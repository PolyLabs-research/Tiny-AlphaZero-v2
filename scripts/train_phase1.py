#!/usr/bin/env python3
"""Phase 1 supervised bootstrap training (spec §5).

Usage:
    python scripts/train_phase1.py \\
        --data runs/data/phase1 \\
        --mate1 runs/data/mate1 \\
        --eval-data runs/data/eval_rand \\
        --eval-mate1 runs/data/eval_mate1 \\
        --out runs/phase1_a \\
        [--epochs 3] [--batch 1024] [--seed 42] [--ckpt <resume>] [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from tinyalphazero.config import load_config
from tinyalphazero.data import Phase1Dataset, assert_dataset_mode
from tinyalphazero.model import ChessNet
from tinyalphazero.train import (
    WeightedPhaseDataset,
    compute_metrics,
    load_checkpoint,
    save_checkpoint,
    train_step,
)


def _build_loader(
    data_dir: Path,
    mate1_dir: Path | None,
    cfg_train,
    seed: int,
    num_workers: int = 8,
    mate2_dir: Path | None = None,
) -> tuple[DataLoader, int]:
    """Mix random play with mate-in-1 and mate-in-2 anchors at their configured fractions.

    Each anchor sample is up-weighted so its dataset contributes exactly its
    configured fraction of the batch, regardless of how many positions it holds.
    Random play takes the remainder, so the mixes must sum below 1.
    """
    anchor_mix = cfg_train.phase1_mate1_mix + getattr(cfg_train, "phase1_mate2_mix", 0.0)
    if not 0.0 <= anchor_mix < 1.0:
        raise ValueError(
            f"anchor mixes must sum to <1, got mate1={cfg_train.phase1_mate1_mix} + "
            f"mate2={getattr(cfg_train, 'phase1_mate2_mix', 0.0)} = {anchor_mix}"
        )

    rand_ds = WeightedPhaseDataset(
        Phase1Dataset(data_dir),
        value_weight=cfg_train.phase1_value_weight_random,
    )

    n_rand = len(rand_ds)
    datasets = [rand_ds]
    weights = np.full(n_rand, 1.0 - anchor_mix)

    for anchor_dir, mix in (
        (mate1_dir, cfg_train.phase1_mate1_mix),
        (mate2_dir, getattr(cfg_train, "phase1_mate2_mix", 0.0)),
    ):
        if anchor_dir is None or not anchor_dir.exists() or mix <= 0:
            continue
        anchor_ds = WeightedPhaseDataset(
            Phase1Dataset(anchor_dir),
            value_weight=cfg_train.phase1_value_weight_anchor,
        )
        n_a = len(anchor_ds)
        datasets.append(anchor_ds)
        weights = np.concatenate(
            [weights, np.full(n_a, (mix / (1.0 - anchor_mix)) * (n_rand / n_a))]
        )

    combined = ConcatDataset(datasets)
    sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(combined),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )

    # persistent_workers/prefetch_factor are only valid with worker processes, and
    # num_workers=0 is what tests and CPU debugging use -- so the shipped builder
    # has to accept it rather than being reachable only in its production shape.
    worker_opts = {"persistent_workers": True, "prefetch_factor": 4} if num_workers > 0 else {}
    loader = DataLoader(
        combined,
        batch_size=cfg_train.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **worker_opts,
    )
    return loader, len(combined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="Random-play dataset dir")
    parser.add_argument("--mate1", default=None, type=Path, help="Mate-in-1 anchor dataset dir")
    parser.add_argument(
        "--mate2", default=None, type=Path, help="Mate-in-2 anchor dataset dir (round d+)"
    )
    parser.add_argument(
        "--eval-data", default=None, type=Path, help="Held-out random-play dir for metrics"
    )
    parser.add_argument(
        "--eval-mate1", default=None, type=Path, help="Held-out mate-in-1 dir for metrics"
    )
    parser.add_argument(
        "--eval-mate2", default=None, type=Path, help="Held-out mate-in-2 dir for metrics"
    )
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Stop after N steps (timing/smoke slice)"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model. BROKEN on ROCm/gfx1151 as measured on "
        "torch 2.12.1+rocm7.2 (HIP 7.2.53211): produces non-finite gradients on ~95%% "
        "of steps and the run aborts. Retest when the ROCm/HIP build changes — compare "
        "torch.__version__ and torch.version.hip against those, not the date.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output run directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override phase1_epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="Warm start: load MODEL WEIGHTS ONLY from this checkpoint, then train with "
        "a fresh optimizer, fresh schedule and step counter at 0. Distinct from --ckpt, "
        "which resumes optimizer state and step for crash recovery within a round. "
        "Mutually exclusive with --ckpt.",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.phase1_epochs = args.epochs
    if args.batch is not None:
        cfg.train.batch_size = args.batch

    # ---- Fail fast on mis-generated datasets (see runs/phase1_a post-mortem) ----
    assert_dataset_mode(args.data, "random_games")
    if args.ckpt is not None and args.init_weights is not None:
        parser.error(
            "--ckpt and --init-weights are mutually exclusive: --ckpt resumes a run "
            "(optimizer + step), --init-weights starts a new one from borrowed weights."
        )
    if args.mate1 is not None:
        assert_dataset_mode(args.mate1, "mate_in_1")
    if args.mate2 is not None:
        assert_dataset_mode(args.mate2, "mate_in_2")
    if args.eval_mate2 is not None:
        assert_dataset_mode(args.eval_mate2, "mate_in_2")
    if args.eval_data is not None:
        assert_dataset_mode(args.eval_data, "random_games")
    if args.eval_mate1 is not None:
        assert_dataset_mode(args.eval_mate1, "mate_in_1")

    eval_data = args.eval_data or args.data
    eval_mate1 = args.eval_mate1 or args.mate1
    if args.mate2 is not None and args.eval_mate2 is None:
        print("WARNING: --mate2 given without --eval-mate2; the round d gate cannot be read.")
    if args.eval_data is None:
        print("WARNING: --eval-data not set; metrics computed on the TRAINING set.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "metrics.jsonl"
    if args.ckpt is None and log_path.exists():
        # Fresh run into an existing dir: start a clean log rather than appending
        # onto a previous run's rows.
        log_path.unlink()

    # ---- Model & optimiser ----
    start_step = 0
    if args.ckpt is not None:
        net, meta = load_checkpoint(args.ckpt, device=device, reference_model_cfg=cfg.model)
        start_step = meta.get("step", 0)
        print(f"Resumed from {args.ckpt} at step {start_step}")
    elif args.init_weights is not None:
        # Warm restart: borrow the weights, discard everything else. start_step
        # stays 0 so the schedule, epoch numbering and metrics.jsonl all describe
        # *this* round rather than continuing the donor's bookkeeping.
        net, donor = load_checkpoint(
            args.init_weights, device=device, reference_model_cfg=cfg.model
        )
        print(
            f"Warm start from {args.init_weights} "
            f"(donor step={donor.get('step', '?')}, sha={donor.get('git_sha', '?')}); "
            "fresh optimizer, fresh schedule, step=0"
        )
    else:
        net = ChessNet(cfg.model).to(device)

    # Keep the uncompiled module: torch.compile wraps it, and the wrapper's
    # state_dict() carries `_orig_mod.` prefixes that would break checkpoint I/O.
    # The two share parameters, so training through `net` updates `raw_net`.
    raw_net = net
    if args.compile:
        net = torch.compile(net)

    loader, n_samples = _build_loader(
        args.data,
        args.mate1,
        cfg.train,
        args.seed,
        num_workers=args.workers,
        mate2_dir=args.mate2,
    )
    # len(loader) accounts for drop_last; ceil(n/batch) does not, which left the old
    # run with a 3-step orphan "epoch 4".
    steps_per_epoch = len(loader)
    total_steps = cfg.train.phase1_epochs * steps_per_epoch
    # Timing/smoke slice: bound the loop only. total_steps still sets the cosine
    # T_max, so a sliced run sees the real schedule's opening — it is not a
    # shortened training run, and its metrics are not final-run metrics.
    stop_at = min(total_steps, args.max_steps) if args.max_steps is not None else total_steps

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=cfg.train.phase1_lr_start,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg.train.phase1_lr_end,
    )

    if args.ckpt is not None:
        meta = meta  # already loaded above
        if "optimizer" in meta:
            optimizer.load_state_dict(meta["optimizer"])

    # Fast-forward scheduler to start_step
    for _ in range(start_step):
        scheduler.step()

    # ---- Training loop ----
    step = start_step
    net.train()
    # Derive from step, else a resume restarts epoch numbering at 1 and clobbers the
    # existing epoch_01.pt.
    epoch = start_step // steps_per_epoch

    print(
        f"Phase 1 training | device={device} | bf16={use_bf16} | "
        f"steps={stop_at}/{total_steps} | batch={cfg.train.batch_size} | "
        f"workers={args.workers} | masked_policy={cfg.train.phase1_policy_masked} | "
        f"compile={args.compile}"
    )

    loss_keys = ("loss", "policy_loss", "promo_loss", "value_loss", "legality_loss")
    total_skipped = 0

    while step < stop_at:
        epoch += 1
        epoch_losses: dict[str, list[float]] = {k: [] for k in loss_keys}
        epoch_skipped = 0
        t0 = time.perf_counter()

        for batch in loader:
            if step >= stop_at:
                break

            losses = train_step(
                batch,
                net,
                optimizer,
                device,
                cfg.train,
                use_bf16=use_bf16,
                masked_policy=cfg.train.phase1_policy_masked,
            )
            scheduler.step()
            step += 1
            if losses["skipped"]:
                epoch_skipped += 1
                total_skipped += 1
                print(
                    f"  step {step:6d}: non-finite gradient (norm={losses['grad_norm']}), "
                    f"update skipped ({total_skipped} total)"
                )
            for k in loss_keys:
                if np.isfinite(losses[k]):
                    epoch_losses[k].append(losses[k])

            # A steady trickle of skips is a real defect, not a transient. Bail out
            # rather than burn hours producing a model trained on a fraction of its
            # batches.
            if total_skipped > 20 and total_skipped > 0.01 * step:
                raise RuntimeError(
                    f"Aborting: {total_skipped} non-finite gradients in {step} steps "
                    f"({100 * total_skipped / step:.2f}%). Investigate before continuing — "
                    f"see the ROCm SDPA note in PROGRESS.md."
                )

            if step % 200 == 0:
                # Log components, not just the total: a policy term pinned at its
                # floor from step 1 is the signature that diagnosed runs/phase1_a.
                recent = {
                    k: (float(np.mean(epoch_losses[k][-200:])) if epoch_losses[k] else float("nan"))
                    for k in loss_keys
                }
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  step {step:6d}/{stop_at}  loss={recent['loss']:.4f}  "
                    f"policy={recent['policy_loss']:.4f}  promo={recent['promo_loss']:.4f}  "
                    f"value={recent['value_loss']:.4f}  leg={recent['legality_loss']:.4f}  "
                    f"lr={lr:.2e}"
                )

        elapsed = time.perf_counter() - t0
        mean_losses = {
            k: (float(np.mean(v)) if v else float("nan")) for k, v in epoch_losses.items()
        }
        mean_epoch_loss = mean_losses["loss"]

        # Per-epoch metrics — held-out sets, never the training dirs
        metrics = compute_metrics(
            raw_net,
            eval_data,
            device,
            mate1_dir=eval_mate1,
            mate2_dir=args.eval_mate2,
            n_random=1000,
            n_mate1=200,
        )
        metrics.update(mean_losses)
        metrics.update(
            {
                "step": step,
                "epoch": epoch,
                "elapsed_s": round(elapsed, 1),
                "lr": scheduler.get_last_lr()[0],
                "skipped_steps": epoch_skipped,
                "skipped_steps_total": total_skipped,
            }
        )

        print(
            f"Epoch {epoch} | loss={mean_epoch_loss:.4f} "
            f"(policy={mean_losses['policy_loss']:.4f}) | "
            f"mate2_top16={metrics.get('mate2_top16_coverage', float('nan')):.4f} | "  # gated (round d)
            f"mate1_top8={metrics['mate1_top8_masked']:.4f} | "  # gated (BLOCKED-2026-08-06)
            f"auc={metrics['legality_auc']:.4f} | "
            f"wdl_anchor={metrics['wdl_anchor_mass']:.4f} | "
            f"top1_legal={metrics['top1_legal_rate']:.4f}ⁱ | "
            f"mate1_solve={metrics['mate1_solve_rate']:.4f}ⁱ"
            + (f" | skipped={epoch_skipped}" if epoch_skipped else "")
        )

        with open(log_path, "a") as fh:
            fh.write(json.dumps(metrics) + "\n")

        # Epoch checkpoint
        ckpt_path = args.out / f"epoch_{epoch:02d}.pt"
        save_checkpoint(ckpt_path, raw_net, optimizer, cfg, step=step, phase="1")
        print(f"Saved {ckpt_path}")

        net.train()

    # Final checkpoint
    save_checkpoint(args.out / "final.pt", raw_net, optimizer, cfg, step=step, phase="1")
    print(f"Done. Final checkpoint: {args.out / 'final.pt'}")


if __name__ == "__main__":
    main()
