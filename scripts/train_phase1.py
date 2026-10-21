#!/usr/bin/env python3
"""Phase 1 supervised bootstrap training (spec §5).

Usage:
    python scripts/train_phase1.py \\
        --data runs/data/phase1 \\
        --mate1 runs/data/mate1 \\
        --eval-data runs/data/eval_rand \\
        --eval-mate1 runs/data/eval_mate1 \\
        --out runs/phase1_a \\
        [--config configs/phase1_d.yaml] [--epochs 3] [--batch 1024] [--seed 42]

    # resume: config, epochs, batch, mixes and seed come from the checkpoint
    python scripts/train_phase1.py --data ... --mate1 ... --out runs/phase1_a \\
        --ckpt runs/phase1_a/epoch_02.pt

Without --config the dataclass defaults apply (Phase 1 values: dropout 0.1,
legality auxiliary on). configs/default.yaml is the *Phase 2* campaign config
and is not a Phase 1 config.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tinyalphazero.config import Config, config_fingerprint, load_config, save_config
from tinyalphazero.data import assert_dataset_mode
from tinyalphazero.envinfo import environment_report
from tinyalphazero.model import ChessNet
from tinyalphazero.phase1 import build_phase1_loader
from tinyalphazero.train import (
    _git_sha,
    compute_metrics,
    load_checkpoint,
    save_checkpoint,
    train_step,
)


def _config_diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    """Dotted keys whose values differ, with both values, so the refusal names them."""
    out = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(_config_diff(va, vb, f"{prefix}{key}."))
        elif va != vb:
            out.append(f"{prefix}{key}: {va!r} vs {vb!r}")
    return out


def _write_run_artifacts(
    out: Path, cfg: Config, device: torch.device, resume: dict | None
) -> list[str]:
    """The AGENTS §4 run-dir artifacts: resolved config.yaml and provenance.json.

    provenance.json carries the git SHA of the running code, the config
    fingerprint, torch/device and the check_env report (the matmul probe only on
    GPU, where it costs well under a second). A resume appends to a ``resumes``
    list rather than overwriting the record of the run's start; a run dir whose
    config.yaml disagrees with the checkpoint being resumed is refused.
    Returns the conflicts found (empty on success) so the caller can report them.
    """
    config_path = out / "config.yaml"
    if resume is not None and config_path.exists():
        diff = _config_diff(asdict(load_config(str(config_path))), asdict(cfg))
        if diff:
            return [f"{config_path} does not describe the checkpoint being resumed: {diff}"]
    else:
        save_config(cfg, str(config_path))

    report = environment_report(matmul=device.type == "cuda")
    entry = {
        "git_sha": _git_sha(),
        "config_fingerprint": config_fingerprint(cfg),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": report.get("device_name", "cpu"),
        "seed": cfg.seed,
        "argv": sys.argv,
        "environment": report,
    }
    prov_path = out / "provenance.json"
    if resume is None:
        record = entry
    else:
        record = json.loads(prov_path.read_text()) if prov_path.exists() else dict(entry)
        record.setdefault("resumes", []).append({**entry, **resume})
    prov_path.write_text(json.dumps(record, indent=2) + "\n")
    return []


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
    parser.add_argument("--seed", type=int, default=None, help="Run seed (default: config seed)")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Resume a run from its checkpoint: weights, optimizer, step AND config "
        "(epochs, batch, schedule, mixes, seed) come from the checkpoint; --config, "
        "--epochs, --batch and --seed are refused if they disagree with it.",
    )
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

    if args.ckpt is not None and args.init_weights is not None:
        parser.error(
            "--ckpt and --init-weights are mutually exclusive: --ckpt resumes a run "
            "(optimizer + step), --init-weights starts a new one from borrowed weights."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"

    # ---- Config: the command line for a fresh run, the checkpoint for a resume ----
    meta: dict = {}
    if args.ckpt is not None:
        # A resumed run cannot silently change what it is (REVIEW-2026-10-21 E2):
        # epochs set the cosine T_max, the mixes set the data, the seed sets the
        # draw. All of it comes from the checkpoint, and a command line that says
        # otherwise is refused rather than overridden either way.
        net, meta = load_checkpoint(args.ckpt, device=device)
        cfg = meta["config_obj"]
        conflicts = []
        if args.config is not None:
            diff = _config_diff(asdict(load_config(args.config)), asdict(cfg))
            if diff:
                conflicts.append(
                    f"--config {args.config} differs from the checkpoint config: {diff}"
                )
        for flag, value, have in (
            ("--epochs", args.epochs, cfg.train.phase1_epochs),
            ("--batch", args.batch, cfg.train.batch_size),
            ("--seed", args.seed, cfg.seed),
        ):
            if value is not None and value != have:
                conflicts.append(f"{flag} {value} vs checkpoint {have}")
        if conflicts:
            parser.error(
                f"--ckpt {args.ckpt} resumes the run its config describes; drop the "
                "conflicting flags or start a new run with --init-weights.\n  "
                + "\n  ".join(conflicts)
            )
    else:
        cfg = load_config(args.config)
        if args.epochs is not None:
            cfg.train.phase1_epochs = args.epochs
        if args.batch is not None:
            cfg.train.batch_size = args.batch
        if args.seed is not None:
            # Into the config, so the checkpoint and config.yaml record the seed
            # the run actually used (E9: cfg.seed was dead for Phase 1).
            cfg.seed = args.seed
    seed = cfg.seed

    # ---- Fail fast on mis-generated datasets (see runs/phase1_a post-mortem) ----
    assert_dataset_mode(args.data, "random_games")
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
        print("WARNING: --eval-data not set; random-play metrics computed on the TRAINING set.")
    if args.eval_mate1 is None and args.mate1 is not None:
        print(
            "WARNING: --eval-mate1 not set; the gated mate-in-1 metrics (mate1_top8_masked, "
            "wdl_anchor_mass) are computed on the TRAINING anchors."
        )
    if args.eval_mate1 is None and args.mate1 is None:
        print(
            "WARNING: no mate-in-1 dir at all; compute_metrics will score mate-in-1 "
            "metrics on 200 positions it generates itself. They gate nothing here."
        )
    for name, mix, given in (
        ("mate1", cfg.train.phase1_mate1_mix, args.mate1),
        ("mate2", cfg.train.phase1_mate2_mix, args.mate2),
    ):
        if mix > 0 and given is None:
            print(
                f"WARNING: train.phase1_{name}_mix={mix} but no --{name} dir given; "
                f"no {name} anchors will be sampled."
            )

    torch.manual_seed(seed)
    np.random.seed(seed)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "metrics.jsonl"
    if args.ckpt is None and log_path.exists():
        # Fresh run into an existing dir: start a clean log rather than appending
        # onto a previous run's rows.
        log_path.unlink()

    # ---- Model & optimiser ----
    start_step = 0
    if args.ckpt is not None:
        start_step = meta.get("step", 0)
        print(
            f"Resumed from {args.ckpt} at step {start_step} "
            f"(config from the checkpoint: epochs={cfg.train.phase1_epochs} "
            f"batch={cfg.train.batch_size} seed={seed})"
        )
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

    # ---- Run artifacts (AGENTS §4): resolved config, git SHA, check_env report ----
    conflicts = _write_run_artifacts(
        args.out,
        cfg,
        device,
        resume=None if args.ckpt is None else {"from": str(args.ckpt), "step": start_step},
    )
    if conflicts:
        parser.error("\n  ".join(["refusing to resume into this run dir:", *conflicts]))

    # Keep the uncompiled module: torch.compile wraps it, and the wrapper's
    # state_dict() carries `_orig_mod.` prefixes that would break checkpoint I/O.
    # The two share parameters, so training through `net` updates `raw_net`.
    raw_net = net
    if args.compile:
        net = torch.compile(net)

    p1 = build_phase1_loader(
        args.data,
        args.mate1,
        cfg.train,
        seed,
        num_workers=args.workers,
        mate2_dir=args.mate2,
    )
    loader = p1.loader
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
    if args.ckpt is not None and "optimizer" in meta:
        optimizer.load_state_dict(meta["optimizer"])
        # The saved param groups carry the LR *at the checkpoint*. CosineAnnealingLR
        # steps recursively from whatever lr the group holds, so fast-forwarding
        # from lr_k lands at lr_2k, not lr_k (measured: 9.75e-5 where 1.65e-4 was
        # due). Rewind to the start value and replay the schedule from step 0,
        # which reproduces the uninterrupted trajectory operation for operation.
        for group in optimizer.param_groups:
            group["lr"] = cfg.train.phase1_lr_start
            group["initial_lr"] = cfg.train.phase1_lr_start
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg.train.phase1_lr_end,
    )

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
    # The realised batch fraction, not the configured mix: under the legacy
    # weighting the two differ (REVIEW-2026-10-21 E1), and the log should say
    # which one this run is actually drawing.
    print(
        f"  anchor weighting={cfg.train.phase1_anchor_weighting} | realised fractions: "
        + (
            ", ".join(f"{k}={v:.4f}" for k, v in p1.anchor_fractions.items())
            if p1.anchor_fractions
            else "none (no anchor dirs given)"
        )
    )

    loss_keys = ("loss", "policy_loss", "promo_loss", "value_loss", "legality_loss")
    total_skipped = 0

    while step < stop_at:
        epoch += 1
        # Epoch k's multiset is a function of (seed, k): a resumed run draws what
        # the uninterrupted one would have, instead of replaying epoch 1 (E2).
        p1.sampler.set_epoch(epoch)
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
