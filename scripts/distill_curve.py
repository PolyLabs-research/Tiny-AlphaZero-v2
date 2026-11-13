#!/usr/bin/env python3
"""The capacity-or-data curve (DIAG-2026-11-13): train on teacher rows at nested sizes.

    python scripts/distill_curve.py --start runs/p2_w_quarter_down/ckpt_iter0010.pt \\
        --d1 runs/p2_u_fresh/teacher --new runs/diag_cod/teacher \\
        --holdout runs/p2_u_fresh/teacher_holdout --out runs/diag_cod/curve

Each size is D1 (whole) plus the first ``size - len(D1)`` rows of ``--new``, trained from
``--start`` with a fresh AdamW at ``--lr`` (weight decay and loss weights from the start's
``config.yaml``), teacher rows only, by ``distill.train_to_best``: the held-out ring is
measured at the start and every ``--eval-every`` samples, and training stops after
``--patience`` measurements without a new best, or at ``--max-samples``. Each measurement
also reads a fixed ``--probe-rows`` sample of D1 (in every set), for the fit gap.

Per size, under ``<out>/size_<rows>/``: ``log.jsonl`` (one row per measurement),
``best.pt``, ``ce_rows.npy`` (the best checkpoint's held-out per-row CE) and
``result.json``; a size with a ``result.json`` is skipped on rerun. At the end,
``<out>/summary.json`` holds the curve, Δ_top (paired bootstrap, D4 minus D3) and the
plan's reading (``distill.read_curve``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", required=True, type=Path)
    ap.add_argument("--config", default=None, help="Default: config.yaml beside --start")
    ap.add_argument("--d1", required=True, type=Path, help="existing teacher ring, used whole")
    ap.add_argument("--new", required=True, type=Path, help="pooled teacher ring, used as prefixes")
    ap.add_argument("--holdout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sizes", default="475696,1000000,2000000,4000000")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=16_000_000)
    ap.add_argument("--probe-rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import torch

    from tinyalphazero.config import load_config
    from tinyalphazero.distill import (
        open_prefix,
        paired_mean_diff,
        read_curve,
        train_to_best,
    )
    from tinyalphazero.provenance import git_sha
    from tinyalphazero.replay import ReplayBuffer
    from tinyalphazero.teacher import agreement
    from tinyalphazero.train import load_checkpoint, save_checkpoint, train_step_phase2

    config_path = args.config
    if config_path is None and (args.start.parent / "config.yaml").exists():
        config_path = str(args.start.parent / "config.yaml")
    cfg = load_config(config_path)
    cfg = replace(cfg, train=replace(cfg.train, lr=args.lr))
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda"
    sha = git_sha()

    d1 = open_prefix(args.d1, None)
    new_ring = ReplayBuffer.open(args.new, readonly=True)
    holdout = ReplayBuffer.open(args.holdout, readonly=True)
    probe = np.sort(
        np.random.default_rng(args.seed).choice(len(d1), size=args.probe_rows, replace=False)
    )
    available = len(d1) + len(new_ring)
    sizes = sorted({min(int(s), available) for s in args.sizes.split(",")})
    if sizes[0] < len(d1):
        print(f"every size must hold D1 ({len(d1)} rows)", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"distill_curve: sizes {sizes} (available {available}); lr {args.lr}, batch "
        f"{args.batch}, eval every {args.eval_every}, patience {args.patience}, cap "
        f"{args.max_samples}; device {device}",
        flush=True,
    )

    results: dict[int, dict] = {}
    ce: dict[int, np.ndarray] = {}
    for size in sizes:
        d = args.out / f"size_{size}"
        if (d / "result.json").exists():
            results[size] = json.loads((d / "result.json").read_text())
            ce[size] = np.load(d / "ce_rows.npy")
            print(f"size {size}: done before, skipped", flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        net, _ = load_checkpoint(args.start, device=device)
        net.train()
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=args.lr, weight_decay=cfg.train.weight_decay
        )
        parts = [d1] if size == len(d1) else [d1, open_prefix(args.new, size - len(d1))]

        def measure(model: object) -> dict:
            h = agreement(model, holdout, device)
            p = agreement(model, d1, device, rows=probe)
            return {**h, "probe_ce": p["teacher_policy_ce"]}

        def step(batch: dict, net: object = net, optimizer: object = optimizer) -> dict:
            return train_step_phase2(
                batch, net, optimizer, device, cfg.train, use_bf16=use_bf16,
                masked_policy=cfg.phase2.masked_policy,
            )  # fmt: skip

        log_path = d / "log.jsonl"
        log_path.write_text("")

        def on_measure(row: dict, log_path: Path = log_path, size: int = size) -> None:
            with log_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"size {size} samples {row['samples']}: held-out CE "
                f"{row['teacher_policy_ce']:.4f} top1 {row['top1_agreement']:.4f} probe CE "
                f"{row['probe_ce']:.4f}",
                flush=True,
            )

        point = train_to_best(
            net, optimizer, parts, measure, step, args.batch, args.eval_every, args.patience,
            args.max_samples, np.random.default_rng(args.seed), on_measure,
        )  # fmt: skip
        net.load_state_dict(point.state_dict)
        save_checkpoint(d / "best.pt", net, optimizer, cfg, point.samples_at_best, "diag", sha)
        np.save(d / "ce_rows.npy", point.ce_rows)
        result = {
            **point.best,
            "samples_at_best": point.samples_at_best,
            "samples_run": point.samples_run,
            "git_sha": sha,
        }
        (d / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        results[size], ce[size] = result, point.ce_rows

    start = json.loads((args.out / f"size_{sizes[0]}" / "log.jsonl").read_text().splitlines()[0])
    summary: dict = {
        "start": {k: start[k] for k in ("teacher_policy_ce", "top1_agreement", "probe_ce")},
        "curve": [
            {
                "rows": results[s]["rows"],
                "teacher_policy_ce": results[s]["teacher_policy_ce"],
                "top1_agreement": results[s]["top1_agreement"],
                "value_mse_vs_teacher": results[s]["value_mse_vs_teacher"],
                "probe_ce": results[s]["probe_ce"],
                "samples_at_best": results[s]["samples_at_best"],
            }
            for s in sizes
        ],
        "teacher_entropy": results[sizes[-1]]["teacher_entropy"],
    }
    if len(sizes) >= 2:
        d3, d4 = sizes[-2], sizes[-1]
        delta_top = paired_mean_diff(ce[d3], ce[d4], resamples=args.resamples, seed=0)
        fit_gap = results[d4]["teacher_policy_ce"] - results[d4]["probe_ce"]
        summary.update(
            {
                "delta_top": {"from_rows": d3, "to_rows": d4, **delta_top},
                "delta_first_to_last": paired_mean_diff(
                    ce[sizes[0]], ce[d4], resamples=args.resamples, seed=0
                ),
                "fit_gap_top": fit_gap,
                "reading": read_curve(delta_top, fit_gap),
            }
        )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary.get(k) for k in ("delta_top", "fit_gap_top", "reading")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
