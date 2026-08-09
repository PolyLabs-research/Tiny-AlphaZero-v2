#!/usr/bin/env python3
"""Forward-pass latency at self-play batch sizes (chunk 7, spec §10).

    python benchmarks/bench_inference.py --ckpt runs/phase1_d/final.pt --device cuda

§10 target: **< 8 ms forward at batch 512**, bf16. That number is the one the
batched runner's throughput budget is built on — at 48 sims and 256 concurrent
games the runner issues a batch-512 forward roughly every 512 leaves, so the
forward time multiplied by the flush count is the floor on wall-clock no amount
of tree optimisation can go under.

Measured with `torch.inference_mode()` and bf16 autocast, matching
``torch_evaluate_batch``: a benchmark run under different precision or grad
settings than the code path it stands in for is measuring a program nobody runs.
Reports median and p95 rather than the mean — a mean over a distribution with a
long right tail flatters exactly the case (a stalled flush) that hurts.

The measurement is recorded as a **measurement**, not a pass/fail gate. Chunk
7's escalation path keys off aggregate sims/sec in bench_selfplay.py; this
script exists to say where the time goes when that number disappoints.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tinyalphazero.train import load_checkpoint

BATCHES = (64, 256, 512)
TARGET_MS_AT_512 = 8.0


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "no-git"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_batch(
    net, device: torch.device, batch: int, iters: int, warmup: int, seq_len: int
) -> dict:
    """Median/p95 forward latency at one batch size.

    Inputs are random *token* ids in the encoder's vocabulary rather than real
    positions: the trunk is a fixed-shape transformer, so latency does not depend
    on the board — and generating real positions here would benchmark
    python-chess instead of the network.
    """
    rng = np.random.default_rng(0)
    tokens = torch.from_numpy(
        rng.integers(0, 12, size=(batch, seq_len), dtype=np.int64)
    ).to(device)

    samples: list[float] = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for i in range(warmup + iters):
            _sync(device)
            t0 = time.perf_counter()
            net(tokens)
            _sync(device)  # without this the timer reads the launch, not the work
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= warmup:
                samples.append(dt)

    samples.sort()
    return {
        "batch": batch,
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 3),
        "min_ms": round(samples[0], 3),
        "positions_per_s": round(batch / (statistics.median(samples) / 1000.0), 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path, help="Checkpoint to time")
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument("--iters", type=int, default=50, help="Timed iterations per batch size")
    ap.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Untimed iterations first. ROCm compiles kernels on first use; without "
        "a warmup the batch-64 row absorbs that cost and reads slowest of the three.",
    )
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results.jsonl"))
    ap.add_argument("--no-record", action="store_true", help="Print only; do not append")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    net, meta = load_checkpoint(args.ckpt, device=device)
    net.eval()
    seq_len = meta["config_obj"].model.seq_len
    sha = _git_sha()

    rows = []
    for batch in BATCHES:
        row = time_batch(net, device, batch, args.iters, args.warmup, seq_len)
        row.update(
            {
                "benchmark": "inference_latency",
                "git_sha": sha,
                "ckpt": str(args.ckpt),
                "device": args.device,
                "dtype": "bf16",
                "torch": torch.__version__,
                "hip": getattr(torch.version, "hip", None),
            }
        )
        rows.append(row)
        print(
            f"batch {batch:>4}: median {row['median_ms']:>7.3f} ms  "
            f"p95 {row['p95_ms']:>7.3f} ms  {row['positions_per_s']:>9.1f} pos/s"
        )

    at512 = next(r for r in rows if r["batch"] == 512)
    verdict = "meets" if at512["median_ms"] < TARGET_MS_AT_512 else "misses"
    print(
        f"\n§10 target < {TARGET_MS_AT_512} ms at batch 512: "
        f"{verdict} ({at512['median_ms']:.3f} ms median)"
    )

    if not args.no_record:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"appended {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
