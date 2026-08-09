#!/usr/bin/env python3
"""Self-play throughput benchmark for the batched runner (chunk 7, spec §10).

    python benchmarks/bench_selfplay.py --games 32 --sims 16 --device cpu
    python benchmarks/bench_selfplay.py --games 256 --sims 48 --ckpt runs/phase1_anchor.pt

Targets on the Strix Halo (NOT on CPU/CI): >= 3,000 sims/sec aggregate, forward
< 8 ms at batch 512. Below ~1,500 sims/sec the sanctioned response is to profile
with py-spy and record the findings in benchmarks/NOTES.md *before* changing
anything — the §10 bottleneck ordering predicts python-chess move generation
first, tree ops second, GPU last, and a fix aimed at the wrong one is how v1
spent its time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from tinyalphazero.config import MCTSConfig, load_config
from tinyalphazero.mcts.batch_runner import (
    BatchRunner,
    torch_evaluate_batch,
)
from tinyalphazero.mcts.batch_runner import (
    uniform_stub_batch as stub_batch,
)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parents[1]),
                    "rev-parse",
                    "--short",
                    "HEAD",
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "no-git"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--batch-leaves", type=int, default=None)
    ap.add_argument("--max-plies", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", type=Path, default=None, help="Use a real net instead of the stub")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results.jsonl"))
    ap.add_argument("--no-record", action="store_true", help="Print only; do not append")
    args = ap.parse_args()

    cfg: MCTSConfig = load_config(args.config).mcts if args.config else MCTSConfig()
    if args.batch_leaves is not None:
        cfg.batch_leaves = args.batch_leaves

    if args.ckpt is not None:
        import torch

        from tinyalphazero.train import load_checkpoint

        device = torch.device(args.device)
        net, _meta = load_checkpoint(args.ckpt, device=device)
        net.eval()
        evaluate = torch_evaluate_batch(net, device)
        eval_name = str(args.ckpt)
    else:
        evaluate = stub_batch
        eval_name = "stub"

    runner = BatchRunner(evaluate, cfg, device=args.device, max_plies=args.max_plies)
    t0 = time.perf_counter()
    traces = list(runner.play_games(args.games, seeds=list(range(args.games)), num_sims=args.sims))
    elapsed = time.perf_counter() - t0

    s = runner.stats
    row = {
        "date": None,  # stamped by the caller; scripts here must stay deterministic
        "git_sha": _git_sha(),
        "bench": "selfplay",
        "config": (
            f"games={args.games} sims={args.sims} batch_leaves={cfg.batch_leaves} "
            f"device={args.device} eval={eval_name}"
        ),
        "elapsed_s": round(elapsed, 2),
        "games": len(traces),
        "plies": sum(t.plies for t in traces),
        "sims": s["sims"],
        "leaves": s["leaves"],
        "forwards": s["forwards"],
        "terminal_shortcircuits": s["terminal_shortcircuits"],
        "sims_per_s": round(s["sims"] / elapsed, 1),
        "leaves_per_s": round(s["leaves"] / elapsed, 1),
        "forwards_per_s": round(s["forwards"] / elapsed, 2),
        "mean_batch_fill": round(s["leaves"] / max(s["forwards"], 1), 2),
        "ms_per_forward": round(1000 * elapsed / max(s["forwards"], 1), 3),
    }

    print(
        f"games={row['games']} plies={row['plies']} in {row['elapsed_s']}s\n"
        f"  sims/s        {row['sims_per_s']}\n"
        f"  leaves/s      {row['leaves_per_s']}\n"
        f"  forwards/s    {row['forwards_per_s']}\n"
        f"  mean fill     {row['mean_batch_fill']} / {cfg.batch_leaves}\n"
        f"  terminal s-c  {row['terminal_shortcircuits']} (never reach the evaluator)"
    )
    if args.device != "cpu" and row["sims_per_s"] < 1500:
        print(
            "\n  BELOW 1,500 sims/s — profile with py-spy and write benchmarks/NOTES.md\n"
            "  BEFORE changing anything (spec §10)."
        )

    if not args.no_record:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"  appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
