#!/usr/bin/env python3
"""Anchor ladder evaluation (spec §8) — the trustworthy progress signal.

Offline and read-only with respect to training: it takes checkpoints, plays the
frozen book against frozen opponents, runs the frozen suites, and appends JSONL.
Nothing here writes to a replay buffer or a checkpoint, which is what lets it be
re-run against any past checkpoint at any time and get the same answer.

    python scripts/eval_ladder.py --ckpt runs/phase1_anchor.pt \\
        --anchors random,greedy --games 20 --out runs/ladder_smoke/ladder.jsonl

    # the full pass M3 runs every 10 iterations
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python scripts/eval_ladder.py \\
        --ckpt runs/p2_a/ckpt_iter0010.pt --anchors random,greedy,phase1 \\
        --anchor-ckpt runs/phase1_anchor.pt --out runs/p2_a/ladder.jsonl

**Resumable mid-pass.** Rows are appended per matchup, and a matchup already in
the log for this checkpoint is skipped rather than recomputed. A killed pass costs
the matchup in flight and nothing else — the same write-ordering contract the
replay ring states, for the same reason: a 45-minute pass that has to restart from
zero will eventually be a pass nobody runs.

This file is an argparse wrapper and a printer. The pass itself is
``evaluate.run_ladder_pass``, because the campaign's ``--ladder-every`` hook needs
the same code and a library module must not import a script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from tinyalphazero.config import load_config
from tinyalphazero.evaluate import format_ladder_event, run_ladder_pass

REPO = Path(__file__).resolve().parents[1]


def report(kind: str, payload: dict) -> None:
    """Print one ladder event. The rendering itself lives in the library, so the
    campaign's inline ladder produces byte-identical output."""
    line = format_ladder_event(kind, payload)
    if line is not None:
        print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, type=Path, help="Checkpoint under evaluation")
    ap.add_argument(
        "--anchors",
        default="random,greedy,phase1",
        help="Comma-separated: random, greedy, phase1, stockfish[:skill]. A "
        "stockfish rung with no binary present is skipped, not an error.",
    )
    ap.add_argument(
        "--anchor-ckpt",
        type=Path,
        default=REPO / "runs" / "phase1_anchor.pt",
        help="Frozen Phase 1 opponent for the `phase1` rung (anchor v2 by digest; "
        "see TRAINING_STRATEGY.md).",
    )
    ap.add_argument("--out", type=Path, default=None, help="ladder.jsonl (default: beside --ckpt)")
    ap.add_argument(
        "--games",
        type=int,
        default=None,
        help="Total games per matchup. Halved to a number of openings, each played "
        "twice with colours swapped — the pairing the CI resamples over. Default: "
        "the full 200.",
    )
    ap.add_argument("--sims", type=int, default=None, help="Match sims (default: eval_match_sims)")
    ap.add_argument("--device", default=None, help="cuda | cpu")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iteration", type=int, default=None, help="Campaign iteration, for the row")
    ap.add_argument("--stockfish", default="", help="Explicit Stockfish binary path")
    ap.add_argument(
        "--no-suites", action="store_true", help="Matches only; skip the eval row entirely"
    )
    ap.add_argument(
        "--role",
        choices=["protocol", "diagnostic"],
        default="protocol",
        help="What the rows are for. `protocol` is a pass a gate or verdict may "
        "read; `diagnostic` is instrumentation no gate ever reads -- an off-cadence "
        "checkpoint, a climbed Stockfish rung. It is stamped into every row this "
        "pass writes, so the distinction travels with the data instead of living "
        "in a document the reader has to find.",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_ladder_pass(
        args.ckpt,
        cfg,
        device,
        out=args.out or (args.ckpt.parent / "ladder.jsonl"),
        anchors=args.anchors,
        anchor_ckpt=args.anchor_ckpt,
        iteration=args.iteration,
        seed=args.seed,
        games=args.games,
        sims=args.sims,
        stockfish=args.stockfish,
        suites=not args.no_suites,
        role=args.role,
        produced="offline",
        on_event=report,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
