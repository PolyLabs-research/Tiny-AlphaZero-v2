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
from dataclasses import replace
from pathlib import Path

import torch

from tinyalphazero.config import config_fingerprint, load_config
from tinyalphazero.evaluate import format_ladder_event, run_ladder_pass


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
    # Protocol parameters default to the config — and the config defaults to the
    # run's own config.yaml beside the checkpoint — so an offline pass of a
    # campaign checkpoint is the same measurement as the campaign's inline pass
    # (REVIEW-2026-10-21 D6). Every flag below overrides one field for this pass.
    ap.add_argument(
        "--anchors",
        default=None,
        help="Comma-separated: random, greedy, phase1, stockfish[:skill]. A "
        "stockfish rung with no binary present is skipped, not an error. "
        "Default: eval.ladder_anchors.",
    )
    ap.add_argument(
        "--anchor-ckpt",
        type=Path,
        default=None,
        help="Frozen Phase 1 opponent for the `phase1` rung (default: eval.anchor_ckpt; "
        "its sha256 must equal eval.anchor_digest — see runs/ANCHORS.sha256).",
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
    ap.add_argument(
        "--config",
        default=None,
        help="YAML config. Default: the config.yaml beside --ckpt when there is one "
        "(a campaign checkpoint), else the dataclass defaults.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Default: eval.eval_seed, else the config's run seed — what the inline "
        "pass used, so the opponents' move streams and the bootstrap match.",
    )
    ap.add_argument("--iteration", type=int, default=None, help="Campaign iteration, for the row")
    ap.add_argument(
        "--stockfish",
        default="",
        help="Explicit Stockfish binary path (default: eval.stockfish_path)",
    )
    ap.add_argument(
        "--proof-directed",
        action="store_true",
        help="Run the pass with proof-directed reply expansion ON (p2_d Track 1), via "
        "mcts.eval_proof_directed_expansion — the measurement-side switch; self-play "
        "is untouched. Rows record the resulting config fingerprint.",
    )
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

    config_path = args.config
    if config_path is None and (args.ckpt.parent / "config.yaml").exists():
        config_path = str(args.ckpt.parent / "config.yaml")
    cfg = load_config(config_path)
    if args.proof_directed:
        cfg = replace(cfg, mcts=replace(cfg.mcts, eval_proof_directed_expansion=True))
    print(
        f"config: {config_path or 'dataclass defaults'} | fingerprint {config_fingerprint(cfg)[:12]}"
    )
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
