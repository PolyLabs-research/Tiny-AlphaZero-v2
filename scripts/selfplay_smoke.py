#!/usr/bin/env python3
"""Play a few self-play games into a replay ring and print the throughput line.

    python scripts/selfplay_smoke.py --stub --games 8 --out runs/sp_smoke
    python scripts/selfplay_smoke.py --ckpt runs/phase1_anchor.pt --games 32 \\
        --out runs/sp_smoke --device cuda

A smoke test for the actor + ring, not a campaign: ``scripts/train_phase2.py``
is the loop. This CLI used to live inside ``tinyalphazero.selfplay`` as a
``main()``; it is a script now because library code does not print
(AGENTS.md §4, REVIEW-2026-10-21 E11).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tinyalphazero.config import load_config
from tinyalphazero.mcts.batch_runner import uniform_stub_batch
from tinyalphazero.replay import ReplayBuffer
from tinyalphazero.selfplay import play_games


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", type=Path, default=None, help="Checkpoint to play with")
    ap.add_argument("--stub", action="store_true", help="Zero-cost evaluator instead of a net")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--sims", type=int, default=None, help="Simulations per move")
    ap.add_argument("--out", type=Path, required=True, help="Run directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None, help="default: the config's seed")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--capacity", type=int, default=None, help="Replay capacity (default: config)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.ckpt is None and not args.stub:
        print("error: pass --ckpt or --stub")
        return 2

    if args.stub:
        evaluate_batch = uniform_stub_batch
        which = "stub"
    else:
        import torch

        from tinyalphazero.mcts.batch_runner import torch_evaluate_batch
        from tinyalphazero.train import load_checkpoint

        device = torch.device(args.device)
        net, _meta = load_checkpoint(args.ckpt, device=device)
        net.eval()
        evaluate_batch = torch_evaluate_batch(net, device)
        which = str(args.ckpt)

    args.out.mkdir(parents=True, exist_ok=True)
    capacity = args.capacity or cfg.train.replay_capacity
    buffer = ReplayBuffer.open_or_create(args.out / "replay", capacity)
    seed = cfg.seed if args.seed is None else args.seed

    _traces, stats = play_games(
        evaluate_batch,
        args.games,
        cfg=cfg,
        seed=seed,
        num_sims=args.sims,
        device=args.device,
        iteration=args.iteration,
        buffer=buffer,
    )

    row = {"eval": which, "sims": args.sims or cfg.selfplay.sims_per_move, **stats.to_dict()}
    with (args.out / "selfplay.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")

    print(
        f"{stats.games} games, {stats.positions} positions in {stats.seconds:.1f}s "
        f"({row['games_per_hour']} games/h) | eval={which}"
    )
    print(f"  results: {stats.results}")
    print(
        f"  resign-disabled {stats.resign_disabled_games}, would-have-resigned "
        f"{stats.would_have_resigned}, false-resign rate {stats.false_resign_rate}"
    )
    print(f"  replay: {len(buffer)} positions at {args.out / 'replay'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
