#!/usr/bin/env python3
"""Generate Phase 1 training data: random-play games and mate-in-1 anchors (spec §5)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tinyalphazero.data import generate_mate_in_1, generate_random_games, write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 1 training data")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--games", type=int, metavar="N", help="Generate N random-play games")
    mode.add_argument(
        "--mate1", type=int, metavar="N", help="Generate N mate-in-1 anchor positions"
    )
    parser.add_argument("--out", required=True, metavar="DIR", help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--min-plies", type=int, default=10)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    t0 = time.perf_counter()

    if args.games is not None:
        records = generate_random_games(
            args.games, rng, max_plies=args.max_plies, min_plies=args.min_plies
        )
        cfg = {
            "mode": "random_games",
            "num_games": args.games,
            "seed": args.seed,
            "max_plies": args.max_plies,
            "min_plies": args.min_plies,
        }
        meta = write_dataset(records, out_dir, generation_config=cfg)
        elapsed = time.perf_counter() - t0
        print(
            f"Generated {meta.n} positions from {args.games} games "
            f"in {elapsed:.1f}s → {out_dir}"
        )
    else:
        records = generate_mate_in_1(args.mate1, rng)
        cfg = {"mode": "mate_in_1", "num_positions": args.mate1, "seed": args.seed}
        meta = write_dataset(records, out_dir, generation_config=cfg)
        elapsed = time.perf_counter() - t0
        print(
            f"Generated {meta.n} mate-in-1 positions in {elapsed:.1f}s → {out_dir}"
        )


if __name__ == "__main__":
    main()
