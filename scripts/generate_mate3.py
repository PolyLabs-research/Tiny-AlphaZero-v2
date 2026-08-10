#!/usr/bin/env python3
"""Harvest the mate-in-3 suite (plan chunk 10, spec §8).

Mate-in-3 is not mate-in-2 with a bigger number. Verification is a 5-ply forced
search — our move, any reply, our move, any reply, our mate — so the naive cost is
O(b^5), and at the b≈30 of a random-play position that is tens of millions of
nodes *per candidate*. Two things make it tractable, and both are stated here
because both bias what ends up in the suite:

1. **Candidates are pre-filtered by branching factor** (``--max-branch``, default
   24). Cheap, and the same trick the mate-in-2 harvest used at 40. It biases the
   suite toward constrained positions — which is where forced mates live anyway.
2. **The mating side's moves are searched checks-first** (``_ordered``). This is
   an ordering, not a restriction: a quiet mating move is still searched, just
   later. No position is excluded by it, so the suite stays sound; it only makes
   the common case return early.

The defender's replies are *never* restricted — that would be unsound, because
missing one reply turns "forced" into "hoped for".

Positions that are already mate-in-1 or mate-in-2 are rejected: a suite entry
whose answer is a shorter mate measures the shorter suite.

This suite is **informational trajectory, not a gate** (§8). It is committed, so
this script exists for provenance and regeneration.

    python scripts/generate_mate3.py --count 200 --seed 1010 --workers 14
    python scripts/generate_mate3.py --timing-sample 20     # project before committing hours
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import numpy as np

from tinyalphazero.mcts.search import _forces_mate_in_3, _ordered_moves, can_force_mate


def is_mate_in_3(board: chess.Board) -> chess.Move | None:
    """First move of a forced mate in **exactly** 3, or None.

    "Exactly": positions already solvable in 1 or 2 are rejected, so the suite
    cannot be beaten by a shallower search than it advertises.

    The solver and the grader are the same code — ``can_force_mate`` and
    ``_forces_mate_in_3`` both come from ``mcts.search``. A suite whose generator
    and whose grader disagree about what "mate in 3" means measures neither.
    """
    if can_force_mate(board, 2):
        return None
    for mv in _ordered_moves(board):
        if _forces_mate_in_3(board, mv):
            return mv
    return None


def _random_positions(rng: np.random.Generator, max_plies: int = 160):
    while True:
        board = chess.Board()
        for _ in range(max_plies):
            if board.is_game_over():
                break
            yield board.copy()
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])


def _candidates(seed: int, max_branch: int, limit: int) -> list[str]:
    """Cheap pass: EPDs worth paying the 5-ply verification for."""
    rng = np.random.default_rng(seed)
    out: list[str] = []
    seen: set[str] = set()
    for board in _random_positions(rng):
        if len(out) >= limit:
            break
        if board.is_check():
            continue  # side to move is the one being mated, not the mater
        n = board.legal_moves.count()
        if n > max_branch or n == 0:
            continue
        epd = board.epd()
        if epd in seen:
            continue
        seen.add(epd)
        out.append(epd)
    return out


def _verify(epd: str) -> tuple[str, str] | None:
    board = chess.Board()
    board.set_epd(epd)
    mv = is_mate_in_3(board)
    return (epd, board.san(mv)) if mv is not None else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=200, help="Positions to harvest")
    ap.add_argument("--out", type=Path, default=Path("tests/suites/mate_in_3.epd"))
    ap.add_argument("--seed", type=int, default=1010, help="Fresh seed — must not be 5 or 42")
    ap.add_argument("--max-branch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument(
        "--timing-sample",
        type=int,
        default=0,
        help="Verify N candidates, report the rate and the projected full-run cost, "
        "and write nothing. Run this before committing hours to a harvest.",
    )
    ap.add_argument("--chunk", type=int, default=2000, help="Candidates per verification round")
    args = ap.parse_args()

    if args.timing_sample:
        cands = _candidates(args.seed, args.max_branch, args.timing_sample)
        t0 = time.perf_counter()
        with Pool(args.workers) as pool:
            hits = [r for r in pool.map(_verify, cands) if r is not None]
        dt = time.perf_counter() - t0
        rate = len(hits) / len(cands) if cands else 0.0
        per = dt / len(cands) if cands else 0.0
        print(
            f"timing sample: {len(cands)} candidates in {dt:.1f}s "
            f"({per * 1000:.0f} ms/candidate at {args.workers} workers), "
            f"{len(hits)} were mate-in-3 (hit rate {rate:.3f})"
        )
        if rate > 0:
            projected = args.count / rate * per
            print(f"projected for --count {args.count}: {projected / 60:.1f} min")
        else:
            print(f"projected for --count {args.count}: unknown — zero hits in the sample")
        return

    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    t0 = time.perf_counter()
    round_no = 0
    seed = args.seed
    while len(found) < args.count:
        round_no += 1
        cands = [e for e in _candidates(seed, args.max_branch, args.chunk) if e not in seen]
        seen.update(cands)
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(_verify, cands, chunksize=8):
                if r is not None:
                    found.append(r)
                    if len(found) >= args.count:
                        break
        print(
            f"round {round_no}: {len(found)}/{args.count} after {len(seen)} candidates, "
            f"{(time.perf_counter() - t0) / 60:.1f} min elapsed",
            flush=True,
        )
        seed += 1  # a fresh walk; `seen` keeps the suite deduplicated across rounds

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'{epd} bm {san}; id "mate3_{i + 1:03d}";'
        for i, (epd, san) in enumerate(found[: args.count])
    ]
    args.out.write_text("\n".join(lines) + "\n")
    print(
        f"wrote {len(lines)} mate-in-3 positions to {args.out} "
        f"in {(time.perf_counter() - t0) / 60:.1f} min "
        f"({len(seen)} candidates verified, seed {args.seed}, max_branch {args.max_branch})"
    )


if __name__ == "__main__":
    main()
