#!/usr/bin/env python3
"""Generate the fixed 4-ply opening book for the anchor ladder (spec §8).

100 positions reached by 4 random legal plies from the start, seed 42, committed.
Every ladder matchup plays each of them **twice with colours swapped**, so the
book is 200 games and both sides get every opening.

Why a fixed book at all: two engines that both play deterministically from the
start position play one game, not two hundred. The book is the variance the match
needs, and freezing it is what makes two evaluations of different checkpoints
comparable — a resampled book would confound "the checkpoint changed" with "the
openings changed", which is exactly the confound the anchor ladder exists to
remove.

Constraints asserted at generation *and* re-asserted in the tests, because a book
is committed once and read for the rest of the project:

* exactly ``--count`` positions, deduplicated by EPD;
* none terminal (checkmate, stalemate, insufficient material) — a terminal
  opening is a game with no moves in it;
* none in check — the side to move would be answering a threat rather than
  choosing an opening, and the colour-swapped twin would not be its mirror in any
  meaningful sense;
* both sides retain material and legal moves, i.e. 4 plies is 4 plies.

    python scripts/generate_openings.py --count 100 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np

PLIES = 4


def random_opening(rng: np.random.Generator, plies: int = PLIES) -> chess.Board | None:
    """Walk ``plies`` uniformly-random legal moves. None if the walk is unusable."""
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            return None
        board.push(moves[int(rng.integers(len(moves)))])
    if board.is_game_over() or board.is_check():
        return None
    return board


def generate(count: int, seed: int, plies: int = PLIES) -> list[chess.Board]:
    rng = np.random.default_rng(seed)
    out: list[chess.Board] = []
    seen: set[str] = set()
    while len(out) < count:
        board = random_opening(rng, plies)
        if board is None:
            continue
        epd = board.epd()
        if epd in seen:
            continue
        seen.add(epd)
        out.append(board)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plies", type=int, default=PLIES)
    ap.add_argument("--out", type=Path, default=Path("tests/suites/openings_4ply.epd"))
    args = ap.parse_args()

    boards = generate(args.count, args.seed, args.plies)
    for i, b in enumerate(boards):
        assert not b.is_game_over(), f"opening {i} is terminal"
        assert not b.is_check(), f"opening {i} is in check"
        assert b.legal_moves.count() > 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'{b.epd()} id "open_{i + 1:03d}";' for i, b in enumerate(boards)]
    args.out.write_text("\n".join(lines) + "\n")
    white_to_move = sum(b.turn == chess.WHITE for b in boards)
    print(
        f"wrote {len(lines)} openings ({args.plies} plies, seed {args.seed}) to {args.out}; "
        f"{white_to_move} with white to move"
    )


if __name__ == "__main__":
    main()
