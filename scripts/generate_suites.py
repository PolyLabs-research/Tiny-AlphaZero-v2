#!/usr/bin/env python3
"""Generate the mate-in-1 and mate-in-2 EPD suites (plan chunk 5).

Random play, then python-chess verification of the mate distance. The suites are
committed, so this script exists for provenance and regeneration, not as part of
any gate.

Usage:
    VIRTUAL_ENV=.venv .venv/bin/python scripts/generate_suites.py \
        --mate1 100 --mate2 200 --out tests/suites --seed 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np


def _mate_in_1_moves(board: chess.Board) -> list[chess.Move]:
    """Moves that deliver immediate checkmate."""
    out = []
    for mv in board.legal_moves:
        board.push(mv)
        if board.is_checkmate():
            out.append(mv)
        board.pop()
    return out


def _is_mate_in_2(board: chess.Board) -> chess.Move | None:
    """Return a move forcing mate in 2 (our move, any reply, our mate), else None.

    Requires that the position is *not* already mate-in-1: a suite position whose
    answer is a one-mover would test nothing beyond the mate-in-1 suite.
    """
    if _mate_in_1_moves(board):
        return None
    for mv in board.legal_moves:
        board.push(mv)
        if board.is_checkmate() or board.is_stalemate():
            board.pop()
            continue
        replies = list(board.legal_moves)
        forced = bool(replies)
        for reply in replies:
            board.push(reply)
            if not _mate_in_1_moves(board):
                forced = False
            board.pop()
            if not forced:
                break
        board.pop()
        if forced:
            return mv
    return None


def _random_positions(rng: np.random.Generator, max_plies: int = 160):
    """Walk random games, yielding every position encountered."""
    while True:
        board = chess.Board()
        for _ in range(max_plies):
            if board.is_game_over():
                break
            yield board.copy()
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])


def _write_epd(path: Path, entries: list[tuple[chess.Board, chess.Move, str]]) -> None:
    lines = []
    for board, mv, ident in entries:
        san = board.san(mv)
        lines.append(f'{board.epd()} bm {san}; id "{ident}";')
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mate1", type=int, default=100)
    ap.add_argument("--mate2", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("tests/suites"))
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    m1: list[tuple[chess.Board, chess.Move, str]] = []
    m2: list[tuple[chess.Board, chess.Move, str]] = []
    seen: set[str] = set()

    for board in _random_positions(rng):
        if len(m1) >= args.mate1 and len(m2) >= args.mate2:
            break
        key = board.epd()
        if key in seen:
            continue

        mates = _mate_in_1_moves(board)
        if mates and len(m1) < args.mate1:
            seen.add(key)
            m1.append((board, mates[0], f"mate1_{len(m1) + 1:03d}"))
            continue
        if not mates and len(m2) < args.mate2:
            # Cheap gate first: mate-in-2 search is O(b^3), so only pay it for
            # positions where the side to move already has the opponent in trouble.
            if len(list(board.legal_moves)) > 40:
                continue
            mv = _is_mate_in_2(board)
            if mv is not None:
                seen.add(key)
                m2.append((board, mv, f"mate2_{len(m2) + 1:03d}"))

    _write_epd(args.out / "mate_in_1.epd", m1)
    _write_epd(args.out / "mate_in_2.epd", m2)
    print(f"wrote {len(m1)} mate-in-1 and {len(m2)} mate-in-2 positions to {args.out}")

    # Guarded: --mate1 0 is a legitimate call (harvest a mate-in-2 tuning suite
    # only), and this stats line used to crash on the empty list *after* both
    # files were already written — a traceback on a run that had in fact
    # succeeded, which is the kind of thing that gets a good artefact thrown away.
    branching = [len(list(b.legal_moves)) for b, _, _ in m1]
    if branching:
        print(
            f"mate-in-1 legal-move counts: min={min(branching)} median="
            f"{int(np.median(branching))} max={max(branching)}"
        )


if __name__ == "__main__":
    main()
