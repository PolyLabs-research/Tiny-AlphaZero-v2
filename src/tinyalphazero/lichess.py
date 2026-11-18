"""Positions from Lichess's open game database, as a ring to label (PREREG-p2_ac).

p2_ac needs more distinct positions than this project's self-play rings hold, and takes
the rest from rated human games (the source DeepMind's searchless 9M model was trained
on, there at a far larger scale). A game is kept by its headers alone
(:func:`header_ok`), then up to ``k`` of its positions are sampled
(:func:`sample_positions`) and stored as replay rows: the position's tokens (with its
repetition count from the game's own history), its legal-move mask, the move the human
played from it, and the game's result from the side to move. Labelling replaces the
human move with the teacher's (``scripts/label_teacher.py``); the result stays as the
row's outcome, as a self-play row's does.

Pure functions over one game's text, so the extraction can run them in worker processes
and the tests can run them on a handful of games.
"""

from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass
from typing import Any

import chess
import chess.pgn
import numpy as np

RESULTS = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

# Rows from human games are tagged in the ring's iteration field: self-play rows are >= 0, rehearsal
# anchors -1, teacher rows -2 (teacher.TEACHER_ITERATION).
LICHESS_ITERATION = -3
SOURCES = ("lichess", "selfplay")


def source_rows(tags: np.ndarray, source: str, limit: int | None = None) -> np.ndarray:
    """The slots of a pooled ring whose rows came from ``source``, in ring order (PREREG-p2_ad).

    ``tags`` is the ring's ``iteration`` field as pooled (``pool_positions`` keeps each
    row's source tag): ``LICHESS_ITERATION`` for a human-game row, an iteration ``>= 0``
    for a self-play row. Any other tag raises: such a ring is not a pool of these two
    sources. ``limit`` takes the first ``limit`` such slots and raises if there are fewer.
    """
    tags = np.asarray(tags)
    lichess = tags == LICHESS_ITERATION
    selfplay = tags >= 0
    other = int(len(tags) - lichess.sum() - selfplay.sum())
    if other:
        raise ValueError(f"{other} rows carry a tag that is neither Lichess nor self-play")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, not {source!r}")
    rows = np.flatnonzero(lichess if source == "lichess" else selfplay)
    if limit is not None:
        if limit > len(rows):
            raise ValueError(f"{source}: {len(rows)} rows, fewer than the limit {limit}")
        rows = rows[:limit]
    return rows


@dataclass(frozen=True)
class Filters:
    """Which games are kept, by their headers; and which of their positions are sampled."""

    min_elo: int = 1600  # both players
    min_base_s: int = 180  # no bullet; correspondence ("-") excluded
    min_plies: int = 30
    first_ply: int = 10  # positions before this ply are opening book territory
    per_game: int = 10


def base_seconds(time_control: str) -> int | None:
    """``"300+3"`` -> 300; ``"-"`` or anything unreadable -> None."""
    head = time_control.split("+", 1)[0]
    return int(head) if head.isdigit() else None


def header_ok(headers: dict[str, str], f: Filters) -> bool:
    """A rated standard game between two players at ``min_elo``+, at blitz or slower, ended
    on the board (``Termination: Normal``), with a decisive or drawn result."""
    if headers.get("Result") not in RESULTS:
        return False
    if headers.get("Termination", "Normal") != "Normal":
        return False
    if headers.get("Variant", "Standard") != "Standard":
        return False
    base = base_seconds(headers.get("TimeControl", "-"))
    if base is None or base < f.min_base_s:
        return False
    try:
        return min(int(headers["WhiteElo"]), int(headers["BlackElo"])) >= f.min_elo
    except (KeyError, ValueError):
        return False


def sample_positions(game_text: str, f: Filters, seed: int) -> list[dict[str, Any]]:
    """Up to ``f.per_game`` positions of one game, sampled without replacement (seeded).

    Only positions from ``f.first_ply`` on that have a move after them, so every row carries
    the move played. Returns rows as ``ReplayBuffer.add`` keywords; empty if the game is
    shorter than ``f.min_plies`` or does not parse.
    """
    from tinyalphazero.encoding import encode_board, encode_move, legal_moves_packed

    game = chess.pgn.read_game(io.StringIO(game_text))
    if game is None or game.errors:
        return []
    z = RESULTS.get(game.headers.get("Result", ""))
    moves = list(game.mainline_moves())
    if z is None or len(moves) < f.min_plies:
        return []
    candidates = list(range(f.first_ply, len(moves)))
    rng = np.random.default_rng(seed)
    chosen = set(
        rng.choice(candidates, size=min(f.per_game, len(candidates)), replace=False).tolist()
    )
    rows: list[dict[str, Any]] = []
    board = game.board()
    seen: Counter = Counter()
    for ply, move in enumerate(moves):
        key = board._transposition_key()
        if ply in chosen:
            fr, to, promo = encode_move(move)
            mover_white = board.turn == chess.WHITE
            rows.append(
                {
                    "tokens": encode_board(board, seen[key]),
                    "legals": legal_moves_packed(board),
                    "move_ft": fr * 64 + to,
                    "promo": promo,
                    "outcome": z if mover_white else -z,
                    "improved_policy": [(fr * 64 + to, promo, 1.0)],
                }
            )
        seen[key] += 1
        board.push(move)
    return rows
