"""A rating on a public scale: games against engines with published ratings.

The ladder's numbers are scores against fixed opponents under this repo's own
protocol; they rank checkpoints but say nothing about where the network stands
among chess programs. This module plays the network (through its UCI front end,
``scripts/uci_engine.py``) against opponents whose ratings are published on one
scale -- CCRL -- and fits the one rating that best explains the results.

Two pieces, both pure enough to test without a chess engine on the box:

* :func:`play_game` -- one game from a book position under a Fischer clock,
  between two objects with python-chess's ``play(board, limit, game=...)``
  method. Rules end the game (mate, stalemate, insufficient material, and a
  threefold repetition or fifty-move count once it has occurred); a flag fall
  loses unless the opponent cannot mate; nothing is adjudicated on engine
  scores.
* :func:`anchored_rating` / :func:`bootstrap_rating` -- the maximum-likelihood
  rating against opponents of known rating (draws as half points, the model
  Ordo fits), and a percentile interval resampling **opening pairs** within each
  opponent, the unit the design randomises over (as ``paired_bootstrap_elo``).

The opponents' ratings are inputs, not estimates: their own published error
bars are a systematic term the interval does not contain.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import chess
import numpy as np

# Hard stop for a game nobody ends: python-chess claims every draw the rules allow,
# so this binds only on a pathological engine pair. Recorded, never silent.
MAX_PLIES = 600
RATING_BOUNDS = (-1000.0, 5000.0)
# Seconds past its own flag before a silent engine is closed and forfeits.
HANG_GRACE_S = 30.0


@dataclass(frozen=True)
class Clock:
    """Fischer time control: ``base`` seconds plus ``inc`` per move made."""

    base: float
    inc: float

    @classmethod
    def parse(cls, text: str) -> Clock:
        """``"60+0.6"`` -> Clock(60.0, 0.6)."""
        base, _, inc = text.partition("+")
        return cls(float(base), float(inc or 0.0))

    def pgn(self) -> str:
        return f"{self.base:g}+{self.inc:g}"


@dataclass
class GameRecord:
    """One finished game, scored from the network's side."""

    start_fen: str
    ours_white: bool
    moves: list[str] = field(default_factory=list)
    result: str = "*"  # PGN result, White's perspective
    termination: str = ""
    score: float = 0.5  # the network's score: 1, 0.5, 0
    ours_seconds: float = 0.0
    theirs_seconds: float = 0.0
    ours_moves: int = 0
    theirs_moves: int = 0

    @property
    def plies(self) -> int:
        return len(self.moves)


def _score_for_white(result: str) -> float:
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]


def play_game(
    ours: Any,
    theirs: Any,
    start_fen: str,
    ours_white: bool,
    clock: Clock,
    game_key: object,
    now: Callable[[], float] = time.monotonic,
    max_plies: int = MAX_PLIES,
    hang_grace: float = HANG_GRACE_S,
) -> GameRecord:
    """Play one game from ``start_fen``; ``ours``/``theirs`` are python-chess engines.

    Each side's clock is charged the wall time its ``play`` call took and
    credited ``clock.inc`` after every move it makes. ``game_key`` is passed to
    python-chess, which sends ``ucinewgame`` / ``new`` whenever it changes, so no
    engine carries hash or state from one game into the next.

    An engine that raises, returns no move, or returns an illegal move loses
    (termination ``"error"``) -- a benchmark that quietly substituted a move
    would no longer be measuring the engine.
    """
    import chess.engine

    board = chess.Board(start_fen)
    rec = GameRecord(start_fen=board.fen(), ours_white=ours_white)
    remaining = {chess.WHITE: clock.base, chess.BLACK: clock.base}
    ours_side = chess.WHITE if ours_white else chess.BLACK

    def finish(result: str, termination: str) -> GameRecord:
        rec.result, rec.termination = result, termination
        white_score = _score_for_white(result)
        rec.score = white_score if ours_white else 1.0 - white_score
        return rec

    def loss_for(side: chess.Color) -> str:
        return "0-1" if side == chess.WHITE else "1-0"

    start_ply = board.ply()
    while True:
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            return finish(outcome.result(), outcome.termination.name.lower())
        # Claimable draws once they have *happened*, as match software ends them:
        # python-chess's claim_draw=True also ends a game whenever the side to move
        # merely could repeat, which no arbiter and no engine is obliged to do.
        if board.is_repetition(3):
            return finish("1/2-1/2", "threefold_repetition")
        if board.halfmove_clock >= 100:
            return finish("1/2-1/2", "fifty_moves")
        if board.ply() - start_ply >= max_plies:
            return finish("1/2-1/2", "max_plies")

        mover = board.turn
        engine = ours if mover == ours_side else theirs
        limit = chess.engine.Limit(
            white_clock=max(0.0, remaining[chess.WHITE]),
            black_clock=max(0.0, remaining[chess.BLACK]),
            white_inc=clock.inc,
            black_inc=clock.inc,
        )
        # python-chess arms no watchdog for a clock-only limit (it waits forever on a
        # silent engine; REVIEW-2026-10-21 D12 met the same thing). An engine still
        # thinking `hang_grace` seconds past its own flag is closed from a timer,
        # which ends the pending `play` with EngineTerminatedError.
        watchdog, fired = None, threading.Event()
        if hasattr(engine, "close"):

            def fire(engine: Any = engine, fired: threading.Event = fired) -> None:
                fired.set()
                engine.close()

            watchdog = threading.Timer(max(0.0, remaining[mover]) + hang_grace, fire)
            watchdog.daemon = True
            watchdog.start()
        t0 = now()
        try:
            move = engine.play(board, limit, game=game_key).move
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
            move = None
        finally:
            if watchdog is not None:
                watchdog.cancel()
        spent = now() - t0
        hung = fired.is_set()
        if mover == ours_side:
            rec.ours_seconds += spent
            rec.ours_moves += 1
        else:
            rec.theirs_seconds += spent
            rec.theirs_moves += 1

        if hung:
            # Past its flag by `hang_grace`: a time forfeit, recorded as `hung` so
            # the runner restarts the engine and the report can count them.
            if board.has_insufficient_material(not mover):
                return finish("1/2-1/2", "hung_draw")
            return finish(loss_for(mover), "hung")
        if move is None or move not in board.legal_moves:
            return finish(loss_for(mover), "error")
        remaining[mover] -= spent
        if remaining[mover] < 0.0:
            # A flag fall is a loss unless the side with time cannot possibly mate.
            if board.has_insufficient_material(not mover):
                return finish("1/2-1/2", "time_forfeit_draw")
            return finish(loss_for(mover), "time_forfeit")
        remaining[mover] += clock.inc
        rec.moves.append(move.uci())
        board.push(move)


# ---------------------------------------------------------------------------
# Rating against anchors of known rating
# ---------------------------------------------------------------------------


def expected_score(rating: float, opponent: np.ndarray | float) -> np.ndarray | float:
    """Logistic expected score of ``rating`` against ``opponent`` (Elo's 400 scale)."""
    return 1.0 / (1.0 + 10.0 ** ((np.asarray(opponent, dtype=np.float64) - rating) / 400.0))


def anchored_rating(
    opponent_ratings: Sequence[float], scores: Sequence[float]
) -> tuple[float, bool]:
    """Maximum-likelihood rating from games against opponents of known rating.

    Draws count as half a win and half a loss, so the likelihood's stationary
    point is where the expected total equals the actual total:
    ``sum(E(R, r_i)) == sum(s_i)``. The left side rises monotonically in ``R``,
    so bisection finds the unique root. Returns ``(rating, clamped)``: a clean
    sweep (or whitewash) has no finite maximum and is reported at the bound with
    ``clamped=True``, as ``elo_from_score`` reports one.
    """
    opp = np.asarray(opponent_ratings, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if opp.size == 0 or opp.size != s.size:
        raise ValueError("need one score per game and at least one game")
    total = float(s.sum())
    lo, hi = RATING_BOUNDS
    if total <= 0.0:
        return lo, True
    if total >= opp.size:
        return hi, True
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if float(np.sum(expected_score(mid, opp))) < total:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


def bootstrap_rating(
    groups: Sequence[tuple[float, np.ndarray]],
    resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """The anchored rating with a percentile interval over opening pairs.

    ``groups`` holds one ``(opponent_rating, pair_scores)`` per opponent, where
    ``pair_scores[i]`` is the network's mean over opening ``i``'s two
    colour-swapped games. Each resample draws that opponent's openings with
    replacement, independently per opponent, and refits. A pair contributes two
    games at its mean score, which leaves the point estimate unchanged.
    """
    rng = np.random.default_rng(seed)
    ratings = [float(r) for r, _ in groups]
    pairs = [np.asarray(p, dtype=np.float64) for _, p in groups]

    def fit(samples: list[np.ndarray]) -> tuple[float, bool]:
        opp = np.concatenate(
            [np.full(2 * len(p), r) for r, p in zip(ratings, samples, strict=True)]
        )
        sc = np.concatenate([np.repeat(p, 2) for p in samples])
        return anchored_rating(opp, sc)

    point, clamped = fit(pairs)
    draws = np.empty(resamples)
    for k in range(resamples):
        draws[k] = fit([p[rng.integers(0, len(p), size=len(p))] for p in pairs])[0]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "rating": point,
        "clamped": clamped,
        "ci95": [float(lo), float(hi)],
        "se": float(draws.std(ddof=1)),
        "resamples": resamples,
        "seed": seed,
        "games": int(sum(2 * len(p) for p in pairs)),
    }


def performance(opponent_rating: float, pair_scores: np.ndarray) -> dict[str, Any]:
    """One opponent's result as a performance rating (anchored fit on its games alone)."""
    p = np.asarray(pair_scores, dtype=np.float64)
    rating, clamped = anchored_rating(np.full(2 * len(p), opponent_rating), np.repeat(p, 2))
    return {
        "opponent_rating": opponent_rating,
        "games": 2 * len(p),
        "score": float(p.mean()) if len(p) else math.nan,
        "performance": rating,
        "clamped": clamped,
    }
