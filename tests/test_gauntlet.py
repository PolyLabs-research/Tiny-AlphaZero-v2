"""Game rules and the anchored rating (tinyalphazero.gauntlet), with scripted engines."""

from __future__ import annotations

import math

import chess
import chess.engine
import numpy as np
import pytest

from tinyalphazero.gauntlet import (
    Clock,
    anchored_rating,
    bootstrap_rating,
    expected_score,
    performance,
    play_game,
)

START = chess.STARTING_FEN


class Scripted:
    """Plays a fixed list of moves, in order, whatever the position."""

    def __init__(self, moves: list[str]) -> None:
        self.moves = list(moves)

    def play(self, board: chess.Board, limit: chess.engine.Limit, game: object = None):
        return chess.engine.PlayResult(chess.Move.from_uci(self.moves.pop(0)), None)


class FakeTime:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_clock_parse() -> None:
    assert Clock.parse("60+0.6") == Clock(60.0, 0.6)
    assert Clock.parse("120") == Clock(120.0, 0.0)
    assert Clock(60.0, 0.6).pgn() == "60+0.6"


def test_checkmate_is_scored_for_the_network() -> None:
    # Fool's mate with the network as Black.
    ours, theirs = Scripted(["e7e5", "d8h4"]), Scripted(["f2f3", "g2g4"])
    rec = play_game(ours, theirs, START, ours_white=False, clock=Clock(60, 0), game_key=1)
    assert (rec.result, rec.termination, rec.score) == ("0-1", "checkmate", 1.0)
    assert rec.moves == ["f2f3", "e7e5", "g2g4", "d8h4"]
    assert (rec.ours_moves, rec.theirs_moves) == (2, 2)


def test_threefold_ends_the_game_only_once_it_has_happened() -> None:
    shuffle_w = ["g1f3", "f3g1", "g1f3", "f3g1"]
    shuffle_b = ["g8f6", "f6g8", "g8f6", "f6g8"]
    rec = play_game(Scripted(shuffle_w), Scripted(shuffle_b), START, True, Clock(60, 0), game_key=1)
    # The start position occurs at plies 0, 4 and 8: drawn on the 8th ply, not before.
    assert (rec.result, rec.termination, rec.plies) == ("1/2-1/2", "threefold_repetition", 8)


def test_illegal_move_loses() -> None:
    rec = play_game(Scripted(["e2e5"]), Scripted([]), START, True, Clock(60, 0), game_key=1)
    assert (rec.result, rec.termination, rec.score) == ("0-1", "error", 0.0)


def test_flag_fall_loses_and_increment_is_credited() -> None:
    now = FakeTime()

    class Slow(Scripted):
        def __init__(self, moves: list[str], seconds: float) -> None:
            super().__init__(moves)
            self.seconds = seconds

        def play(self, board, limit, game=None):
            now.t += self.seconds
            return super().play(board, limit, game)

    # 15+5: White spends 12 s a move. After move 1: 15-12+5 = 8; move 2: 8-12 < 0.
    ours = Slow(["g8f6", "f6g8", "g8f6"], 0.0)
    theirs = Slow(["g1f3", "f3g1", "g1f3"], 12.0)
    rec = play_game(ours, theirs, START, False, Clock(15, 5), game_key=1, now=now)
    assert (rec.result, rec.termination, rec.score) == ("0-1", "time_forfeit", 1.0)
    assert rec.plies == 2  # the flagging move is not played


def test_flag_fall_against_a_bare_king_is_a_draw() -> None:
    now = FakeTime()

    class Slow(Scripted):
        def play(self, board, limit, game=None):
            now.t += 100.0
            return super().play(board, limit, game)

    fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"  # White K+P to move, Black bare king
    rec = play_game(Slow(["e2e4"]), Scripted([]), fen, True, Clock(1, 0), game_key=1, now=now)
    assert (rec.result, rec.termination) == ("1/2-1/2", "time_forfeit_draw")


def test_anchored_rating_equals_performance_against_one_opponent() -> None:
    for score in (0.2, 0.5, 0.73):
        n = 100
        scores = [1.0] * int(score * n) + [0.0] * (n - int(score * n))
        rating, clamped = anchored_rating([1800.0] * n, scores)
        s = sum(scores) / n
        assert not clamped
        assert rating == pytest.approx(1800 + 400 * math.log10(s / (1 - s)), abs=1e-6)


def test_anchored_rating_solves_the_likelihood_equation() -> None:
    opp = np.array([1500.0] * 40 + [1900.0] * 40 + [2300.0] * 40)
    scores = np.array([0.9] * 40 + [0.5] * 40 + [0.1] * 40)
    rating, _ = anchored_rating(opp, scores)
    assert float(np.sum(expected_score(rating, opp))) == pytest.approx(scores.sum(), abs=1e-6)
    assert rating == pytest.approx(1900.0, abs=1e-6)  # symmetric design


def test_sweeps_are_clamped_not_infinite() -> None:
    assert anchored_rating([1500.0] * 4, [1.0] * 4)[1] is True
    assert anchored_rating([1500.0] * 4, [0.0] * 4)[1] is True
    with pytest.raises(ValueError):
        anchored_rating([], [])


def test_bootstrap_is_reproducible_and_brackets_the_point() -> None:
    rng = np.random.default_rng(1)
    groups = [
        (1500.0, rng.choice([0.0, 0.5, 1.0], size=50, p=[0.1, 0.3, 0.6])),
        (1900.0, rng.choice([0.0, 0.5, 1.0], size=50, p=[0.4, 0.3, 0.3])),
    ]
    a = bootstrap_rating(groups, resamples=500, seed=0)
    b = bootstrap_rating(groups, resamples=500, seed=0)
    assert a == b
    assert a["ci95"][0] < a["rating"] < a["ci95"][1]
    assert a["games"] == 200


def test_performance_reports_score_and_games() -> None:
    p = performance(2000.0, np.array([0.5, 0.5, 1.0, 0.0]))
    assert p["games"] == 8 and p["score"] == 0.5
    assert p["performance"] == pytest.approx(2000.0, abs=1e-6)


def test_a_silent_engine_is_closed_and_forfeits() -> None:
    """The watchdog: python-chess would wait forever on an engine that never answers."""
    import threading

    class Silent:
        def __init__(self) -> None:
            self.closed = threading.Event()

        def play(self, board, limit, game=None):
            assert self.closed.wait(timeout=10.0), "watchdog never fired"
            raise chess.engine.EngineTerminatedError("closed by watchdog")

        def close(self) -> None:
            self.closed.set()

    silent = Silent()
    rec = play_game(
        Scripted(["e2e4"]), silent, START, True, Clock(0.05, 0), game_key=1, hang_grace=0.05
    )
    assert silent.closed.is_set()
    assert (rec.result, rec.termination, rec.score) == ("1-0", "hung", 1.0)
