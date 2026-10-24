"""The random-opening lever (``selfplay.opening_random_plies``, PREREG-p2_g).

Self-play reached 2 distinct positions after one ply and none of the skill-0 rung's
100 openings in p2_f's whole ring (RUNLOG-p2_f §6). The lever starts games k random
plies in. What must hold: off is exactly the old behaviour; on, the draws are
per-game reproducible, never a finished game and never one of the rung's openings;
and the collapse detector still reads the initial position only.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

from tinyalphazero.config import Config
from tinyalphazero.encoding import encode_board
from tinyalphazero.mcts.batch_runner import uniform_stub_batch
from tinyalphazero.replay import ReplayBuffer
from tinyalphazero.selfplay import _excluded_starts, play_games, wave_specs


def _sp(plies: int, exclude: str | None = None):
    sp = Config().selfplay
    sp.opening_random_plies = plies
    if exclude is not None:
        sp.opening_exclude = exclude
    return sp


def test_off_is_the_initial_position_for_every_game() -> None:
    assert all(s.start_fen is None for s in wave_specs(7, 64, _sp(0)))


def test_on_is_reproducible_bounded_unfinished_and_never_a_book_opening() -> None:
    sp = _sp(8)
    specs = wave_specs(12345, 256, sp)
    assert [s.start_fen for s in specs] == [s.start_fen for s in wave_specs(12345, 256, sp)]
    book = _excluded_starts(sp.opening_exclude)
    assert len(book) == 100
    plies = []
    for s in specs:
        board = chess.Board(s.start_fen) if s.start_fen else chess.Board()
        plies.append(board.ply())
        assert not board.is_game_over()
        assert board.epd() not in book
    assert min(plies) == 0 and max(plies) == 8
    assert len(set(plies)) == 9, "every k in 0..8 should occur in 256 draws"


def test_the_lever_leaves_the_waves_other_draws_alone() -> None:
    """Seeds and resignation flags are the same with the lever on or off."""
    off, on = wave_specs(99, 32, _sp(0)), wave_specs(99, 32, _sp(6))
    assert [(s.seed, s.resign_disabled) for s in off] == [(s.seed, s.resign_disabled) for s in on]


def test_an_excluded_start_is_redrawn(tmp_path: Path) -> None:
    """Exclude every one-ply position: with k in {0, 1}, every game must start at k = 0."""
    epd = tmp_path / "one_ply.epd"
    lines = []
    for move in chess.Board().legal_moves:
        b = chess.Board()
        b.push(move)
        lines.append(f'{b.epd()} id "p{move.uci()}";')
    epd.write_text("\n".join(lines) + "\n")
    specs = wave_specs(5, 64, _sp(1, exclude=str(epd)))
    assert all(s.start_fen is None for s in specs)


def test_games_are_searched_and_stored_from_their_start_and_h1_reads_the_initial_position(
    tmp_path: Path,
) -> None:
    cfg = Config()
    cfg.selfplay.opening_random_plies, cfg.selfplay.max_plies = 6, 8
    specs = wave_specs(31, 12, cfg.selfplay)
    assert any(s.start_fen for s in specs) and any(s.start_fen is None for s in specs)
    buf = ReplayBuffer.create(tmp_path / "r", capacity=1024)
    traces, stats = play_games(uniform_stub_batch, 12, cfg=cfg, seed=31, num_sims=4, buffer=buf)

    row = 0
    for spec, trace in zip(specs, traces, strict=True):
        start = chess.Board(spec.start_fen) if spec.start_fen else chess.Board()
        assert np.array_equal(buf._arr["boards"][row], encode_board(start, 0))
        row += len(trace.positions)
    from_initial = sum(1 for s in specs if s.start_fen is None)
    assert len(stats.move1_prior_entropies) == from_initial
