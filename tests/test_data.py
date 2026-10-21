"""Tests for Phase 1 data generation and Dataset (chunk 2, spec §5)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from tinyalphazero.data import (
    GameRecord,
    Phase1Dataset,
    PositionRecord,
    generate_mate_in_1,
    generate_random_games,
    write_dataset,
)
from tinyalphazero.encoding import (
    TURN_B,
    TURN_W,
    decode_board,
    decode_move,
    legal_moves_packed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _smoke_games(n: int = 50, seed: int = 0) -> list[GameRecord]:
    return list(generate_random_games(n, _rng(seed)))


# ---------------------------------------------------------------------------
# generate_random_games
# ---------------------------------------------------------------------------


def test_random_games_yields_game_records() -> None:
    games = _smoke_games(5)
    assert all(isinstance(g, GameRecord) for g in games)


def test_random_games_positions_have_correct_shapes() -> None:
    games = _smoke_games(10)
    for game in games:
        for pos in game.positions:
            assert pos.tokens.shape == (72,), "board tokens wrong shape"
            assert pos.legals.shape == (512,), "legals wrong shape"
            assert 0 <= pos.from_sq < 64
            assert 0 <= pos.to_sq < 64
            assert 0 <= pos.promo <= 4
            assert pos.outcome in (-1, 0, 1)


def test_random_games_min_plies_respected() -> None:
    # With min_plies=20, every game should have at least 20 positions
    games = list(generate_random_games(20, _rng(1), min_plies=20))
    for game in games:
        assert len(game.positions) >= 20, f"Game too short: {len(game.positions)}"


def _replay_to_end(game) -> chess.Board:
    """Rebuild the final board from a GameRecord's stored moves."""
    from tinyalphazero.encoding import decode_board, decode_move

    board = decode_board(game.positions[0].tokens)
    for pos in game.positions:
        board.push(decode_move(pos.from_sq, pos.to_sq, pos.promo))
    return board


def test_random_games_outcome_perspective() -> None:
    """Side-to-move perspective, checked against the *actual* winner (spec §5).

    The previous version asserted only that adjacent decisive positions alternate
    — which an inverted sign satisfies just as well — and the "known win" tests
    asserted a literal they had themselves constructed. A global sign flip in
    ``generate_random_games`` passed all three (REVIEW-2026-10-21 B2). Here the
    stored moves are replayed to the terminal board, the winner is read off it,
    and every position's outcome must be +1 exactly when its mover is the winner.
    """
    rng = _rng(42)
    decisive = drawn = 0
    for game in generate_random_games(600, rng):
        final = _replay_to_end(game)
        if final.is_checkmate():
            decisive += 1
            winner_is_white = not final.turn  # the side to move at a checkmate is mated
            for pos in game.positions:
                mover_is_white = int(pos.tokens[64]) == TURN_W
                expected = 1 if mover_is_white == winner_is_white else -1
                assert pos.outcome == expected, (pos.outcome, expected, final.fen())
            # The mover of the last stored position delivered the mate.
            assert game.positions[-1].outcome == 1
        else:
            drawn += 1
            assert all(p.outcome == 0 for p in game.positions), final.fen()
    assert decisive >= 5 and drawn >= 5, (decisive, drawn)


def test_random_games_draw_outcome_all_zero() -> None:
    """In a drawn game all outcomes must be 0 — checked on generated games, not a literal."""
    rng = _rng(7)
    seen = 0
    for game in generate_random_games(300, rng):
        if not _replay_to_end(game).is_checkmate():
            seen += 1
            assert all(p.outcome == 0 for p in game.positions)
    assert seen > 0, "no drawn game in 300 random games"


def test_random_games_tokens_in_vocab_range() -> None:
    from tinyalphazero.encoding import VOCAB_SIZE

    games = _smoke_games(20)
    for game in games:
        for pos in game.positions:
            assert pos.tokens.min() >= 0
            assert pos.tokens.max() < VOCAB_SIZE


# ---------------------------------------------------------------------------
# generate_mate_in_1
# ---------------------------------------------------------------------------


def test_mate_in_1_yields_position_records() -> None:
    positions = list(generate_mate_in_1(5, _rng(0)))
    assert all(isinstance(p, PositionRecord) for p in positions)
    assert len(positions) == 5


def test_mate_in_1_all_outcomes_plus_one() -> None:
    for pos in generate_mate_in_1(20, _rng(1)):
        assert pos.outcome == 1, "Mate-in-1 anchor must have outcome +1"


def test_mate_in_1_move_is_checkmate() -> None:
    """Push the stored mating move and verify it produces checkmate."""
    for pos in generate_mate_in_1(20, _rng(2)):
        board = decode_board(pos.tokens)
        move = decode_move(pos.from_sq, pos.to_sq, pos.promo)
        assert board.is_legal(move), f"Stored mating move {move} is illegal on decoded board"
        board.push(move)
        assert board.is_checkmate(), f"Stored mating move {move} did not produce checkmate"


def test_mate_in_1_correct_shapes() -> None:
    for pos in generate_mate_in_1(10, _rng(3)):
        assert pos.tokens.shape == (72,)
        assert pos.legals.shape == (512,)


# ---------------------------------------------------------------------------
# write_dataset / Phase1Dataset round-trip
# ---------------------------------------------------------------------------


def test_write_read_roundtrip_random_games() -> None:
    """Dataset write/read round-trip on a 50-game sample."""
    rng = _rng(10)
    games = list(generate_random_games(50, rng))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "data"
        meta = write_dataset(iter(games), out)

        assert meta.n > 0
        assert meta.version == "1"

        ds = Phase1Dataset(out)
        assert len(ds) == meta.n

        # Spot-check a few samples
        for idx in [0, 1, len(ds) // 2, len(ds) - 1]:
            item = ds[idx]
            assert "tokens" in item
            assert "from_sq" in item
            assert "to_sq" in item
            assert "promo" in item
            assert "legals" in item
            assert "outcome" in item
            assert item["tokens"].shape == (72,)
            assert item["legals"].shape == (512,)
            assert item["outcome"].item() in (-1, 0, 1)


def test_write_read_roundtrip_mate_in_1() -> None:
    """Dataset write/read round-trip for mate-in-1 anchors."""
    rng = _rng(11)
    positions = list(generate_mate_in_1(30, rng))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "mate1"
        meta = write_dataset(iter(positions), out)

        assert meta.n == 30
        ds = Phase1Dataset(out)
        assert len(ds) == 30
        for item in ds:
            assert item["outcome"].item() == 1


def test_dataset_tokens_dtype() -> None:
    rng = _rng(20)
    games = list(generate_random_games(10, rng))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "d"
        write_dataset(iter(games), out)
        ds = Phase1Dataset(out)
        item = ds[0]
        assert item["tokens"].dtype == torch.int64
        assert item["from_sq"].dtype == torch.int64
        assert item["legals"].dtype == torch.uint8


def test_meta_json_written() -> None:
    rng = _rng(21)
    games = list(generate_random_games(5, rng))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "d"
        write_dataset(iter(games), out, generation_config={"mode": "test"})
        import json

        with open(out / "meta.json") as fh:
            meta_dict = json.load(fh)
        assert meta_dict["version"] == "1"
        assert meta_dict["n"] > 0
        assert meta_dict["generation_config"]["mode"] == "test"


# ---------------------------------------------------------------------------
# Legal bitset re-validation
# ---------------------------------------------------------------------------


def test_stored_legal_bitset_matches_board_reconstruction() -> None:
    """Every stored legal bitset must match a fresh legal_moves_packed on the decoded board."""
    rng = _rng(30)
    games = list(generate_random_games(10, rng))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "d"
        write_dataset(iter(games), out)
        ds = Phase1Dataset(out)

        for idx in range(min(len(ds), 200)):
            item = ds[idx]
            board = decode_board(item["tokens"].numpy().astype(np.int8))
            expected = legal_moves_packed(board)
            stored = item["legals"].numpy()
            assert np.array_equal(stored, expected), (
                f"Legal bitset mismatch at idx {idx}, FEN: {board.fen()}"
            )


# ---------------------------------------------------------------------------
# Outcome perspective test on a known forced win
# ---------------------------------------------------------------------------


def test_outcome_perspective_known_win() -> None:
    """The mate anchors label the mover +1 — read from the generator, for both colours."""
    from tinyalphazero.data import generate_mate_in_1
    from tinyalphazero.encoding import decode_board, decode_move

    rng = _rng(3)
    colours = set()
    for rec in generate_mate_in_1(12, rng):
        board = decode_board(rec.tokens)
        colours.add(board.turn)
        board.push(decode_move(rec.from_sq, rec.to_sq, rec.promo))
        assert board.is_checkmate(), "the stored move is the mate"
        assert rec.outcome == 1, "the mover delivers the mate and is labelled +1"
        assert int(rec.tokens[64]) == (TURN_W if not board.turn else TURN_B)
    assert colours == {chess.WHITE, chess.BLACK}, "both colours must be exercised"


def test_write_dataset_empty_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(ValueError, match="no records"):
        write_dataset(iter([]), Path(tmp) / "empty")


def test_write_dataset_mixed_position_and_game_records() -> None:
    """write_dataset should handle an iterator of PositionRecord items."""
    rng = _rng(40)
    positions = list(generate_mate_in_1(10, rng))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "d"
        meta = write_dataset(iter(positions), out)
        assert meta.n == 10
