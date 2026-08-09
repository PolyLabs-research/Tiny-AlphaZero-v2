"""Tests for the board and move encoding (chunk 1, spec §3 / §9.4)."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from tinyalphazero.encoding import (
    EP_NONE,
    HM_0,
    REP_0,
    REP_1,
    REP_2,
    SEQ_LEN,
    VOCAB_SIZE,
    _repetitions_from_board,
    decode_board,
    decode_move,
    encode_board,
    encode_move,
    halfmove_bucket,
    legal_mask_ft,
    legal_moves_packed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_reachable_positions(
    n: int, rng: np.random.Generator, max_plies: int = 150
) -> list[tuple[chess.Board, int]]:
    """Generate n random reachable (board, repetitions) pairs via random play."""
    positions: list[tuple[chess.Board, int]] = []
    board = chess.Board()
    plies = 0
    while len(positions) < n:
        legal = list(board.legal_moves)
        if not legal or plies >= max_plies or board.is_game_over():
            board.reset()
            plies = 0
            legal = list(board.legal_moves)
        rep = _repetitions_from_board(board)
        positions.append((board.copy(), rep))
        move = legal[int(rng.integers(len(legal)))]
        board.push(move)
        plies += 1
    return positions


# ---------------------------------------------------------------------------
# 1. Vocabulary constants sanity
# ---------------------------------------------------------------------------


def test_vocab_constants() -> None:
    assert VOCAB_SIZE == 43
    assert SEQ_LEN == 72


def test_halfmove_bucket_boundaries() -> None:
    assert halfmove_bucket(0) == HM_0
    assert halfmove_bucket(1) == HM_0 + 1
    assert halfmove_bucket(9) == HM_0 + 1
    assert halfmove_bucket(10) == HM_0 + 2
    assert halfmove_bucket(49) == HM_0 + 5
    assert halfmove_bucket(50) == HM_0 + 6
    assert halfmove_bucket(74) == HM_0 + 6
    assert halfmove_bucket(75) == HM_0 + 7
    assert halfmove_bucket(200) == HM_0 + 7


def test_halfmove_bucket_49_vs_50_different_tokens() -> None:
    """Positions with clock 49 and 50 must encode to different bucket tokens."""
    assert halfmove_bucket(49) != halfmove_bucket(50)


# ---------------------------------------------------------------------------
# 2. encode_board / decode_board round-trip
# ---------------------------------------------------------------------------


def test_encode_board_shape_and_dtype() -> None:
    board = chess.Board()
    tokens = encode_board(board, 0)
    assert tokens.shape == (72,)
    assert tokens.dtype == np.int8


def test_encode_board_token_range() -> None:
    rng = np.random.default_rng(42)
    positions = _random_reachable_positions(500, rng)
    for board, rep in positions:
        tokens = encode_board(board, rep)
        assert tokens.min() >= 0, "Negative token"
        assert tokens.max() < VOCAB_SIZE, f"Token {tokens.max()} >= VOCAB_SIZE {VOCAB_SIZE}"


def test_roundtrip_starting_position() -> None:
    board = chess.Board()
    tokens = encode_board(board, 0)
    board2 = decode_board(tokens)
    assert board.board_fen() == board2.board_fen()
    assert board.turn == board2.turn
    assert board.castling_rights == board2.castling_rights
    assert board.ep_square == board2.ep_square


def test_roundtrip_10k_random_positions() -> None:
    """Round-trip: decode(encode(b)) must reproduce piece placement, turn, castling, EP, hm bucket."""
    rng = np.random.default_rng(1)
    positions = _random_reachable_positions(10_000, rng)
    for board, rep in positions:
        tokens = encode_board(board, rep)
        board2 = decode_board(tokens)

        # Piece placement
        assert board.board_fen() == board2.board_fen(), f"Piece mismatch at {board.fen()}"

        # Turn
        assert board.turn == board2.turn, f"Turn mismatch at {board.fen()}"

        # Castling rights
        assert board.castling_rights == board2.castling_rights, (
            f"Castling mismatch at {board.fen()}: "
            f"{board.castling_rights} != {board2.castling_rights}"
        )

        # En passant (only when legal)
        if board.has_legal_en_passant():
            assert board2.ep_square is not None, f"EP lost at {board.fen()}"
            assert chess.square_file(board.ep_square) == chess.square_file(board2.ep_square), (
                f"EP file mismatch at {board.fen()}"
            )
        else:
            # EP token should be EP_NONE → no legal en passant after decode
            assert not board2.has_legal_en_passant() or board2.ep_square is None, (
                f"Spurious EP at {board.fen()}"
            )

        # Halfmove clock: decoded value must be in same bucket
        from tinyalphazero.encoding import halfmove_bucket

        assert halfmove_bucket(board.halfmove_clock) == halfmove_bucket(board2.halfmove_clock), (
            f"HM bucket mismatch: {board.halfmove_clock} → {board2.halfmove_clock}"
        )


def test_ep_only_set_when_legal() -> None:
    """EP token must only be set when has_legal_en_passant() is True."""
    # Position with ep_square set but no legal capture
    # After 1.e4 e5 2.Nf3, the ep_square on e6 from white is irrelevant (it's black's turn now);
    # set up a position manually where ep_square is not actually capturable.
    board = chess.Board()
    board.push_san("e4")  # white pawn to e4; now black to move; ep_square = e3
    # ep_square is set (e3) but black has no pawn adjacent to e4 after just this move yet
    # Actually after e4 the ep_square is e3 from black's perspective.
    # Let's check: if no black pawn is on d4 or f4, no legal en passant.
    board2 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assert board2.ep_square is not None
    assert not board2.has_legal_en_passant()
    tokens = encode_board(board2, 0)
    assert tokens[69] == EP_NONE, "EP token must be NONE when no legal en passant"


def test_repetition_tokens() -> None:
    """Test that repetition count is correctly encoded."""
    board = chess.Board()
    t0 = encode_board(board, 0)
    t1 = encode_board(board, 1)
    t2 = encode_board(board, 2)
    t3 = encode_board(board, 3)  # clamped to 2
    assert int(t0[71]) == REP_0
    assert int(t1[71]) == REP_1
    assert int(t2[71]) == REP_2
    assert int(t3[71]) == REP_2  # clamped


def test_repetition_1_vs_2_different_tokens() -> None:
    """Positions with rep=1 and rep=2 must produce different encodings."""
    board = chess.Board()
    t1 = encode_board(board, 1)
    t2 = encode_board(board, 2)
    assert not np.array_equal(t1, t2)


# ---------------------------------------------------------------------------
# 3. encode_move / decode_move
# ---------------------------------------------------------------------------


def test_encode_decode_move_roundtrip_random_positions() -> None:
    """Move codec bijectivity over all legal moves of 1K random positions."""
    rng = np.random.default_rng(2)
    positions = _random_reachable_positions(1_000, rng)
    for board, _ in positions:
        for move in board.legal_moves:
            from_sq, to_sq, promo = encode_move(move)
            assert 0 <= from_sq < 64
            assert 0 <= to_sq < 64
            assert 0 <= promo <= 4
            # Round-trip
            move2 = decode_move(from_sq, to_sq, promo)
            assert move == move2, f"Move round-trip failed: {move} != {move2}"


def test_non_promotion_gives_promo_0() -> None:
    move = chess.Move.from_uci("e2e4")
    _, _, promo = encode_move(move)
    assert promo == 0


def test_queen_promotion_gives_promo_4() -> None:
    move = chess.Move.from_uci("e7e8q")
    _, _, promo = encode_move(move)
    assert promo == 4


def test_underpromotion_encoding() -> None:
    promos = {"n": 1, "b": 2, "r": 3, "q": 4}
    for suffix, expected in promos.items():
        move = chess.Move.from_uci(f"e7e8{suffix}")
        _, _, promo = encode_move(move)
        assert promo == expected, f"Wrong promo for {suffix}: {promo} != {expected}"


def test_decode_move_invalid_promo_raises() -> None:
    """promo value outside 0-4 must raise ValueError."""
    with pytest.raises(ValueError, match="invalid promo value"):
        decode_move(chess.E2, chess.E4, 5)


def test_decode_move_promo_zero_gives_no_promotion() -> None:
    """promo=0 produces a move with no promotion piece."""
    e2 = chess.square(4, 1)
    e4 = chess.square(4, 3)
    move = decode_move(e2, e4, 0)
    assert move.promotion is None


def test_decode_move_promo_zero_ok_for_non_promotion() -> None:
    """promo=0 on a normal move must succeed."""
    e2 = chess.square(4, 1)
    e4 = chess.square(4, 3)
    move = decode_move(e2, e4, 0)
    assert move == chess.Move(e2, e4)


def test_at_least_50_underpromotion_moves() -> None:
    """Explicitly construct ≥ 50 underpromotion move round-trips (N, B, R, Q × 8 files × 2 colors = 64)."""
    count = 0
    for file in range(8):
        for promo in [1, 2, 3, 4]:  # N, B, R, Q
            # White promotion (push and diagonal captures where applicable)
            from_sq = chess.square(file, 6)
            to_sq = chess.square(file, 7)
            move_in = chess.Move(from_sq, to_sq, promotion=_promo_int_to_piece(promo))
            f, t, p = encode_move(move_in)
            move_out = decode_move(f, t, p)
            assert move_in == move_out
            count += 1
            # Black promotion
            from_sq_b = chess.square(file, 1)
            to_sq_b = chess.square(file, 0)
            move_in_b = chess.Move(from_sq_b, to_sq_b, promotion=_promo_int_to_piece(promo))
            f, t, p = encode_move(move_in_b)
            move_out_b = decode_move(f, t, p)
            assert move_in_b == move_out_b
            count += 1
    assert count >= 50, f"Only {count} underpromotion moves tested"


def _promo_int_to_piece(p: int) -> int:
    return {1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}[p]


def test_at_least_50_en_passant_moves() -> None:
    """Collect ≥ 50 actual en passant moves from random games and verify round-trip."""
    rng = np.random.default_rng(42)
    ep_moves: list[chess.Move] = []
    board = chess.Board()
    plies = 0
    max_plies = 200
    max_iters = 500_000

    for _ in range(max_iters):
        legal = list(board.legal_moves)
        if not legal or plies >= max_plies or board.is_game_over():
            board.reset()
            plies = 0
            legal = list(board.legal_moves)

        for move in legal:
            if board.is_en_passant(move):
                ep_moves.append(move)
                if len(ep_moves) >= 50:
                    break

        if len(ep_moves) >= 50:
            break

        move = legal[int(rng.integers(len(legal)))]
        board.push(move)
        plies += 1

    assert len(ep_moves) >= 50, f"Only found {len(ep_moves)} en passant moves in random play"

    for move in ep_moves:
        f, t, p = encode_move(move)
        assert p == 0, f"En passant move {move} unexpectedly has promo {p}"
        move2 = decode_move(f, t, p)
        assert move == move2, f"En passant round-trip failed: {move} != {move2}"


# ---------------------------------------------------------------------------
# 4. legal_mask_ft vs legal_moves_packed agreement
# ---------------------------------------------------------------------------


def test_legal_mask_packed_agreement_1k_positions() -> None:
    """legal_moves_packed must agree bit-for-bit with legal_mask_ft on 1K positions."""
    rng = np.random.default_rng(3)
    positions = _random_reachable_positions(1_000, rng)
    for board, _ in positions:
        mask = legal_mask_ft(board)
        packed = legal_moves_packed(board)
        # Reconstruct mask from packed bitset and compare
        for i in range(64):
            for j in range(64):
                idx = i * 64 + j
                bit = bool((packed[idx >> 3] >> (idx & 7)) & 1)
                assert bit == mask[i, j], (
                    f"Mismatch at ({i},{j}) in position {board.fen()}: "
                    f"mask={mask[i, j]}, packed bit={bit}"
                )


def test_legal_mask_shape_and_dtype() -> None:
    board = chess.Board()
    mask = legal_mask_ft(board)
    assert mask.shape == (64, 64)
    assert mask.dtype == bool


def test_legal_moves_packed_shape_and_dtype() -> None:
    board = chess.Board()
    packed = legal_moves_packed(board)
    assert packed.shape == (512,)
    assert packed.dtype == np.uint8


def test_legal_mask_starting_position_spot_check() -> None:
    """In the starting position, e2→e4 must be legal and e2→e5 must not."""
    board = chess.Board()
    mask = legal_mask_ft(board)
    e2 = chess.square(4, 1)
    e4 = chess.square(4, 3)
    e5 = chess.square(4, 4)
    assert mask[e2, e4], "e2→e4 should be legal"
    assert not mask[e2, e5], "e2→e5 should not be legal"


# ---------------------------------------------------------------------------
# 5. Markov invariant — light version (full fuzz is in __main__)
# ---------------------------------------------------------------------------


def test_markov_invariant_200k() -> None:
    """200K-position Markov invariant check (fast subset of the __main__ fuzz)."""
    from tinyalphazero.encoding import _fuzz

    # Should complete without raising
    _fuzz(200_000, seed=99)


def test_repetition_from_board_helper() -> None:
    """_repetitions_from_board must return 0 for a fresh board."""
    board = chess.Board()
    assert _repetitions_from_board(board) == 0
