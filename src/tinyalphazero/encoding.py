"""Board and move encoding for TinyAlphaZero v2 (spec §3).

72-token Markovian board encoding + factorized (from, to, promo) move codec.
This module is the single source of truth for all vocabulary constants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chess
import numpy as np

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Vocabulary constants (spec §3.1)
# ---------------------------------------------------------------------------

# Piece tokens 0-12: empty=0, white pieces 1-6, black pieces 7-12
ES = 0
WP, WN, WB, WR, WQ, WK = 1, 2, 3, 4, 5, 6
BP, BN, BB, BR, BQ, BK = 7, 8, 9, 10, 11, 12

# Side to move
TURN_W, TURN_B = 13, 14

# Castling rights (Y=right exists, N=right lost)
CWK_Y, CWK_N = 15, 16
CWQ_Y, CWQ_N = 17, 18
CBK_Y, CBK_N = 19, 20
CBQ_Y, CBQ_N = 21, 22

# En passant file (set ONLY when has_legal_en_passant() is true)
EP_NONE = 23
EP_A, EP_B, EP_C, EP_D, EP_E, EP_F, EP_G, EP_H = 24, 25, 26, 27, 28, 29, 30, 31

# Halfmove clock buckets: {0}, 1-9, 10-19, 20-29, 30-39, 40-49, 50-74, 75+
HM_0, HM_1, HM_2, HM_3, HM_4, HM_5, HM_6, HM_7 = 32, 33, 34, 35, 36, 37, 38, 39

# Repetition count (times current position already occurred earlier)
REP_0, REP_1, REP_2 = 40, 41, 42

VOCAB_SIZE: int = 43
SEQ_LEN: int = 72

# ---------------------------------------------------------------------------
# Internal lookup tables
# ---------------------------------------------------------------------------

# python-chess piece_type (1-6) × color → token
_PIECE_TOKEN: dict[tuple[int, bool], int] = {
    (chess.PAWN, chess.WHITE): WP,
    (chess.KNIGHT, chess.WHITE): WN,
    (chess.BISHOP, chess.WHITE): WB,
    (chess.ROOK, chess.WHITE): WR,
    (chess.QUEEN, chess.WHITE): WQ,
    (chess.KING, chess.WHITE): WK,
    (chess.PAWN, chess.BLACK): BP,
    (chess.KNIGHT, chess.BLACK): BN,
    (chess.BISHOP, chess.BLACK): BB,
    (chess.ROOK, chess.BLACK): BR,
    (chess.QUEEN, chess.BLACK): BQ,
    (chess.KING, chess.BLACK): BK,
}

# Reverse: token → (piece_type, color)
_TOKEN_PIECE: dict[int, tuple[int, bool]] = {v: k for k, v in _PIECE_TOKEN.items()}

# Halfmove bucket thresholds (lower-bound of each bucket)
_HM_LOWER: list[int] = [0, 1, 10, 20, 30, 40, 50, 75]

# Promotion mapping: our promo int → python-chess piece type
_PROMO_TO_PIECE: dict[int, int | None] = {
    0: None,
    1: chess.KNIGHT,
    2: chess.BISHOP,
    3: chess.ROOK,
    4: chess.QUEEN,
}
_PIECE_TO_PROMO: dict[int | None, int] = {v: k for k, v in _PROMO_TO_PIECE.items()}

# EP file token for each file index 0-7
_EP_FILE_TOKEN: list[int] = [EP_A, EP_B, EP_C, EP_D, EP_E, EP_F, EP_G, EP_H]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def halfmove_bucket(clock: int) -> int:
    """Map a halfmove clock value to its bucket token (HM_0..HM_7)."""
    if clock == 0:
        return HM_0
    if clock < 10:
        return HM_1
    if clock < 20:
        return HM_2
    if clock < 30:
        return HM_3
    if clock < 40:
        return HM_4
    if clock < 50:
        return HM_5
    if clock < 75:
        return HM_6
    return HM_7


def encode_board(board: chess.Board, repetitions: int) -> np.ndarray:
    """Encode a board position into a 72-token int8 array (spec §3.1).

    repetitions: how many times this position already occurred earlier in the
    game (0, 1, or ≥2 clamped to 2). The caller is responsible for tracking
    this; encode_board does not scan board history.
    """
    tokens = np.empty(SEQ_LEN, dtype=np.int8)

    # Squares 0-63: piece tokens
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            tokens[sq] = ES
        else:
            tokens[sq] = _PIECE_TOKEN[(piece.piece_type, piece.color)]

    # Token 64: side to move
    tokens[64] = TURN_W if board.turn == chess.WHITE else TURN_B

    # Tokens 65-68: castling rights
    tokens[65] = CWK_Y if board.has_kingside_castling_rights(chess.WHITE) else CWK_N
    tokens[66] = CWQ_Y if board.has_queenside_castling_rights(chess.WHITE) else CWQ_N
    tokens[67] = CBK_Y if board.has_kingside_castling_rights(chess.BLACK) else CBK_N
    tokens[68] = CBQ_Y if board.has_queenside_castling_rights(chess.BLACK) else CBQ_N

    # Token 69: en passant file (only when capture is actually legal)
    if board.has_legal_en_passant():
        ep_file = chess.square_file(board.ep_square)  # type: ignore[arg-type]
        tokens[69] = _EP_FILE_TOKEN[ep_file]
    else:
        tokens[69] = EP_NONE

    # Token 70: halfmove clock bucket
    tokens[70] = halfmove_bucket(board.halfmove_clock)

    # Token 71: repetition count (clamped to 2)
    tokens[71] = REP_0 + min(repetitions, 2)

    return tokens


def decode_board(tokens: np.ndarray) -> chess.Board:
    """Reconstruct a chess.Board from a 72-token encoding.

    Halfmove clock is set to the lower bound of its bucket.
    Fullmove number is set to 1 (not encoded).
    """
    board = chess.Board(fen=None)  # empty board, no pieces
    board.clear()

    # Squares 0-63
    for sq in range(64):
        t = int(tokens[sq])
        if t != ES:
            piece_type, color = _TOKEN_PIECE[t]
            board.set_piece_at(sq, chess.Piece(piece_type, color))

    # Side to move
    board.turn = chess.WHITE if int(tokens[64]) == TURN_W else chess.BLACK

    # Castling rights — build castling mask
    castling = chess.BB_EMPTY
    if int(tokens[65]) == CWK_Y:
        castling |= chess.BB_H1
    if int(tokens[66]) == CWQ_Y:
        castling |= chess.BB_A1
    if int(tokens[67]) == CBK_Y:
        castling |= chess.BB_H8
    if int(tokens[68]) == CBQ_Y:
        castling |= chess.BB_A8
    board.castling_rights = castling

    # En passant
    ep_token = int(tokens[69])
    if ep_token != EP_NONE:
        ep_file = ep_token - EP_A  # 0-7
        # EP square is on rank 6 for white (rank index 5) or rank 3 for black (rank index 2)
        # The side that can capture determines the rank of the ep square.
        # By convention ep_square is the square the capturing pawn moves *to*.
        # White captures upward → ep square on rank 6 (index 5, rows 40-47)
        # Black captures downward → ep square on rank 3 (index 2, rows 16-23)
        if board.turn == chess.WHITE:
            board.ep_square = chess.square(ep_file, 5)  # rank 6 (0-indexed: 5)
        else:
            board.ep_square = chess.square(ep_file, 2)  # rank 3 (0-indexed: 2)
    else:
        board.ep_square = None

    # Halfmove clock: lower bound of bucket
    hm_token = int(tokens[70])
    board.halfmove_clock = _HM_LOWER[hm_token - HM_0]

    # Fullmove number not encoded; set to 1
    board.fullmove_number = 1

    return board


def encode_move(move: chess.Move) -> tuple[int, int, int]:
    """Encode a chess.Move as (from_sq, to_sq, promo).

    promo: 0=none, 1=N, 2=B, 3=R, 4=Q
    """
    promo = _PIECE_TO_PROMO.get(move.promotion, 0)
    return move.from_square, move.to_square, promo


def decode_move(from_sq: int, to_sq: int, promo: int) -> chess.Move:
    """Reconstruct a chess.Move from (from_sq, to_sq, promo).

    promo must be 0 (none), 1 (N), 2 (B), 3 (R), or 4 (Q).
    Note: callers are responsible for passing the correct promo value; distinguishing
    a pawn promotion from a non-pawn move to the back rank requires board context
    that decode_move does not have.
    """
    if promo not in _PROMO_TO_PIECE:
        raise ValueError(f"decode_move: invalid promo value {promo!r}; must be 0–4")
    promotion_piece = _PROMO_TO_PIECE[promo]
    return chess.Move(from_sq, to_sq, promotion=promotion_piece)


def legal_mask_ft(board: chess.Board) -> np.ndarray:
    """Return a (64, 64) bool array: True where (from, to) has any legal move."""
    mask = np.zeros((64, 64), dtype=bool)
    for move in board.legal_moves:
        mask[move.from_square, move.to_square] = True
    return mask


def legal_moves_packed(board: chess.Board) -> np.ndarray:
    """Return a 512-byte bitset: bit i*64+j is set if (i→j) has any legal move."""
    bits = np.zeros(512, dtype=np.uint8)
    for move in board.legal_moves:
        idx = move.from_square * 64 + move.to_square
        bits[idx >> 3] |= np.uint8(1 << (idx & 7))
    return bits


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_pawn_promotion_pair(from_sq: int, to_sq: int) -> bool:
    """True if (from_sq, to_sq) is geometrically a pawn promotion move."""
    from_rank = from_sq >> 3
    to_rank = to_sq >> 3
    from_file = from_sq & 7
    to_file = to_sq & 7
    file_diff = abs(to_file - from_file)
    # White pawn: from rank 6 (0-indexed) to rank 7; forward or diagonal capture
    if from_rank == 6 and to_rank == 7 and file_diff <= 1:
        return True
    # Black pawn: from rank 1 to rank 0
    return from_rank == 1 and to_rank == 0 and file_diff <= 1


def _repetitions_from_board(board: chess.Board) -> int:
    """Helper: derive repetition count from python-chess board history."""
    if board.is_repetition(3):
        return 2
    if board.is_repetition(2):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Markov-invariant fuzz (driven by scripts/fuzz_encoding.py and the tests)
# ---------------------------------------------------------------------------


def fuzz_markov_invariant(n_positions: int, seed: int = 0) -> dict[str, int]:
    """Generate n_positions random positions and verify the Markov invariant.

    Buckets all positions by their 72-token encoding and asserts that any two
    positions sharing an encoding have identical legal move sets and identical
    50-move / threefold draw-claim status. Raises ``AssertionError`` on the
    first violation; returns ``{"positions", "unique", "collisions"}`` otherwise.
    """
    rng = np.random.default_rng(seed)

    # Map token_bytes → (legal_uci_frozenset, can_fifty, can_threefold)
    seen: dict[bytes, tuple[frozenset[str], bool, bool]] = {}
    collisions = 0
    total = 0

    board = chess.Board()
    plies_since_reset = 0
    max_plies = 150

    while total < n_positions:
        legal = list(board.legal_moves)
        if not legal or plies_since_reset >= max_plies or board.is_game_over():
            board.reset()
            plies_since_reset = 0
            legal = list(board.legal_moves)

        rep = _repetitions_from_board(board)
        tokens = encode_board(board, rep)
        key = tokens.tobytes()

        legal_set = frozenset(m.uci() for m in legal)
        can_fifty = board.can_claim_fifty_moves()
        can_threefold = board.can_claim_threefold_repetition()

        if key in seen:
            prev_legal, prev_fifty, prev_threefold = seen[key]
            if legal_set != prev_legal:
                raise AssertionError(
                    f"Markov invariant violated: same encoding, different legal moves!\n"
                    f"  FEN: {board.fen()}\n"
                    f"  extra in current: {legal_set - prev_legal}\n"
                    f"  missing from current: {prev_legal - legal_set}"
                )
            if can_fifty != prev_fifty or can_threefold != prev_threefold:
                raise AssertionError(
                    f"Markov invariant violated: same encoding, different draw status!\n"
                    f"  FEN: {board.fen()}\n"
                    f"  fifty: {prev_fifty} → {can_fifty}, threefold: {prev_threefold} → {can_threefold}"
                )
            collisions += 1
        else:
            seen[key] = (legal_set, can_fifty, can_threefold)

        # Advance game
        move = legal[int(rng.integers(len(legal)))]
        board.push(move)
        plies_since_reset += 1
        total += 1

    return {"positions": total, "unique": len(seen), "collisions": collisions}
