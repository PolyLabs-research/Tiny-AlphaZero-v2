"""Drive an xboard protocol-1 engine (Micro-Max 4.8) from python-chess (PREREG-p2_y).

python-chess speaks xboard protocol 2 and refuses an engine that does not declare
``ping`` and ``setboard``. Micro-Max 4.8 (``umax4_8w.c``) predates both: it answers
``protover`` with ``Error (unknown command)``, sets up positions only through
``edit`` mode, has no ``ping``, and prints its moves in four characters, a promotion
included. The adapter (``scripts/xboard1_adapter.py``) sits between them and changes
exactly four things; every other line passes through:

* ``protover`` is answered with the features python-chess requires, on the engine's
  behalf; ``accepted``/``rejected`` replies to them are dropped.
* ``setboard FEN`` becomes the protocol-1 sequence: side to move, ``force``,
  ``edit``, ``#`` (clear), White's pieces, ``c``, Black's pieces, ``.``. Edit mode
  cannot carry an en-passant square, and it infers castling rights from where the
  kings and rooks stand, so a FEN is accepted only if its en-passant field is ``-``
  and its castling rights equal the inferred ones (:func:`edit_commands` raises
  otherwise). Every position of ``tests/suites/openings_4ply.epd`` qualifies
  (``tests/test_xboard1.py``).
* ``ping N`` becomes a marker command the engine does not know. The engine reads
  input only between searches and handles commands in order, so its
  ``Error (unknown command): <marker>`` reply comes after everything sent before
  the ping, including a search in progress -- which is what ``pong N`` promises.
  That reply is turned into ``pong N``; every other unknown-command error is dropped
  (a WinBoard GUI ignores them; python-chess treats them as fatal).
* ``move XXXX`` from the engine that is a pawn reaching the last rank gets ``q``
  appended. Micro-Max promotes only to a queen (``umax4_8w.c`` lines 151-152) and
  prints no piece letter; WinBoard assumes the queen, python-chess rejects the move.
  Telling a promotion from a rook or king move needs the position, so
  :class:`Relay` tracks the board from ``new``, ``setboard``, the moves sent in,
  ``undo``/``remove`` and the engine's own moves.

The engine's other lines (results, ``Illegal move``, ``resign``) pass unchanged.
"""

from __future__ import annotations

import threading

import chess

UNKNOWN_PREFIX = "Error (unknown command): "
# Starts with 'z' so the engine can never read it as a move (files run a-h).
SYNC_MARKER = "zping"


def features_line(name: str) -> str:
    """The features python-chess requires, declared on the engine's behalf."""
    return f'feature ping=1 setboard=1 usermove=0 san=0 reuse=1 myname="{name}" done=1'


def inferred_castling(board: chess.Board) -> int:
    """Castling rights edit mode infers: a king and rook still on their home squares."""
    rights = 0
    for color, king_sq, short_rook, long_rook in (
        (chess.WHITE, chess.E1, chess.H1, chess.A1),
        (chess.BLACK, chess.E8, chess.H8, chess.A8),
    ):
        if board.piece_at(king_sq) != chess.Piece(chess.KING, color):
            continue
        for rook_sq in (short_rook, long_rook):
            if board.piece_at(rook_sq) == chess.Piece(chess.ROOK, color):
                rights |= chess.BB_SQUARES[rook_sq]
    return rights


def edit_commands(fen: str) -> list[str]:
    """The protocol-1 lines that set up ``fen``; ValueError if edit mode cannot express it."""
    board = chess.Board(fen)
    if board.ep_square is not None:
        raise ValueError(f"en-passant square in {fen!r}: edit mode cannot set it")
    if board.castling_rights != inferred_castling(board):
        raise ValueError(f"castling rights of {fen!r} differ from what edit mode infers")
    lines = ["white" if board.turn == chess.WHITE else "black", "force", "edit", "#"]
    for color in (chess.WHITE, chess.BLACK):
        if color == chess.BLACK:
            lines.append("c")
        for square, piece in sorted(board.piece_map().items()):
            if piece.color == color:
                lines.append(piece.symbol().upper() + chess.square_name(square))
    lines.append(".")
    return lines


def _legal(board: chess.Board, text: str) -> chess.Move | None:
    """``text`` as a legal move on ``board``, or None."""
    try:
        move = chess.Move.from_uci(text)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


class Relay:
    """The adapter's translation, with the board it needs for promotions.

    ``from_gui`` runs on the adapter's input thread and ``from_engine`` on its output
    thread; both touch the board, so both hold ``_lock``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.board = chess.Board()
        self._lock = threading.Lock()

    def from_gui(self, line: str) -> tuple[list[str], list[str]]:
        """One line from python-chess -> (lines for the engine, lines back to python-chess)."""
        text = line.strip()
        cmd, _, rest = text.partition(" ")
        with self._lock:
            if cmd == "protover":
                return [], [features_line(self.name)]
            if cmd in ("accepted", "rejected"):
                return [], []
            if cmd == "ping":
                return [f"{SYNC_MARKER}{rest.strip()}"], []
            if cmd == "setboard":
                try:
                    lines = edit_commands(rest.strip())
                except ValueError as exc:
                    return [], [f"Error (setboard): {exc}"]
                self.board = chess.Board(rest.strip())
                return lines, []
            if cmd == "new":
                self.board = chess.Board()
            elif cmd == "undo" and self.board.move_stack:
                self.board.pop()
            elif cmd == "remove":
                for _ in range(min(2, len(self.board.move_stack))):
                    self.board.pop()
            else:
                move = _legal(self.board, cmd)
                if move is not None:
                    self.board.push(move)
            return [text], []

    def from_engine(self, line: str) -> list[str]:
        """One line from the engine -> lines for python-chess (empty to drop it)."""
        text = line.rstrip("\n")
        if text.startswith(UNKNOWN_PREFIX):
            unknown = text[len(UNKNOWN_PREFIX) :].strip()
            if unknown.startswith(SYNC_MARKER):
                return [f"pong {unknown[len(SYNC_MARKER) :]}"]
            return []
        cmd, _, rest = text.partition(" ")
        if cmd != "move":
            return [text]
        uci = rest.strip()
        with self._lock:
            move = _legal(self.board, uci)
            if move is None and len(uci) == 4:
                move = _legal(self.board, uci + "q")
                if move is not None:
                    uci = move.uci()
            if move is not None:
                self.board.push(move)
        return [f"move {uci}"]
