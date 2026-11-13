"""The protocol-1 adapter (tinyalphazero.xboard1, scripts/xboard1_adapter.py)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import chess
import pytest

from tinyalphazero.evaluate import load_openings
from tinyalphazero.xboard1 import (
    SYNC_MARKER,
    Relay,
    edit_commands,
    features_line,
    inferred_castling,
)

REPO = Path(__file__).resolve().parent.parent


def test_every_book_opening_is_representable_in_edit_mode() -> None:
    """p2_y's games start from these 100 positions; each must survive the translation."""
    book = load_openings()
    assert len(book) == 100
    for epd, ident in book:
        board = chess.Board()
        board.set_epd(epd)
        lines = edit_commands(board.fen())
        assert lines[:4] == ["white" if board.turn else "black", "force", "edit", "#"], ident
        assert lines[-1] == "."


def test_edit_commands_place_every_piece_once_in_colour_blocks() -> None:
    board = chess.Board()
    lines = edit_commands(board.fen())
    c = lines.index("c")
    white, black = lines[4:c], lines[c + 1 : -1]
    assert len(white) == len(black) == 16
    assert "Ke1" in white and "Ra1" in white and "Pe2" in white
    assert "Ke8" in black and "Qd8" in black and "Pe7" in black  # uppercase after 'c'
    assert lines[0] == "white"


def test_unrepresentable_positions_are_refused() -> None:
    # En-passant square set.
    with pytest.raises(ValueError, match="en-passant"):
        edit_commands("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    # King and rooks home, but White's rights already lost (king moved and returned).
    with pytest.raises(ValueError, match="castling"):
        edit_commands("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w kq - 4 3")


def test_inferred_castling_reads_home_squares() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K1R1 w Qkq - 0 1")
    assert inferred_castling(board) == chess.BB_A1 | chess.BB_A8 | chess.BB_H8
    assert inferred_castling(board) == board.castling_rights


def test_gui_side() -> None:
    r = Relay("mm")
    assert r.from_gui("protover 2\n") == ([], [features_line("mm")])
    assert "ping=1" in features_line("mm") and "setboard=1" in features_line("mm")
    assert r.from_gui("accepted ping\n") == ([], [])
    assert r.from_gui("ping 17\n") == ([f"{SYNC_MARKER}17"], [])
    assert r.from_gui("level 0 1:00 0.6\n") == (["level 0 1:00 0.6"], [])
    assert r.from_gui("e2e4\n") == (["e2e4"], [])
    assert r.board.move_stack == [chess.Move.from_uci("e2e4")]
    to_engine, back = r.from_gui(
        "setboard rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3\n"
    )
    assert to_engine == [] and back[0].startswith("Error (setboard)")


def test_engine_side() -> None:
    r = Relay("mm")
    assert r.from_engine(f"Error (unknown command): {SYNC_MARKER}17\n") == ["pong 17"]
    assert r.from_engine("Error (unknown command): protover\n") == []
    assert r.from_engine("Illegal move:e2e5\n") == ["Illegal move:e2e5"]
    r.from_gui("e2e4\n")
    assert r.from_engine("move e7e5\n") == ["move e7e5"]
    assert r.board.peek() == chess.Move.from_uci("e7e5")


def test_a_four_character_promotion_becomes_a_queen() -> None:
    """umax4_8w.c prints promotions without a piece letter and promotes only to a queen."""
    r = Relay("mm")
    r.from_gui("setboard 8/1p6/4k2p/P1pp4/3r4/4K3/1p6/3b4 b - - 0 48\n")  # the smoke game's
    assert r.from_engine("move b2b1\n") == ["move b2b1q"]
    # A rook reaching the back rank is not a promotion and keeps its four characters.
    r.from_gui("setboard 4k3/8/8/8/8/8/1r6/4K3 b - - 0 1\n")
    assert r.from_engine("move b2b1\n") == ["move b2b1"]


def test_board_tracking_follows_new_undo_and_remove() -> None:
    r = Relay("mm")
    for mv in ("e2e4", "e7e5", "g1f3"):
        r.from_gui(mv + "\n")
    r.from_gui("undo\n")
    assert len(r.board.move_stack) == 2
    r.from_gui("remove\n")
    assert r.board.move_stack == []
    r.from_gui("d2d4\n")
    r.from_gui("new\n")
    assert r.board == chess.Board()


FAKE_ENGINE = textwrap.dedent(
    """
    import sys
    # A protocol-1 engine in miniature: knows a few commands, answers 'go' with a move,
    # reports everything else as unknown -- the way umax4_8w.c does.
    replies = iter(["move e7e5", "move b2b1"])  # the second: a promotion, four characters
    for line in sys.stdin:
        cmd = line.split()[0] if line.split() else ""
        if cmd == "quit":
            break
        if cmd == "go":
            print(next(replies), flush=True)
        elif cmd in ("new", "force", "white", "black", "edit", "#", "c", ".", "level",
                     "time", "otim", "xboard") or (len(cmd) == 3 and cmd[0] in "PNBRQK") or (
                     len(cmd) == 4 and cmd[0] in "abcdefgh" and cmd[1] in "12345678"):
            continue
        else:
            print(f"Error (unknown command): {cmd}", flush=True)
    """
)


def test_adapter_lets_python_chess_play_a_protocol1_engine(tmp_path: Path) -> None:
    """End to end through the real script: handshake, setboard, a move, ping/pong, quit."""
    import chess.engine

    fake = tmp_path / "fake_engine.py"
    fake.write_text(FAKE_ENGINE)
    cmd = [
        sys.executable,
        str(REPO / "scripts/xboard1_adapter.py"),
        "--",
        sys.executable,
        str(fake),
    ]
    engine = chess.engine.SimpleEngine.popen_xboard(cmd, timeout=30)
    try:
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        limit = chess.engine.Limit(white_clock=60, black_clock=60, white_inc=0.6, black_inc=0.6)
        result = engine.play(board, limit, game="g")
        assert result.move == chess.Move.from_uci("e7e5")
        engine.ping()
        # A new game: Black's pawn on b2 promotes; the engine prints "b2b1".
        board = chess.Board("8/1p6/4k2p/P1pp4/3r4/4K3/1p6/3b4 b - - 0 48")
        result = engine.play(board, limit, game="h")
        assert result.move == chess.Move.from_uci("b2b1q")
    finally:
        engine.quit()


def test_adapter_exits_when_its_input_closes(tmp_path: Path) -> None:
    fake = tmp_path / "fake_engine.py"
    fake.write_text(FAKE_ENGINE)
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/xboard1_adapter.py"), "--", sys.executable, str(fake)],
        input="xboard\nprotover 2\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip().splitlines() == [features_line("micro-Max 4.8")]
