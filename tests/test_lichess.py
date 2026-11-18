"""Positions from Lichess games (tinyalphazero.lichess)."""

from __future__ import annotations

import numpy as np

from tinyalphazero.encoding import decode_board
from tinyalphazero.lichess import Filters, base_seconds, header_ok, sample_positions

# A 40-ply game White wins, with a knight shuffle (g1f3/f3g1 and g8f6/f6g8) at plies 10-17 so
# that the start of that shuffle's position repeats in the game's own history.
MOVES = (
    "1. e4 e5 2. Nc3 Nc6 3. Bc4 Bc5 4. d3 d6 5. Be3 Be6 6. Nf3 Nf6 7. Ng1 Ng8 8. Nf3 Nf6 "
    "9. Ng1 Ng8 10. Qd2 Qd7 11. O-O-O O-O-O 12. f3 f6 13. g4 g5 14. h4 h6 15. a3 a6 "
    "16. Kb1 Kb8 17. Ka1 Ka8 18. Nge2 Nge7 19. Bxc5 dxc5 20. Bxe6 Qxe6 1-0"
)


def _pgn(**headers: str) -> str:
    base = {
        "Event": "Rated Blitz game", "Result": "1-0", "WhiteElo": "1900", "BlackElo": "1750",
        "TimeControl": "300+3", "Termination": "Normal",
    }  # fmt: skip
    base.update(headers)
    head = "".join(f'[{k} "{v}"]\n' for k, v in base.items())
    return head + "\n" + MOVES + "\n"


def _headers(text: str) -> dict[str, str]:
    import io

    import chess.pgn

    return dict(chess.pgn.read_headers(io.StringIO(text)))


def test_header_filters() -> None:
    f = Filters()
    assert base_seconds("300+3") == 300 and base_seconds("-") is None
    assert header_ok(_headers(_pgn()), f)
    assert not header_ok(_headers(_pgn(TimeControl="60+0")), f)  # bullet
    assert not header_ok(_headers(_pgn(TimeControl="-")), f)  # correspondence
    assert not header_ok(_headers(_pgn(BlackElo="1500")), f)  # one player below 1600
    assert not header_ok(_headers(_pgn(Termination="Time forfeit")), f)
    assert not header_ok(_headers(_pgn(Result="*")), f)


def test_sampled_positions_are_the_games_own_with_its_result_and_history() -> None:
    f = Filters(per_game=10, first_ply=10, min_plies=30)
    rows = sample_positions(_pgn(), f, seed=3)
    assert len(rows) == 10
    import io

    import chess.pgn

    game = chess.pgn.read_game(io.StringIO(_pgn()))
    boards, moves = [], list(game.mainline_moves())
    b = game.board()
    for mv in moves:
        boards.append((b.fen(), b.turn, mv))
        b.push(mv)
    fens = {fen: (turn, mv) for fen, turn, mv in boards[10:]}
    for row in rows:
        board = decode_board(np.asarray(row["tokens"]))
        fen = board.fen()
        key = next(k for k in fens if k.split(" ")[:4] == fen.split(" ")[:4])
        turn, mv = fens[key]
        assert row["outcome"] == (1 if turn == chess.WHITE else -1)  # White won
        assert row["move_ft"] == mv.from_square * 64 + mv.to_square
    again = sample_positions(_pgn(), f, seed=3)  # seeded: the same positions again
    assert all(np.array_equal(a["tokens"], b["tokens"]) for a, b in zip(rows, again, strict=True))
    other = sample_positions(_pgn(), f, seed=4)
    assert not all(
        np.array_equal(a["tokens"], b["tokens"]) for a, b in zip(rows, other, strict=True)
    )


def test_repetition_count_comes_from_the_games_history() -> None:
    from tinyalphazero.encoding import REP_0

    f = Filters(per_game=40, first_ply=0, min_plies=30)  # every position
    rows = sample_positions(_pgn(), f, seed=0)
    assert len(rows) == 40
    # Ply 14 (after 7...Ng8) repeats ply 10's position; ply 18 repeats it a second time.
    assert int(rows[10]["tokens"][71]) == REP_0
    assert int(rows[14]["tokens"][71]) != REP_0
    assert int(rows[18]["tokens"][71]) != int(rows[14]["tokens"][71])


def test_short_or_broken_games_give_nothing() -> None:
    f = Filters(min_plies=60)
    assert sample_positions(_pgn(), f, seed=0) == []
    assert sample_positions('[Result "1-0"]\n\n1. e4 e9 1-0\n', Filters(), seed=0) == []


def test_source_rows_selects_by_tag_in_ring_order_and_refuses_other_tags() -> None:
    import pytest

    from tinyalphazero.lichess import LICHESS_ITERATION, source_rows

    tags = np.array([LICHESS_ITERATION, 0, 5, LICHESS_ITERATION, 60, LICHESS_ITERATION])
    assert source_rows(tags, "lichess").tolist() == [0, 3, 5]
    assert source_rows(tags, "selfplay").tolist() == [1, 2, 4]
    assert source_rows(tags, "lichess", limit=2).tolist() == [0, 3]
    with pytest.raises(ValueError):
        source_rows(tags, "selfplay", limit=4)  # only three
    with pytest.raises(ValueError):
        source_rows(np.append(tags, -2), "lichess")  # a teacher tag: not a pool of these sources
    with pytest.raises(ValueError):
        source_rows(tags, "human")
