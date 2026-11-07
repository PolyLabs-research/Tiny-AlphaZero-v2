"""The error analysis's grading (scripts/where_points_go.py), on games small enough to check by hand."""

from __future__ import annotations

import chess
import pytest

from scripts.where_points_go import grade_game, summarise
from tinyalphazero.teacher import MATE_CP

START = chess.Board().epd()


def _r(cp: int, wdl: tuple[float, float, float], pv: list[str]) -> dict:
    return {"lines": [[pv[0] if pv else "0000", cp]], "wdl": list(wdl), "depth": 10, "nodes": 1,
            "pv": pv}  # fmt: skip


def test_fools_mate_is_an_allowed_mate_in_the_opening() -> None:
    game = {
        "epd": START, "learner_white": True, "score": 0.0, "result": "checkmate",
        "moves": ["f2f3", "e7e5", "g2g4", "d8h4"], "learner_q": [[0, 0.0], [2, 0.1]],
    }  # fmt: skip
    annot = {
        "readings": [
            _r(20, (0.05, 0.9, 0.05), ["e2e4"]),  # white to move: 0.5
            _r(60, (0.10, 0.88, 0.02), ["e7e5"]),  # black to move: white 0.46
            _r(-80, (0.0, 0.85, 0.15), ["e2e4"]),  # white to move: 0.425
            _r(MATE_CP - 1, (1.0, 0.0, 0.0), ["d8h4"]),  # black mates: white 0.0
            None,  # checkmate
        ]
    }
    g = grade_game(game, annot)
    learner = [m for m in g["moves"] if m["drop"] >= 0.10]
    assert len(learner) == 1
    m = learner[0]
    assert m["ply"] == 2 and m["kind"] == "allowed_mate" and m["phase"] == "opening"
    assert m["drop"] == pytest.approx(0.425)
    assert m["model_e"] == pytest.approx(0.55)  # the search thought it was fine
    assert g["e_end"] == 0.0 and g["score"] == 0.0
    s = summarise([g])
    assert s["largest_cell"] == {"cell": "allowed_mate/opening", "share": 1.0}
    assert s["one_recurring_cause"] is True
    assert s["blunders_search_did_not_see"] == 1


def test_a_drop_with_no_material_lost_while_winning_is_a_slip() -> None:
    game = {
        "epd": "4k3/8/8/2p5/8/8/8/3QK3 w - -", "learner_white": True, "score": 0.5,
        "result": "ply_cap", "moves": ["d1d5", "c5c4"], "learner_q": [[0, 0.9]],
    }  # fmt: skip
    annot = {
        "readings": [
            _r(900, (0.95, 0.05, 0.0), ["d1d4", "e8e7"]),  # white to move: 0.975
            _r(-300, (0.0, 0.3, 0.7), ["c5c4"]),  # black to move: white 1 - 0.15 = 0.85
            _r(300, (0.6, 0.4, 0.0), ["d5c4"]),
        ]
    }
    m = grade_game(game, annot)["moves"][0]
    assert m["drop"] == pytest.approx(0.125)
    assert m["kind"] == "slip_while_winning"  # nothing changes hands along either line


def test_a_hung_queen_is_material_in_the_endgame() -> None:
    game2 = {
        "epd": "4k3/8/8/4p3/8/8/8/3QK3 w - -", "learner_white": True, "score": 0.0,
        "result": "checkmate", "moves": ["d1d4", "e5d4"], "learner_q": [],
    }  # fmt: skip
    annot2 = {
        "readings": [
            _r(800, (0.6, 0.4, 0.0), ["d1d3", "e8e7"]),  # 0.8
            _r(800, (0.9, 0.1, 0.0), ["e5d4"]),  # black to move, winning the queen: white 0.05
            _r(-800, (0.0, 0.2, 0.8), ["e1d2"]),
        ]
    }
    g2 = grade_game(game2, annot2)
    m2 = g2["moves"][0]
    assert m2["drop"] == pytest.approx(0.75)
    assert m2["kind"] == "material" and m2["phase"] == "endgame"
