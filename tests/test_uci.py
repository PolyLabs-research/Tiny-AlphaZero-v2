"""The UCI front end plays the ladder's moves (tinyalphazero.uci).

The decisive test is ``test_uci_replays_batch_runner_games``: games played by the
ladder's own scheduler (``BatchRunner`` under the evaluation profile, a real
network on cpu) are replayed position by position through the UCI session, and
every move must come back the same. The rest pin the protocol and the
repetition state.
"""

from __future__ import annotations

from collections import Counter

import chess
import numpy as np
import pytest
import torch

from tinyalphazero.config import MCTSConfig, ModelConfig
from tinyalphazero.encoding import decode_move
from tinyalphazero.evaluate import load_openings
from tinyalphazero.mcts.batch_runner import (
    BatchRunner,
    masked_policy_probs,
    torch_evaluate_batch,
)
from tinyalphazero.mcts.batch_runner import uniform_stub_batch as stub_batch
from tinyalphazero.mcts.search import Search
from tinyalphazero.model import ChessNet
from tinyalphazero.uci import UciEngine, UciSession, q_to_cp, single_from_batch

_CPU = torch.device("cpu")
_EVAL = MCTSConfig(root_noise=False)


@pytest.fixture(scope="module")
def tiny_batch_evaluator():
    torch.manual_seed(0)
    net = ChessNet(
        ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, d_policy=16, dropout=0.0)
    ).eval()
    return torch_evaluate_batch(net, _CPU)


def _session(evaluate_batch, sims: int = 16) -> tuple[UciSession, list[str]]:
    out: list[str] = []
    session = UciSession(
        lambda s: UciEngine(single_from_batch(evaluate_batch), MCTSConfig(), s),
        out.append,
        default_sims=sims,
    )
    return session, out


def _bestmove(session: UciSession, out: list[str], position: str, go: str = "go") -> str:
    out.clear()
    session.handle(position)
    session.handle(go)
    lines = [line for line in out if line.startswith("bestmove ")]
    assert len(lines) == 1, out
    return lines[0].split()[1]


def test_uci_replays_batch_runner_games(tiny_batch_evaluator) -> None:
    """Three concurrent ladder games, every ply re-asked over UCI: same move each time.

    The runner batches leaves across the three games; the session asks one
    position at a time. Equal moves therefore cover the adapter, the masking,
    the eval profile and the history count together.
    """
    sims, plies = 16, 24
    starts = []
    for epd, _id in load_openings()[:3]:
        board = chess.Board()
        board.set_epd(epd)  # as the ladder's `_pairing` builds its boards
        starts.append(board)
    fens = [b.fen() for b in starts]
    runner = BatchRunner(tiny_batch_evaluator, _EVAL, max_plies=plies)
    traces = list(runner.play_games(3, seeds=[0, 1, 2], num_sims=sims, boards=starts))

    session, out = _session(tiny_batch_evaluator, sims)
    checked = 0
    for fen, board, trace in zip(fens, starts, traces, strict=True):
        moves = [m.uci() for m in board.move_stack]
        assert len(moves) == len(trace.positions) > 0
        for k, played in enumerate(moves):
            prefix = " ".join(moves[:k])
            got = _bestmove(session, out, f"position fen {fen} moves {prefix}".rstrip())
            assert got == played, f"game from {fen}, ply {k}: uci {got} vs ladder {played}"
            checked += 1
    assert checked >= 30


def test_repetition_state_matches_the_ladder() -> None:
    """A position met for the second time is encoded and searched as the ladder does.

    Six scripted knight moves return to the position after ply 2; the runner
    then searches it with its own history. The session reaches the same
    position from ``position startpos moves ...`` and must produce the same
    root tokens (the repetition token included) and the same move.
    """
    script = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]

    def move_fn(_gi: int, board: chess.Board) -> chess.Move | None:
        ply = len(board.move_stack)
        return chess.Move.from_uci(script[ply]) if ply < len(script) else None

    board = chess.Board()
    runner = BatchRunner(stub_batch, _EVAL, max_plies=len(script) + 1)
    (trace,) = list(runner.play_games(1, seeds=[0], num_sims=8, boards=[board], move_fn=move_fn))
    assert len(trace.positions) == 1
    row = trace.positions[0]

    engine = UciEngine(single_from_batch(stub_batch), MCTSConfig(), 8)
    engine.set_position(None, script)
    assert engine.history[engine.board._transposition_key()] == 1, "seen once before"
    path_reps: Counter = Counter({engine.board._transposition_key(): 1})
    tokens = Search(engine.evaluate_fn, engine.mcts).leaf_tokens(
        engine.board, path_reps, lambda b: engine.history[b._transposition_key()]
    )
    np.testing.assert_array_equal(tokens, row.tokens)
    played = decode_move(row.played_ft // 64, row.played_ft % 64, row.played_promo)
    assert engine.search().move == played


def test_handshake_and_ready() -> None:
    session, out = _session(stub_batch)
    session.handle("uci")
    assert out[0].startswith("id name ")
    assert any(line.startswith("option name Sims type spin default 16") for line in out)
    assert out[-1] == "uciok"
    out.clear()
    session.handle("isready")
    assert out == ["readyok"]


def test_engine_is_built_lazily_and_once() -> None:
    built: list[int] = []

    def make(sims: int) -> UciEngine:
        built.append(sims)
        return UciEngine(single_from_batch(stub_batch), MCTSConfig(), sims)

    session = UciSession(make, lambda _line: None, default_sims=48)
    session.handle("uci")
    assert built == []
    session.handle("isready")
    session.handle("ucinewgame")
    session.handle("position startpos")
    assert built == [48]


def test_bestmove_is_legal_and_reports_its_budget() -> None:
    session, out = _session(stub_batch, sims=12)
    move = _bestmove(session, out, "position startpos moves e2e4 e7e5")
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert chess.Move.from_uci(move) in board.legal_moves
    info = [line for line in out if line.startswith("info ")]
    assert info and " nodes 12 " in info[0]


def test_go_nodes_and_the_sims_option_set_the_budget() -> None:
    session, out = _session(stub_batch, sims=12)
    _bestmove(session, out, "position startpos", "go nodes 5")
    assert " nodes 5 " in out[0]
    session.handle("setoption name Sims value 7")
    _bestmove(session, out, "position startpos", "go wtime 1000 btime 1000")
    assert " nodes 7 " in out[0]
    out.clear()
    session.handle("setoption name Sims value 0")
    assert out and out[0].startswith("info string")
    assert session.sims == 7


def test_go_infinite_waits_for_stop() -> None:
    session, out = _session(stub_batch)
    session.handle("position startpos")
    out.clear()
    session.handle("go infinite")
    assert out == []
    session.handle("stop")
    assert out[-1].startswith("bestmove ")


def test_terminal_position_answers_null_move() -> None:
    session, out = _session(stub_batch)
    mated = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert _bestmove(session, out, f"position fen {mated}") == "0000"


def test_bad_position_is_reported_and_leaves_the_old_one() -> None:
    session, out = _session(stub_batch)
    session.handle("position startpos moves e2e4")
    out.clear()
    session.handle("position startpos moves e2e5")
    assert out and out[0].startswith("info string position rejected")
    assert (
        session.engine.board.fen()
        == chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").fen()
    )


def test_quit_and_unknown_commands() -> None:
    session, out = _session(stub_batch)
    assert session.handle("debug on") is True
    assert session.handle("") is True
    assert session.handle("quit") is False
    assert out == []


def test_q_to_cp_is_odd_monotone_and_finite() -> None:
    qs = np.linspace(-1.0, 1.0, 41)
    cps = [q_to_cp(q) for q in qs]
    assert cps == sorted(cps)
    assert q_to_cp(0.0) == 0
    assert q_to_cp(0.5) == -q_to_cp(-0.5)
    assert abs(q_to_cp(1.0)) < 2000


def test_masked_policy_probs_normalises_over_legal_cells() -> None:
    logits = np.random.default_rng(0).normal(size=(64, 64)).astype(np.float32)
    mask = np.zeros((64, 64), dtype=bool)
    mask[12, 28] = mask[6, 21] = True
    probs = masked_policy_probs(logits, mask)
    assert probs[~mask].sum() == 0.0
    assert probs.sum() == pytest.approx(1.0)
    assert masked_policy_probs(logits, np.zeros((64, 64), dtype=bool)).sum() == 0.0
