"""Chunk 8: the self-play actor (spec §7).

The rules under test are bookkeeping rules, and bookkeeping errors here are the
kind that never raise: a sign flip in outcomes trains the value head to predict
the opponent's result, and a resignation streak counted over the wrong side
resigns twice as fast as specified. Both produce well-formed data.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from tinyalphazero.config import Config, SelfPlayConfig
from tinyalphazero.encoding import TURN_B, TURN_W
from tinyalphazero.mcts.batch_runner import GameTrace, TracePosition, uniform_stub_batch
from tinyalphazero.replay import ReplayBuffer
from tinyalphazero.selfplay import (
    RESIGN_RESULT,
    CheckpointPool,
    ResignTracker,
    fill_outcomes,
    game_seed,
    play_games,
    winner_of,
)


def _pos(ply: int, q: float, side: bool) -> TracePosition:
    return TracePosition(
        tokens=np.zeros(72, dtype=np.int8),
        improved_policy=[(64, 0, 1.0)],
        side_to_move=side,
        ply=ply,
        root_q=q,
    )


def _trace(qs: list[float]) -> GameTrace:
    """Alternating-colour trace from a list of per-move root values."""
    t = GameTrace()
    for i, q in enumerate(qs):
        t.positions.append(_pos(i, q, chess.WHITE if i % 2 == 0 else chess.BLACK))
    return t


# ---------------------------------------------------------------------------
# Resignation (synthetic Q sequences, per the plan)
# ---------------------------------------------------------------------------


def test_resign_needs_eight_of_the_SAME_side_s_moves() -> None:
    """The subtlety the plan names: eight consecutive *own* moves, not eight rows.

    A trace alternates colours, so a streak counted over rows fires after four
    moves from each side — half the specified patience, on every game.
    """
    tr = ResignTracker()
    trace = GameTrace()
    fired_at = None
    # 8 white moves below threshold, interleaved with healthy black moves.
    for i in range(16):
        white = i % 2 == 0
        trace.positions.append(
            _pos(i, -0.95 if white else 0.5, chess.WHITE if white else chess.BLACK)
        )
        if tr(trace) == RESIGN_RESULT and fired_at is None:
            fired_at = i
    assert fired_at == 14, f"fired at ply {fired_at}; expected the 8th white move"
    assert tr.would_resign_side == chess.WHITE


def test_resign_streak_resets_on_a_recovery() -> None:
    tr = ResignTracker()
    trace = GameTrace()
    qs = [-0.95] * 7 + [0.1] + [-0.95] * 7  # all white moves
    for i, q in enumerate(qs):
        trace.positions.append(_pos(i, q, chess.WHITE))
        assert tr(trace) is None, f"resigned at {i} despite the recovery at 7"
    assert tr.would_resign_ply is None


def test_resign_threshold_is_strict() -> None:
    """−0.90 exactly must not resign: the rule is "below", not "at or below"."""
    tr = ResignTracker()
    trace = GameTrace()
    for i in range(12):
        trace.positions.append(_pos(i, -0.90, chess.WHITE))
        assert tr(trace) is None
    assert tr.would_resign_ply is None


def test_disabled_games_record_the_would_be_resignation_but_play_on() -> None:
    """The 10% sample: the rule fires, the game continues, the ply is recorded.

    Without this the false-resign rate is unmeasurable — a stopped game has no
    ground truth about what would have happened.
    """
    tr = ResignTracker(disabled=True)
    trace = GameTrace()
    for i in range(12):
        trace.positions.append(_pos(i, -0.99, chess.WHITE))
        assert tr(trace) is None, "a disabled game must never stop"
    assert tr.would_resign_ply == 7
    assert tr.would_resign_side == chess.WHITE


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def test_outcome_perspective_alternates_with_the_mover() -> None:
    trace = _trace([0.0] * 5)
    fill_outcomes(trace, chess.WHITE)
    got = [p.outcome for p in trace.positions]
    assert got == [1, -1, 1, -1, 1], got


def test_draws_are_zero_everywhere() -> None:
    trace = _trace([0.0] * 4)
    fill_outcomes(trace, None)
    assert [p.outcome for p in trace.positions] == [0, 0, 0, 0]


@pytest.mark.parametrize(
    ("result", "resign_side", "expect"),
    [
        ("draw_claim", None, None),
        ("ply_cap", None, None),
        ("draw", None, None),
        (RESIGN_RESULT, chess.WHITE, chess.BLACK),
        (RESIGN_RESULT, chess.BLACK, chess.WHITE),
    ],
)
def test_winner_of_scores_each_result_kind(result, resign_side, expect) -> None:
    """§7's scoring table, including that a resignation is a loss for the resigner."""
    trace = GameTrace()
    trace.result = result
    assert winner_of(trace, chess.Board(), resign_side) == expect


def test_checkmate_winner_is_the_side_not_to_move() -> None:
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    trace = GameTrace()
    trace.result = "checkmate"
    assert winner_of(trace, board, None) == chess.BLACK


# ---------------------------------------------------------------------------
# League
# ---------------------------------------------------------------------------


def test_checkpoint_pool_keeps_only_the_last_n() -> None:
    pool = CheckpointPool(SelfPlayConfig(league_pool_size=3))
    for i in range(6):
        pool.add(f"ckpt_{i}")
    assert pool.snapshots == ["ckpt_3", "ckpt_4", "ckpt_5"]
    assert pool.latest == "ckpt_5"


def test_checkpoint_pool_sampling_ratio_chi_squared() -> None:
    """80/20 latest-vs-pool over 10K draws, per the plan.

    χ² against the specified split rather than an eyeball on the ratio: the
    league mix determines what the network is actually trained against, and a
    drift to "almost always latest" is self-play against a single opponent
    wearing a league's clothes.
    """
    cfg = SelfPlayConfig(league_latest_frac=0.8, league_pool_size=20)
    pool = CheckpointPool(cfg)
    for i in range(20):
        pool.add(f"ckpt_{i}")

    rng = np.random.default_rng(11)
    draws = [pool.sample(rng) for _ in range(10_000)]
    latest = sum(d == "ckpt_19" for d in draws)

    # Latest is drawn directly 80% of the time, plus 20%/20 = 1% from the pool.
    expected_latest = 10_000 * (0.8 + 0.2 / 20)
    other = 10_000 - latest
    chi2 = (latest - expected_latest) ** 2 / expected_latest + (
        other - (10_000 - expected_latest)
    ) ** 2 / (10_000 - expected_latest)
    assert chi2 < 10.83, f"chi2={chi2:.2f} (p<0.001, 1 dof) latest={latest}"

    # And the non-latest mass really is spread over the pool, not one snapshot.
    assert len(set(draws)) >= 15


def test_empty_pool_samples_none() -> None:
    assert CheckpointPool().sample(np.random.default_rng(0)) is None


# ---------------------------------------------------------------------------
# Seeding and the end-to-end path
# ---------------------------------------------------------------------------


def test_game_seed_is_distinct_and_reproducible() -> None:
    """Any single game replayable in isolation (spec §9.2)."""
    seeds = [game_seed(42, i) for i in range(100)]
    assert len(set(seeds)) == 100, "seed fan-out collided"
    assert seeds == [game_seed(42, i) for i in range(100)], "not reproducible"
    assert seeds != [game_seed(43, i) for i in range(100)], "base seed ignored"


def test_play_games_scores_and_stores(tmp_path) -> None:
    cfg = Config()
    cfg.selfplay.max_plies = 6
    buf = ReplayBuffer.create(tmp_path / "replay", capacity=256)
    traces, stats = play_games(
        uniform_stub_batch, num_games=3, cfg=cfg, seed=7, num_sims=8, buffer=buf
    )

    assert len(traces) == 3
    assert stats.games == 3
    assert stats.positions == sum(len(t.positions) for t in traces)
    assert len(buf) == stats.positions
    for t in traces:
        assert t.result != "unfinished"
        assert all(p.outcome in (-1, 0, 1) for p in t.positions)

    batch = buf.sample(8, np.random.default_rng(0))
    assert batch["policy"].shape[0] == 8
    assert float(batch["policy"].sum(dim=1).min()) > 0.99


def _lost_for(loser: bool):
    """Flat policy; every position is worth -1 to ``loser`` and +1 to the other side.

    Values are from the side to move's view, read off the turn token, so the
    loser's root completed-Q sits at -1 wherever the search looks.
    """
    loser_turn = TURN_W if loser == chess.WHITE else TURN_B

    def evaluate(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits, promo, _ = uniform_stub_batch(tokens)
        return logits, promo, np.where(tokens[:, 64] == loser_turn, -1.0, 1.0).astype(np.float32)

    return evaluate


@pytest.mark.parametrize(
    "loser", [chess.WHITE, chess.BLACK], ids=["white_resigns", "black_resigns"]
)
def test_a_decisive_game_reaches_the_ring_signed_for_each_mover(tmp_path, loser) -> None:
    """``play_games → _store`` with a decided game: winner's rows +1, loser's -1.

    ``test_play_games_scores_and_stores`` caps every game at 6 plies, so the only
    outcome it ever stores is 0 and a sign error anywhere on the decisive path —
    ``winner_of`` on a resignation, ``fill_outcomes``, the ring write — passes it
    (REVIEW-2026-10-21 C10/T11). Here the evaluator makes one side lost, the
    resignation rule fires on its eighth own move, and every stored row is
    checked against the turn token in the ring and against the search's own
    ``root_q`` for that row.
    """
    cfg = Config()
    cfg.selfplay.resign_disabled_frac = 0.0  # every game may resign
    buf = ReplayBuffer.create(tmp_path / "replay", capacity=256)
    traces, stats = play_games(
        _lost_for(loser), num_games=2, cfg=cfg, seed=7, num_sims=8, buffer=buf
    )

    assert stats.results == {RESIGN_RESULT: 2}
    own_moves = cfg.selfplay.resign_consecutive
    resign_ply = 2 * (own_moves - 1) + (0 if loser == chess.WHITE else 1)
    for t in traces:
        assert t.would_resign_ply == resign_ply
        assert len(t.positions) == resign_ply + 1

    n = len(buf)
    assert n == stats.positions
    loser_turn = TURN_W if loser == chess.WHITE else TURN_B
    loser_rows = np.asarray(buf._arr["boards"][:n, 64]) == loser_turn
    outcomes = np.asarray(buf._arr["outcomes"][:n])
    root_q = np.asarray(buf._arr["root_q"][:n], dtype=np.float32)
    assert loser_rows.any() and (~loser_rows).any(), "non-vacuous: both movers stored"
    assert (outcomes[loser_rows] == -1).all(), "the resigner's moves are losses"
    assert (outcomes[~loser_rows] == 1).all(), "the winner's moves are wins"
    assert (np.sign(root_q) == outcomes).all(), "label and search disagree in sign"


def test_the_ring_stores_the_move_actually_played(tmp_path) -> None:
    """`moves.dat` is the game's move, not the improved-policy argmax (REVIEW-2026-10-21 C4).

    The acting rule is the argmax over the *survivors* of g + logits + σ(q̂); the
    improved policy is softmax(logits + σ(q̂)) over all legal moves with no g. Under
    self-play noise the two differed on ~95% of positions, and the ring recorded
    the latter under the name "the move actually played" — so the promotion head
    was trained at a cell the game may never have reached.
    """
    import chess

    from tinyalphazero.mcts.batch_runner import BatchRunner

    cfg = Config()  # root_noise on: the self-play profile, where the two diverge
    runner = BatchRunner(uniform_stub_batch, cfg.mcts, max_plies=12, keep_debug_trees=False)
    traces = list(runner.play_games(2, seeds=[1, 2], num_sims=8))
    for gi, trace in enumerate(traces):
        board = chess.Board()
        for pos, move in zip(
            trace.positions, runner._games_debug[gi].board.move_stack, strict=True
        ):
            assert pos.played_ft == move.from_square * 64 + move.to_square
            assert pos.played_promo == (0 if move.promotion is None else pos.played_promo)
            board.push(move)
    positions = [p for t in traces for p in t.positions]
    argmax_ft = [max(p.improved_policy, key=lambda e: e[2])[0] for p in positions]
    played_ft = [p.played_ft for p in positions]
    assert played_ft != argmax_ft, (
        "non-vacuous: the acting move and the argmax must differ somewhere"
    )

    buf = ReplayBuffer.create(tmp_path / "replay", capacity=256)
    cfg.selfplay.max_plies = 12
    traces, stats = play_games(
        uniform_stub_batch, num_games=2, cfg=cfg, seed=7, num_sims=8, buffer=buf
    )
    stored = sorted(int(f) * 64 + int(t) for f, t in np.asarray(buf._arr["moves"][: len(buf)]))
    assert stored == sorted(p.played_ft for t in traces for p in t.positions)
    assert buf.version == 4, "a ring whose moves are the played move says so"


def test_play_games_is_reproducible_from_the_seed(tmp_path) -> None:
    cfg = Config()
    cfg.selfplay.max_plies = 4
    a, _ = play_games(uniform_stub_batch, 2, cfg=cfg, seed=3, num_sims=8)
    b, _ = play_games(uniform_stub_batch, 2, cfg=cfg, seed=3, num_sims=8)
    assert [t.result for t in a] == [t.result for t in b]
    assert [len(t.positions) for t in a] == [len(t.positions) for t in b]
    for ta, tb in zip(a, b, strict=True):
        for pa, pb in zip(ta.positions, tb.positions, strict=True):
            np.testing.assert_array_equal(pa.tokens, pb.tokens)


def test_false_resign_rate_is_none_before_it_is_measured() -> None:
    """An honest "not measured" rather than a 0.0 that reads as calibrated."""
    from tinyalphazero.selfplay import SelfPlayStats

    assert SelfPlayStats().false_resign_rate is None
    s = SelfPlayStats(would_have_resigned=4, false_resigns=1)
    assert s.false_resign_rate == 0.25
