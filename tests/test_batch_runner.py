"""Chunk 7: batched leaf evaluation across concurrent games (spec §6.3).

The equivalence gate is the load-bearing test. Everything else in this chunk is a
throughput optimisation, and a throughput optimisation that changes the answer is
a bug wearing a benchmark.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from tinyalphazero.config import MCTSConfig
from tinyalphazero.mcts.batch_runner import BatchRunner, GameTrace
from tinyalphazero.mcts.search import Search, load_epd, uniform_stub_evaluator
from tinyalphazero.mcts.tree import Tree

MATE_IN_1 = "tests/suites/mate_in_1.epd"
MATE_IN_2 = "tests/suites/mate_in_2.epd"
MIDGAME = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"


def stub_batch(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched analogue of the uniform stub: flat logits, zero value.

    Flat logits become a uniform distribution once the runner applies the legal
    mask, which is exactly what `uniform_stub_evaluator` produces — so the two
    paths are comparable by construction.
    """
    b = tokens.shape[0]
    return np.zeros((b, 64, 64), dtype=np.float32), np.zeros(b, dtype=np.float32)


def _board(epd_or_fen: str) -> chess.Board:
    board = chess.Board()
    if " bm " in epd_or_fen or 'id "' in epd_or_fen:
        board.set_epd(epd_or_fen)
    else:
        board.set_fen(epd_or_fen)
    return board


# Fixtures chosen to exercise the post-spec search features, per the chunk 5/6
# review: a solver-proving position and a position with several mating moves
# where the immediate-win bonus decides the tie.
FIXTURES = {
    "midgame": MIDGAME,
    "mate_in_1": load_epd(MATE_IN_1)[0][0],
    "mate_in_2": load_epd(MATE_IN_2)[0][0],
    "multi_mate": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
}


def _cfg(**kw) -> MCTSConfig:
    """Self-play profile by default, pinned explicitly.

    ``root_noise`` is a dispatch flag on a defaulted field, which is precisely
    the shape that silently re-pointed three chunk-5 tests when ``root_mode``
    arrived. Set it here so a future default flip fails these tests instead of
    quietly changing what they measure.
    """
    cfg = MCTSConfig()
    cfg.batch_leaves = 1
    cfg.root_noise = True
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# 1. THE EQUIVALENCE GATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("sims", [16, 48])
@pytest.mark.parametrize("root_noise", [True, False], ids=["selfplay", "eval"])
def test_gate_equivalence_with_synchronous_search(
    name: str, sims: int, root_noise: bool
) -> None:
    """batch_leaves=1, num_games=1, same seed → identical search, not merely a
    matching move.

    Compares visit counts, solver `proven` flags, terminal flags and the acting
    move. Two searches can agree on the move while disagreeing about everything
    behind it, and it is the disagreement that would surface later as an
    unreproducible self-play run.

    Run under **both** profiles. The runner reimplements the Gumbel root rather
    than calling into ``Search``, so "the eval profile works" is a claim about
    two independent pieces of code, and this gate is the only place the two are
    compared directly.
    """
    epd = FIXTURES[name]
    cfg = _cfg(root_noise=root_noise)

    sync = Search(uniform_stub_evaluator, cfg, np.random.default_rng(7)).run(
        _board(epd), None, sims
    )

    runner = BatchRunner(stub_batch, cfg, max_plies=1)
    list(runner.play_games(1, seeds=[7], num_sims=sims, boards=[_board(epd)]))
    tree, root, sims_run = runner.debug_trees[(0, 0)]
    sl = tree.edge_slice(root)

    assert sims_run == sync.sims_run == sims
    assert list(tree.N[sl]) == list(sync.visits), f"{name}: visit counts diverged"
    assert list(tree.terminal[sl]) == list(
        _sync_tree_flags(epd, cfg, sims, "terminal")
    ), f"{name}: terminal flags diverged"
    assert list(tree.proven[sl]) == list(
        _sync_tree_flags(epd, cfg, sims, "proven")
    ), f"{name}: solver proven flags diverged"


def _sync_tree_flags(epd: str, cfg: MCTSConfig, sims: int, attr: str) -> np.ndarray:
    """Re-run the synchronous search and read a flag array off its tree."""
    search = Search(uniform_stub_evaluator, cfg, np.random.default_rng(7))
    board = _board(epd)
    tree, root, _ = search.run_tree(board, None, 0)
    # run_gumbel builds its own tree; re-drive it to get the post-search tree.
    search2 = Search(uniform_stub_evaluator, cfg, np.random.default_rng(7))
    search2.run(_board(epd), None, sims)
    tree2, root2 = search2._last_tree, search2._last_root
    return getattr(tree2, attr)[tree2.edge_slice(root2)]


def test_equivalence_holds_across_a_whole_game() -> None:
    """Not just move 1: divergence usually starts a few plies in."""
    cfg = _cfg()
    board_a, board_b = _board(MIDGAME), _board(MIDGAME)

    runner = BatchRunner(stub_batch, cfg, max_plies=4)
    list(runner.play_games(1, seeds=[3], num_sims=16, boards=[board_b]))

    search = Search(uniform_stub_evaluator, cfg, np.random.default_rng(3))
    for ply in range(4):
        result = search.run(board_a, None, 16)
        tree, root, _ = runner.debug_trees[(0, ply)]
        assert list(tree.N[tree.edge_slice(root)]) == list(result.visits), f"ply {ply}"
        board_a.push(result.best_move())


# ---------------------------------------------------------------------------
# 2. Virtual loss × proven values
# ---------------------------------------------------------------------------


def test_vloss_diverges_concurrent_paths() -> None:
    """Two in-flight sims in one tree must not stack on the same edge."""
    cfg = _cfg()
    search = Search(uniform_stub_evaluator, cfg)
    board = _board(MIDGAME)
    tree, root, _ = search.run_tree(board, None, 0)

    from collections import Counter

    reps: Counter = Counter()
    reps[board._transposition_key()] += 1

    g1 = search.simulate_gen(tree, root, board, reps, lambda _b: 0, use_vloss=True)
    next(g1)
    in_flight = np.where(tree.vloss[tree.edge_slice(root)] > 0)[0]
    assert len(in_flight) == 1, "one in-flight sim should mark exactly one root edge"

    chosen_second = search._select_edge(tree, root) - tree.edge_slice(root).start
    assert chosen_second != in_flight[0], (
        "the second simulation picked the edge already in flight — virtual loss is "
        "not steering concurrent sims apart"
    )


def test_proven_edges_ignore_virtual_loss() -> None:
    """A certainty must not look worse the more simulations agree with it.

    Q_eff = (W - vloss)/(N + vloss) applied to a proven win would drive a +1 edge
    downward exactly as the search concentrates on it. Exact values are exempt.
    """
    tree = Tree(cap_nodes=4)
    node = tree.add_node([(1, 0, 0.5), (2, 0, 0.5)])
    sl = tree.edge_slice(node)

    tree.terminal[sl.start] = True
    tree.terminal_value[sl.start] = -1.0  # child is mated → +1 for this node
    tree.vloss[sl.start] = 8  # heavy in-flight traffic on the proven edge
    tree.N[sl.start + 1] = 4
    tree.W[sl.start + 1] = 3.0  # a merely good sibling

    search = Search(uniform_stub_evaluator, _cfg())
    assert search._select_edge(tree, node) == sl.start, (
        "virtual loss dragged a proven win below a heuristic sibling"
    )


# ---------------------------------------------------------------------------
# 3. Q4 — terminal short-circuits must not desync leaf accounting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_leaves", [1, 4, 32])
def test_leaf_accounting_balances_with_terminals_present(batch_leaves: int) -> None:
    """leaves == sims - terminal_shortcircuits + root_expansions. Exactly.

    A terminal short-circuit never reaches the evaluator, so it must never enter
    the buffer; a root expansion does. If these drift, the batch runner is either
    dropping evaluations or double-counting them, and the throughput numbers stop
    describing the work done.
    """
    cfg = _cfg(batch_leaves=batch_leaves)
    runner = BatchRunner(stub_batch, cfg, max_plies=6)
    boards = [_board(FIXTURES["mate_in_1"]), _board(MIDGAME), _board(FIXTURES["mate_in_2"])]
    traces = list(runner.play_games(3, seeds=[1, 2, 3], num_sims=16, boards=boards))

    root_expansions = sum(len(t.positions) for t in traces)
    expected = runner.stats["sims"] - runner.stats["terminal_shortcircuits"] + root_expansions
    assert runner.stats["leaves"] == expected, (
        f"leaf accounting desynced: leaves={runner.stats['leaves']} != {expected} "
        f"(stats={runner.stats}, roots={root_expansions})"
    )
    assert runner.stats["terminal_shortcircuits"] > 0, (
        "fixture set did not actually exercise terminal short-circuits"
    )


def test_vloss_is_zero_after_every_flush() -> None:
    """Virtual loss is a flight-window artefact; none may survive the batch."""
    cfg = _cfg(batch_leaves=8)
    runner = BatchRunner(stub_batch, cfg, max_plies=4)
    list(runner.play_games(4, seeds=[1, 2, 3, 4], num_sims=16))
    for (_g, _ply), (tree, _root, _s) in runner.debug_trees.items():
        assert (tree.vloss[: tree.n_edges] == 0).all(), "leaked virtual loss"


# ---------------------------------------------------------------------------
# 4. Runner behaviour
# ---------------------------------------------------------------------------


def test_games_run_to_completion_and_trace() -> None:
    cfg = _cfg(batch_leaves=16)
    runner = BatchRunner(stub_batch, cfg, max_plies=8)
    traces = list(runner.play_games(4, seeds=[1, 2, 3, 4], num_sims=16))

    assert len(traces) == 4
    for t in traces:
        assert isinstance(t, GameTrace)
        assert t.result in {"checkmate", "draw", "ply_cap"}
        assert len(t.positions) == t.plies
        for pos in t.positions:
            assert pos.tokens.shape == (72,)
            total = sum(p for _ft, _pr, p in pos.improved_policy)
            assert total == pytest.approx(1.0, abs=1e-4)
            assert -1.0 <= pos.root_q <= 1.0


def test_sims_per_move_is_exactly_the_budget() -> None:
    """Concurrency must not change how much search each move receives."""
    cfg = _cfg(batch_leaves=16)
    runner = BatchRunner(stub_batch, cfg, max_plies=5)
    traces = list(runner.play_games(3, seeds=[5, 6, 7], num_sims=24))
    moves = sum(len(t.positions) for t in traces)
    assert runner.stats["sims"] == 24 * moves


def test_batching_actually_batches() -> None:
    """More concurrent games must mean fuller batches, or none of this is worth it."""
    cfg = _cfg(batch_leaves=64)
    lonely = BatchRunner(stub_batch, cfg, max_plies=3)
    list(lonely.play_games(1, seeds=[1], num_sims=16))
    crowd = BatchRunner(stub_batch, cfg, max_plies=3)
    list(crowd.play_games(16, seeds=list(range(16)), num_sims=16))

    fill_one = lonely.stats["leaves"] / max(lonely.stats["forwards"], 1)
    fill_many = crowd.stats["leaves"] / max(crowd.stats["forwards"], 1)
    assert fill_one == pytest.approx(1.0, abs=1e-9)
    assert fill_many > 4 * fill_one, f"batching gained nothing: {fill_one} → {fill_many}"


def test_inline_evaluation_is_refused() -> None:
    """If the batched path is ever bypassed the runner must fail loudly."""
    from tinyalphazero.mcts.batch_runner import _unused_evaluator

    with pytest.raises(AssertionError, match="batched path was bypassed"):
        _unused_evaluator(np.zeros(72, dtype=np.int8))


# ---------------------------------------------------------------------------
# The evaluation profile on the batched path (BLOCKED-2026-08-08)
#
# Chunk 10's ladder plays its matches through BatchRunner, so the profile has to
# hold here as well as in Search. Before this was wired the runner drew Gumbel
# noise unconditionally: suite numbers would have been deterministic and ladder
# numbers noisy, with nothing to say the two disagreed.
# ---------------------------------------------------------------------------


def test_eval_profile_batched_is_seed_independent() -> None:
    """Different seeds, identical trees — the property the ladder needs."""
    cfg = _cfg(root_noise=False, batch_leaves=8)
    trees = []
    for seed in (11, 22):
        runner = BatchRunner(stub_batch, cfg, max_plies=3)
        list(runner.play_games(2, seeds=[seed, seed + 1], num_sims=24))
        trees.append(
            {k: tree.N[tree.edge_slice(root)].tolist()
             for k, (tree, root, _) in runner.debug_trees.items()}
        )
    assert trees[0] == trees[1], "batched eval profile still depends on the seed"


def test_selfplay_profile_batched_still_varies() -> None:
    """The other side of it: self-play data diversity must survive."""
    seen = set()
    for seed in range(5):
        runner = BatchRunner(stub_batch, _cfg(batch_leaves=8), max_plies=3)
        list(runner.play_games(1, seeds=[seed], num_sims=24))
        tree, root, _ = runner.debug_trees[(0, 0)]
        seen.add(tuple(tree.N[tree.edge_slice(root)].tolist()))
    assert len(seen) > 1, "batched self-play profile went deterministic"


def test_batched_runner_resolves_eval_overrides() -> None:
    """``eval_m`` must reach the runner's own considered-set arithmetic.

    The runner reads ``self.cfg.gumbel_m`` directly, so without resolving the
    profile it would consider 16 actions while ``Search`` considered 32 — the
    two implementations disagreeing about the same config.
    """
    cfg = _cfg(root_noise=False, eval_m=32, eval_c_visit=25.0, eval_c_scale=2.0)
    runner = BatchRunner(stub_batch, cfg, max_plies=1)
    assert runner.cfg.gumbel_m == 32
    assert runner.cfg.c_visit == 25.0
    assert runner.cfg.c_scale == 2.0
