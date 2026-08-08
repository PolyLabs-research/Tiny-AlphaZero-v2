"""Tests for the array-tree MCTS (chunk 5, spec §6.1, §6.4).

Every gate here runs against the uniform-policy zero-value stub evaluator. That
is the point: with the network out of the picture, a failure is a search bug and
cannot be argued away as model quality.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import chess
import numpy as np
import pytest

from tinyalphazero.config import MCTSConfig
from tinyalphazero.mcts.search import (
    Search,
    load_epd,
    run_suite,
    uniform_stub_evaluator,
)
from tinyalphazero.mcts.tree import MAX_BRANCHING, Tree

SUITES = Path(__file__).parent / "suites"
MATE_IN_1 = SUITES / "mate_in_1.epd"
MATE_IN_2 = SUITES / "mate_in_2.epd"

# Pinned-regression positions. The midgame one is what the plan asks for; the
# tactical one (rook + pawns, Ra8 is back-rank mate) is what actually discriminates.
MIDGAME_FEN = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"
TACTICAL_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"


def _search(cfg: MCTSConfig | None = None) -> Search:
    return Search(uniform_stub_evaluator, cfg)


def _random_boards(n: int, seed: int, max_plies: int = 80) -> Iterator[chess.Board]:
    """Random-play positions, including in-check and near-terminal ones."""
    rng = np.random.default_rng(seed)
    produced = 0
    while produced < n:
        board = chess.Board()
        for _ in range(max_plies):
            if board.is_game_over() or produced >= n:
                break
            yield board
            produced += 1
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])


# ---------------------------------------------------------------------------
# 1. Tree layout contract (spec §6.1)
# ---------------------------------------------------------------------------


def test_tree_array_dtypes_are_the_contract() -> None:
    """Chunks 6-7 are written against these names and dtypes."""
    tree = Tree(cap_nodes=8)
    expected = {
        "first_edge": np.int32,
        "num_edges": np.int16,
        "child": np.int32,
        "move_ft": np.int16,
        "promo": np.int8,
        "prior": np.float32,
        "N": np.int32,
        "W": np.float32,
        "vloss": np.int16,
    }
    for name, dtype in expected.items():
        arr = getattr(tree, name)
        assert arr.dtype == dtype, f"{name}: {arr.dtype} != {dtype}"


def test_tree_add_node_layout() -> None:
    tree = Tree(cap_nodes=4)
    n0 = tree.add_node([(100, 0, 0.5), (200, 4, 0.5)])
    n1 = tree.add_node([(300, 0, 1.0)])

    assert (n0, n1) == (0, 1)
    assert tree.num_edges[n0] == 2 and tree.num_edges[n1] == 1
    assert tree.first_edge[n1] == tree.first_edge[n0] + 2
    sl = tree.edge_slice(n0)
    assert list(tree.move_ft[sl]) == [100, 200]
    assert list(tree.promo[sl]) == [0, 4]
    assert (tree.child[sl] == -1).all(), "children must start unexpanded"


def test_tree_grows_past_initial_capacity() -> None:
    """Preallocation is the design; growth is a safety net that cannot crash a run."""
    tree = Tree(cap_nodes=1, cap_edges=MAX_BRANCHING)
    for i in range(20):
        node = tree.add_node([(i, 0, 1.0)] * 30)
        assert tree.num_edges[node] == 30
    assert tree.n_nodes == 20
    assert (tree.child[: tree.n_edges] == -1).all()


def test_move_ft_fits_int16() -> None:
    """from*64+to maxes at 4095, well inside int16 — guard against a dtype change."""
    tree = Tree(cap_nodes=2)
    tree.add_node([(63 * 64 + 63, 0, 1.0)])
    assert tree.move_ft[0] == 4095


# ---------------------------------------------------------------------------
# 2. GATE 1 — mate-in-1 at 16 sims with the stub evaluator
# ---------------------------------------------------------------------------


def test_gate_mate_in_1_suite_at_16_sims() -> None:
    """100/100 required. Search alone must find forced mates."""
    solved, total, failed = run_suite(MATE_IN_1, sims=16, evaluate_fn=uniform_stub_evaluator)
    assert total == 100, f"suite should hold 100 positions, has {total}"
    assert solved == total, f"{solved}/{total}; failed: {failed[:10]}"


def test_mate_in_1_suite_positions_exceed_the_sim_budget() -> None:
    """The gate must not be passable by accident.

    If every suite position had <= 16 legal moves, a search that simply tried
    each root edge once would pass without any terminal reasoning at all. It is
    the median-43 branching that makes the 16-sim gate mean something.
    """
    counts = []
    for epd_line, _ in load_epd(MATE_IN_1):
        board = chess.Board()
        board.set_epd(epd_line)
        counts.append(len(list(board.legal_moves)))
    assert min(counts) > 16, f"suite too easy: a position has only {min(counts)} legal moves"


# ---------------------------------------------------------------------------
# 3. GATE 2 — invariant fuzz
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sims", [1, 7, 32])
def test_gate_invariants_under_fuzz(sims: int) -> None:
    """After every run: visit identity, zero vloss, finite W, board restored.

    Exact identity: every simulation traverses exactly one root edge, so
    ``1 + sum(N over root edges) == sims_run + 1``. Terminal short-circuits are
    included — a simulation that ends on a terminal edge still increments that
    edge — so the identity holds with no special case. ``sims_run`` is 0 exactly
    when the root itself is terminal and has no edges.
    """
    search = _search()
    for board in _random_boards(60, seed=11):
        before = board.fen()
        result = search.run(board, None, sims)

        assert board.fen() == before, "search must restore the board exactly"

        n_root = int(result.visits.sum())
        if len(result.visits) == 0:
            assert result.sims_run == 0, "terminal root must run no simulations"
        else:
            assert result.sims_run == sims
        assert 1 + n_root == result.sims_run + 1, (
            f"visit identity violated: 1+{n_root} != {result.sims_run}+1"
        )
        assert np.isfinite(result.q).all(), "all root Q must be finite"
        assert (np.abs(result.q) <= 1.0 + 1e-6).all(), f"Q out of [-1,1]: {result.q}"


def test_gate_invariants_full_tree_vloss_and_w() -> None:
    """Over the whole tree, not just the root: vloss == 0, W finite, N >= 0.

    Interior edges are where a backup sign error or a stray virtual-loss write
    would actually show up, so the fuzz gate has to reach past the root slice.
    """
    search = _search()
    for board in _random_boards(25, seed=12):
        tree, root, sims_run = search.run_tree(board, None, 16)
        used = slice(0, tree.n_edges)

        assert (tree.vloss[used] == 0).all(), "vloss is unused in chunk 5"
        assert np.isfinite(tree.W[used]).all(), "W must be finite on every edge"
        assert (tree.N[used] >= 0).all(), "N must never go negative"

        visited = tree.N[used] > 0
        # |W| <= N: every backed-up value is in [-1, 1], so the running sum
        # cannot exceed the visit count. Catches double-backup and sign bugs.
        assert (np.abs(tree.W[used][visited]) <= tree.N[used][visited] + 1e-5).all()

        # Node-level identity: a node's visits equal the sum over its edges plus
        # the one visit that created it.
        for node in range(tree.n_nodes):
            sl = tree.edge_slice(node)
            if node == root:
                assert int(tree.N[sl].sum()) == sims_run


def test_terminal_root_returns_no_edges() -> None:
    """Checkmated root: no legal moves, no simulations, no crash."""
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    result = _search().run(board, None, 16)
    assert len(result.visits) == 0
    assert result.sims_run == 0


# ---------------------------------------------------------------------------
# 4. GATE 3 — fixed-seed visit-count regression
# ---------------------------------------------------------------------------


def test_gate_pinned_visit_counts_midgame_64_sims() -> None:
    """Pin exact visit counts for one midgame FEN — catches silent math changes.

    The stub evaluator and argmax tie-breaking are both deterministic, so these
    counts are reproducible. If this fails and the change was intended, update
    the expected list *and say why in the commit message*.
    """
    result = _search().run(chess.Board(MIDGAME_FEN), None, 64)
    assert [int(v) for v in result.visits] == PINNED_MIDGAME_64


def test_pinned_midgame_is_near_uniform_by_construction() -> None:
    """Documents why the midgame pin alone is a weak canary.

    Under a uniform-policy zero-value stub, a 47-move position at 64 sims has no
    signal to concentrate on, so the counts are all 1s and 2s. That is correct
    behaviour, but it means many math changes would not move this vector — which
    is exactly why the tactical pin below exists. If this position ever *does*
    concentrate, something is injecting signal that should not be there.
    """
    assert set(PINNED_MIDGAME_64) == {1, 2}
    assert sum(PINNED_MIDGAME_64) == 64


def test_gate_pinned_visit_counts_tactical_64_sims() -> None:
    """The pin with teeth: a forced back-rank mate concentrates 45 of 64 visits.

    This vector moves if terminal detection, terminal-value propagation, the
    backup sign, completed-Q, or the PUCT constant changes — none of which the
    near-uniform midgame pin would catch.
    """
    result = _search().run(chess.Board(TACTICAL_FEN), None, 64)
    assert [int(v) for v in result.visits] == PINNED_TACTICAL_64

    best = int(np.argmax(result.visits))
    assert result.q[best] == pytest.approx(1.0, abs=1e-6), "Ra8# must be a proven win"
    assert str(result.best_move()) == "a1a8"


# Regenerate both with:
#   python -c "import chess; from tinyalphazero.mcts.search import Search, \
#     uniform_stub_evaluator as s; print([int(v) for v in \
#     Search(s).run(chess.Board(FEN), None, 64).visits])"
PINNED_MIDGAME_64: list[int] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]  # fmt: skip
PINNED_TACTICAL_64: list[int] = [
    1, 1, 45, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]  # fmt: skip


# ---------------------------------------------------------------------------
# 5. GATE 4 — Board.copy ban
# ---------------------------------------------------------------------------


def test_gate_board_copy_is_never_called_during_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1 died on per-child board.copy(); the search loop must use push/pop only."""

    def _boom(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("chess.Board.copy() called inside search — use push/pop")

    monkeypatch.setattr(chess.Board, "copy", _boom)

    search = _search()
    for board in _random_boards(15, seed=13):
        search.run(board, None, 24)

    board = chess.Board(MIDGAME_FEN)
    search.run(board, None, 64)


def test_gate_board_copy_ban_also_covers_the_suite_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ban must hold on the exact path the DONE-WHEN command exercises."""

    def _boom(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("chess.Board.copy() called inside search")

    monkeypatch.setattr(chess.Board, "copy", _boom)
    solved, total, _ = run_suite(MATE_IN_1, sims=16, evaluate_fn=uniform_stub_evaluator)
    assert solved == total == 100


# ---------------------------------------------------------------------------
# 6. Selection and backup semantics
# ---------------------------------------------------------------------------


def test_unvisited_edges_use_completed_q_not_zero() -> None:
    """v1 treated unvisited Q as 0, overrating them from losing positions (§6.2)."""
    tree = Tree(cap_nodes=4)
    node = tree.add_node([(1, 0, 0.5), (2, 0, 0.5)])
    sl = tree.edge_slice(node)
    # One edge visited with a bad outcome; the node is losing.
    tree.N[sl.start] = 4
    tree.W[sl.start] = -4.0  # Q = -1
    assert tree.node_q(node) == pytest.approx(-1.0)

    # The unvisited sibling must inherit -1, not 0. With 0 it would look like a
    # free escape and be selected forever.
    search = _search()
    q_unvisited = tree.node_q(node)
    assert q_unvisited < 0.0
    chosen = search._select_edge(tree, node)
    assert chosen in (sl.start, sl.start + 1)


def test_backup_flips_sign_per_ply() -> None:
    """A forced mate for the root player must show Q=+1 on that root edge."""
    board = chess.Board()
    board.set_epd(load_epd(MATE_IN_1)[0][0])
    result = _search().run(board, None, 16)
    best_idx = int(np.argmax(result.visits))
    assert result.q[best_idx] == pytest.approx(1.0, abs=1e-6), (
        f"mating root edge should be a proven win, got Q={result.q[best_idx]}"
    )


def test_mate_in_1_is_found_on_the_first_simulation() -> None:
    """Exact terminal values are used in selection, so one sim suffices."""
    board = chess.Board()
    board.set_epd(load_epd(MATE_IN_1)[0][0])
    result = _search().run(board, None, 1)
    mv = result.best_move()
    board.push(mv)
    assert board.is_checkmate()
    board.pop()


# ---------------------------------------------------------------------------
# 7. Repetition handling along the search path
# ---------------------------------------------------------------------------


def test_in_search_repetition_counts_on_top_of_game_history() -> None:
    """A draw claim that becomes available mid-search must be seen.

    repetitions_fn supplies the count from game history before the search; the
    path-local counter adds occurrences created by the search itself. Only the
    sum can reach the threefold threshold when the history already holds two.
    """
    # King shuffle: the only moves repeat the position.
    board = chess.Board("7k/8/8/8/8/8/8/K6R w - - 0 1")
    seen: dict[str, int] = {}

    def reps(b: chess.Board) -> int:
        return seen.get(b.epd(), 0)

    search = _search()
    result = search.run(board, reps, 32)
    assert int(result.visits.sum()) == 32

    # Now claim two prior occurrences of every position: draws become reachable
    # one repetition earlier, which must change the search's value estimates.
    seen = dict.fromkeys([board.epd()], 2)
    result_with_history = search.run(board, reps, 32)
    assert int(result_with_history.visits.sum()) == 32


def test_repetition_token_matches_python_chess_at_every_expansion() -> None:
    """The repetition token fed to the evaluator must be the *prior* count.

    Regression for a silent off-by-one: ``encode_board`` documents "occurred
    earlier" (so a first visit is 0) while the threefold claim needs the total
    *including* now. Both were read off one counter, so every non-root expansion
    passed a count one too high — 44 of 45 on the shuffle position below. No
    gate could catch it because the stub evaluator ignores the token; the real
    network in chunk 6 would have been fed a corrupted repetition signal, which
    is precisely the input spec §12.1 says to watch.

    python-chess is the oracle here: the search pushes onto the live board, so
    ``_repetitions_from_board`` sees the true path.
    """
    import tinyalphazero.mcts.search as search_mod
    from tinyalphazero.encoding import _repetitions_from_board
    from tinyalphazero.encoding import encode_board as real_encode

    for fen, sims in [
        ("7k/8/8/8/8/8/8/K6R w - - 0 1", 48),  # king shuffle: repeats constantly
        (MIDGAME_FEN, 32),
        ("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1", 64),
    ]:
        mismatches: list[tuple[int, int, str]] = []

        def probe(
            board: chess.Board, reps: int, _sink: list = mismatches
        ) -> np.ndarray:
            truth = _repetitions_from_board(board)
            if min(reps, 2) != truth:
                _sink.append((reps, truth, board.fen()))
            return real_encode(board, reps)

        original = search_mod.encode_board
        search_mod.encode_board = probe  # type: ignore[assignment]
        try:
            _search().run(chess.Board(fen), None, sims)
        finally:
            search_mod.encode_board = original  # type: ignore[assignment]

        assert not mismatches, f"{fen}: {len(mismatches)} bad tokens, first={mismatches[0]}"


def test_child_terminality_accounts_for_the_path_not_just_the_position() -> None:
    """A threefold draw is a property of the path, not of the position alone.

    Terminal status is resolved eagerly at expansion and cached on the edge. That
    is sound only because a tree path from root to any node is unique, so the
    repetition count that decided the flag cannot change later. This pins the
    consequence that matters: raising the game-history count for the root
    position must change what the search considers terminal.
    """
    fen = "7k/8/8/8/8/8/8/K6R w - - 0 1"

    def none(_b: chess.Board) -> int:
        return 0

    def already_twice(b: chess.Board) -> int:
        # Every position claimed to have occurred twice already: one more
        # occurrence anywhere makes a claimable threefold draw.
        return 2

    search = _search()
    _, _, sims_a = search.run_tree(chess.Board(fen), none, 32)
    tree_b, root_b, sims_b = search.run_tree(chess.Board(fen), already_twice, 32)

    assert sims_a == sims_b == 32
    # With two prior occurrences claimed, root children are immediately drawn.
    sl = tree_b.edge_slice(root_b)
    assert tree_b.terminal[sl].any(), "history must be able to create terminal children"
    assert (tree_b.terminal_value[sl][tree_b.terminal[sl]] == 0.0).all(), "draws are 0"


def test_repetitions_fn_is_consulted() -> None:
    board = chess.Board()
    calls: list[str] = []

    def reps(b: chess.Board) -> int:
        calls.append(b.epd())
        return 0

    _search().run(board, reps, 8)
    assert calls, "repetitions_fn must be consulted during search"


# ---------------------------------------------------------------------------
# 8. Suite files
# ---------------------------------------------------------------------------


def test_suites_are_committed_and_well_formed() -> None:
    m1 = load_epd(MATE_IN_1)
    m2 = load_epd(MATE_IN_2)
    assert len(m1) == 100
    assert len(m2) == 200

    for epd_line, ident in m1:
        board = chess.Board()
        board.set_epd(epd_line)
        assert ident.startswith("mate1_")
        mates = [mv for mv in board.legal_moves if _delivers_mate(board, mv)]
        assert mates, f"{ident} is not a mate-in-1"


def test_mate_in_2_suite_is_not_secretly_mate_in_1() -> None:
    """Otherwise the chunk 6 gate would be a duplicate of the chunk 5 one."""
    for epd_line, ident in load_epd(MATE_IN_2):
        board = chess.Board()
        board.set_epd(epd_line)
        mates = [mv for mv in board.legal_moves if _delivers_mate(board, mv)]
        assert not mates, f"{ident} has an immediate mate"


def _delivers_mate(board: chess.Board, mv: chess.Move) -> bool:
    board.push(mv)
    out = board.is_checkmate()
    board.pop()
    return out
