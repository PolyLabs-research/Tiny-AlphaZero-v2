"""Tests for the array-tree MCTS (chunk 5, spec §6.1, §6.4).

Every gate here runs against the uniform-policy zero-value stub evaluator. That
is the point: with the network out of the picture, a failure is a search bug and
cannot be argued away as model quality.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import pytest

from tinyalphazero.config import MCTSConfig
from tinyalphazero.mcts.gumbel import allocate_within_phase, halving_phase_budgets
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


def _puct_cfg() -> MCTSConfig:
    """Chunk 5 sections below test *PUCT* selection specifically.

    MCTSConfig.root_mode defaults to "gumbel" (the self-play default), so these
    must pin the mode explicitly -- otherwise adding chunk 6's dispatch silently
    changes what they measure, which is exactly what it did: the pinned visit
    vectors are PUCT artefacts and mean nothing under Sequential Halving.
    """
    cfg = MCTSConfig()
    cfg.root_mode = "puct"
    cfg.dirichlet_frac = 0.0  # deterministic: the pins are exact-value regressions
    return cfg


def _search(cfg: MCTSConfig | None = None) -> Search:
    return Search(uniform_stub_evaluator, cfg if cfg is not None else _puct_cfg())


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


def _repetition_oracle_mismatches(
    board: chess.Board, sims: int, base_reps: object = None
) -> list[tuple[int, int, str]]:
    """Run a search, checking every expansion's repetition token against python-chess.

    The oracle is sound because the search pushes onto the *live* board, so its
    move stack carries ground truth — for FEN-rooted positions (empty history)
    and full-history self-play boards alike. It is only valid when the history
    lives in the move stack, i.e. ``repetitions_fn`` contributes 0; a non-zero
    ``repetitions_fn`` describes occurrences python-chess cannot see.
    """
    import tinyalphazero.mcts.search as search_mod
    from tinyalphazero.encoding import _repetitions_from_board
    from tinyalphazero.encoding import encode_board as real_encode

    bad: list[tuple[int, int, str]] = []

    def probe(b: chess.Board, reps: int, _sink: list = bad) -> np.ndarray:
        truth = _repetitions_from_board(b)
        if min(reps, 2) != truth:
            _sink.append((reps, truth, b.fen()))
        return real_encode(b, reps)

    original = search_mod.encode_board
    search_mod.encode_board = probe  # type: ignore[assignment]
    try:
        _search().run(board, base_reps, sims)  # type: ignore[arg-type]
    finally:
        search_mod.encode_board = original  # type: ignore[assignment]
    return bad


# Shuffle-heavy positions: sparse material, few captures, so search paths
# revisit positions constantly and the repetition counter is exercised hard.
SHUFFLE_FENS = [
    "7k/8/8/8/8/8/8/K6R w - - 0 1",  # rook + kings
    "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1",  # rook endgame, open board
    "8/8/4k3/8/8/4K3/8/6Q1 w - - 0 1",  # queen + kings
    "8/5k2/8/8/8/8/5K2/6N1 w - - 0 1",  # knight: no mate, pure shuffling
    "4k3/8/8/8/8/8/8/4K2R b K - 0 1",  # black to move, castling rights live
    "8/8/8/3k4/8/3K4/8/3B4 w - - 40 60",  # high halfmove clock: near 50-move
    "6k1/6p1/8/8/8/8/6P1/6K1 w - - 0 1",  # blocked pawns, shuffle-forced
]


@pytest.mark.parametrize("fen", SHUFFLE_FENS)
@pytest.mark.parametrize("sims", [16, 64])
def test_repetition_token_fuzz_shuffle_heavy(fen: str, sims: int) -> None:
    """GATE: the repetition token must equal python-chess's prior count. Always.

    This is the gate that was missing when the off-by-one shipped — the stub
    ignores the token, so no existing gate could observe it, and the first
    diagnosable symptom would have been vague value-miscalibration at M3, by
    which point §12.1's escalation (add history frames) would have been applied
    to a bug and made nothing better.
    """
    bad = _repetition_oracle_mismatches(chess.Board(fen), sims)
    assert not bad, f"{fen} @{sims} sims: {len(bad)} bad tokens, first={bad[0]}"


def test_repetition_token_fuzz_with_real_move_history() -> None:
    """Same gate on boards carrying genuine history, as in self-play.

    A FEN-rooted board starts with an empty move stack; a self-play board does
    not, and the prior-occurrence count then spans both the pre-search history
    and the in-search path. Both must agree with the oracle.
    """
    rng = np.random.default_rng(23)
    board = chess.Board()
    checked = 0
    for ply in range(60):
        if board.is_game_over():
            break
        if ply >= 8 and ply % 4 == 0:
            bad = _repetition_oracle_mismatches(board, 24)
            assert not bad, f"ply {ply}: {len(bad)} bad tokens, first={bad[0]}"
            checked += 1
        moves = list(board.legal_moves)
        board.push(moves[int(rng.integers(len(moves)))])
    assert checked >= 8, f"fuzz did not exercise enough history depth ({checked})"


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


# ---------------------------------------------------------------------------
# 9. Chunk 6 — Gumbel root selection and Sequential Halving (spec §6.2)
# ---------------------------------------------------------------------------

ANCHOR = Path("runs/phase1_anchor.pt")
requires_anchor = pytest.mark.skipif(
    not ANCHOR.exists(), reason="runs/phase1_anchor.pt not present (untracked, 75 MB)"
)


def _gumbel_cfg(**kw) -> MCTSConfig:
    """The **self-play** Gumbel profile: noise on, spec constants.

    Both dispatch flags are pinned explicitly rather than inherited from the
    dataclass defaults. Same lesson as ``_puct_cfg``: every test below that
    reasons about g — force-include, the noise-driven tiebreak that
    ``IMMEDIATE_WIN_BONUS`` fixes, the 46/100 regression — is only meaningful
    with noise actually on, and a future default flip would silently repoint the
    whole block at a different search instead of failing.
    """
    cfg = MCTSConfig()
    cfg.root_mode = "gumbel"
    cfg.root_noise = True
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _eval_cfg(**kw) -> MCTSConfig:
    """The **evaluation** profile: deterministic root, no Gumbel draw."""
    cfg = MCTSConfig()
    cfg.root_mode = "gumbel"
    cfg.root_noise = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.parametrize(
    ("n", "m"), [(48, 16), (128, 16), (16, 16), (50, 16), (48, 8), (64, 32), (200, 16), (7, 4)]
)
def test_halving_phase_budgets_total_exactly_the_budget(n: int, m: int) -> None:
    """Sequential Halving must spend the budget exactly — no more, no less."""
    budgets = halving_phase_budgets(n, m)
    assert sum(budgets) == n, f"n={n} m={m}: {budgets} sums to {sum(budgets)}"
    assert all(b >= 0 for b in budgets)


def test_halving_worked_example_matches_the_docstring() -> None:
    """n=48, m=16 → [16, 8, 12, 12]. Pinned so the documented example stays true."""
    assert halving_phase_budgets(48, 16) == [16, 8, 12, 12]


@pytest.mark.parametrize(("budget", "k"), [(12, 16), (12, 8), (13, 2), (7, 3), (0, 4)])
def test_allocate_within_phase_sums_to_phase_budget(budget: int, k: int) -> None:
    alloc = allocate_within_phase(budget, k)
    assert sum(alloc) == budget
    assert len(alloc) == k


@pytest.mark.parametrize(("sims", "m"), [(48, 16), (128, 16), (32, 8)])
@pytest.mark.parametrize("root_noise", [True, False], ids=["selfplay", "eval"])
def test_gumbel_simulations_total_exactly_the_budget(
    sims: int, m: int, root_noise: bool
) -> None:
    """GATE: simulations == budget exactly, in **both** profiles.

    Chunk 5 ratified that terminal hits *consume* budget, so this is the
    identity that holds unconditionally. Evaluator calls are a different
    quantity — see the next test — and asserting "leaf evaluations == budget"
    would silently be asserting the absence of terminal positions.

    Both profiles, because the eval profile changes *which* actions are
    considered and a considered set of the wrong size would surface here first:
    Sequential Halving derives its per-phase arithmetic from
    ``len(considered)``, so a wrong-sized set misspends the budget rather than
    raising.
    """
    cfg = (_gumbel_cfg if root_noise else _eval_cfg)(gumbel_m=m)
    search = Search(uniform_stub_evaluator, cfg, np.random.default_rng(0))
    for board in _random_boards(12, seed=41):
        result = search.run(board, None, sims)
        if len(result.visits) == 0:
            continue
        assert result.sims_run == sims
        assert int(result.visits.sum()) == sims


def test_evaluations_equal_budget_only_without_terminals() -> None:
    """Evaluator calls <= simulations, strictly less when terminals short-circuit."""
    search = Search(uniform_stub_evaluator, _gumbel_cfg(), np.random.default_rng(0))

    # Quiet midgame: no terminal inside a 48-sim horizon.
    quiet = search.run(chess.Board(MIDGAME_FEN), None, 48)
    assert quiet.evaluations == quiet.sims_run + 1, "root expansion plus one per sim"

    # Mate-in-1 position: the mating edge is terminal, so sims that take it
    # never reach the evaluator.
    board = chess.Board()
    board.set_epd(load_epd(MATE_IN_1)[0][0])
    tactical = search.run(board, None, 48)
    assert tactical.evaluations < tactical.sims_run + 1
    assert tactical.sims_run == 48


def test_gate_gumbel_mate_in_1_at_48_sims_stub() -> None:
    """GATE: Gumbel at 48 sims still solves mate-in-1 100% with the stub."""
    solved, total, failed = run_suite(
        MATE_IN_1, sims=48, evaluate_fn=uniform_stub_evaluator,
        cfg=_gumbel_cfg(), mate_in=1, rng=np.random.default_rng(0),
    )
    assert solved == total == 100, f"{solved}/{total}; failed {failed[:8]}"


def test_proven_wins_are_never_dropped_from_the_considered_set() -> None:
    """Top-m by g+logits is a random subset under uniform priors.

    Without force-include the mate-in-1 suite scores 46/100 at 48 sims, because
    the mating move simply is not among the 16 considered actions on a median-43
    move position. An exactly-known win must not be discarded for noise.
    """
    board = chess.Board()
    board.set_epd(load_epd(MATE_IN_1)[0][0])
    search = Search(uniform_stub_evaluator, _gumbel_cfg(), np.random.default_rng(0))
    result = search.run(board, None, 48)
    board.push(result.best_move())
    assert board.is_checkmate()
    board.pop()


def test_improved_policy_is_a_distribution_over_legal_actions() -> None:
    search = Search(uniform_stub_evaluator, _gumbel_cfg(), np.random.default_rng(0))
    for board in _random_boards(15, seed=42):
        result = search.run(board, None, 48)
        if len(result.visits) == 0:
            continue
        probs = result.improved_policy_probs
        assert probs is not None
        assert probs.shape == result.visits.shape
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert (probs >= 0).all()

        legal = {(m.from_square * 64 + m.to_square) for m in board.legal_moves}
        sparse = result.improved_policy()
        assert len(sparse) == len(probs)
        for move_ft, _promo, p in sparse:
            assert move_ft in legal, "improved_policy must be zero off the legal set"
            assert p >= 0


def test_improved_policy_concentrates_on_a_proven_mate() -> None:
    """σ(completed_q) must carry the *exact* terminal value (review question 3).

    With the estimate instead of the truth the target stays diffuse, and the
    network would be taught that a proven win is worth about as much as a quiet
    move — a degradation that never raises, because the softmax still looks
    perfectly well-formed.
    """
    board = chess.Board()
    board.set_epd(load_epd(MATE_IN_1)[0][0])
    search = Search(uniform_stub_evaluator, _gumbel_cfg(), np.random.default_rng(0))
    result = search.run(board, None, 48)

    mating = [
        i for i, mv in enumerate(result.moves()) if _delivers_mate(board, mv)
    ]
    assert mating
    mass = float(result.improved_policy_probs[mating].sum())
    assert mass > 0.9, f"improved policy put only {mass:.3f} on the mate"


# ---------------------------------------------------------------------------
# 8. Evaluation profile (BLOCKED-2026-08-08)
#
# Contract: root_noise=False draws no Gumbel at all, so the considered set is
# the deterministic top-m by raw logits and the acting argmax is
# logits + σ(q̂). Sequential Halving, σ, the solver and the budget convention
# are unchanged — the budget half of that claim is gated above, parametrized
# over both profiles.
# ---------------------------------------------------------------------------


def test_eval_profile_is_seed_independent_position_by_position() -> None:
    """GATE (a): no Gumbel draw ⇒ two seeds agree exactly, not just on average.

    Asserted per position over the full root statistics rather than on a suite
    total, because two different searches can reach the same score by solving
    different positions — which would pass a total-only check while leaving the
    number seed-dependent, i.e. exactly the property this profile exists to
    remove.
    """
    a = Search(uniform_stub_evaluator, _eval_cfg(), np.random.default_rng(1))
    b = Search(uniform_stub_evaluator, _eval_cfg(), np.random.default_rng(2))
    checked = 0
    for board_a, board_b in zip(
        _random_boards(20, seed=71), _random_boards(20, seed=71), strict=True
    ):
        ra = a.run(board_a, None, 48)
        rb = b.run(board_b, None, 48)
        if len(ra.visits) == 0:
            continue
        assert ra.chosen == rb.chosen
        np.testing.assert_array_equal(ra.visits, rb.visits)
        np.testing.assert_array_equal(ra.improved_policy_probs, rb.improved_policy_probs)
        checked += 1
    assert checked >= 15, f"only {checked} non-terminal positions exercised"


def test_eval_profile_suite_result_is_identical_across_seeds() -> None:
    """GATE (a), at the level the gate is actually read off: the suite runner."""
    runs = [
        run_suite(
            MATE_IN_1, sims=48, evaluate_fn=uniform_stub_evaluator,
            cfg=_eval_cfg(), mate_in=1, rng=np.random.default_rng(seed),
        )
        for seed in (1, 2)
    ]
    assert runs[0] == runs[1], "eval profile must not depend on the seed"


def test_selfplay_profile_still_draws_noise() -> None:
    """GATE (b), the other side: the default profile is still stochastic.

    Guards against "made it deterministic everywhere", which would pass every
    determinism test above and quietly destroy self-play data diversity — the
    one thing the Gumbel draw is actually for.
    """
    seen = set()
    for seed in range(6):
        search = Search(uniform_stub_evaluator, _gumbel_cfg(), np.random.default_rng(seed))
        board = chess.Board(MIDGAME_FEN)
        seen.add(tuple(search.run(board, None, 48).visits.tolist()))
    assert len(seen) > 1, "self-play profile produced identical trees across 6 seeds"


def test_eval_profile_considers_the_prior_top_m_exactly() -> None:
    """The considered set is top-m by raw logits — the property the gate needs.

    Under noise this set is a *perturbed* top-m: on the mate-in-2 suite a move
    the prior ranks first is knocked out often enough to cost 6.5 points of
    consideration. Pinned here on a skewed synthetic prior, so the claim is
    tested rather than asserted in a docstring.
    """
    board = chess.Board(MIDGAME_FEN)
    n_legal = board.legal_moves.count()
    ranked = sorted(board.legal_moves, key=lambda mv: (mv.from_square, mv.to_square))

    def skewed(tokens: np.ndarray) -> tuple[np.ndarray, float]:
        # Geometric prior over a fixed move order: unambiguous, well-separated
        # ranking, no ties for the considered-set boundary to arbitrate.
        policy = np.zeros((64, 64), dtype=np.float32)
        for i, mv in enumerate(ranked):
            policy[mv.from_square, mv.to_square] = 0.5 ** (i + 1)
        return policy / policy.sum(), 0.0

    m = 8
    search = Search(uniform_stub_evaluator, _eval_cfg(eval_m=m), np.random.default_rng(0))
    search.evaluate_fn = skewed
    result = search.run(board, None, 64)

    assert n_legal > m, "position must have more legal moves than m for this to bite"
    visited = {i for i, n in enumerate(result.visits) if n > 0}
    top_m = set(np.argsort(-result.prior, kind="stable")[:m].tolist())
    assert visited <= top_m, f"visited outside the prior top-{m}: {visited - top_m}"


def test_eval_overrides_are_scoped_to_the_eval_profile() -> None:
    """``eval_*`` must not touch self-play constants — the firewall, in code.

    Tuning search constants against a gate suite is defensible for a
    measurement and is *not* defensible for the data generator. Keeping that
    separation in the config rather than in a convention is the point.
    """
    cfg = MCTSConfig(eval_m=32, eval_c_visit=25.0, eval_c_scale=2.0)

    selfplay = cfg.resolved()  # root_noise defaults True
    assert (selfplay.gumbel_m, selfplay.c_visit, selfplay.c_scale) == (16, 50.0, 1.0)

    ev = replace(cfg, root_noise=False).resolved()
    assert (ev.gumbel_m, ev.c_visit, ev.c_scale) == (32, 25.0, 2.0)
    assert ev.resolved() == ev, "resolved() must be idempotent"


def test_eval_overrides_default_to_the_selfplay_values() -> None:
    """Unset ``eval_*`` inherits, so the profile alone changes only the noise."""
    ev = MCTSConfig(root_noise=False).resolved()
    base = MCTSConfig()
    assert (ev.gumbel_m, ev.c_visit, ev.c_scale) == (
        base.gumbel_m,
        base.c_visit,
        base.c_scale,
    )


def test_gate_eval_profile_mate_in_1_at_48_sims_stub() -> None:
    """GATE (b): the chunk 5/6 mate-in-1 gate survives the profile change.

    100/100 with a uniform stub is the result that proved mate-in-1 is search's
    job and no network's. If the eval profile broke it, force-include or the
    solver would be interacting with the g ≡ 0 substitution.
    """
    solved, total, failed = run_suite(
        MATE_IN_1, sims=48, evaluate_fn=uniform_stub_evaluator,
        cfg=_eval_cfg(), mate_in=1, rng=np.random.default_rng(0),
    )
    assert solved == total == 100, f"{solved}/{total}; failed {failed[:8]}"


def test_root_completed_q_prefers_exact_over_estimate() -> None:
    """Review question 3, at the unit level."""
    board = chess.Board()
    board.set_epd(load_epd(MATE_IN_1)[0][0])
    search = Search(uniform_stub_evaluator, _gumbel_cfg(), np.random.default_rng(0))
    tree, root, _ = search.run_tree(board, None, 0)
    q = search._root_completed_q(tree, root)

    sl = tree.edge_slice(root)
    terminal_wins = np.where(tree.terminal[sl] & (-tree.terminal_value[sl] >= 1.0))[0]
    assert len(terminal_wins) > 0
    assert np.allclose(q[terminal_wins], 1.0), (
        f"a mate must read exactly +1 before any simulation has run, got {q[terminal_wins]}"
    )


def test_puct_ablation_mode_still_runs_and_gates() -> None:
    """PUCT + Dirichlet is retained as the control condition (spec §6.2)."""
    cfg = MCTSConfig()
    cfg.root_mode = "puct"
    solved, total, _ = run_suite(
        MATE_IN_1, sims=16, evaluate_fn=uniform_stub_evaluator,
        cfg=cfg, mate_in=1, rng=np.random.default_rng(0),
    )
    assert solved == total == 100, f"PUCT control regressed: {solved}/{total}"

    result = Search(uniform_stub_evaluator, cfg, np.random.default_rng(0)).run(
        chess.Board(MIDGAME_FEN), None, 32
    )
    assert result.improved_policy_probs is None, "improved_policy is gumbel-only"
    assert int(result.visits.sum()) == 32


def test_unknown_root_mode_raises() -> None:
    cfg = MCTSConfig()
    cfg.root_mode = "greedy"
    with pytest.raises(ValueError, match="unknown root_mode"):
        Search(uniform_stub_evaluator, cfg).run(chess.Board(), None, 4)


def test_solver_marks_a_forced_mate_proven() -> None:
    """A refuted reply set makes the parent edge proven, not merely well-scored."""
    board = chess.Board()
    board.set_epd(load_epd(MATE_IN_2)[0][0])
    search = Search(uniform_stub_evaluator, _gumbel_cfg(), np.random.default_rng(0))
    tree, root, _ = search.run_tree(board, None, 256)
    sl = tree.edge_slice(root)
    assert tree.proven[sl].any() or tree.terminal[sl].any(), (
        "256 sims on a mate-in-2 should prove at least one edge"
    )
