"""Batched leaf evaluation across concurrent games (spec §6.3, chunk 7).

The throughput engine. G games advance concurrently in one thread; each game's
current simulation runs until it reaches an unevaluated leaf, applies virtual
loss along its path, and suspends. When the shared buffer holds ``batch_leaves``
entries — or every live game is blocked — one batched forward answers them all.

**This is a scheduler, not a second search.** Every game drives
``Search.simulate_gen``, the same coroutine the synchronous runner drives; the
only difference is how long an evaluation request waits for its answer. That is
why ``batch_leaves=1, num_games=1`` reproduces chunk 6 exactly, and why the
equivalence gate is a structural property rather than a coincidence two
implementations have to keep re-establishing.

No transposition table, deliberately: per-edge terminality, per-path repetition
counting and the ``path_reps`` invariant all assume tree paths are unique. A TT
turns the tree into a DAG and breaks all three at once — it is a different data
structure, not an optimisation (recorded in PROGRESS.md as a blocker).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import chess
import numpy as np

from tinyalphazero.config import MCTSConfig
from tinyalphazero.encoding import legal_mask_ft
from tinyalphazero.mcts import gumbel
from tinyalphazero.mcts.search import (
    FIFTY_MOVE_PLIES,
    THREEFOLD_COUNT,
    RepetitionsFn,
    Search,
)
from tinyalphazero.mcts.tree import Tree


@dataclass
class TracePosition:
    """One decision point in a game (spec §7 value targets fill `outcome` later)."""

    tokens: np.ndarray  # (72,) int8
    improved_policy: list[tuple[int, int, float]]  # sparse (move_ft, promo, prob)
    side_to_move: bool  # chess.WHITE / chess.BLACK
    ply: int
    root_q: float  # best root value estimate — chunk 8 reads this to decide resignation
    outcome: int = 0  # +1/0/-1 from THIS position's mover's view; filled after the game
    # Entropy (nats) of the network's *prior* over legal root moves, before search.
    # Distinct from the entropy of `improved_policy` and not a substitute for it:
    # the improved policy is softmax(logits + σ(q̂)) with σ ≈ (c_visit + max_N), so
    # once the value head becomes informative its entropy collapses by design.
    # Spec §8's collapse detector is a question about the *policy* — has the
    # network stopped considering alternatives — and only the prior answers it.
    root_prior_entropy: float = 0.0


@dataclass
class GameTrace:
    positions: list[TracePosition] = field(default_factory=list)
    # "checkmate" | "draw" | "draw_claim" | "ply_cap" | "unfinished"
    result: str = "unfinished"
    outcome: int | None = None  # filled by the caller (chunk 8)
    plies: int = 0
    # Chunk 8 self-play bookkeeping, filled by the actor after scoring.
    resign_disabled: bool = False
    would_resign_ply: int | None = None


@dataclass
class _LeafRequest:
    game: int
    tokens: np.ndarray
    legal_mask: np.ndarray  # (64, 64) bool — computed CPU-side when the leaf yields
    # Which network answers this leaf. None (the default) means the runner's single
    # evaluator, and the buffer stays one undivided batch; chunk 9's league sets it
    # per game+colour so latest-vs-pool games can share one concurrent pool.
    evaluator: Any = None


class _GameState:
    """One live game: board, tree, and the in-flight simulation coroutine."""

    def __init__(self, board: chess.Board, seed: int, cfg: MCTSConfig, index: int = 0) -> None:
        self.board = board
        self.index = index  # position in the runner's game list; what move_fn is keyed on
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.trace = GameTrace()
        self.ply = 0
        self.done = False
        # Side to move at the *root* of the move currently being searched. The
        # board is pushed to the leaf when a simulation yields, so `board.turn`
        # at request time is the leaf's mover, not the searcher's — and it is the
        # searcher whose weights answer the leaf.
        self.root_turn: bool = board.turn
        # Positions this game has already LEFT, by transposition key. Incremented
        # once per committed move -- never per simulation -- so the hot path
        # never rescans a move stack. Counts occurrences strictly BEFORE the
        # current position, which is exactly what `base_reps` is defined to
        # return and what `path_reps` (which includes the present position) is
        # built to compose with.
        self.history: Counter = Counter()
        # Per-move search state, (re)built by start_move().
        self.tree: Tree | None = None
        self.root = -1
        self.in_root = False
        self.path_reps: Counter = Counter()
        self.gen: Any = None
        self.schedule: list[int] = []  # remaining Sequential Halving phase budgets
        self.phase_queue: list[int] = []  # survivor-local indices left in this phase
        self.phase_ran = False  # did the phase just drained actually spend visits?
        self.sims_run = 0
        # Evaluator answers this move's search has consumed, root included. The
        # shared Search never evaluates on the batched path, so its own counter
        # would report 0 for every RootResult built here (review A6).
        self.evaluations = 0
        self.g: np.ndarray | None = None
        self.logits: np.ndarray | None = None
        self.survivors: list[int] = []


class BatchRunner:
    """Play games concurrently, evaluating leaves in batches."""

    def __init__(
        self,
        evaluate_batch: Any,
        cfg: MCTSConfig | None = None,
        device: str = "cpu",
        max_plies: int = 160,
        keep_debug_trees: bool = False,
    ) -> None:
        """``evaluate_batch(tokens (B,72)) -> (policy (B,64,64), promo (B,64,64,4), value (B,))``.

        Raw policy logits: the legal mask is applied runner-side, because the mask
        is cheap on CPU exactly when the leaf yields and would otherwise have to be
        shipped to the device and back.

        ``keep_debug_trees`` retains every searched ``(tree, root, sims_run)`` in
        ``debug_trees`` for the equivalence gate and the parity probe. Off by
        default: a ``Tree(cap_nodes=49)`` is ~63 KiB, so a 256-game self-play
        wave at chunk-8 ply counts held ~1.9 GiB of trees for the whole
        ``play_games`` call, on the box whose scarce resource is host RAM
        (review A4/C7).
        """
        self.evaluate_batch = evaluate_batch
        # Resolve the evaluation profile here for the same reason Search does:
        # this class reimplements the Gumbel root, so it is a second consumer of
        # gumbel_m / c_visit / c_scale and would otherwise read the self-play
        # constants while the synchronous path read the eval ones.
        self.cfg = (cfg if cfg is not None else MCTSConfig()).resolved()
        self.device = device
        self.max_plies = max_plies
        # Counters the invariant fuzz reads; see Q4 in PROGRESS.md.
        self.stats = {"forwards": 0, "leaves": 0, "sims": 0, "terminal_shortcircuits": 0}
        # (game_index, ply) -> (tree, root, sims_run) for the equivalence gate. The
        # gate has to compare visit counts and solver flags, not just chosen moves:
        # two searches can agree on the move while disagreeing about everything else.
        # Filled only when keep_debug_trees is set; empty otherwise.
        self.keep_debug_trees = keep_debug_trees
        self.debug_trees: dict[tuple[int, int], tuple[Tree, int, int]] = {}
        self._stop_fn: Any = None
        self._evaluator_for: Any = None
        self._move_fn: Any = None
        # Per-call search context, stashed so ``_resume`` can finish a root the
        # moment its evaluation lands (see ``_resume`` for why it must).
        self._search: Search | None = None
        self._num_sims: int = 0

    # -- evaluation plumbing ---------------------------------------------

    def _flush(self, pending: list[_LeafRequest], games: list[_GameState]) -> None:
        """Resume every suspended simulation, one batched forward per network.

        With a single evaluator (the default, and everything before chunk 9) this
        is exactly one forward over the whole buffer — the equivalence gate's
        ``batch_leaves=1`` path is untouched. Under league play the buffer holds
        leaves from games searched by different snapshots; grouping is what lets
        all 512 games stay in *one* concurrent pool instead of being generated in
        per-opponent waves, which is the difference between a wide forward and
        twenty latency-bound narrow ones.
        """
        if not pending:
            return
        if self._evaluator_for is None:
            self._forward(self.evaluate_batch, pending, games)
        else:
            groups: dict[Any, list[_LeafRequest]] = {}
            for req in pending:
                groups.setdefault(req.evaluator, []).append(req)
            for fn, reqs in groups.items():
                self._forward(fn, reqs, games)
        pending.clear()

    def _forward(
        self, evaluate_batch: Any, reqs: list[_LeafRequest], games: list[_GameState]
    ) -> None:
        tokens = np.stack([r.tokens for r in reqs]).astype(np.int64)
        policies, promos, values = evaluate_batch(tokens)
        self.stats["forwards"] += 1
        self.stats["leaves"] += len(reqs)

        for i, req in enumerate(reqs):
            policy = np.asarray(policies[i], dtype=np.float32).reshape(64, 64)
            masked = np.where(req.legal_mask, policy, -np.inf)
            m = masked.max()
            probs = np.zeros((64, 64), dtype=np.float32)
            if np.isfinite(m):
                e = np.exp(masked - m)
                probs = (e / e.sum()).astype(np.float32)
            promo = np.asarray(promos[i], dtype=np.float32).reshape(64, 64, 4)
            self._resume(games[req.game], (probs, promo, float(values[i])))

    def _resume(self, gs: _GameState, answer: tuple[np.ndarray, np.ndarray, float]) -> None:
        """Hand one evaluation back to its suspended simulation.

        A root that completes here is finished *here*, synchronously. It used to
        be queued and finished at the top of the next ``while`` pass, which left
        a window — ``gen is None`` and ``in_root`` still True — that the
        per-game ``for`` loop could reach first whenever a flush fired at a
        lower game index than the root's owner (only possible with
        ``batch_leaves <= concurrent games``). The loop read that state as "root
        not yet requested" and opened a **second** ``expand_root_gen`` on the
        same tree: Gumbel noise drawn twice, ``root`` re-pointed to node 1 with
        node 0 orphaned, or the second build counted as a simulation (review
        A1). Finishing inline leaves no state in which a completed root is
        ambiguous, so the scheduler cannot change a game's search — which is
        the whole contract of this class.
        """
        gs.evaluations += 1  # one answer, one evaluation of this game's leaf
        try:
            gs.gen.send(answer)
        except StopIteration as stop:
            gs.gen = None
            if gs.in_root:
                self._finish_root(gs, self._search, self._num_sims, stop.value)
            else:
                gs.sims_run += 1
                self.stats["sims"] += 1

    # -- per-move Gumbel scheduling --------------------------------------

    def _start_move(
        self, gs: _GameState, search: Search, num_sims: int, base_reps: RepetitionsFn
    ) -> None:
        """Begin a move: open the root-expansion coroutine (it yields one leaf)."""
        gs.tree = Tree(cap_nodes=num_sims + 1)
        gs.root_turn = gs.board.turn
        gs.path_reps = Counter()
        gs.path_reps[gs.board._transposition_key()] += 1
        gs.sims_run = 0
        gs.evaluations = 0
        gs.schedule = []
        gs.phase_queue = []
        gs.phase_ran = False
        gs.survivors = []
        gs.root = -1
        gs.in_root = True
        gs.gen = None  # the loop creates and primes it, so all leaves enter one buffer

    def _finish_root(self, gs: _GameState, search: Search, num_sims: int, root: int) -> None:
        """Root node built: derive Gumbel noise and the halving schedule."""
        gs.root = root
        gs.in_root = False
        n_actions = int(gs.tree.num_edges[gs.root])
        if n_actions == 0:
            return

        # Shared with Search.run_gumbel (review S1) — one definition of the
        # considered set, consumed by both drivers, neither re-deriving it.
        gs.g, gs.logits, considered = gumbel.form_considered_set(
            gs.tree, gs.root, self.cfg, gs.rng, num_sims
        )
        # Phase budgets up front; the per-phase allocation is computed at phase
        # entry, because re-ranking needs stats concurrent games have not
        # produced yet at scheduling time.
        gs.survivors = list(considered)
        gs.schedule = list(gumbel.halving_phase_budgets(num_sims, len(considered)))
        gs.phase_queue = []
        gs.phase_ran = False

    def _next_forced_edge(self, gs: _GameState, search: Search) -> int | None:
        """The root edge this simulation must spend itself on, or None if done.

        Mirrors ``Search.run_gumbel``'s loop exactly, one call at a time:

        * a phase's visits come from ``allocate_within_phase`` — best-ranked
          survivors absorb the remainder — not from a round-robin;
        * survivors halve **after** a phase that actually ran, and a
          zero-budget phase is skipped *without* halving, which is what the
          synchronous ``continue`` does.

        Both were wrong before (review S2). The round-robin gave phase
        remainders to ``survivors[sims_run % k]``, and the old ``while`` loop
        halved on zero-budget phases the synchronous driver skips over.
        """
        if gs.tree is None or not gs.survivors:
            return None
        sl = gs.tree.edge_slice(gs.root)

        while not gs.phase_queue:
            if gs.phase_ran and len(gs.survivors) > 1:
                scores = search._gumbel_scores(gs.tree, gs.root, gs.g, gs.logits)
                gs.survivors = gumbel.halve_survivors(gs.survivors, scores)
            gs.phase_ran = False
            if not gs.schedule:
                return None
            budget = gs.schedule.pop(0)
            if budget <= 0 or not gs.survivors:
                continue
            gs.phase_queue = gumbel.phase_visit_queue(budget, len(gs.survivors))
            gs.phase_ran = True

        return sl.start + int(gs.survivors[gs.phase_queue.pop(0)])

    # -- the loop ---------------------------------------------------------

    def play_games(
        self,
        num_games: int,
        seeds: list[int] | None = None,
        num_sims: int | None = None,
        boards: list[chess.Board] | None = None,
        repetitions_fn: RepetitionsFn | None = None,
        stop_fn: Any = None,
        evaluator_for: Any = None,
        move_fn: Any = None,
    ) -> Iterator[GameTrace]:
        """``stop_fn(trace) -> str | None`` is consulted after every committed
        move; a returned string ends that game with it as the result.

        This is how chunk 8 implements resignation without the runner knowing
        what resignation is. The runner's job is to expose ``root_q`` per
        decision (spec §7 says exactly that); deciding when a position is lost
        enough to abandon is the actor's policy, and keeping it out here is what
        lets the actor disable it for 10% of games without a second code path.

        ``evaluator_for(game_index, root_side_to_move) -> evaluate_batch`` is the
        same idea for the league (chunk 9): the runner does not know what a
        snapshot pool is, only that a leaf belongs to whoever is searching. None —
        the default — keeps a single evaluator and a single forward per flush.

        ``move_fn(game_index, board) -> chess.Move | None`` is the third (chunk
        10): returning a move means *this side does not search*, so the move is
        played directly and no tree, trace row, or simulation budget is spent on
        it. That is how the ladder plays a random mover, a 1-ply greedy or an
        external engine while the learner still gets full batched concurrency —
        rather than growing a second scheduler beside this one. A network
        opponent is not a ``move_fn``: it searches, so it belongs in
        ``evaluator_for``.
        """
        num_sims = self.cfg.sims if num_sims is None else num_sims
        seeds = seeds or list(range(num_games))
        search = Search(_unused_evaluator, self.cfg)

        # One base_reps per game, reading that game's own history. An override
        # is honoured for tests that want to inject a synthetic history.
        def game_reps(gs: _GameState) -> RepetitionsFn:
            if repetitions_fn is not None:
                return repetitions_fn
            return lambda b: gs.history[b._transposition_key()]

        self._games_ref = None
        games = [
            _GameState(
                (boards[i] if boards else chess.Board()),
                seeds[i % len(seeds)],
                self.cfg,
                index=i,
            )
            for i in range(num_games)
        ]
        # Exposed for tests that assert on per-game history/adjudication state;
        # the equivalence gate already reads debug_trees for the same reason.
        self._games_debug = games
        self._stop_fn = stop_fn
        self._evaluator_for = evaluator_for
        self._move_fn = move_fn
        self._search = search
        self._num_sims = num_sims
        for gs in games:
            # _begin_move, not _start_move: a game whose opening has the opponent
            # to move must play that move before any tree is built for it.
            self._begin_move(gs, search, num_sims, game_reps(gs))

        pending: list[_LeafRequest] = []
        batch_leaves = max(1, self.cfg.batch_leaves)

        while any(not gs.done for gs in games):
            progressed = False
            for gi, gs in enumerate(games):
                if gs.done or gs.gen is not None:
                    continue  # finished, or suspended awaiting its evaluation

                if gs.in_root:
                    # The root is a leaf like any other; it goes through the buffer.
                    # ``in_root`` with no generator means exactly one thing here —
                    # "not yet requested" — because ``_resume`` clears it the
                    # moment the root's evaluation lands (review A1).
                    gs.gen = search.expand_root_gen(gs.tree, gs.board, gs.path_reps, game_reps(gs))
                else:
                    forced = self._next_forced_edge(gs, search)
                    if forced is None:
                        self._commit_move(gs, search, num_sims, game_reps(gs))
                        progressed = True
                        continue
                    gs.gen = search.simulate_gen(
                        gs.tree,
                        gs.root,
                        gs.board,
                        gs.path_reps,
                        game_reps(gs),
                        forced_edge=forced,
                        use_vloss=True,
                    )

                try:
                    tokens = next(gs.gen)
                except StopIteration:
                    # Terminal short-circuit: never reached the evaluator, so it must
                    # NOT enter the buffer. Sims advance, leaves do not -- this is the
                    # accounting review Q4 requires not to desync.
                    gs.gen = None
                    gs.sims_run += 1
                    self.stats["sims"] += 1
                    self.stats["terminal_shortcircuits"] += 1
                    progressed = True
                    continue

                pending.append(
                    _LeafRequest(
                        gi,
                        tokens,
                        legal_mask_ft(gs.board),
                        None if evaluator_for is None else evaluator_for(gi, gs.root_turn),
                    )
                )
                progressed = True
                if len(pending) >= batch_leaves:
                    self._flush(pending, games)

            if pending and not progressed:
                # Every live game is blocked on evaluation: flush short rather than
                # wait for a full batch that can never arrive.
                self._flush(pending, games)
                continue  # the flush unblocked games -- that IS progress
            if not progressed and not pending:
                break

        self._flush(pending, games)
        # Re-key debug trees by game index now that the identity map is known.
        by_id = {id(gs): i for i, gs in enumerate(games)}
        self.debug_trees = {
            (by_id[k[0]], k[1]): v for k, v in self.debug_trees.items() if k[0] in by_id
        }
        yield from (gs.trace for gs in games)

    def _commit_move(
        self, gs: _GameState, search: Search, num_sims: int, base_reps: RepetitionsFn
    ) -> None:
        """Record the trace row, play the acting move, and set up the next search."""
        tree, root = gs.tree, gs.root
        if tree is None or int(tree.num_edges[root]) == 0:
            gs.done = True
            gs.trace.result = "checkmate" if gs.board.is_checkmate() else "draw"
            gs.trace.plies = gs.ply
            return

        sl = tree.edge_slice(root)
        result = search._root_result(tree, root, sl, gs.sims_run)
        # _root_result reports the Search's own counter, which never moves on
        # this path; the game's count is the real one (review A6).
        result.evaluations = gs.evaluations
        completed_q = search._root_completed_q(tree, root)
        max_n = int(tree.N[sl].max())
        result.improved_policy_probs = gumbel.improved_policy(
            gs.logits, completed_q, max_n, self.cfg
        )
        scores = search._gumbel_scores(tree, root, gs.g, gs.logits)
        result.chosen = int(max(gs.survivors, key=lambda a: scores[a]))

        gs.trace.positions.append(
            TracePosition(
                # Review S3: the real base_reps, not a zero stub. The trace row
                # is training data, and the repetition token is part of the
                # Markov state (spec §3.1) — encoding every position as
                # first-occurrence teaches the network that a position it has
                # already reached twice is fresh, which is exactly the
                # non-Markovian defect v1 was rebuilt to remove. Harmless today
                # only because chunk 7 passes no history; chunk 8 supplies one.
                tokens=search.leaf_tokens(gs.board, gs.path_reps, base_reps),
                improved_policy=result.improved_policy(),
                side_to_move=gs.board.turn,
                ply=gs.ply,
                root_q=float(completed_q.max()),
                root_prior_entropy=gumbel.entropy(tree.prior[sl]),
            )
        )

        if self.keep_debug_trees:
            self.debug_trees[(id(gs), gs.ply)] = (tree, root, gs.sims_run)
        if self._play(gs, result.best_move()):
            return
        self._begin_move(gs, search, num_sims, base_reps)

    def _play(self, gs: _GameState, move: chess.Move) -> bool:
        """Push ``move``, update history, adjudicate. Returns True if the game ended.

        The single place a move is played, whether the search chose it or an
        external opponent did (``move_fn``). Adjudication that lived in only one
        of two paths would be two rulebooks: a ladder match that ended by a
        different rule than a self-play game would make Elo and training data
        disagree about what a draw is.
        """
        # The position we are about to leave joins the history. One Counter
        # increment per committed move; nothing rescans the move stack.
        gs.history[gs.board._transposition_key()] += 1
        gs.board.push(move)
        gs.ply += 1
        stop = self._stop_fn(gs.trace) if self._stop_fn is not None else None
        return self._adjudicate(gs, stop)

    def _adjudicate(self, gs: _GameState, stop: str | None = None) -> bool:
        """Is the position ``gs`` stands on over? If so mark the game done and say why.

        One rulebook, consulted after every push by ``_play`` and once on the
        opening by ``_begin_move``. The opening case exists because a terminal
        opening has nothing to push: it used to fall straight through to
        ``move_fn``, and ``RandomMover`` then drew from zero legal moves and
        raised (review D15). ``stop`` is the actor's verdict on the trace; the
        opening passes None because no decision has been made yet.

        Claim draws at game level with exactly the in-search semantics: a
        rational player claims, so the actor must treat these as over rather
        than playing on into a position the search already scores 0. Using the
        same THREEFOLD_COUNT / FIFTY_MOVE_PLIES constants keeps the two in
        step by construction — a game that ended one way inside the tree and
        another way at the actor would put contradictory value targets into
        the replay buffer.
        """
        occurrences = gs.history[gs.board._transposition_key()] + 1
        claim_draw = occurrences >= THREEFOLD_COUNT or gs.board.halfmove_clock >= FIFTY_MOVE_PLIES

        if stop or gs.board.is_game_over() or claim_draw or gs.ply >= self.max_plies:
            gs.done = True
            gs.trace.plies = gs.ply
            gs.trace.result = (
                stop
                if stop
                else "checkmate"
                if gs.board.is_checkmate()
                else "draw_claim"
                if claim_draw
                else "ply_cap"
                if gs.ply >= self.max_plies
                else "draw"
            )
            return True
        return False

    def _begin_move(
        self, gs: _GameState, search: Search, num_sims: int, base_reps: RepetitionsFn
    ) -> None:
        """Set up the next searched move, playing any external moves that precede it.

        ``move_fn`` returning a move means "this side is not the searcher" — a
        ladder opponent that is a random mover, a 1-ply greedy, or an engine. The
        loop is a ``while`` because both sides can be external in principle, and
        because a game must never be left sitting on a position nobody will move.

        The opening is adjudicated before anyone is asked for a move. Every
        later position has already been through ``_play``'s adjudication, so
        the check is made once, at ply 0 (review D15).
        """
        if gs.ply == 0 and self._adjudicate(gs):
            return  # a terminal opening: no side has a move to make
        while not gs.done:
            move = None if self._move_fn is None else self._move_fn(gs.index, gs.board)
            if move is None:
                self._start_move(gs, search, num_sims, base_reps)
                return
            if self._play(gs, move):
                return


def _unused_evaluator(_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Search's inline evaluator is never called on the batched path.

    Loud rather than silent: if a simulation ever evaluates inline here, the
    batching has been bypassed and throughput numbers would be quietly wrong.
    """
    raise AssertionError("BatchRunner drove an inline evaluation — the batched path was bypassed")


def uniform_stub_batch(
    tokens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched analogue of ``uniform_stub_evaluator``: flat logits, zero value.

    Flat logits become a uniform distribution once the runner applies the legal
    mask, and the 0.25 promo plane divides a promotion cell exactly as the old
    split-by-count did — so the two evaluators agree by construction and any
    sync-vs-batched divergence is the scheduler's, not the evaluator's.

    Lives here, once. It previously existed as three separate copies (the tests,
    the benchmark, the parity probe) that had to be kept in step by hand, which
    is the duplication rule in AGENTS.md §5 and the thing that made the
    evaluator signature change touch three files it should not have.
    """
    b = tokens.shape[0]
    return (
        np.zeros((b, 64, 64), dtype=np.float32),
        np.full((b, 64, 64, 4), 0.25, dtype=np.float32),
        np.zeros(b, dtype=np.float32),
    )


def torch_evaluate_batch(net: Any, device: Any) -> Any:
    """Adapt a ChessNet into an ``evaluate_batch`` callable."""
    import torch

    from tinyalphazero.model import value_scalar

    @torch.inference_mode()
    def evaluate(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = torch.from_numpy(np.ascontiguousarray(tokens)).to(device)
        use_bf16 = getattr(device, "type", str(device)) == "cuda"
        ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_bf16
            else __import__("contextlib").nullcontext()
        )
        with ctx:
            out = net(t)
        # Promotions are softmaxed on device: it is a (B,64,64,4) reduction over
        # the last axis, and doing it here rather than per-leaf on the CPU keeps
        # the scatter loop free of per-cell work. The transfer grows by 4x over
        # the policy plane, which the §10 erratum's decomposition says is not
        # where the time goes (GPU ~28% of a simulation; movegen dominates).
        return (
            out.policy_ft.float().cpu().numpy(),
            torch.softmax(out.promo, dim=-1).float().cpu().numpy(),
            value_scalar(out.value_wdl).float().cpu().numpy(),
        )

    return evaluate
