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
from tinyalphazero.mcts.search import RepetitionsFn, Search
from tinyalphazero.mcts.tree import Tree


@dataclass
class TracePosition:
    """One decision point in a game (spec §7 value targets fill `outcome` later)."""

    tokens: np.ndarray  # (72,) int8
    improved_policy: list[tuple[int, int, float]]  # sparse (move_ft, promo, prob)
    side_to_move: bool  # chess.WHITE / chess.BLACK
    ply: int
    root_q: float  # best root value estimate — chunk 8 reads this to decide resignation


@dataclass
class GameTrace:
    positions: list[TracePosition] = field(default_factory=list)
    result: str = "unfinished"  # "checkmate" | "draw" | "ply_cap" | "unfinished"
    outcome: int | None = None  # filled by the caller (chunk 8)
    plies: int = 0


@dataclass
class _LeafRequest:
    game: int
    tokens: np.ndarray
    legal_mask: np.ndarray  # (64, 64) bool — computed CPU-side when the leaf yields


class _GameState:
    """One live game: board, tree, and the in-flight simulation coroutine."""

    def __init__(self, board: chess.Board, seed: int, cfg: MCTSConfig) -> None:
        self.board = board
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.trace = GameTrace()
        self.ply = 0
        self.done = False
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
    ) -> None:
        """``evaluate_batch(tokens (B,72)) -> (policy (B,64,64), promo (B,64,64,4), value (B,))``.

        Raw policy logits: the legal mask is applied runner-side, because the mask
        is cheap on CPU exactly when the leaf yields and would otherwise have to be
        shipped to the device and back.
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
        self.debug_trees: dict[tuple[int, int], tuple[Tree, int, int]] = {}

    # -- evaluation plumbing ---------------------------------------------

    def _flush(self, pending: list[_LeafRequest], games: list[_GameState]) -> None:
        """One batched forward; resume every suspended simulation it answers."""
        if not pending:
            return
        tokens = np.stack([r.tokens for r in pending]).astype(np.int64)
        policies, promos, values = self.evaluate_batch(tokens)
        self.stats["forwards"] += 1
        self.stats["leaves"] += len(pending)

        for i, req in enumerate(pending):
            policy = np.asarray(policies[i], dtype=np.float32).reshape(64, 64)
            masked = np.where(req.legal_mask, policy, -np.inf)
            m = masked.max()
            probs = np.zeros((64, 64), dtype=np.float32)
            if np.isfinite(m):
                e = np.exp(masked - m)
                probs = (e / e.sum()).astype(np.float32)
            promo = np.asarray(promos[i], dtype=np.float32).reshape(64, 64, 4)
            self._resume(games[req.game], (probs, promo, float(values[i])))
        pending.clear()

    def _resume(self, gs: _GameState, answer: tuple[np.ndarray, np.ndarray, float]) -> None:
        try:
            gs.gen.send(answer)
        except StopIteration as stop:
            gs.gen = None
            if gs.in_root:
                self._pending_roots.append((gs, stop.value))
            else:
                gs.sims_run += 1
                self.stats["sims"] += 1

    # -- per-move Gumbel scheduling --------------------------------------

    def _start_move(
        self, gs: _GameState, search: Search, num_sims: int, base_reps: RepetitionsFn
    ) -> None:
        """Begin a move: open the root-expansion coroutine (it yields one leaf)."""
        gs.tree = Tree(cap_nodes=num_sims + 1)
        gs.path_reps = Counter()
        gs.path_reps[gs.board._transposition_key()] += 1
        gs.sims_run = 0
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
    ) -> Iterator[GameTrace]:
        num_sims = self.cfg.sims if num_sims is None else num_sims
        seeds = seeds or list(range(num_games))
        search = Search(_unused_evaluator, self.cfg)
        base_reps: RepetitionsFn = repetitions_fn or (lambda _b: 0)

        self._games_ref = None
        games = [
            _GameState(
                (boards[i] if boards else chess.Board()),
                seeds[i % len(seeds)],
                self.cfg,
            )
            for i in range(num_games)
        ]
        for gs in games:
            self._start_move(gs, search, num_sims, base_reps)

        pending: list[_LeafRequest] = []
        self._pending_roots: list[tuple[_GameState, int]] = []
        batch_leaves = max(1, self.cfg.batch_leaves)

        while any(not gs.done for gs in games):
            for gs_done, root_idx in self._pending_roots:
                self._finish_root(gs_done, search, num_sims, root_idx)
            self._pending_roots.clear()

            progressed = False
            for gi, gs in enumerate(games):
                if gs.done or gs.gen is not None:
                    continue  # finished, or suspended awaiting its evaluation

                if gs.in_root:
                    # The root is a leaf like any other; it goes through the buffer.
                    gs.gen = search.expand_root_gen(
                        gs.tree, gs.board, gs.path_reps, base_reps
                    )
                else:
                    forced = self._next_forced_edge(gs, search)
                    if forced is None:
                        self._commit_move(gs, search, num_sims, base_reps)
                        progressed = True
                        continue
                    gs.gen = search.simulate_gen(
                        gs.tree, gs.root, gs.board, gs.path_reps, base_reps,
                        forced_edge=forced, use_vloss=True,
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

                pending.append(_LeafRequest(gi, tokens, legal_mask_ft(gs.board)))
                progressed = True
                if len(pending) >= batch_leaves:
                    self._flush(pending, games)

            if pending and not progressed:
                # Every live game is blocked on evaluation: flush short rather than
                # wait for a full batch that can never arrive.
                self._flush(pending, games)
                continue  # the flush unblocked games -- that IS progress
            if not progressed and not pending and not self._pending_roots:
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
            )
        )

        self.debug_trees[(id(gs), gs.ply)] = (tree, root, gs.sims_run)
        gs.board.push(result.best_move())
        gs.ply += 1

        if gs.board.is_game_over() or gs.ply >= self.max_plies:
            gs.done = True
            gs.trace.plies = gs.ply
            gs.trace.result = (
                "checkmate" if gs.board.is_checkmate()
                else "ply_cap" if gs.ply >= self.max_plies
                else "draw"
            )
            return
        self._start_move(gs, search, num_sims, base_reps)


def _unused_evaluator(_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Search's inline evaluator is never called on the batched path.

    Loud rather than silent: if a simulation ever evaluates inline here, the
    batching has been bypassed and throughput numbers would be quietly wrong.
    """
    raise AssertionError(
        "BatchRunner drove an inline evaluation — the batched path was bypassed"
    )


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
