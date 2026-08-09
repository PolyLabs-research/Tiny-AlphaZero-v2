"""Synchronous single-game MCTS with classic PUCT selection (spec §6.1, §6.4).

Correctness before speed: no batching, no Gumbel, uniform root. Gumbel arrives
in chunk 6; this chunk exists to make search bugs impossible to confuse with
model quality, which is why every gate runs against a stub evaluator.

Run the mate-in-1 suite:

    python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_1.epd --sims 16 --stub

**Evaluation profile.** Suites, the anchor ladder, and match play run with
``root_noise=False``; self-play runs with noise. Gumbel's root draw is a
data-diversity device, and leaving it on turns a strength measurement into a
measurement of exploration noise — on the mate-in-2 suite it costs 6.5 points of
consideration and gives the gate a ±2.5-point spread under reseeding
(``BLOCKED-2026-08-08-mate2-gate.md``). This runner therefore defaults to the
eval profile; pass ``--root-noise`` to measure the self-play profile instead.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from tinyalphazero.config import MCTSConfig
from tinyalphazero.encoding import decode_board, encode_board
from tinyalphazero.mcts import gumbel
from tinyalphazero.mcts.tree import Tree

# evaluate_fn(tokens (72,)) -> (policy_ft (64,64) probs already legal-masked,
#                               promo (64,64,4) per-cell promotion distribution,
#                               value float)
#
# The promo plane is read **only** on cells that carry a promotion move, and it
# exists because a (from, to) cell is not one action when it is a promotion — it
# is four. Splitting that cell's prior by count treats a queen promotion and an
# underpromotion as equally likely, which is wrong on essentially every position
# where promotions are legal and is exactly the kind of prior error Gumbel's
# considered set has no way to recover from. Dense (64,64,4) costs two linears
# and a broadcast; the model already computes it in `forward`.
EvaluateFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, float]]
# repetitions_fn(board) -> times this position already occurred in game history
# *before* the search started. In-search repetitions are counted separately.
RepetitionsFn = Callable[[chess.Board], int]

# Draw-claim thresholds (spec §6.3: "50-move and threefold via claim checks").
FIFTY_MOVE_PLIES = 100
THREEFOLD_COUNT = 3

# Lifts every exactly-known win above every estimate, and orders wins by mate
# distance. All proven wins sit at q̂ = +1, so without this the acting argmax
# falls through to `g + logits` — noise under self-play, the raw policy prior
# under the evaluation profile. Either way the search discards what it has
# *proved* in favour of what the network *guesses*.
#
# Scale: logits are O(10) and σ is O(100), so WIN_DEPTH_STEP dominates both and
# one extra ply of mate distance always outweighs any prior preference. The
# bonus stays positive out past 900 plies, well beyond any reachable mate.
EXACT_WIN_BONUS = 1e6
WIN_DEPTH_STEP = 1e3


# --- repetition counting ----------------------------------------------------
#
# Two callers want two *different* counts off the same data, and conflating them
# is a silent off-by-one that no stub-evaluator gate can catch:
#
#   encode_board(board, repetitions)  wants occurrences strictly EARLIER than now
#   threefold claim                   wants TOTAL occurrences, including now
#
# Invariant: ``path_reps`` always includes the position the board is standing on,
# root included. Both helpers below are derived from that one invariant so the
# distinction lives in their names rather than in a caller's arithmetic.


def _total_occurrences(board: chess.Board, path_reps: Counter, base_reps: RepetitionsFn) -> int:
    """Times this position has occurred, counting the present one."""
    return base_reps(board) + path_reps[board._transposition_key()]


def _prior_occurrences(board: chess.Board, path_reps: Counter, base_reps: RepetitionsFn) -> int:
    """Times this position occurred *before* now — what ``encode_board`` wants."""
    return max(0, _total_occurrences(board, path_reps, base_reps) - 1)


@dataclass
class RootResult:
    """Root statistics after a search."""

    move_ft: np.ndarray  # int16 (n_edges,)  from*64+to
    promo: np.ndarray  # int8  (n_edges,)
    visits: np.ndarray  # int32 (n_edges,)
    q: np.ndarray  # float32 (n_edges,) root side-to-move perspective
    prior: np.ndarray  # float32 (n_edges,)
    sims_run: int
    # Gumbel additions (chunk 6). ``chosen`` is the acting move's edge index;
    # under PUCT mode it is the most-visited edge.
    improved_policy_probs: np.ndarray | None = None  # float32 (n_edges,)
    chosen: int = -1
    evaluations: int = 0  # evaluator calls — < sims_run when terminals short-circuit

    def moves(self) -> list[chess.Move]:
        from tinyalphazero.encoding import decode_move

        return [
            decode_move(int(ft) // 64, int(ft) % 64, int(p))
            for ft, p in zip(self.move_ft, self.promo, strict=True)
        ]

    def best_move(self) -> chess.Move:
        """The acting move.

        Under Gumbel this is the argmax of ``g + logits + σ(q̂)`` over survivors,
        which is *not* in general the most-visited action — Sequential Halving
        spends its budget to discriminate, not to vote.
        """
        if len(self.visits) == 0:
            raise ValueError("no root edges — position is terminal")
        if self.chosen >= 0:
            return self.moves()[self.chosen]
        best = max(
            range(len(self.visits)),
            key=lambda i: (int(self.visits[i]), float(self.q[i]), -i),
        )
        return self.moves()[best]

    def improved_policy(self) -> list[tuple[int, int, float]]:
        """Sparse ``(move_ft, promo, prob)`` policy target over all legal actions."""
        if self.improved_policy_probs is None:
            raise ValueError("improved_policy is only produced in gumbel root mode")
        return [
            (int(ft), int(pr), float(p))
            for ft, pr, p in zip(self.move_ft, self.promo, self.improved_policy_probs, strict=True)
        ]


def network_evaluator(ckpt_path: str | Path, device: str | None = None) -> EvaluateFn:
    """Build an ``evaluate_fn`` backed by a trained checkpoint.

    Returns legal-masked policy probabilities, a per-cell promotion
    distribution, and a scalar value in the side-to-move perspective
    (``P(win) - P(loss)``), matching the sign convention the tree backs up.
    """
    import torch

    from tinyalphazero.encoding import legal_mask_ft
    from tinyalphazero.model import masked_policy, value_scalar
    from tinyalphazero.train import load_checkpoint

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net, _meta = load_checkpoint(ckpt_path, device=dev)
    net.eval()

    @torch.no_grad()
    def evaluate(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        board = decode_board(tokens)
        t = torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.int64)).unsqueeze(0).to(dev)
        out = net(t)
        mask = torch.from_numpy(legal_mask_ft(board)).unsqueeze(0).to(dev)
        probs = masked_policy(out.policy_ft, mask)[0].reshape(64, 64).float().cpu().numpy()
        promo = torch.softmax(out.promo[0], dim=-1).float().cpu().numpy()
        value = float(value_scalar(out.value_wdl)[0].item())
        return probs, promo, value

    return evaluate


# Promotion pieces in the order the codec numbers them (promo 1..4), so a promo
# plane index is `_PROMO_OF[piece] - 1`.
UNIFORM_PROMO = np.full((64, 64, 4), 0.25, dtype=np.float32)


def uniform_stub_evaluator(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Uniform policy over legal moves, uniform promotions, zero value.

    Decodes the board from ``tokens`` — legitimate precisely because the
    encoding is Markovian (spec §3.1); if this ever stops working, the Markov
    invariant is broken and that is the bug worth finding.

    The 0.25-everywhere promo plane is chosen so the stub's behaviour is
    **unchanged** by the interface: splitting a promotion cell's mass by a
    uniform distribution over four pieces is arithmetically identical to the
    split-by-count it replaces. Every stub-evaluator gate — which is every gate
    in chunks 5 and 6 — therefore measures the same search it did before.
    """
    board = decode_board(tokens)
    policy = np.zeros((64, 64), dtype=np.float32)
    legal = list(board.legal_moves)
    if legal:
        for mv in legal:
            policy[mv.from_square, mv.to_square] = 1.0
        policy /= policy.sum()
    return policy, UNIFORM_PROMO, 0.0


class Search:
    """Synchronous PUCT search over an array tree."""

    def __init__(
        self,
        evaluate_fn: EvaluateFn,
        cfg: MCTSConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.evaluate_fn = evaluate_fn
        # Resolve the eval profile once, here, so every downstream reader of
        # cfg.gumbel_m / c_visit / c_scale sees the constants this search is
        # actually running with. Mutating ``search.cfg`` afterwards is therefore
        # not a supported way to switch profile — build a new Search.
        self.cfg = (cfg if cfg is not None else MCTSConfig()).resolved()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._evaluations = 0
        # Post-search tree, for tests that must compare more than the chosen move.
        self._last_tree: Tree | None = None
        self._last_root = -1

    # -- root value completion (spec §6.2) --------------------------------

    def _root_completed_q(self, tree: Tree, root: int) -> np.ndarray:
        """Per-root-action q̂ from the root side-to-move perspective.

        Three cases, in priority order:

        1. **Terminal** — the exact value, negated into this node's perspective.
        2. **Visited** — the running average ``W/N``.
        3. **Unvisited** — completed-Q, i.e. the root's own value estimate.

        Case 1 must come first and must be applied *here*, not only in
        ``_select_edge``. Gumbel consumes this vector twice — inside σ(q̂) for
        the acting argmax and inside the improved-policy target — and a mate
        scored by estimate on one path while non-root selection uses the exact
        value on another is an inconsistency that never raises: it just yields a
        policy target that teaches the network a proven win is worth ~0.
        """
        sl = tree.edge_slice(root)
        n = tree.N[sl].astype(np.float32)
        q = np.where(n > 0, tree.W[sl] / np.maximum(n, 1.0), tree.node_q(root))
        has_exact, exact = tree.exact_value(sl)
        if has_exact.any():
            q = np.where(has_exact, -exact, q)
        return q.astype(np.float32)

    # -- terminal handling ----------------------------------------------

    def _terminal_value(self, board: chess.Board, total_reps: int) -> float | None:
        """Exact value from the side-to-move's perspective, or None if not terminal.

        Claim-based draws are treated as draws: a rational player claims. Uses
        cheap explicit checks rather than ``board.can_claim_*``, which are both
        expensive and unnecessary once repetitions are counted along the path.
        """
        if board.is_checkmate():
            return -1.0  # side to move has been mated
        if not any(True for _ in board.generate_legal_moves()):
            return 0.0  # stalemate
        if board.is_insufficient_material():
            return 0.0
        if board.halfmove_clock >= FIFTY_MOVE_PLIES:
            return 0.0
        if total_reps >= THREEFOLD_COUNT:
            return 0.0
        return None

    # -- expansion -------------------------------------------------------

    def _expand(
        self, tree: Tree, board: chess.Board, path_reps: Counter, base_reps: RepetitionsFn
    ) -> tuple[int, float]:
        """Create a node for ``board``; return ``(node_index, value)``.

        ``value`` is the evaluator's estimate from the side-to-move perspective.

        Terminal status of each child is resolved here, with one push/pop per
        legal move, rather than lazily on first traversal. This is deliberate
        and it is what makes the 16-sim mate-in-1 gate meaningful: PUCT with
        uniform priors visits one fresh root edge per simulation, so a lazy
        search on a 30-move position could not see a mate at 16 sims however
        correct it was — the gate would measure branching factor, not search.
        Cost is O(branching) push/pops per expansion; chunk 5 is correctness
        before speed (plan §chunk-5) and chunk 7 can revisit it after profiling.
        """
        tokens = encode_board(board, _prior_occurrences(board, path_reps, base_reps))
        policy, promo_probs, value = self.evaluate_fn(tokens)
        self._evaluations += 1
        return self._build_node(tree, board, path_reps, base_reps, policy, promo_probs, value)

    def expand_root_gen(
        self,
        tree: Tree,
        board: chess.Board,
        path_reps: Counter,
        base_reps: RepetitionsFn,
    ):
        """Root expansion as a coroutine, so it can be batched like any other leaf.

        A root is a leaf awaiting evaluation exactly like the ones inside a
        simulation. Evaluating it inline would mean one unbatched forward per game
        per move -- at 256 concurrent games that is 256 single-row forwards before
        any real batch forms, which is the throughput this chunk exists to fix.
        """
        tokens = self.leaf_tokens(board, path_reps, base_reps)
        policy, promo_probs, value = yield tokens
        node, _ = self._build_node(tree, board, path_reps, base_reps, policy, promo_probs, value)
        return node

    def leaf_tokens(
        self, board: chess.Board, path_reps: Counter, base_reps: RepetitionsFn
    ) -> np.ndarray:
        """Encoding of a leaf awaiting evaluation (batched path)."""
        return encode_board(board, _prior_occurrences(board, path_reps, base_reps))

    def _build_node(
        self,
        tree: Tree,
        board: chess.Board,
        path_reps: Counter,
        base_reps: RepetitionsFn,
        policy: np.ndarray,
        promo_probs: np.ndarray,
        value: float,
    ) -> tuple[int, float]:
        """Create the node given an already-computed evaluation.

        Split out of ``_expand`` so the synchronous and batched runners share one
        implementation: the only difference between them is *where* the policy and
        value came from, and equivalence at batch_leaves=1 is then structural
        rather than something a test has to keep rediscovering.
        """

        edges: list[tuple[int, int, float]] = []
        terminal_flags: list[tuple[bool, float]] = []
        for mv in board.legal_moves:
            promo = _PROMO_OF[mv.promotion]
            ft = mv.from_square * 64 + mv.to_square

            board.push(mv)
            key = board._transposition_key()
            path_reps[key] += 1
            tv = self._terminal_value(board, _total_occurrences(board, path_reps, base_reps))
            path_reps[key] -= 1
            board.pop()

            # A promotion cell holds four distinct actions sharing one (from, to)
            # score, so the cell's mass is divided by the promo head's own
            # distribution rather than equally. Equal division says a rook
            # promotion is as likely as a queen, which is wrong nearly always and
            # occasionally wrong in the other direction — an underpromotion that
            # is the only winning move gets a quarter of the mass it deserves,
            # and Gumbel's considered set has no mechanism to recover a prior
            # error. `promo` is 0 for non-promotions, and index 0 of the plane is
            # never read, so the branch is only taken where it means something.
            cell = float(policy[mv.from_square, mv.to_square])
            if promo and self.cfg.promo_prior_from_head:
                # Phase 2 path: the four promotions *share* the cell's mass,
                # divided by the head's distribution over pieces.
                cell *= float(promo_probs[mv.from_square, mv.to_square, promo - 1])

            edges.append((ft, promo, cell))
            terminal_flags.append((tv is not None, 0.0 if tv is None else tv))

        # Normalize priors over the edges actually created.
        #
        # Note what happens to a promotion cell when `promo_prior_from_head` is
        # off: all four promotion edges take the *full* cell score, so the cell
        # contributes 4x its weight to this sum and each promotion ends with the
        # same prior a normal move of equal score would get. That is the chunk 5
        # behaviour, and its docstring described it as "the four of them split
        # that cell's mass by count" — which is not what the code did, and not
        # what it does now. It is load-bearing: making promotions genuinely share
        # the cell (x0.25 each) drops the chunk 6 gate 0.985 -> 0.965, because on
        # four gate positions the key promotion falls out of Gumbel's top-16.
        # Kept deliberately rather than "fixed", and revisited when Phase 2 turns
        # the flag on and the head has real signal to divide by.
        total = sum(p for _, _, p in edges)
        if total > 0:
            edges = [(ft, pr, p / total) for ft, pr, p in edges]
        elif edges:
            edges = [(ft, pr, 1.0 / len(edges)) for ft, pr, _ in edges]

        node = tree.add_node(edges)
        sl = tree.edge_slice(node)
        tree.terminal[sl] = [t for t, _ in terminal_flags]
        tree.terminal_value[sl] = [v for _, v in terminal_flags]

        # If expansion just revealed that the side to move has a mate available,
        # this node's value is exactly +1 — it is not an estimate. Returning the
        # evaluator's guess instead dilutes the parent's Q with a number we know
        # to be wrong, and costs an extra simulation per line before the win
        # registers. Same principle as _select_edge: prefer truth to estimate.
        for is_term, tv in terminal_flags:
            if is_term and -tv >= 1.0:
                return node, 1.0
        return node, float(value)

    # -- selection -------------------------------------------------------

    def _select_edge(self, tree: Tree, node: int) -> int:
        """PUCT: ``Q + c_puct * prior * sqrt(N_parent) / (1 + N_edge)``.

        Unvisited edges take completed-Q = this node's Q rather than 0.
        """
        sl = tree.edge_slice(node)
        n = tree.N[sl].astype(np.float32)
        w = tree.W[sl]
        prior = tree.prior[sl]

        # Virtual loss: in-flight simulations count as pending losses so concurrent
        # sims diverge instead of stacking on one path. Q_eff = (W - vloss)/(N + vloss).
        # With no sims in flight vloss is 0 everywhere and this reduces exactly to
        # the synchronous formula -- which is what keeps batch_leaves=1 equivalent.
        vl = tree.vloss[sl].astype(np.float32)
        n_eff = n + vl
        parent_q = tree.node_q(node)
        q = np.where(n_eff > 0, (w - vl) / np.maximum(n_eff, 1.0), parent_q)
        # Terminal edges have an exactly known value, which strictly dominates
        # both the running average and the completed-Q estimate. terminal_value
        # is from the child's perspective; negate for this node's. Without this
        # the search would rediscover a proven mate by sampling, which at small
        # sim budgets it simply cannot do: the mate-in-1 suite has a median of 43
        # legal moves, so 16 simulations touch under 40% of the root edges.
        # Exact values override, and they ignore vloss entirely: a proven mate is
        # worth +1 no matter how many simulations are currently traversing it.
        # Deducting virtual loss from a proven value would make a *certainty*
        # look worse the more the search agreed with it.
        has_exact, exact = tree.exact_value(sl)
        if has_exact.any():
            q = np.where(has_exact, -exact, q)
        u = self.cfg.c_puct * prior * np.sqrt(max(float(n_eff.sum()), 1.0)) / (1.0 + n_eff)
        return int(np.argmax(q + u)) + sl.start

    # -- the loop ---------------------------------------------------------

    def run(
        self,
        board: chess.Board,
        repetitions_fn: RepetitionsFn | None = None,
        num_sims: int | None = None,
    ) -> RootResult:
        """Run ``num_sims`` simulations from ``board`` and return root statistics.

        Dispatches on ``cfg.root_mode``: ``"gumbel"`` (default) or ``"puct"``
        (the ablation control). ``board`` is mutated during the search but is
        restored exactly: every ``push`` is matched by a ``pop``. No ``copy()``
        is taken anywhere.
        """
        if self.cfg.root_mode == "gumbel":
            return self.run_gumbel(board, repetitions_fn, num_sims)
        if self.cfg.root_mode != "puct":
            raise ValueError(f"unknown root_mode {self.cfg.root_mode!r}; use gumbel|puct")
        tree, root, sims_run = self.run_tree(board, repetitions_fn, num_sims)
        root_slice = tree.edge_slice(root)
        self._last_tree, self._last_root = tree, root
        return self._root_result(tree, root, root_slice, sims_run)

    def _root_result(self, tree: Tree, root: int, root_slice: slice, sims_run: int) -> RootResult:
        return RootResult(
            move_ft=tree.move_ft[root_slice].copy(),
            promo=tree.promo[root_slice].copy(),
            visits=tree.N[root_slice].copy(),
            q=np.where(
                tree.N[root_slice] > 0,
                tree.W[root_slice] / np.maximum(tree.N[root_slice], 1),
                0.0,
            ).astype(np.float32),
            prior=tree.prior[root_slice].copy(),
            sims_run=sims_run,
            evaluations=self._evaluations,
        )

    def run_gumbel(
        self,
        board: chess.Board,
        repetitions_fn: RepetitionsFn | None = None,
        num_sims: int | None = None,
    ) -> RootResult:
        """Gumbel top-m root selection with Sequential Halving (spec §6.2).

        Budget accounting, stated explicitly because chunk 5 ratified that
        terminal hits consume budget: **simulations** always total exactly
        ``num_sims``; **evaluator calls** are ``<= num_sims``, strictly less
        whenever a scheduled simulation lands on a terminal edge and
        short-circuits without touching the network. A test asserting
        "leaf evaluations == budget" would be asserting the absence of
        terminals, so ``RootResult.evaluations`` is reported separately.
        """
        num_sims = self.cfg.sims if num_sims is None else num_sims
        tree, root, _ = self.run_tree(board, repetitions_fn, 0)
        sl = tree.edge_slice(root)
        n_actions = int(tree.num_edges[root])
        if n_actions == 0:
            return self._root_result(tree, root, sl, 0)

        # Shared with BatchRunner (review S1): the noise draw or its g ≡ 0
        # substitute, top-m by g + logits, and force-include of proven wins all
        # live in one place so the two drivers cannot drift apart.
        g, logits, considered = gumbel.form_considered_set(tree, root, self.cfg, self.rng, num_sims)

        base_reps: RepetitionsFn = repetitions_fn if repetitions_fn is not None else (lambda _b: 0)
        path_reps: Counter = Counter()
        path_reps[board._transposition_key()] += 1

        sims_run = 0
        survivors = considered
        for budget in gumbel.halving_phase_budgets(num_sims, len(considered)):
            if budget <= 0 or not survivors:
                continue
            for local, visits in enumerate(gumbel.allocate_within_phase(budget, len(survivors))):
                for _ in range(visits):
                    self._simulate(
                        tree,
                        root,
                        board,
                        path_reps,
                        base_reps,
                        forced_edge=sl.start + int(survivors[local]),
                    )
                    sims_run += 1
            if len(survivors) > 1:
                scores = self._gumbel_scores(tree, root, g, logits)
                survivors = gumbel.halve_survivors(survivors, scores)

        self._last_tree, self._last_root = tree, root
        result = self._root_result(tree, root, sl, sims_run)
        completed_q = self._root_completed_q(tree, root)
        max_n = int(tree.N[sl].max()) if n_actions else 0
        result.improved_policy_probs = gumbel.improved_policy(logits, completed_q, max_n, self.cfg)
        final_scores = self._gumbel_scores(tree, root, g, logits)
        result.chosen = int(max(survivors, key=lambda a: final_scores[a]))
        return result

    def _solve_edge(self, tree: Tree, e: int) -> None:
        """Backward induction on ``e``'s child (MCTS-Solver, Winands et al. 2008).

        Two rules, both exact:

        * child's mover has *any* winning move  → child is a win for them;
        * *every* child move is already solved   → child's value is the best of them.

        Averaged statistics cannot express either fact. A mate-in-2 whose whole
        reply set has been refuted still shows Q well under 1 after backup, so
        Sequential Halving can eliminate a line that is *proven* won — which is
        what the 76% plateau was. Marking it proven makes the value permanent
        and immune to elimination.

        Kept separate from ``terminal`` so an immediate mate still outranks a
        longer forced one (see ``EXACT_WIN_BONUS``).

        Mate distance is inducted alongside the value: the winning side takes
        the **fastest** proved line, the losing side the **slowest**. Without a
        distance every proven win ties at q̂ = +1, and the acting argmax then
        breaks the tie on the prior — which is how a proven mate-in-5 was played
        over a proven mate-in-2 on nine positions of the gate suite.
        """
        c = int(tree.child[e])
        if c < 0 or tree.terminal[e] or tree.proven[e]:
            return
        sl = tree.edge_slice(c)
        if sl.stop <= sl.start:
            return
        has_exact, val = tree.exact_value(sl)
        mover_view = -val  # child's own side-to-move perspective
        depths = tree.exact_depth(sl)

        if has_exact.any() and (mover_view[has_exact] >= 1.0).any():
            solved = 1.0
            winning = has_exact & (mover_view >= 1.0)
            depth = 1 + int(depths[winning].min())  # winner mates as fast as it can
        elif has_exact.all():
            solved = float(mover_view.max())
            # Losing: every line is refuted, so the defender picks the longest.
            # Otherwise the distance belongs to the line actually chosen.
            depth = 1 + int(depths.max() if solved <= -1.0 else depths[int(np.argmax(mover_view))])
        else:
            return
        tree.proven[e] = True
        tree.proven_value[e] = solved
        tree.proven_depth[e] = min(depth, np.iinfo(np.int16).max)

    def _gumbel_scores(
        self, tree: Tree, root: int, g: np.ndarray, logits: np.ndarray
    ) -> np.ndarray:
        """``g + logits + σ(q̂)`` over all root actions, q̂ exact where known.

        **Every exactly-known win** — immediate mate or deeper proven line —
        gets ``EXACT_WIN_BONUS``, reduced by ``WIN_DEPTH_STEP`` per ply to mate.
        Two properties follow, and both are load-bearing:

        1. an exactly-known win always outranks a merely-estimated move, however
           good the prior thinks the estimate is;
        2. among wins, the shorter mate is always preferred.

        (1) is the fix for nine gate-suite failures where the search *proved* a
        mate-in-2 and played a proven mate-in-5 instead: all proven wins tie at
        q̂ = +1, so the acting argmax fell through to ``g + logits`` and picked
        whichever line the policy prior happened to like. (2) is the older
        ``IMMEDIATE_WIN_BONUS`` argument — a shorter forced mate is strictly
        better play — now stated over mate distance rather than over the single
        special case of depth 1.

        The bonus only ever reorders actions already at the +1 ceiling relative
        to each other, and lifts that whole group above everything else, so it
        cannot promote a worse action.
        """
        sl = tree.edge_slice(root)
        max_n = int(tree.N[sl].max()) if int(tree.num_edges[root]) else 0
        scores = g + logits + gumbel.sigma(self._root_completed_q(tree, root), max_n, self.cfg)
        exact_win = (tree.terminal[sl] & (-tree.terminal_value[sl] >= 1.0)) | (
            tree.proven[sl] & (-tree.proven_value[sl] >= 1.0)
        )
        depth = tree.exact_depth(sl)
        return np.where(exact_win, scores + EXACT_WIN_BONUS - depth * WIN_DEPTH_STEP, scores)

    def run_tree(
        self,
        board: chess.Board,
        repetitions_fn: RepetitionsFn | None = None,
        num_sims: int | None = None,
    ) -> tuple[Tree, int, int]:
        """Same as ``run`` but returns ``(tree, root, sims_run)``.

        Exposed so invariant tests can assert over the *whole* tree rather than
        only the root slice — checking vloss and W on interior edges is the
        entire point of the fuzz gate.
        """
        num_sims = self.cfg.sims if num_sims is None else num_sims
        base_reps: RepetitionsFn = repetitions_fn if repetitions_fn is not None else (lambda _b: 0)

        tree = Tree(cap_nodes=num_sims + 1)
        path_reps: Counter = Counter()
        # The root occupies the path too — without this the root is the one node
        # whose prior-occurrence count is off by one in the other direction, and
        # the two helpers above stop agreeing on what path_reps means.
        path_reps[board._transposition_key()] += 1
        self._evaluations = 0
        root, _ = self._expand(tree, board, path_reps, base_reps)

        if self.cfg.root_mode == "puct" and int(tree.num_edges[root]) > 0:
            # Ablation control only: Dirichlet noise on the root prior, v1
            # semantics. Gumbel mode deliberately has no noise knob at all.
            sl = tree.edge_slice(root)
            tree.prior[sl] = gumbel.dirichlet_mixed_prior(tree.prior[sl], self.rng, self.cfg)

        sims_run = 0
        if int(tree.num_edges[root]) > 0:
            for _ in range(num_sims):
                self._simulate(tree, root, board, path_reps, base_reps)
                sims_run += 1
        return tree, root, sims_run

    def _simulate(
        self,
        tree: Tree,
        root: int,
        board: chess.Board,
        path_reps: Counter,
        base_reps: RepetitionsFn,
        forced_edge: int | None = None,
    ) -> None:
        """Drive one simulation to completion, evaluating leaves inline.

        The work lives in ``simulate_gen``; this is the zero-latency driver that
        answers each evaluation request immediately. ``BatchRunner`` drives the
        same generator but defers the answer until a batch is full, which is why
        batch_leaves=1 reproduces this exactly.
        """
        gen = self.simulate_gen(tree, root, board, path_reps, base_reps, forced_edge)
        try:
            tokens = next(gen)
            while True:
                self._evaluations += 1
                tokens = gen.send(self.evaluate_fn(tokens))
        except StopIteration:
            pass

    def simulate_gen(
        self,
        tree: Tree,
        root: int,
        board: chess.Board,
        path_reps: Counter,
        base_reps: RepetitionsFn,
        forced_edge: int | None = None,
        use_vloss: bool = False,
    ):
        """One simulation as a coroutine: yields leaf tokens, receives (policy, value).

        Suspends at most once — a simulation either short-circuits on a terminal
        edge (no evaluation at all) or expands exactly one leaf. That single
        suspension point is what makes the batched runner a scheduling change
        rather than a second search implementation.

        ``use_vloss`` applies virtual loss along the in-flight path so concurrent
        simulations in the same tree diverge; it is removed on resume. Terminal
        short-circuits never yield, so they never apply or remove vloss — the
        buffer accounting stays balanced with terminals present (review Q4).
        """
        path: list[int] = []
        node = root
        pushed = 0
        value: float

        while True:
            if forced_edge is not None and node == root and not path:
                e = forced_edge
            else:
                e = self._select_edge(tree, node)
            path.append(e)

            mv = _move_of(int(tree.move_ft[e]), int(tree.promo[e]))
            board.push(mv)
            pushed += 1
            path_reps[board._transposition_key()] += 1

            if tree.terminal[e]:
                value = float(tree.terminal_value[e])
                break
            if tree.child[e] == -1:
                tokens = self.leaf_tokens(board, path_reps, base_reps)
                if use_vloss:
                    for pe in path:
                        tree.vloss[pe] += 1
                policy, promo_probs, value_est = yield tokens
                if use_vloss:
                    for pe in path:
                        tree.vloss[pe] -= 1
                child, value = self._build_node(
                    tree, board, path_reps, base_reps, policy, promo_probs, value_est
                )
                tree.child[e] = child
                break
            node = int(tree.child[e])

        for e in reversed(path):
            value = -value
            tree.N[e] += 1
            tree.W[e] += value
            self._solve_edge(tree, e)

        for _ in range(pushed):
            path_reps[board._transposition_key()] -= 1
            board.pop()


# --- move codec helpers (hot path: avoid re-deriving these per call) --------

_PROMO_OF = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
_PIECE_OF = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def _move_of(move_ft: int, promo: int) -> chess.Move:
    return chess.Move(move_ft // 64, move_ft % 64, promotion=_PIECE_OF[promo])


# --- suite runner -----------------------------------------------------------


def load_epd(path: str | Path) -> list[tuple[str, str]]:
    """Read an EPD suite as ``[(epd_line, id)]``, skipping blanks and comments."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ident = ""
        if 'id "' in line:
            ident = line.split('id "', 1)[1].split('"', 1)[0]
        out.append((line, ident))
    return out


def _forces_mate_in_2(board: chess.Board, mv: chess.Move) -> bool:
    """After ``mv``, is every opponent reply answerable by mate?

    The mate-in-2 gate cannot be scored by ``bm`` matching: positions often have
    several winning first moves, and scoring against one of them measures suite
    authorship rather than search.
    """
    board.push(mv)
    try:
        if board.is_checkmate():
            return True
        replies = list(board.legal_moves)
        if not replies:
            return False
        for reply in replies:
            board.push(reply)
            try:
                if not any(_delivers_mate(board, m2) for m2 in board.legal_moves):
                    return False
            finally:
                board.pop()
        return True
    finally:
        board.pop()


def _delivers_mate(board: chess.Board, mv: chess.Move) -> bool:
    board.push(mv)
    out = board.is_checkmate()
    board.pop()
    return out


def solve_positions(
    positions: list[tuple[str, str]],
    sims: int,
    evaluate_fn: EvaluateFn,
    cfg: MCTSConfig | None = None,
    mate_in: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, list[str]]:
    """Solve a list of ``(epd_line, id)`` entries. Returns ``(solved, total, failed)``.

    Split out from ``run_suite`` so a caller can shard a suite across worker
    processes and sum the parts. That is only a legitimate thing to do because
    each position gets an independent search — which is exactly true under the
    evaluation profile, and *not* true under self-play noise, where the shards
    would consume different draws from their own RNGs and the total would depend
    on how the work happened to be divided.
    """
    search = Search(evaluate_fn, cfg, rng)
    solved = 0
    failed: list[str] = []
    for epd_line, ident in positions:
        board = chess.Board()
        board.set_epd(epd_line)
        result = search.run(board, None, sims)
        mv = result.best_move()
        ok = _delivers_mate(board, mv) if mate_in == 1 else _forces_mate_in_2(board, mv)
        if ok:
            solved += 1
        else:
            failed.append(ident or board.epd())
    return solved, len(positions), failed


def run_suite(
    epd_path: str | Path,
    sims: int,
    evaluate_fn: EvaluateFn,
    cfg: MCTSConfig | None = None,
    mate_in: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, list[str]]:
    """Solve every position in an EPD suite. Returns ``(solved, total, failed_ids)``.

    Solved means the chosen move *is* a solution — delivers mate (``mate_in=1``)
    or forces mate next move (``mate_in=2``) — not that it matches the recorded
    ``bm``. Several positions have more than one solution and pinning one of them
    would be testing the suite, not the search.
    """
    return solve_positions(load_epd(epd_path), sims, evaluate_fn, cfg, mate_in, rng)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, type=Path, help="EPD suite path")
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument(
        "--stub",
        action="store_true",
        help="Use the uniform-policy zero-value stub evaluator: search with the "
        "network out of the picture, so a failure is unambiguously a search bug.",
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint for the network evaluator, e.g. runs/phase1_anchor.pt. "
        "Mutually exclusive with --stub.",
    )
    ap.add_argument(
        "--root-mode", choices=["gumbel", "puct"], default=None, help="Override cfg.root_mode"
    )
    ap.add_argument(
        "--root-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw Gumbel noise at the root (the SELF-PLAY profile). Suites default "
        "to --no-root-noise, the evaluation profile: deterministic top-m by raw "
        "logits and an acting argmax of logits + sigma(q), which is what a "
        "strength measurement should read. Turning this on makes the result "
        "seed-dependent.",
    )
    ap.add_argument(
        "--eval-m",
        type=int,
        default=None,
        help="Considered-set size m for the evaluation profile only; self-play's "
        "gumbel_m is untouched. Default: inherit gumbel_m (16).",
    )
    ap.add_argument(
        "--eval-c-visit",
        type=float,
        default=None,
        help="sigma c_visit for the evaluation profile only. Default: inherit (50.0).",
    )
    ap.add_argument(
        "--eval-c-scale",
        type=float,
        default=None,
        help="sigma c_scale for the evaluation profile only. Default: inherit (1.0).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Gumbel noise seed. Has no effect under the evaluation profile, which "
        "draws no noise at all — two seeds must give byte-identical results.",
    )
    ap.add_argument(
        "--mate-in",
        type=int,
        default=None,
        help="Solution depth to score against; inferred from the suite name if omitted.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Pass threshold as a fraction, e.g. 0.95. Defaults to 1.0 for mate-in-1 "
        "and 0.95 for mate-in-2, matching the §6.4 gates.",
    )
    args = ap.parse_args(argv)

    if bool(args.stub) == bool(args.ckpt):
        print("error: pass exactly one of --stub or --ckpt")
        return 2

    mate_in = args.mate_in if args.mate_in else (2 if "mate_in_2" in args.suite.name else 1)
    threshold = args.threshold if args.threshold else (0.95 if mate_in == 2 else 1.0)

    cfg = MCTSConfig()
    if args.root_mode:
        cfg.root_mode = args.root_mode
    cfg.root_noise = bool(args.root_noise)
    cfg.eval_m = args.eval_m
    cfg.eval_c_visit = args.eval_c_visit
    cfg.eval_c_scale = args.eval_c_scale
    active = cfg.resolved()

    if args.stub:
        evaluate_fn = uniform_stub_evaluator
        which = "stub"
    else:
        evaluate_fn = network_evaluator(args.ckpt)
        which = str(args.ckpt)

    solved, total, failed = run_suite(
        args.suite,
        args.sims,
        evaluate_fn,
        cfg,
        mate_in=mate_in,
        rng=np.random.default_rng(args.seed),
    )
    frac = solved / total if total else 0.0
    # The profile and the active constants go in the result line because these
    # numbers get pasted into gate documents, where "77.0%" with no profile
    # attached is exactly the ambiguity this whole investigation is about.
    profile = "selfplay(noise)" if cfg.root_noise else "eval(deterministic)"
    print(
        f"{args.suite.name}: {solved}/{total} solved at {args.sims} sims "
        f"({100 * frac:.1f}%) | eval={which} root={cfg.root_mode} "
        f"profile={profile} m={active.gumbel_m} "
        f"c_visit={active.c_visit:g} c_scale={active.c_scale:g} "
        f"| gate >= {100 * threshold:.0f}%"
    )
    if failed:
        print(f"  failed: {', '.join(failed[:10])}{' …' if len(failed) > 10 else ''}")
    return 0 if frac >= threshold else 1


if __name__ == "__main__":
    sys.exit(main())
