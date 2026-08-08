"""Synchronous single-game MCTS with classic PUCT selection (spec §6.1, §6.4).

Correctness before speed: no batching, no Gumbel, uniform root. Gumbel arrives
in chunk 6; this chunk exists to make search bugs impossible to confuse with
model quality, which is why every gate runs against a stub evaluator.

Run the mate-in-1 suite:

    python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_1.epd --sims 16 --stub
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
from tinyalphazero.mcts.tree import Tree

# evaluate_fn(tokens (72,)) -> (policy_ft (64,64) probs already legal-masked, value float)
EvaluateFn = Callable[[np.ndarray], tuple[np.ndarray, float]]
# repetitions_fn(board) -> times this position already occurred in game history
# *before* the search started. In-search repetitions are counted separately.
RepetitionsFn = Callable[[chess.Board], int]

# Draw-claim thresholds (spec §6.3: "50-move and threefold via claim checks").
FIFTY_MOVE_PLIES = 100
THREEFOLD_COUNT = 3


@dataclass
class RootResult:
    """Root statistics after a search."""

    move_ft: np.ndarray  # int16 (n_edges,)  from*64+to
    promo: np.ndarray  # int8  (n_edges,)
    visits: np.ndarray  # int32 (n_edges,)
    q: np.ndarray  # float32 (n_edges,) root side-to-move perspective
    prior: np.ndarray  # float32 (n_edges,)
    sims_run: int

    def moves(self) -> list[chess.Move]:
        from tinyalphazero.encoding import decode_move

        return [
            decode_move(int(ft) // 64, int(ft) % 64, int(p))
            for ft, p in zip(self.move_ft, self.promo, strict=True)
        ]

    def best_move(self) -> chess.Move:
        """Most-visited root move, ties broken by Q then by index (deterministic)."""
        if len(self.visits) == 0:
            raise ValueError("no root edges — position is terminal")
        best = max(
            range(len(self.visits)),
            key=lambda i: (int(self.visits[i]), float(self.q[i]), -i),
        )
        return self.moves()[best]


def uniform_stub_evaluator(tokens: np.ndarray) -> tuple[np.ndarray, float]:
    """Uniform policy over legal moves, zero value.

    Decodes the board from ``tokens`` — legitimate precisely because the
    encoding is Markovian (spec §3.1); if this ever stops working, the Markov
    invariant is broken and that is the bug worth finding.
    """
    board = decode_board(tokens)
    policy = np.zeros((64, 64), dtype=np.float32)
    legal = list(board.legal_moves)
    if legal:
        for mv in legal:
            policy[mv.from_square, mv.to_square] = 1.0
        policy /= policy.sum()
    return policy, 0.0


class Search:
    """Synchronous PUCT search over an array tree."""

    def __init__(self, evaluate_fn: EvaluateFn, cfg: MCTSConfig | None = None) -> None:
        self.evaluate_fn = evaluate_fn
        self.cfg = cfg if cfg is not None else MCTSConfig()

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
        rep_here = base_reps(board) + path_reps[board._transposition_key()]
        tokens = encode_board(board, rep_here)
        policy, value = self.evaluate_fn(tokens)

        edges: list[tuple[int, int, float]] = []
        terminal_flags: list[tuple[bool, float]] = []
        for mv in board.legal_moves:
            promo = _PROMO_OF[mv.promotion]
            ft = mv.from_square * 64 + mv.to_square

            board.push(mv)
            key = board._transposition_key()
            path_reps[key] += 1
            child_reps = base_reps(board) + path_reps[key]
            tv = self._terminal_value(board, child_reps)
            path_reps[key] -= 1
            board.pop()

            edges.append((ft, promo, float(policy[mv.from_square, mv.to_square])))
            terminal_flags.append((tv is not None, 0.0 if tv is None else tv))

        # Normalize priors over the edges actually created. Promotions share a
        # (from, to) cell, so the four of them split that cell's mass by count.
        total = sum(p for _, _, p in edges)
        if total > 0:
            edges = [(ft, pr, p / total) for ft, pr, p in edges]
        elif edges:
            edges = [(ft, pr, 1.0 / len(edges)) for ft, pr, _ in edges]

        node = tree.add_node(edges)
        sl = tree.edge_slice(node)
        tree.terminal[sl] = [t for t, _ in terminal_flags]
        tree.terminal_value[sl] = [v for _, v in terminal_flags]
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

        parent_q = tree.node_q(node)
        q = np.where(n > 0, w / np.maximum(n, 1.0), parent_q)
        # Terminal edges have an exactly known value, which strictly dominates
        # both the running average and the completed-Q estimate. terminal_value
        # is from the child's perspective; negate for this node's. Without this
        # the search would rediscover a proven mate by sampling, which at small
        # sim budgets it simply cannot do: the mate-in-1 suite has a median of 43
        # legal moves, so 16 simulations touch under 40% of the root edges.
        term = tree.terminal[sl]
        if term.any():
            q = np.where(term, -tree.terminal_value[sl], q)
        u = self.cfg.c_puct * prior * np.sqrt(max(float(n.sum()), 1.0)) / (1.0 + n)
        return int(np.argmax(q + u)) + sl.start

    # -- the loop ---------------------------------------------------------

    def run(
        self,
        board: chess.Board,
        repetitions_fn: RepetitionsFn | None = None,
        num_sims: int | None = None,
    ) -> RootResult:
        """Run ``num_sims`` simulations from ``board`` and return root statistics.

        ``board`` is mutated during the search but is restored exactly: every
        ``push`` is matched by a ``pop``. No ``copy()`` is taken anywhere.
        """
        tree, root, sims_run = self.run_tree(board, repetitions_fn, num_sims)
        root_slice = tree.edge_slice(root)
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
        )

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
        base_reps: RepetitionsFn = (
            repetitions_fn if repetitions_fn is not None else (lambda _b: 0)
        )

        tree = Tree(cap_nodes=num_sims + 1)
        path_reps: Counter = Counter()
        root, _ = self._expand(tree, board, path_reps, base_reps)

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
    ) -> None:
        """One simulation: descend, expand or short-circuit, back up, unwind."""
        path: list[int] = []  # edge indices traversed
        node = root
        pushed = 0
        value: float

        while True:
            e = self._select_edge(tree, node)
            path.append(e)

            mv = _move_of(int(tree.move_ft[e]), int(tree.promo[e]))
            board.push(mv)
            pushed += 1
            path_reps[board._transposition_key()] += 1

            if tree.terminal[e]:
                # Exact outcome — never touches the evaluator (spec §6.3).
                value = float(tree.terminal_value[e])
                break
            if tree.child[e] == -1:
                child, value = self._expand(tree, board, path_reps, base_reps)
                tree.child[e] = child
                break
            node = int(tree.child[e])

        # Back up. `value` is from the perspective of the side to move at the
        # leaf; flip once per ply on the way up.
        for e in reversed(path):
            value = -value
            tree.N[e] += 1
            tree.W[e] += value

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


def run_suite(
    epd_path: str | Path, sims: int, evaluate_fn: EvaluateFn, cfg: MCTSConfig | None = None
) -> tuple[int, int, list[str]]:
    """Solve every position in an EPD suite. Returns ``(solved, total, failed_ids)``.

    A mate-in-1 position counts as solved if the chosen move *delivers mate* —
    not if it matches the recorded ``bm``. Several positions have more than one
    mating move and pinning one of them would be testing the suite, not the search.
    """
    search = Search(evaluate_fn, cfg)
    positions = load_epd(epd_path)
    solved = 0
    failed: list[str] = []
    for epd_line, ident in positions:
        board = chess.Board()
        board.set_epd(epd_line)
        result = search.run(board, None, sims)
        mv = result.best_move()
        board.push(mv)
        if board.is_checkmate():
            solved += 1
        else:
            failed.append(ident or board.epd())
        board.pop()
    return solved, len(positions), failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, type=Path, help="EPD suite path")
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument(
        "--stub",
        action="store_true",
        help="Use the uniform-policy zero-value stub evaluator. Required in chunk 5: "
        "all gates run with the network out of the picture so a failure is a search bug.",
    )
    args = ap.parse_args(argv)

    if not args.stub:
        print("error: only --stub is available in chunk 5 (network arrives in chunk 6)")
        return 2

    solved, total, failed = run_suite(args.suite, args.sims, uniform_stub_evaluator)
    pct = 100.0 * solved / total if total else 0.0
    print(f"{args.suite.name}: {solved}/{total} solved at {args.sims} sims ({pct:.1f}%)")
    if failed:
        print(f"  failed: {', '.join(failed[:10])}{' …' if len(failed) > 10 else ''}")
    return 0 if solved == total else 1


if __name__ == "__main__":
    sys.exit(main())
