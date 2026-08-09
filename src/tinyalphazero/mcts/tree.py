"""Array-backed search tree (spec §6.1).

No board is stored in the tree. Every simulation replays its path on a single
scratch ``chess.Board`` via ``push()``/``pop()``; it was the per-child ``copy()``
that killed v1.

The nine arrays below are a **contract** — chunks 6–7 are written against these
names, dtypes and semantics, so do not rename or re-type them. ``terminal`` and
``terminal_value`` are additions, not modifications: they carry exact outcomes
for edges whose child position is over, so a simulation ending there never
touches the evaluator. See ``Search._expand`` for why they are filled eagerly.
"""

from __future__ import annotations

import numpy as np

# Maximum legal moves in any reachable chess position is 218; use it as the
# per-node edge budget so a preallocated tree cannot overflow in practice.
MAX_BRANCHING = 218


class Tree:
    """Preallocated node/edge arrays for one search.

    Sized for ``cap_nodes`` nodes (spec: ``sims + 1``) and a parallel edge table.
    Arrays grow by doubling if a pathological position exceeds the estimate —
    preallocation is the design, growth is a safety net that cannot crash a run.
    """

    def __init__(self, cap_nodes: int, cap_edges: int | None = None) -> None:
        if cap_nodes < 1:
            raise ValueError(f"cap_nodes must be >= 1, got {cap_nodes}")
        cap_edges = cap_nodes * 40 if cap_edges is None else cap_edges
        cap_edges = max(cap_edges, MAX_BRANCHING)

        # --- node arrays (contract) ---
        self.first_edge = np.zeros(cap_nodes, dtype=np.int32)
        self.num_edges = np.zeros(cap_nodes, dtype=np.int16)
        # --- edge arrays (contract) ---
        self.child = np.full(cap_edges, -1, dtype=np.int32)
        self.move_ft = np.zeros(cap_edges, dtype=np.int16)
        self.promo = np.zeros(cap_edges, dtype=np.int8)
        self.prior = np.zeros(cap_edges, dtype=np.float32)
        self.N = np.zeros(cap_edges, dtype=np.int32)
        self.W = np.zeros(cap_edges, dtype=np.float32)
        self.vloss = np.zeros(cap_edges, dtype=np.int16)
        # --- additions (see module docstring) ---
        # terminal: the child position is game-over (set once, at expansion).
        self.terminal = np.zeros(cap_edges, dtype=bool)
        self.terminal_value = np.zeros(cap_edges, dtype=np.float32)
        # proven: the child *subtree* is solved by backward induction, even
        # though the position itself is not over. Kept separate from `terminal`
        # so an immediate mate can still be preferred over a longer forced one.
        self.proven = np.zeros(cap_edges, dtype=bool)
        self.proven_value = np.zeros(cap_edges, dtype=np.float32)

        self.n_nodes = 0
        self.n_edges = 0

    # -- capacity -------------------------------------------------------

    def _grow_nodes(self) -> None:
        new_cap = max(2 * len(self.first_edge), self.n_nodes + 1)
        self.first_edge = np.resize(self.first_edge, new_cap)
        self.num_edges = np.resize(self.num_edges, new_cap)

    def _grow_edges(self, needed: int) -> None:
        new_cap = max(2 * len(self.child), self.n_edges + needed)
        old = len(self.child)
        self.child = np.resize(self.child, new_cap)
        self.child[old:] = -1  # np.resize tiles old contents; -1 means unexpanded
        for name in ("move_ft", "promo", "prior", "N", "W", "vloss", "terminal",
                     "terminal_value", "proven", "proven_value"):
            arr = np.resize(getattr(self, name), new_cap)
            arr[old:] = 0
            setattr(self, name, arr)

    # -- construction ---------------------------------------------------

    def add_node(self, edges: list[tuple[int, int, float]]) -> int:
        """Append a node owning ``edges`` of ``(move_ft, promo, prior)``. Returns its index."""
        if self.n_nodes >= len(self.first_edge):
            self._grow_nodes()
        if self.n_edges + len(edges) > len(self.child):
            self._grow_edges(len(edges))

        node = self.n_nodes
        self.n_nodes += 1
        start = self.n_edges
        self.first_edge[node] = start
        self.num_edges[node] = len(edges)

        for i, (move_ft, promo, prior) in enumerate(edges):
            e = start + i
            self.move_ft[e] = move_ft
            self.promo[e] = promo
            self.prior[e] = prior
            self.child[e] = -1
            self.N[e] = 0
            self.W[e] = 0.0
            self.vloss[e] = 0
            self.terminal[e] = False
            self.terminal_value[e] = 0.0
            self.proven[e] = False
            self.proven_value[e] = 0.0
        self.n_edges += len(edges)
        return node

    # -- access ---------------------------------------------------------

    def edge_slice(self, node: int) -> slice:
        start = int(self.first_edge[node])
        return slice(start, start + int(self.num_edges[node]))

    def exact_value(self, sl: slice) -> tuple[np.ndarray, np.ndarray]:
        """``(has_exact, value)`` per edge in ``sl``, child-perspective.

        Terminal (position over) takes precedence over proven (subtree solved);
        both are exact, and where both hold they agree.
        """
        has = self.terminal[sl] | self.proven[sl]
        val = np.where(self.terminal[sl], self.terminal_value[sl], self.proven_value[sl])
        return has, val.astype(np.float32)

    def node_q(self, node: int) -> float:
        """Value of ``node`` from its own side-to-move perspective.

        Used as completed-Q for unvisited edges: v1 treated those as 0, which
        systematically overrated them from losing positions (spec §6.2).
        Derived from the contract arrays alone — no node-level N/W needed.
        """
        sl = self.edge_slice(node)
        n_sum = int(self.N[sl].sum())
        if n_sum == 0:
            return 0.0
        return float(self.W[sl].sum() / n_sum)
