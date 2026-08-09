#!/usr/bin/env python3
"""Decompose the chunk 6 mate-in-2 gate into consideration × conversion.

    VIRTUAL_ENV=.venv .venv/bin/python scripts/diagnose_mate2.py \\
        --suite tests/suites/mate_in_2.epd --ckpt runs/phase1_d/final.pt --sims 128

The gate is a product, and round `d` was spent moving the wrong factor. Under
Gumbel the chain is

    P(solved) = P(a forcing move is considered) × P(solved | considered)

and under the evaluation profile the first factor is just prior top-m coverage,
because there is no noise left to knock a top-ranked move out of the set. This
script reports both factors against the same run that produces the gate number,
so a future round can see which one it would be aiming at *before* it spends
compute — the mistake round `d` made was treating an upper bound as an estimator.

Diagnosis only. It selects nothing and gates nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np

from tinyalphazero.config import MCTSConfig
from tinyalphazero.mcts.search import (
    Search,
    _forces_mate_in_2,
    load_epd,
    network_evaluator,
)


def forcing_moves(board: chess.Board) -> list[chess.Move]:
    """Every first move that forces mate next move — not just the recorded bm."""
    return [mv for mv in board.legal_moves if _forces_mate_in_2(board, mv)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--eval-m", type=int, default=None)
    ap.add_argument("--eval-c-visit", type=float, default=None)
    ap.add_argument("--eval-c-scale", type=float, default=None)
    args = ap.parse_args(argv)

    cfg = MCTSConfig(
        root_noise=False,
        eval_m=args.eval_m,
        eval_c_visit=args.eval_c_visit,
        eval_c_scale=args.eval_c_scale,
    )
    active = cfg.resolved()
    m = active.gumbel_m
    evaluate_fn = network_evaluator(args.ckpt)
    search = Search(evaluate_fn, cfg)

    considered_hits = 0
    solved = 0
    solved_given_considered = 0
    ranks: list[int] = []
    missed_ids: list[str] = []

    for epd_line, ident in load_epd(args.suite):
        board = chess.Board()
        board.set_epd(epd_line)
        forcing = forcing_moves(board)
        if not forcing:
            continue

        # Prior rank of the best-ranked forcing move. The considered set under
        # the eval profile is exactly the top-m of this ordering.
        tree, root, _ = search.run_tree(board, None, 0)
        sl = tree.edge_slice(root)
        order = np.argsort(-tree.prior[sl], kind="stable")
        rank_of = {int(e): r for r, e in enumerate(order)}
        forcing_ft = {(mv.from_square * 64 + mv.to_square) for mv in forcing}
        best_rank = min(
            rank_of[i]
            for i in range(int(tree.num_edges[root]))
            if int(tree.move_ft[sl.start + i]) in forcing_ft
        )
        ranks.append(best_rank)
        is_considered = best_rank < m
        considered_hits += is_considered

        result = search.run(board, None, args.sims)
        ok = _forces_mate_in_2(board, result.best_move())
        solved += ok
        if is_considered:
            solved_given_considered += ok
        elif ok:
            # Not impossible: force-include rescues a *proven* win found during
            # root expansion regardless of prior rank.
            missed_ids.append(ident)

    total = len(ranks)
    cov = considered_hits / total
    conv = solved_given_considered / considered_hits if considered_hits else 0.0
    ranks_arr = np.array(ranks)
    print(
        f"suite={args.suite.name} n={total} m={m} "
        f"c_visit={active.c_visit:g} c_scale={active.c_scale:g} sims={args.sims}"
    )
    print(f"  prior coverage @top-{m:<3}        {cov:.3f}  ({considered_hits}/{total})")
    print(
        f"  solved | considered            {conv:.3f}  "
        f"({solved_given_considered}/{considered_hits})"
    )
    print(f"  product                        {cov * conv:.3f}")
    print(f"  measured gate                  {solved / total:.3f}  ({solved}/{total})")
    for k in (1, 8, 16, 32):
        print(f"  prior top-{k:<3} coverage         {(ranks_arr < k).mean():.3f}")
    print(f"  median forcing-move rank       {int(np.median(ranks_arr))}")
    if missed_ids:
        print(f"  solved despite not considered  {len(missed_ids)} (force-include)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
