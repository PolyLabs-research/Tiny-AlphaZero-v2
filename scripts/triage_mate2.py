#!/usr/bin/env python3
"""Classify every mate-in-2 gate failure as BUG or CAPABILITY (brief part B2).

    VIRTUAL_ENV=.venv .venv/bin/python scripts/triage_mate2.py \\
        --suite tests/suites/mate_in_2.epd --ckpt runs/phase1_d/final.pt --sims 128

A residual of "conversion failures" is not actionable until you know which kind
each one is, and the two kinds have opposite remedies:

* **BUG** — the proof was *available* and the search did not act on it: either
  every opponent reply beneath the key move was already exact and the edge was
  not marked proven, or it was marked proven and still lost the acting argmax.
  Both are defects in code that exists, and both get fixed.
* **CAPABILITY** — the key move was considered but never received enough
  simulations to expand its full reply set, so no proof was reachable and
  Sequential Halving eliminated it on an interim *value estimate*. Nothing is
  broken; the value head is not yet sharp enough to hold the move in contention.
  These are the positions self-play value learning has to convert, so they
  become the M3 watchlist.
* **COVERAGE** — the key move was outside the considered set. Prior-side, and
  already known to be ~1% of the suite.

Proving a mate-in-2 costs at least ``1 + len(opponent replies)`` simulations on
the key edge: one to expand the child, then one per reply so each reply's own
mate becomes visible and the solver can induct back. That arithmetic is what
separates "under-visited" from "broken".

Diagnosis only. Selects nothing, gates nothing.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import chess

from tinyalphazero.config import load_config
from tinyalphazero.mcts import gumbel
from tinyalphazero.mcts.search import (
    Search,
    _forces_mate_in_2,
    load_epd,
    network_evaluator,
)


def drive_gumbel(search: Search, board: chess.Board, num_sims: int):
    """Reproduce ``Search.run_gumbel`` while keeping the tree.

    ``run_gumbel`` returns a ``RootResult`` and drops the tree, and triage needs
    the tree. This re-drives the *shared* helpers rather than reimplementing
    them, and ``--validate`` asserts the reproduction picks the same move as the
    real driver on every position, so a drift here fails loudly instead of
    quietly producing a plausible triage.
    """
    tree, root, _ = search.run_tree(board, None, 0)
    sl = tree.edge_slice(root)
    n_actions = int(tree.num_edges[root])
    if n_actions == 0:
        return None

    g, logits, considered = gumbel.form_considered_set(tree, root, search.cfg, search.rng, num_sims)
    path_reps: Counter = Counter()
    path_reps[board._transposition_key()] += 1

    survivors = list(considered)
    for budget in gumbel.halving_phase_budgets(num_sims, len(considered)):
        if budget <= 0 or not survivors:
            continue
        for local in gumbel.phase_visit_queue(budget, len(survivors)):
            search._simulate(
                tree,
                root,
                board,
                path_reps,
                lambda _b: 0,
                forced_edge=sl.start + int(survivors[local]),
            )
        if len(survivors) > 1:
            survivors = gumbel.halve_survivors(
                survivors, search._gumbel_scores(tree, root, g, logits)
            )

    scores = search._gumbel_scores(tree, root, g, logits)
    chosen = int(max(survivors, key=lambda a: scores[a]))
    return tree, root, sl, considered, survivors, chosen


def classify(tree, root, sl, considered, chosen, key_locals) -> tuple[str, dict]:
    """BUG / CAPABILITY / COVERAGE, with the evidence that decided it."""
    considered_keys = [k for k in key_locals if k in considered]
    if not considered_keys:
        return "COVERAGE", {"reason": "no key move in the considered set"}

    best = max(considered_keys, key=lambda k: int(tree.N[sl.start + k]))
    e = sl.start + best
    info = {
        "key_edge_visits": int(tree.N[e]),
        "key_edge_proven": bool(tree.proven[e]),
        "key_edge_q": round(float(tree.W[e] / max(int(tree.N[e]), 1)), 4),
        "chosen_was_key": chosen in key_locals,
    }

    if tree.proven[e] and chosen not in key_locals:
        return "BUG", {**info, "reason": "key edge proven but not selected"}

    child = int(tree.child[e])
    if child < 0:
        return "CAPABILITY", {
            **info,
            "replies_expanded": "0/?",
            "reason": "key move never expanded",
        }

    csl = tree.edge_slice(child)
    has_exact, _ = tree.exact_value(csl)
    n_replies = int(tree.num_edges[child])
    n_exact = int(has_exact.sum())
    info["replies_expanded"] = f"{n_exact}/{n_replies}"
    info["sims_needed_for_proof"] = 1 + n_replies

    if n_exact == n_replies and not tree.proven[e]:
        return "BUG", {**info, "reason": "all replies exact but edge not proven"}
    if n_exact < n_replies:
        return "CAPABILITY", {**info, "reason": "reply set never fully expanded"}
    return "CAPABILITY", {**info, "reason": "proof available but move not chosen"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--out", type=Path, default=None, help="JSON sink for the watchlist")
    ap.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--config",
        default=None,
        help="Campaign YAML whose mcts section the search runs under (eval profile: "
        "root_noise forced off). Default: the dataclass defaults, i.e. what the "
        "diagnostic measured before --config existed (REVIEW-2026-10-21 E13).",
    )
    args = ap.parse_args(argv)

    cfg = replace(load_config(args.config).mcts, root_noise=False)
    evaluate_fn = network_evaluator(args.ckpt)
    search = Search(evaluate_fn, cfg)

    rows, mismatches = [], 0
    for epd_line, ident in load_epd(args.suite):
        board = chess.Board()
        board.set_epd(epd_line)
        out = drive_gumbel(search, board, args.sims)
        if out is None:
            continue
        tree, root, sl, considered, _survivors, chosen = out

        moves = [
            chess.Move(
                int(tree.move_ft[sl.start + i]) // 64,
                int(tree.move_ft[sl.start + i]) % 64,
                promotion=[None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][
                    int(tree.promo[sl.start + i])
                ],
            )
            for i in range(int(tree.num_edges[root]))
        ]
        if args.validate:
            real = search.run(board, None, args.sims).best_move()
            mismatches += real != moves[chosen]

        if _forces_mate_in_2(board, moves[chosen]):
            continue  # solved

        key_locals = [i for i, mv in enumerate(moves) if _forces_mate_in_2(board, mv)]
        verdict, info = classify(tree, root, sl, considered, chosen, key_locals)

        key_child_value = None
        if key_locals:
            best_key = max(key_locals, key=lambda k: int(tree.N[sl.start + k]))
            board.push(moves[best_key])
            try:
                from tinyalphazero.encoding import encode_board

                # Value head on the position AFTER the key move, opponent to
                # move: the number self-play has to sharpen for this position to
                # survive halving. Negated into our perspective.
                key_child_value = -float(evaluate_fn(encode_board(board, 0))[2])
            finally:
                board.pop()

        rows.append(
            {
                "id": ident,
                "verdict": verdict,
                "fen": board.fen(),
                "n_key_moves": len(key_locals),
                "value_head_on_key_child": (
                    None if key_child_value is None else round(key_child_value, 4)
                ),
                **info,
            }
        )

    if args.validate:
        print(f"reproduction check: {mismatches} mismatch(es) vs Search.run\n")
    counts = Counter(r["verdict"] for r in rows)
    print(f"{len(rows)} failures: " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
    print()
    hdr = f"{'id':<12} {'verdict':<11} {'vis':>4} {'replies':>9} {'need':>5} {'q':>8} {'v_head':>7}  reason"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['id']:<12} {r['verdict']:<11} {r.get('key_edge_visits', 0):>4} "
            f"{r.get('replies_expanded', '-'):>9} {r.get('sims_needed_for_proof', '-'):>5} "
            f"{r.get('key_edge_q', float('nan')):>8} "
            f"{r['value_head_on_key_child'] if r['value_head_on_key_child'] is not None else '-':>7}  "
            f"{r['reason']}"
        )

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
