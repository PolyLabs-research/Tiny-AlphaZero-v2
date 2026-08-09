#!/usr/bin/env python3
"""Sync-vs-batched root parity probe (review 2026-08-09, S2).

    VIRTUAL_ENV=.venv .venv/bin/python scripts/parity_probe.py

At ``batch_leaves=1`` with one game, ``BatchRunner`` must reproduce
``Search.run_gumbel`` *exactly* — same seed, same root visit vector. That is the
gate keeping the concurrent path honest, and it only bites at combinations where
Sequential Halving has a remainder to distribute: when every phase budget
divides evenly across its survivors, a round-robin and a proper allocation agree
by accident.

So this sweeps m × sims rather than testing one comfortable pair. The reviewed
defect reproduces at ``m=5, sims=6``: the schedule is ``[5, 1, 0]``, and the
single visit of phase 2 belongs to the *best-ranked* survivor under
``allocate_within_phase`` but landed on ``survivors[sims_run % 3]`` — the worst
of the three — under the round-robin it replaced.

Exit status is 1 if any combination disagrees, so this is usable as a gate.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import chess
import numpy as np

from tinyalphazero.config import MCTSConfig
from tinyalphazero.mcts.batch_runner import BatchRunner
from tinyalphazero.mcts.batch_runner import uniform_stub_batch as stub_batch
from tinyalphazero.mcts.search import Search, load_epd, uniform_stub_evaluator

REPO = Path(__file__).resolve().parents[1]

# Positions with enough branching that m < n_legal actually bites, plus the
# tactical ones where the solver and force-include interact with the schedule.
FENS = {
    "start": chess.STARTING_FEN,
    "midgame": "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
    "backrank": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
}

M_VALUES = (3, 5, 12, 16)
SIMS_VALUES = (6, 16, 31, 48)


def _board(epd_or_fen: str) -> chess.Board:
    board = chess.Board()
    if " bm " in epd_or_fen or ' id "' in epd_or_fen:
        board.set_epd(epd_or_fen)
    else:
        board.set_fen(epd_or_fen)
    return board


def root_visits_sync(fen: str, cfg: MCTSConfig, sims: int, seed: int) -> list[int]:
    result = Search(uniform_stub_evaluator, cfg, np.random.default_rng(seed)).run(
        _board(fen), None, sims
    )
    return list(result.visits)


def root_visits_batched(fen: str, cfg: MCTSConfig, sims: int, seed: int) -> list[int]:
    runner = BatchRunner(stub_batch, cfg, max_plies=1)
    list(runner.play_games(1, seeds=[seed], num_sims=sims, boards=[_board(fen)]))
    ((tree, root, _sims_run),) = runner.debug_trees.values()
    return list(tree.N[tree.edge_slice(root)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quiet", action="store_true", help="Only print disagreements")
    args = ap.parse_args(argv)

    fens = dict(FENS)
    suite = REPO / "tests" / "suites" / "mate_in_2.epd"
    if suite.exists():
        fens["mate_in_2[0]"] = load_epd(suite)[0][0]

    failures = 0
    for name, fen in fens.items():
        for m, sims in itertools.product(M_VALUES, SIMS_VALUES):
            if m > sims:
                continue  # halving_phase_budgets refuses m > num_sims by contract
            cfg = MCTSConfig(batch_leaves=1, root_noise=True, gumbel_m=m)
            sync = root_visits_sync(fen, cfg, sims, args.seed)
            batched = root_visits_batched(fen, cfg, sims, args.seed)
            equal = sync == batched
            failures += not equal
            if not equal or not args.quiet:
                print(
                    f"{name:<14} m={m:<3} sims={sims:<3} visits_equal={equal}"
                    + ("" if equal else f"\n    sync   ={sync}\n    batched={batched}")
                )

    print(f"\n{failures} disagreement(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
