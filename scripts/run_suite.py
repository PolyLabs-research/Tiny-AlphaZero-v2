#!/usr/bin/env python3
"""Solve a mate-in-N EPD suite with the search and report the gate line.

    python scripts/run_suite.py --suite tests/suites/mate_in_1.epd --sims 16 --stub
    python scripts/run_suite.py --suite tests/suites/mate_in_2.epd --sims 128 \\
        --ckpt runs/phase1_anchor.pt

Solved means the chosen move *is* a solution (delivers mate, or forces it), not
that it matches the recorded ``bm``. The result line names the evaluator, the
root mode, the search profile and the active σ constants, because these numbers
get pasted into gate documents and "77.0%" with no profile attached is exactly
the ambiguity BLOCKED-2026-08-08 was about.

This CLI used to live inside ``tinyalphazero.mcts.search`` as a ``main()``; it
is a script now because library code does not print (AGENTS.md §4,
REVIEW-2026-10-21 E11). The search itself is unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tinyalphazero.config import MCTSConfig
from tinyalphazero.mcts.search import network_evaluator, run_suite, uniform_stub_evaluator


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        "--proof-directed",
        action="store_true",
        help="Proof-directed reply expansion ON for this measurement "
        "(mcts.eval_proof_directed_expansion; self-play untouched).",
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

    inferred = 3 if "mate_in_3" in args.suite.name else 2 if "mate_in_2" in args.suite.name else 1
    mate_in = args.mate_in if args.mate_in else inferred
    # mate-in-3 is informational trajectory, not a gate (§8), so it has no
    # threshold of its own; 0.0 reports the number without implying a verdict.
    default_threshold = 0.0 if mate_in == 3 else 0.95 if mate_in == 2 else 1.0
    threshold = args.threshold if args.threshold else default_threshold

    cfg = MCTSConfig()
    if args.root_mode:
        cfg.root_mode = args.root_mode
    cfg.root_noise = bool(args.root_noise)
    cfg.eval_m = args.eval_m
    cfg.eval_c_visit = args.eval_c_visit
    cfg.eval_c_scale = args.eval_c_scale
    if args.proof_directed:
        cfg.eval_proof_directed_expansion = True
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
    profile = "selfplay(noise)" if cfg.root_noise else "eval(deterministic)"
    print(
        f"{args.suite.name}: {solved}/{total} solved at {args.sims} sims "
        f"({100 * frac:.1f}%) | eval={which} root={cfg.root_mode} "
        f"profile={profile} m={active.gumbel_m} "
        f"c_visit={active.c_visit:g} c_scale={active.c_scale:g} "
        f"proof_directed={active.proof_directed_expansion} "
        f"| gate >= {100 * threshold:.0f}%"
    )
    if failed:
        print(f"  failed: {', '.join(failed[:10])}{' …' if len(failed) > 10 else ''}")
    return 0 if frac >= threshold else 1


if __name__ == "__main__":
    sys.exit(main())
