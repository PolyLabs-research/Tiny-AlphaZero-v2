#!/usr/bin/env python3
"""Phase 1 acceptance evaluation (spec §5).

Prints the acceptance table and exits non-zero if any gate fails.

Usage:
    python scripts/eval_phase1.py --ckpt runs/phase1_anchor.pt --data runs/data/eval_rand \\
        --mate1 runs/data/eval_mate1_1k

--data/--mate1 should point at held-out sets, not the training dirs. --mate1 is
required for a full verdict: two of the three gates are anchor-set metrics, and
without it they are reported as "not measured" and the verdict FAILS — they are
never computed on positions generated on the fly (REVIEW-2026-10-21 E7).
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch

from tinyalphazero.data import Phase1Dataset, assert_dataset_mode
from tinyalphazero.train import MATE1_TOPK, compute_metrics, load_checkpoint

# Gates that read the mate-in-1 anchor set: without --mate1 they are not measured.
ANCHOR_GATES = ("wdl_anchor_mass", "mate1_top8_masked")

# §5 acceptance gates, as revised by BLOCKED-2026-08-06-phase1-gates.md.
# The two unmasked-top-1 proxies were retired (not lowered) after diagnosis showed
# they over-demand on the one axis legality masking neutralizes; downstream
# consumption is masked-only (spec §6). They stay reported as informational.
GATES = {
    "legality_auc": ("≥ 0.995", lambda v: v >= 0.995),
    "wdl_anchor_mass": ("≥ 0.80", lambda v: v >= 0.80),
    "mate1_top8_masked": ("≥ 0.97", lambda v: v >= 0.97),
    # Round d (BLOCKED-2026-08-08). Gated at k=16 because that is Gumbel's
    # considered-set size: a key move outside the prior's top-16 is never
    # simulated at the Phase 2 root. Only applied when --eval-mate2 is given.
    "mate2_top16_coverage": ("≥ 0.95", lambda v: v >= 0.95),
}

RETIRED = {
    "mate2_solve_rate": "informational (top-1)",
    "top1_legal_rate": "retired 2026-08-06 (was ≥ 0.99)",
    "mate1_solve_rate": "retired 2026-08-06 (was ≥ 0.90)",
    "top1_exact_rate": "informational",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="Random-play dataset dir")
    parser.add_argument("--mate1", default=None, type=Path, help="Mate-in-1 dataset dir (optional)")
    parser.add_argument(
        "--eval-mate2",
        default=None,
        type=Path,
        help="Held-out mate-in-2 dir. Enables the round-d mate2_top16_coverage gate; "
        "without it that gate is not evaluated and the verdict covers the others only.",
    )
    parser.add_argument("--n-random", type=int, default=2000)
    # Full frozen 1K anchor set: gated metrics carry no sampling noise, and
    # mate1_solve_rate then equals mate1_top1_masked by construction.
    parser.add_argument("--n-mate1", type=int, default=1000)
    parser.add_argument(
        "--n-mate2",
        type=int,
        default=inspect.signature(compute_metrics).parameters["n_mate2"].default,
        help="Mate-in-2 positions sampled from --eval-mate2 (default: the library's)",
    )
    args = parser.parse_args()

    assert_dataset_mode(args.data, "random_games")
    if args.mate1 is not None:
        assert_dataset_mode(args.mate1, "mate_in_1")
    if args.eval_mate2 is not None:
        assert_dataset_mode(args.eval_mate2, "mate_in_2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    net, meta = load_checkpoint(args.ckpt, device=device)
    net.eval()
    print(f"  step={meta.get('step', '?')}  phase={meta.get('phase', '?')}")

    # The denominators actually used: compute_metrics samples min(n, len(dataset)).
    n_random = min(args.n_random, len(Phase1Dataset(args.data)))
    n_mate1 = min(args.n_mate1, len(Phase1Dataset(args.mate1))) if args.mate1 is not None else 0
    n_mate2 = (
        min(args.n_mate2, len(Phase1Dataset(args.eval_mate2))) if args.eval_mate2 is not None else 0
    )
    print(
        f"\nComputing metrics (n_random={n_random}, n_mate1={n_mate1}"
        + (f", n_mate2={n_mate2}" if args.eval_mate2 is not None else "")
        + ")..."
    )
    if args.mate1 is None:
        print(
            "  --mate1 absent: the anchor gates are NOT measured (and are not computed "
            "on generated positions). Pass a held-out mate-in-1 dir for a verdict."
        )
    metrics = compute_metrics(
        net,
        args.data,
        device,
        mate1_dir=args.mate1,
        n_random=args.n_random,
        # n_mate1=0 with mate1_dir=None: the library would otherwise generate
        # 200 positions with a fixed rng and score them as if they were the set.
        n_mate1=args.n_mate1 if args.mate1 is not None else 0,
        mate2_dir=args.eval_mate2,
        n_mate2=args.n_mate2,
    )

    # ---- Print acceptance table ----
    print("\n" + "=" * 72)
    print(f"{'Metric':<28} {'Value':>10}  {'Gate':>12}  {'Pass':>6}")
    print("-" * 72)

    all_pass = True
    not_measured: list[str] = []
    for key, (gate_str, gate_fn) in GATES.items():
        val = metrics.get(key, float("nan"))
        if key == "mate2_top16_coverage" and args.eval_mate2 is None:
            continue  # not requested; do not fail a gate that was not asked for
        if key in ANCHOR_GATES and args.mate1 is None:
            not_measured.append(key)
            passed = False
            val_str = "not measured"
        elif val != val:  # NaN
            passed = False
            val_str = "   N/A"
        else:
            passed = gate_fn(val)
            val_str = f"{val:10.4f}"
        flag = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"  {key:<26} {val_str:>12}  {gate_str:>12}  {flag:>6}")

    # Retired gates and informational metrics — reported, never gated.
    print("-" * 72)
    for key, note in RETIRED.items():
        if key in metrics and metrics[key] == metrics[key]:
            print(f"  {key:<26} {metrics[key]:10.4f}  {note}")
    for k in MATE1_TOPK:
        key = f"mate1_top{k}_masked"
        if k not in (1, 8) and key in metrics and metrics[key] == metrics[key]:
            # k=1 is mate1_solve_rate above; k=8 is gated
            print(f"  {key:<26} {metrics[key]:10.4f}  informational")

    print("=" * 72)

    if all_pass:
        print("\nAll gates PASSED — ready for Phase 2.")
        sys.exit(0)
    if not_measured:
        print(
            f"\nFAILED: {', '.join(not_measured)} not measured — no --mate1 held-out "
            "anchor set was given, so there is no verdict."
        )
    else:
        print("\nOne or more gates FAILED.")
    sys.exit(1)


if __name__ == "__main__":
    main()
