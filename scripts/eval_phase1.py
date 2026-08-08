#!/usr/bin/env python3
"""Phase 1 acceptance evaluation (spec §5).

Prints the acceptance table and exits non-zero if any gate fails.

Usage:
    python scripts/eval_phase1.py --ckpt runs/phase1_anchor.pt --data runs/data/eval_rand \\
        --mate1 runs/data/eval_mate1_1k

--data/--mate1 should point at held-out sets, not the training dirs. --mate1 is
required for a full verdict: two of the three gates are anchor-set metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from tinyalphazero.data import Phase1Dataset, assert_dataset_mode
from tinyalphazero.train import compute_metrics, load_checkpoint, unpack_legal_mask

# §5 acceptance gates, as revised by BLOCKED-2026-08-06-phase1-gates.md.
# The two unmasked-top-1 proxies were retired (not lowered) after diagnosis showed
# they over-demand on the one axis legality masking neutralizes; downstream
# consumption is masked-only (spec §6). They stay reported as informational.
GATES = {
    "legality_auc": ("≥ 0.995", lambda v: v >= 0.995),
    "wdl_anchor_mass": ("≥ 0.80", lambda v: v >= 0.80),
    "mate1_top8_masked": ("≥ 0.97", lambda v: v >= 0.97),
}

RETIRED = {
    "top1_legal_rate": "retired 2026-08-06 (was ≥ 0.99)",
    "mate1_solve_rate": "retired 2026-08-06 (was ≥ 0.90)",
    "top1_exact_rate": "informational",
}


@torch.no_grad()
def mate1_topk_masked(net, mate1_dir: Path, device, ks=(1, 3, 8, 16), bs: int = 256) -> dict[int, float]:
    """Masked top-k coverage of the mating move over the FULL anchor set.

    Full-set (not sampled) so the gated number carries no sampling noise; this is
    the same computation as scripts/diagnose_phase1.py's mate_topk.
    """
    ds = Phase1Dataset(mate1_dir)
    n = len(ds)
    tokens = torch.stack([ds[i]["tokens"] for i in range(n)])
    legals = torch.stack([ds[i]["legals"] for i in range(n)])
    target = torch.tensor([int(ds[i]["from_sq"]) * 64 + int(ds[i]["to_sq"]) for i in range(n)])

    pol = torch.cat([net(tokens[i:i + bs].to(device)).policy_ft.float().cpu() for i in range(0, n, bs)])
    pol = pol.reshape(n, 4096).masked_fill(~unpack_legal_mask(legals).reshape(n, 4096), float("-inf"))
    order = pol.argsort(dim=1, descending=True)
    return {k: (order[:, :k] == target[:, None]).any(1).float().mean().item() for k in ks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="Random-play dataset dir")
    parser.add_argument("--mate1", default=None, type=Path, help="Mate-in-1 dataset dir (optional)")
    parser.add_argument("--n-random", type=int, default=2000)
    # Full frozen 1K anchor set: gated metrics carry no sampling noise, and
    # mate1_solve_rate then equals mate1_top1_masked by construction.
    parser.add_argument("--n-mate1", type=int, default=1000)
    args = parser.parse_args()

    assert_dataset_mode(args.data, "random_games")
    if args.mate1 is not None:
        assert_dataset_mode(args.mate1, "mate_in_1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    net, meta = load_checkpoint(args.ckpt, device=device)
    net.eval()
    print(f"  step={meta.get('step', '?')}  phase={meta.get('phase', '?')}")

    print(f"\nComputing metrics (n_random={args.n_random}, n_mate1={args.n_mate1})...")
    metrics = compute_metrics(
        net,
        args.data,
        device,
        mate1_dir=args.mate1,
        n_random=args.n_random,
        n_mate1=args.n_mate1,
    )

    topk: dict[int, float] = {}
    if args.mate1 is not None:
        topk = mate1_topk_masked(net, args.mate1, device)
        metrics["mate1_top8_masked"] = topk[8]

    # ---- Print acceptance table ----
    print("\n" + "=" * 72)
    print(f"{'Metric':<28} {'Value':>10}  {'Gate':>12}  {'Pass':>6}")
    print("-" * 72)

    all_pass = True
    for key, (gate_str, gate_fn) in GATES.items():
        val = metrics.get(key, float("nan"))
        if val != val:  # NaN
            passed = False
            val_str = "   N/A"
        else:
            passed = gate_fn(val)
            val_str = f"{val:10.4f}"
        flag = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"  {key:<26} {val_str}  {gate_str:>12}  {flag:>6}")

    # Retired gates and informational metrics — reported, never gated.
    print("-" * 72)
    for key, note in RETIRED.items():
        if key in metrics:
            print(f"  {key:<26} {metrics[key]:10.4f}  {note}")
    for k in sorted(topk):
        if k not in (1, 8):  # k=1 is mate1_solve_rate above; k=8 is gated
            print(f"  {f'mate1_top{k}_masked':<26} {topk[k]:10.4f}  informational")

    print("=" * 72)

    if all_pass:
        print("\nAll gates PASSED — ready for Phase 2.")
        sys.exit(0)
    else:
        print("\nOne or more gates FAILED.")
        sys.exit(1)



if __name__ == "__main__":
    main()
