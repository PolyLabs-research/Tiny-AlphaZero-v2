#!/usr/bin/env python3
"""Sweep the evaluation profile's σ constants and considered-set size.

Stage 2 of the resolution path in ``BLOCKED-2026-08-08-mate2-gate.md``. Stage 1
(the evaluation profile itself) moved the chunk 6 mate-in-2 gate from 0.890 to
0.940 and still fell short of 0.95, so exactly one bounded tuning pass is
permitted — under a firewall:

* the sweep runs on a **disjoint tuning suite**, never on a gate suite;
* the tuning suite is burned as tuning data the moment it is used;
* selection is by tuning-suite solve rate alone, ties broken toward the paper
  defaults (m=16, c_scale=1.0, c_visit=50);
* the gate suite is then touched **exactly once**, with the chosen config.

Re-picking after seeing the gate number is the overfit the firewall exists to
prevent, so this script deliberately has no way to read the gate suite: point it
at ``--suite`` and it will happily sweep whatever you give it, which is why the
tuning suite's ``meta.json`` records what it is for.

Usage:

    VIRTUAL_ENV=.venv .venv/bin/python scripts/sweep_eval_profile.py \\
        --suite runs/data/mate2_tune_500/mate_in_2.epd --sims 128 \\
        --ckpt runs/phase1_d/final.pt --workers 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

from tinyalphazero.config import MCTSConfig
from tinyalphazero.mcts.search import load_epd, network_evaluator, solve_positions

# The grid, pre-registered in BLOCKED-2026-08-08. Not widened after seeing
# results: a grid that grows once the first pass disappoints is a search over
# stopping rules, not a tuning pass.
C_VISIT = (25.0, 50.0, 100.0)
C_SCALE = (0.5, 1.0, 2.0)
EVAL_M = (16, 32)

# Tie-break target: minimal delta from the published Gumbel-AZ configuration.
DEFAULTS = {"eval_c_visit": 50.0, "eval_c_scale": 1.0, "eval_m": 16}

_WORKER: dict = {}


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - provenance is best-effort
        return "unknown"


def _init_worker(ckpt: str, device: str | None) -> None:
    """Load the network once per worker process, not once per position."""
    _WORKER["evaluate_fn"] = network_evaluator(ckpt, device)


def _run_shard(job: tuple[list[tuple[str, str]], int, dict, int]) -> tuple[int, int, list[str]]:
    positions, sims, cfg_kw, mate_in = job
    cfg = MCTSConfig(root_noise=False, **cfg_kw)
    return solve_positions(positions, sims, _WORKER["evaluate_fn"], cfg, mate_in=mate_in)


def _shard(positions: list, n: int) -> list[list]:
    """Contiguous shards. Position order never affects a deterministic search."""
    if n <= 1:
        return [positions]
    size = (len(positions) + n - 1) // n
    return [positions[i : i + size] for i in range(0, len(positions), size)]


def evaluate_config(
    positions: list[tuple[str, str]],
    sims: int,
    cfg_kw: dict,
    mate_in: int,
    pool: mp.pool.Pool | None,
    workers: int,
) -> tuple[float, int, int, list[str], float]:
    t0 = time.perf_counter()
    if pool is None:
        solved, total, failed = _run_shard((positions, sims, cfg_kw, mate_in))
    else:
        jobs = [(sh, sims, cfg_kw, mate_in) for sh in _shard(positions, workers)]
        parts = pool.map(_run_shard, jobs)
        solved = sum(p[0] for p in parts)
        total = sum(p[1] for p in parts)
        failed = [f for p in parts for f in p[2]]
    return solved / total if total else 0.0, solved, total, failed, time.perf_counter() - t0


def select(rows: list[dict]) -> dict:
    """Highest solve rate; ties broken toward the paper defaults.

    Distance is the count of fields that differ from the published
    configuration, then the raw field values, so the ordering is total and does
    not depend on dict iteration order.
    """

    def key(r: dict) -> tuple:
        deltas = sum(1 for k, v in DEFAULTS.items() if r[k] != v)
        return (-r["solve_rate"], deltas, r["eval_m"], r["eval_c_scale"], r["eval_c_visit"])

    return sorted(rows, key=key)[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suite",
        required=True,
        type=Path,
        help="EPD tuning suite. MUST NOT be a gate suite — see the module docstring.",
    )
    ap.add_argument("--ckpt", required=True, type=Path, help="Checkpoint to evaluate")
    ap.add_argument("--sims", type=int, default=128, help="Simulations per move")
    ap.add_argument(
        "--positions",
        type=int,
        default=None,
        help="Use only the first N positions — the coarse pass. Omit for the full suite.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes sharding the positions. Each loads its own copy of "
        "the network, so keep this modest on a unified-memory box. 1 runs "
        "in-process.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results.jsonl"),
        help="JSONL sink for the full table, one row per config, with git SHA.",
    )
    ap.add_argument(
        "--tag",
        default="eval_profile_sweep",
        help="Row tag written into the JSONL, e.g. eval_profile_sweep_coarse.",
    )
    ap.add_argument(
        "--time-one-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the default config alone first and project the full grid, so a "
        "grid that would overrun its budget is caught before it is spent.",
    )
    args = ap.parse_args(argv)

    positions = load_epd(args.suite)
    if args.positions:
        positions = positions[: args.positions]
    mate_in = 2 if "mate_in_2" in args.suite.name else 1
    sha = git_sha()

    grid = [
        {"eval_c_visit": cv, "eval_c_scale": cs, "eval_m": m}
        for cv, cs, m in itertools.product(C_VISIT, C_SCALE, EVAL_M)
    ]

    pool = None
    if args.workers > 1:
        ctx = mp.get_context("spawn")  # ROCm contexts do not survive fork
        pool = ctx.Pool(
            args.workers, initializer=_init_worker, initargs=(str(args.ckpt), None)
        )
    else:
        _init_worker(str(args.ckpt), None)

    try:
        print(
            f"sweep: {len(grid)} configs x {len(positions)} positions @ {args.sims} sims "
            f"| ckpt={args.ckpt} workers={args.workers} sha={sha[:8]}"
        )
        if args.time_one_first:
            _rate, _s, _t, _f, secs = evaluate_config(
                positions, args.sims, dict(DEFAULTS), mate_in, pool, args.workers
            )
            projected = secs * len(grid) / 3600.0
            print(
                f"  timing slice: default config {secs:.1f}s "
                f"→ full grid ≈ {projected:.2f} h"
            )
            if projected > 4.0:
                print(
                    "  ABORT: projected past the 4 h budget. Re-run with "
                    "--positions 200 --tag eval_profile_sweep_coarse for the coarse "
                    "pass, then refine the top 3 on the full suite."
                )
                return 3

        rows = []
        for i, cfg_kw in enumerate(grid, 1):
            rate, solved, total, failed, secs = evaluate_config(
                positions, args.sims, cfg_kw, mate_in, pool, args.workers
            )
            row = {
                "benchmark": args.tag,
                "git_sha": sha,
                "ckpt": str(args.ckpt),
                "suite": str(args.suite),
                "positions": total,
                "sims": args.sims,
                "profile": "eval",
                **cfg_kw,
                "solved": solved,
                "solve_rate": rate,
                "seconds": round(secs, 2),
            }
            rows.append(row)
            print(
                f"  [{i:2d}/{len(grid)}] m={cfg_kw['eval_m']:<3} "
                f"c_visit={cfg_kw['eval_c_visit']:<6g} c_scale={cfg_kw['eval_c_scale']:<4g} "
                f"→ {solved}/{total} ({100 * rate:.1f}%)  {secs:.0f}s"
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    best = select(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")
    print(
        f"SELECTED (tuning-suite solve, ties → paper defaults): "
        f"m={best['eval_m']} c_visit={best['eval_c_visit']:g} "
        f"c_scale={best['eval_c_scale']:g} → {100 * best['solve_rate']:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
