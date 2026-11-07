#!/usr/bin/env python3
"""Relabel a self-play ring's positions with Stockfish and write the teacher rings (PREREG-p2_t).

    python scripts/label_teacher.py --ring runs/p2_s_cont/replay --out runs/p2_t_teach/teacher \\
        --holdout-out runs/p2_t_teach/teacher_holdout --holdout 5000

Positions: each distinct position of the source ring once, minus every position of the
book, the start position and the mate suites, minus rows that repeat a position of
their game (``teacher.select_rows``). A seeded ``--holdout`` of them is labelled the
same way but written to a separate ring that training never reads; it measures how far
a network has moved toward the teacher (``scripts/teacher_agreement.py``).

Labelling runs one single-threaded Stockfish per worker, as many workers as the cgroup
allows (``teacher.available_cpus``), each appending to its own shard under
``<out>_labels/``; a rerun skips labelled rows. The rings are written once every row is
labelled, the training ring's outcome model fitted on its own rows and reused for the
held-out ring.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tinyalphazero.opponents import stockfish_path
from tinyalphazero.provenance import git_sha, sha256_file
from tinyalphazero.replay import ReplayBuffer
from tinyalphazero.teacher import (
    TeacherSettings,
    available_cpus,
    build_teacher_ring,
    engine_analyse,
    exclusion_keys,
    label_rows,
    open_engine,
    read_labels,
    select_rows,
)

REPO = Path(__file__).resolve().parents[1]
EXCLUDE = [
    "tests/suites/openings_4ply.epd",
    "tests/suites/openings_startpos.epd",
    "tests/suites/mate_in_1.epd",
    "tests/suites/mate_in_2.epd",
    "tests/suites/mate_in_3.epd",
]


def _worker(job: tuple[str, list[int], dict, str, int]) -> int:
    ring_dir, rows, settings_d, shard, k = job
    settings = TeacherSettings(**settings_d)
    ring = ReplayBuffer.open(ring_dir, readonly=True)
    engine = open_engine(settings.stockfish, settings.hash_mb)
    try:
        return label_rows(ring._arr["boards"], rows, engine_analyse(engine), settings, Path(shard))
    finally:
        engine.quit()


def stockfish_version(path: str) -> str:
    out = subprocess.run([path], input="quit\n", capture_output=True, text=True, timeout=30)
    return out.stdout.splitlines()[0].strip() if out.stdout else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ring", required=True, type=Path, help="source self-play ring (read-only)")
    ap.add_argument("--out", required=True, type=Path, help="teacher ring to create")
    ap.add_argument("--holdout-out", required=True, type=Path)
    ap.add_argument("--holdout", type=int, default=5000)
    ap.add_argument("--nodes", type=int, default=100_000)
    ap.add_argument("--multipv", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=50.0, help="centipawns")
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0, help="0: the cgroup's CPU allowance")
    ap.add_argument("--seed", type=int, default=0, help="the held-out split")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="label a seeded sample of N selected rows (a smoke test)",
    )
    ap.add_argument("--stockfish", default="")
    args = ap.parse_args(argv)

    for path in (args.out, args.holdout_out):
        if (path / "teacher.json").exists():
            print(f"{path} is already a finished teacher ring", file=sys.stderr)
            return 1
    sf = stockfish_path(args.stockfish)
    if sf is None:
        print("no Stockfish binary", file=sys.stderr)
        return 1
    settings = TeacherSettings(
        nodes=args.nodes, multipv=args.multipv, temperature_cp=args.temperature,
        hash_mb=args.hash, stockfish=sf, stockfish_version=stockfish_version(sf),
    )  # fmt: skip

    source = ReplayBuffer.open(args.ring, readonly=True)
    n = len(source)
    excluded = exclusion_keys([REPO / p for p in EXCLUDE])
    rows, counts = select_rows(source._arr["boards"][:n], excluded)
    rng = np.random.default_rng(args.seed)
    if args.limit:
        rows = np.sort(rng.choice(rows, size=min(args.limit, len(rows)), replace=False))
    hold_mask = np.zeros(len(rows), dtype=bool)
    hold_mask[rng.choice(len(rows), size=min(args.holdout, len(rows) // 10), replace=False)] = True
    train_rows, hold_rows = rows[~hold_mask], rows[hold_mask]
    print(
        f"source {args.ring}: {n} rows, total_written {source.total_written}; {counts}", flush=True
    )
    print(f"label {len(rows)}: train {len(train_rows)}, held out {len(hold_rows)}", flush=True)

    workers = args.workers or available_cpus()
    labels_dir = args.out.parent / f"{args.out.name}_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(args.ring), [int(r) for r in rows[k::workers]], asdict(settings),
         str(labels_dir / f"shard_{k:03d}.jsonl"), k)
        for k in range(workers)
    ]  # fmt: skip
    t0 = time.time()
    print(f"{workers} workers, {settings.nodes} nodes, multipv {settings.multipv}", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        done = sum(pool.map(_worker, jobs, chunksize=1))
    labels = read_labels(sorted(labels_dir.glob("shard_*.jsonl")))
    missing = [int(r) for r in rows if int(r) not in labels]
    print(f"labelled {done} this run in {time.time() - t0:.0f} s; {len(labels)} in all", flush=True)
    if missing:
        print(
            f"{len(missing)} rows unlabelled, e.g. {missing[:5]}; rerun to finish", file=sys.stderr
        )
        return 1

    provenance = {
        "source_ring": str(args.ring),
        "source_total_written": int(source.total_written),
        "selection": counts,
        "excluded_suites": {p: sha256_file(REPO / p) for p in EXCLUDE},
        "holdout": {"seed": args.seed, "rows": len(hold_rows)},
        "git_sha": git_sha(),
        "labelling_seconds": round(time.time() - t0, 1),
        "workers": workers,
    }
    meta = build_teacher_ring(
        source, labels, train_rows, args.out, settings, {**provenance, "split": "train"}
    )
    model = (meta["outcome_model"]["a"], meta["outcome_model"]["s"])
    build_teacher_ring(
        source, labels, hold_rows, args.holdout_out, settings,
        {**provenance, "split": "holdout"}, model=model,
    )  # fmt: skip
    print(json.dumps({"rows": meta["rows"], "outcome_model": meta["outcome_model"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
