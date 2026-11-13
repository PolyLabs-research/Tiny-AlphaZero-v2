#!/usr/bin/env python3
"""Pool the positions of several self-play rings into one ring to label (DIAG capacity-or-data).

    python scripts/pool_positions.py --out runs/diag_cod/pool --limit 3524304 \\
        --ring runs/p2_t_teach/replay --ring runs/p2_t_teach_down/replay ... \\
        --exclude-ring runs/p2_u_fresh/teacher --exclude-ring runs/p2_u_fresh/teacher_holdout

Each ring's rows are selected as ``scripts/label_teacher.py`` selects them (the book, the
start position and the mate suites out; rows that repeat a position of their own game out;
each position once), with one "already taken" set across all rings, in the order given.
Every position of an ``--exclude-ring`` is out as well. The kept rows are shuffled with
``--seed`` and cut to ``--limit``, and written, every field copied, to ``--out`` in that
order: any prefix of the pool is a uniform sample of it (``teacher.pool_rows``,
``teacher.write_pool``). ``pool.json`` beside it records the sources and the counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUITES = [
    "tests/suites/openings_4ply.epd",
    "tests/suites/openings_startpos.epd",
    "tests/suites/mate_in_1.epd",
    "tests/suites/mate_in_2.epd",
    "tests/suites/mate_in_3.epd",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ring", action="append", required=True, type=Path, help="source ring")
    ap.add_argument(
        "--exclude-ring", action="append", default=[], type=Path, help="positions to leave out"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from tinyalphazero.provenance import git_sha, sha256_file
    from tinyalphazero.replay import ReplayBuffer
    from tinyalphazero.teacher import exclusion_keys, pool_rows, position_key, write_pool

    if (args.out / "meta.json").exists():
        print(f"{args.out} exists; refusing to overwrite a pool", file=sys.stderr)
        return 1
    excluded = exclusion_keys([REPO / p for p in SUITES])
    for path in args.exclude_ring:
        ring = ReplayBuffer.open(path, readonly=True)
        boards = ring._arr["boards"]
        excluded.update(position_key(boards[i]) for i in range(len(ring)))
    rings = [ReplayBuffer.open(p, readonly=True) for p in args.ring]
    pairs, counts = pool_rows(
        [r._arr["boards"][: len(r)] for r in rings], excluded, args.seed, args.limit
    )
    available = sum(c["selected"] for c in counts)
    print(f"{available} distinct positions across {len(rings)} rings; pooling {len(pairs)}")
    write_pool(rings, pairs, args.out)
    record = {
        "sources": [
            {"ring": str(p), "total_written": int(r.total_written), **c}
            for p, r, c in zip(args.ring, rings, counts, strict=True)
        ],
        "exclude_rings": [str(p) for p in args.exclude_ring],
        "excluded_suites": {p: sha256_file(REPO / p) for p in SUITES},
        "available": available,
        "rows": int(len(pairs)),
        "seed": args.seed,
        "limit": args.limit,
        "git_sha": git_sha(),
    }
    (args.out / "pool.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("available", "rows", "seed")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
