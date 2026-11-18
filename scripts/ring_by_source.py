#!/usr/bin/env python3
"""One source's rows of a labelled pool, as their own ring (PREREG-p2_ad).

    python scripts/ring_by_source.py --ring runs/p2_ac/teacher --tags runs/p2_ac/pool \\
        --source selfplay --out runs/p2_ad/ring_selfplay
    python scripts/ring_by_source.py --ring runs/p2_ac/teacher --tags runs/p2_ac/pool \\
        --source lichess --limit 8845607 --out runs/p2_ad/ring_lichess

A teacher ring's ``iteration`` field holds the teacher tag for every row, so each row's
source is read from the pool the ring was labelled from (``--tags``), whose rows are the
ring's in the same order: the two rings' boards are compared, all of them, before a row
is selected. The rows of ``--source`` (``lichess.source_rows``: the first ``--limit``,
in ring order) are copied, every field, into ``--out`` (``ReplayBuffer.from_rows``), with
the ring's ``teacher.json`` beside them and ``subset.json`` recording the selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

CHUNK = 2_000_000


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ring", required=True, type=Path, help="the labelled ring")
    ap.add_argument("--tags", required=True, type=Path, help="the pool it was labelled from")
    ap.add_argument("--source", required=True, choices=("lichess", "selfplay"))
    ap.add_argument("--limit", type=int, default=None, help="default: every row of the source")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    from tinyalphazero.lichess import LICHESS_ITERATION, source_rows
    from tinyalphazero.provenance import git_sha
    from tinyalphazero.replay import ReplayBuffer

    if (args.out / "meta.json").exists():
        print(f"{args.out} exists; remove it to rebuild", file=sys.stderr)
        return 1
    t0 = time.time()
    ring = ReplayBuffer.open(args.ring, readonly=True)
    pool = ReplayBuffer.open(args.tags, readonly=True)
    n = len(ring)
    if len(pool) != n:
        print(f"the ring holds {n} rows, the pool {len(pool)}", file=sys.stderr)
        return 1
    for s in range(0, n, CHUNK):
        if not np.array_equal(
            ring._arr["boards"][s : s + CHUNK], pool._arr["boards"][s : s + CHUNK]
        ):
            print(f"boards differ in rows {s}..{min(n, s + CHUNK) - 1}", file=sys.stderr)
            return 1
    tags = np.asarray(pool._arr["iteration"][:n])
    rows = source_rows(tags, args.source, args.limit)
    print(f"{args.source}: {len(rows)} rows of {n} (boards aligned)", flush=True)
    out = ReplayBuffer.from_rows(ring, rows, args.out)
    teacher = args.ring / "teacher.json"
    if teacher.exists():
        shutil.copy2(teacher, args.out / "teacher.json")
    record = {
        "ring": str(args.ring),
        "tags": str(args.tags),
        "source": args.source,
        "rule": f"iteration == {LICHESS_ITERATION}"
        if args.source == "lichess"
        else "iteration >= 0",
        "available": int(np.count_nonzero(tags == LICHESS_ITERATION))
        if args.source == "lichess"
        else int(np.count_nonzero(tags >= 0)),
        "limit": args.limit,
        "rows": len(out),
        "first_slot": int(rows[0]),
        "last_slot": int(rows[-1]),
        "slots_sha256": hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest(),
        "boards_aligned": True,
        "git_sha": git_sha(),
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "subset.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
