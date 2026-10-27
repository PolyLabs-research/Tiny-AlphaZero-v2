#!/usr/bin/env python3
"""L2, the contamination check: rows of a run's replay ring equal to a frozen-book position.

PREREG-p2_h registered the count (expected 0) and RUNLOG-p2_h §2 found 4 with a
script that was never committed; RUNLOG-p2_i §2 reports it per arm (2 and 3).
This is the committed instrument: ``tinyalphazero.evaluate.book_rows_in_ring``,
which keys on the position proper (squares, side, castling, en passant) and reads
the ring read-only. A full 500,000-row ring takes about half a minute.

    python scripts/ring_book_rows.py runs/p2_i_cont runs/p2_i_row
    python scripts/ring_book_rows.py runs/p2_h --json runs/p2_h/book_rows.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tinyalphazero.evaluate import book_rows_in_ring, load_openings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", type=Path, help="run directories, each with a replay/ ring")
    ap.add_argument(
        "--openings",
        default=None,
        help="EPD book, repo-relative (default: the frozen 4-ply book the ladder plays)",
    )
    ap.add_argument("--json", type=Path, default=None, help="also write every result here")
    args = ap.parse_args(argv)
    book = load_openings(args.openings) if args.openings else None
    results: dict[str, dict] = {}
    for run in args.runs:
        ring = run / "replay"
        if not (ring / "meta.json").exists():
            print(f"{run}: no replay ring", file=sys.stderr)
            return 2
        res = book_rows_in_ring(ring, book)
        results[str(run)] = res
        print(
            f"{run}: {len(res['book_rows'])} book row(s) in {res['rows']:,} "
            f"(total_written {res['total_written']:,}); {res['rows_at_initial_position']:,} at the "
            f"initial position; {res['distinct_positions']:,} distinct positions"
        )
        for row, ident, iteration in res["book_rows"]:
            print(f"  row {row}: {ident} (iteration {iteration})")
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
