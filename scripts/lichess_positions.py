#!/usr/bin/env python3
"""Sample positions from a Lichess game database file into a ring to label (PREREG-p2_ac).

    python scripts/lichess_positions.py --pgn-zst <dir>/lichess_db_standard_rated_2016-01.pgn.zst \\
        --pgn-zst <dir>/lichess_db_standard_rated_2016-02.pgn.zst \\
        --out runs/_lichess/ring --max-rows 16000000 --workers 4

The files are streamed through ``zstdcat``, in the order given, as one sequence of games. Each game is kept or dropped on its headers alone
(``lichess.header_ok``: both players rated, blitz or slower, ended on the board); a kept
game gives up to ``--per-game`` positions from ``--first-ply`` on (``lichess.sample_positions``,
seeded by ``--seed`` and the game's index in the sequence, so a rerun over the same files
gives the same rows). Games are processed in order by ``--workers`` processes and written in
that order until ``--max-rows`` rows or the end of the file. ``lichess.json`` beside the ring
records the source's sha256, the filters and the counts. Positions are not de-duplicated
here: ``scripts/pool_positions.py`` keeps each position once across all its sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

HEADER = re.compile(r'^\[(\w+) "(.*)"\]\s*$')

_FILTERS = None


def _init(filters_d: dict) -> None:
    global _FILTERS
    from tinyalphazero.lichess import Filters

    _FILTERS = Filters(**filters_d)


def _work(item: tuple[int, str, int]) -> list[dict]:
    from tinyalphazero.lichess import sample_positions

    index, text, seed = item
    return sample_positions(text, _FILTERS, seed * 1_000_000_007 + index)


def games(stream):
    """(headers, text) for each game of a PGN text stream, in order."""
    headers: dict[str, str] = {}
    lines: list[str] = []
    in_moves = False
    for line in stream:
        if line.startswith("[Event ") and lines:
            yield headers, "".join(lines)
            headers, lines, in_moves = {}, [], False
        lines.append(line)
        if not in_moves:
            m = HEADER.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
            elif line.strip():
                in_moves = True
    if lines:
        yield headers, "".join(lines)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    from tinyalphazero.lichess import Filters

    defaults = Filters()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pgn-zst", required=True, type=Path, action="append", help="repeatable")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-rows", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-elo", type=int, default=defaults.min_elo)
    ap.add_argument("--min-base-s", type=int, default=defaults.min_base_s)
    ap.add_argument("--min-plies", type=int, default=defaults.min_plies)
    ap.add_argument("--first-ply", type=int, default=defaults.first_ply)
    ap.add_argument("--per-game", type=int, default=defaults.per_game)
    args = ap.parse_args(argv)

    from tinyalphazero.lichess import LICHESS_ITERATION, header_ok
    from tinyalphazero.provenance import git_sha
    from tinyalphazero.replay import ReplayBuffer

    if (args.out / "meta.json").exists():
        print(f"{args.out} exists; refusing to overwrite", file=sys.stderr)
        return 1
    filters = Filters(
        min_elo=args.min_elo, min_base_s=args.min_base_s, min_plies=args.min_plies,
        first_ply=args.first_ply, per_game=args.per_game,
    )  # fmt: skip
    t0 = time.time()
    ring = ReplayBuffer.create(args.out, capacity=args.max_rows)
    cmd = ["zstdcat", *[str(p) for p in args.pgn_zst]]  # concatenated, in order
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    counts = {"games_read": 0, "games_kept": 0, "games_with_rows": 0, "rows": 0}

    def kept():
        for index, (headers, text) in enumerate(games(proc.stdout)):
            counts["games_read"] += 1
            if header_ok(headers, filters):
                counts["games_kept"] += 1
                yield index, text, args.seed

    full = False
    with mp.get_context("spawn").Pool(args.workers, _init, (asdict(filters),)) as pool:
        for rows in pool.imap(_work, kept(), chunksize=256):
            if rows:
                counts["games_with_rows"] += 1
            for row in rows:
                ring.add(**row, iteration=LICHESS_ITERATION, mover_is_latest=False)
                counts["rows"] += 1
                if counts["rows"] >= args.max_rows:
                    full = True
                    break
            if full:
                break
            if counts["games_with_rows"] % 200_000 == 0 and rows:
                print(f"{counts} ({time.time() - t0:.0f} s)", flush=True)
        pool.terminate()
    proc.kill()
    ring.flush()
    record = {
        "sources": [{"file": str(p), "sha256": sha256_of(p)} for p in args.pgn_zst],
        "filters": asdict(filters),
        "seed": args.seed,
        "max_rows": args.max_rows,
        "stopped_at_max_rows": full,
        **counts,
        "iteration_tag": LICHESS_ITERATION,
        "seconds": round(time.time() - t0, 1),
        "git_sha": git_sha(),
    }
    (args.out / "lichess.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
