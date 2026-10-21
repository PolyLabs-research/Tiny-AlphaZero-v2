#!/usr/bin/env python3
"""Generate Phase 1 training data: random-play games, mate-in-1 and mate-in-2 anchors (spec §5)."""

from __future__ import annotations

import argparse
import inspect
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from tinyalphazero.data import (
    generate_mate_in_1,
    generate_mate_in_2,
    generate_random_games,
    write_dataset,
)


def _library_default(fn, name: str):
    """The generator's own default for ``name`` — read, not restated (AGENTS §4)."""
    return inspect.signature(fn).parameters[name].default


def load_exclusion_keys(paths: list[str]) -> set[str]:
    """Position keys to skip: dataset dirs and/or .epd suites.

    Anchors must never contain a position a gate reads. Filtering at generation
    time is the only place this can be fixed without touching the frozen eval
    sets (campaign rule 4) -- and it has to exist, because shallow mates form a
    small space that independently-seeded harvests collide inside.
    """
    import chess

    from tinyalphazero.data import Phase1Dataset, position_key
    from tinyalphazero.encoding import encode_board

    keys: set[str] = set()
    for raw in paths:
        path = Path(raw)
        if path.suffix == ".epd":
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    board = chess.Board()
                    board.set_epd(line)
                    keys.add(position_key(encode_board(board, 0)))
        else:
            ds = Phase1Dataset(path)
            keys.update(position_key(ds[i]["tokens"].numpy()) for i in range(len(ds)))
    return keys


def _mate2_shard(spec: tuple[int, int, frozenset, int]) -> list:
    """Harvest one shard of mate-in-2 positions. Top level so it is picklable."""
    from tinyalphazero.data import position_key

    count, seed, excluded, max_plies = spec
    rng = np.random.default_rng(seed)
    out = []
    if not excluded:
        return list(generate_mate_in_2(count, rng, max_plies=max_plies))
    # Over-draw and filter: the generator counts what it yields, so filtering
    # afterwards would come up short.
    for rec in generate_mate_in_2(count * 2, rng, max_plies=max_plies):
        if position_key(rec.tokens) not in excluded:
            out.append(rec)
            if len(out) >= count:
                break
    return out


def _generate_mate2(
    total: int,
    seed: int,
    workers: int,
    excluded: frozenset = frozenset(),
    max_plies: int | None = None,
) -> tuple[list, float, int]:
    """Harvest `total` mate-in-2 positions, sharded across processes.

    ~3.3 positions/sec/core measured, so 50K is ~4.3 h single-core -- past the 2 h
    threshold, hence the sharding. Each shard gets its own derived seed, so the
    output is reproducible from (seed, workers); both go into meta.json, because a
    different worker count gives a different (still valid) dataset.
    """
    if max_plies is None:
        max_plies = _library_default(generate_mate_in_2, "max_plies")
    workers = max(1, min(workers, os.cpu_count() or 1))
    base = total // workers
    counts = [base + (1 if i < total % workers else 0) for i in range(workers)]
    specs = [(c, seed * 10_000 + i, excluded, max_plies) for i, c in enumerate(counts) if c > 0]
    t0 = time.perf_counter()
    if len(specs) == 1:
        shards = [_mate2_shard(specs[0])]
    else:
        with mp.Pool(len(specs)) as pool:
            shards = pool.map(_mate2_shard, specs)
    elapsed = time.perf_counter() - t0
    return [r for shard in shards for r in shard], elapsed, workers


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 1 training data")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--games", type=int, metavar="N", help="Generate N random-play games")
    mode.add_argument(
        "--mate1", type=int, metavar="N", help="Generate N mate-in-1 anchor positions"
    )
    mode.add_argument(
        "--mate2",
        type=int,
        metavar="N",
        help="Generate N mate-in-2 anchor positions (forced mate in two; positions "
        "that already have a mate-in-1 are excluded). ~3.3 pos/sec/core, so this "
        "shards across --workers processes by default.",
    )
    parser.add_argument("--out", required=True, metavar="DIR", help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-plies",
        type=int,
        default=None,
        help="Game length cap for the random walk. --games and --mate2 accept it "
        f"(library defaults {_library_default(generate_random_games, 'max_plies')} / "
        f"{_library_default(generate_mate_in_2, 'max_plies')}); --mate1 fixes its own "
        "and rejects the flag.",
    )
    parser.add_argument(
        "--min-plies",
        type=int,
        default=None,
        help="Skip random games shorter than this (--games only; library default "
        f"{_library_default(generate_random_games, 'min_plies')}). The anchor "
        "harvesters have no such parameter and reject the flag.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATH",
        help="Dataset dir or .epd suite whose positions must NOT appear in the "
        "output. Repeatable. Use it to keep anchors disjoint from every held-out "
        "set and committed suite; tests/test_contamination.py gates this.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 8),
        help="Processes for --mate2 harvesting. Leaves headroom so the box stays "
        "usable; ignored by --games and --mate1.",
    )
    args = parser.parse_args()

    # Inert flags are worse than rejected ones (REVIEW-2026-10-21 E12): a dataset
    # whose meta.json records a ply cap the harvester never saw misdescribes itself.
    if args.mate1 is not None and (args.max_plies is not None or args.min_plies is not None):
        parser.error(
            "--max-plies/--min-plies do not apply to --mate1: generate_mate_in_1 fixes its "
            "walk at 150 plies (src/tinyalphazero/data.py) and takes neither parameter."
        )
    if args.mate2 is not None and args.min_plies is not None:
        parser.error(
            "--min-plies does not apply to --mate2: generate_mate_in_2 has no such parameter."
        )

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    t0 = time.perf_counter()

    excluded = frozenset(load_exclusion_keys(args.exclude)) if args.exclude else frozenset()
    if excluded:
        print(f"Excluding {len(excluded)} positions from {len(args.exclude)} source(s)")

    if args.games is not None:
        kw = {
            k: v
            for k, v in (("max_plies", args.max_plies), ("min_plies", args.min_plies))
            if v is not None
        }
        records = generate_random_games(args.games, rng, **kw)
        cfg = {
            "mode": "random_games",
            "num_games": args.games,
            "seed": args.seed,
            # The effective values: the flag if given, else the generator's own default.
            "max_plies": kw.get("max_plies", _library_default(generate_random_games, "max_plies")),
            "min_plies": kw.get("min_plies", _library_default(generate_random_games, "min_plies")),
        }
        meta = write_dataset(records, out_dir, generation_config=cfg)
        elapsed = time.perf_counter() - t0
        print(f"Generated {meta.n} positions from {args.games} games in {elapsed:.1f}s → {out_dir}")
    elif args.mate2 is not None:
        max_plies = (
            args.max_plies
            if args.max_plies is not None
            else _library_default(generate_mate_in_2, "max_plies")
        )
        records, harvest_s, workers = _generate_mate2(
            args.mate2, args.seed, args.workers, excluded, max_plies=max_plies
        )
        cfg = {
            "mode": "mate_in_2",
            "num_positions": args.mate2,
            "seed": args.seed,
            "workers": workers,
            "max_plies": max_plies,
            "positions_per_sec": round(len(records) / max(harvest_s, 1e-9), 2),
            "excluded_sources": list(args.exclude),
        }
        meta = write_dataset(iter(records), out_dir, generation_config=cfg)
        elapsed = time.perf_counter() - t0
        print(
            f"Generated {meta.n} mate-in-2 positions in {elapsed:.1f}s "
            f"({cfg['positions_per_sec']} pos/s across {workers} workers) → {out_dir}"
        )
    else:
        from tinyalphazero.data import position_key

        def _filtered_mate1():
            made = 0
            for rec in generate_mate_in_1(args.mate1 * 3, rng):
                if position_key(rec.tokens) in excluded:
                    continue
                yield rec
                made += 1
                if made >= args.mate1:
                    return

        records = _filtered_mate1() if excluded else generate_mate_in_1(args.mate1, rng)
        cfg = {
            "mode": "mate_in_1",
            "num_positions": args.mate1,
            "seed": args.seed,
            "excluded_sources": list(args.exclude),
        }
        meta = write_dataset(records, out_dir, generation_config=cfg)
        elapsed = time.perf_counter() - t0
        print(f"Generated {meta.n} mate-in-1 positions in {elapsed:.1f}s → {out_dir}")


if __name__ == "__main__":
    main()
