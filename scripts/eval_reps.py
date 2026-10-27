#!/usr/bin/env python3
"""Repetitions of a nondeterministic rung, and the verdict over them (p2_d).

The skill-0 Stockfish rung randomizes its own play (RUNLOG-p2_d.md §2), so a
single ladder row of it is one draw from a distribution and two such rows cannot
be compared. p2_d's registered statistic replays the rung N times per checkpoint
and compares *per-opening rep-means* with a paired bootstrap over openings
(PREREG-p2_d.md Addendum 2). This is the harness that produced
``runs/p2_{c,d}/reps_iter0020_r{1,2,3}.jsonl`` and ``runs/p2_d/p2_verdict_reps.json``;
it was run from an uncommitted script at the time and is committed now
(REVIEW-2026-10-21 P5).

    # three replays of the skill-0 rung against a checkpoint
    python scripts/eval_reps.py run --ckpt runs/p2_c/ckpt_iter0020.pt \\
        --anchor stockfish:0 --reps 3 --out-dir runs/p2_c

    # the verdict: b against a, over the rows the runs above wrote
    python scripts/eval_reps.py verdict --a p2_c runs/p2_c/reps_iter0020_r*.jsonl \\
        --b p2_d runs/p2_d/reps_iter0020_r*.jsonl --out runs/p2_d/p2_verdict_reps.json

Every row is written ``role=diagnostic`` (no gate reads a single rep) with
``nondeterministic`` as the opponent declares it. The generation side cannot be
reproduced bit-for-bit — that is the whole point of repeating it — but the
``verdict`` side is a pure function of the rows and is pinned by a test against
the committed artefacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tinyalphazero import logschema
from tinyalphazero.config import load_config
from tinyalphazero.evaluate import (
    format_ladder_event,
    one_sample_rep_interval,
    paired_rep_verdict,
    run_ladder_pass,
)


def _report(kind: str, payload: dict) -> None:
    line = format_ladder_event(kind, payload)
    if line is not None:
        print(line, flush=True)


def _iteration_of(ckpt: Path) -> int | None:
    stem = ckpt.stem  # ckpt_iter0020
    digits = stem.rsplit("iter", 1)[-1]
    return int(digits) if digits.isdigit() else None


def cmd_run(args: argparse.Namespace) -> int:
    import torch

    config_path = args.config
    if config_path is None and (args.ckpt.parent / "config.yaml").exists():
        config_path = str(args.ckpt.parent / "config.yaml")
    cfg = load_config(config_path)
    if args.openings is not None:
        # A different book for this measurement only; the run's config.yaml is untouched.
        cfg.eval.openings_path = args.openings
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    iteration = args.iteration if args.iteration is not None else _iteration_of(args.ckpt)
    out_dir = args.out_dir or args.ckpt.parent
    tag = f"iter{iteration:04d}" if iteration is not None else args.ckpt.stem
    for rep in range(1, args.reps + 1):
        out = out_dir / f"reps_{tag}_r{rep}.jsonl"
        print(f"rep {rep}/{args.reps} -> {out}")
        rows = run_ladder_pass(
            args.ckpt,
            cfg,
            device,
            out=out,
            anchors=args.anchor,
            iteration=iteration,
            seed=args.seed,
            games=args.games,
            stockfish=args.stockfish,
            suites=False,
            role=logschema.ROLE_DIAGNOSTIC,
            produced=logschema.PRODUCED_OFFLINE,
            on_event=_report,
        )
        if not rows:
            print("  (already in the log, or the rung was skipped)")
    return 0


def _read_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        for row in logschema.read_jsonl(p):
            if row.get("row_type") == logschema.MATCH_ROW:
                logschema.validate_ladder(row)
                rows.append(row)
    return rows


def cmd_verdict(args: argparse.Namespace) -> int:
    a_name, *a_paths = args.a
    b_name, *b_paths = args.b
    a_rows, b_rows = _read_rows(a_paths), _read_rows(b_paths)
    verdict = paired_rep_verdict(
        a_rows, b_rows, a_name, b_name, resamples=args.resamples, seed=args.seed
    )
    text = json.dumps(verdict, indent=2)
    print(text)
    if args.out is not None:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    return 0 if verdict["excludes_zero"] else 1


def cmd_m4(args: argparse.Namespace) -> int:
    name, *paths = args.rows
    stat = one_sample_rep_interval(
        _read_rows(paths), name, resamples=args.resamples, seed=args.seed
    )
    stat["bar"] = args.bar
    stat["met"] = bool(stat["ci95"][0] > args.bar)
    text = json.dumps(stat, indent=2)
    print(text)
    if args.out is not None:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    return 0 if stat["met"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="replay one rung N times against a checkpoint")
    run.add_argument("--ckpt", required=True, type=Path)
    run.add_argument("--anchor", default="stockfish:0", help="one opponent spec")
    run.add_argument("--reps", type=int, default=3)
    run.add_argument("--seed", type=int, default=None, help="default: the config's seed")
    run.add_argument("--games", type=int, default=None, help="default: eval.anchor_games")
    run.add_argument(
        "--iteration", type=int, default=None, help="default: parsed from the file name"
    )
    run.add_argument("--config", default=None, help="default: config.yaml beside --ckpt")
    run.add_argument("--out-dir", type=Path, default=None, help="default: beside --ckpt")
    run.add_argument("--device", default=None)
    run.add_argument("--stockfish", default="", help="default: eval.stockfish_path")
    run.add_argument(
        "--openings",
        default=None,
        help="EPD book, repo-relative (default: eval.openings_path, the frozen 4-ply book)",
    )
    run.set_defaults(fn=cmd_run)

    ver = sub.add_parser("verdict", help="paired bootstrap over per-opening rep-means, b - a")
    ver.add_argument("--a", nargs="+", required=True, metavar="NAME ROWS", help="label then rows")
    ver.add_argument("--b", nargs="+", required=True, metavar="NAME ROWS", help="label then rows")
    ver.add_argument("--resamples", type=int, default=10_000)  # reproduces p2_verdict_reps.json
    ver.add_argument("--seed", type=int, default=0)
    ver.add_argument("--out", type=Path, default=None)
    ver.set_defaults(fn=cmd_verdict)

    m4 = sub.add_parser(
        "m4", help="one-sample bootstrap of the mean-over-reps book score against a bar (M4)"
    )
    m4.add_argument("--rows", nargs="+", required=True, metavar="NAME ROWS", help="label then rows")
    m4.add_argument("--bar", type=float, default=0.675, help="RUNLOG-p2_g §4's bar")
    m4.add_argument("--resamples", type=int, default=10_000)
    m4.add_argument("--seed", type=int, default=0)
    m4.add_argument("--out", type=Path, default=None)
    m4.set_defaults(fn=cmd_m4)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
