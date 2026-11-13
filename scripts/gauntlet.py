"""Gauntlet: the network against engines with published ratings (the public benchmark).

    # play (resumable: finished games in <out>/games.jsonl are skipped)
    .venv/bin/python scripts/gauntlet.py play --ckpt runs/<run>/ckpt_iter0010.pt \
        --opponents sf1500,sf1900,hoichess --openings 100 --tc 60+0.6 \
        --concurrency 2 --out runs/bench/<name>

    # rate (anchored maximum-likelihood rating, CI over opening pairs)
    .venv/bin/python scripts/gauntlet.py rate --out runs/bench/<name>

Opponents:
  sf<elo>                 Stockfish (--stockfish or PATH), UCI_LimitStrength on at
                          UCI_Elo=<elo>, Threads 1, Hash 16. Stockfish 16 fits its
                          UCI_Elo to CCRL (goldfish 1.13 = 2000) at TC 60+0.6.
  hoichess                HoiChess 0.22.0 (Ubuntu package, extracted under
                          runs/bench/engines/hoichess), xboard; CCRL Blitz 1806.
  name=proto:cmd@rating   any other engine: proto uci|xboard, cmd a shell-quoted
                          command line, rating its published rating.

Each opening of the book is played twice, the network taking each colour once
(the ladder's pairing). The network is ``scripts/uci_engine.py`` as a subprocess:
the evaluation profile at the run's ``eval_match_sims``, clocks ignored. See
``tinyalphazero.gauntlet`` for the game rules and the rating model.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

from tinyalphazero.gauntlet import (
    Clock,
    bootstrap_rating,
    families_agree,
    performance,
    play_game,
)

REPO = Path(__file__).resolve().parent.parent
HOICHESS = REPO / "runs/bench/engines/hoichess/usr/games/hoichess"
# CCRL Blitz (2'+1" on an i7-4770K equivalent), list of 2026-11-07: 1806 +/- 20, 936 games.
HOICHESS_RATING = 1806.0


@dataclass(frozen=True)
class EngineSpec:
    name: str
    proto: str  # "uci" | "xboard"
    cmd: tuple[str, ...]
    rating: float
    options: tuple[tuple[str, object], ...] = ()

    def open(self) -> chess.engine.SimpleEngine:
        popen = (
            chess.engine.SimpleEngine.popen_uci
            if self.proto == "uci"
            else chess.engine.SimpleEngine.popen_xboard
        )
        engine = popen(list(self.cmd), timeout=120)
        if self.options:
            engine.configure(dict(self.options))
        return engine


def parse_opponent(text: str, stockfish: str) -> EngineSpec:
    if text.startswith("sf") and text[2:].isdigit():
        elo = int(text[2:])
        return EngineSpec(
            name=text,
            proto="uci",
            cmd=(stockfish,),
            rating=float(elo),
            options=(("UCI_LimitStrength", True), ("UCI_Elo", elo), ("Threads", 1), ("Hash", 16)),
        )
    if text == "hoichess":
        return EngineSpec("hoichess", "xboard", (str(HOICHESS), "-x"), HOICHESS_RATING)
    if "=" in text and "@" in text:
        name, rest = text.split("=", 1)
        proto_cmd, rating = rest.rsplit("@", 1)
        proto, cmd = proto_cmd.split(":", 1)
        if proto not in ("uci", "xboard"):
            raise ValueError(f"{text}: protocol must be uci or xboard")
        return EngineSpec(name, proto, tuple(shlex.split(cmd)), float(rating))
    raise ValueError(f"unrecognised opponent {text!r} (see --help)")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cmd_play(args: argparse.Namespace) -> int:
    from tinyalphazero.config import config_fingerprint, load_config
    from tinyalphazero.evaluate import load_openings

    logging.getLogger("chess.engine").setLevel(logging.ERROR)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    games_path, pgn_path = out / "games.jsonl", out / "games.pgn"

    config_path = args.config
    if config_path is None and (args.ckpt.parent / "config.yaml").exists():
        config_path = str(args.ckpt.parent / "config.yaml")
    cfg = load_config(config_path)
    sims = args.sims if args.sims is not None else cfg.eval.eval_match_sims
    ours_cmd = [
        sys.executable,
        str(REPO / "scripts/uci_engine.py"),
        "--ckpt",
        str(args.ckpt),
        "--device",
        args.device,
        "--sims",
        str(sims),
    ]
    if args.config:
        ours_cmd += ["--config", args.config]
    ours_spec = EngineSpec("tinyalphazero", "uci", tuple(ours_cmd), float("nan"))
    opponents = [parse_opponent(t, args.stockfish) for t in args.opponents.split(",")]
    clock = Clock.parse(args.tc)
    book = load_openings()[: args.openings]

    done: set[tuple[str, str, bool]] = set()
    if games_path.exists():
        for line in games_path.read_text().splitlines():
            row = json.loads(line)
            done.add((row["opponent"], row["opening"], row["ours_white"]))

    tasks = []
    for spec in opponents:
        for i, (epd, ident) in enumerate(book):
            for ours_white in (True, False):
                if (spec.name, ident or epd, ours_white) not in done:
                    tasks.append((spec, i, epd, ident or epd, ours_white))
    meta = {
        "checkpoint": str(args.ckpt),
        "config": config_path,
        "config_fingerprint": config_fingerprint(cfg),
        "sims": sims,
        "device": args.device,
        "tc": clock.pgn(),
        "git_sha": git_sha(),
    }
    print(
        f"gauntlet: {len(tasks)} games to play ({len(done)} already in {games_path}); "
        f"{meta['checkpoint']} sims {sims} on {args.device}; tc {clock.pgn()}; "
        f"concurrency {args.concurrency}",
        flush=True,
    )

    local = threading.local()
    lock = threading.Lock()
    # Every engine any worker opens, so all are quit at the end: python-chess keeps
    # a non-daemon thread per engine, and one left open keeps the process alive
    # after the last game (seen in the smoke run, 2026-11-12).
    opened: list[chess.engine.SimpleEngine] = []

    def engines_for(
        spec: EngineSpec,
    ) -> tuple[chess.engine.SimpleEngine, chess.engine.SimpleEngine]:
        pool = getattr(local, "pool", None)
        if pool is None:
            pool = local.pool = {}
        for s in (ours_spec, spec):
            if s.name not in pool:
                pool[s.name] = s.open()
                with lock:
                    opened.append(pool[s.name])
        return pool[ours_spec.name], pool[spec.name]

    def drop(spec: EngineSpec) -> None:
        engine = getattr(local, "pool", {}).pop(spec.name, None)
        if engine is not None:
            try:
                engine.quit()
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError):
                engine.close()

    def run(task: tuple) -> dict:
        spec, i, epd, ident, ours_white = task
        board = chess.Board()
        board.set_epd(epd)
        ours, theirs = engines_for(spec)
        key = (spec.name, ident, ours_white, time.monotonic_ns())
        rec = play_game(ours, theirs, board.fen(), ours_white, clock, key)
        if rec.termination in ("error", "hung", "hung_draw"):
            # Whoever erred or hung, restart both so the next game starts clean.
            drop(spec)
            drop(ours_spec)
        row = {
            "row_type": "gauntlet_game",
            "opponent": spec.name,
            "opponent_rating": spec.rating,
            "opening": ident,
            "opening_index": i,
            "ours_white": ours_white,
            "score": rec.score,
            "result": rec.result,
            "termination": rec.termination,
            "plies": rec.plies,
            "ours_seconds": round(rec.ours_seconds, 3),
            "theirs_seconds": round(rec.theirs_seconds, 3),
            "ours_moves": rec.ours_moves,
            "theirs_moves": rec.theirs_moves,
            "start_fen": rec.start_fen,
            "moves": rec.moves,
            **meta,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        game = chess.pgn.Game()
        game.setup(chess.Board(rec.start_fen))
        node = game
        for mv in rec.moves:
            node = node.add_variation(chess.Move.from_uci(mv))
        white, black = (
            ("TinyAlphaZero-v2", spec.name) if ours_white else (spec.name, "TinyAlphaZero-v2")
        )
        game.headers.update(
            {
                "Event": f"gauntlet {out.name}",
                "Site": "local",
                "Date": datetime.now(UTC).strftime("%Y.%m.%d"),
                "Round": f"{i + 1}.{1 if ours_white else 2}",
                "White": white,
                "Black": black,
                "Result": rec.result,
                "TimeControl": clock.pgn(),
                "Termination": rec.termination,
                "Opening": ident,
            }
        )
        with lock:
            with games_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            with pgn_path.open("a") as f:
                f.write(str(game) + "\n\n")
        return row

    n = 0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(run, t) for t in tasks]
            for fut in as_completed(futures):
                row = fut.result()
                n += 1
                print(
                    f"[{n}/{len(tasks)}] {row['opponent']} {row['opening']} "
                    f"{'W' if row['ours_white'] else 'B'} -> {row['score']} ({row['termination']}, "
                    f"{row['plies']} plies, ours {row['ours_seconds']:.1f}s theirs "
                    f"{row['theirs_seconds']:.1f}s)",
                    flush=True,
                )
    finally:
        for engine in opened:
            try:
                engine.quit()
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError):
                engine.close()
    return 0


def load_rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "games.jsonl").read_text().splitlines()]


def pair_groups(rows: list[dict]) -> dict[str, tuple[float, np.ndarray]]:
    """Per opponent, ordered by rating: (its rating, the network's mean over each complete pair)."""
    by_opp: dict[str, dict] = {}
    for r in rows:
        o = by_opp.setdefault(r["opponent"], {"rating": r["opponent_rating"], "open": {}})
        o["open"].setdefault(r["opening"], {})[r["ours_white"]] = r["score"]
    return {
        name: (
            o["rating"],
            np.array([(v[True] + v[False]) / 2 for v in o["open"].values() if len(v) == 2]),
        )
        for name, o in sorted(by_opp.items(), key=lambda kv: kv[1]["rating"])
    }


def cmd_rate(args: argparse.Namespace) -> int:
    rows = load_rows(args.out)
    groups, per_opponent = [], {}
    for name, (rating, pairs) in pair_groups(rows).items():
        mine = [r for r in rows if r["opponent"] == name]
        terms: dict[str, int] = {}
        for r in mine:
            terms[r["termination"]] = terms.get(r["termination"], 0) + 1
        per_opponent[name] = {
            **performance(rating, pairs),
            "wins": sum(r["score"] == 1.0 for r in mine),
            "draws": sum(r["score"] == 0.5 for r in mine),
            "losses": sum(r["score"] == 0.0 for r in mine),
            "unpaired_games": len(mine) - 2 * len(pairs),
            "terminations": terms,
            "ours_seconds_per_move": round(
                sum(r["ours_seconds"] for r in mine) / max(1, sum(r["ours_moves"] for r in mine)), 3
            ),
            "theirs_seconds_per_move": round(
                sum(r["theirs_seconds"] for r in mine)
                / max(1, sum(r["theirs_moves"] for r in mine)),
                3,
            ),
        }
        if len(pairs):
            groups.append((rating, pairs))
    report = {
        "overall": bootstrap_rating(groups, resamples=args.resamples, seed=args.seed),
        "per_opponent": per_opponent,
        "checkpoints": sorted({r["checkpoint"] for r in rows}),
        "tc": sorted({r["tc"] for r in rows}),
        "sims": sorted({r["sims"] for r in rows}),
    }
    path = args.out / "rating.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    ov = report["overall"]
    print(
        f"rating {ov['rating']:.0f} [{ov['ci95'][0]:.0f}, {ov['ci95'][1]:.0f}] over {ov['games']} "
        f"paired games{' (CLAMPED)' if ov['clamped'] else ''}"
    )
    for name, p in per_opponent.items():
        print(
            f"  {name:>12} ({p['opponent_rating']:.0f}): {p['wins']}-{p['draws']}-{p['losses']} "
            f"score {p['score']:.3f} perf {p['performance']:.0f}"
        )
    print(f"-> {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Two anchor families' ratings, each with its own interval (PREREG-p2_x K2, Addendum A).

    Each family is fitted by ``bootstrap_rating`` on its own opponents' pairs alone, with the
    same resamples and seed as ``rate``. They *agree* when each point lies inside the other's
    interval.
    """
    groups = pair_groups(load_rows(args.out))
    fams = {"a": args.family_a.split(","), "b": args.family_b.split(",")}
    for names in fams.values():
        missing = [n for n in names if n not in groups]
        if missing:
            raise SystemExit(f"no games against {missing} in {args.out}/games.jsonl")
    fits = {
        k: bootstrap_rating([groups[n] for n in names], resamples=args.resamples, seed=args.seed)
        for k, names in fams.items()
    }
    a, b = fits["a"], fits["b"]
    agree = families_agree(a, b)
    report = {
        "family_a": {"opponents": fams["a"], **a},
        "family_b": {"opponents": fams["b"], **b},
        "difference_b_minus_a": b["rating"] - a["rating"],
        "agree": agree,
        "rule": "agree iff each point lies inside the other's 95% interval",
    }
    path = args.out / args.file
    path.write_text(json.dumps(report, indent=2) + "\n")
    for k in ("a", "b"):
        f = fits[k]
        print(
            f"{','.join(fams[k]):>28}: {f['rating']:.0f} [{f['ci95'][0]:.0f}, {f['ci95'][1]:.0f}] "
            f"over {f['games']} games"
        )
    print(
        f"{'agree' if agree else 'DISAGREE'} (b - a = {report['difference_b_minus_a']:+.0f}) -> {path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play", help="play (or resume) a gauntlet")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--config", default=None, help="Default: config.yaml beside --ckpt")
    p.add_argument("--sims", type=int, default=None, help="Default: eval.eval_match_sims")
    p.add_argument("--device", default="cuda", help="the network's device (default cuda)")
    p.add_argument("--opponents", required=True, help="comma-separated (see above)")
    p.add_argument("--openings", type=int, default=100, help="book openings, 2 games each")
    p.add_argument("--tc", default="60+0.6", help="Fischer clock, base+inc seconds")
    p.add_argument("--concurrency", type=int, default=2, help="games at once (CPU thermal!)")
    p.add_argument("--stockfish", default="/usr/games/stockfish")
    p.add_argument("--out", required=True, type=Path)
    r = sub.add_parser("rate", help="rate the games in <out>/games.jsonl")
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--resamples", type=int, default=10_000)
    r.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare", help="two anchor families' ratings and their agreement")
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--family-a", required=True, help="comma-separated opponents")
    c.add_argument("--family-b", required=True, help="comma-separated opponents")
    c.add_argument("--file", default="k2.json", help="written under --out")
    c.add_argument("--resamples", type=int, default=10_000)
    c.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return {"play": cmd_play, "rate": cmd_rate, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
