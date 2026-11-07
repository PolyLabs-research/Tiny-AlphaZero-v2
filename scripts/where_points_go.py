#!/usr/bin/env python3
"""Where the points go against a rung: record games, grade every move, sort the losses.

DIAG-2026-11-07-where-points-go.md states the question and the reading rules; this is
its instrument. Three steps, each restartable and each a pure reader of the last:

    # 1. play the rung exactly as eval_reps does, keeping the moves (GPU, under gpu hold)
    python scripts/where_points_go.py play --ckpt runs/p2_s_cont/ckpt_iter0030.pt \\
        --anchor stockfish:5 --reps 2 --out-dir runs/diag_points/p2_s_cont_0030
    # 2. Stockfish grades every position of every game (CPU; keep --workers low here)
    python scripts/where_points_go.py annotate --dir runs/diag_points/p2_s_cont_0030 --workers 2
    # 3. sort the learner's losses by kind and phase
    python scripts/where_points_go.py report --dir runs/diag_points/p2_s_cont_0030
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import chess
import numpy as np

from tinyalphazero.teacher import (
    Reading,
    engine_analyse,
    material_after_line,
    non_pawn_material,
    open_engine,
    reading_to_json,
)

REPO = Path(__file__).resolve().parents[1]
MISTAKE = 0.10  # expected-score drop that counts as a mistake (DIAG §2)
BLUNDER = 0.20
PV_PLIES = 8  # how far along the engine's line material is counted
WINNING = 0.75  # expected score from which a drop is a slip while winning
THROWN = 0.85  # a game reached this and did not win: a thrown win
OPENING_PLIES = 16  # plies from the book position (itself 4 plies in) still "opening"
ENDGAME_NPM = 26  # both sides' non-pawn material at or below this is an endgame


# ---------------------------------------------------------------------------
# play
# ---------------------------------------------------------------------------


def cmd_play(args: argparse.Namespace) -> int:
    import torch

    from tinyalphazero.config import load_config
    from tinyalphazero.evaluate import load_openings, play_match
    from tinyalphazero.mcts.batch_runner import torch_evaluate_batch
    from tinyalphazero.opponents import build_opponent
    from tinyalphazero.train import load_checkpoint

    cfg = load_config(str(args.ckpt.parent / "config.yaml"))
    ev = cfg.eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = load_checkpoint(args.ckpt, device=device)
    net.eval()
    batched = torch_evaluate_batch(net, device)
    book = load_openings(REPO / ev.openings_path)
    seed = ev.seed_or(cfg.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for rep in range(1, args.reps + 1):
        out = args.out_dir / f"games_r{rep}.jsonl"
        if out.exists():
            print(f"rep {rep}: {out} exists, skipped")
            continue
        opponent = build_opponent(
            args.anchor, sims=ev.eval_match_sims, device=str(device), stockfish=ev.stockfish_path
        )
        if opponent is None:
            print("no Stockfish binary", file=sys.stderr)
            return 1
        try:
            result = play_match(
                batched,
                opponent,
                cfg,
                sims=ev.eval_match_sims,
                device=str(device),
                openings=book,
                seed=seed,
                max_openings=ev.anchor_games // 2,
            )
        finally:
            opponent.close()
        tmp = out.with_suffix(".tmp")
        with tmp.open("w") as fh:
            for gi, (o, moves, trace) in enumerate(
                zip(result.outcomes, result.moves, result.traces, strict=True)
            ):
                fh.write(
                    json.dumps(
                        {
                            "rep": rep,
                            "game": gi,
                            "opening": o.opening,
                            "epd": book[gi // 2][0],
                            "learner_white": o.learner_white,
                            "result": o.result,
                            "score": o.score,
                            "plies": o.plies,
                            "moves": moves,
                            "learner_q": [
                                [p.ply, round(float(p.root_q), 4)] for p in trace.positions
                            ],
                            "ckpt": str(args.ckpt),
                            "anchor": result.anchor,
                            "match_sims": ev.eval_match_sims,
                            "seed": seed,
                        }
                    )
                    + "\n"
                )
        tmp.rename(out)
        print(
            f"rep {rep}: {result.anchor} score {result.score:.4f} "
            f"W{result.wins} D{result.draws} L{result.losses} -> {out}"
        )
    return 0


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------

_ANALYSE = None


def _init_worker(stockfish: str, hash_mb: int) -> None:
    global _ANALYSE
    _ANALYSE = engine_analyse(open_engine(stockfish, hash_mb))


def terminal_value(board: chess.Board) -> float | None:
    """The side to move's score if the position ends the game by rule, else None."""
    if board.is_checkmate():
        return 0.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.5
    return None


def _annotate_game(task: tuple[dict[str, Any], int]) -> dict[str, Any]:
    game, nodes = task
    board = chess.Board()
    board.set_epd(game["epd"])
    readings: list[dict[str, Any] | None] = []
    for ply in range(len(game["moves"]) + 1):
        if ply > 0:
            board.push_uci(game["moves"][ply - 1])
        if terminal_value(board) is not None:
            readings.append(None)
            continue
        reading = _ANALYSE(board, nodes, 1)
        item = reading_to_json(ply, reading)
        item["pv"] = list(reading.lines[0].pv[:PV_PLIES])
        readings.append(item)
    return {"rep": game["rep"], "game": game["game"], "nodes": nodes, "readings": readings}


def cmd_annotate(args: argparse.Namespace) -> int:
    from tinyalphazero.opponents import stockfish_path

    sf = stockfish_path(args.stockfish)
    if sf is None:
        print("no Stockfish binary", file=sys.stderr)
        return 1
    for games_file in sorted(args.dir.glob("games_r*.jsonl")):
        out = games_file.with_name(games_file.name.replace("games_", "annot_"))
        done = set()
        if out.exists():
            for raw in out.read_text().splitlines():
                with contextlib.suppress(ValueError, KeyError):
                    done.add(json.loads(raw)["game"])
        games = [json.loads(x) for x in games_file.read_text().splitlines()]
        todo = [(g, args.nodes) for g in games if g["game"] not in done]
        print(f"{games_file.name}: {len(todo)} of {len(games)} games to grade", flush=True)
        with (
            mp.get_context("spawn").Pool(args.workers, _init_worker, (sf, args.hash)) as pool,
            out.open("a") as fh,
        ):
            for k, item in enumerate(pool.imap_unordered(_annotate_game, todo), 1):
                fh.write(json.dumps(item) + "\n")
                fh.flush()
                if k % 20 == 0:
                    print(f"  {k}/{len(todo)}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _reading(item: dict[str, Any] | None) -> Reading | None:
    from tinyalphazero.teacher import Line

    if item is None:
        return None
    return Reading(
        lines=tuple(
            Line(m, int(c), tuple(item.get("pv", [])) if i == 0 else ())
            for i, (m, c) in enumerate(item["lines"])
        ),
        wdl=tuple(item["wdl"]),
        depth=item["depth"],
        nodes=item["nodes"],
    )


def _expected(board: chess.Board, reading: Reading | None) -> float:
    """The side to move's expected score: the rule's if the game is over, else the engine's."""
    rule = terminal_value(board)
    if rule is not None:
        return rule
    assert reading is not None
    return reading.expected


def phase_of(board: chess.Board, ply: int) -> str:
    if non_pawn_material(board) <= ENDGAME_NPM:
        return "endgame"
    return "opening" if ply < OPENING_PLIES else "middlegame"


def classify(
    before: chess.Board,
    r_before: Reading,
    after: chess.Board,
    r_after: Reading | None,
    e_before: float,
    learner: chess.Color,
) -> str:
    """One kind per mistake, first match wins (DIAG §2)."""
    if r_after is not None and r_after.mate_for_mover and not r_before.mate_against_mover:
        return "allowed_mate"
    if (
        r_before.mate_for_mover
        and not (r_after is None and after.is_checkmate())
        and not (r_after is not None and r_after.mate_against_mover)
    ):
        return "missed_mate"
    best = material_after_line(before, r_before.lines[0].pv, learner, PV_PLIES)
    played = (
        material_after_line(after, r_after.lines[0].pv, learner, PV_PLIES - 1)
        if r_after is not None
        else best
    )
    if played <= best - 2:
        return "material"
    if e_before >= WINNING:
        return "slip_while_winning"
    return "other"


def grade_game(game: dict[str, Any], annot: dict[str, Any]) -> dict[str, Any]:
    """Every learner move's drop in expected score, its kind and phase; the game's ledger."""
    learner = chess.WHITE if game["learner_white"] else chess.BLACK
    q_at = {int(p): float(q) for p, q in game["learner_q"]}
    readings = [_reading(x) for x in annot["readings"]]
    boards = []
    b = chess.Board()
    b.set_epd(game["epd"])
    boards.append(b.copy())
    for uci in game["moves"]:
        b.push_uci(uci)
        boards.append(b.copy())

    def e_learner(ply: int) -> float:
        e = _expected(boards[ply], readings[ply])
        return e if boards[ply].turn == learner else 1.0 - e

    moves: list[dict[str, Any]] = []
    opp_gifts = 0.0
    max_e = 0.0
    for ply in range(len(game["moves"])):
        e0, e1 = e_learner(ply), e_learner(ply + 1)
        if boards[ply].turn != learner:
            opp_gifts += max(0.0, e1 - e0)
            continue
        max_e = max(max_e, e0)
        drop = e0 - e1
        item = {"ply": ply, "drop": drop, "e_before": e0, "phase": phase_of(boards[ply], ply)}
        if drop >= MISTAKE:
            item["kind"] = classify(
                boards[ply], readings[ply], boards[ply + 1], readings[ply + 1], e0, learner
            )
            if ply in q_at:
                item["model_e"] = (q_at[ply] + 1) / 2
        moves.append(item)
    end = len(game["moves"])
    return {
        "score": game["score"],
        "result": game["result"],
        "e_start": e_learner(0),
        "e_end": e_learner(end),
        "max_e": max_e,
        "moves": moves,
        "opp_gifts": opp_gifts,
    }


def summarise(graded: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(graded)
    points = sum(g["score"] for g in graded)
    mistakes = [m for g in graded for m in g["moves"] if m["drop"] >= MISTAKE]
    lost = sum(m["drop"] for m in mistakes)
    cell: dict[str, float] = defaultdict(float)
    kind: dict[str, float] = defaultdict(float)
    phase: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for m in mistakes:
        cell[f"{m['kind']}/{m['phase']}"] += m["drop"]
        kind[m["kind"]] += m["drop"]
        phase[m["phase"]] += m["drop"]
        count[m["kind"]] += 1
    blunders = [m for m in mistakes if m["drop"] >= BLUNDER]
    seen = [m for m in blunders if "model_e" in m]
    unseen = [m for m in seen if m["model_e"] - (m["e_before"] - m["drop"]) >= 0.25]
    thrown = [g for g in graded if g["max_e"] >= THROWN and g["score"] < 1.0]
    thrown_by: dict[str, float] = defaultdict(float)
    for g in thrown:
        thrown_by[g["result"]] += 1.0 - g["score"]
    share = {k: v / lost for k, v in sorted(cell.items(), key=lambda x: -x[1])} if lost else {}
    top = next(iter(share.items()), (None, 0.0))
    return {
        "games": games,
        "points": points,
        "score": points / games if games else 0.0,
        "learner_moves": sum(1 for g in graded for _ in g["moves"]),
        "mistakes": len(mistakes),
        "blunders": len(blunders),
        "expected_lost_on_mistakes": lost,
        "expected_lost_all_drops": sum(max(0.0, m["drop"]) for g in graded for m in g["moves"]),
        "opponent_gifts": sum(g["opp_gifts"] for g in graded),
        "e_start_mean": float(np.mean([g["e_start"] for g in graded])),
        "end_adjustment": sum(g["score"] - g["e_end"] for g in graded),
        "share_by_kind": {k: v / lost for k, v in sorted(kind.items(), key=lambda x: -x[1])}
        if lost
        else {},
        "count_by_kind": dict(count),
        "share_by_phase": {k: v / lost for k, v in sorted(phase.items(), key=lambda x: -x[1])}
        if lost
        else {},
        "share_by_cell": share,
        "largest_cell": {"cell": top[0], "share": top[1]},
        "one_recurring_cause": bool(top[1] >= 0.5),
        "blunders_with_model_q": len(seen),
        "blunders_search_did_not_see": len(unseen),
        "thrown_wins": len(thrown),
        "thrown_points_by_result": dict(thrown_by),
    }


def cmd_report(args: argparse.Namespace) -> int:
    graded = []
    for games_file in sorted(args.dir.glob("games_r*.jsonl")):
        annot_file = games_file.with_name(games_file.name.replace("games_", "annot_"))
        if not annot_file.exists():
            print(f"{annot_file} missing: annotate first", file=sys.stderr)
            return 1
        annots = {json.loads(x)["game"]: json.loads(x) for x in annot_file.read_text().splitlines()}
        for raw in games_file.read_text().splitlines():
            game = json.loads(raw)
            if game["game"] in annots:
                graded.append(grade_game(game, annots[game["game"]]))
    summary = summarise(graded)
    out = args.out or args.dir / "where_points_go.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--anchor", default="stockfish:5")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--out-dir", required=True, type=Path)
    a = sub.add_parser("annotate")
    a.add_argument("--dir", required=True, type=Path)
    a.add_argument("--nodes", type=int, default=100_000)
    a.add_argument("--workers", type=int, default=2)
    a.add_argument("--hash", type=int, default=64)
    a.add_argument("--stockfish", default="")
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True, type=Path)
    r.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    return {"play": cmd_play, "annotate": cmd_annotate, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
