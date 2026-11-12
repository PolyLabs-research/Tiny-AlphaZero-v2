"""Does the UCI engine play the ladder's moves? A gate on a real checkpoint.

    .venv/bin/python scripts/uci_parity.py --ckpt runs/<run>/ckpt_iter0010.pt \
        --device cuda --openings 10 --out runs/bench/uci_parity_cuda.json

Plays ``--openings`` book openings twice, colours swapped, exactly as a ladder
rung does (``BatchRunner`` under the evaluation profile, every game concurrent,
Stockfish at fixed skill and nodes on the other side). Then replays every one of
the network's decisions through ``scripts/uci_engine.py`` as a subprocess -- the
real protocol over pipes, one position at a time, the way a gauntlet will drive
it -- and counts the moves that come back the same.

On cpu the two paths compute in fp32 and should agree everywhere. On cuda the
ladder evaluates bf16 batches of many games while the engine evaluates batches
of one, so near-ties can fall the other way; the agreement is the number a
benchmark run reports beside its rating.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import chess
import chess.engine
import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", default=None, help="Default: config.yaml beside --ckpt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--openings", type=int, default=10, help="Book openings (2 games each)")
    ap.add_argument("--skill", type=int, default=5, help="Stockfish skill on the other side")
    ap.add_argument("--nodes", type=int, default=10_000, help="Stockfish nodes per move")
    ap.add_argument("--max-plies", type=int, default=None, help="Default: eval_match_max_plies")
    ap.add_argument("--stockfish", default="", help="Stockfish binary (default: on PATH)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="JSON report")
    args = ap.parse_args(argv)

    import torch

    from tinyalphazero.config import config_fingerprint, load_config
    from tinyalphazero.evaluate import _pairing, load_openings
    from tinyalphazero.mcts.batch_runner import BatchRunner, torch_evaluate_batch
    from tinyalphazero.opponents import StockfishOpponent, stockfish_path
    from tinyalphazero.train import load_checkpoint

    config_path = args.config
    if config_path is None and (args.ckpt.parent / "config.yaml").exists():
        config_path = str(args.ckpt.parent / "config.yaml")
    cfg = load_config(config_path)
    sims = cfg.eval.eval_match_sims
    max_plies = args.max_plies or cfg.eval.eval_match_max_plies
    sf = stockfish_path(args.stockfish)
    if sf is None:
        print("no Stockfish binary found", file=sys.stderr)
        return 2

    torch.set_num_threads(1)
    device = torch.device(args.device)
    net, _meta = load_checkpoint(args.ckpt, device=device)
    net.eval()

    boards, learner_white, opening_ids = _pairing(load_openings()[: args.openings])
    starts = [b.fen() for b in boards]
    opponent = StockfishOpponent(path=sf, skill=args.skill, nodes=args.nodes)
    rngs = [np.random.default_rng([args.seed, i]) for i in range(len(boards))]

    def move_fn(gi: int, board: chess.Board) -> chess.Move | None:
        if board.turn == (chess.WHITE if learner_white[gi] else chess.BLACK):
            return None
        return opponent.choose(board, rngs[gi])

    t0 = time.perf_counter()
    runner = BatchRunner(
        torch_evaluate_batch(net, device),
        replace(cfg.mcts, root_noise=False),
        device=args.device,
        max_plies=max_plies,
    )
    traces = list(runner.play_games(len(boards), num_sims=sims, boards=boards, move_fn=move_fn))
    opponent.close()
    t_ladder = time.perf_counter() - t0
    print(f"ladder: {len(traces)} games in {t_ladder:.0f} s", file=sys.stderr, flush=True)

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("uci_engine.py")),
        "--ckpt",
        str(args.ckpt),
        "--device",
        args.device,
    ]
    if args.config:
        cmd += ["--config", args.config]
    engine = chess.engine.SimpleEngine.popen_uci(cmd, timeout=120)
    asked = agreed = 0
    mismatches: list[dict] = []
    t1 = time.perf_counter()
    try:
        for gi, (fen, board) in enumerate(zip(starts, boards, strict=True)):
            side = chess.WHITE if learner_white[gi] else chess.BLACK
            replay = chess.Board(fen)
            for move in board.move_stack:
                if replay.turn == side:
                    # A wall-clock ceiling only so python-chess arms its watchdog;
                    # the engine ignores clocks and searches its fixed budget.
                    got = engine.play(replay, chess.engine.Limit(time=60.0), game=gi).move
                    asked += 1
                    if got == move:
                        agreed += 1
                    else:
                        mismatches.append(
                            {
                                "game": gi,
                                "opening": opening_ids[gi],
                                "ply": len(replay.move_stack),
                                "fen": replay.fen(),
                                "ladder": move.uci(),
                                "uci": got.uci() if got else None,
                            }
                        )
                replay.push(move)
    finally:
        engine.quit()
    t_uci = time.perf_counter() - t1

    report = {
        "checkpoint": str(args.ckpt),
        "config": config_path,
        "config_fingerprint": config_fingerprint(cfg),
        "device": args.device,
        "sims": sims,
        "games": len(boards),
        "opponent": f"stockfish_s{args.skill}@{args.nodes}",
        "positions": asked,
        "agreed": agreed,
        "agreement": agreed / asked if asked else None,
        "ladder_seconds": round(t_ladder, 1),
        "uci_seconds": round(t_uci, 1),
        "uci_seconds_per_move": round(t_uci / asked, 3) if asked else None,
        "mismatches": mismatches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{args.device}: {agreed}/{asked} agree ({report['agreement']:.4f}); "
        f"{report['uci_seconds_per_move']} s/move over UCI -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
