#!/usr/bin/env python3
"""Track 1 instrument: the mate-in-2 suite flag-off vs flag-on, for one checkpoint.

p2_d's Track 1 (RUNLOG-p2_d.md §3) measured proof-directed reply expansion
offline on frozen checkpoints: the same suite, the same sims, the evaluation
profile, once with the lever off and once on, and the per-position solve sets
compared. This is the harness that produced ``runs/p2_{c,d}/track1_ckpt_iter*.json``;
it ran from an uncommitted script at the time and is committed now
(REVIEW-2026-10-21 P5).

    python scripts/eval_track1.py --ckpt runs/p2_c/ckpt_iter0020.pt
    # -> runs/p2_c/track1_ckpt_iter0020.json

The lever is switched through ``mcts.eval_proof_directed_expansion`` — the
measurement-side field ``resolved()`` folds under the eval profile — so
self-play is never touched (REVIEW-2026-10-21 P2). Coverage and the conditional
are the decomposition ``scripts/diagnose_mate2.py`` prints: a position is
*considered* when a forcing move sits inside the prior's top-m, and the gate is
coverage × solved-given-considered.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import torch

from tinyalphazero.config import Config, config_fingerprint, load_config
from tinyalphazero.evaluate import SUITES, suite_result
from tinyalphazero.mcts.search import (
    Search,
    _forces_mate_in_2,
    load_epd,
    network_evaluator_from_net,
)
from tinyalphazero.provenance import git_sha
from tinyalphazero.train import load_checkpoint


def _measure(
    net: torch.nn.Module, cfg: Config, device: torch.device, flag: bool, sims: int
) -> dict:
    cfg = replace(cfg, mcts=replace(cfg.mcts, eval_proof_directed_expansion=flag))
    evaluate_fn = network_evaluator_from_net(net, device)
    mcts = replace(cfg.mcts, root_noise=False)
    m = mcts.resolved().gumbel_m
    search = Search(evaluate_fn, mcts)

    solved_ids: list[str] = []
    failed_ids: list[str] = []
    considered = solved_given_considered = total = 0
    for epd_line, ident in load_epd(SUITES / "mate_in_2.epd"):
        board = chess.Board()
        board.set_epd(epd_line)
        forcing_ft = {
            mv.from_square * 64 + mv.to_square
            for mv in board.legal_moves
            if _forces_mate_in_2(board, mv)
        }
        if not forcing_ft:
            continue
        total += 1
        tree, root, _ = search.run_tree(board, None, 0)
        sl = tree.edge_slice(root)
        order = np.argsort(-tree.prior[sl], kind="stable")
        rank_of = {int(e): r for r, e in enumerate(order)}
        best_rank = min(
            rank_of[i]
            for i in range(int(tree.num_edges[root]))
            if int(tree.move_ft[sl.start + i]) in forcing_ft
        )
        is_considered = best_rank < m
        considered += is_considered
        ok = _forces_mate_in_2(board, search.run(board, None, sims).best_move())
        (solved_ids if ok else failed_ids).append(ident)
        solved_given_considered += ok and is_considered

    m3_solved, m3_total, _ = suite_result(
        SUITES / "mate_in_3.epd", evaluate_fn, cfg, mate_in=3, sims=sims
    )
    return {
        "coverage": round(considered / total, 4),
        "conditional": round(solved_given_considered / considered, 4) if considered else 0.0,
        "gate": round(len(solved_ids) / total, 4),
        "solved_ids": solved_ids,
        "failed_ids": failed_ids,
        "mate3_rate": round(m3_solved / m3_total, 4) if m3_total else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", default=None, help="default: config.yaml beside --ckpt")
    ap.add_argument("--sims", type=int, default=None, help="default: eval.suite_sims (128)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None, help="default: <ckpt dir>/track1_<ckpt>.json")
    args = ap.parse_args(argv)

    config_path = args.config
    if config_path is None and (args.ckpt.parent / "config.yaml").exists():
        config_path = str(args.ckpt.parent / "config.yaml")
    cfg = load_config(config_path)
    sims = args.sims if args.sims is not None else cfg.eval.suite_sims
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    net, _meta = load_checkpoint(args.ckpt, device=device)
    net.eval()

    off = _measure(net, cfg, device, flag=False, sims=sims)
    on = _measure(net, cfg, device, flag=True, sims=sims)
    report = {
        "run": args.ckpt.parent.name,
        "checkpoint": args.ckpt.name,
        "off": off,
        "on": on,
        "converted_by_flag": sorted(set(on["solved_ids"]) - set(off["solved_ids"])),
        "broken_by_flag": sorted(set(off["solved_ids"]) - set(on["solved_ids"])),
        "sims": sims,
        "git_sha": git_sha(),
        "config_fingerprint": config_fingerprint(cfg),
    }
    out = args.out or (args.ckpt.parent / f"track1_{args.ckpt.stem}.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    for label, r in (("off", off), ("on", on)):
        print(
            f"{label:3} gate {r['gate']:.3f} = coverage {r['coverage']:.3f} x "
            f"conditional {r['conditional']:.4f} | mate3 {r['mate3_rate']}"
        )
    print(f"converted {report['converted_by_flag']} broken {report['broken_by_flag']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
