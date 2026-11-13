#!/usr/bin/env python3
"""How close a network is to the teacher on positions training never saw (PREREG-p2_t, T1).

    python scripts/teacher_agreement.py --ckpt runs/p2_t_teach_down/ckpt_iter0010.pt \\
        --ring runs/p2_t_teach/teacher_holdout --out runs/p2_t_teach_down/teacher_agreement.json

Over every row of a held-out teacher ring: how often the network's prior (no search)
puts its top move where Stockfish's best move is, the cross-entropy of the network's
prior against the teacher's policy, and the squared error of the network's value
(``P(win) - P(loss)``) against the teacher's ``root_q``. The rows are read in order;
the result does not depend on a seed. ``teacher_entropy`` is the teacher's own entropy on
the same rows, the floor no prior can pass. The computation is ``teacher.agreement``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--ring", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args(argv)

    import torch

    from tinyalphazero.replay import ReplayBuffer
    from tinyalphazero.teacher import agreement
    from tinyalphazero.train import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = load_checkpoint(args.ckpt, device=device)
    net.eval()
    ring = ReplayBuffer.open(args.ring, readonly=True)
    a = agreement(net, ring, device, batch=args.batch)
    result = {
        "checkpoint": str(args.ckpt),
        "ring": str(args.ring),
        "rows": a["rows"],
        "top1_agreement": a["top1_agreement"],
        "teacher_policy_ce": a["teacher_policy_ce"],
        "value_mse_vs_teacher": a["value_mse_vs_teacher"],
        "teacher_entropy": a["teacher_entropy"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
