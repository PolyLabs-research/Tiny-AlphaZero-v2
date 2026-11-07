#!/usr/bin/env python3
"""How close a network is to the teacher on positions training never saw (PREREG-p2_t, T1).

    python scripts/teacher_agreement.py --ckpt runs/p2_t_teach_down/ckpt_iter0010.pt \\
        --ring runs/p2_t_teach/teacher_holdout --out runs/p2_t_teach_down/teacher_agreement.json

Over every row of a held-out teacher ring: how often the network's prior (no search)
puts its top move where Stockfish's best move is, the cross-entropy of the network's
prior against the teacher's policy, and the squared error of the network's value
(``P(win) - P(loss)``) against the teacher's ``root_q``. The rows are read in order;
the result does not depend on a seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


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

    from tinyalphazero.model import masked_policy, value_scalar
    from tinyalphazero.replay import ReplayBuffer
    from tinyalphazero.train import load_checkpoint, unpack_legal_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = load_checkpoint(args.ckpt, device=device)
    net.eval()
    ring = ReplayBuffer.open(args.ring, readonly=True)
    arr = ring._arr
    n = len(ring)
    top1 = 0
    ce_sum = 0.0
    se_sum = 0.0
    for lo in range(0, n, args.batch):
        idx = np.arange(lo, min(n, lo + args.batch))
        tokens = torch.from_numpy(arr["boards"][idx].astype(np.int64)).to(device)
        legal = unpack_legal_mask(torch.from_numpy(np.asarray(arr["legals"][idx]))).to(device)
        target = torch.from_numpy(np.stack([ring.densify_one(int(i)) for i in idx])).to(device)
        best = torch.from_numpy(arr["moves"][idx].astype(np.int64)).to(device)
        with torch.no_grad():
            out = net(tokens)
            probs = masked_policy(out.policy_ft.float(), legal)
            value = value_scalar(out.value_wdl.float())
        top1 += int((probs.argmax(dim=1) == best[:, 0] * 64 + best[:, 1]).sum())
        ce_sum += float(-(target * probs.clamp_min(1e-12).log()).sum(dim=1).sum())
        root_q = torch.from_numpy(arr["root_q"][idx].astype(np.float32)).to(device)
        se_sum += float(((value - root_q) ** 2).sum())
    result = {
        "checkpoint": str(args.ckpt),
        "ring": str(args.ring),
        "rows": n,
        "top1_agreement": top1 / n,
        "teacher_policy_ce": ce_sum / n,
        "value_mse_vs_teacher": se_sum / n,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
