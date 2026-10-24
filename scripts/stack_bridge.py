#!/usr/bin/env python3
"""Same weights, same batches, two stacks: the numbers a training step computes.

    python scripts/stack_bridge.py --ckpt runs/p2_f/ckpt_iter0020.pt \\
        --replay runs/p2_f/replay --config runs/p2_f/config.yaml --out bridge_local.json

p2_g trains on a CUDA pod while every earlier round trained here on ROCm. This runs
the real ``train_step_phase2`` -- bf16 autocast, the masked soft policy loss, the
z/q value blend -- on fixed batches drawn with fixed seeds from a frozen ring, with
an optimizer whose learning rate is 0, so the weights never move and each batch is
scored by the same net. The model is in eval mode: dropout would make the two stacks
draw different masks, and the question is numerics, not regularisation. Comparing
the JSON from both stacks says whether a training step means the same thing on each.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tinyalphazero.config import load_config
from tinyalphazero.replay import ReplayBuffer
from tinyalphazero.train import load_checkpoint, train_step_phase2


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    net, _meta = load_checkpoint(args.ckpt, device=device, reference_model_cfg=cfg.model)
    net.eval()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0, weight_decay=0.0)
    ring = ReplayBuffer.open(args.replay, readonly=True)
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}

    rows = []
    for b in range(args.batches):
        batch = ring.sample(cfg.train.batch_size, np.random.default_rng([args.seed, b]))
        out = train_step_phase2(
            batch,
            net,
            optimizer,
            device,
            cfg.train,
            use_bf16=device.type == "cuda",
            masked_policy=cfg.phase2.masked_policy,
        )
        rows.append({"batch": b, **{k: round(v, 6) for k, v in out.items()}})
        print(json.dumps(rows[-1]), flush=True)

    moved = sum(int(not torch.equal(before[k], v)) for k, v in net.state_dict().items())
    result = {
        "ckpt": str(args.ckpt),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "tensors_moved": moved,
        "rows": rows,
    }
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if moved == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
