#!/usr/bin/env python3
"""A checkpoint's forward outputs on fixed positions, for comparing two stacks.

    python scripts/forward_dump.py dump --ckpt runs/p2_g/ckpt_iter0060.pt --out before.npz
    python scripts/forward_dump.py compare before.npz after.npz

The measuring instrument is forward-only: the ladder, the suites and the skill-0 rung
read policy logits, promotion probabilities and the value through
``torch_evaluate_batch`` (bf16 autocast on a GPU). ``dump`` runs exactly that function
on positions drawn with a fixed seed from a frozen ring and saves what it returns;
``compare`` says how far two stacks' answers differ -- per output, and as agreement of
the legal-move argmax, which is what search actually consumes first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cmd_dump(args: argparse.Namespace) -> int:
    import torch

    from tinyalphazero.mcts.batch_runner import torch_evaluate_batch
    from tinyalphazero.replay import ReplayBuffer
    from tinyalphazero.train import load_checkpoint, unpack_legal_mask

    device = torch.device(args.device)
    net, _ = load_checkpoint(args.ckpt, device=device)
    net.eval()
    ring = ReplayBuffer.open(args.replay, readonly=True)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(ring), size=args.n, replace=False))
    tokens = np.asarray(ring._arr["boards"][idx]).astype(np.int64)
    legal = unpack_legal_mask(torch.from_numpy(np.asarray(ring._arr["legals"][idx]))).numpy()
    evaluate = torch_evaluate_batch(net, device)
    outs = [evaluate(tokens[i : i + args.batch]) for i in range(0, args.n, args.batch)]
    policy, promo, value = (np.concatenate([o[k] for o in outs]) for k in range(3))
    np.savez(args.out, policy=policy, promo=promo, value=value, legal=legal)
    meta = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "ckpt": str(args.ckpt),
        "n": args.n,
        "seed": args.seed,
    }
    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a, b = np.load(args.a), np.load(args.b)
    ma = json.loads(Path(str(args.a) + ".json").read_text())
    mb = json.loads(Path(str(args.b) + ".json").read_text())
    print(f"a: {ma['torch']} on {ma['device']}   b: {mb['torch']} on {mb['device']}   n={ma['n']}")
    for k in ("policy", "promo", "value"):
        d = np.abs(a[k].astype(np.float64) - b[k].astype(np.float64))
        scale = np.abs(a[k]).mean() or 1.0
        print(
            f"  {k:6s}: max |diff| {d.max():.3e}   mean |diff| {d.mean():.3e}   (mean |a| {scale:.3e})"
        )
    legal = a["legal"].reshape(len(a["legal"]), -1)
    pa = np.where(legal, a["policy"].reshape(len(legal), -1), -np.inf).argmax(1)
    pb = np.where(legal, b["policy"].reshape(len(legal), -1), -np.inf).argmax(1)
    print(
        f"  legal-move argmax agreement: {(pa == pb).mean():.4f} ({int((pa != pb).sum())} of {len(pa)} differ)"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--ckpt", type=Path, required=True)
    d.add_argument("--replay", type=Path, default=Path("runs/p2_g/replay"))
    d.add_argument("--n", type=int, default=2048)
    d.add_argument("--batch", type=int, default=256)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--device", default="cuda")
    d.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    args = ap.parse_args()
    return cmd_dump(args) if args.cmd == "dump" else cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
