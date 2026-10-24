#!/usr/bin/env python3
"""Where a stack's gradient departs from the fp32 CPU reference, and whether its direction does.

    # one variant per process (the AOTriton switch is read when torch loads)
    python scripts/grad_diagnostic.py dump --device cpu --out g_cpu.npz
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python scripts/grad_diagnostic.py dump --device cuda --bf16 --out g_rocm.npz
    python scripts/grad_diagnostic.py compare g_cpu.npz g_rocm.npz [more.npz ...]

PREREG-p2_g Addendum 2: this box's ROCm bf16 gradient norms ran high on every bridge
batch (+122% on batch 0) while CUDA bf16 matched fp32 within 1.3%. Norms say nothing
about direction, and with ``grad_clip: 1.0`` direction is what an update is. ``dump``
runs the real ``train_step_phase2`` (zero learning rate, eval mode; the config's
dropout is 0, so eval computes what training computes) on the bridge's fixed batches
and saves each parameter's *raw* gradient -- ``clip_grad_norm_`` scales in place, so
the clipped gradient is multiplied back by ``norm / clip``. ``compare`` reports, per
batch, the norm ratio and the cosine of the whole gradient against the reference,
and the parameters whose gradients are furthest off.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np


def cmd_dump(args: argparse.Namespace) -> int:
    import torch

    from tinyalphazero.config import load_config
    from tinyalphazero.replay import ReplayBuffer
    from tinyalphazero.train import load_checkpoint, train_step_phase2

    cfg = load_config(args.config)
    device = torch.device(args.device)
    net, _ = load_checkpoint(args.ckpt, device=device, reference_model_cfg=cfg.model)
    net.eval()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0, weight_decay=0.0)
    ring = ReplayBuffer.open(args.replay, readonly=True)

    backend = contextlib.nullcontext()
    if args.sdpa != "default":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend = sdpa_kernel(
            {
                "math": SDPBackend.MATH,
                "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "flash": SDPBackend.FLASH_ATTENTION,
            }[args.sdpa]
        )

    out: dict[str, np.ndarray] = {}
    meta = {
        "device": str(device),
        "bf16": args.bf16,
        "sdpa": args.sdpa,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "batches": args.batches,
    }
    with backend:
        for b in [int(x) for x in args.batches.split(",")]:
            batch = ring.sample(cfg.train.batch_size, np.random.default_rng([args.seed, b]))
            row = train_step_phase2(
                batch,
                net,
                optimizer,
                device,
                cfg.train,
                use_bf16=args.bf16,
                masked_policy=cfg.phase2.masked_policy,
            )
            unclip = max(1.0, row["grad_norm"] / cfg.train.grad_clip)
            for name, p in net.named_parameters():
                if p.grad is not None:
                    out[f"b{b}:{name}"] = (p.grad.detach().float().cpu().numpy() * unclip).ravel()
            meta[f"b{b}"] = {"loss": row["loss"], "grad_norm": row["grad_norm"]}
            print(json.dumps({"batch": b, **{k: round(v, 6) for k, v in row.items()}}), flush=True)
    np.savez(args.out, **out)
    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    ref = np.load(args.ref)
    for path in args.variants:
        var = np.load(path)
        meta = json.loads(Path(str(path) + ".json").read_text())
        label = f"{Path(path).stem} ({meta['device']}, bf16={meta['bf16']}, sdpa={meta['sdpa']})"
        print(f"== {label}")
        batches = sorted({k.split(":")[0] for k in ref.files})
        worst: dict[str, float] = {}
        for b in batches:
            keys = [k for k in ref.files if k.startswith(b + ":")]
            r = np.concatenate([ref[k] for k in keys]).astype(np.float64)
            v = np.concatenate([var[k] for k in keys]).astype(np.float64)
            cos = float(r @ v / (np.linalg.norm(r) * np.linalg.norm(v)))
            print(
                f"   {b}: |g| ratio {np.linalg.norm(v) / np.linalg.norm(r):.3f}   cosine {cos:.5f}"
                f"   rel. error {np.linalg.norm(v - r) / np.linalg.norm(r):.3f}"
            )
            for k in keys:
                name = k.split(":", 1)[1]
                denom = np.linalg.norm(ref[k]) or 1.0
                worst[name] = max(
                    worst.get(name, 0.0), float(np.linalg.norm(var[k] - ref[k]) / denom)
                )
        top = sorted(worst.items(), key=lambda kv: -kv[1])[: args.top]
        print("   parameters furthest off (max over batches of ||g - g_ref|| / ||g_ref||):")
        for name, err in top:
            print(f"      {err:8.3f}  {name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--ckpt", type=Path, default=Path("runs/p2_f/ckpt_iter0020.pt"))
    d.add_argument("--replay", type=Path, default=Path("runs/p2_f/replay"))
    d.add_argument("--config", default="runs/p2_f/config.yaml")
    d.add_argument("--batches", default="0,6,7,1")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--device", default="cuda")
    d.add_argument("--bf16", action="store_true")
    d.add_argument("--sdpa", choices=["default", "math", "efficient", "flash"], default="default")
    d.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("ref", type=Path)
    c.add_argument("variants", type=Path, nargs="+")
    c.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    return cmd_dump(args) if args.cmd == "dump" else cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
