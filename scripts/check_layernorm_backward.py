#!/usr/bin/env python3
"""Does this GPU's LayerNorm backward compute the affine gradients correctly?

    python scripts/check_layernorm_backward.py          # exit 1 if any row count is wrong

The reproducer for FINDING-2026-10-24-rocm-layernorm-backward.md. Random fp64 data,
an fp64 CPU LayerNorm as the reference, the same LayerNorm in fp32 on the GPU, and the
relative error of ``weight.grad``, ``bias.grad`` and the input gradient across row
counts. On torch 2.12.1+rocm7.2 / Radeon 8060S (gfx1151) the odd-indexed affine
gradient elements are never written for 128..65,536 rows (pytorch/pytorch#199265,
fixed in torch >= 2.13), so the relative error is ~0.70; exact at 64 and from 73,728;
the input gradient is always right. A GPU check, so it lives here and not in the
CPU-only test suite (AGENTS.md section 3).
"""

from __future__ import annotations

import argparse

import torch


def error(n: int, d: int, dtype: torch.dtype, device: str) -> tuple[float, float, float]:
    x = torch.randn(n, d, dtype=torch.float64)
    go = torch.randn(n, d, dtype=torch.float64)
    ref = torch.nn.LayerNorm(d).double()
    xr = x.clone().requires_grad_()
    ref(xr).backward(go)
    dev = torch.nn.LayerNorm(d).to(device, dtype)
    xd = x.to(device, dtype).requires_grad_()
    dev(xd).backward(go.to(device, dtype))

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return (torch.linalg.norm(a.double().cpu() - b) / torch.linalg.norm(b)).item()

    return (
        rel(dev.weight.grad, ref.weight.grad),
        rel(dev.bias.grad, ref.bias.grad),
        rel(xd.grad, xr.grad),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--rows", default="64,256,512,1024,2048,4096,8192,73728")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(0)
    name = torch.cuda.get_device_name(0) if args.device == "cuda" else args.device
    print(
        f"torch {torch.__version__}  hip {getattr(torch.version, 'hip', None)}  cuda {torch.version.cuda}  {name}"
    )
    print(
        "rows     weight.grad   bias.grad    input.grad   (fp32 on the device vs fp64 on the CPU)"
    )
    bad = 0
    for n in [int(r) for r in args.rows.split(",")]:
        w, b, x = error(n, args.width, torch.float32, args.device)
        wrong = max(w, b, x) > args.tolerance
        bad += wrong
        print(f"{n:6d}   {w:11.3e}   {b:10.3e}   {x:11.3e}{'   <-- WRONG' if wrong else ''}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
