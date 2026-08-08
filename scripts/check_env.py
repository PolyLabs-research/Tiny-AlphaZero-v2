#!/usr/bin/env python3
"""Environment check script (spec §9.3).

Prints torch version, device availability, VRAM if GPU present,
and a 4096×4096 bf16 matmul throughput number.
Writes JSON to stdout when --json is passed.
Exits 0 on CPU-only machines.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def check_env() -> dict:
    import torch

    info: dict = {
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "vram_gb": None,
        "matmul_tflops": None,
        "rocm": False,
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")
        info["device"] = f"cuda:{torch.cuda.current_device()}"
        info["device_name"] = torch.cuda.get_device_name()
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"] = round(vram_bytes / 1024**3, 2)

        # Detect ROCm
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            info["rocm"] = True
            info["rocm_version"] = torch.version.hip
    else:
        device = torch.device("cpu")

    # bf16 matmul throughput benchmark: 4096×4096
    size = 4096
    dtype = torch.bfloat16
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)

    # Warmup
    for _ in range(3):
        _ = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()

    n_iters = 10
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # FLOPs for matmul: 2 * N^3
    flops_per_iter = 2 * (size**3)
    tflops = (flops_per_iter * n_iters) / elapsed / 1e12
    info["matmul_tflops"] = round(tflops, 3)
    info["matmul_matrix_size"] = size

    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyAlphaZero environment check")
    parser.add_argument("--json", action="store_true", help="Output as JSON to stdout")
    args = parser.parse_args()

    try:
        info = check_env()
    except Exception as exc:  # noqa: BLE001
        msg = {"error": str(exc)}
        if args.json:
            print(json.dumps(msg))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"Python:       {info['python_version']}")
        print(f"PyTorch:      {info['torch_version']}")
        print(f"Device:       {info['device']}")
        if info.get("device_name"):
            print(f"GPU name:     {info['device_name']}")
        if info["rocm"]:
            print(f"ROCm:         {info.get('rocm_version', 'detected')}")
        if info["vram_gb"] is not None:
            print(f"VRAM:         {info['vram_gb']} GB")
        print(
            f"bf16 matmul:  {info['matmul_tflops']} TFLOPS "
            f"({info['matmul_matrix_size']}x{info['matmul_matrix_size']})"
        )


if __name__ == "__main__":
    main()
