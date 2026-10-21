#!/usr/bin/env python3
"""Environment check script (spec §9.3).

Prints torch version, device availability, VRAM if GPU present,
and a 4096×4096 bf16 matmul throughput number.
Writes JSON to stdout when --json is passed.
Exits 0 on CPU-only machines, 1 if the environment cannot be probed.

The probe itself is ``tinyalphazero.envinfo.environment_report``; Phase 1 and
Phase 2 run directories embed the same report in ``provenance.json``.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyAlphaZero environment check")
    parser.add_argument("--json", action="store_true", help="Output as JSON to stdout")
    parser.add_argument(
        "--no-matmul",
        action="store_true",
        help="Skip the bf16 matmul throughput probe (seconds on CPU); report device facts only",
    )
    args = parser.parse_args()

    try:
        from tinyalphazero.envinfo import environment_report

        info = environment_report(matmul=not args.no_matmul)
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
        if info["matmul_tflops"] is None:
            print("bf16 matmul:  not measured (--no-matmul)")
        else:
            print(
                f"bf16 matmul:  {info['matmul_tflops']} TFLOPS "
                f"({info['matmul_matrix_size']}x{info['matmul_matrix_size']})"
            )


if __name__ == "__main__":
    main()
