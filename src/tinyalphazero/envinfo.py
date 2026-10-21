"""Environment report (AGENTS §2.3, §4; spec §9.3).

The probing logic behind ``scripts/check_env.py``, as a library function so a
run directory can embed the same report the script prints (AGENTS §4: every run
dir gets the resolved config, the git SHA and the check_env output). The script
is a thin wrapper that prints the dict this returns.

The bf16 matmul throughput probe is opt-in: it is the one part of the report
that costs time (13 × 4096³ bf16 matmuls, seconds on CPU), and a run's
provenance or a unit test wants the device facts without paying for it.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MATMUL_SIZE = 4096
# Stamped by benchmark_row; a caller cannot pass them as fields.
BENCHMARK_ROW_STAMPS = ("benchmark", "git_sha", "date", "device")
MATMUL_ITERS = 10
MATMUL_WARMUP = 3


def matmul_tflops(
    device: Any,
    size: int = MATMUL_SIZE,
    iters: int = MATMUL_ITERS,
    warmup: int = MATMUL_WARMUP,
) -> float:
    """bf16 ``size × size`` matmul throughput in TFLOPS on ``device``."""
    import torch

    device = torch.device(device)
    a = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    b = torch.randn(size, size, dtype=torch.bfloat16, device=device)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        _ = a @ b
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    sync()
    elapsed = time.perf_counter() - start
    return (2 * size**3 * iters) / elapsed / 1e12


def environment_report(
    *,
    matmul: bool = False,
    matmul_size: int = MATMUL_SIZE,
    matmul_iters: int = MATMUL_ITERS,
) -> dict[str, Any]:
    """Torch version, device, VRAM, ROCm, and (if ``matmul``) bf16 matmul TFLOPS.

    Keys are stable — they are the ``check_env.py --json`` contract — and
    ``matmul_tflops`` / ``matmul_matrix_size`` are ``None`` when not probed.
    """
    import torch

    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "vram_gb": None,
        "matmul_tflops": None,
        "matmul_matrix_size": None,
        "rocm": False,
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")
        info["device"] = f"cuda:{torch.cuda.current_device()}"
        info["device_name"] = torch.cuda.get_device_name()
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"] = round(vram_bytes / 1024**3, 2)
        if getattr(torch.version, "hip", None) is not None:
            info["rocm"] = True
            info["rocm_version"] = torch.version.hip
    else:
        device = torch.device("cpu")

    if matmul:
        info["matmul_tflops"] = round(matmul_tflops(device, matmul_size, matmul_iters), 3)
        info["matmul_matrix_size"] = matmul_size

    return info


def benchmark_row(benchmark: str, device: Any = None, **fields: Any) -> dict[str, Any]:
    """The one schema for a ``benchmarks/results.jsonl`` row (AGENTS §5).

    Stamps ``benchmark`` (the row's kind), ``git_sha`` (short SHA of the source
    tree this package lives in — never the cwd's repo), an ISO-8601 UTC ``date``
    and ``device``; the caller's ``fields`` follow. Three writers each had their
    own spelling (``bench`` vs ``benchmark``, 7- vs 40-char SHAs, ``date: None``
    "stamped by the caller" with no caller) — REVIEW-2026-10-21 E6.
    """
    from tinyalphazero.provenance import git_sha

    clash = sorted(set(fields) & set(BENCHMARK_ROW_STAMPS))
    if clash:
        raise ValueError(
            f"{clash} are stamped by benchmark_row; pass them as arguments, not fields"
        )
    row: dict[str, Any] = {
        "benchmark": benchmark,
        "git_sha": git_sha(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": None if device is None else str(device),
    }
    row.update(fields)
    return row


def append_benchmark_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON line; the parent directory is created if missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
