"""The environment report is a library function the scripts and run dirs share."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tinyalphazero.envinfo import (
    BENCHMARK_ROW_STAMPS,
    append_benchmark_row,
    benchmark_row,
    environment_report,
    matmul_tflops,
)
from tinyalphazero.train import _git_sha

REPO = Path(__file__).resolve().parents[1]

# The check_env.py --json contract: every key below is always present.
CONTRACT_KEYS = {
    "torch_version",
    "python_version",
    "cuda_available",
    "device",
    "vram_gb",
    "matmul_tflops",
    "matmul_matrix_size",
    "rocm",
}


def test_report_without_the_probe_has_the_contract_keys_and_no_tflops() -> None:
    info = environment_report()
    assert set(info) >= CONTRACT_KEYS
    assert info["matmul_tflops"] is None
    assert info["matmul_matrix_size"] is None
    assert isinstance(info["cuda_available"], bool)
    assert info["device"] == "cpu" or info["device"].startswith("cuda")
    json.dumps(info)  # serialisable: it is embedded in provenance.json


def test_small_probe_measures_something() -> None:
    tflops = matmul_tflops("cpu", size=64, iters=2, warmup=1)
    assert tflops > 0.0
    info = environment_report(matmul=True, matmul_size=64, matmul_iters=2)
    assert info["matmul_tflops"] > 0.0
    assert info["matmul_matrix_size"] == 64


def test_check_env_script_prints_the_report_as_json() -> None:
    """The script is a wrapper over the library: same keys, exit 0, parseable."""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")])
        ),
    }
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_env.py"), "--json", "--no-matmul"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    info = json.loads(r.stdout)
    assert set(info) >= CONTRACT_KEYS
    assert info["torch_version"] == environment_report()["torch_version"]


# ---------------------------------------------------------------------------
# One benchmark-row writer (E6)
# ---------------------------------------------------------------------------


def test_benchmark_row_stamps_kind_sha_date_device_then_fields() -> None:
    row = benchmark_row("selfplay", device="cpu", games=3, sims_per_s=1.5)
    assert list(row)[:4] == list(BENCHMARK_ROW_STAMPS)
    assert row["benchmark"] == "selfplay"
    assert row["git_sha"] == _git_sha()
    assert row["device"] == "cpu"
    assert row["games"] == 3 and row["sims_per_s"] == 1.5
    # ISO-8601 UTC, to the second: parseable and timezone-aware.
    from datetime import datetime

    stamped = datetime.fromisoformat(row["date"])
    assert stamped.tzinfo is not None
    assert benchmark_row("x")["device"] is None


def test_benchmark_row_refuses_to_let_fields_override_the_stamps() -> None:
    with pytest.raises(ValueError, match="stamped"):
        benchmark_row("selfplay", git_sha="deadbeef")
    with pytest.raises(ValueError, match="stamped"):
        benchmark_row("selfplay", date="yesterday")


def test_append_benchmark_row_writes_one_json_line(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "results.jsonl"
    append_benchmark_row(out, benchmark_row("a", n=1))
    append_benchmark_row(out, benchmark_row("b", n=2))
    lines = out.read_text().splitlines()
    assert [json.loads(line)["benchmark"] for line in lines] == ["a", "b"]
