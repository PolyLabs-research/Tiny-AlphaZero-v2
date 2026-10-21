"""The environment report is a library function the scripts and run dirs share."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tinyalphazero.envinfo import environment_report, matmul_tflops

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
