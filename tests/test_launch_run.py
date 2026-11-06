"""cloud/launch_run.sh spends money unattended: every pod it launches must end up training or terminated.

A stub stands in for runpod_pod.py, stage_pod.sh and systemctl, records every call, and fails on request. No network.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

STUB = r"""#!/bin/bash
# Fake CLI. $STUB_DIR/calls records each call; files in $STUB_DIR choose the behaviour.
echo "$*" >> "$STUB_DIR/calls"
case "$1" in
  launch)
    n=$(($(cat "$STUB_DIR/launches" 2>/dev/null || echo 0) + 1)); echo "$n" > "$STUB_DIR/launches"
    if [ "$n" -le "$(cat "$STUB_DIR/refuse" 2>/dev/null || echo 0)" ]; then
      echo "error: POST /pods 500: no instances"; exit 1
    fi
    echo "plan: fake"; echo "launched pod$n taz-x at \$0.89/h" ;;
  wait-ssh)
    grep -qx "$2" "$STUB_DIR/dead" 2>/dev/null && exit 1
    exit 0 ;;
  ssh)
    case "$*" in
      *sha256sum*) grep -qx "$2" "$STUB_DIR/badsha" 2>/dev/null && { echo "FAILED"; exit 1; }; echo ALL_MATCH ;;
      *nvidia-smi*) echo "RTX 4090, 24564 MiB" ;;
    esac ;;
  enable) echo "$3" > "$STUB_DIR/enabled" ;;
esac
exit 0
"""


def _run(tmp_path: Path, **behaviour: str) -> tuple[subprocess.CompletedProcess, Path]:
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    for name, value in behaviour.items():
        (stub_dir / name).write_text(value)
    stub = stub_dir / "fake.sh"
    stub.write_text(STUB)
    stub.chmod(0o755)
    repo = tmp_path / "repo"
    (repo / "runs" / "start" / "replay").mkdir(parents=True)
    (repo / "runs" / "start" / "state.json").write_text("{}")
    (repo / "runs" / "start" / "replay" / "meta.json").write_text("{}")
    env = {
        **os.environ,
        "STUB_DIR": str(stub_dir),
        "LAUNCH_REPO": str(repo),
        "LAUNCH_RP": str(stub),
        "LAUNCH_STAGE": f"{stub} stage",
        "LAUNCH_SYSTEMCTL": str(stub),
        "TAZ_CAMPAIGNS": str(tmp_path / "campaigns"),
        "RETRY_S": "0",
        "SSH_WINDOW_S": "1",
        "MAX_PODS": "2",
        "TRIES": "3",
    }
    done = subprocess.run(
        [
            "bash",
            str(REPO / "cloud" / "launch_run.sh"),
            "tag",
            "8",
            "p2_x",
            "10 20",
            "20",
            "runs/a/reps_r*.jsonl",
            "runs/a/startpos/reps_r*.jsonl",
            "python3 train.py --out runs/p2_x",
            "runs/start/state.json",
            "runs/start/replay/",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return done, stub_dir


def _calls(stub_dir: Path) -> list[str]:
    return (stub_dir / "calls").read_text().splitlines()


def test_a_refused_launch_is_retried_and_the_run_starts(tmp_path: Path) -> None:
    done, stub = _run(tmp_path, refuse="1")
    assert done.returncode == 0
    calls = _calls(stub)
    assert sum(c.startswith("launch") for c in calls) == 2
    assert not any(c.startswith("terminate") for c in calls)
    assert any("pod_job.sh" in c and "train.py --out runs/p2_x" in c for c in calls)
    assert (stub / "enabled").read_text().strip() == "taz-campaign@p2_x.service"
    env = (tmp_path / "campaigns" / "p2_x.env").read_text()
    assert "POD=pod2\n" in env and 'JUDGED="10 20"\n' in env and "MEASURE_AFTER_PULL=1\n" in env
    assert "START_REPS=runs/a/reps_r*.jsonl\n" in env


def test_a_pod_that_never_comes_up_is_terminated_and_replaced(tmp_path: Path) -> None:
    done, stub = _run(tmp_path, dead="pod1\n")
    assert done.returncode == 0
    calls = _calls(stub)
    assert "terminate pod1" in calls
    assert (tmp_path / "campaigns" / "p2_x.env").read_text().startswith("POD=pod2\n")
    assert not any(c.startswith("ssh pod1") for c in calls), "nothing is staged on a dead pod"


def test_a_checksum_mismatch_never_starts_the_job(tmp_path: Path) -> None:
    done, stub = _run(tmp_path, badsha="pod1\n")
    assert done.returncode == 0
    calls = _calls(stub)
    assert "terminate pod1" in calls
    assert not any("pod_job.sh" in c and c.startswith("ssh pod1") for c in calls)
    assert any("pod_job.sh" in c and c.startswith("ssh pod2") for c in calls)


def test_every_launched_pod_ends_training_or_terminated(tmp_path: Path) -> None:
    done, stub = _run(tmp_path, dead="pod1\npod2\n")
    assert done.returncode == 1
    calls = _calls(stub)
    assert {"terminate pod1", "terminate pod2"} <= set(calls)
    assert sum(c.startswith("launch") for c in calls) == 2, "MAX_PODS bounds the replacements"
    assert not (tmp_path / "campaigns" / "p2_x.env").exists()
    assert not (stub / "enabled").exists()


def test_no_offer_gives_up_without_a_pod(tmp_path: Path) -> None:
    done, stub = _run(tmp_path, refuse="99")
    assert done.returncode == 1
    calls = _calls(stub)
    assert sum(c.startswith("launch") for c in calls) == 3
    assert not any(c.startswith("terminate") for c in calls)
