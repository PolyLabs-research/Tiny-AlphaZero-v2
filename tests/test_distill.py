"""Nested-size training on teacher rows (tinyalphazero.distill, scripts/distill_curve.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.test_teacher import _teacher_ring, _walks
from tinyalphazero.distill import (
    DELTA_TOP_THRESHOLD,
    open_prefix,
    paired_mean_diff,
    read_curve,
    sample_union,
    train_to_best,
)

REPO = Path(__file__).resolve().parent.parent


def test_prefix_ring_reads_no_further_than_its_prefix(tmp_path: Path) -> None:
    _teacher_ring(tmp_path, _walks(30, 1), "t")
    ring = open_prefix(tmp_path / "t", 12)
    assert len(ring) == 12
    batch = ring.sample(400, np.random.default_rng(0))
    first12 = {ring._arr["boards"][i].tobytes() for i in range(12)}
    assert {row.numpy().astype(np.int8).tobytes() for row in batch["tokens"]} <= first12
    assert len(open_prefix(tmp_path / "t", None)) == 30


def test_sample_union_draws_every_row_of_the_union_alike(tmp_path: Path) -> None:
    _teacher_ring(tmp_path, _walks(30, 1), "a")
    _teacher_ring(tmp_path, _walks(30, 2), "b")
    a, b = open_prefix(tmp_path / "a", None), open_prefix(tmp_path / "b", 10)
    batch = sample_union([a, b], 4000, np.random.default_rng(0))
    assert batch["tokens"].shape[0] == 4000
    from_b = {b._arr["boards"][i].tobytes() for i in range(10)}
    share_b = np.mean([r.numpy().astype(np.int8).tobytes() in from_b for r in batch["tokens"]])
    assert share_b == pytest.approx(10 / 40, abs=0.03)


def test_train_to_best_keeps_the_best_and_stops_on_patience() -> None:
    class Net:
        def __init__(self) -> None:
            self.w = 0

        def state_dict(self) -> dict:
            import torch

            return {"w": torch.tensor(float(self.w))}

    class Part:
        def __len__(self) -> int:
            return 100

        def sample(self, n: int, rng: np.random.Generator) -> dict:
            import torch

            return {"x": torch.zeros(n)}

    net = Net()
    curve = [3.0, 2.5, 2.2, 2.3, 2.25, 2.4, 2.21]  # best at the 2nd step, then 4 misses

    def measure(_net: Net) -> dict:
        return {"teacher_policy_ce": curve[_net.w], "ce_rows": np.full(3, curve[_net.w])}

    def step(_batch: dict) -> dict:
        net.w += 1
        return {}

    seen: list[dict] = []
    point = train_to_best(
        net, None, [Part()], measure, step, batch=10, eval_every=10, patience=4,
        max_samples=1000, rng=np.random.default_rng(0), on_measure=seen.append,
    )  # fmt: skip
    assert point.best["teacher_policy_ce"] == 2.2 and point.samples_at_best == 20
    assert point.samples_run == 60 and len(seen) == 7  # the start and six steps
    assert float(point.state_dict["w"]) == 2.0
    assert point.best["rows"] == 100 and point.best["samples"] == 20
    capped = train_to_best(
        Net(), None, [Part()], lambda n: {"teacher_policy_ce": -n.w, "ce_rows": np.zeros(1)},
        lambda b: {}, batch=10, eval_every=25, patience=99, max_samples=40,
        rng=np.random.default_rng(0),
    )  # fmt: skip
    assert capped.samples_run == 40


def test_paired_mean_diff_and_the_reading_rule() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(2.3, 1.0, size=5000)
    shifted = paired_mean_diff(a, a - 0.05 + rng.normal(0, 0.01, size=5000), resamples=500)
    assert shifted["ci95"][1] < DELTA_TOP_THRESHOLD
    assert shifted["ci95"][0] < shifted["mean"] < shifted["ci95"][1]
    assert paired_mean_diff(a, a, resamples=50) == paired_mean_diff(a, a, resamples=50)
    with pytest.raises(ValueError):
        paired_mean_diff(a, a[:10])

    assert read_curve({"ci95": [-0.06, -0.02]}, 0.0) == "data-limited"
    assert read_curve({"ci95": [-0.005, 0.01]}, 0.02) == "capacity-limited"
    assert read_curve({"ci95": [-0.005, 0.01]}, 0.08) == "in between"
    assert read_curve({"ci95": [-0.02, 0.0]}, 0.0) == "not resolved"


def test_distill_curve_runs_end_to_end_on_a_tiny_net(tmp_path: Path) -> None:
    import torch

    from tinyalphazero.config import load_config, save_config
    from tinyalphazero.model import ChessNet
    from tinyalphazero.train import save_checkpoint

    _teacher_ring(tmp_path, _walks(24, 1), "d1")
    _teacher_ring(tmp_path, _walks(36, 2), "new")
    _teacher_ring(tmp_path, _walks(12, 3), "hold")
    cfg = load_config(str(REPO / "configs" / "mini.yaml"))
    torch.manual_seed(0)
    net = ChessNet(cfg.model)
    start = tmp_path / "start" / "ckpt.pt"
    save_checkpoint(start, net, torch.optim.AdamW(net.parameters()), cfg, 0, "2", "test")
    save_config(cfg, str(start.parent / "config.yaml"))
    out = tmp_path / "curve"
    cmd = [
        sys.executable, str(REPO / "scripts/distill_curve.py"), "--start", str(start),
        "--d1", str(tmp_path / "d1"), "--new", str(tmp_path / "new"),
        "--holdout", str(tmp_path / "hold"), "--out", str(out), "--sizes", "24,40,60",
        "--batch", "8", "--eval-every", "16", "--patience", "2", "--max-samples", "64",
        "--probe-rows", "6", "--resamples", "100", "--device", "cpu",
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    summary = json.loads((out / "summary.json").read_text())
    assert [p["rows"] for p in summary["curve"]] == [24, 40, 60]
    assert summary["reading"] in {"data-limited", "capacity-limited", "in between", "not resolved"}
    assert summary["delta_top"]["from_rows"] == 40 and summary["delta_top"]["to_rows"] == 60
    assert np.load(out / "size_60" / "ce_rows.npy").shape == (12,)
    assert (out / "size_24" / "best.pt").exists()
    # Rerun: every size is done, nothing retrains, the summary is the same.
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    assert json.loads((out / "summary.json").read_text()) == summary
