"""A batch mixing all three Phase 1 datasets must collate and train.

Named test for a named failure. The mate-in-2 dataset is the only one carrying
`key_moves`, so a batch drawn across the mixture had inconsistent keys and died
in `default_collate` with `KeyError: 'key_moves'`. Nothing in the unit tests
touched it — each dataset was individually fine, and only the *mixture* broke.

The 300-step timing slice caught it, which is what integration slices are for,
but a failure that has happened once should not depend on someone remembering to
run a 3-minute GPU job. This is the cheap CPU version of that catch.

Generalised deliberately: the assertion is that every dataset in the mixture
yields the same key set, so the next optional per-mode field fails here rather
than 18,926 steps into a run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from tinyalphazero.config import TrainConfig
from tinyalphazero.data import (
    Phase1Dataset,
    generate_mate_in_1,
    generate_mate_in_2,
    generate_random_games,
    write_dataset,
)
from tinyalphazero.model import ChessNet
from tinyalphazero.train import WeightedPhaseDataset, train_step

_CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def three_datasets() -> list[Path]:
    """Tiny random-play / mate-in-1 / mate-in-2 datasets on disk."""
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)

    rand_dir = tmp / "rand"
    write_dataset(generate_random_games(3, rng), rand_dir, {"mode": "random_games"})

    m1_dir = tmp / "m1"
    write_dataset(generate_mate_in_1(6, rng), m1_dir, {"mode": "mate_in_1"})

    m2_dir = tmp / "m2"
    write_dataset(generate_mate_in_2(4, rng), m2_dir, {"mode": "mate_in_2"})

    return [rand_dir, m1_dir, m2_dir]


def test_only_mate2_carries_key_moves(three_datasets: list[Path]) -> None:
    """The precondition that made the mixture break. If this changes, so did the schema."""
    rand_dir, m1_dir, m2_dir = three_datasets
    assert "key_moves" not in Phase1Dataset(rand_dir)[0]
    assert "key_moves" not in Phase1Dataset(m1_dir)[0]
    assert "key_moves" in Phase1Dataset(m2_dir)[0]


def test_wrapped_datasets_expose_identical_keys(three_datasets: list[Path]) -> None:
    """The training wrapper must erase per-mode differences, not merely tolerate them."""
    keysets = []
    for d in three_datasets:
        wrapped = WeightedPhaseDataset(Phase1Dataset(d), value_weight=1.0)
        keysets.append(frozenset(wrapped[0].keys()))
    assert len(set(keysets)) == 1, (
        "datasets in the mixture expose different keys, which default_collate cannot "
        f"reconcile: {[sorted(k) for k in keysets]}"
    )


def test_mixed_batch_collates_and_trains(three_datasets: list[Path]) -> None:
    """The regression itself: draw across all three, collate, and take a real step."""
    rand_dir, m1_dir, m2_dir = three_datasets
    datasets = [
        WeightedPhaseDataset(Phase1Dataset(rand_dir), value_weight=0.25),
        WeightedPhaseDataset(Phase1Dataset(m1_dir), value_weight=1.0),
        WeightedPhaseDataset(Phase1Dataset(m2_dir), value_weight=1.0),
    ]
    combined = ConcatDataset(datasets)
    # Uniform weights: every dataset is guaranteed representation in the draw,
    # which is the condition that triggers the bug.
    sampler = WeightedRandomSampler(
        weights=[1.0] * len(combined),
        num_samples=16,
        replacement=True,
        generator=torch.Generator().manual_seed(0),
    )
    loader = DataLoader(combined, batch_size=8, sampler=sampler, num_workers=0)

    batch = next(iter(loader))  # this is the line that used to raise KeyError
    assert set(batch) >= {
        "tokens", "from_sq", "to_sq", "promo", "legals", "outcome", "value_weight"
    }

    from tinyalphazero.config import ModelConfig

    cfg = TrainConfig(batch_size=8, grad_clip=1.0)
    net = ChessNet(
        ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, d_policy=16, dropout=0.0)
    ).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    losses = train_step(batch, net, opt, _CPU, cfg)
    assert all(np.isfinite(v) for v in losses.values()), losses


def test_full_mixture_through_the_real_loader_builder(three_datasets: list[Path]) -> None:
    """Exercise `_build_loader` itself, so the shipped path is what is covered."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_phase1 import _build_loader

    rand_dir, m1_dir, m2_dir = three_datasets
    cfg = TrainConfig(batch_size=4)
    cfg.phase1_mate1_mix = 0.3
    cfg.phase1_mate2_mix = 0.3
    loader, n = _build_loader(rand_dir, m1_dir, cfg, seed=0, num_workers=0, mate2_dir=m2_dir)
    assert n == len(Phase1Dataset(rand_dir)) + len(Phase1Dataset(m1_dir)) + len(
        Phase1Dataset(m2_dir)
    )
    batch = next(iter(loader))
    assert batch["tokens"].shape[0] == 4


def test_anchor_mixes_must_sum_below_one(three_datasets: list[Path]) -> None:
    """Random play takes the remainder, so a >=1 total is a config error, not a clamp."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_phase1 import _build_loader

    rand_dir, m1_dir, m2_dir = three_datasets
    cfg = TrainConfig(batch_size=4)
    cfg.phase1_mate1_mix = 0.6
    cfg.phase1_mate2_mix = 0.5
    with pytest.raises(ValueError, match="must sum to <1"):
        _build_loader(rand_dir, m1_dir, cfg, seed=0, num_workers=0, mate2_dir=m2_dir)
