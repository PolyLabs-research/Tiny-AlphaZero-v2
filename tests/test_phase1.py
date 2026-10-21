"""The Phase 1 sampler draws what the config says it draws.

Two properties, both of which were false in the script-resident loader the
2026-10-21 review found (E1, E2):

* the fraction of a batch drawn from an anchor set is the configured mix under
  ``"exact"`` and the documented ``mix / ((1-a)^2 + a)`` under ``"legacy"``
  (every Phase 1 round trained under legacy, so a checkpoint's embedded config
  must keep meaning what it meant);
* epoch k's multiset is a function of ``(seed, k)``, so a resumed run does not
  replay epoch 1.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from tinyalphazero.config import TrainConfig
from tinyalphazero.data import (
    Phase1Dataset,
    generate_mate_in_1,
    generate_mate_in_2,
    generate_random_games,
    write_dataset,
)
from tinyalphazero.phase1 import (
    EpochSeededSampler,
    anchor_sample_weights,
    build_phase1_loader,
    epoch_seed,
    realised_anchor_fractions,
)

# (mixes, anchor set sizes): the single-anchor 5 % default, phase1_d's 8/7 split,
# and a mate-2-only mixture. Sizes are deliberately unequal to the random set so
# a formula that forgot the n_rand/n_a rescaling would fail here.
CASES = [
    ((0.05,), (37,)),
    ((0.08, 0.07), (37, 11)),
    ((0.07,), (11,)),
]
N_RAND = 1000


def _legacy(mix: float, a: float) -> float:
    return mix / ((1.0 - a) ** 2 + a)


@pytest.mark.parametrize(("mixes", "sizes"), CASES)
def test_realised_fraction_matches_the_documented_arithmetic(mixes, sizes) -> None:
    a = sum(mixes)
    exact = realised_anchor_fractions(N_RAND, sizes, mixes, "exact")
    legacy = realised_anchor_fractions(N_RAND, sizes, mixes, "legacy")
    for mix, e, lg in zip(mixes, exact, legacy, strict=True):
        assert e == pytest.approx(mix, abs=1e-9), (mix, e)
        assert lg == pytest.approx(_legacy(mix, a), abs=1e-9), (mix, lg)
    # Random play takes exactly the remainder under "exact".
    w = anchor_sample_weights(N_RAND, sizes, mixes, "exact")
    assert w[:N_RAND].sum() / w.sum() == pytest.approx(1.0 - a, abs=1e-9)


def test_legacy_is_not_exact() -> None:
    """The two modes must differ, or the field names nothing (0.05 -> 0.0525)."""
    (legacy,) = realised_anchor_fractions(N_RAND, (37,), (0.05,), "legacy")
    assert legacy == pytest.approx(0.05 / (0.95**2 + 0.05), abs=1e-9)
    assert abs(legacy - 0.05) > 2e-3


def test_unknown_weighting_raises() -> None:
    with pytest.raises(ValueError, match="phase1_anchor_weighting"):
        anchor_sample_weights(N_RAND, (37,), (0.05,), "approximately")


def test_mixes_must_sum_below_one() -> None:
    with pytest.raises(ValueError, match="must sum to <1"):
        anchor_sample_weights(N_RAND, (37, 11), (0.6, 0.5), "exact")


@pytest.mark.parametrize("weighting", ["exact", "legacy"])
def test_empirical_fraction_over_many_draws(weighting: str) -> None:
    """The weights are what the sampler consumes, so draw from them and count."""
    mixes, sizes = (0.08, 0.07), (37, 11)
    n_rand = 200
    weights = anchor_sample_weights(n_rand, sizes, mixes, weighting)
    predicted = realised_anchor_fractions(n_rand, sizes, mixes, weighting)
    n_draws = 400_000
    sampler = EpochSeededSampler(weights, num_samples=n_draws, seed=123)
    idx = np.fromiter(iter(sampler), dtype=np.int64, count=n_draws)
    lo = n_rand
    for mix, n_a, p in zip(mixes, sizes, predicted, strict=True):
        hit = ((idx >= lo) & (idx < lo + n_a)).mean()
        lo += n_a
        # sd at p~0.09 over 400k draws is ~0.00045; 0.002 is >4 sd, and the gap
        # between the two modes (0.0917 vs 0.08) is six times the tolerance.
        assert abs(hit - p) < 2e-3, (weighting, mix, hit, p)


# ---------------------------------------------------------------------------
# Per-epoch seeding (E2)
# ---------------------------------------------------------------------------


def test_epoch_seed_is_distinct_per_epoch_and_per_seed() -> None:
    seeds = {epoch_seed(42, k) for k in range(1, 50)}
    assert len(seeds) == 49
    assert epoch_seed(42, 1) != epoch_seed(43, 1)
    with pytest.raises(ValueError):
        epoch_seed(42, 0)


def test_resumed_epoch_k_equals_uninterrupted_epoch_k() -> None:
    weights = anchor_sample_weights(200, (37,), (0.05,), "exact")

    uninterrupted = EpochSeededSampler(weights, num_samples=500, seed=7)
    e1 = list(uninterrupted)
    e2 = list(uninterrupted)
    e3 = list(uninterrupted)

    resumed = EpochSeededSampler(weights, num_samples=500, seed=7)
    resumed.set_epoch(3)
    r3 = list(resumed)
    r4 = list(resumed)  # auto-advances: the next draw is epoch 4

    assert r3 == e3
    assert e3 != e1 and e2 != e1 and e3 != e2
    assert r4 != e3
    assert Counter(e1) != Counter(e3), "the multiset itself must change, not just the order"

    # Explicit positioning wins over the auto-advance, and is idempotent.
    resumed.set_epoch(1)
    assert list(resumed) == e1
    resumed.set_epoch(1)
    assert list(resumed) == e1


def test_other_seed_draws_other_epochs() -> None:
    weights = anchor_sample_weights(200, (37,), (0.05,), "exact")
    a = EpochSeededSampler(weights, num_samples=500, seed=7)
    b = EpochSeededSampler(weights, num_samples=500, seed=8)
    assert list(a) != list(b)


# ---------------------------------------------------------------------------
# Through the real builder on tiny datasets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def three_datasets() -> list[Path]:
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    rand_dir = tmp / "rand"
    write_dataset(generate_random_games(3, rng), rand_dir, {"mode": "random_games"})
    m1_dir = tmp / "m1"
    write_dataset(generate_mate_in_1(6, rng), m1_dir, {"mode": "mate_in_1"})
    m2_dir = tmp / "m2"
    write_dataset(generate_mate_in_2(4, rng), m2_dir, {"mode": "mate_in_2"})
    return [rand_dir, m1_dir, m2_dir]


@pytest.mark.parametrize("weighting", ["exact", "legacy"])
def test_builder_reports_the_fraction_its_weights_realise(three_datasets, weighting) -> None:
    rand_dir, m1_dir, m2_dir = three_datasets
    cfg = TrainConfig(batch_size=4, phase1_anchor_weighting=weighting)
    cfg.phase1_mate1_mix = 0.08
    cfg.phase1_mate2_mix = 0.07
    p1 = build_phase1_loader(rand_dir, m1_dir, cfg, seed=0, num_workers=0, mate2_dir=m2_dir)

    n_rand = len(Phase1Dataset(rand_dir))
    n1, n2 = len(Phase1Dataset(m1_dir)), len(Phase1Dataset(m2_dir))
    w = np.asarray(p1.sampler.weights, dtype=np.float64)
    assert len(w) == n_rand + n1 + n2 == p1.n_samples
    share1 = w[n_rand : n_rand + n1].sum() / w.sum()
    share2 = w[n_rand + n1 :].sum() / w.sum()
    expect1, expect2 = (
        (0.08, 0.07) if weighting == "exact" else (_legacy(0.08, 0.15), _legacy(0.07, 0.15))
    )
    assert share1 == pytest.approx(expect1, abs=1e-9)
    assert share2 == pytest.approx(expect2, abs=1e-9)
    assert p1.anchor_fractions == {
        "mate1": pytest.approx(expect1, abs=1e-9),
        "mate2": pytest.approx(expect2, abs=1e-9),
    }


def test_builder_resume_draws_the_same_batches(three_datasets) -> None:
    """End to end through the DataLoader: epoch 2 of a fresh loader positioned at 2
    yields the same batches as epoch 2 of a loader that already drew epoch 1."""
    rand_dir, m1_dir, _ = three_datasets
    cfg = TrainConfig(batch_size=8, phase1_anchor_weighting="exact")
    cfg.phase1_mate1_mix = 0.2

    full = build_phase1_loader(rand_dir, m1_dir, cfg, seed=3, num_workers=0)
    full.sampler.set_epoch(1)
    first = [b["tokens"].clone() for b in full.loader]
    second = [b["tokens"].clone() for b in full.loader]

    resumed = build_phase1_loader(rand_dir, m1_dir, cfg, seed=3, num_workers=0)
    resumed.sampler.set_epoch(2)
    second_again = [b["tokens"].clone() for b in resumed.loader]

    assert len(second) == len(second_again) == len(first) > 0
    assert all(torch.equal(x, y) for x, y in zip(second, second_again, strict=True))
    assert not all(torch.equal(x, y) for x, y in zip(first, second, strict=True))


def test_anchor_dir_with_zero_mix_raises(three_datasets) -> None:
    """E12: `--mate2 DIR` with phase1_mate2_mix=0 loaded the anchors and never drew one."""
    rand_dir, m1_dir, m2_dir = three_datasets
    cfg = TrainConfig(batch_size=4)  # phase1_mate2_mix defaults to 0.0
    with pytest.raises(ValueError, match="phase1_mate2_mix=0.0"):
        build_phase1_loader(rand_dir, m1_dir, cfg, seed=0, num_workers=0, mate2_dir=m2_dir)
    cfg.phase1_mate1_mix = 0.0
    with pytest.raises(ValueError, match="phase1_mate1_mix=0.0"):
        build_phase1_loader(rand_dir, m1_dir, cfg, seed=0, num_workers=0)


def test_no_anchor_dirs_is_pure_random_play(three_datasets) -> None:
    rand_dir, _, _ = three_datasets
    cfg = TrainConfig(batch_size=4)
    p1 = build_phase1_loader(rand_dir, None, cfg, seed=0, num_workers=0)
    assert p1.anchor_fractions == {}
    assert p1.n_samples == len(Phase1Dataset(rand_dir))
