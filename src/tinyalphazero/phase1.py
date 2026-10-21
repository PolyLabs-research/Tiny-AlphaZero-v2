"""Phase 1 supervised-bootstrap data pipeline (spec §5).

The loader builder lives here rather than in ``scripts/train_phase1.py`` so the
*shipped* sampling path is importable without ``sys.path`` mutation (AGENTS §3)
and its two properties can be stated as library behaviour and tested:

1. **The realised anchor fraction.** ``anchor_sample_weights`` turns the
   configured mixes into per-sample weights. Under ``"exact"`` a configured mix
   *is* the fraction of the batch drawn from that anchor set; under ``"legacy"``
   it is ``mix / ((1-a)^2 + a)`` (REVIEW-2026-10-21 E1), which is what every
   Phase 1 round up to and including the anchor v2 trained under and therefore
   what a checkpoint whose embedded config says ``legacy`` must keep computing.
2. **Per-epoch determinism.** ``EpochSeededSampler`` derives each epoch's draw
   from ``(seed, epoch)``, so a run resumed at epoch k draws the same multiset
   as an uninterrupted run's epoch k instead of replaying epoch 1 (E2).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from tinyalphazero.config import TrainConfig
from tinyalphazero.data import Phase1Dataset
from tinyalphazero.train import WeightedPhaseDataset

ANCHOR_WEIGHTINGS = ("legacy", "exact")


def anchor_sample_weights(
    n_rand: int,
    anchor_sizes: Sequence[int],
    mixes: Sequence[float],
    weighting: str,
) -> np.ndarray:
    """Per-sample weights for ``WeightedRandomSampler`` over ``[random, *anchors]``.

    Random-play rows get weight ``1 - a`` (``a`` = sum of the mixes). Anchor set
    ``i`` with ``n_i`` rows gets a per-row weight such that the set's total mass is

    * ``"exact"``:  ``mix_i * n_rand``          -> realised fraction ``mix_i``
    * ``"legacy"``: ``mix_i / (1 - a) * n_rand`` -> realised fraction
      ``mix_i / ((1 - a)^2 + a)``: the ``1/(1-a)`` normalisation was applied on
      top of the ``(1-a)`` already on the random rows. Kept verbatim so a
      checkpoint trained under it still describes what it trained on.

    Worked example, ``mix = 0.05``: exact 5.00 %, legacy 5.25 %; phase1_d's
    ``0.08 / 0.07``: exact 8 % / 7 %, legacy 9.17 % / 8.02 %.
    """
    if weighting not in ANCHOR_WEIGHTINGS:
        raise ValueError(
            f"train.phase1_anchor_weighting must be one of {ANCHOR_WEIGHTINGS}, got {weighting!r}"
        )
    if len(anchor_sizes) != len(mixes):
        raise ValueError(f"{len(anchor_sizes)} anchor sets but {len(mixes)} mixes")
    if n_rand <= 0:
        raise ValueError(f"random-play dataset is empty (n_rand={n_rand})")
    anchor_mix = float(sum(mixes))
    if not 0.0 <= anchor_mix < 1.0:
        raise ValueError(f"anchor mixes must sum to <1, got {list(mixes)} = {anchor_mix}")
    for mix, n_a in zip(mixes, anchor_sizes, strict=True):
        if mix <= 0:
            raise ValueError(
                f"an anchor set is present but its mix is {mix}; it would never be drawn"
            )
        if n_a <= 0:
            raise ValueError("anchor dataset is empty")

    parts = [np.full(n_rand, 1.0 - anchor_mix)]
    for mix, n_a in zip(mixes, anchor_sizes, strict=True):
        per_row = mix * n_rand / n_a
        if weighting == "legacy":
            per_row /= 1.0 - anchor_mix
        parts.append(np.full(n_a, per_row))
    return np.concatenate(parts)


def realised_anchor_fractions(
    n_rand: int,
    anchor_sizes: Sequence[int],
    mixes: Sequence[float],
    weighting: str,
) -> tuple[float, ...]:
    """The batch fraction each anchor set actually receives under ``weighting``."""
    weights = anchor_sample_weights(n_rand, anchor_sizes, mixes, weighting)
    total = float(weights.sum())
    out = []
    offset = n_rand
    for n_a in anchor_sizes:
        out.append(float(weights[offset : offset + n_a].sum()) / total)
        offset += n_a
    return tuple(out)


def epoch_seed(seed: int, epoch: int) -> int:
    """One seed per (run seed, epoch), distinct across epochs and across runs."""
    if epoch < 1:
        raise ValueError(f"epochs are numbered from 1, got {epoch}")
    return int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0])


class EpochSeededSampler(WeightedRandomSampler):
    """``WeightedRandomSampler`` whose draw is a function of ``(seed, epoch)``.

    A generator seeded once at construction makes epoch k's multiset depend on
    how many epochs were drawn before it in *this process*, so a resumed run's
    first epoch replays epoch 1 (REVIEW-2026-10-21 E2). Here every ``__iter__``
    reseeds from ``epoch_seed(seed, epoch)`` and then advances ``epoch``, so an
    uninterrupted run draws epochs 1, 2, 3, ... and a run resumed with
    ``set_epoch(k)`` draws exactly the uninterrupted run's epoch k.
    """

    def __init__(self, weights: Sequence[float] | np.ndarray, num_samples: int, seed: int) -> None:
        super().__init__(
            weights=np.asarray(weights, dtype=np.float64).tolist(),
            num_samples=num_samples,
            replacement=True,
            generator=torch.Generator(),
        )
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        """Position the sampler so the next ``__iter__`` draws ``epoch``'s multiset."""
        if epoch < 1:
            raise ValueError(f"epochs are numbered from 1, got {epoch}")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        self.generator.manual_seed(epoch_seed(self.seed, self.epoch))
        self.epoch += 1
        return super().__iter__()


@dataclass(frozen=True)
class Phase1Loader:
    loader: DataLoader
    n_samples: int
    #: realised batch fraction per anchor set, keyed "mate1" / "mate2"
    anchor_fractions: dict[str, float]

    @property
    def sampler(self) -> EpochSeededSampler:
        return self.loader.sampler  # type: ignore[return-value]


def build_phase1_loader(
    data_dir: Path,
    mate1_dir: Path | None,
    cfg_train: TrainConfig,
    seed: int,
    num_workers: int = 8,
    mate2_dir: Path | None = None,
) -> Phase1Loader:
    """Mix random play with mate-in-1 / mate-in-2 anchors at their configured mixes.

    Each anchor set is up-weighted so its share of the batch is independent of
    how many positions it holds (see ``anchor_sample_weights`` for what "share"
    means under each ``cfg_train.phase1_anchor_weighting``). Random play takes
    the remainder, so the mixes must sum below 1. An anchor directory passed with
    a zero mix raises: it would be loaded, counted in ``n_samples`` and never
    drawn, while an ``--eval-mate2`` gate downstream reads as if it had been.
    """
    rand_ds = WeightedPhaseDataset(
        Phase1Dataset(data_dir), value_weight=cfg_train.phase1_value_weight_random
    )
    datasets: list[WeightedPhaseDataset] = [rand_ds]
    names: list[str] = []
    sizes: list[int] = []
    mixes: list[float] = []
    for name, anchor_dir, mix in (
        ("mate1", mate1_dir, cfg_train.phase1_mate1_mix),
        ("mate2", mate2_dir, cfg_train.phase1_mate2_mix),
    ):
        if anchor_dir is None:
            continue
        if mix <= 0:
            raise ValueError(
                f"{name} anchor dir {anchor_dir} given but train.phase1_{name}_mix={mix}: "
                "the anchors would never be sampled. Drop the dir or set the mix."
            )
        anchor_ds = WeightedPhaseDataset(
            Phase1Dataset(anchor_dir), value_weight=cfg_train.phase1_value_weight_anchor
        )
        datasets.append(anchor_ds)
        names.append(name)
        sizes.append(len(anchor_ds))
        mixes.append(mix)

    weighting = cfg_train.phase1_anchor_weighting
    weights = anchor_sample_weights(len(rand_ds), sizes, mixes, weighting)
    fractions = realised_anchor_fractions(len(rand_ds), sizes, mixes, weighting)

    combined = ConcatDataset(datasets)
    sampler = EpochSeededSampler(weights, num_samples=len(combined), seed=seed)

    # persistent_workers/prefetch_factor are only valid with worker processes, and
    # num_workers=0 is what tests and CPU debugging use -- so the shipped builder
    # has to accept it rather than being reachable only in its production shape.
    worker_opts = {"persistent_workers": True, "prefetch_factor": 4} if num_workers > 0 else {}
    loader = DataLoader(
        combined,
        batch_size=cfg_train.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **worker_opts,
    )
    return Phase1Loader(
        loader=loader,
        n_samples=len(combined),
        anchor_fractions=dict(zip(names, fractions, strict=True)),
    )
