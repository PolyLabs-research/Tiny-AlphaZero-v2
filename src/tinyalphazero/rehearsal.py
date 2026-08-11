"""Anchor rehearsal: mixing frozen Phase 1 anchor rows into Phase 2 batches.

p2_b's single lever. p2_a's decomposition (2026-08-11, `RUNLOG-p2_a.md`) split the
mate-in-2 gate exactly:

    prior coverage @top-16   0.890  (178/200)
    solved | considered      0.933  (166/178)
    product                  0.830  = the measured gate

Both factors are addressed by the same intervention and neither is addressed by
it alone. Coverage fails because the key move has drifted out of the prior's
top-16 as self-play funnels the opening book (book entropy fell monotonically,
2.796 → 1.905, across all five ladder passes); rehearsal re-promotes it. The
conditional fails because the value head is under-convicted on the key child
(median key-edge q̂ 0.795 across the CAPABILITY dozen); the anchor sets carry
+1 labels on exactly those positions and firm it.

**Why rehearsal rather than a fresh anchor phase.** The rows are already
generated, already contamination-proven, and already in the schema the trainer
reads. The lever is a sampler, not a curriculum.

**The honesty condition.** The rehearsal sets are `mate1_100k_clean` and
`mate2_50k_clean`, both proven disjoint from every gate suite and held-out set,
and `tests/test_contamination.py` asserts that the *config defaults these fields
carry* are inside the set it checks. The gate stays a held-out generalisation
test; if it did not, "recovered mate-in-2" would mean "memorised the suite" and
the run would prove nothing while looking like a success.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyalphazero.data import Phase1Dataset
from tinyalphazero.train import unpack_legal_mask

POLICY_DIM = 4096

# Anchor rows carry this in the batch's ``iteration`` field. Real ring rows carry
# the iteration that generated them (>= 0), so a negative value identifies a
# rehearsal row unambiguously anywhere downstream.
ANCHOR_ITERATION = -1


@dataclass
class RehearsalMix:
    """How many rows of each origin make up one batch."""

    ring: int
    mate1: int
    mate2: int

    @property
    def total(self) -> int:
        return self.ring + self.mate1 + self.mate2


def plan_mix(batch_size: int, mate1_frac: float, mate2_frac: float) -> RehearsalMix:
    """Row counts for one batch, rounded so the total is always ``batch_size``.

    The anchor counts round to nearest and the ring takes the remainder, so the
    batch size the optimiser sees never depends on the mix — a rehearsal run and
    the control it is compared against take identically-shaped steps.
    """
    n1 = int(round(batch_size * mate1_frac))
    n2 = int(round(batch_size * mate2_frac))
    if n1 + n2 > batch_size:
        raise ValueError(f"rehearsal fractions ask for {n1 + n2} rows of a {batch_size}-row batch")
    return RehearsalMix(ring=batch_size - n1 - n2, mate1=n1, mate2=n2)


class RehearsalSampler:
    """Wraps a :class:`~tinyalphazero.replay.ReplayBuffer` with anchor rehearsal.

    ``sample`` returns exactly what ``ReplayBuffer.sample`` returns — same keys,
    same dtypes, same shapes — so ``train_step_phase2`` is untouched by this
    feature existing. When the configured fractions are zero the ring batch is
    passed straight through, which is what keeps every pre-p2_b run bit-identical.

    Anchor draws are seeded from ``(run seed, iteration, step)`` rather than from
    the training loop's sequential generator: reproducing iteration 12's step 300
    should not require replaying the 4,300 ring draws before it, for the same
    reason §9.2 fans the self-play seed out to the individual game.
    """

    def __init__(self, buffer: Any, cfg_train: Any, seed: int, repo: Path | None = None) -> None:
        self.buffer = buffer
        self.cfg_train = cfg_train
        self.seed = int(seed)
        repo = repo or Path(__file__).resolve().parents[2]

        self.mate1_frac = float(cfg_train.rehearsal_mate1_frac)
        self.mate2_frac = float(cfg_train.rehearsal_mate2_frac)
        self.enabled = (self.mate1_frac + self.mate2_frac) > 0.0

        self._sets: dict[str, Phase1Dataset] = {}
        if self.enabled:
            for name, frac, rel in (
                ("mate1", self.mate1_frac, cfg_train.rehearsal_mate1_dir),
                ("mate2", self.mate2_frac, cfg_train.rehearsal_mate2_dir),
            ):
                if frac <= 0.0:
                    continue
                path = Path(rel)
                if not path.is_absolute():
                    path = repo / path
                if not (path / "meta.json").exists():
                    raise FileNotFoundError(
                        f"rehearsal is on (train.rehearsal_{name}_frac={frac}) but its "
                        f"anchor set is missing: {path}. Generate it or set the "
                        f"fraction to 0 — silently training without the lever would "
                        f"make the run uninterpretable."
                    )
                self._sets[name] = Phase1Dataset(path)

    def mix_for(self, batch_size: int) -> RehearsalMix:
        return plan_mix(batch_size, self.mate1_frac, self.mate2_frac)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        iteration: int,
        step: int,
        latest_only: bool = False,
    ) -> dict[str, Any]:
        if not self.enabled:
            return self.buffer.sample(batch_size, rng, latest_only=latest_only)

        mix = self.mix_for(batch_size)
        batch = self.buffer.sample(mix.ring, rng, latest_only=latest_only)
        anchor_rng = np.random.default_rng([self.seed, iteration, step])

        parts = [batch]
        for name, count in (("mate1", mix.mate1), ("mate2", mix.mate2)):
            if count > 0:
                parts.append(self._anchor_rows(self._sets[name], count, anchor_rng))
        if len(parts) == 1:
            return batch
        return {k: torch.cat([p[k] for p in parts], dim=0) for k in batch}

    # -- anchor rows -> the Phase 2 batch schema -------------------------------

    def _anchor_rows(
        self, ds: Phase1Dataset, count: int, rng: np.random.Generator
    ) -> dict[str, Any]:
        """Draw ``count`` anchor rows and adapt them to what ``sample`` returns.

        Three fields have to be *built* rather than copied, because Phase 1 stores
        a played move where Phase 2 stores a distribution:

        * ``policy`` — dense (4096,). For a set with ``keymoves.dat`` (mate-in-2)
          the target is uniform over every key move, not one-hot on the stored
          canonical one: several first moves force mate, the stored choice is an
          arbitrary tie-break, and one-hot would spend gradient teaching the model
          to prefer that tie-break over equally-correct siblings. Since the whole
          point of the lever is getting *a* key move back into the prior's top-16,
          training against the tie-break would work against the objective.
        * ``promo_target`` / ``promo_mask`` — Phase 1's scalar (0=none, 1..4=NBRQ)
          becomes the one-hot conditional at the acted cell the promo head reads,
          matching ``ReplayBuffer.sample``'s convention exactly.
        * ``iteration`` — :data:`ANCHOR_ITERATION`, so a rehearsal row is
          identifiable downstream instead of masquerading as ring row 0.
        """
        idx = rng.integers(0, len(ds), size=count)
        tokens, legals, outcomes = [], [], []
        from_sq, to_sq = [], []
        policy = torch.zeros((count, POLICY_DIM), dtype=torch.float32)
        promo_target = torch.zeros((count, 4), dtype=torch.float32)
        promo_mask = torch.zeros(count, dtype=torch.bool)

        for b, j in enumerate(idx):
            item = ds[int(j)]
            tokens.append(item["tokens"])
            legals.append(item["legals"])
            outcomes.append(item["outcome"])
            f, t = int(item["from_sq"]), int(item["to_sq"])
            from_sq.append(f)
            to_sq.append(t)

            keys = None
            if "key_moves" in item:
                keys = unpack_legal_mask(item["key_moves"].unsqueeze(0))[0].reshape(POLICY_DIM)
                if not bool(keys.any()):
                    keys = None  # empty key set: fall back rather than emit a zero target
            if keys is None:
                policy[b, f * 64 + t] = 1.0
            else:
                policy[b] = keys.float() / float(keys.sum())

            promo = int(item["promo"])
            if promo > 0:
                promo_mask[b] = True
                promo_target[b, promo - 1] = 1.0

        return {
            "tokens": torch.stack(tokens),
            "legals": torch.stack(legals),
            "outcome": torch.stack(outcomes).long(),
            "policy": policy,
            "promo_target": promo_target,
            "promo_mask": promo_mask,
            "from_sq": torch.tensor(from_sq, dtype=torch.long),
            "to_sq": torch.tensor(to_sq, dtype=torch.long),
            "iteration": torch.full((count,), ANCHOR_ITERATION, dtype=torch.long),
            # Not a self-play row, so "did the latest weights choose this move" has
            # no answer. 0 is the honest one; nothing reads it after the mix,
            # because ``latest_only`` filters the ring *before* anchors are added.
            "mover_is_latest": torch.zeros(count, dtype=torch.long),
        }
