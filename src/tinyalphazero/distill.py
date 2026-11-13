"""Train on teacher rows only, at nested data sizes (DIAG-2026-11-13 capacity-or-data).

The diagnostic asks whether the network's distance from the teacher is set by its size
or by how many labelled positions it has seen. Each training set is ``D1`` (an existing
teacher ring, whole) plus a prefix of a second ring whose rows were pooled in a seeded
order (``teacher.pool_rows``), so the sets nest by construction. Each is trained from
one start checkpoint with the loss Phase 2 applies to teacher rows (``train_step_phase2``
on ``ReplayBuffer.sample`` batches), measured on a held-out teacher ring at a fixed
cadence, and stopped when the held-out cross-entropy has not improved for ``patience``
measurements. :func:`read_curve` is the reading rule of the plan, as code.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tinyalphazero.replay import ReplayBuffer

# The plan's thresholds (DIAG-2026-11-13 §3), in nats.
DELTA_TOP_THRESHOLD = -0.01
FIT_GAP_THRESHOLD = 0.05


class PrefixRing(ReplayBuffer):
    """A ring read as its first ``prefix`` rows: ``sample`` and ``len`` see no further."""

    prefix: int | None = None

    def __len__(self) -> int:
        n = super().__len__()
        return n if self.prefix is None else min(n, self.prefix)


def open_prefix(path: Any, prefix: int | None) -> PrefixRing:
    ring = PrefixRing.open(path, readonly=True)
    ring.prefix = prefix
    return ring


def sample_union(parts: Sequence[Any], batch: int, rng: np.random.Generator) -> dict[str, Any]:
    """A uniform draw, with replacement, over the union of ``parts``' rows.

    The draw is split across rings by a multinomial on their sizes, then each ring
    samples its share; the result is concatenated in ring order. Every row of the union
    is equally likely, as one ``sample`` over a single ring of all of them would be.
    """
    import torch

    sizes = np.array([len(p) for p in parts], dtype=np.float64)
    counts = rng.multinomial(batch, sizes / sizes.sum())
    pieces = [p.sample(int(c), rng) for p, c in zip(parts, counts, strict=True) if c > 0]
    return {k: torch.cat([b[k] for b in pieces], dim=0) for k in pieces[0]}


@dataclass
class CurvePoint:
    """One training set's result: its best measurement and the per-row CE behind it."""

    rows: int
    samples_at_best: int
    samples_run: int
    best: dict[str, Any]
    ce_rows: np.ndarray
    state_dict: dict[str, Any]
    log: list[dict[str, Any]]


def train_to_best(
    net: Any,
    optimizer: Any,
    parts: Sequence[Any],
    measure: Callable[[Any], dict[str, Any]],
    step: Callable[[dict[str, Any]], dict[str, float]],
    batch: int,
    eval_every: int,
    patience: int,
    max_samples: int,
    rng: np.random.Generator,
    on_measure: Callable[[dict[str, Any]], None] | None = None,
) -> CurvePoint:
    """Train on ``parts`` until held-out CE stops improving; return the best point.

    ``measure(net)`` returns a dict with at least ``teacher_policy_ce`` and ``ce_rows``
    (``teacher.agreement`` on the held-out ring, plus whatever else the caller adds);
    ``step(batch)`` is one optimizer step. The start is measured first (samples 0), then
    every ``eval_every`` samples; a measurement that does not beat the best by any
    amount counts toward ``patience``. The best weights are kept in memory (on CPU).
    """
    rows = int(sum(len(p) for p in parts))
    log: list[dict[str, Any]] = []

    def record(samples: int) -> tuple[dict[str, Any], dict[str, Any]]:
        m = measure(net)
        row = {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
        # ``agreement``'s ``rows`` counts the held-out ring; the log's ``rows`` is the set.
        row["heldout_rows"] = row.pop("rows", None)
        row.update({"rows": rows, "samples": samples})
        log.append(row)
        if on_measure is not None:
            on_measure(row)
        return m, row

    def snapshot() -> dict[str, Any]:
        return {k: v.detach().to("cpu").clone() for k, v in net.state_dict().items()}

    best_m, best_row = record(0)
    best_samples, best_state = 0, snapshot()
    since_best = 0
    samples = 0
    next_eval = eval_every
    while samples < max_samples and since_best < patience:
        step(sample_union(parts, batch, rng))
        samples += batch
        if samples >= next_eval or samples >= max_samples:
            next_eval += eval_every
            m, row = record(samples)
            if m["teacher_policy_ce"] < best_m["teacher_policy_ce"]:
                best_m, best_row, best_samples = m, row, samples
                best_state, since_best = snapshot(), 0
            else:
                since_best += 1
    return CurvePoint(
        rows=rows,
        samples_at_best=best_samples,
        samples_run=samples,
        best=best_row,
        ce_rows=np.asarray(best_m["ce_rows"], dtype=np.float64),
        state_dict=best_state,
        log=log,
    )


def paired_mean_diff(
    a: np.ndarray, b: np.ndarray, resamples: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> dict[str, Any]:
    """Mean of ``b - a`` over paired rows, with a percentile interval resampling rows."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("need two equal-length 1-d arrays of at least two rows")
    d = b - a
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for k in range(resamples):
        means[k] = d[rng.integers(0, len(d), size=len(d))].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "resamples": resamples,
        "seed": seed,
    }


def read_curve(delta_top: dict[str, Any], fit_gap: float) -> str:
    """The plan's reading rule (DIAG-2026-11-13 §3), from ``Δ_top`` and the D4 fit gap.

    ``delta_top`` is :func:`paired_mean_diff` of D3's and D4's held-out per-row CE
    (D4 minus D3); ``fit_gap`` is D4's held-out CE minus its training-probe CE.
    """
    lo, hi = delta_top["ci95"]
    if hi < DELTA_TOP_THRESHOLD:
        return "data-limited"
    if lo > DELTA_TOP_THRESHOLD:
        return "capacity-limited" if fit_gap < FIT_GAP_THRESHOLD else "in between"
    return "not resolved"
