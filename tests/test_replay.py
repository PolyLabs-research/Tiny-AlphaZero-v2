"""Chunk 8: the memmap replay ring (spec §7).

The buffer is the only place a multi-day run's data lives, so the tests that
matter are the ones about *surviving* — wraparound without corrupting the rows
that stay, and reopening a directory without losing the write position.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tinyalphazero.replay import MAX_POLICY_ENTRIES, POLICY_DIM, ReplayBuffer


def _row(i: int, policy: list[tuple[int, int, float]] | None = None) -> dict:
    return {
        "tokens": np.full(72, i % 43, dtype=np.int8),
        "legals": np.full(512, i % 256, dtype=np.uint8),
        "move_ft": (i * 7) % 4096,
        "promo": i % 5,
        "outcome": (i % 3) - 1,
        "improved_policy": policy if policy is not None else [((i * 7) % 4096, 0, 1.0)],
        "iteration": i,
        "resign_disabled": bool(i % 2),
    }


def test_create_open_roundtrip(tmp_path: Path) -> None:
    buf = ReplayBuffer.create(tmp_path / "r", capacity=64)
    for i in range(10):
        buf.add(**_row(i))
    buf.flush()
    assert len(buf) == 10

    again = ReplayBuffer.open(tmp_path / "r")
    assert len(again) == 10
    assert again.capacity == 64
    assert int(again._arr["iteration"][7]) == 7


def test_reopen_resumes_the_write_position(tmp_path: Path) -> None:
    """A killed run must not overwrite what it already wrote.

    This is the resumability requirement in AGENTS.md §6 at its smallest: if
    ``total_written`` did not survive the round trip, the next process would
    restart at index 0 and silently overwrite the head of the buffer while
    ``len()`` still looked correct.
    """
    buf = ReplayBuffer.create(tmp_path / "r", capacity=16)
    for i in range(5):
        buf.add(**_row(i))
    buf.flush()

    reopened = ReplayBuffer.open(tmp_path / "r")
    reopened.add(**_row(99))
    reopened.flush()

    assert len(reopened) == 6
    assert int(reopened._arr["iteration"][0]) == 0, "row 0 must be untouched"
    assert int(reopened._arr["iteration"][5]) == 99, "new row went to the next slot"


def test_ring_wraps_and_overwrites_oldest(tmp_path: Path) -> None:
    cap = 8
    buf = ReplayBuffer.create(tmp_path / "r", capacity=cap)
    for i in range(cap + 3):
        buf.add(**_row(i))
    buf.flush()

    assert len(buf) == cap, "length saturates at capacity"
    assert buf.total_written == cap + 3
    # Slots 0..2 hold the wrapped rows 8..10; slots 3..7 still hold rows 3..7.
    assert [int(buf._arr["iteration"][i]) for i in range(3)] == [8, 9, 10]
    assert [int(buf._arr["iteration"][i]) for i in range(3, cap)] == list(range(3, cap))


def test_sparse_policy_densify_round_trip(tmp_path: Path) -> None:
    buf = ReplayBuffer.create(tmp_path / "r", capacity=4)
    policy = [(100, 0, 0.5), (200, 0, 0.3), (300, 0, 0.2)]
    buf.add(**_row(0, policy))

    dense = buf.densify_one(0)
    assert dense.shape == (POLICY_DIM,)
    assert dense.sum() == pytest.approx(1.0, abs=1e-3)
    for ft, _promo, p in policy:
        assert dense[ft] == pytest.approx(p, abs=1e-3)
    assert (dense > 0).sum() == 3


def test_promotions_share_a_cell_when_densified(tmp_path: Path) -> None:
    """Four promotions on one (from, to) must sum into that one policy index.

    The dense target is over ``from*64+to``; the promotion choice is a separate
    4-way head. Storing them as four sparse entries and densifying with ``add.at``
    is what keeps both true at once.
    """
    buf = ReplayBuffer.create(tmp_path / "r", capacity=4)
    ft = 12 * 64 + 4
    buf.add(**_row(0, [(ft, 1, 0.1), (ft, 2, 0.1), (ft, 3, 0.2), (ft, 4, 0.6)]))
    dense = buf.densify_one(0)
    assert dense[ft] == pytest.approx(1.0, abs=1e-3)
    assert (dense > 0).sum() == 1


def test_oversized_policy_keeps_the_largest_and_renormalises(tmp_path: Path) -> None:
    """A >96-move position must still store a distribution, not a truncation."""
    buf = ReplayBuffer.create(tmp_path / "r", capacity=2)
    n = MAX_POLICY_ENTRIES + 20
    policy = [(i, 0, float(i + 1)) for i in range(n)]
    total = sum(p for _, _, p in policy)
    policy = [(ft, pr, p / total) for ft, pr, p in policy]
    buf.add(**_row(0, policy))

    assert int(buf._arr["policy_n"][0]) == MAX_POLICY_ENTRIES
    dense = buf.densify_one(0)
    assert dense.sum() == pytest.approx(1.0, abs=1e-2), "must remain a distribution"
    # The kept entries are the heaviest ones, i.e. the tail of the index range.
    assert dense[n - 1] > 0 and dense[0] == 0


def test_sample_densifies_policy_and_promo_targets(tmp_path: Path) -> None:
    buf = ReplayBuffer.create(tmp_path / "r", capacity=32)
    ft = 5 * 64 + 6
    for i in range(8):
        buf.add(**_row(i, [(ft, 4, 0.75), (ft, 1, 0.25)]))
    batch = buf.sample(6, np.random.default_rng(0))

    assert batch["tokens"].shape == (6, 72)
    assert batch["policy"].shape == (6, POLICY_DIM)
    assert batch["promo_target"].shape == (6, 4)
    assert bool(batch["promo_mask"].all())
    np.testing.assert_allclose(batch["policy"][:, ft].numpy(), 1.0, atol=1e-3)
    # queen (class 4 -> index 3) got 0.75, knight (class 1 -> index 0) 0.25
    np.testing.assert_allclose(batch["promo_target"][:, 3].numpy(), 0.75, atol=1e-3)
    np.testing.assert_allclose(batch["promo_target"][:, 0].numpy(), 0.25, atol=1e-3)


def test_sample_on_empty_buffer_raises(tmp_path: Path) -> None:
    buf = ReplayBuffer.create(tmp_path / "r", capacity=4)
    with pytest.raises(ValueError, match="empty"):
        buf.sample(2, np.random.default_rng(0))


def test_open_rejects_a_version_mismatch(tmp_path: Path) -> None:
    """Silent format drift across a multi-day run is worse than a crash."""
    import json

    buf = ReplayBuffer.create(tmp_path / "r", capacity=4)
    meta = json.loads((buf.path / "meta.json").read_text())
    meta["version"] = 99
    (buf.path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="version"):
        ReplayBuffer.open(tmp_path / "r")
