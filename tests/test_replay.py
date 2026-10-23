"""Chunk 8: the memmap replay ring (spec §7).

The buffer is the only place a multi-day run's data lives, so the tests that
matter are the ones about *surviving* — wraparound without corrupting the rows
that stay, and reopening a directory without losing the write position.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tinyalphazero.replay import MAX_POLICY_ENTRIES, POLICY_DIM, ReplayBuffer


def _row(i: int, policy: list[tuple[int, int, float]] | None = None) -> dict:
    """One stored position, with the acted move derived exactly as ``_store`` does.

    The acted move must be the heaviest improved-policy entry, because that is the
    only relationship the producer can ever create — and ``sample`` conditions the
    promotion target on the acted cell, so a fixture that stored an unrelated
    ``move_ft`` would be testing a row shape the buffer never sees.
    """
    policy = policy if policy is not None else [((i * 7) % 4096, 0, 1.0)]
    acted = max(policy, key=lambda e: e[2])
    return {
        "tokens": np.full(72, i % 43, dtype=np.int8),
        "legals": np.full(512, i % 256, dtype=np.uint8),
        "move_ft": acted[0],
        "promo": acted[1],
        "outcome": (i % 3) - 1,
        "improved_policy": policy,
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
    # ...and the head is read at the cell the target belongs to
    np.testing.assert_array_equal(batch["from_sq"].numpy(), np.full(6, 5))
    np.testing.assert_array_equal(batch["to_sq"].numpy(), np.full(6, 6))


def test_promo_target_is_conditioned_on_the_acted_cell(tmp_path: Path) -> None:
    """A promotable pawn does not make every other move a promotion target.

    ``P(promo | from, to)`` is a conditional. If the search prefers a rook move in
    a position where a pawn *could* promote, the promotion entries belong to the
    pawn's square pair, and attaching them to the rook's would train the head to
    predict a promotion distribution for a move that promotes nothing. Nothing
    downstream would complain: the batch stays well-formed and the loss stays
    finite.
    """
    buf = ReplayBuffer.create(tmp_path / "r", capacity=4)
    rook_ft = 8 * 64 + 16
    pawn_ft = 55 * 64 + 63
    buf.add(**_row(0, [(rook_ft, 0, 0.7), (pawn_ft, 4, 0.2), (pawn_ft, 1, 0.1)]))
    batch = buf.sample(1, np.random.default_rng(0))

    assert int(batch["from_sq"][0]) == 8 and int(batch["to_sq"][0]) == 16, "acted move is the rook"
    assert not bool(batch["promo_mask"][0]), "the rook move is not a promotion"
    np.testing.assert_allclose(batch["promo_target"][0].numpy(), 0.0)
    # The policy target keeps all three moves — only the promo conditional is scoped.
    dense = batch["policy"][0].numpy()
    assert dense[rook_ft] == pytest.approx(0.7, abs=1e-3)
    assert dense[pawn_ft] == pytest.approx(0.3, abs=1e-3)


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


# ---------------------------------------------------------------------------
# root_q (version 3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fen", "mover_is_white"),
    [
        ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", True),  # Ra8#
        ("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", False),  # Ra1#
    ],
    ids=["white-mates", "black-mates"],
)
def test_root_q_shares_the_perspective_of_the_outcome_it_blends_with(
    tmp_path, fen: str, mover_is_white: bool
) -> None:
    """`root_q` and `outcome` must both be from the position's mover's view.

    This is the whole safety of the z/q blend. `outcome` is documented as
    "+1/0/-1 from THIS position's mover's view" and `root_q` is the root's
    completed-Q, which is also the mover's. If they disagreed in sign, the blend
    would train the value head *against* its own target on every position whose
    mover eventually lost — a silent, plausible-looking corruption of exactly the
    kind this project keeps finding in fields nobody asserted on.

    Asserted where the two are *produced* — a real search from a mate-in-1, scored
    by ``winner_of``/``fill_outcomes``, written through ``_store`` and read back
    from the ring — for both colours. The previous version round-tripped two
    hand-written rows whose signs already agreed, so a sign error in either
    producer passed it (REVIEW-2026-10-21 B8/C6; PREREG-p2_c cites this test).
    """
    import chess

    from tinyalphazero.config import Config
    from tinyalphazero.mcts.batch_runner import BatchRunner, uniform_stub_batch
    from tinyalphazero.selfplay import GameAssignment, _store, fill_outcomes, winner_of

    runner = BatchRunner(uniform_stub_batch, Config().mcts, max_plies=40)
    (trace,) = runner.play_games(1, seeds=[1], num_sims=16, boards=[chess.Board(fen)])
    gs = runner._games_debug[0]
    assert trace.result == "checkmate", trace.result
    winner = winner_of(trace, gs.board, None)
    assert winner == (chess.WHITE if mover_is_white else chess.BLACK)
    fill_outcomes(trace, winner)

    first = trace.positions[0]
    assert first.side_to_move == mover_is_white
    assert first.root_q == pytest.approx(1.0), "the mate is exact at the root"
    assert first.outcome == 1, "the mover who mated is labelled +1"

    buf = ReplayBuffer.create(tmp_path / "ring", capacity=8)
    _store(buf, trace, gs, iteration=1, resign_disabled=False, assignment=GameAssignment())
    assert len(buf) == len(trace.positions) >= 1
    batch = buf.sample(len(buf), np.random.default_rng(0))
    q, z = batch["root_q"], batch["outcome"].float()
    assert bool(torch.all(torch.sign(q) == torch.sign(z))), (
        f"root_q {q.tolist()} and outcome {z.tolist()} disagree in sign; the blend "
        "would train the value head against its own target"
    )
    assert bool(torch.all(batch["root_q_mask"]))
    assert float(q.max()) == pytest.approx(1.0) and int(z.max()) == 1


def test_a_ring_older_than_root_q_reports_it_absent_not_zero(tmp_path) -> None:
    """Absence is a value. 0.0 is a legal draw estimate and must not stand in.

    A v2 ring predates the field. Reporting 0.0 for every row would train the
    blend toward "every historical position was a draw"; reporting the mask False
    is the same signal a rehearsal row carries, which is the point — one meaning
    for absence, one code path.
    """
    import json

    from tinyalphazero.replay import ReplayBuffer

    buf = ReplayBuffer.create(tmp_path / "ring", capacity=4)
    buf.add(
        tokens=np.zeros(72, dtype=np.int8),
        legals=np.zeros(512, dtype=np.uint8),
        move_ft=0,
        promo=0,
        outcome=1,
        improved_policy=[(0, 0, 1.0)],
        root_q=0.7,
    )
    buf.flush()

    meta_path = tmp_path / "ring" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = 2
    meta["fields"].pop("root_q", None)
    meta_path.write_text(json.dumps(meta))
    (tmp_path / "ring" / "root_q.dat").unlink()

    old = ReplayBuffer.open(tmp_path / "ring")
    assert not old.has("root_q")
    batch = old.sample(2, np.random.default_rng(0))
    assert not bool(batch["root_q_mask"].any()), "a v2 ring claimed to have search-Q"


# ---------------------------------------------------------------------------
# seed_from: carrying a ring across a round boundary (PREREG-p2_f)
# ---------------------------------------------------------------------------


def _donor(path: Path, iterations: int, rows_per_iter: int, torn: int, capacity: int):
    """A donor ring: ``iterations`` committed iterations, then ``torn`` uncommitted rows.

    Row ``g`` carries tokens ``g % 43`` (via ``_row``) and the tag of the iteration
    that produced it; the torn rows are tagged ``iterations + 1`` and flushed, so
    the durable head sits past the commit exactly as p2_e's did.
    """
    buf = ReplayBuffer.create(path, capacity=capacity)
    g = 0
    for it in range(1, iterations + 1):
        for _ in range(rows_per_iter):
            buf.add(**{**_row(g), "iteration": it})
            g += 1
    committed = buf.total_written
    for _ in range(torn):
        buf.add(**{**_row(g), "iteration": iterations + 1})
        g += 1
    buf.flush()
    return ReplayBuffer.open(path, readonly=True), committed


def test_seed_from_carries_only_committed_intact_rows_oldest_first(tmp_path: Path) -> None:
    """The torn tail overwrote the oldest committed slots; neither may be carried.

    Donor: capacity 10, 7 iterations × 3 rows (global rows 0..20), then 4 torn rows
    (21..24) that wrapped onto slots 1..4. Committed and intact is rows 15..20 —
    slot 5 onward holds rows 15..19 and slot 0 holds row 20.
    """
    donor, committed = _donor(tmp_path / "d", iterations=7, rows_per_iter=3, torn=4, capacity=10)
    assert (committed, donor.total_written) == (21, 25)

    ring = ReplayBuffer.create(tmp_path / "r", capacity=16)
    assert ring.seed_from(donor, committed=committed, iteration_offset=-7) == 6
    ring.flush()

    reopened = ReplayBuffer.open(tmp_path / "r")
    assert len(reopened) == 6
    assert [int(reopened._arr["boards"][k][0]) for k in range(6)] == [g % 43 for g in range(15, 21)]
    # Rows 15..17 came from iteration 6 and 18..20 from 7: rebased, all <= 0.
    assert [int(reopened._arr["iteration"][k]) for k in range(6)] == [-1, -1, -1, 0, 0, 0]
    # The next write lands after the carried rows, not on top of them.
    reopened.add(**_row(99))
    assert int(reopened._arr["boards"][6][0]) == 99 % 43


def test_seed_from_keeps_the_newest_rows_when_the_donor_window_is_larger(tmp_path: Path) -> None:
    donor, committed = _donor(tmp_path / "d", iterations=7, rows_per_iter=3, torn=0, capacity=10)
    ring = ReplayBuffer.create(tmp_path / "r", capacity=4)
    assert ring.seed_from(donor, committed=committed, iteration_offset=-7) == 4
    assert [int(ring._arr["boards"][k][0]) for k in range(4)] == [g % 43 for g in range(17, 21)]


def test_seed_from_refuses_rows_from_past_the_donor_commit(tmp_path: Path) -> None:
    """A committed head that claims the torn rows must not smuggle them in."""
    donor, _committed = _donor(tmp_path / "d", iterations=7, rows_per_iter=3, torn=4, capacity=10)
    ring = ReplayBuffer.create(tmp_path / "r", capacity=16)
    with pytest.raises(ValueError, match="after its commit"):
        ring.seed_from(donor, committed=25, iteration_offset=-7)
    assert len(ReplayBuffer.open(tmp_path / "r")) == 0, "a refused seed left rows behind"


def test_seed_from_refuses_a_donor_of_another_version(tmp_path: Path) -> None:
    """A v3 ring's `moves` is the policy argmax, a v4 ring's the played move (C4)."""
    import json

    _donor(tmp_path / "d", iterations=2, rows_per_iter=3, torn=0, capacity=10)
    meta = json.loads((tmp_path / "d" / "meta.json").read_text())
    meta["version"] = 3
    (tmp_path / "d" / "meta.json").write_text(json.dumps(meta))
    ring = ReplayBuffer.create(tmp_path / "r", capacity=16)
    with pytest.raises(ValueError, match="version 3"):
        ring.seed_from(
            ReplayBuffer.open(tmp_path / "d", readonly=True), committed=6, iteration_offset=-2
        )


def test_seed_from_fills_only_an_empty_ring(tmp_path: Path) -> None:
    donor, committed = _donor(tmp_path / "d", iterations=2, rows_per_iter=3, torn=0, capacity=10)
    ring = ReplayBuffer.create(tmp_path / "r", capacity=16)
    ring.add(**_row(0))
    with pytest.raises(ValueError, match="empty ring"):
        ring.seed_from(donor, committed=committed, iteration_offset=-2)


def test_seed_from_never_writes_the_donor(tmp_path: Path) -> None:
    """The donor is a finished run's record; seeding reads it through a read-only map."""
    donor, committed = _donor(tmp_path / "d", iterations=3, rows_per_iter=3, torn=2, capacity=10)
    before = {p.name: p.read_bytes() for p in sorted((tmp_path / "d").iterdir())}
    ring = ReplayBuffer.create(tmp_path / "r", capacity=16)
    ring.seed_from(donor, committed=committed, iteration_offset=-3)
    ring.flush()
    assert {p.name: p.read_bytes() for p in sorted((tmp_path / "d").iterdir())} == before
    with pytest.raises(ValueError):
        donor._arr["boards"][0] = 1  # read-only map
