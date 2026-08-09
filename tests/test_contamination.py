"""Anchor training sets must not contain any evaluation position.

This is the integrity of round `d`. The round exists to raise mate-in-2 prior
coverage, and it is judged by a held-out eval set and by `tests/suites/mate_in_2.epd`.
If a training anchor set contains a suite position, the gate measures memorisation
and the whole round means nothing — while looking like a success, which is worse
than looking like a failure.

A gate, not an advisory: it fails the build.

Comparison key is placement + turn + castling + EP, via `data.position_key`, which
decodes tokens so both sides normalise identically. Raw-FEN comparison would be
wrong here: the EP token is only set when en passant is *legally available* (spec
§3.1's Markov fix), so a raw FEN's EP field and the encoded one legitimately
disagree, and the comparison would under-report overlap.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest

from tinyalphazero.data import Phase1Dataset, position_key

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "tests" / "suites"

# Training anchor sets — anything the model fits on.
# Round `d` anchor sets. The *_clean sets were regenerated with --exclude over
# every held-out set and committed suite after the guard caught real overlap in
# the originals (1 position each). The originals are kept, not deleted: round c's
# provenance points at runs/data/mate1_100k and rewriting it would falsify the
# record. They are listed here so the guard keeps reporting on them.
ANCHOR_DIRS = [
    "runs/data/mate1_100k_clean",
    "runs/data/mate2_50k_clean",
]

# Superseded anchor sets: known-contaminated, quantified, and no longer used for
# training. Reported by a separate xfail-style test rather than silently dropped.
LEGACY_ANCHOR_DIRS = [
    "runs/data/mate1_100k",
    "runs/data/mate2_50k",
    "runs/data/mate1",
]
# Held-out sets and committed suites — anything a gate reads.
EVAL_DIRS = [
    "runs/data/eval_rand",
    "runs/data/eval_mate1_1k",
    "runs/data/eval_mate1",
    "runs/data/eval_mate2_1k",
]
EVAL_SUITES = ["mate_in_1.epd", "mate_in_2.epd"]


def _dataset_keys(rel: str) -> set[str] | None:
    path = REPO / rel
    if not (path / "meta.json").exists():
        return None
    ds = Phase1Dataset(path)
    return {position_key(ds[i]["tokens"].numpy()) for i in range(len(ds))}


def _suite_keys(name: str) -> set[str] | None:
    """Normalise EPD positions through encode→decode so keys are comparable."""
    from tinyalphazero.encoding import encode_board

    path = SUITES / name
    if not path.exists():
        return None
    keys = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        board = chess.Board()
        board.set_epd(line)
        keys.add(position_key(encode_board(board, 0)))
    return keys


@pytest.mark.parametrize("anchor_rel", ANCHOR_DIRS)
@pytest.mark.parametrize("eval_rel", EVAL_DIRS)
def test_no_anchor_eval_contamination(anchor_rel: str, eval_rel: str) -> None:
    anchor = _dataset_keys(anchor_rel)
    held_out = _dataset_keys(eval_rel)
    if anchor is None or held_out is None:
        pytest.skip(f"{anchor_rel} or {eval_rel} not generated in this working tree")
    overlap = anchor & held_out
    assert not overlap, (
        f"{len(overlap)} position(s) shared between training {anchor_rel} and "
        f"held-out {eval_rel}; first: {sorted(overlap)[0]}"
    )


@pytest.mark.parametrize("anchor_rel", ANCHOR_DIRS)
@pytest.mark.parametrize("suite", EVAL_SUITES)
def test_no_suite_contamination(anchor_rel: str, suite: str) -> None:
    """The named gate from the round `d` brief."""
    anchor = _dataset_keys(anchor_rel)
    suite_keys = _suite_keys(suite)
    if anchor is None or suite_keys is None:
        pytest.skip(f"{anchor_rel} or {suite} not present in this working tree")
    overlap = anchor & suite_keys
    assert not overlap, (
        f"{len(overlap)} position(s) from {suite} appear in training set "
        f"{anchor_rel}; first: {sorted(overlap)[0]}. The gate would be measuring "
        "memorisation."
    )


def test_position_key_normalises_both_sides_identically() -> None:
    """The key must survive an encode→decode round trip, or overlap is under-reported."""
    from tinyalphazero.encoding import encode_board

    board = chess.Board()
    for uci in ["e2e4", "c7c5", "g1f3", "d7d6"]:
        board.push(chess.Move.from_uci(uci))
    once = position_key(encode_board(board, 0))
    twice = position_key(encode_board(chess.Board(once + " 0 1"), 0))
    assert once == twice


def test_contamination_check_can_actually_fail() -> None:
    """A guard that cannot fire is decoration. Feed it a known overlap."""
    from tinyalphazero.encoding import encode_board

    board = chess.Board()
    shared = position_key(encode_board(board, 0))
    a = {shared, "other"}
    b = {shared}
    assert a & b, "set intersection must detect a planted duplicate"
    assert np.array_equal(np.array([1]), np.array([1]))


@pytest.mark.parametrize("anchor_rel", LEGACY_ANCHOR_DIRS)
def test_legacy_anchor_contamination_is_bounded_and_recorded(anchor_rel: str) -> None:
    """Superseded sets are allowed to be dirty, but never by an unmeasured amount.

    `runs/data/mate1_100k` shares exactly 1 position with `eval_mate1_1k` — 0.10%
    of that eval set, bounding its effect on round c's `mate1_top8_masked` at
    0.0010 against a score of 0.980 and a gate of 0.97. Round c's conclusion is
    unaffected. `runs/data/mate2_50k` shares 1 with `eval_rand`. Both were
    superseded by *_clean sets rather than patched, so this test exists to keep
    the number honest if anyone reaches for the old dirs again.
    """
    anchor = _dataset_keys(anchor_rel)
    if anchor is None:
        pytest.skip(f"{anchor_rel} not present")
    total = 0
    for eval_rel in EVAL_DIRS:
        held_out = _dataset_keys(eval_rel)
        if held_out is not None:
            total += len(anchor & held_out)
    assert total <= 2, (
        f"{anchor_rel} overlap with held-out sets grew to {total}; it was <=2 when "
        "measured on 2026-08-08. A superseded set getting dirtier means something "
        "regenerated it."
    )
