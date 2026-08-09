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

import json
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
    return _epd_keys(SUITES / name)


def _epd_keys(path: Path) -> set[str] | None:
    from tinyalphazero.encoding import encode_board

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


# ---------------------------------------------------------------------------
# The stage-2 tuning suite (BLOCKED-2026-08-08)
#
# runs/data/mate2_tune_500 is where the evaluation-profile sweep picks its σ
# constants and eval_m. Two distinct things have to be true of it, and only one
# of them is the usual contamination story:
#
#   1. it must not overlap the *gate* suites, or the config would be chosen on
#      the very positions it is later scored against — the firewall;
#   2. it must not overlap the *anchor* sets, or the sweep would be ranking
#      configs by how well they exploit memorised positions rather than by
#      search quality.
#
# Legacy anchor dirs are included in (2) deliberately: phase1_d warm-started
# from phase1_c, which trained on runs/data/mate1_100k, so "superseded" does not
# mean "never seen by this checkpoint".
# ---------------------------------------------------------------------------

TUNING_SUITE = REPO / "runs" / "data" / "mate2_tune_500" / "mate_in_2.epd"


def _tuning_keys() -> set[str] | None:
    return _epd_keys(TUNING_SUITE)


@pytest.mark.parametrize("suite", EVAL_SUITES)
def test_tuning_suite_is_disjoint_from_the_gate_suites(suite: str) -> None:
    """The firewall. A shared position makes the single-shot gate self-graded."""
    tuning = _tuning_keys()
    gate = _suite_keys(suite)
    if tuning is None or gate is None:
        pytest.skip("tuning suite or gate suite not present in this working tree")
    overlap = tuning & gate
    assert not overlap, (
        f"{len(overlap)} position(s) shared between the tuning suite and gate suite "
        f"{suite}; first: {sorted(overlap)[0]}. The sweep would be selecting a "
        "config on positions the gate then scores."
    )


@pytest.mark.parametrize("eval_rel", EVAL_DIRS)
def test_tuning_suite_is_disjoint_from_held_out_sets(eval_rel: str) -> None:
    tuning = _tuning_keys()
    held_out = _dataset_keys(eval_rel)
    if tuning is None or held_out is None:
        pytest.skip(f"tuning suite or {eval_rel} not present")
    overlap = tuning & held_out
    assert not overlap, (
        f"{len(overlap)} position(s) shared between the tuning suite and held-out "
        f"{eval_rel}; first: {sorted(overlap)[0]}"
    )


@pytest.mark.parametrize("anchor_rel", ANCHOR_DIRS + LEGACY_ANCHOR_DIRS)
def test_tuning_suite_is_disjoint_from_anchor_sets(anchor_rel: str) -> None:
    """Including superseded sets: phase1_d's weights descend from phase1_c's."""
    tuning = _tuning_keys()
    anchor = _dataset_keys(anchor_rel)
    if tuning is None or anchor is None:
        pytest.skip(f"tuning suite or {anchor_rel} not present")
    overlap = tuning & anchor
    assert not overlap, (
        f"{len(overlap)} tuning-suite position(s) appear in anchor set {anchor_rel}; "
        f"first: {sorted(overlap)[0]}. The sweep would rank configs by memorisation."
    )


def test_tuning_suite_declares_itself_tuning_only() -> None:
    """The suite's own metadata must say it can never gate anything.

    Cheap to assert and worth asserting: the risk with a 500-position suite that
    scores well is somebody quoting it later as a result. The warning has to
    travel with the artefact, not live only in a BLOCKED document.
    """
    meta_path = TUNING_SUITE.parent / "meta.json"
    if not meta_path.exists():
        pytest.skip("tuning suite not generated in this working tree")
    meta = json.loads(meta_path.read_text())
    assert "TUNING ONLY" in meta.get("PURPOSE", "").upper()
    assert "MUST NEVER GATE" in meta.get("PURPOSE", "").upper()


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
