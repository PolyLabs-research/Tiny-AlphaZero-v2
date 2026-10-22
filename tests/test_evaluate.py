"""Chunk 10: the anchor ladder (spec §8).

M3's go/no-go is a statement about *this* code: "monotone Elo vs the Phase 1
anchor with non-overlapping CIs between start and end". So the Elo and CI
machinery is tested on synthetics with known answers before it is ever pointed at
a checkpoint — including the **null** case, because "non-overlapping" is only
meaningful if the machinery can produce an interval that overlaps zero when the
two players are in fact equal.

The opponents and the book get the same treatment: they are frozen for the life of
the project, so what they are is asserted here rather than assumed.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import chess
import numpy as np
import pytest

from tinyalphazero import logschema
from tinyalphazero.config import Config, load_config
from tinyalphazero.evaluate import (
    _pairing,
    book_move1_entropy,
    elo_from_score,
    format_elo,
    load_openings,
    load_watchlist,
    paired_bootstrap_elo,
    play_match,
    play_opponents,
)
from tinyalphazero.mcts.batch_runner import BatchRunner, uniform_stub_batch
from tinyalphazero.mcts.search import can_force_mate, load_epd, move_solves
from tinyalphazero.opponents import (
    GreedyMaterial,
    RandomMover,
    build_opponent,
    stockfish_path,
)

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "tests" / "suites"
BOOK = SUITES / "openings_4ply.epd"
MATE3 = SUITES / "mate_in_3.epd"


# ---------------------------------------------------------------------------
# Elo, on synthetics with known answers
# ---------------------------------------------------------------------------


def test_known_score_gives_known_elo() -> None:
    """400·log10(s/(1−s)), checked against hand-computable points."""
    assert elo_from_score(0.5, 200)[0] == pytest.approx(0.0)
    assert elo_from_score(0.75, 200)[0] == pytest.approx(400 * np.log10(3.0))
    assert elo_from_score(0.25, 200)[0] == pytest.approx(-400 * np.log10(3.0))
    # Symmetry: swapping colours must swap the sign, exactly.
    for s in (0.05, 0.31, 0.62, 0.9):
        assert elo_from_score(s, 200)[0] == pytest.approx(-elo_from_score(1 - s, 200)[0])


def test_a_clean_sweep_is_clamped_and_says_so() -> None:
    """A 200-0 result has infinite implied Elo, which is true and useless.

    The clamp is at half a game — the resolution of the measurement — and the flag
    travels with the number so no reader has to guess whether a bound was hit.
    """
    elo, clamped = elo_from_score(1.0, 200)
    assert clamped and np.isfinite(elo)
    assert elo == pytest.approx(400 * np.log10(399.0))
    tighter, _ = elo_from_score(1.0, 20)
    assert tighter < elo, "fewer games must imply a weaker bound, not a stronger one"
    assert elo_from_score(0.5, 200)[1] is False


def test_ci_shrinks_as_the_number_of_openings_grows() -> None:
    rng = np.random.default_rng(0)
    widths = []
    for n_pairs in (25, 100, 400):
        pairs = rng.choice([0.0, 0.5, 1.0], size=n_pairs, p=[0.25, 0.2, 0.55])
        _elo, lo, hi, _c = paired_bootstrap_elo(
            pairs, n_games=2 * n_pairs, resamples=2000, rng=np.random.default_rng(1)
        )
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2], widths


def test_the_null_case_produces_a_ci_that_straddles_zero() -> None:
    """The case M3's "non-overlapping CIs" criterion is judged against.

    Two equal players score 50%. If the machinery could not produce an interval
    containing 0 here, "the CIs stopped overlapping" would be a property of the
    estimator rather than evidence of progress.
    """
    pairs = np.full(100, 0.5)
    elo, lo, hi, clamped = paired_bootstrap_elo(pairs, n_games=200, resamples=5000)
    assert elo == pytest.approx(0.0) and not clamped
    assert lo <= 0.0 <= hi, (lo, hi)

    # And a genuinely even but noisy match, where individual games are decisive.
    rng = np.random.default_rng(7)
    noisy = rng.choice([0.0, 0.5, 1.0], size=100, p=[0.4, 0.2, 0.4])
    _e, lo2, hi2, _c = paired_bootstrap_elo(noisy, n_games=200, resamples=5000, rng=rng)
    assert lo2 <= 0.0 <= hi2, (lo2, hi2)


def test_bootstrap_is_deterministic_given_a_seed() -> None:
    pairs = np.random.default_rng(2).random(60)
    a = paired_bootstrap_elo(pairs, 120, resamples=1000, rng=np.random.default_rng(5))
    b = paired_bootstrap_elo(pairs, 120, resamples=1000, rng=np.random.default_rng(5))
    assert a == b


def test_a_lopsided_result_excludes_zero() -> None:
    """The other half of the criterion: real separation must be reported as such."""
    pairs = np.full(100, 0.8)
    _elo, lo, hi, _c = paired_bootstrap_elo(pairs, 200, resamples=5000)
    assert lo > 0.0, (lo, hi)


# ---------------------------------------------------------------------------
# The colour-swap pairing
# ---------------------------------------------------------------------------


def test_pairing_plays_each_opening_from_both_sides_adjacently() -> None:
    """``_summarise`` reshapes scores to (-1, 2); the pairing must match that."""
    book = [("8/8/8/8/8/8/8/K6k w - -", "a"), ("8/8/8/8/8/8/8/K6k b - -", "b")]
    boards, white, ids = _pairing(book)
    assert len(boards) == 4
    assert white == [True, False, True, False]
    assert ids == ["a", "a", "b", "b"]


def test_pair_scores_average_the_two_colours() -> None:
    cfg = Config()
    cfg.eval.eval_match_max_plies = 12
    result = play_opponents(
        RandomMover(), RandomMover(), cfg, openings=load_openings()[:6], seed=11
    )
    assert result.games == 12
    assert len(result.pair_scores) == 6
    scores = [o.score for o in result.outcomes]
    for i, p in enumerate(result.pair_scores):
        assert p == pytest.approx((scores[2 * i] + scores[2 * i + 1]) / 2)


# ---------------------------------------------------------------------------
# Opponents
# ---------------------------------------------------------------------------


def test_greedy_beats_random_decisively() -> None:
    """Spec §8's floor calibration, through the ladder's own protocol.

    Run at the ladder's ply cap, not self-play's: greedy has no mating technique,
    so it grinds, and at 160 plies 28 of these games are cut off mid-grind and
    scored as draws — an artefact of a *cost* bound, not of the opponents.
    """
    cfg = Config()
    result = play_opponents(GreedyMaterial(), RandomMover(), cfg, seed=3)
    assert result.games == 200
    assert result.losses == 0, "a 1-ply material grabber must never lose to a random mover"
    assert result.score >= 0.90, f"greedy scored {result.score:.3f}"
    assert result.elo_lo > 0.0


def test_greedy_takes_a_free_queen() -> None:
    greedy = GreedyMaterial()
    # White rook on a1 can take the black queen on a8; nothing else is a capture.
    board = chess.Board("q6k/8/8/8/8/8/7P/R6K w - - 0 1")
    assert greedy.choose(board, np.random.default_rng(0)) == chess.Move.from_uci("a1a8")


def test_greedy_prefers_mate_over_material() -> None:
    """``mate1_002`` from the committed suite: mate available, and a free rook.

    Taken from an artefact rather than hand-built. A constructed FEN that turns
    out to have no mate in it tests the test author, and that is exactly what the
    first version of this did.
    """
    board = chess.Board()
    board.set_epd("8/2k1n1b1/3q3p/P1P3p1/1R2r3/6pP/4Brb1/1NB1K1R1 b - -")
    greedy = GreedyMaterial()
    material = max(greedy.score(board, mv) for mv in board.legal_moves if not _mates(board, mv))
    assert material >= 5.0, "the position must actually tempt a material grab"

    chosen = greedy.choose(board, np.random.default_rng(0))
    board.push(chosen)
    assert board.is_checkmate(), f"greedy played {chosen} instead of mate"


def _mates(board: chess.Board, mv: chess.Move) -> bool:
    board.push(mv)
    out = board.is_checkmate()
    board.pop()
    return out


def test_greedy_is_reproducible_from_the_generator() -> None:
    """Without a seeded tie-break the book would produce far fewer distinct games."""
    board = chess.Board()
    a = [GreedyMaterial().choose(board, np.random.default_rng(4)) for _ in range(5)]
    b = [GreedyMaterial().choose(board, np.random.default_rng(4)) for _ in range(5)]
    assert a == b
    c = [GreedyMaterial().choose(board, np.random.default_rng(9)) for _ in range(5)]
    assert set(a) != set(c) or a != c


def test_stockfish_absence_is_a_skip_not_a_failure() -> None:
    """§8's Stockfish rungs are M4 calibration; M3's go/no-go never reads them."""
    assert build_opponent("stockfish:0", stockfish="/definitely/not/here") is None
    if stockfish_path() is None:
        assert build_opponent("stockfish:0") is None
    else:
        pytest.skip("a Stockfish binary is installed on this machine")


def test_unknown_opponent_is_an_error_not_a_silent_skip() -> None:
    with pytest.raises(ValueError, match="unknown opponent"):
        build_opponent("houdini")


# ---------------------------------------------------------------------------
# The external-move hook in the batched runner
# ---------------------------------------------------------------------------


def test_move_fn_plays_without_building_a_tree() -> None:
    """A non-searching opponent must cost no simulation budget at all."""
    runner = BatchRunner(uniform_stub_batch, Config().mcts, max_plies=8)

    def always_first(_gi: int, board: chess.Board) -> chess.Move:
        return next(iter(board.legal_moves))

    traces = list(runner.play_games(2, num_sims=16, move_fn=always_first))
    assert runner.stats["sims"] == 0, "an all-external game ran simulations"
    assert runner.stats["forwards"] == 0
    assert all(t.plies == 8 for t in traces)
    assert all(t.positions == [] for t in traces), "external moves must not write trace rows"


def test_move_fn_none_is_the_unmodified_search_path() -> None:
    runner = BatchRunner(uniform_stub_batch, Config().mcts, max_plies=4)
    baseline = list(runner.play_games(2, num_sims=8))
    stats = dict(runner.stats)

    runner2 = BatchRunner(uniform_stub_batch, Config().mcts, max_plies=4)
    same = list(runner2.play_games(2, num_sims=8, move_fn=lambda _gi, _b: None))
    assert runner2.stats == stats
    assert [t.plies for t in same] == [t.plies for t in baseline]


def test_only_the_searching_side_writes_trace_rows() -> None:
    """The calibration pass reads trace rows; a row from an unsearched ply is a lie."""
    cfg = Config()
    cfg.eval.eval_match_max_plies = 10
    result = play_match(
        uniform_stub_batch, RandomMover(), cfg, sims=4, openings=load_openings()[:2], seed=1
    )
    for outcome, trace in zip(result.outcomes, result.traces, strict=True):
        learner_side = chess.WHITE if outcome.learner_white else chess.BLACK
        assert all(p.side_to_move == learner_side for p in trace.positions)


# ---------------------------------------------------------------------------
# The frozen opening book
# ---------------------------------------------------------------------------


def test_the_book_is_committed_and_the_right_size() -> None:
    assert BOOK.exists(), "the opening book is committed, not generated at eval time"
    assert len(load_epd(BOOK)) == 100


def test_book_positions_are_playable_and_distinct() -> None:
    """A terminal or in-check opening is not an opening (see the generator).

    Ply count is *not* asserted from the file: EPD carries no move counters, so
    ``set_epd`` always reports fullmove 1. What the file can support is checked
    here; that the positions are 4 plies deep is established by regenerating them
    from the seed in the test below.
    """
    entries = load_epd(BOOK)
    seen = set()
    for epd, ident in entries:
        board = chess.Board()
        board.set_epd(epd)
        assert not board.is_game_over(), f"{ident} is terminal"
        assert not board.is_check(), f"{ident} starts in check"
        assert board.legal_moves.count() > 0
        assert board.turn == chess.WHITE, f"{ident} is not an even number of plies in"
        # Four plies allow at most one capture.
        assert len(board.piece_map()) >= 31, f"{ident} has lost too much material"
        key = board.epd()
        assert key not in seen, f"{ident} duplicates an earlier opening"
        seen.add(key)


def test_the_book_is_reproducible_from_its_seed() -> None:
    """Committed, but regenerable — provenance the record can be checked against."""
    from scripts.generate_openings import generate

    regenerated = [b.epd() for b in generate(100, seed=42)]
    committed = []
    for epd, _ident in load_epd(BOOK):
        board = chess.Board()
        board.set_epd(epd)  # strips the id operation; compare positions, not lines
        committed.append(board.epd())
    assert regenerated == committed


# ---------------------------------------------------------------------------
# The mate-in-3 suite
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MATE3.exists(), reason="mate_in_3.epd not generated in this tree")
def test_mate_in_3_positions_are_exactly_mate_in_3() -> None:
    """Sound and tight: every entry is forceable in 3 and **not** in 2.

    Generator and grader are the same code (``mcts.search.can_force_mate``), so
    this is not a tautology only because it re-checks the committed *artefact*:
    a suite file that drifted from the code that made it would fail here.
    """
    entries = load_epd(MATE3)
    assert len(entries) >= 100
    for epd, ident in entries:
        board = chess.Board()
        board.set_epd(epd)
        assert not can_force_mate(board, 2), f"{ident} is solvable in 2 or fewer"
        assert can_force_mate(board, 3), f"{ident} is not a forced mate in 3"


@pytest.mark.skipif(not MATE3.exists(), reason="mate_in_3.epd not generated in this tree")
def test_mate_in_3_bm_moves_are_solutions_and_ids_unique() -> None:
    entries = load_epd(MATE3)
    ids = [ident for _e, ident in entries]
    assert len(set(ids)) == len(ids)
    for epd, ident in entries:
        board = chess.Board()
        board.set_epd(epd)
        bm = epd.split(" bm ", 1)[1].split(";", 1)[0].strip()
        assert move_solves(board, board.parse_san(bm), 3), f"{ident}'s bm does not solve it"


@pytest.mark.skipif(not MATE3.exists(), reason="mate_in_3.epd not generated in this tree")
def test_mate_in_3_is_disjoint_from_the_other_suites() -> None:
    """A shared position would let one suite's number be read off another's."""
    from tests.test_contamination import _epd_keys

    m3 = _epd_keys(MATE3)
    for other in ("mate_in_1.epd", "mate_in_2.epd"):
        keys = _epd_keys(SUITES / other)
        assert keys is not None
        assert not (m3 & keys), f"mate_in_3 overlaps {other}"


# ---------------------------------------------------------------------------
# Row schema and the watchlist
# ---------------------------------------------------------------------------


def _match_row(**over) -> dict:
    base = {
        "checkpoint": "runs/x.pt",
        "iteration": 10,
        "anchor": "phase1",
        "games": 200,
        "wins": 100,
        "draws": 50,
        "losses": 50,
        "score": 0.625,
        "elo": 88.7,
        "elo_lo": 40.0,
        "elo_hi": 140.0,
        "elo_clamped": False,
        "draw_rate": 0.25,
        "mean_plies": 120.0,
        "match_sims": 48,
        "max_plies": 300,
        "seconds": 1200.0,
        "git_sha": "abc1234",
        "config_fingerprint": "f" * 64,
        "anchor_digest": None,
        "nondeterministic": False,
    }
    base.update(over)
    return logschema.ladder_match_row(**base)


def test_a_saturated_elo_renders_as_a_bound_not_a_precise_interval() -> None:
    """A clean sweep is the most precise-looking row in the table and the least
    informative one in it.

    Clamping the point estimate *and* collapsing the bootstrap (every resampled
    pair scores alike) yields ``+636.4 [+636.4, +636.4]``, which reads to a skimmer
    as an extraordinarily tight measurement. Presentation is the hazard, so
    presentation is where it is fixed — the JSONL keeps the raw numbers.
    """
    swept = {"elo": 636.4, "elo_lo": 636.4, "elo_hi": 636.4, "elo_clamped": True, "games": 20}
    rendered = format_elo(swept)
    assert "≥ +636.4" in rendered and "saturated" in rendered
    assert "[" not in rendered, "a saturated result must not be shown as an interval"

    lost = {**swept, "elo": -636.4, "elo_lo": -636.4, "elo_hi": -636.4}
    assert "≤ -636.4" in format_elo(lost)

    ordinary = {"elo": 127.0, "elo_lo": 17.4, "elo_hi": 269.4, "elo_clamped": False, "games": 20}
    assert format_elo(ordinary) == "Elo +127.0 [+17.4, +269.4]"

    null = {"elo": 0.0, "elo_lo": -30.0, "elo_hi": 30.0, "elo_clamped": False, "games": 200}
    assert "CI straddles 0" in format_elo(null)

    degenerate = {"elo": 12.0, "elo_lo": 12.0, "elo_hi": 12.0, "elo_clamped": False, "games": 200}
    assert "degenerate CI" in format_elo(degenerate)

    # The self-match null is both at once, and both notes must survive: the
    # straddle is the result, the degeneracy is why it is exact.
    self_match = {"elo": 0.0, "elo_lo": 0.0, "elo_hi": 0.0, "elo_clamped": False, "games": 20}
    rendered = format_elo(self_match)
    assert "CI straddles 0" in rendered and "degenerate CI" in rendered


def test_a_deterministic_self_match_is_an_exact_null() -> None:
    """A checkpoint against itself scores 0.500 by identity, not by luck.

    Under the evaluation profile both players are the same deterministic function,
    so the two colour-swapped games of an opening are *the same game* with the
    labels swapped: whoever wins it wins as white in one and loses as black in the
    other, and the pair scores exactly 0.5. Every pair. The bootstrap then has
    nothing to resample and the interval collapses onto zero.

    That makes this stronger than a statistical null in two further ways.

    It is a **pairing proof**: if ``_pairing`` mis-ordered the games or the colour
    assignment were not a true swap, the pair scores could not all be 0.5.

    It is a **no-hidden-state detector**: all-pairs-exactly-0.5 holds if and only
    if move choice is a pure function of position. Any per-game RNG draw, any
    seat-dependent branch, any cache-ordering effect that leaks into selection
    breaks it immediately and visibly. That is why the shipped ``GreedyMaterial``
    cannot stand in here — its tie-break is drawn per game by design, and it
    returns pair scores of 0.75 and 0.25.
    """
    cfg = Config()
    book = load_openings()[:12]
    exact = play_opponents(
        _DeterministicGreedy(), _DeterministicGreedy(), cfg, openings=book, seed=5
    )

    # Non-vacuous: the identity must survive a *decided* game, not merely hold
    # because everything drew. Deterministic greedy self-play decides one of these.
    assert exact.wins > 0 and exact.wins == exact.losses, (exact.wins, exact.losses)
    assert all(p == pytest.approx(0.5) for p in exact.pair_scores), exact.pair_scores
    assert exact.score == pytest.approx(0.5)
    assert exact.elo == pytest.approx(0.0)
    assert exact.elo_lo <= 0.0 <= exact.elo_hi

    # The premise is *identical and position-pure*. Two ways to break it, both of
    # which must show up as pair scores that are not all 0.5:
    mixed = play_opponents(_DeterministicGreedy(), GreedyMaterial(), cfg, openings=book, seed=5)
    assert not all(p == 0.5 for p in mixed.pair_scores), mixed.pair_scores

    # ...and identical players that carry per-game state. This is the hidden-state
    # detector firing on a real instance of hidden state, not a synthetic one.
    impure = play_opponents(GreedyMaterial(), GreedyMaterial(), cfg, openings=book, seed=5)
    assert not all(p == 0.5 for p in impure.pair_scores), impure.pair_scores


class _DeterministicGreedy(GreedyMaterial):
    """Greedy with the random tie-break removed — the identity's premise.

    The shipped ``GreedyMaterial`` draws its tie-break from a per-game generator,
    and the two halves of a colour-swapped pair are different games, so they get
    different generators and the identity does not apply to it.
    """

    name = "det-greedy"

    def choose(self, board: chess.Board, _rng) -> chess.Move:
        moves = list(board.legal_moves)
        scores = [self.score(board, mv) for mv in moves]
        best = max(scores)
        return min(
            (mv for mv, s in zip(moves, scores, strict=True) if s == best),
            key=lambda m: (m.from_square, m.to_square, m.promotion or 0),
        )


def test_ladder_rows_validate_and_reject_drift() -> None:
    row = _match_row()
    logschema.validate_ladder(row)
    row["surprise"] = 1
    with pytest.raises(ValueError, match="unknown"):
        logschema.validate_ladder(row)
    with pytest.raises(ValueError, match="row_type"):
        logschema.validate_ladder({"row_type": "something-else"})


def test_completed_matchups_makes_a_pass_resumable(tmp_path: Path) -> None:
    """A killed pass costs the matchup in flight and nothing already measured."""
    log = tmp_path / "ladder.jsonl"
    assert logschema.completed_matchups(log) == set()
    assert logschema.completed_keys(log) == set()
    logschema.append_ladder_row(log, _match_row(anchor="random"))
    logschema.append_ladder_row(log, _match_row(anchor="greedy"))
    assert logschema.completed_matchups(log) == {
        ("runs/x.pt", "random"),
        ("runs/x.pt", "greedy"),
    }
    assert logschema.match_key("runs/x.pt", "random", 200, 48, 300, "protocol") in (
        logschema.completed_keys(log)
    )


def test_resume_key_is_the_matchup_and_its_protocol(tmp_path: Path) -> None:
    """A 20-game smoke row must not stand in for the 200-game protocol row.

    `(checkpoint, anchor)` alone let a smoke pass, a different-sims diagnostic or a
    different ply cap block the real measurement for the same pair — and
    `scripts/eval_ladder.py` writes beside the checkpoint by default, i.e. into
    the campaign's own log (REVIEW-2026-10-21 D5). The eval row is keyed too,
    because a re-run of a completed pass used to append a second one (D4).
    """
    log = tmp_path / "ladder.jsonl"
    logschema.append_ladder_row(log, _match_row(anchor="random", games=20))
    logschema.append_ladder_row(log, _match_row(anchor="greedy", role="diagnostic"))
    done = logschema.completed_keys(log)
    protocol = logschema.match_key("runs/x.pt", "random", 200, 48, 300, "protocol")
    smoke = logschema.match_key("runs/x.pt", "random", 20, 48, 300, "protocol")
    assert smoke in done and protocol not in done
    assert logschema.match_key("runs/x.pt", "greedy", 200, 48, 300, "protocol") not in done
    assert logschema.match_key("runs/x.pt", "greedy", 200, 48, 300, "diagnostic") in done
    # Every protocol parameter is part of the identity.
    base = ("runs/x.pt", "random", 200, 48, 300, "protocol")
    for i, alt in ((2, 100), (3, 128), (4, 160), (5, "diagnostic")):
        changed = list(base)
        changed[i] = alt
        assert logschema.match_key(*changed) != logschema.match_key(*base)
    # A pre-v2 row has no role field; it lives in a campaign log, so it is protocol.
    v1 = {
        "row_type": "match",
        "checkpoint": "c",
        "anchor": "a",
        "games": 200,
        "match_sims": 48,
        "max_plies": 300,
        "schema_version": 1,
    }
    assert logschema.ladder_key(v1) == logschema.match_key("c", "a", 200, 48, 300, "protocol")
    assert logschema.eval_key("c", 128, "protocol") != logschema.eval_key("c", 16, "protocol")


def test_appending_after_a_torn_tail_keeps_every_row_readable(tmp_path: Path) -> None:
    """The kill-then-resume sequence that used to poison a log (REVIEW-2026-10-21 D1).

    A torn fragment with no newline, then an append: the new row must be its own
    line, visible to the resume key set, and the file must still read after a
    *second* append — which is where the old behaviour raised for the whole file.
    """
    log = tmp_path / "ladder.jsonl"
    first = _match_row(anchor="random")
    logschema.append_ladder_row(log, first)
    with log.open("a") as fh:
        fh.write('{"row_type": "match", "checkpoint": "run')  # killed mid-write
    second = _match_row(anchor="greedy")
    logschema.append_ladder_row(log, second)
    assert logschema.read_jsonl(log) == [first, second]
    assert logschema.ladder_key(second) in logschema.completed_keys(log)
    third = _match_row(anchor="phase1", anchor_digest="d" * 64)
    logschema.append_ladder_row(log, third)
    assert [r["anchor"] for r in logschema.read_jsonl(log)] == ["random", "greedy", "phase1"]
    # Nothing but the fragment was cut: the first row is byte-identical.
    assert log.read_text().splitlines()[0] == log.read_text().splitlines()[0].strip()


_RECORD_LOGS = sorted(
    p
    for pattern in ("runs/*/ladder*.jsonl", "runs/*/iterations.jsonl", "runs/*/reps_*.jsonl")
    for p in REPO.glob(pattern)
)


@pytest.mark.parametrize("log", _RECORD_LOGS, ids=lambda p: str(p.relative_to(REPO)))
def test_every_committed_row_validates_under_its_own_version(log: Path) -> None:
    """A committed row never becomes invalid because the code moved on.

    Until 2026-10-21 one SCHEMA_VERSION served both row families, so the ladder's
    bump to 3 made the validator reject the 70 unchanged iteration rows of p2_a and
    p2_b, and `nondeterministic` added to version 3 after p2_d's rows were written
    rejected 17 more (REVIEW-2026-10-21 P10, P1). These files are tracked, so this
    runs on every clone.
    """
    rows = logschema.read_jsonl(log)
    assert rows, f"{log} is empty"
    keys: list[tuple] = []
    for row in rows:
        if "row_type" in row:
            logschema.validate_ladder(row)
            keys.append(logschema.ladder_key(row))
        else:
            logschema.validate(row)
    assert len(keys) == len(set(keys)), f"{log} holds two rows for one resume key"


def test_an_eval_row_says_why_calibration_is_missing() -> None:
    """None means "not measured" (the house rule); the note says which.

    Calibration samples the phase1 anchor matchup, so a partial pass legitimately
    measures nothing — and a bare null leaves a reader to decide for themselves
    whether the value head was uncalibrated or simply never asked.
    """
    base = {
        "checkpoint": "runs/x.pt",
        "iteration": None,
        "suite_sims": 128,
        "mate2_solved": 197,
        "mate2_total": 200,
        "mate2_rate": 0.985,
        "mate2_baseline": 0.985,
        "mate2_no_regress": True,
        "mate2_m3_clause": True,
        "mate2_failed": ["mate2_129"],
        "mate3_solved": 138,
        "mate3_total": 200,
        "mate3_rate": 0.69,
        "watchlist": [],
        "ece": None,
        "calibration_n": None,
        "calibration_bins": [],
        "calibration_note": "not measured — calibration samples the phase1 anchor matchup",
        "book_move1_entropy": 3.1623,
        "seconds": 90.0,
        "git_sha": "abc1234",
        "config_fingerprint": "f" * 64,
    }
    row = logschema.ladder_eval_row(**base)
    logschema.validate_ladder(row)
    assert row["ece"] is None and "not measured" in row["calibration_note"]

    with pytest.raises(ValueError, match="null"):
        logschema.validate_ladder({**row, "calibration_note": None})


def test_watchlist_is_read_from_the_record_not_re_derived() -> None:
    """These are gate-suite positions; re-harvesting would eventually change them."""
    entries = load_watchlist()
    assert [e["id"] for e in entries] == ["mate2_129", "mate2_165", "mate2_166"]
    suite = {e for e, _i in load_epd(SUITES / "mate_in_2.epd")}
    ids = {i for _e, i in load_epd(SUITES / "mate_in_2.epd")}
    assert all(e["id"] in ids for e in entries)
    assert suite, "the gate suite must be present for the watchlist to mean anything"
    for e in entries:
        board = chess.Board(e["fen"])
        assert any(move_solves(board, mv, 2) for mv in board.legal_moves)


def test_book_entropy_is_finite_and_bounded() -> None:
    """The ladder-side twin of the loop's H_prior, on positions never trained on."""
    from tinyalphazero.config import ModelConfig
    from tinyalphazero.model import ChessNet

    net = ChessNet(
        ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, d_policy=16, dropout=0.0)
    ).eval()
    h = book_move1_entropy(net, load_openings()[:5])
    assert 0.0 < h <= np.log(218), h


def test_generator_scripts_are_runnable() -> None:
    """Provenance is only provenance if the script still runs."""
    import sys

    for script in (
        "generate_openings.py",
        "generate_mate3.py",
        "eval_ladder.py",
        "eval_reps.py",
        "eval_track1.py",
        "run_suite.py",
        "selfplay_smoke.py",
        "fuzz_encoding.py",
    ):
        # The interpreter running the tests, not whatever `python` is on PATH: on
        # this box there is none outside an activated venv (REVIEW-2026-10-21 D13).
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / script), "--help"],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        assert r.returncode == 0, r.stderr[-2000:]


# ---------------------------------------------------------------------------
# Row roles (schema_version 2)
# ---------------------------------------------------------------------------


def test_ladder_rows_carry_role_and_provenance() -> None:
    """A row states what it is for, so a reader cannot mistake one kind for another.

    The failure this prevents is not hypothetical: p2_b's Stockfish rungs are
    climbed until one resolves unclamped, and M4's criterion is a match against a
    *fixed* skill 0. Both land in the same log. A skill-5 diagnostic row that
    announces itself cannot be read into M4's gate six weeks later by someone who
    never opened the pre-registration.
    """
    from tinyalphazero import logschema

    base = {
        "checkpoint": "runs/x/ckpt_iter0001.pt",
        "iteration": 1,
        "anchor": "stockfish_s5",
        "games": 200,
        "wins": 100,
        "draws": 0,
        "losses": 100,
        "score": 0.5,
        "elo": 0.0,
        "elo_lo": -50.0,
        "elo_hi": 50.0,
        "elo_clamped": False,
        "draw_rate": 0.0,
        "mean_plies": 60.0,
        "match_sims": 48,
        "max_plies": 300,
        "seconds": 1.0,
        "git_sha": "abc",
        "config_fingerprint": "f" * 64,
        "anchor_digest": None,
        "nondeterministic": True,
    }
    default = logschema.ladder_match_row(**base)
    assert default["role"] == logschema.ROLE_PROTOCOL
    assert default["produced"] == logschema.PRODUCED_OFFLINE

    diag = logschema.ladder_match_row(
        **base, role=logschema.ROLE_DIAGNOSTIC, produced=logschema.PRODUCED_INLINE
    )
    assert diag["role"] == "diagnostic"

    with pytest.raises(ValueError, match="role"):
        logschema.ladder_match_row(**base, role="whatever-i-felt-like")
    with pytest.raises(ValueError, match="produced"):
        logschema.ladder_match_row(**base, produced="somehow")


def test_pre_v2_rows_report_no_role_rather_than_a_guessed_one() -> None:
    """Absence carries a reason: an old row's role lives in its file name, not in it.

    Defaulting a pre-v2 row to `protocol` would silently relabel every row in
    `ladder_diagnostic.jsonl` as something a gate may read — the exact misreading
    the field was added to prevent, introduced by the field itself.
    """
    from tinyalphazero import logschema

    assert logschema.ladder_role({"row_type": "match", "schema_version": 1}) is None
    assert logschema.ladder_role({"role": "diagnostic"}) == "diagnostic"


def test_ladder_pass_closes_a_subprocess_owning_opponent() -> None:
    """An opponent holding a subprocess is released when its matchup ends.

    Regression for 2026-08-12: `StockfishOpponent.close()` shipped in chunk 10 and
    was called by nothing. `chess.engine`'s transport thread is non-daemon, so the
    first Stockfish rung ever run wrote its row correctly and then hung at
    interpreter exit for seven hours. The measurement was never wrong; the process
    simply would not end, and the chain behind it never advanced.

    Asserted through `run_ladder_pass` rather than on the opponent, because the
    defect was never in `close()` — it was in nobody calling it.
    """
    from tinyalphazero import evaluate as ev

    closed: list[str] = []

    class _FakeEngineOpponent:
        name = "stockfish_s0"
        kind = "move"

        def choose(self, board, rng):  # noqa: ANN001, ARG002
            moves = list(board.legal_moves)
            return moves[int(rng.integers(len(moves)))]

        def close(self) -> None:
            closed.append(self.name)

    def _fake_play_match(*_a, **_k):
        raise RuntimeError("matchup blew up mid-play")

    cfg = load_config(None)
    # run_ladder_pass imports these inside the function, so they are patched at
    # their source modules rather than on `evaluate`.
    with (
        mock.patch("tinyalphazero.opponents.build_opponent", lambda *a, **k: _FakeEngineOpponent()),
        mock.patch.object(ev, "play_match", _fake_play_match),
        mock.patch("tinyalphazero.train.load_checkpoint", lambda *a, **k: (mock.MagicMock(), {})),
        mock.patch("tinyalphazero.mcts.batch_runner.torch_evaluate_batch", lambda *a, **k: None),
        mock.patch("tinyalphazero.mcts.search.network_evaluator_from_net", lambda *a, **k: None),
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(RuntimeError, match="blew up"),
    ):
        ev.run_ladder_pass(
            Path(tmp) / "ckpt.pt",
            cfg,
            "cpu",
            out=Path(tmp) / "ladder.jsonl",
            anchors="stockfish:0",
            suites=False,
        )
    # Released even though the matchup raised: the finally clause is the point.
    assert closed == ["stockfish_s0"], "the opponent's subprocess was never released"


def test_stockfish_rung_is_resumable_by_name_not_spec() -> None:
    """A logged Stockfish matchup is skipped on re-run, like every other anchor.

    Regression for 2026-08-12: rows record `result.anchor` (the opponent's *name*)
    while the skip set was keyed on the *spec* from the CLI. For random, greedy
    and phase1 those strings are identical, so resumability worked everywhere it
    was ever exercised. "stockfish:0" resolves to a row named "stockfish_s0", so
    the one anchor family whose spec carries a parameter silently recomputed a
    five-minute matchup and appended a duplicate row.

    This is what made the "surgical by construction" claim about backfilling onto
    a live ladder true for three rungs and false for the fourth.
    """
    from tinyalphazero import evaluate as ev
    from tinyalphazero import logschema

    played: list[str] = []

    class _Opp:
        name = "stockfish_s0"
        kind = "move"

        def choose(self, board, rng):  # noqa: ANN001, ARG002
            raise AssertionError("a logged matchup must not be replayed")

        def close(self) -> None:
            pass

    def _spy_play_match(*_a, **_k):
        played.append("ran")
        raise AssertionError("a logged matchup must not be replayed")

    cfg = load_config(None)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ladder.jsonl"
        logschema.append_ladder_row(
            out,
            logschema.ladder_match_row(
                checkpoint=str(Path(tmp) / "ckpt.pt"),
                iteration=0,
                anchor="stockfish_s0",  # as a real row records it
                games=200,
                wins=3,
                draws=6,
                losses=191,
                score=0.03,
                elo=-603.9,
                elo_lo=-759.0,
                elo_hi=-502.6,
                elo_clamped=False,
                draw_rate=0.03,
                mean_plies=79.0,
                match_sims=48,
                max_plies=300,
                seconds=294.0,
                git_sha="abc",
                config_fingerprint="f" * 64,
                anchor_digest=None,
                nondeterministic=True,
                role=logschema.ROLE_DIAGNOSTIC,
            ),
        )
        with (
            mock.patch("tinyalphazero.opponents.build_opponent", lambda *a, **k: _Opp()),
            mock.patch.object(ev, "play_match", _spy_play_match),
            mock.patch(
                "tinyalphazero.train.load_checkpoint", lambda *a, **k: (mock.MagicMock(), {})
            ),
            mock.patch(
                "tinyalphazero.mcts.batch_runner.torch_evaluate_batch", lambda *a, **k: None
            ),
            mock.patch(
                "tinyalphazero.mcts.search.network_evaluator_from_net", lambda *a, **k: None
            ),
        ):
            rows = ev.run_ladder_pass(
                Path(tmp) / "ckpt.pt",
                cfg,
                "cpu",
                out=out,
                anchors="stockfish:0",  # the spec, which is NOT the logged name
                suites=False,
                # The logged row is diagnostic; a protocol pass would (rightly)
                # not treat it as done -- the resume key carries the role.
                role=logschema.ROLE_DIAGNOSTIC,
            )
    assert played == [], "the already-logged stockfish matchup was replayed"
    assert rows == []


def test_stockfish_opponent_is_a_context_manager() -> None:
    """Closure is an idiom, not something a caller must remember (cure for 1231b65)."""
    from tinyalphazero.opponents import StockfishOpponent

    opp = StockfishOpponent(path="/nonexistent", skill=0, nodes=10)
    calls: list[str] = []
    opp.close = lambda: calls.append("closed")  # type: ignore[method-assign]
    with opp as ctx:
        assert ctx is opp
    assert calls == ["closed"], "__exit__ did not release the engine"


def test_anchor_identity_is_the_single_spelling() -> None:
    """Row-writer and skip-set derive identity from one function (cure for 5bb9216)."""
    from tinyalphazero.opponents import RandomMover, StockfishOpponent, anchor_identity

    sf = StockfishOpponent(path="/nonexistent", skill=0, nodes=10)
    assert anchor_identity(sf) == "stockfish_s0" != "stockfish:0"
    assert anchor_identity(RandomMover()) == "random"


def test_watchlist_verdict_is_live_and_the_pinned_one_says_so() -> None:
    """A field that looks live must be live; a pinned one must carry its date."""
    import json

    from tinyalphazero.evaluate import WATCHLIST, load_watchlist

    entries = load_watchlist()
    assert entries, "watchlist is empty"
    for e in entries:
        assert "asof" in e, f"{e['id']} has no asof; its pinned verdict is undated"
    raw = json.loads(WATCHLIST.read_text())
    assert all(r.get("asof") for r in raw)


def test_opponent_families_declare_whether_they_randomize() -> None:
    """Construction gate: a new opponent family must answer the determinism question.

    Promoted from incident to law on 2026-08-13. This project has now paid three
    times for the same mistake — a property proven for a set, then silently
    extended when the set grew: promotion-free fixtures, the `stockfish_s0` dedup
    key, and the ladder's reproducibility claim, which was true for net-vs-net
    rungs and false for the skill-limited engine rung that arrived later.

    Measured, not assumed: five fresh engine processes per position, both
    polarities. Skill 0 returned two distinct moves on quiet positions; skill 20
    returned one every time.
    """
    from dataclasses import MISSING, fields

    from tinyalphazero.opponents import OPPONENT_SPECS, StockfishOpponent, build_opponent
    from tinyalphazero.provenance import sha256_file

    assert StockfishOpponent(path="/nonexistent", skill=0, nodes=10).nondeterministic is True
    assert StockfishOpponent(path="/nonexistent", skill=20, nodes=10).nondeterministic is False

    # The gate gates the *set*, not one member of it (REVIEW-2026-10-21 P3): every
    # family build_opponent can return declares the property as a bool. Stockfish
    # may be absent from the box, so that family is constructed directly.
    with tempfile.TemporaryDirectory() as tmp:
        fake_anchor = Path(tmp) / "anchor.pt"
        fake_anchor.write_bytes(b"not a real checkpoint")
        for spec in OPPONENT_SPECS:
            if spec.startswith("stockfish"):
                opp = StockfishOpponent(path="/nonexistent", skill=int(spec.split(":")[1]), nodes=1)
            else:
                opp = build_opponent(spec, anchor_ckpt=fake_anchor, anchor_digest="")
            assert isinstance(opp.nondeterministic, bool), spec
            if spec == "phase1":
                assert opp.digest == sha256_file(fake_anchor)
    # And the row cannot fall back to "deterministic": the field has no default.
    (nd,) = [f for f in fields(logschema.LadderMatchRow) if f.name == "nondeterministic"]
    assert nd.default is MISSING and nd.default_factory is MISSING


def test_ladder_row_carries_the_nondeterminism_flag() -> None:
    """The row says whether it is reproducible, so a reader never has to know."""
    from tinyalphazero import logschema

    base = {
        "checkpoint": "runs/x/ckpt.pt",
        "iteration": 20,
        "anchor": "stockfish_s0",
        "games": 200,
        "wins": 37,
        "draws": 19,
        "losses": 144,
        "score": 0.2325,
        "elo": -207.5,
        "elo_lo": -260.5,
        "elo_hi": -159.8,
        "elo_clamped": False,
        "draw_rate": 0.095,
        "mean_plies": 119.9,
        "match_sims": 48,
        "max_plies": 300,
        "seconds": 427.0,
        "git_sha": "abc",
        "config_fingerprint": "f" * 64,
        "anchor_digest": None,
    }
    assert logschema.ladder_match_row(**base, nondeterministic=False)["nondeterministic"] is False
    assert logschema.ladder_match_row(**base, nondeterministic=True)["nondeterministic"] is True
    with pytest.raises(TypeError):  # no default to fall back to (REVIEW-2026-10-21 P3)
        logschema.ladder_match_row(**base)
    good = logschema.ladder_match_row(**base, nondeterministic=True)
    with pytest.raises(ValueError, match="bool"):
        logschema.validate_ladder({**good, "nondeterministic": "no"})


def test_rep_verdict_reproduces_the_committed_p2_d_statistic() -> None:
    """`paired_rep_verdict` on the committed rep rows gives `p2_verdict_reps.json`.

    The verdict P2 was closed on was computed by a script that was never committed
    (REVIEW-2026-10-21 P5). It is a library function now, and this pins it to the
    artefact: same rep scores, means, paired difference, bootstrap CI and SE.
    """
    import json

    from tinyalphazero.evaluate import paired_rep_verdict

    def rows(run: str) -> list[dict]:
        return [
            logschema.read_jsonl(REPO / "runs" / run / f"reps_iter0020_r{k}.jsonl")[0]
            for k in (1, 2, 3)
        ]

    recorded = json.loads((REPO / "runs" / "p2_d" / "p2_verdict_reps.json").read_text())
    got = paired_rep_verdict(rows("p2_c"), rows("p2_d"), "p2_c", "p2_d")
    assert got.keys() == recorded.keys()
    for key in ("p2_c_reps", "p2_d_reps", "openings", "excludes_zero"):
        assert got[key] == recorded[key], key
    for key in ("p2_c_mean", "p2_d_mean", "paired_difference", "se"):
        assert got[key] == pytest.approx(recorded[key], abs=1e-12), key
    assert got["ci95"] == pytest.approx(recorded["ci95"], abs=1e-12)
    assert got["paired_difference"] == pytest.approx(0.1033, abs=5e-5)
    assert got["excludes_zero"] is True


def test_phase1_rung_refuses_a_checkpoint_with_the_wrong_digest() -> None:
    """The anchor is pinned by digest in the code, not only in a document.

    `runs/phase1_anchor.pt` is a 78 MB untracked file at a path two anchor versions
    have lived at; a restored v1 would have moved the ladder's zero point with no
    trace in any row (REVIEW-2026-10-21 D3).
    """
    from tinyalphazero.opponents import build_opponent
    from tinyalphazero.provenance import sha256_file

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "anchor.pt"
        ckpt.write_bytes(b"weights")
        digest = sha256_file(ckpt)
        with pytest.raises(ValueError, match="pinned anchor"):
            build_opponent("phase1", anchor_ckpt=ckpt, anchor_digest="0" * 64)
        ok = build_opponent("phase1", anchor_ckpt=ckpt, anchor_digest=digest)
        assert ok.digest == digest
        unchecked = build_opponent("phase1", anchor_ckpt=ckpt, anchor_digest="")
        assert unchecked.digest == digest  # recorded even when not enforced


def test_one_opening_match_reports_no_interval_and_still_serialises() -> None:
    """`±inf` is not JSON; the bounds are None, "not measured" (REVIEW-2026-10-21 D14)."""
    import json

    from tinyalphazero.evaluate import paired_bootstrap_elo

    elo, lo, hi, _clamped = paired_bootstrap_elo(np.array([0.5]), n_games=2)
    assert elo == 0.0 and lo is None and hi is None
    row = _match_row(games=2, elo_lo=None, elo_hi=None)
    text = json.dumps(row)
    assert "Infinity" not in text
    assert json.loads(text)["elo_lo"] is None


def test_stockfish_searches_each_game_from_its_own_table() -> None:
    """`play()` carries a game key and a watchdog ceiling (REVIEW-2026-10-21 P9, D12).

    Without the key one engine process kept its transposition table across the
    ~200 games of a matchup, so a move in game k depended on games 1..k-1 and on
    the scheduler's interleaving. Without the ceiling python-chess arms no timeout
    for a nodes-only limit and a hung engine blocks forever.
    """
    import chess.engine

    from tinyalphazero.opponents import StockfishOpponent

    calls: list[dict] = []

    class _Engine:
        def play(self, board, limit, game=None):  # noqa: ANN001
            calls.append({"limit": limit, "game": game})
            return chess.engine.PlayResult(next(iter(board.legal_moves)), None)

        def quit(self) -> None:
            raise chess.engine.EngineTerminatedError("engine event loop dead")

    opp = StockfishOpponent(path="/nonexistent", skill=0, nodes=777)
    opp._engine = _Engine()
    rng = np.random.default_rng(0)
    a, b = chess.Board(), chess.Board()
    opp.choose(a, rng)
    opp.choose(b, rng)
    opp.choose(a, rng, game="g1")
    assert [c["limit"].nodes for c in calls] == [777, 777, 777]
    assert all(c["limit"].time == opp.time_ceiling_s for c in calls)
    assert calls[0]["game"] == id(a) != calls[1]["game"] == id(b)
    assert calls[2]["game"] == "g1"
    opp.close()  # the dead engine's own error must not replace the caller's
    assert opp._engine is None


def test_ladder_rows_stamp_the_launch_sha_and_the_config_fingerprint() -> None:
    """Rows name the code that produced them and the config they ran under.

    `git_sha` used to be HEAD at *write time* — p2_b's rows carry four SHAs from
    one process (REVIEW-2026-10-21 C2/D2) — and nothing in a row said which config
    a backfill used (D6). The pass takes the SHA from its caller and stamps it, and
    both row types carry `config_fingerprint`.
    """
    from tinyalphazero import evaluate as ev
    from tinyalphazero import logschema
    from tinyalphazero.config import config_fingerprint
    from tinyalphazero.evaluate import MatchResult

    def _fake_match(_evaluate, opponent, cfg, **kw):
        return MatchResult(
            anchor=opponent.name,
            games=2,
            wins=1,
            draws=0,
            losses=1,
            score=0.5,
            elo=0.0,
            elo_lo=None,
            elo_hi=None,
            elo_clamped=False,
            draw_rate=0.0,
            mean_plies=40.0,
            seconds=0.1,
            pair_scores=[0.5],
            traces=[],
        )

    cfg = load_config(None)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ladder.jsonl"
        with (
            mock.patch.object(ev, "play_match", _fake_match),
            mock.patch(
                "tinyalphazero.train.load_checkpoint", lambda *a, **k: (mock.MagicMock(), {})
            ),
            mock.patch(
                "tinyalphazero.mcts.batch_runner.torch_evaluate_batch", lambda *a, **k: None
            ),
            mock.patch(
                "tinyalphazero.mcts.search.network_evaluator_from_net", lambda *a, **k: None
            ),
        ):
            rows = ev.run_ladder_pass(
                Path(tmp) / "ckpt.pt",
                cfg,
                "cpu",
                out=out,
                anchors="random,greedy",
                suites=False,
                git_sha="launch00",
            )
    assert [r["anchor"] for r in rows] == ["random", "greedy"]
    for r in rows:
        logschema.validate_ladder(r)
        assert r["git_sha"] == "launch00"
        assert r["config_fingerprint"] == config_fingerprint(cfg)
        assert r["nondeterministic"] is False and r["anchor_digest"] is None
        assert r["schema_version"] == logschema.LADDER_SCHEMA_VERSION
