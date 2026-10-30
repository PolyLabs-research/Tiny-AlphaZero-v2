"""The anchor ladder: matches, Elo with paired-bootstrap CIs, suites (spec §8).

This is the module M3's go/no-go is read off. v1 never had a defensible "is it
actually improving" measurement, and everything upstream exists to make one plot
trustworthy — so the rules that make it trustworthy are stated here rather than
distributed across call sites.

**Profile: every ladder participant that searches runs the evaluation profile**
(``root_noise=False``: deterministic top-m by raw logits, acting argmax on
``logits + σ(q̂)``). Self-play noise exists to diversify *training data*; carrying
it into a match measures exploration on a match-play question, and a rung that
moves ±2.5 points under reseeding cannot resolve Elo.

**Sims: matches at ``eval.eval_match_sims`` (48), suites at ``eval.suite_sims``
(128), and the two are never conflated.** 128 is the frozen definition of the
mate-in-2 gate — the number every recorded gate result was measured at — while 48
is self-play-comparable strength at affordable cost. Running the gate at 48 would
produce a number that is not comparable to any gate on record; running 200-game
matches at 128 would cost 2.7× for a strength question that does not need it.

**Openings: a frozen 4-ply book, each played twice with colours swapped.** The
pairing is what the CI is computed over — see :func:`paired_bootstrap_elo`.
Resampling games rather than opening *pairs* would treat the two halves of a
colour-swapped pair as independent when they share an opening, and would report a
CI narrower than the evidence supports.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import chess
import numpy as np

from tinyalphazero.config import Config
from tinyalphazero.mcts.batch_runner import BatchRunner, GameTrace
from tinyalphazero.mcts.search import load_epd, run_suite
from tinyalphazero.opponents import GreedyMaterial, NetSearchOpponent, RandomMover
from tinyalphazero.selfplay import fill_outcomes

REPO = Path(__file__).resolve().parents[2]
OPENINGS = REPO / "tests" / "suites" / "openings_4ply.epd"
WATCHLIST = REPO / "benchmarks" / "m3_watchlist.json"


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


def elo_from_score(score: float, n_games: int) -> tuple[float, bool]:
    """Elo difference implied by a score in [0, 1]. Returns ``(elo, clamped)``.

    A clean sweep has infinite implied Elo, which is true and useless. The score
    is clamped to ``[1/(2N), 1 - 1/(2N)]`` — half a game, the resolution of the
    measurement — so a 200-game whitewash reports "≥ +1043" rather than an
    infinity that breaks every downstream mean. ``clamped`` travels with the
    number so a reader is never left guessing whether a bound was hit.
    """
    eps = 1.0 / (2 * max(1, n_games))
    clamped = not (eps <= score <= 1.0 - eps)
    s = min(max(score, eps), 1.0 - eps)
    return 400.0 * math.log10(s / (1.0 - s)), clamped


def paired_bootstrap_elo(
    pair_scores: np.ndarray,
    n_games: int,
    resamples: int = 10_000,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> tuple[float, float | None, float | None, bool]:
    """Point Elo and a percentile CI, resampling **opening pairs** with replacement.

    ``pair_scores[i]`` is the learner's mean score over the two colour-swapped
    games of opening ``i``, so each resampled unit is one opening played from both
    sides. That is the unit the design actually randomises over; resampling
    individual games would double-count each opening's difficulty and shrink the
    interval below what the evidence supports.

    Returns ``(elo, lo, hi, clamped)`` where ``clamped`` refers to the point
    estimate. With fewer than two openings there is nothing to resample and the
    bounds are ``None`` — "not measured", the same word the row schema uses —
    rather than ``±inf``, which ``json.dumps`` writes as the non-JSON literals
    ``-Infinity``/``Infinity`` (REVIEW-2026-10-21 D14).
    """
    rng = rng or np.random.default_rng(0)
    pair_scores = np.asarray(pair_scores, dtype=np.float64)
    n_pairs = len(pair_scores)
    elo, clamped = elo_from_score(float(pair_scores.mean()), n_games)
    if n_pairs < 2:
        return elo, None, None, clamped

    idx = rng.integers(0, n_pairs, size=(resamples, n_pairs))
    means = pair_scores[idx].mean(axis=1)
    elos = np.array([elo_from_score(float(m), n_games)[0] for m in means])
    lo, hi = np.percentile(elos, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return elo, float(lo), float(hi), clamped


def paired_rep_verdict(
    a_rows: list[dict],
    b_rows: list[dict],
    a_name: str,
    b_name: str,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """p2_d's registered statistic (PREREG-p2_d.md Addendum 2; RUNLOG-p2_d.md §1).

    Each row is one offline replay of a nondeterministic rung (``pair_scores``
    per opening). Per opening, the *rep-mean* removes the engine's own
    randomisation; the paired difference of rep-means ``b - a`` over openings is
    then bootstrapped over openings — the unit the design randomises over, as in
    :func:`paired_bootstrap_elo`. ``ci95`` is the 2.5/97.5 percentile of the
    bootstrap means and ``se`` their standard deviation (ddof=1). The defaults —
    10,000 resamples, ``default_rng(0)`` — are the ones that reproduce
    ``runs/p2_d/p2_verdict_reps.json`` from the committed rep rows to the digit;
    a test pins that.

    Lives here so that the verdict is a library function with a test, not a
    script that was never committed (REVIEW-2026-10-21 P5).
    """

    def _stack(rows: list[dict], name: str) -> np.ndarray:
        if not rows:
            raise ValueError(f"{name}: no repetition rows")
        ps = np.array([r["pair_scores"] for r in rows], dtype=np.float64)
        if ps.ndim != 2 or ps.shape[1] == 0:
            raise ValueError(f"{name}: rows must carry equal-length pair_scores (schema >= 3)")
        return ps

    pa, pb = _stack(a_rows, a_name), _stack(b_rows, b_name)
    if pa.shape[1] != pb.shape[1]:
        raise ValueError(f"openings differ: {a_name} {pa.shape[1]} vs {b_name} {pb.shape[1]}")
    # A difference of rep-means against two different opponents is a number about nothing; from p2_l on, a run's
    # directory holds reps against more than one Stockfish skill (PREREG-p2_l), so the mix is refused, not computed.
    anchors = {str(r.get("anchor")) for r in a_rows + b_rows}
    if len(anchors) != 1:
        raise ValueError(f"rows from more than one anchor: {sorted(anchors)}")
    n = pa.shape[1]
    diff = pb.mean(axis=0) - pa.mean(axis=0)  # per-opening difference of rep-means
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    a_scores = [float(r["score"]) for r in a_rows]
    b_scores = [float(r["score"]) for r in b_rows]
    return {
        f"{a_name}_reps": a_scores,
        f"{b_name}_reps": b_scores,
        f"{a_name}_mean": sum(a_scores) / len(a_scores),
        f"{b_name}_mean": sum(b_scores) / len(b_scores),
        "paired_difference": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "se": float(boot.std(ddof=1)),
        "openings": int(n),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def one_sample_rep_interval(
    rows: list[dict], name: str, resamples: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """M4's interval: the mean-over-reps book score with a one-sample bootstrap CI.

    RUNLOG-p2_g §4 defined it at scoring (the registration had named the bar,
    0.675, but not the interval): per-opening rep-means bootstrapped over the 100
    openings, 10,000 resamples, seed 0, percentile interval — the one-sample
    counterpart of :func:`paired_rep_verdict`, and it draws the same opening
    indices for the same seed (``default_rng(seed).integers(0, n, size=(R, n))``).
    RUNLOG-p2_h and RUNLOG-p2_i report it at every judged checkpoint; a test pins
    p2_h/0060's value to the committed rep rows.
    """
    if not rows:
        raise ValueError(f"{name}: no repetition rows")
    ps = np.array([r["pair_scores"] for r in rows], dtype=np.float64)
    if ps.ndim != 2 or ps.shape[1] == 0:
        raise ValueError(f"{name}: rows must carry equal-length pair_scores (schema >= 3)")
    n = ps.shape[1]
    per_opening = ps.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    boot = per_opening[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        f"{name}_reps": [float(r["score"]) for r in rows],
        f"{name}_mean": float(per_opening.mean()),
        "ci95": [float(lo), float(hi)],
        "se": float(boot.std(ddof=1)),
        "openings": int(n),
    }


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------


@dataclass
class GameOutcome:
    opening: str
    learner_white: bool
    result: str
    score: float  # from the learner's side: 1.0 win, 0.5 draw, 0.0 loss
    plies: int


@dataclass
class MatchResult:
    anchor: str
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    elo: float
    elo_lo: float | None  # None: fewer than two openings, no interval to report
    elo_hi: float | None
    elo_clamped: bool
    draw_rate: float
    mean_plies: float
    seconds: float
    pair_scores: list[float] = field(default_factory=list)
    outcomes: list[GameOutcome] = field(default_factory=list)
    traces: list[GameTrace] = field(default_factory=list, repr=False)


def load_openings(path: str | Path = OPENINGS) -> list[tuple[str, str]]:
    """The frozen book as ``[(epd, id)]``."""
    return load_epd(path)


def book_rows_in_ring(
    ring_dir: str | Path, openings: list[tuple[str, str]] | None = None
) -> dict[str, Any]:
    """L2, the contamination check: rows of a replay ring equal to a book position.

    PREREG-p2_h registered the count (expected 0) and RUNLOG-p2_h §2 scored it by a
    script that was never committed; RUNLOG-p2_i §2 reports it per arm. The key is
    the position proper — squares, side to move, castling rights and the en passant
    file, ``encode_board()[:70]`` — not the halfmove bucket or the repetition
    count, which a stored row carries and an EPD does not. Every stored row is
    compared, so a game that starts off the book and plays into a book position
    counts: that is the mechanism RUNLOG-p2_h §2 found behind its four rows. The
    ring is opened read-only; ``book_rows`` lists ``(row, opening_id, iteration)``.
    """
    from tinyalphazero.encoding import encode_board
    from tinyalphazero.replay import ReplayBuffer

    book = load_openings() if openings is None else openings
    keys: dict[bytes, str] = {}
    for epd, ident in book:
        b = chess.Board()
        b.set_epd(epd)
        keys[encode_board(b, 0)[:70].tobytes()] = ident
    start_key = encode_board(chess.Board(), 0)[:70].tobytes()
    buf = ReplayBuffer.open(ring_dir, readonly=True)
    n = len(buf)
    boards = buf._arr["boards"][:n]
    iterations = buf._arr["iteration"][:n]
    hits: list[tuple[int, str, int]] = []
    at_start = 0
    distinct: set[bytes] = set()
    for i in range(n):
        key = boards[i, :70].tobytes()
        distinct.add(key)
        if key == start_key:
            at_start += 1
        ident = keys.get(key)
        if ident is not None:
            hits.append((i, ident, int(iterations[i])))
    return {
        "rows": int(n),
        "total_written": int(buf.total_written),
        "book_rows": hits,
        "rows_at_initial_position": at_start,
        "distinct_positions": len(distinct),
    }


def _score_for_learner(trace: GameTrace, board: chess.Board, learner_white: bool) -> float:
    """1.0 / 0.5 / 0.0 from the learner's side.

    Checkmate is read off the final board — the side to move at a checkmate is the
    side that has been mated — rather than inferred from whose turn it was in the
    trace, which is only the same thing when every ply was searched.
    """
    if trace.result != "checkmate":
        return 0.5
    loser_is_white = board.turn == chess.WHITE
    return 0.0 if loser_is_white == learner_white else 1.0


def play_match(
    learner_evaluate: Any,
    opponent: Any,
    cfg: Config,
    sims: int | None = None,
    device: str = "cpu",
    openings: list[tuple[str, str]] | None = None,
    seed: int = 0,
    max_openings: int | None = None,
) -> MatchResult:
    """Play the frozen book twice colour-swapped against ``opponent``.

    Both opponent kinds go through the one batched scheduler: a searching anchor
    via ``evaluator_for`` (two networks, one concurrent pool, one forward each per
    flush) and a non-searching one via ``move_fn`` (no tree, no simulation budget).
    A separate match loop would have been a second scheduler with a second set of
    adjudication rules — see ``BatchRunner._play``.
    """
    book = openings if openings is not None else load_openings()
    if max_openings is not None:
        book = book[:max_openings]
    sims = sims if sims is not None else cfg.eval.eval_match_sims

    boards, learner_white, opening_ids = _pairing(book)
    n_games = len(boards)
    # The evaluation profile, applied once, here — see the module docstring.
    match_cfg = replace(cfg.mcts, root_noise=False)
    runner = BatchRunner(
        learner_evaluate, match_cfg, device=device, max_plies=cfg.eval.eval_match_max_plies
    )

    def learner_side(gi: int) -> bool:
        return chess.WHITE if learner_white[gi] else chess.BLACK

    evaluator_for = None
    move_fn = None
    if getattr(opponent, "kind", "move") == "network":
        opponent_evaluate = opponent.evaluate_batch()

        def evaluator_for(gi: int, root_side: bool):  # noqa: F811
            return learner_evaluate if root_side == learner_side(gi) else opponent_evaluate
    else:
        rngs = [np.random.default_rng([seed, i]) for i in range(n_games)]

        def move_fn(gi: int, board: chess.Board):  # noqa: F811
            if board.turn == learner_side(gi):
                return None
            # Each game owns one Board object for its whole life, so an engine
            # opponent keys its transposition table on it (StockfishOpponent.choose).
            return opponent.choose(board, rngs[gi])

    t0 = time.perf_counter()
    traces = list(
        runner.play_games(
            n_games,
            seeds=[
                int(np.random.SeedSequence([seed, i]).generate_state(1)[0]) for i in range(n_games)
            ],
            num_sims=sims,
            boards=boards,
            evaluator_for=evaluator_for,
            move_fn=move_fn,
        )
    )
    seconds = time.perf_counter() - t0
    return _summarise(opponent.name, traces, runner, learner_white, opening_ids, seconds, cfg, seed)


def play_opponents(
    player: Any,
    opponent: Any,
    cfg: Config,
    openings: list[tuple[str, str]] | None = None,
    seed: int = 0,
    max_openings: int | None = None,
) -> MatchResult:
    """Two non-searching opponents over the same book, pairing and Elo machinery.

    Spec §8's floor calibration — greedy must beat random decisively — and the
    live counterpart to the Elo synthetics. Deliberately routed through the same
    protocol the real ladder uses rather than a bespoke loop: a sanity check that
    exercises different code from the thing it is sanity-checking is worth very
    little. Neither side searches, so ``move_fn`` answers every ply and no tree is
    ever built.
    """
    book = openings if openings is not None else load_openings()
    if max_openings is not None:
        book = book[:max_openings]

    boards, player_white, opening_ids = _pairing(book)
    n_games = len(boards)
    runner = BatchRunner(
        None, replace(cfg.mcts, root_noise=False), max_plies=cfg.eval.eval_match_max_plies
    )
    rngs = [np.random.default_rng([seed, i]) for i in range(n_games)]

    def move_fn(gi: int, board: chess.Board) -> chess.Move:
        side = chess.WHITE if player_white[gi] else chess.BLACK
        actor = player if board.turn == side else opponent
        return actor.choose(board, rngs[gi])

    t0 = time.perf_counter()
    traces = list(runner.play_games(n_games, num_sims=1, boards=boards, move_fn=move_fn))
    seconds = time.perf_counter() - t0
    return _summarise(opponent.name, traces, runner, player_white, opening_ids, seconds, cfg, seed)


def _pairing(book: list[tuple[str, str]]) -> tuple[list[chess.Board], list[bool], list[str]]:
    """Two games per opening — the player takes each colour once (spec §8).

    Colour-swapped pairs are the unit the CI resamples, so they must stay adjacent
    and in order: ``_summarise`` reshapes the score vector to (-1, 2) and would
    silently pair the wrong games if this interleaving ever changed.
    """
    boards: list[chess.Board] = []
    player_white: list[bool] = []
    ids: list[str] = []
    for epd, ident in book:
        for as_white in (True, False):
            b = chess.Board()
            b.set_epd(epd)
            boards.append(b)
            player_white.append(as_white)
            ids.append(ident or epd)
    return boards, player_white, ids


def _summarise(
    anchor: str,
    traces: list[GameTrace],
    runner: BatchRunner,
    player_white: list[bool],
    opening_ids: list[str],
    seconds: float,
    cfg: Config,
    seed: int,
) -> MatchResult:
    outcomes: list[GameOutcome] = []
    for i, trace in enumerate(traces):
        final_board = runner._games_debug[i].board
        # Fill per-position outcomes so the calibration pass has ground truth.
        # Reusing the actor's function rather than a second sign convention: a
        # flip here would report a well-calibrated head as badly calibrated, and
        # the ECE would look plausible either way.
        winner = None if trace.result != "checkmate" else (not final_board.turn)
        fill_outcomes(trace, winner)
        outcomes.append(
            GameOutcome(
                opening=opening_ids[i],
                learner_white=player_white[i],
                result=trace.result,
                score=_score_for_learner(trace, final_board, player_white[i]),
                plies=trace.plies,
            )
        )

    scores = np.array([o.score for o in outcomes])
    pair_scores = scores.reshape(-1, 2).mean(axis=1)  # (as white, as black) per opening
    elo, lo, hi, clamped = paired_bootstrap_elo(
        pair_scores,
        len(outcomes),
        resamples=cfg.eval.bootstrap_resamples,
        rng=np.random.default_rng([seed, 99]),
    )
    return MatchResult(
        anchor=anchor,
        games=len(outcomes),
        wins=int((scores == 1.0).sum()),
        draws=int((scores == 0.5).sum()),
        losses=int((scores == 0.0).sum()),
        score=float(scores.mean()),
        elo=elo,
        elo_lo=lo,
        elo_hi=hi,
        elo_clamped=clamped,
        draw_rate=float((scores == 0.5).mean()),
        mean_plies=float(np.mean([o.plies for o in outcomes])) if outcomes else 0.0,
        seconds=seconds,
        pair_scores=[float(p) for p in pair_scores],
        outcomes=outcomes,
        traces=traces,
    )


# ---------------------------------------------------------------------------
# Suites, watchlist, calibration, entropy
# ---------------------------------------------------------------------------


def suite_result(
    suite_path: Path, evaluate_fn: Any, cfg: Config, mate_in: int, sims: int | None = None
) -> tuple[int, int, list[str]]:
    """Run a mate suite at the frozen gate settings: ``suite_sims``, eval profile."""
    return run_suite(
        suite_path,
        sims if sims is not None else cfg.eval.suite_sims,
        evaluate_fn,
        replace(cfg.mcts, root_noise=False),
        mate_in=mate_in,
        rng=np.random.default_rng(0),
    )


def load_watchlist(path: str | Path = WATCHLIST) -> list[dict]:
    """The three residual gate positions, by name (TRAINING_STRATEGY.md).

    Read from the record rather than re-derived — they are gate-suite positions,
    and re-harvesting them would eventually produce a different three.
    """
    return json.loads(Path(path).read_text())


def _live_watchlist_verdict(net: Any, board: chess.Board, suite_cfg: Any, solved: bool) -> str:
    """Why this position fails *now*: SOLVED / COVERAGE / CAPABILITY.

    COVERAGE means no key move is inside the prior's considered set, so search
    never simulates one and no amount of value quality can help. CAPABILITY means
    a key move was considered and lost anyway. BUG is not diagnosed here — it
    needs the tree, which is ``scripts/triage_mate2.py``'s job; this is the two-way
    split that the ladder can afford on every pass.
    """
    import torch

    from tinyalphazero.encoding import encode_board
    from tinyalphazero.mcts.search import move_solves

    if solved:
        return "SOLVED"
    m = suite_cfg.resolved().gumbel_m
    device_t = next(net.parameters()).device
    tokens = torch.from_numpy(encode_board(board, 0).astype(np.int64)).unsqueeze(0)
    with torch.no_grad():
        logits = net(tokens.to(device_t)).policy_ft[0].reshape(4096)
    legal = torch.zeros(4096, dtype=torch.bool, device=logits.device)
    for mv in board.legal_moves:
        legal[mv.from_square * 64 + mv.to_square] = True
    order = torch.argsort(logits.masked_fill(~legal, float("-inf")), descending=True)
    considered = {int(c) for c in order[:m].tolist()}
    for mv in board.legal_moves:
        if move_solves(board, mv, 2) and mv.from_square * 64 + mv.to_square in considered:
            return "CAPABILITY"
    return "COVERAGE"


def watchlist_report(net: Any, evaluate_fn: Any, cfg: Config, device: str = "cpu") -> list[dict]:
    """Per watchlist position: does search solve it, and what is value on the key child?

    Both numbers, because they answer different questions. "Solved" is the visible
    outcome; the value head's estimate on the key move's child is the mechanism —
    all three positions are *won* and the Phase 1 head scores them at ≈ +0.03, so
    the number to watch converting is that one, and it can move a long way before
    the solve flips.
    """
    import torch

    from tinyalphazero.encoding import encode_board
    from tinyalphazero.mcts.search import Search, move_solves
    from tinyalphazero.model import value_scalar

    suite_cfg = replace(cfg.mcts, root_noise=False)
    device_t = next(net.parameters()).device
    out: list[dict] = []
    for entry in load_watchlist():
        board = chess.Board(entry["fen"])
        search = Search(evaluate_fn, suite_cfg, np.random.default_rng(0))
        result = search.run(board, None, cfg.eval.suite_sims)
        solved = move_solves(board, result.best_move(), 2)

        # Value on the key child: the position after a key move.
        key_value = None
        for mv in board.legal_moves:
            if not move_solves(board, mv, 2):
                continue
            board.push(mv)
            tokens = torch.from_numpy(encode_board(board, 0).astype(np.int64)).unsqueeze(0)
            with torch.no_grad():
                wdl = net(tokens.to(device_t)).value_wdl
            # Negated: the child is the *opponent's* turn, and the watchlist
            # records the value from the mating side's view. Without the flip the
            # number would improve when the position got worse.
            key_value = -float(value_scalar(wdl)[0])
            board.pop()
            break
        # Live verdict. The pinned one from the watchlist file describes why the
        # position failed on the day it was pinned, and printing it beside live
        # numbers made every ladder row since chunk 10 assert a stale fact — it
        # read "COVERAGE" for positions whose key move had since been re-promoted
        # into the considered set, and it steered a reading before being caught
        # (RUNLOG-p2_b §8). A field that looks live must be live; a field that is
        # pinned must say so, which is what baseline_asof is for.
        live = _live_watchlist_verdict(net, board, suite_cfg, bool(solved))
        out.append(
            {
                "id": entry["id"],
                "verdict": live,
                "baseline_verdict": entry["verdict"],
                "baseline_asof": entry.get("asof", "unknown"),
                "solved": bool(solved),
                "value_on_key_child": None if key_value is None else round(key_value, 4),
                "baseline_value": entry["value_head_on_key_child"],
            }
        )
    return out


def value_calibration(
    net: Any, traces: list[GameTrace], max_positions: int = 5000, bins: int = 10, seed: int = 0
) -> dict:
    """10-bin ECE of predicted P(win) against realized outcome (spec §8).

    Positions come from the ladder's own anchor-matchup games, so the calibration
    is measured on the distribution the checkpoint actually plays into rather than
    on self-play data it was trained on. A draw counts as 0.5 realized: the value
    head predicts a distribution over win/draw/loss and scoring a draw as a loss
    would manufacture overconfidence that is not there.
    """
    import torch

    from tinyalphazero.model import value_scalar

    rows = [(pos.tokens, pos.outcome) for tr in traces for pos in tr.positions]
    if not rows:
        return {"ece": None, "n": 0, "bins": []}
    rng = np.random.default_rng(seed)
    if len(rows) > max_positions:
        pick = rng.choice(len(rows), size=max_positions, replace=False)
        rows = [rows[int(i)] for i in pick]

    tokens = torch.from_numpy(np.stack([r[0] for r in rows]).astype(np.int64))
    realized = np.array([(r[1] + 1) / 2.0 for r in rows])  # -1/0/+1 -> 0/0.5/1
    device = next(net.parameters()).device
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(tokens), 512):
            wdl = net(tokens[i : i + 512].to(device)).value_wdl
            preds.append(((value_scalar(wdl) + 1) / 2).float().cpu().numpy())
    predicted = np.concatenate(preds)

    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(predicted, edges) - 1, 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = which == b
        if not m.any():
            table.append({"bin": b, "n": 0, "predicted": None, "realized": None})
            continue
        p, r = float(predicted[m].mean()), float(realized[m].mean())
        ece += (m.sum() / len(predicted)) * abs(p - r)
        table.append(
            {"bin": b, "n": int(m.sum()), "predicted": round(p, 4), "realized": round(r, 4)}
        )
    return {"ece": round(float(ece), 4), "n": len(predicted), "bins": table}


def book_move1_entropy(net: Any, openings: list[tuple[str, str]] | None = None) -> float:
    """Mean prior entropy (nats) over the frozen book — the ladder-side H_prior.

    The twin of the loop's ``move1_prior_entropy``, measured on positions the
    campaign never trains on, so a collapse cannot hide behind familiarity with
    its own openings.
    """
    import torch

    from tinyalphazero.encoding import encode_board, legal_moves_packed
    from tinyalphazero.mcts import gumbel
    from tinyalphazero.train import unpack_legal_mask

    book = openings if openings is not None else load_openings()
    device = next(net.parameters()).device
    values: list[float] = []
    for epd, _ident in book:
        board = chess.Board()
        board.set_epd(epd)
        tokens = torch.from_numpy(encode_board(board, 0).astype(np.int64)).unsqueeze(0).to(device)
        legals = torch.from_numpy(legal_moves_packed(board)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = net(tokens).policy_ft[0].reshape(4096)
        mask = unpack_legal_mask(legals)[0].reshape(4096)
        p = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
        values.append(gumbel.entropy(p.float().cpu().numpy()))
    return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# One ladder pass
# ---------------------------------------------------------------------------

SUITES = REPO / "tests" / "suites"

#: The pinned M3 no-regress baseline (TRAINING_STRATEGY.md, 2026-08-10). A
#: constant, not something recomputed from the checkpoint under test — a baseline
#: the ladder derives from its own subject is not a baseline.
MATE2_BASELINE = 0.985
#: The M3 clause itself, which is absolute and does not move (spec §11).
M3_CLAUSE = 0.95


def run_ladder_pass(
    ckpt: str | Path,
    cfg: Config,
    device: Any,
    out: str | Path,
    anchors: str | None = None,
    anchor_ckpt: str | Path | None = None,
    iteration: int | None = None,
    seed: int | None = None,
    games: int | None = None,
    sims: int | None = None,
    stockfish: str = "",
    suites: bool = True,
    role: str = "protocol",
    produced: str = "offline",
    on_event: Any = None,
    git_sha: str | None = None,
    anchor_digest: str | None = None,
) -> list[dict]:
    """One complete ladder pass over ``ckpt``; returns the rows it appended.

    Lives in the library rather than in ``scripts/eval_ladder.py`` because two
    callers need it — the CLI and the campaign's ``--ladder-every`` hook — and the
    alternative was a library module importing a script. ``on_event(kind, payload)``
    is how the CLI gets its printing without this function acquiring a ``print``
    (AGENTS.md §4: library code raises, only scripts print).

    Every protocol parameter defaults to ``cfg.eval`` — anchors, anchor checkpoint
    and digest, openings, Stockfish path, seed, games, sims — so that the inline
    pass and an offline backfill of the same checkpoint under the same config are
    the same measurement (REVIEW-2026-10-21 C3/D6/D7). The rows record the
    config's fingerprint so a reader can check that they were.

    Resumable: a matchup already logged for this checkpoint *at these protocol
    parameters* is skipped (``logschema.match_key``), the opponent-free eval row
    likewise, and rows are appended as each finishes.

    ``git_sha`` is the SHA of the running code; a campaign passes the one it
    captured at launch. Left None it is read once here, for this pass — never
    once per row.
    """
    from tinyalphazero import logschema
    from tinyalphazero.config import config_fingerprint
    from tinyalphazero.mcts.batch_runner import torch_evaluate_batch
    from tinyalphazero.mcts.search import network_evaluator_from_net
    from tinyalphazero.opponents import anchor_identity, build_opponent
    from tinyalphazero.provenance import git_sha as read_git_sha
    from tinyalphazero.train import load_checkpoint

    ckpt = Path(ckpt)
    out = Path(out)
    ev = cfg.eval
    sims = sims if sims is not None else ev.eval_match_sims
    seed = ev.seed_or(cfg.seed) if seed is None else seed
    anchors = ev.ladder_anchors if anchors is None else anchors
    anchor_ckpt = Path(anchor_ckpt) if anchor_ckpt else REPO / ev.anchor_ckpt
    anchor_digest = ev.anchor_digest if anchor_digest is None else anchor_digest
    stockfish = stockfish or ev.stockfish_path
    sha = read_git_sha() if git_sha is None else git_sha
    fingerprint = config_fingerprint(cfg)

    net, _meta = load_checkpoint(ckpt, device=device)
    net.eval()
    batched = torch_evaluate_batch(net, device)
    single = network_evaluator_from_net(net, device)

    book = load_openings(REPO / ev.openings_path)
    # §8 fixes the matchup size at ``eval.anchor_games`` (200 = the whole book,
    # twice each). Taking it from config rather than defaulting to the full book is
    # what lets configs/mini.yaml run a real ladder pass in seconds instead of a
    # scaled-down imitation of one.
    n_games = games if games is not None else ev.anchor_games
    max_openings = max(1, min(len(book), n_games // 2))
    ckpt_key = str(ckpt)
    done = logschema.completed_keys(out)

    def emit(kind: str, payload: Any) -> None:
        if on_event is not None:
            on_event(kind, payload)

    emit(
        "start",
        {
            "checkpoint": ckpt_key,
            "device": str(device),
            "match_sims": sims,
            "suite_sims": cfg.eval.suite_sims,
            "openings": max_openings,
            "max_plies": cfg.eval.eval_match_max_plies,
            "out": str(out),
        },
    )

    rows: list[dict] = []
    anchor_traces: list[GameTrace] = []
    calib_sources: list[str] = []
    for spec in [s for s in anchors.split(",") if s]:
        # Build first, then test resumability on the opponent's *name*. Rows record
        # ``result.anchor``, which is the name, and for every anchor but one the
        # name and the spec are the same string — so keying the skip on the spec
        # worked for random/greedy/phase1 and silently failed for the family whose
        # spec carries a parameter: "stockfish:0" never matches the logged
        # "stockfish_s0", so a re-run recomputed a 5-minute matchup and appended a
        # duplicate row instead of skipping. Found 2026-08-12 alongside the
        # unreleased-engine bug, on the same first-ever Stockfish rung.
        # Construction is cheap for every opponent (checkpoints and engines both
        # load lazily), so building before the check costs nothing.
        opponent = build_opponent(
            spec,
            anchor_ckpt=anchor_ckpt,
            sims=sims,
            device=str(device),
            stockfish=stockfish,
            anchor_digest=anchor_digest,
        )
        if opponent is None:
            emit("skip", {"anchor": spec, "reason": "no Stockfish binary"})
            continue
        identity = anchor_identity(opponent)
        key = logschema.match_key(
            ckpt_key, identity, 2 * max_openings, sims, ev.eval_match_max_plies, role
        )
        if key in done:
            emit("skip", {"anchor": identity, "reason": "already in the log"})
            closer = getattr(opponent, "close", None)
            if callable(closer):
                closer()
            continue

        try:
            result = play_match(
                batched,
                opponent,
                cfg,
                sims=sims,
                device=str(device),
                openings=book,
                seed=seed,
                max_openings=max_openings,
            )
        finally:
            # An opponent that owns a subprocess must be told to let go of it.
            # StockfishOpponent.close() existed from chunk 10 and was called by
            # nobody, which stayed invisible for exactly as long as no Stockfish
            # rung had ever run: chess.engine's transport thread is non-daemon, so
            # the *measurement* completes and is logged and then the interpreter
            # hangs at exit, forever, holding the GPU-adjacent slot behind it.
            # Found 2026-08-12 on the first Stockfish rung ever run, after it sat
            # for 7 hours having already written its row.
            closer = getattr(opponent, "close", None)
            if callable(closer):
                closer()
        # Calibration positions. The phase1 matchup is one the learner wins ~99% of,
        # so its value distribution is degenerate — at p2_a pass 20, bins 0 and 9
        # held 4,652 of 5,000 positions between them, and ECE over that is easy to
        # pass and says little (RUNLOG-p2_a §6). A Stockfish matchup is the same
        # measurement where the model mostly *loses*, which is the un-saturated
        # half of the same instrument, and p2_c's lever is aimed exactly at value
        # quality. Both are sampled when both are run.
        if spec == "phase1" or spec.startswith("stockfish"):
            anchor_traces = anchor_traces + list(result.traces)
            calib_sources.append(identity)
        row = logschema.ladder_match_row(
            checkpoint=ckpt_key,
            iteration=iteration,
            anchor=result.anchor,
            games=result.games,
            wins=result.wins,
            draws=result.draws,
            losses=result.losses,
            score=round(result.score, 4),
            elo=round(result.elo, 2),
            elo_lo=None if result.elo_lo is None else round(result.elo_lo, 2),
            elo_hi=None if result.elo_hi is None else round(result.elo_hi, 2),
            elo_clamped=result.elo_clamped,
            draw_rate=round(result.draw_rate, 4),
            mean_plies=round(result.mean_plies, 2),
            match_sims=sims,
            max_plies=ev.eval_match_max_plies,
            seconds=round(result.seconds, 2),
            git_sha=sha,
            config_fingerprint=fingerprint,
            anchor_digest=getattr(opponent, "digest", None) or None,
            # Read directly: every family declares it (P3). An opponent that did
            # not would raise here, which is the point — "deterministic" is never
            # the answer a missing attribute falls back to.
            nondeterministic=bool(opponent.nondeterministic),
            role=role,
            produced=produced,
            pair_scores=[round(float(x), 4) for x in result.pair_scores],
        )
        logschema.append_ladder_row(out, row)
        rows.append(row)
        emit("match", row)

    if not suites:
        return rows
    if logschema.eval_key(ckpt_key, ev.suite_sims, role) in done:
        # A re-run of a completed pass used to re-measure the suites and append a
        # second eval row for the same checkpoint (REVIEW-2026-10-21 D4).
        emit("skip", {"anchor": logschema.EVAL_ROW, "reason": "already in the log"})
        return rows

    t0 = time.perf_counter()
    m2_solved, m2_total, m2_failed = suite_result(SUITES / "mate_in_2.epd", single, cfg, mate_in=2)
    m2_rate = m2_solved / m2_total if m2_total else 0.0

    m3_solved = m3_total = m3_rate = None
    if (SUITES / "mate_in_3.epd").exists():
        m3_solved, m3_total, _failed3 = suite_result(
            SUITES / "mate_in_3.epd", single, cfg, mate_in=3
        )
        m3_rate = m3_solved / m3_total if m3_total else 0.0

    watch = watchlist_report(net, single, cfg)
    if anchor_traces:
        calib = value_calibration(net, anchor_traces)
        calib_note = (
            f"measured on {calib['n']} positions from the "
            f"{' + '.join(calib_sources) or 'phase1'} matchup(s), "
            f"{len(calib['bins'])} bins"
        )
    else:
        calib = {"ece": None, "n": 0, "bins": []}
        calib_note = (
            "not measured — calibration samples the phase1 and stockfish matchups, "
            "and this pass ran neither"
        )
    entropy = book_move1_entropy(net, book)

    # "No regress" allows one position of slack: the baseline is a 200-position
    # proportion, so 196/200 against a 197/200 baseline is one position, not a
    # regression. Anything worse is reported as one.
    no_regress = m2_rate >= MATE2_BASELINE - (1.0 / max(1, m2_total))
    row = logschema.ladder_eval_row(
        checkpoint=ckpt_key,
        iteration=iteration,
        suite_sims=ev.suite_sims,
        mate2_solved=m2_solved,
        mate2_total=m2_total,
        mate2_rate=round(m2_rate, 4),
        mate2_baseline=MATE2_BASELINE,
        mate2_no_regress=bool(no_regress),
        mate2_m3_clause=bool(m2_rate >= M3_CLAUSE),
        mate2_failed=m2_failed,
        mate3_solved=m3_solved,
        mate3_total=m3_total,
        mate3_rate=None if m3_rate is None else round(m3_rate, 4),
        watchlist=watch,
        ece=calib["ece"],
        calibration_n=calib["n"] or None,
        calibration_bins=calib["bins"],
        calibration_note=calib_note,
        book_move1_entropy=round(entropy, 4),
        seconds=round(time.perf_counter() - t0, 2),
        git_sha=sha,
        config_fingerprint=fingerprint,
        role=role,
        produced=produced,
    )
    logschema.append_ladder_row(out, row)
    rows.append(row)
    emit("eval", row)
    return rows


def format_elo(row: dict) -> str:
    """Render a match row's Elo so a saturated result cannot read as a precise one.

    A clean sweep clamps the point estimate to the measurement's resolution *and*
    collapses the bootstrap: every resampled opening pair scores the same, so the
    interval degenerates to a point. Printed as ``+636.4 [+636.4, +636.4]`` that is
    the most precise-looking row in the table and the least informative one in it.

    Saturation is therefore shown as a bound, and a degenerate-but-unclamped
    interval is labelled too. Presentation only — the statistics are untouched, and
    the JSONL keeps the raw numbers plus ``elo_clamped`` for any reader that wants
    them.
    """
    elo, lo, hi = row["elo"], row["elo_lo"], row["elo_hi"]
    if row["elo_clamped"]:
        bound = "≥" if elo > 0 else "≤"
        return f"Elo {bound} {elo:+.1f} (saturated at {row['games']} games)"

    body = f"Elo {elo:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    # Both notes, independently. A self-match produces an interval that is
    # degenerate *and* straddles zero, and those say different things: the
    # straddle is the null result, the degeneracy is why it is exact rather than
    # approximate. Reporting only one of them loses whichever the reader needed.
    if lo <= 0.0 <= hi:
        body += "  CI straddles 0"
    if lo == hi:
        body += "  (degenerate CI — every opening pair scored alike)"
    return body


def format_ladder_event(kind: str, payload: dict) -> str | None:
    """Render one ladder event as a line, or None if it has nothing to say.

    A *formatter*, not a printer: two scripts want the same rendering (the ladder
    CLI and the campaign's ``--ladder-every`` hook), and the alternative to one
    library-side formatter was either a duplicated one or a script importing a
    script. Library code still does not print — the caller does.
    """
    if kind == "start":
        return (
            f"ladder | ckpt={Path(payload['checkpoint']).name} device={payload['device']} "
            f"match_sims={payload['match_sims']} suite_sims={payload['suite_sims']} "
            f"openings={payload['openings']} max_plies={payload['max_plies']} "
            f"-> {payload['out']}"
        )
    if kind == "skip":
        note = payload["reason"]
        if note == "no Stockfish binary":
            note = (
                "no Stockfish binary. It is an M4 calibration rung, not an M3 one; "
                "installing it is a system change and your call."
            )
        return f"  {payload['anchor']:12s} SKIPPED — {note}"
    if kind == "match":
        r = payload
        return (
            f"  {r['anchor']:12s} {r['wins']}W/{r['draws']}D/{r['losses']}L "
            f"score={r['score']:.3f}  {format_elo(r)}"
            f"  plies={r['mean_plies']:.0f}  {r['seconds'] / 60:.1f} min"
        )
    if kind == "eval":
        r = payload
        lines = [
            f"  mate-in-2    {r['mate2_solved']}/{r['mate2_total']} = {r['mate2_rate']:.3f} "
            f"(baseline {MATE2_BASELINE}, M3 clause >= {M3_CLAUSE}) "
            f"{'no-regress OK' if r['mate2_no_regress'] else 'REGRESSED'}"
            f"{'' if r['mate2_m3_clause'] else '  M3 CLAUSE FAILED'}"
        ]
        if r["mate3_rate"] is not None:
            lines.append(
                f"  mate-in-3    {r['mate3_solved']}/{r['mate3_total']} = "
                f"{r['mate3_rate']:.3f}  (informational, no gate)"
            )
        for w in r["watchlist"]:
            lines.append(
                f"  watch {w['id']:11s} {w['verdict']:11s} solved={str(w['solved']):5s} "
                f"value_on_key_child={w['value_on_key_child']} (was {w['baseline_value']})"
            )
        lines.append(f"  calibration: {r['calibration_note']}")
        if r["ece"] is not None:
            lines.append(f"  ECE {r['ece']} over {r['calibration_n']} positions")
        lines.append(f"  book H_prior {r['book_move1_entropy']:.4f} | {r['seconds'] / 60:.1f} min")
        return "\n".join(lines)
    return None


__all__ = [
    "M3_CLAUSE",
    "MATE2_BASELINE",
    "format_ladder_event",
    "run_ladder_pass",
    "GameOutcome",
    "GreedyMaterial",
    "MatchResult",
    "NetSearchOpponent",
    "RandomMover",
    "book_move1_entropy",
    "elo_from_score",
    "load_openings",
    "load_watchlist",
    "paired_bootstrap_elo",
    "play_match",
    "suite_result",
    "value_calibration",
    "watchlist_report",
]
