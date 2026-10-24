"""Self-play actor: games, resignation, adjudication, league (spec §7, chunk 8).

**Profile.** Self-play runs the *self-play* profile — ``root_noise=True`` — and
this is the file where that matters most, because it is the only consumer whose
purpose is data diversity rather than strength. Suites, the anchor ladder and
match play run the evaluation profile (deterministic root); see
``mcts/search.py``. Getting these the wrong way round does not fail: it quietly
produces either a noisy strength measurement or a self-play corpus with no
exploration in it.

**Value targets are depth-blind, deliberately.** The improved policy written
into a trace is orthodox completed-Q — it does not encode mate distance, so a
proven mate-in-5 and a proven mate-in-2 receive equal target mass. The acting
fix in chunk 6 already guarantees the shorter mate is the one *played*, so
games do not lengthen; only the target is indifferent. Standing ruling: leave it
until M3 shows mate-lengthening in played games, measured as mean plies of
decisive games (already in the iteration log). Revisiting earlier would mean
inventing a target weighting with no evidence to calibrate it against.

Smoke test (actor + replay ring, no training):

    python scripts/selfplay_smoke.py --ckpt runs/phase1_anchor.pt \\
        --games 8 --sims 16 --out runs/sp_smoke --device cpu
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np

from tinyalphazero.config import Config, MCTSConfig, SelfPlayConfig
from tinyalphazero.encoding import legal_moves_packed
from tinyalphazero.mcts import gumbel
from tinyalphazero.mcts.batch_runner import BatchRunner, GameTrace
from tinyalphazero.replay import ReplayBuffer

RESIGN_RESULT = "resign"

#: Opponent key standing for "the current learner weights" in a league assignment.
LATEST = "latest"


# ---------------------------------------------------------------------------
# League
# ---------------------------------------------------------------------------


class CheckpointPool:
    """The league's opponent pool (spec §7).

    **Exactly one implementation of this concept exists, and it is this one** —
    v1 carried two copies in two modules that drifted apart, which is defect 7 in
    the spec's post-mortem. If another module needs opponent sampling, it imports
    this.

    Per game: the latest weights with probability ``league_latest_frac``,
    otherwise a uniform draw from the last ``league_pool_size`` snapshots.
    """

    def __init__(self, cfg: SelfPlayConfig | None = None) -> None:
        self.cfg = cfg or SelfPlayConfig()
        self.snapshots: list[str] = []

    def add(self, path: str | Path) -> None:
        self.snapshots.append(str(path))
        excess = len(self.snapshots) - self.cfg.league_pool_size
        if excess > 0:
            del self.snapshots[:excess]

    @property
    def latest(self) -> str | None:
        return self.snapshots[-1] if self.snapshots else None

    def sample(self, rng: np.random.Generator) -> str | None:
        """One opponent. Returns None only when the pool is empty."""
        if not self.snapshots:
            return None
        if len(self.snapshots) == 1 or rng.random() < self.cfg.league_latest_frac:
            return self.snapshots[-1]
        return self.snapshots[int(rng.integers(len(self.snapshots)))]


@dataclass(frozen=True)
class GameAssignment:
    """Which weights play which colour in one game (§7's 80/20 league).

    §7 says *latest-vs-pool*, not pool-vs-pool: in a league game the learner plays
    one colour and a snapshot the other, so the two sides of a single game are
    searched by different networks. That is why the replay tag is per position
    (``mover_is_latest``) rather than per game, and why the batched runner needed
    a per-(game, colour) evaluator rather than one call per opponent.
    """

    white: str = LATEST
    black: str = LATEST

    def key_for(self, side: bool) -> str:
        """``side`` is ``chess.WHITE`` / ``chess.BLACK`` (i.e. ``board.turn``)."""
        return self.white if side == chess.WHITE else self.black


def assign_league(
    pool: CheckpointPool,
    num_games: int,
    rng: np.random.Generator,
    max_opponents: int = 4,
) -> list[GameAssignment]:
    """Per-game league draw, coalesced onto at most ``max_opponents`` snapshots.

    The draw itself is ``CheckpointPool.sample`` per game — chunk 8's contract,
    unchanged — so the *count* of league games is exactly §7's 20%. What the
    coalescing changes is only how many distinct snapshots one iteration spans:
    games that drew a snapshot outside the ``max_opponents`` most-drawn are
    reassigned uniformly among the ones that survived.

    The reason is wall clock, not statistics. A 20-snapshot pool spreads ~102
    league games over 20 networks, and a forward over five leaves costs the same
    GPU latency as one over 256; coalescing keeps each network's share wide enough
    to be worth a batch. Across a campaign every snapshot still appears — the
    marginal is preserved, only within-iteration diversity is traded.

    The learner's colour is drawn per game, so a systematic first-move advantage
    cannot accumulate on one side of the league.
    """
    latest = pool.latest
    draws = [pool.sample(rng) for _ in range(num_games)]

    counts: dict[str, int] = {}
    for d in draws:
        if d is not None and d != latest:
            counts[d] = counts.get(d, 0) + 1
    kept = sorted(counts, key=lambda k: (-counts[k], k))[: max(1, max_opponents)]

    out: list[GameAssignment] = []
    for d in draws:
        if d is None or d == latest:
            out.append(GameAssignment())
            continue
        opponent = d if d in kept else kept[int(rng.integers(len(kept)))]
        learner_is_white = bool(rng.random() < 0.5)
        out.append(
            GameAssignment(white=LATEST, black=opponent)
            if learner_is_white
            else GameAssignment(white=opponent, black=LATEST)
        )
    return out


# ---------------------------------------------------------------------------
# Resignation
# ---------------------------------------------------------------------------


@dataclass
class ResignTracker:
    """Spec §7 resignation, as a policy over a trace the runner is building.

    A side resigns when its own best root value has been below
    ``threshold`` for ``consecutive`` of *its own* moves. Counting own moves is
    the whole subtlety: a trace alternates sides, so a naive streak over trace
    rows resigns after four of each side's moves, not eight.

    ``disabled`` games play on. They are the measurement sample: among games
    where the rule *would* have fired, the fraction the would-be resigner did
    not ultimately lose is the false-resign rate, and it cannot be estimated at
    all from games that were stopped.
    """

    threshold: float = -0.90
    consecutive: int = 8
    disabled: bool = False
    streak: dict[bool, int] = field(default_factory=lambda: {True: 0, False: 0})
    would_resign_ply: int | None = None
    would_resign_side: bool | None = None

    def __call__(self, trace: GameTrace) -> str | None:
        pos = trace.positions[-1]
        side = pos.side_to_move
        if pos.root_q < self.threshold:
            self.streak[side] += 1
        else:
            self.streak[side] = 0

        if self.streak[side] >= self.consecutive and self.would_resign_ply is None:
            self.would_resign_ply = pos.ply
            self.would_resign_side = side
        if self.would_resign_ply is not None and not self.disabled:
            return RESIGN_RESULT
        return None


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def winner_of(trace: GameTrace, board: chess.Board, resign_side: bool | None) -> int | None:
    """Winning colour as ``chess.WHITE``/``chess.BLACK``, or None for a draw.

    ``"draw_claim"`` and ``"ply_cap"`` are draws by §7; a resignation is a loss
    for the resigner. Checkmate is read off the final board rather than inferred,
    because the side to move at a checkmate is the side that has been mated.
    """
    if trace.result == RESIGN_RESULT:
        return None if resign_side is None else (not resign_side)
    if trace.result == "checkmate":
        return not board.turn
    return None


def fill_outcomes(trace: GameTrace, winner: int | None) -> None:
    """Write +1/0/−1 into every position, from *that position's* mover's view.

    The per-position perspective flip is the part that is easy to get wrong and
    impossible to notice later: a sign error here trains the value head to
    predict the opponent's result, and every downstream metric still looks
    well-formed.
    """
    trace.outcome = 0 if winner is None else 1
    for pos in trace.positions:
        pos.outcome = 0 if winner is None else (1 if pos.side_to_move == winner else -1)


# ---------------------------------------------------------------------------
# The actor
# ---------------------------------------------------------------------------


def game_seed(base_seed: int, game_index: int) -> int:
    """Per-game seed fan-out (spec §9.2) so any single game is replayable alone.

    A run that can only be reproduced by replaying every game before it is not
    reproducible in practice — the point is being able to re-run game 4,731 of a
    24-hour run on its own when it produces something strange.
    """
    return int(np.random.SeedSequence([base_seed, game_index]).generate_state(1)[0])


@dataclass
class SelfPlayStats:
    games: int = 0
    positions: int = 0
    results: dict[str, int] = field(default_factory=dict)
    resign_disabled_games: int = 0
    would_have_resigned: int = 0
    false_resigns: int = 0
    seconds: float = 0.0
    sims: int = 0
    forwards: int = 0
    pool_positions: int = 0  # positions whose mover was a league opponent
    plies: list[int] = field(default_factory=list)
    decisive_plies: list[int] = field(default_factory=list)
    move1_entropies: list[float] = field(default_factory=list)
    move1_prior_entropies: list[float] = field(default_factory=list)

    @property
    def false_resign_rate(self) -> float | None:
        """Fraction of would-be resignations that were not in fact losses.

        None when no disabled game ever hit the threshold — an honest "not
        measured yet" rather than a 0.0 that reads as "calibrated".
        """
        if self.would_have_resigned == 0:
            return None
        return self.false_resigns / self.would_have_resigned

    @property
    def sims_per_s(self) -> float:
        return self.sims / self.seconds if self.seconds > 0 else 0.0

    @property
    def mean_plies(self) -> float:
        return float(np.mean(self.plies)) if self.plies else 0.0

    @property
    def mean_plies_decisive(self) -> float | None:
        """None, not 0.0, when no game was decided — see logschema's None rule."""
        return float(np.mean(self.decisive_plies)) if self.decisive_plies else None

    @property
    def move1_policy_entropy(self) -> float:
        """Mean entropy (nats) of the *improved policy* at each game's first move.

        This is the sharpness of the training target, and it is **not** spec §8's
        collapse detector — read ``move1_prior_entropy`` for that. Measured on the
        GPU smoke: between iterations 1 and 2 this fell 2.909 → 0.226 while the
        prior at the start position barely moved (2.972 → 2.778 of a 2.996 max).
        The cause is σ, which scales by ``(c_visit + max_N)`` ≈ 70: the value head
        went from indifferent (child-value range 0.010) to discriminating (0.885),
        and 0.885 × 70 is 62 logits of spread. That is Gumbel working as designed.
        """
        return float(np.mean(self.move1_entropies)) if self.move1_entropies else 0.0

    @property
    def move1_prior_entropy(self) -> float:
        """Mean entropy (nats) of the network's *prior* at each game's first move.

        Spec §8's collapse detector: the question is whether the policy still
        considers alternatives at all, and only the pre-search prior answers it.
        A network whose prior has collapsed onto one move cannot recover, because
        nothing outside its top-m is ever simulated.
        """
        return float(np.mean(self.move1_prior_entropies)) if self.move1_prior_entropies else 0.0

    def absorb(self, other: SelfPlayStats) -> None:
        """Fold another wave's stats into this one (spec §7 generates in waves).

        Per-game lists are concatenated rather than averaged: a mean of two means
        is only the mean when the waves are equal-sized, and the last wave of an
        iteration is a remainder.
        """
        self.games += other.games
        self.positions += other.positions
        self.resign_disabled_games += other.resign_disabled_games
        self.would_have_resigned += other.would_have_resigned
        self.false_resigns += other.false_resigns
        self.seconds += other.seconds
        self.sims += other.sims
        self.forwards += other.forwards
        self.pool_positions += other.pool_positions
        self.plies.extend(other.plies)
        self.decisive_plies.extend(other.decisive_plies)
        self.move1_entropies.extend(other.move1_entropies)
        self.move1_prior_entropies.extend(other.move1_prior_entropies)
        for k, v in other.results.items():
            self.results[k] = self.results.get(k, 0) + v

    def to_dict(self) -> dict:
        return {
            "games": self.games,
            "positions": self.positions,
            "results": self.results,
            "resign_disabled_games": self.resign_disabled_games,
            "would_have_resigned": self.would_have_resigned,
            "false_resigns": self.false_resigns,
            "false_resign_rate": self.false_resign_rate,
            "sims": self.sims,
            "forwards": self.forwards,
            "sims_per_s": round(self.sims_per_s, 1),
            "mean_plies": round(self.mean_plies, 1),
            "seconds": round(self.seconds, 2),
            "games_per_hour": (
                round(self.games / self.seconds * 3600, 1) if self.seconds > 0 else None
            ),
        }


@dataclass(frozen=True)
class GameSpec:
    """Everything that decides one self-play game before it is played.

    A game's trajectory is a function of its seed, its resignation flag, which
    weights search each colour, and the network's answers to its leaves — never of
    which other games share its batch (the BatchRunner's contract). Fixing these
    up front is what lets a wave be played in one process or split across many
    (``actors``) and still store the same rows.
    """

    seed: int
    resign_disabled: bool
    assignment: GameAssignment = GameAssignment()
    # The position the game is searched from; None is the initial position. Set by
    # the random-opening lever (``selfplay.opening_random_plies``).
    start_fen: str | None = None


@dataclass
class GameResult:
    """One played and scored game: the trace, with outcomes filled, and its verdict."""

    trace: GameTrace
    winner: int | None
    would_resign_side: bool | None


def wave_specs(
    seed: int,
    num_games: int,
    sp: SelfPlayConfig,
    assignments: list[GameAssignment] | None = None,
) -> list[GameSpec]:
    """The per-game draws for one wave, exactly as ``play_games`` has always made them.

    One ``rng.random(num_games)`` for the resignation flags, then ``game_seed(seed,
    i)`` per game. The order of those draws is the definition of the wave; it is
    written once, here, so the in-process and multi-process paths cannot drift.
    """
    rng = np.random.default_rng(seed)
    disabled_flags = rng.random(num_games) < sp.resign_disabled_frac
    excluded = _excluded_starts(sp.opening_exclude) if sp.opening_random_plies > 0 else set()
    return [
        GameSpec(
            seed=game_seed(seed, i),
            resign_disabled=bool(disabled_flags[i]),
            assignment=assignments[i] if assignments is not None else GameAssignment(),
            start_fen=(
                _random_opening(seed, i, sp.opening_random_plies, excluded)
                if sp.opening_random_plies > 0
                else None
            ),
        )
        for i in range(num_games)
    ]


#: Stream tag for the opening draw, so it cannot collide with ``game_seed``'s.
_OPENING_STREAM = 0x0BE7


def _random_opening(seed: int, game_index: int, max_plies: int, excluded: set[str]) -> str | None:
    """A start position k uniformly random legal plies in, k ~ U{0..max_plies}.

    Drawn from its own stream of (wave seed, game index), so a wave's other draws
    are untouched and each game's opening is replayable alone. A position that is
    over, or whose EPD is in ``excluded``, is redrawn (k included); after 1,000 tries
    the game starts from the initial position, which is never excluded. Returns
    None for the initial position, so k = 0 is indistinguishable from the lever off.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, game_index, _OPENING_STREAM]))
    for _ in range(1000):
        k = int(rng.integers(0, max_plies + 1))
        board = chess.Board()
        for _ply in range(k):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.integers(len(moves)))])
        if k == 0:
            return None
        if board.is_game_over() or board.epd() in excluded:
            continue
        return board.fen()
    return None


_EXCLUDED_CACHE: dict[str, set[str]] = {}


def _excluded_starts(path: str) -> set[str]:
    """EPD position keys of a suite, relative to the repo root unless absolute."""
    if not path:
        return set()
    if path not in _EXCLUDED_CACHE:
        from tinyalphazero.mcts.search import load_epd

        full = (
            Path(path) if Path(path).is_absolute() else Path(__file__).resolve().parents[2] / path
        )
        _EXCLUDED_CACHE[path] = {
            chess.Board.from_epd(line)[0].epd() for line, _id in load_epd(full)
        }
    return _EXCLUDED_CACHE[path]


def play_games(
    evaluate_batch,
    num_games: int,
    cfg: Config | None = None,
    seed: int = 0,
    num_sims: int | None = None,
    device: str = "cpu",
    iteration: int = 0,
    buffer: ReplayBuffer | None = None,
    evaluators: dict[str, Any] | None = None,
    assignments: list[GameAssignment] | None = None,
) -> tuple[list[GameTrace], SelfPlayStats]:
    """Play ``num_games`` self-play games, score them, and optionally store them.

    ``assignments`` turns on league play (§7): with it, ``evaluators`` must map
    every key the assignments name — including ``LATEST`` — to an
    ``evaluate_batch`` callable, and each leaf is answered by the network of the
    side that is *searching*. Without it every game is latest-vs-latest and every
    stored position is tagged ``mover_is_latest=True``, which is exactly chunk 8's
    behaviour.
    """
    cfg = cfg or Config()
    sp: SelfPlayConfig = cfg.selfplay
    num_sims = num_sims if num_sims is not None else sp.sims_per_move
    specs = wave_specs(seed, num_games, sp, assignments)

    stats = SelfPlayStats()
    t0 = time.perf_counter()
    results, runner_stats = play_specs(
        evaluate_batch,
        specs,
        cfg,
        num_sims,
        device=device,
        evaluators=evaluators,
        league=assignments is not None,
    )
    record_games(results, specs, stats, buffer, iteration)
    stats.seconds = time.perf_counter() - t0
    stats.sims = runner_stats["sims"]
    stats.forwards = runner_stats["forwards"]
    if buffer is not None:
        buffer.flush()
    return [r.trace for r in results], stats


def play_specs(
    evaluate_batch,
    specs: list[GameSpec],
    cfg: Config,
    num_sims: int,
    device: str = "cpu",
    evaluators: dict[str, Any] | None = None,
    league: bool = False,
) -> tuple[list[GameResult], dict[str, int]]:
    """Play ``specs`` concurrently in one runner and score them; store nothing.

    The part of self-play that an actor process can do on its own: it needs the
    games' specs and something that answers leaves, and returns scored traces in
    spec order plus the runner's counters. Storing is left to the caller because
    the ring has exactly one writer (``replay``'s contract, point 1).
    """
    mcts: MCTSConfig = cfg.mcts
    sp: SelfPlayConfig = cfg.selfplay
    num_games = len(specs)
    trackers = [
        ResignTracker(
            threshold=sp.resign_threshold,
            consecutive=sp.resign_consecutive,
            disabled=spec.resign_disabled,
        )
        for spec in specs
    ]

    # ONE runner for all games, so chunk 7's batching actually applies: with a
    # runner per game every forward is batch 1, which is the exact throughput
    # failure the batched runner exists to prevent. Resignation stays per game by
    # keying the trackers on trace identity — the runner hands the callback the
    # trace it is building, which is the only per-game handle it has.
    by_trace: dict[int, ResignTracker] = {}

    def stop_fn(trace: GameTrace) -> str | None:
        # Bound lazily: play_games is a generator, so the runner's game list does
        # not exist until iteration starts, which is after this closure is built.
        tr = by_trace.get(id(trace))
        if tr is None:
            for i, gs in enumerate(runner._games_debug):
                if gs.trace is trace:
                    tr = by_trace[id(trace)] = trackers[i]
                    break
        if tr is None:
            # Returning None here would silently disable resignation for this
            # game and nothing downstream could tell (REVIEW-2026-10-21 C10).
            raise RuntimeError("resignation callback received a trace the runner does not own")
        return tr(trace)

    evaluator_for = None
    if league:
        if evaluators is None:
            raise ValueError("league assignments given without an evaluators map")
        missing = {
            spec.assignment.key_for(s) for spec in specs for s in (chess.WHITE, chess.BLACK)
        } - set(evaluators)
        if missing:
            raise ValueError(f"no evaluator supplied for league key(s): {sorted(missing)}")

        def evaluator_for(gi: int, root_side: bool):  # noqa: F811
            return evaluators[specs[gi].assignment.key_for(root_side)]

    runner = BatchRunner(evaluate_batch, mcts, device=device, max_plies=sp.max_plies)
    boards = [chess.Board(spec.start_fen) if spec.start_fen else chess.Board() for spec in specs]

    # The runner creates its _GameState list up front, so the trackers can be
    # bound before the first committed move.
    produced = runner.play_games(
        num_games,
        seeds=[spec.seed for spec in specs],
        num_sims=num_sims,
        boards=boards,
        stop_fn=stop_fn,
        evaluator_for=evaluator_for,
    )
    results: list[GameResult] = []
    for i, trace in enumerate(produced):
        gs = runner._games_debug[i]
        tr = trackers[i]

        winner = winner_of(trace, gs.board, tr.would_resign_side)
        fill_outcomes(trace, winner)
        trace.resign_disabled = tr.disabled
        trace.would_resign_ply = tr.would_resign_ply
        results.append(GameResult(trace, winner, tr.would_resign_side))

    return results, {"sims": int(runner.stats["sims"]), "forwards": int(runner.stats["forwards"])}


def record_games(
    results: list[GameResult],
    specs: list[GameSpec],
    stats: SelfPlayStats,
    buffer: ReplayBuffer | None,
    iteration: int,
) -> None:
    """Fold scored games into ``stats`` and the ring, in spec order.

    Spec order is the order ``play_games`` has always stored in (the runner yields
    traces by game index), so a ring filled here from games played anywhere is
    byte-identical to one filled in-process.
    """
    for res, spec in zip(results, specs, strict=True):
        trace, winner = res.trace, res.winner
        stats.games += 1
        stats.positions += len(trace.positions)
        stats.results[trace.result] = stats.results.get(trace.result, 0) + 1
        stats.plies.append(trace.plies)
        if winner is not None:
            stats.decisive_plies.append(trace.plies)
        if trace.positions and spec.start_fen is None:
            # Spec §8's collapse detector is a reading at the *initial* position, so a
            # game started elsewhere by the opening lever does not contribute to it.
            first = trace.positions[0]
            stats.move1_entropies.append(gumbel.entropy([e[2] for e in first.improved_policy]))
            stats.move1_prior_entropies.append(first.root_prior_entropy)
        if trace.resign_disabled:
            stats.resign_disabled_games += 1
            if trace.would_resign_ply is not None:
                stats.would_have_resigned += 1
                # "Not ultimately lost by that side" — a draw counts as a false
                # resign, because resigning a drawn game throws away half a point.
                lost = winner is not None and winner != res.would_resign_side
                stats.false_resigns += not lost

        if buffer is not None:
            stats.pool_positions += _store(
                buffer, trace, iteration, trace.resign_disabled, spec.assignment
            )


def _store(
    buffer: ReplayBuffer,
    trace: GameTrace,
    iteration: int,
    resign_disabled: bool,
    assignment: GameAssignment,
) -> int:
    """Write a scored trace into the ring.

    Legal masks are recomputed from the decoded position rather than carried on
    the trace: the trace holds tokens, the encoding is Markovian, and storing a
    512-byte mask per position through the runner would double the trace's
    memory for something reconstructible in microseconds.

    Returns the number of positions whose mover was a league opponent.
    """
    from tinyalphazero.encoding import decode_board

    pool_positions = 0
    for pos in trace.positions:
        board = decode_board(pos.tokens)
        if pos.played_ft >= 0:
            # The move the game continued with (REVIEW-2026-10-21 C4) — the
            # promotion head's target is read at this cell.
            move_ft, promo = int(pos.played_ft), int(pos.played_promo)
        else:
            # A trace built without the runner: the improved-policy argmax.
            acted = (
                max(pos.improved_policy, key=lambda e: e[2]) if pos.improved_policy else (0, 0, 0.0)
            )
            move_ft, promo = int(acted[0]), int(acted[1])
        mover_is_latest = assignment.key_for(pos.side_to_move) == LATEST
        pool_positions += not mover_is_latest
        buffer.add(
            tokens=pos.tokens,
            legals=legal_moves_packed(board),
            move_ft=move_ft,
            promo=promo,
            outcome=int(pos.outcome),
            improved_policy=pos.improved_policy,
            iteration=iteration,
            resign_disabled=resign_disabled,
            mover_is_latest=mover_is_latest,
            root_q=float(pos.root_q),
        )
    return pool_positions


# The smoke-test CLI that lived here is scripts/selfplay_smoke.py
# (REVIEW-2026-10-21 E11: library code does not print).
