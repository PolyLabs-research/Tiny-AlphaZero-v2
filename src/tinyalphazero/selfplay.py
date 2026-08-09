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

    python -m tinyalphazero.selfplay --ckpt runs/phase1_anchor.pt \\
        --games 8 --sims 16 --out runs/sp_smoke --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np

from tinyalphazero.config import Config, MCTSConfig, SelfPlayConfig, load_config
from tinyalphazero.encoding import legal_moves_packed
from tinyalphazero.mcts.batch_runner import BatchRunner, GameTrace, uniform_stub_batch
from tinyalphazero.replay import ReplayBuffer

RESIGN_RESULT = "resign"


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

    @property
    def false_resign_rate(self) -> float | None:
        """Fraction of would-be resignations that were not in fact losses.

        None when no disabled game ever hit the threshold — an honest "not
        measured yet" rather than a 0.0 that reads as "calibrated".
        """
        if self.would_have_resigned == 0:
            return None
        return self.false_resigns / self.would_have_resigned

    def to_dict(self) -> dict:
        return {
            "games": self.games,
            "positions": self.positions,
            "results": self.results,
            "resign_disabled_games": self.resign_disabled_games,
            "would_have_resigned": self.would_have_resigned,
            "false_resigns": self.false_resigns,
            "false_resign_rate": self.false_resign_rate,
            "seconds": round(self.seconds, 2),
            "games_per_hour": (
                round(self.games / self.seconds * 3600, 1) if self.seconds > 0 else None
            ),
        }


def play_games(
    evaluate_batch,
    num_games: int,
    cfg: Config | None = None,
    seed: int = 0,
    num_sims: int | None = None,
    device: str = "cpu",
    iteration: int = 0,
    buffer: ReplayBuffer | None = None,
) -> tuple[list[GameTrace], SelfPlayStats]:
    """Play ``num_games`` self-play games, score them, and optionally store them."""
    cfg = cfg or Config()
    mcts: MCTSConfig = cfg.mcts
    sp: SelfPlayConfig = cfg.selfplay
    num_sims = num_sims if num_sims is not None else sp.sims_per_move

    rng = np.random.default_rng(seed)
    disabled_flags = rng.random(num_games) < sp.resign_disabled_frac
    trackers = [
        ResignTracker(
            threshold=sp.resign_threshold,
            consecutive=sp.resign_consecutive,
            disabled=bool(disabled_flags[i]),
        )
        for i in range(num_games)
    ]

    stats = SelfPlayStats()
    t0 = time.perf_counter()
    traces: list[GameTrace] = []

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
        return tr(trace) if tr is not None else None

    runner = BatchRunner(evaluate_batch, mcts, device=device, max_plies=sp.max_plies)
    boards = [chess.Board() for _ in range(num_games)]
    seeds = [game_seed(seed, i) for i in range(num_games)]

    # The runner creates its _GameState list up front, so the trackers can be
    # bound before the first committed move.
    produced = runner.play_games(
        num_games, seeds=seeds, num_sims=num_sims, boards=boards, stop_fn=stop_fn
    )
    for i, trace in enumerate(produced):
        gs = runner._games_debug[i]
        tr = trackers[i]

        winner = winner_of(trace, gs.board, tr.would_resign_side)
        fill_outcomes(trace, winner)
        trace.resign_disabled = tr.disabled
        trace.would_resign_ply = tr.would_resign_ply

        stats.games += 1
        stats.positions += len(trace.positions)
        stats.results[trace.result] = stats.results.get(trace.result, 0) + 1
        if tr.disabled:
            stats.resign_disabled_games += 1
            if tr.would_resign_ply is not None:
                stats.would_have_resigned += 1
                # "Not ultimately lost by that side" — a draw counts as a false
                # resign, because resigning a drawn game throws away half a point.
                lost = winner is not None and winner != tr.would_resign_side
                stats.false_resigns += not lost

        if buffer is not None:
            _store(buffer, trace, gs, iteration, tr.disabled)
        traces.append(trace)

    stats.seconds = time.perf_counter() - t0
    if buffer is not None:
        buffer.flush()
    return traces, stats


def _store(
    buffer: ReplayBuffer, trace: GameTrace, gs, iteration: int, resign_disabled: bool
) -> None:
    """Write a scored trace into the ring.

    Legal masks are recomputed from the decoded position rather than carried on
    the trace: the trace holds tokens, the encoding is Markovian, and storing a
    512-byte mask per position through the runner would double the trace's
    memory for something reconstructible in microseconds.
    """
    from tinyalphazero.encoding import decode_board

    for pos in trace.positions:
        board = decode_board(pos.tokens)
        acted = max(pos.improved_policy, key=lambda e: e[2]) if pos.improved_policy else (0, 0, 0.0)
        buffer.add(
            tokens=pos.tokens,
            legals=legal_moves_packed(board),
            move_ft=int(acted[0]),
            promo=int(acted[1]),
            outcome=int(pos.outcome),
            improved_policy=pos.improved_policy,
            iteration=iteration,
            resign_disabled=resign_disabled,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=None, help="Checkpoint to play with")
    ap.add_argument("--stub", action="store_true", help="Zero-cost evaluator instead of a net")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--sims", type=int, default=None, help="Simulations per move")
    ap.add_argument("--out", type=Path, required=True, help="Run directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument(
        "--capacity", type=int, default=None, help="Replay capacity (default: config)"
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.ckpt is None and not args.stub:
        print("error: pass --ckpt or --stub")
        return 2

    if args.stub:
        evaluate_batch = uniform_stub_batch
        which = "stub"
    else:
        import torch

        from tinyalphazero.mcts.batch_runner import torch_evaluate_batch
        from tinyalphazero.train import load_checkpoint

        device = torch.device(args.device)
        net, _meta = load_checkpoint(args.ckpt, device=device)
        net.eval()
        evaluate_batch = torch_evaluate_batch(net, device)
        which = str(args.ckpt)

    args.out.mkdir(parents=True, exist_ok=True)
    capacity = args.capacity or cfg.train.replay_capacity
    buffer = ReplayBuffer.open_or_create(args.out / "replay", capacity)

    _traces, stats = play_games(
        evaluate_batch,
        args.games,
        cfg=cfg,
        seed=args.seed,
        num_sims=args.sims,
        device=args.device,
        iteration=args.iteration,
        buffer=buffer,
    )

    row = {"eval": which, "sims": args.sims or cfg.selfplay.sims_per_move, **stats.to_dict()}
    with (args.out / "selfplay.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")

    print(
        f"{stats.games} games, {stats.positions} positions in {stats.seconds:.1f}s "
        f"({row['games_per_hour']} games/h) | eval={which}"
    )
    print(f"  results: {stats.results}")
    print(
        f"  resign-disabled {stats.resign_disabled_games}, would-have-resigned "
        f"{stats.would_have_resigned}, false-resign rate {stats.false_resign_rate}"
    )
    print(f"  replay: {len(buffer)} positions at {args.out / 'replay'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
