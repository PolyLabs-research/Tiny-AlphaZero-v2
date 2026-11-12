"""The network under the evaluation profile, as a UCI engine (the public benchmark).

A UCI client -- cutechess-cli, fastchess, lichess-bot, a GUI -- sends positions
and ``go``; the engine answers ``bestmove``. What it searches with is exactly what
the ladder measures, which is the whole point of the module:

* **the evaluation profile** -- ``root_noise=False`` on the run's own ``mcts``
  section, resolved by ``Search`` as for a suite or a match: no Gumbel draw, so a
  position and its history determine the move;
* **the ladder's simulation budget** -- ``eval.eval_match_sims`` (48) unless the
  client asks for ``nodes`` or sets the ``Sims`` option;
* **the ladder's leaf evaluation** -- the batched evaluator
  (``torch_evaluate_batch``: bf16 autocast on cuda, fp32 on cpu) called with a
  batch of one, masked by ``masked_policy_probs``, the function
  ``BatchRunner._apply`` masks with;
* **the ladder's repetition state** -- the positions the game has *left*, counted
  from the ``position`` command's move list exactly as ``BatchRunner`` counts
  them from the moves it commits (``_GameState.history``).

``Search`` (synchronous) and ``BatchRunner`` (batched) drive one simulation
coroutine and are held equal by the equivalence gate in
``tests/test_batch_runner.py``; ``tests/test_uci.py`` replays ``BatchRunner``
games through this module and requires the same moves.

Clock fields (``wtime``, ``movetime``, ...) are ignored: the budget is a number of
simulations, not a time, as on the ladder. Library code: nothing here prints;
the session writes through the callable it is given.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import chess
import numpy as np

from tinyalphazero.config import MCTSConfig
from tinyalphazero.encoding import decode_board, legal_mask_ft
from tinyalphazero.mcts.batch_runner import masked_policy_probs
from tinyalphazero.mcts.search import EvaluateFn, Search

ENGINE_NAME = "TinyAlphaZero-v2"
ENGINE_AUTHOR = "PolyLabs-research"
MAX_SIMS = 100_000
# q is P(win) - P(loss) in [-1, 1]; clip before the log so a proven result
# (q = +/-1) reports a large finite score instead of infinity.
_Q_CLIP = 0.999


def q_to_cp(q: float) -> int:
    """Root value -> centipawns, by the logistic Elo relation on expected score.

    Expected score s = (1 + q) / 2, and 400 * log10(s / (1 - s)) is the rating
    gap a score of s implies. It is a display convention for clients, not a
    calibrated pawn scale: the network was never fitted to centipawns.
    """
    q = max(-_Q_CLIP, min(_Q_CLIP, float(q)))
    return round(400.0 * math.log10((1.0 + q) / (1.0 - q)))


def single_from_batch(evaluate_batch: Any) -> EvaluateFn:
    """Adapt a batched evaluator to the single-position ``EvaluateFn`` ``Search`` calls.

    Tokens go through as a batch of one; the leaf is masked by
    ``masked_policy_probs`` over ``legal_mask_ft`` of the decoded board -- the
    encoding is Markovian (spec §3.1), so the decoded board's legal moves are the
    leaf's.
    """

    def evaluate(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        batch = np.ascontiguousarray(np.asarray(tokens, dtype=np.int64)[None, :])
        policies, promos, values = evaluate_batch(batch)
        probs = masked_policy_probs(policies[0], legal_mask_ft(decode_board(tokens)))
        promo = np.asarray(promos[0], dtype=np.float32).reshape(64, 64, 4)
        return probs, promo, float(values[0])

    return evaluate


@dataclass
class SearchReport:
    """One answered ``go``. ``move`` is None when the position has no legal move."""

    move: chess.Move | None
    sims_run: int = 0
    q: float = 0.0

    def info_line(self) -> str | None:
        if self.move is None:
            return None
        return f"info depth 1 nodes {self.sims_run} score cp {q_to_cp(self.q)} pv {self.move.uci()}"

    def bestmove_line(self) -> str:
        return f"bestmove {self.move.uci() if self.move is not None else '0000'}"


@dataclass
class UciEngine:
    """The position, its history, and the search that answers ``go``."""

    evaluate_fn: EvaluateFn
    mcts: MCTSConfig
    sims: int
    board: chess.Board = field(default_factory=chess.Board)
    # Positions the game has LEFT, by transposition key -- BatchRunner's
    # `_GameState.history`. The present position is not in it.
    history: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        # The evaluation profile, applied once, here -- as `play_match` applies it.
        self.mcts = replace(self.mcts, root_noise=False)
        if not 1 <= self.sims <= MAX_SIMS:
            raise ValueError(f"sims must be in [1, {MAX_SIMS}], got {self.sims}")

    def set_position(self, fen: str | None, moves: list[str]) -> None:
        """``position``: a start (None = the initial position) and the moves since.

        Raises ValueError on a bad FEN or a move that is not legal where it is
        played; the engine's position is then left as it was.
        """
        board = chess.Board(fen) if fen is not None else chess.Board()
        history: Counter = Counter()
        for text in moves:
            try:
                move = chess.Move.from_uci(text)
            except ValueError as exc:
                raise ValueError(f"unparseable move {text!r}") from exc
            if move not in board.legal_moves:
                raise ValueError(f"illegal move {text!r} in {board.fen()}")
            history[board._transposition_key()] += 1
            board.push(move)
        self.board, self.history = board, history

    def search(self, sims: int | None = None) -> SearchReport:
        """Search the current position and report the acting move."""
        if not any(self.board.legal_moves):
            return SearchReport(move=None)
        budget = self.sims if sims is None else sims
        history = self.history
        result = Search(self.evaluate_fn, self.mcts).run(
            self.board, lambda b: history[b._transposition_key()], budget
        )
        move = result.best_move()
        chosen = result.chosen if result.chosen >= 0 else result.moves().index(move)
        return SearchReport(move=move, sims_run=result.sims_run, q=float(result.q[chosen]))


class UciSession:
    """The UCI command loop over one engine, built lazily on first need.

    ``make_engine(sims)`` builds the engine (loading a checkpoint takes seconds,
    so ``uci`` is answered first and the load happens at ``isready`` or the first
    command that needs a position). ``write`` receives one protocol line at a
    time, without its newline.
    """

    def __init__(
        self,
        make_engine: Callable[[int], UciEngine],
        write: Callable[[str], None],
        default_sims: int,
        name: str = ENGINE_NAME,
    ) -> None:
        self._make_engine = make_engine
        self._write = write
        self._engine: UciEngine | None = None
        self.sims = default_sims
        self.name = name
        # A `go infinite` answer is held until `stop` (UCI: no bestmove before it).
        self._held: SearchReport | None = None

    @property
    def engine(self) -> UciEngine:
        if self._engine is None:
            self._engine = self._make_engine(self.sims)
        return self._engine

    def handle(self, line: str) -> bool:
        """Process one input line. Returns False when the client sent ``quit``."""
        tokens = line.split()
        if not tokens:
            return True
        cmd, args = tokens[0], tokens[1:]
        if cmd == "quit":
            return False
        if cmd == "uci":
            self._write(f"id name {self.name}")
            self._write(f"id author {ENGINE_AUTHOR}")
            self._write(f"option name Sims type spin default {self.sims} min 1 max {MAX_SIMS}")
            self._write("uciok")
        elif cmd == "isready":
            _ = self.engine
            self._write("readyok")
        elif cmd == "setoption":
            self._setoption(args)
        elif cmd == "ucinewgame":
            self.engine.set_position(None, [])
        elif cmd == "position":
            self._position(args)
        elif cmd == "go":
            self._go(args)
        elif cmd in ("stop", "ponderhit"):
            if self._held is not None:
                self._emit(self._held)
                self._held = None
        # Anything else (debug, register, unknown) is ignored, as UCI requires.
        return True

    def _emit(self, report: SearchReport) -> None:
        info = report.info_line()
        if info is not None:
            self._write(info)
        self._write(report.bestmove_line())

    def _setoption(self, args: list[str]) -> None:
        # setoption name <id> [value <x>]; option names may contain spaces.
        text = " ".join(args)
        if not text.startswith("name "):
            return
        name, _, value = text[len("name ") :].partition(" value ")
        if name.strip().lower() != "sims":
            return
        try:
            sims = int(value.strip())
        except ValueError:
            self._write(f"info string Sims must be an integer, got {value.strip()!r}")
            return
        if not 1 <= sims <= MAX_SIMS:
            self._write(f"info string Sims must be in [1, {MAX_SIMS}], got {sims}")
            return
        self.sims = sims
        if self._engine is not None:
            self._engine.sims = sims

    def _position(self, args: list[str]) -> None:
        if not args:
            return
        if "moves" in args:
            split = args.index("moves")
            head, moves = args[:split], args[split + 1 :]
        else:
            head, moves = args, []
        if head[0] == "startpos":
            fen = None
        elif head[0] == "fen" and len(head) > 1:
            fen = " ".join(head[1:])
        else:
            self._write(f"info string unrecognised position command: {' '.join(args)}")
            return
        try:
            self.engine.set_position(fen, moves)
        except ValueError as exc:
            self._write(f"info string position rejected: {exc}")

    def _go(self, args: list[str]) -> None:
        sims = None
        if "nodes" in args:
            i = args.index("nodes")
            try:
                sims = max(1, min(MAX_SIMS, int(args[i + 1])))
            except (IndexError, ValueError):
                sims = None
        report = self.engine.search(sims)
        if "infinite" in args or "ponder" in args:
            self._held = report
        else:
            self._emit(report)
