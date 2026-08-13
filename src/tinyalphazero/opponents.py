"""Frozen ladder opponents (spec §8, chunk 10).

The anchor ladder's whole value is that its opponents never change. A moving
opponent turns "Elo went up" into an uninterpretable statement, which is the
defect v1's arena had and the reason §8 specifies *fixed* anchors. So: these are
deterministic given a seed, they take no configuration from the campaign, and the
network anchor is pinned **by digest**, not by a path that later checkpoints might
be promoted into.

Two kinds of opponent, because they cost different things:

* ``kind == "move"`` — decides from the board alone (random, greedy, an external
  engine). The batched runner plays these through ``move_fn``, spending no
  simulation budget on them.
* ``kind == "network"`` — searches, so it is expressed as an ``evaluate_batch``
  and routed through ``evaluator_for``. Both sides then share one concurrent
  pool and one batched forward per network.

Stockfish is optional and **skips cleanly when the binary is absent**: it is an
M4 calibration anchor, not an M3 one, and installing it is a system change that
is the operator's call (AGENTS.md §5).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np

#: Standard relative values. The king is 0 — it is never captured, and giving it
#: a value would make every check look like a hanging piece.
PIECE_VALUE: dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

MATE_SCORE = 1e6


@dataclass
class RandomMover:
    """Uniform over legal moves. Spec §8's floor sanity check.

    A checkpoint that cannot beat this is broken in a way no other anchor would
    isolate, which is why it is on the ladder despite being trivially weak.
    """

    name: str = "random"
    kind: str = "move"

    def choose(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        moves = list(board.legal_moves)
        return moves[int(rng.integers(len(moves)))]


@dataclass
class GreedyMaterial:
    """1-ply material grab with a seeded random tie-break (spec §8: basic tactics).

    Takes mate when it is available, otherwise maximises immediate material won.
    It does **not** look at what it hangs in reply — that is the point: it is a
    one-ply opponent, and a checkpoint that beats it has learned something about
    consequences rather than about capture ordering.

    **The tie-break is drawn per game, so this opponent is deliberately not a pure
    function of position.** Measured: a shipped-greedy self-match returns opening
    pair scores of 0.75 and 0.25, not the uniform 0.5 that a position-pure player
    produces (see ``test_a_deterministic_self_match_is_an_exact_null``). That is
    correct for its role and worth stating so nobody "fixes" it: a fixed tie-break
    would make greedy play python-chess's generation order among equal captures
    forever, which is an arbitrary systematic preference rather than a neutral one.
    The generator comes from the caller so the choice is still reproducible from
    the run seed alone.

    The consequence to keep straight: the exact-null identity belongs to the
    *learner under the evaluation profile*, whose move choice really is a pure
    function of position. It does not apply to this class, and a test that expected
    it to would be testing the wrong object.
    """

    name: str = "greedy"
    kind: str = "move"

    def score(self, board: chess.Board, mv: chess.Move) -> float:
        board.push(mv)
        try:
            if board.is_checkmate():
                return MATE_SCORE
        finally:
            board.pop()

        gain = 0.0
        if board.is_en_passant(mv):
            gain += PIECE_VALUE[chess.PAWN]
        else:
            captured = board.piece_at(mv.to_square)
            if captured is not None:
                gain += PIECE_VALUE[captured.piece_type]
        if mv.promotion:
            gain += PIECE_VALUE[mv.promotion] - PIECE_VALUE[chess.PAWN]
        return gain

    def choose(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        moves = list(board.legal_moves)
        scores = np.array([self.score(board, mv) for mv in moves])
        best = np.flatnonzero(scores == scores.max())
        return moves[int(best[int(rng.integers(len(best)))])]


@dataclass
class NetSearchOpponent:
    """A frozen checkpoint playing under the evaluation profile (spec §8).

    The M3 criterion is Elo against *this*: the Phase 1 anchor at a comparable
    node count. It is pinned by digest in TRAINING_STRATEGY.md so that promoting
    `phase1_anchor.pt` later cannot silently move the ladder's zero point.
    """

    ckpt: Path
    sims: int
    device: str = "cpu"
    name: str = "phase1"
    kind: str = "network"
    digest: str = ""
    _evaluate: Any = field(default=None, repr=False)

    def evaluate_batch(self) -> Any:
        """Lazily build the batched evaluator; cached for the life of the opponent."""
        if self._evaluate is None:
            import torch

            from tinyalphazero.mcts.batch_runner import torch_evaluate_batch
            from tinyalphazero.train import load_checkpoint

            device = torch.device(self.device)
            net, _meta = load_checkpoint(self.ckpt, device=device)
            net.eval()
            self._evaluate = torch_evaluate_batch(net, device)
        return self._evaluate


# ---------------------------------------------------------------------------
# Stockfish — optional, skipped cleanly when absent
# ---------------------------------------------------------------------------


def stockfish_path(configured: str = "") -> str | None:
    """Resolve a usable Stockfish binary, or None.

    None is a first-class answer, not an error: §8's Stockfish rungs calibrate an
    external scale for M4, and M3's go/no-go does not read them. Every caller
    treats None as "that rung is not on the ladder today".
    """
    if configured:
        return configured if Path(configured).exists() else None
    return shutil.which("stockfish")


@dataclass
class StockfishOpponent:
    """Fixed-skill, fixed-node Stockfish via ``chess.engine`` (spec §8).

    Fixed *nodes* rather than time: a time limit makes the anchor a function of
    what else the box was doing, and an anchor that moves with machine load is
    not an anchor.
    """

    path: str
    skill: int
    nodes: int
    name: str = ""
    kind: str = "move"
    _engine: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"stockfish_s{self.skill}"

    def _open(self) -> Any:
        if self._engine is None:
            import chess.engine

            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
            self._engine.configure({"Skill Level": self.skill})
        return self._engine

    def choose(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        import chess.engine

        result = self._open().play(board, chess.engine.Limit(nodes=self.nodes))
        if result.move is None:  # engine resigned or returned nothing playable
            moves = list(board.legal_moves)
            return moves[int(rng.integers(len(moves)))]
        return result.move

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    # Context-manager protocol so releasing the subprocess is an idiom rather
    # than something a caller has to remember. close() existed from chunk 10 and
    # was called by nobody until 2026-08-12, when the first Stockfish rung played
    # its games, wrote a correct row, and then hung at interpreter exit for seven
    # hours on a non-daemon transport thread.
    def __enter__(self) -> StockfishOpponent:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def anchor_identity(opponent: Any) -> str:
    """The single spelling of an anchor's identity, for rows and for skip-sets.

    Ladder rows record an opponent's *name*; resumability used to test the CLI
    *spec*. For random, greedy and phase1 those strings are identical, so the
    mismatch was invisible everywhere it had ever run — and "stockfish:0" never
    matched the logged "stockfish_s0", so the one anchor family whose spec carries
    a parameter silently recomputed its matchup and appended a duplicate row.

    Two spellings of one intent is the defect. This is the one spelling: every
    writer and every reader derives identity from here.
    """
    return str(opponent.name)


def build_opponent(
    spec: str,
    anchor_ckpt: Path | None = None,
    sims: int = 48,
    device: str = "cpu",
    stockfish: str = "",
) -> RandomMover | GreedyMaterial | NetSearchOpponent | StockfishOpponent | None:
    """Resolve an opponent name from the CLI. Returns None if it is unavailable.

    ``None`` means "skip this rung", which only Stockfish can produce.
    """
    if spec == "random":
        return RandomMover()
    if spec == "greedy":
        return GreedyMaterial()
    if spec == "phase1":
        if anchor_ckpt is None:
            raise ValueError("the phase1 rung needs --anchor-ckpt")
        return NetSearchOpponent(ckpt=Path(anchor_ckpt), sims=sims, device=device)
    if spec.startswith("stockfish"):
        path = stockfish_path(stockfish)
        if path is None:
            return None
        skill = int(spec.split(":", 1)[1]) if ":" in spec else 0
        return StockfishOpponent(path=path, skill=skill, nodes=10_000)
    raise ValueError(f"unknown opponent {spec!r}")
