"""Phase 1 data generation: random-play games and mate-in-1 anchors (spec §5).

Records are written straight to memmap with a small JSON sidecar (meta.json).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from tinyalphazero.encoding import (
    _repetitions_from_board,
    encode_board,
    encode_move,
    legal_moves_packed,
)

# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------

_VERSION = "1"

_DTYPE_MAP = {
    "boards": "int8",
    "moves": "int16",
    "promos": "int8",
    "legals": "uint8",
    "outcomes": "int8",
}


@dataclass
class PositionRecord:
    tokens: np.ndarray   # (72,) int8
    from_sq: int
    to_sq: int
    promo: int
    legals: np.ndarray   # (512,) uint8
    outcome: int         # +1, 0, -1 from side-to-move perspective


@dataclass
class GameRecord:
    positions: list[PositionRecord] = field(default_factory=list)


@dataclass
class DatasetMeta:
    n: int
    version: str
    git_sha: str
    dtype_map: dict
    generation_config: dict

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "version": self.version,
            "git_sha": self.git_sha,
            "dtype_map": self.dtype_map,
            "generation_config": self.generation_config,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DatasetMeta:
        return cls(
            n=d["n"],
            version=d["version"],
            git_sha=d.get("git_sha", "unknown"),
            dtype_map=d["dtype_map"],
            generation_config=d.get("generation_config", {}),
        )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_random_games(
    num_games: int,
    rng: np.random.Generator,
    max_plies: int = 200,
    min_plies: int = 10,
) -> Iterator[GameRecord]:
    """Yield GameRecord for each random-legal game. Skips games shorter than min_plies."""
    for _ in range(num_games):
        board = chess.Board()
        raw: list[tuple[np.ndarray, int, int, int, np.ndarray, bool]] = []
        # (tokens, from_sq, to_sq, promo, legals_packed, stm_is_white)

        plies = 0
        while plies < max_plies:
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break

            rep = _repetitions_from_board(board)
            tokens = encode_board(board, rep)
            legals_packed = legal_moves_packed(board)
            stm_is_white = board.turn == chess.WHITE

            move = legal[int(rng.integers(len(legal)))]
            from_sq, to_sq, promo = encode_move(move)

            raw.append((tokens, from_sq, to_sq, promo, legals_packed, stm_is_white))
            board.push(move)
            plies += 1

        if plies < min_plies:
            continue

        outcome_from_white = _board_outcome_white(board)

        game = GameRecord()
        for tokens, from_sq, to_sq, promo, legals_packed, stm_is_white in raw:
            if outcome_from_white == 0:
                stm_outcome = 0
            elif outcome_from_white == 1:
                stm_outcome = 1 if stm_is_white else -1
            else:
                stm_outcome = -1 if stm_is_white else 1
            game.positions.append(
                PositionRecord(tokens, from_sq, to_sq, promo, legals_packed, stm_outcome)
            )

        yield game


def generate_mate_in_1(
    num_positions: int,
    rng: np.random.Generator,
) -> Iterator[PositionRecord]:
    """Yield PositionRecord for positions where a mate-in-1 exists.

    Harvests as a by-product of random play — does not rejection-sample whole games.
    """
    found = 0
    board = chess.Board()
    plies = 0
    max_plies = 150

    while found < num_positions:
        legal = list(board.legal_moves)
        if not legal or plies >= max_plies or board.is_game_over():
            board.reset()
            plies = 0
            continue

        # Scan for a mating move without modifying the board permanently
        mating_move: chess.Move | None = None
        for move in legal:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                mating_move = move
                break
            board.pop()

        if mating_move is not None:
            rep = _repetitions_from_board(board)
            tokens = encode_board(board, rep)
            legals_packed = legal_moves_packed(board)
            from_sq, to_sq, promo = encode_move(mating_move)
            yield PositionRecord(
                tokens=tokens,
                from_sq=from_sq,
                to_sq=to_sq,
                promo=promo,
                legals=legals_packed,
                outcome=1,  # side to move mates → +1
            )
            found += 1

        # Advance game with a random move
        move = legal[int(rng.integers(len(legal)))]
        board.push(move)
        plies += 1


# ---------------------------------------------------------------------------
# Write / read
# ---------------------------------------------------------------------------


def write_dataset(
    records: Iterator[GameRecord | PositionRecord],
    out_dir: Path,
    generation_config: dict | None = None,
) -> DatasetMeta:
    """Collect all position records and write them to the memmap layout.

    Accepts an iterator of GameRecord (flattened) or PositionRecord.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boards_list: list[np.ndarray] = []
    moves_list: list[tuple[int, int]] = []
    promos_list: list[int] = []
    legals_list: list[np.ndarray] = []
    outcomes_list: list[int] = []

    for record in records:
        positions = record.positions if isinstance(record, GameRecord) else [record]
        for pos in positions:
            boards_list.append(pos.tokens)
            moves_list.append((pos.from_sq, pos.to_sq))
            promos_list.append(pos.promo)
            legals_list.append(pos.legals)
            outcomes_list.append(pos.outcome)

    n = len(boards_list)
    if n == 0:
        raise ValueError("write_dataset: no records to write")

    boards_arr = np.stack(boards_list).astype(np.int8)         # (N, 72)
    moves_arr = np.array(moves_list, dtype=np.int16)           # (N, 2)
    promos_arr = np.array(promos_list, dtype=np.int8)          # (N,)
    legals_arr = np.stack(legals_list).astype(np.uint8)        # (N, 512)
    outcomes_arr = np.array(outcomes_list, dtype=np.int8)      # (N,)

    _write_memmap(out_dir / "boards.dat", boards_arr)
    _write_memmap(out_dir / "moves.dat", moves_arr)
    _write_memmap(out_dir / "promos.dat", promos_arr)
    _write_memmap(out_dir / "legals.dat", legals_arr)
    _write_memmap(out_dir / "outcomes.dat", outcomes_arr)

    # Record the exact invocation: runs/data/mate1 was once generated with --games
    # instead of --mate1, and nothing downstream could tell.
    gen_cfg = dict(generation_config or {})
    gen_cfg.setdefault("argv", list(sys.argv))

    meta = DatasetMeta(
        n=n,
        version=_VERSION,
        git_sha=_git_sha(),
        dtype_map=_DTYPE_MAP,
        generation_config=gen_cfg,
    )
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta.to_dict(), fh, indent=2)

    return meta


def assert_dataset_mode(data_dir: Path, expected_mode: str) -> dict:
    """Fail fast if a dataset dir was not generated in the expected mode.

    Guards against the runs/phase1_a failure, where runs/data/mate1 held random-play
    games because the generator was invoked with --games instead of --mate1. Returns
    the meta dict on success.
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No meta.json in {data_dir} — cannot verify dataset mode")

    with open(meta_path) as fh:
        meta = json.load(fh)

    actual = meta.get("generation_config", {}).get("mode")
    if actual != expected_mode:
        raise ValueError(
            f"Dataset mode mismatch in {meta_path}: expected mode={expected_mode!r}, "
            f"found mode={actual!r} (n={meta.get('n')}). "
            f"Regenerate with: scripts/generate_phase1.py "
            f"{'--mate1 N' if expected_mode == 'mate_in_1' else '--games N'} --out {data_dir}"
        )
    return meta


def _write_memmap(path: Path, arr: np.ndarray) -> None:
    mm = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Phase1Dataset(Dataset):
    """Read-only memmap-backed dataset. __getitem__ returns a dict of tensors."""

    def __init__(self, data_dir: Path) -> None:
        data_dir = Path(data_dir)
        with open(data_dir / "meta.json") as fh:
            self.meta = DatasetMeta.from_dict(json.load(fh))
        n = self.meta.n

        self._boards = np.memmap(data_dir / "boards.dat", dtype=np.int8, mode="r", shape=(n, 72))
        self._moves = np.memmap(data_dir / "moves.dat", dtype=np.int16, mode="r", shape=(n, 2))
        self._promos = np.memmap(data_dir / "promos.dat", dtype=np.int8, mode="r", shape=(n,))
        self._legals = np.memmap(data_dir / "legals.dat", dtype=np.uint8, mode="r", shape=(n, 512))
        self._outcomes = np.memmap(
            data_dir / "outcomes.dat", dtype=np.int8, mode="r", shape=(n,)
        )

    def __len__(self) -> int:
        return self.meta.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.from_numpy(self._boards[idx].astype(np.int64)),
            "from_sq": torch.tensor(int(self._moves[idx, 0]), dtype=torch.long),
            "to_sq": torch.tensor(int(self._moves[idx, 1]), dtype=torch.long),
            "promo": torch.tensor(int(self._promos[idx]), dtype=torch.long),
            "legals": torch.from_numpy(self._legals[idx].copy()),
            "outcome": torch.tensor(int(self._outcomes[idx]), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _board_outcome_white(board: chess.Board) -> int:
    """Return game outcome from white's perspective: +1, 0, -1."""
    result = board.result()
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    return 0  # draw or unfinished (hit ply cap → draw adjudication)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"
