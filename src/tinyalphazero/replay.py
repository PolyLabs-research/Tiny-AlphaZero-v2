"""Memmap-backed replay ring (spec §7, chunk 8).

Capacity is fixed at construction and the buffer wraps; the file set *is* the
state, so reopening a run directory resumes exactly where it stopped. That is
the resumability requirement in AGENTS.md §6 — a five-day run must survive being
killed at an arbitrary moment — and it is why nothing here is held only in RAM.

Layout, one directory per buffer (the chunk-2 fields plus what search targets
need):

    boards.dat        int8    (C, 72)     position tokens
    moves.dat         int16   (C, 2)      the move actually played (from, to)
    promos.dat        int8    (C,)        its promotion class, 0..4
    legals.dat        uint8   (C, 512)    packed legal-move bitset
    outcomes.dat      int8    (C,)        +1/0/-1, side-to-move perspective
    policy_ft.dat     int16   (C, 96)     improved-policy move_ft, -1 padded
    policy_pr.dat     float16 (C, 96)     improved-policy probabilities
    policy_n.dat      int16   (C,)        entries actually used
    promo_cls.dat     int8    (C, 96)     promo class per sparse entry
    iteration.dat     int32   (C,)        producing iteration
    resign_off.dat    int8    (C,)        1 if resignation was disabled for the game
    meta.json                             capacity, total_written, dtypes, version

The improved policy is stored **sparse** because it is supported only on legal
moves — a dense (4096,) float32 row would be 16 KB of which ~99% is zero, and
500K of those is 8 GB. At 96 entries it is 384 B. ``MAX_POLICY_ENTRIES`` is a
hard cap, not a guess: a position with more than 96 legal moves keeps its
96 highest-probability entries and renormalises, which is recorded rather than
silent (see ``_pack_policy``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

VERSION = 1
MAX_POLICY_ENTRIES = 96
SEQ_LEN = 72
LEGAL_BYTES = 512
POLICY_DIM = 4096

_FIELDS: dict[str, tuple[str, tuple[int, ...]]] = {
    "boards": ("int8", (SEQ_LEN,)),
    "moves": ("int16", (2,)),
    "promos": ("int8", ()),
    "legals": ("uint8", (LEGAL_BYTES,)),
    "outcomes": ("int8", ()),
    "policy_ft": ("int16", (MAX_POLICY_ENTRIES,)),
    "policy_pr": ("float16", (MAX_POLICY_ENTRIES,)),
    "policy_n": ("int16", ()),
    "promo_cls": ("int8", (MAX_POLICY_ENTRIES,)),
    "iteration": ("int32", ()),
    "resign_off": ("int8", ()),
}


def _pack_policy(
    sparse: list[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sparse improved policy → fixed-width arrays.

    Over-length positions keep their highest-probability entries and renormalise
    rather than truncating in move order, so the target stays a distribution and
    the mass that survives is the mass that mattered. Chess tops out around 218
    legal moves in constructed positions; real ones essentially never exceed 96,
    but "essentially never" is not a guarantee a memmap width can rely on.
    """
    ft = np.full(MAX_POLICY_ENTRIES, -1, dtype=np.int16)
    pr = np.zeros(MAX_POLICY_ENTRIES, dtype=np.float16)
    cls = np.zeros(MAX_POLICY_ENTRIES, dtype=np.int8)
    if not sparse:
        return ft, pr, cls, 0

    entries = sorted(sparse, key=lambda e: -e[2])[:MAX_POLICY_ENTRIES]
    total = sum(e[2] for e in entries)
    if total <= 0:
        total = 1.0
    for i, (move_ft, promo, prob) in enumerate(entries):
        ft[i] = move_ft
        pr[i] = prob / total
        cls[i] = promo
    return ft, pr, cls, len(entries)


class ReplayBuffer:
    """Fixed-capacity ring over memmapped arrays."""

    def __init__(self, path: Path, capacity: int, total_written: int) -> None:
        self.path = Path(path)
        self.capacity = capacity
        self.total_written = total_written
        self._arr: dict[str, np.memmap] = {}
        for name, (dtype, shape) in _FIELDS.items():
            self._arr[name] = np.memmap(
                self.path / f"{name}.dat",
                dtype=dtype,
                mode="r+",
                shape=(capacity, *shape) if shape else (capacity,),
            )

    # -- lifecycle -------------------------------------------------------

    @classmethod
    def create(cls, path: str | Path, capacity: int = 500_000) -> ReplayBuffer:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, (dtype, shape) in _FIELDS.items():
            full = (capacity, *shape) if shape else (capacity,)
            np.memmap(path / f"{name}.dat", dtype=dtype, mode="w+", shape=full).flush()
        buf = cls(path, capacity, 0)
        buf._write_meta()
        return buf

    @classmethod
    def open(cls, path: str | Path) -> ReplayBuffer:
        """Reopen an existing buffer, resuming its write position."""
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        if meta.get("version") != VERSION:
            raise ValueError(f"replay version {meta.get('version')} != {VERSION}")
        return cls(path, int(meta["capacity"]), int(meta["total_written"]))

    @classmethod
    def open_or_create(cls, path: str | Path, capacity: int = 500_000) -> ReplayBuffer:
        return cls.open(path) if (Path(path) / "meta.json").exists() else cls.create(path, capacity)

    def _write_meta(self) -> None:
        (self.path / "meta.json").write_text(
            json.dumps(
                {
                    "version": VERSION,
                    "capacity": self.capacity,
                    "total_written": self.total_written,
                    "max_policy_entries": MAX_POLICY_ENTRIES,
                    "fields": {k: v[0] for k, v in _FIELDS.items()},
                },
                indent=2,
            )
            + "\n"
        )

    def flush(self) -> None:
        for arr in self._arr.values():
            arr.flush()
        self._write_meta()

    def __len__(self) -> int:
        return min(self.total_written, self.capacity)

    # -- writing ---------------------------------------------------------

    def add(
        self,
        tokens: np.ndarray,
        legals: np.ndarray,
        move_ft: int,
        promo: int,
        outcome: int,
        improved_policy: list[tuple[int, int, float]],
        iteration: int = 0,
        resign_disabled: bool = False,
    ) -> None:
        i = self.total_written % self.capacity
        ft, pr, cls, n = _pack_policy(improved_policy)
        self._arr["boards"][i] = tokens
        self._arr["legals"][i] = legals
        self._arr["moves"][i] = (move_ft // 64, move_ft % 64)
        self._arr["promos"][i] = promo
        self._arr["outcomes"][i] = outcome
        self._arr["policy_ft"][i] = ft
        self._arr["policy_pr"][i] = pr
        self._arr["promo_cls"][i] = cls
        self._arr["policy_n"][i] = n
        self._arr["iteration"][i] = iteration
        self._arr["resign_off"][i] = int(resign_disabled)
        self.total_written += 1

    # -- reading ---------------------------------------------------------

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, Any]:
        """Uniform sample, densified into training tensors.

        Returns torch tensors so the caller can feed ``train_step`` directly;
        ``policy`` is (B, 4096) over from*64+to, and ``promo_target`` is (B, 4)
        over the promotion classes with ``promo_mask`` marking the rows where any
        promotion carried mass. Densifying here rather than in the training loop
        keeps the 4096-wide array off disk and out of every other consumer.
        """
        import torch

        n = len(self)
        if n == 0:
            raise ValueError("replay buffer is empty")
        idx = rng.integers(0, n, size=batch_size)

        policy = np.zeros((batch_size, POLICY_DIM), dtype=np.float32)
        promo_target = np.zeros((batch_size, 4), dtype=np.float32)
        promo_mask = np.zeros(batch_size, dtype=bool)
        for b, j in enumerate(idx):
            k = int(self._arr["policy_n"][j])
            if k == 0:
                continue
            ft = self._arr["policy_ft"][j, :k].astype(np.int64)
            pr = self._arr["policy_pr"][j, :k].astype(np.float32)
            cls = self._arr["promo_cls"][j, :k].astype(np.int64)
            np.add.at(policy[b], ft, pr)  # promotions share a (from,to) cell
            is_promo = cls > 0
            if is_promo.any():
                promo_mask[b] = True
                np.add.at(promo_target[b], cls[is_promo] - 1, pr[is_promo])
                s = promo_target[b].sum()
                if s > 0:
                    promo_target[b] /= s

        return {
            "tokens": torch.from_numpy(self._arr["boards"][idx].astype(np.int64)),
            "legals": torch.from_numpy(np.asarray(self._arr["legals"][idx])),
            "outcome": torch.from_numpy(self._arr["outcomes"][idx].astype(np.int64)),
            "policy": torch.from_numpy(policy),
            "promo_target": torch.from_numpy(promo_target),
            "promo_mask": torch.from_numpy(promo_mask),
            "iteration": torch.from_numpy(self._arr["iteration"][idx].astype(np.int64)),
        }

    def densify_one(self, i: int) -> np.ndarray:
        """Dense (4096,) improved policy for row ``i`` — used by the round-trip test."""
        out = np.zeros(POLICY_DIM, dtype=np.float32)
        k = int(self._arr["policy_n"][i])
        if k:
            np.add.at(
                out,
                self._arr["policy_ft"][i, :k].astype(np.int64),
                self._arr["policy_pr"][i, :k].astype(np.float32),
            )
        return out
