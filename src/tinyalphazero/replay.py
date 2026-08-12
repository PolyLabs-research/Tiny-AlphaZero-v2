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
    mover_latest.dat  int8    (C,)        1 if this position's mover searched with the latest weights
    meta.json                             capacity, total_written, dtypes, version

The improved policy is stored **sparse** because it is supported only on legal
moves — a dense (4096,) float32 row would be 16 KB of which ~99% is zero, and
500K of those is 8 GB. At 96 entries it is 384 B. ``MAX_POLICY_ENTRIES`` is a
hard cap, not a guess: a position with more than 96 legal moves keeps its
96 highest-probability entries and renormalises, which is recorded rather than
silent (see ``_pack_policy``).

**Crash-safety is a write-ordering contract, and this is it (chunk 9).**

1. **Single writer.** One process owns a buffer directory. Nothing here locks;
   the orchestrator is alternating and single-process by §7, and the contract is
   stated rather than enforced because enforcing it would mean inventing a
   locking protocol for a concurrency mode the design does not have.
2. **Position data is written before the head advances.** ``add`` fills every
   field of slot ``i`` and only then increments ``total_written``, which lives in
   RAM; the *durable* head is ``meta.json`` and moves only in ``flush``. A
   process killed between the two is indistinguishable from one that never wrote
   the rows: the slots hold bytes nothing points at, and the next writer
   overwrites them.
3. **A torn iteration loses at most its own partial games.** Everything written
   before the last ``flush`` is intact and readable, because the head never moved
   backwards over it. Resume uses :meth:`rewind_to` to discard a *flushed* but
   uncommitted iteration — see that method for the one honest caveat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# v2 (chunk 9) adds mover_latest. Bumping rather than defaulting: a Phase 2 run
# that silently read a chunk-8 ring would see mover_is_latest=0 everywhere and,
# with train_on_pool_moves=False, sample from an empty set.
VERSION = 3
#: Versions this build can open. Older rings lose the fields they predate.
_READABLE_VERSIONS = frozenset({2, 3})
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
    "mover_latest": ("int8", ()),
    # Root completed-Q at the moment this position was searched, side-to-move
    # perspective — the same convention as ``outcomes``. Version 3. Stored rather
    # than recomputed because a recomputation would use whatever net asks later,
    # not the net that actually searched: a different quantity wearing the same
    # name. float16 is ample for a target bounded in [-1, 1].
    "root_q": ("float16", ()),
}

#: Fields absent from rings written before their version. A ring that predates a
#: field does not get a fabricated one — it reports the field as *absent*, and
#: every consumer treats absence identically to a row that never had a search
#: (a rehearsal row). One meaning, one code path.
_FIELDS_SINCE: dict[str, int] = {"root_q": 3}


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

    def __init__(
        self, path: Path, capacity: int, total_written: int, version: int = VERSION
    ) -> None:
        self.path = Path(path)
        self.capacity = capacity
        self.total_written = total_written
        self.version = version
        self._arr: dict[str, np.memmap] = {}
        for name, (dtype, shape) in _FIELDS.items():
            if version < _FIELDS_SINCE.get(name, 0):
                continue  # this ring predates the field; it is absent, not zero
            self._arr[name] = np.memmap(
                self.path / f"{name}.dat",
                dtype=dtype,
                mode="r+",
                shape=(capacity, *shape) if shape else (capacity,),
            )

    def has(self, field: str) -> bool:
        """Does this ring carry ``field``? False for a ring older than it."""
        return field in self._arr

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
        version = int(meta.get("version", 0))
        # Older rings open read-and-write-able, minus the fields they predate.
        # Refusing them would strand every campaign written before a field
        # existed — including the two this project's results rest on — for no
        # gain, because "absent" is already a meaning the batch schema carries.
        if version not in _READABLE_VERSIONS:
            raise ValueError(
                f"replay version {version} is not readable; this build reads "
                f"{sorted(_READABLE_VERSIONS)}"
            )
        return cls(path, int(meta["capacity"]), int(meta["total_written"]), version=version)

    @classmethod
    def open_or_create(cls, path: str | Path, capacity: int = 500_000) -> ReplayBuffer:
        return cls.open(path) if (Path(path) / "meta.json").exists() else cls.create(path, capacity)

    def _write_meta(self) -> None:
        fields = {k: v[0] for k, v in _FIELDS.items() if self.has(k)}
        (self.path / "meta.json").write_text(
            json.dumps(
                {
                    "version": self.version,
                    "capacity": self.capacity,
                    "total_written": self.total_written,
                    "max_policy_entries": MAX_POLICY_ENTRIES,
                    "fields": fields,
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
        mover_is_latest: bool = True,
        root_q: float = 0.0,
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
        if self.has("root_q"):
            self._arr["root_q"][i] = root_q
        self._arr["iteration"][i] = iteration
        self._arr["resign_off"][i] = int(resign_disabled)
        self._arr["mover_latest"][i] = int(mover_is_latest)
        # Last: every field of slot `i` is on the page before anything points at
        # it. See the write-ordering contract in the module docstring.
        self.total_written += 1

    def rewind_to(self, total_written: int) -> None:
        """Restore the write position recorded at the end of the last iteration.

        Resume-only repair. If a process died after ``play_games`` flushed but
        before its iteration was committed, the durable head is ahead of the last
        committed state; rewinding drops those games so the re-run does not store
        them twice and ``replay_size`` in the log stays consistent with the
        iteration count.

        **The honest caveat:** this restores the write *position*, it does not
        un-write bytes. If the torn iteration wrapped past the end of the ring,
        the slots it overwrote now hold valid, newer-than-expected rows until
        writing catches up again. Uniform sampling cannot tell the difference —
        every row is still a real position with a real search target — but the
        buffer is not byte-identical to one that never saw the crash.
        """
        if total_written > self.total_written:
            raise ValueError(
                f"rewind_to({total_written}) is ahead of the current head "
                f"{self.total_written}; the ring only ever moves forward"
            )
        self.total_written = int(total_written)
        self._write_meta()

    # -- reading ---------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        latest_only: bool = False,
    ) -> dict[str, Any]:
        """Uniform sample, densified into training tensors.

        Returns torch tensors so the caller can feed ``train_step_phase2``
        directly; ``policy`` is (B, 4096) over from*64+to, and ``promo_target`` is
        (B, 4) over the promotion classes at the **acted** (from, to) cell, with
        ``promo_mask`` marking the rows where the acted move is itself a
        promotion. Densifying here rather than in the training loop keeps the
        4096-wide array off disk and out of every other consumer.

        ``latest_only`` restricts the draw to positions whose mover searched with
        the latest weights (``phase2.train_on_pool_moves=False``). It is a filter
        over what is already stored, which is the entire reason the tag is written
        for every row rather than inferred later: the ablation costs a config flag,
        not a regenerated buffer.
        """
        import torch

        n = len(self)
        if n == 0:
            raise ValueError("replay buffer is empty")
        if latest_only:
            eligible = np.flatnonzero(np.asarray(self._arr["mover_latest"][:n]))
            if eligible.size == 0:
                raise ValueError(
                    "replay buffer holds no latest-weights positions "
                    "(phase2.train_on_pool_moves=False with a pool-only buffer)"
                )
            idx = eligible[rng.integers(0, eligible.size, size=batch_size)]
        else:
            idx = rng.integers(0, n, size=batch_size)

        moves = self._arr["moves"][idx].astype(np.int64)  # (B, 2) from, to
        acted_ft = moves[:, 0] * 64 + moves[:, 1]

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
            # The promotion head models P(promo | from, to) — a *conditional*. Its
            # target therefore has to be built from the promotion entries sitting
            # on the acted cell, not from every promotion entry in the row: a
            # position where the search prefers a rook move but a pawn could also
            # promote would otherwise train the head at the rook's square pair
            # toward the pawn's promotion distribution. Legal, well-formed, and
            # teaching the wrong conditional — the S3 defect family exactly.
            on_acted = (cls > 0) & (ft == acted_ft[b])
            if on_acted.any():
                promo_mask[b] = True
                np.add.at(promo_target[b], cls[on_acted] - 1, pr[on_acted])
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
            # The (from, to) cell the promotion head is read at. The promo head is
            # conditioned on a square pair, so a (B, 4) target is meaningless
            # without saying which cell it belongs to.
            "from_sq": torch.from_numpy(moves[:, 0]),
            "to_sq": torch.from_numpy(moves[:, 1]),
            "iteration": torch.from_numpy(self._arr["iteration"][idx].astype(np.int64)),
            "mover_is_latest": torch.from_numpy(self._arr["mover_latest"][idx].astype(np.int64)),
            # Search-Q and whether it is real. A ring older than the field reports
            # every row as unmasked rather than as 0.0, which is a legal value and
            # would silently train the blend toward "the game was a draw".
            "root_q": torch.from_numpy(
                self._arr["root_q"][idx].astype(np.float32)
                if self.has("root_q")
                else np.zeros(batch_size, dtype=np.float32)
            ),
            "root_q_mask": torch.from_numpy(np.full(batch_size, self.has("root_q"), dtype=bool)),
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
