"""Parallel self-play: CPU actor processes feeding one GPU-owning process (AGENTS.md §6).

Self-play is ~89% of an iteration and, until this module, ran in one Python
process: during p2_f the box's 32 hardware threads were 92.6% idle (``sar``,
2026-10-23) while the GPU sat 20–35% busy. The search is python-chess move
generation and tree bookkeeping, which is CPU work that parallelises across
processes and not across threads.

**The shape is the one AGENTS.md §6 prescribes.** ``processes`` actor processes
each play a shard of the iteration's games with their own ``BatchRunner``. When an
actor's runner flushes, the leaf batch goes to this (the calling) process, which
owns the GPU and the networks, runs the forward, and writes the answer back. The
actors never import torch, never see a CUDA context, and never touch the ring.

**Same games, by construction.** Every game is fixed by its ``GameSpec`` (seed,
resignation flag, league assignment), and a game's search cannot depend on which
other games share its batch (the ``BatchRunner`` contract). Specs are drawn by
``selfplay.wave_specs`` exactly as the in-process path draws them, and the results
are recorded in spec order by ``selfplay.record_games`` — the order the in-process
path has always stored in. With an evaluator whose rows do not depend on their
batch, the ring is therefore byte-identical to the in-process ring; the equivalence
test asserts it. On the GPU, bf16 rows do depend slightly on batch composition, so
the games are the same draws but not bit-identical — as they already were between
any two in-process runs (``nondeterministic`` on every rung).

**Transport.** One request in flight per actor, so each actor owns one fixed slot
in shared memory: tokens in, policy / promotion / value out, sized for its shard.
Control messages are small and go through one queue (actor → server) and one pipe
per actor (server → actor). The server answers every request queued at the time it
looks, grouped by network, so concurrent actors share forwards.

**Failure is loud.** An actor that raises sends its traceback and the server
re-raises it; an actor that dies without reporting is noticed within a poll
interval; an actor whose server dies exits on its next wait. Shared memory is
unlinked by the server in ``finally`` and, if the server is killed outright, by
multiprocessing's resource tracker.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
import secrets
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from tinyalphazero.config import Config
from tinyalphazero.selfplay import LATEST, GameResult, GameSpec, play_specs

_SEQ = 72
_POLICY = 64 * 64
_PROMO = 64 * 64 * 4
#: Shared-memory segment prefix, so a leak is attributable from ``/dev/shm`` alone.
SHM_PREFIX = "taz_actor_"


class _Slot:
    """One actor's request/response arrays in shared memory."""

    FIELDS: dict[str, tuple[Any, int]] = {
        "tokens": (np.int64, _SEQ),
        "policy": (np.float32, _POLICY),
        "promo": (np.float32, _PROMO),
        "value": (np.float32, 1),
    }

    def __init__(self, cap: int, names: dict[str, str] | None = None) -> None:
        self.cap = cap
        self._shm: dict[str, shared_memory.SharedMemory] = {}
        self.arr: dict[str, np.ndarray] = {}
        for field, (dtype, width) in self.FIELDS.items():
            if names is None:
                nbytes = max(1, cap * width * np.dtype(dtype).itemsize)
                shm = shared_memory.SharedMemory(
                    create=True, size=nbytes, name=SHM_PREFIX + secrets.token_hex(8)
                )
            else:
                shm = shared_memory.SharedMemory(name=names[field])
            self._shm[field] = shm
            shape = (cap, width) if width > 1 else (cap,)
            self.arr[field] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    @property
    def names(self) -> dict[str, str]:
        return {field: shm.name for field, shm in self._shm.items()}

    def close(self) -> None:
        self.arr.clear()  # views first: a buffer with live exports cannot close
        for shm in self._shm.values():
            shm.close()

    def unlink(self) -> None:
        for shm in self._shm.values():
            with suppress(FileNotFoundError):
                shm.unlink()


class _ProxyEvaluator:
    """An actor's ``evaluate_batch``: ships the batch to the server and waits.

    One instance per network key, so the runner's per-evaluator grouping in
    ``_flush`` still separates the learner's leaves from a league opponent's.
    The answer is copied out of the slot: the search keeps references into what
    it is handed, and the slot is overwritten by the next request.
    """

    def __init__(self, wid: int, key: str, slot: _Slot, conn: Any, requests: Any) -> None:
        self.wid, self.key, self.slot, self.conn, self.requests = wid, key, slot, conn, requests

    def __call__(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(tokens)
        self.slot.arr["tokens"][:n] = tokens
        self.requests.put(("eval", self.wid, n, self.key))
        _await_answer(self.conn)
        a = self.slot.arr
        return (
            a["policy"][:n].reshape(n, 64, 64).copy(),
            a["promo"][:n].reshape(n, 64, 64, 4).copy(),
            a["value"][:n].copy(),
        )


def _await_answer(conn: Any, poll_s: float = 1.0) -> None:
    """Block for the server's answer, giving up if the server has died."""
    parent = mp.parent_process()
    while not conn.poll(poll_s):
        if parent is not None and not parent.is_alive():
            os._exit(3)  # nothing to report to; the run is already gone
    conn.recv()


def _actor_main(
    wid: int,
    specs: list[GameSpec],
    cfg: Config,
    num_sims: int,
    league: bool,
    keys: list[str],
    slot_names: dict[str, str],
    conn: Any,
    requests: Any,
) -> None:
    """Actor process entry: play this shard, send the scored games back."""
    slot = None
    try:
        slot = _Slot(len(specs), names=slot_names)
        proxies = {k: _ProxyEvaluator(wid, k, slot, conn, requests) for k in keys}
        results, runner_stats = play_specs(
            proxies[LATEST],
            specs,
            cfg,
            num_sims,
            evaluators=proxies if league else None,
            league=league,
        )
        requests.put(("done", wid, results, runner_stats))
    except BaseException:
        requests.put(("error", wid, traceback.format_exc()))
    finally:
        if slot is not None:
            slot.close()


@contextmanager
def _actor_environment() -> Iterator[None]:
    """What a spawned actor inherits at ``start()``, set for exactly that long.

    Single-threaded math libraries: an actor's numpy work is small and per-row,
    and N actors each spinning a BLAS pool over 32 threads would fight the very
    cores they were started to use. And no ``__main__``: spawn re-imports the
    parent's main module in every child, and ``scripts/train_phase2.py`` imports
    torch — ~300 MB of host RAM per actor, on the box where host RAM is the scarce
    pool, for a library the actors never call. The actor's entry point lives in
    this module, so the child needs nothing from ``__main__``.
    """
    env_keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved_env = {k: os.environ.get(k) for k in env_keys}
    main = sys.modules.get("__main__")
    saved_main = {
        k: main.__dict__[k] for k in ("__file__", "__spec__") if main and k in main.__dict__
    }
    try:
        for k in env_keys:
            os.environ[k] = "1"
        if main is not None:
            main.__dict__["__spec__"] = None
            main.__dict__.pop("__file__", None)
        yield
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if main is not None:
            main.__dict__.update(saved_main)
            if "__spec__" not in saved_main:
                main.__dict__.pop("__spec__", None)


def play_specs_parallel(
    evaluators: dict[str, Any],
    specs: list[GameSpec],
    cfg: Config,
    num_sims: int,
    processes: int,
    league: bool = True,
    poll_s: float = 1.0,
) -> tuple[list[GameResult], dict[str, int]]:
    """``selfplay.play_specs`` across ``processes`` actor processes; results in spec order.

    ``evaluators`` are the real ``evaluate_batch`` callables, keyed as the specs'
    assignments name them (``LATEST`` always), and run here, in the caller. They
    must return float32 arrays, as ``torch_evaluate_batch`` does: the slots are
    float32, and a float64 value would be rounded on the way back where the
    in-process path would not round it.

    Games are dealt round-robin, so each actor gets a like mix of league and
    learner-only games. Returns the scored games and ``{"sims", "forwards"}``,
    where ``forwards`` counts the forwards actually run here (fewer than the
    actors' flushes, because concurrent requests share one).
    """
    if LATEST not in evaluators:
        raise ValueError(f"evaluators must include {LATEST!r}")
    if not specs:
        return [], {"sims": 0, "forwards": 0}
    k = max(1, min(processes, len(specs)))
    shards = [list(range(w, len(specs), k)) for w in range(k)]
    keys = sorted(evaluators)
    ctx = mp.get_context("spawn")
    requests = ctx.Queue()
    slots: list[_Slot] = []
    conns: list[Any] = []
    procs: list[Any] = []
    try:
        with _actor_environment():
            for w, shard in enumerate(shards):
                slot = _Slot(len(shard))
                slots.append(slot)
                recv_end, send_end = ctx.Pipe(duplex=False)
                proc = ctx.Process(
                    target=_actor_main,
                    args=(
                        w,
                        [specs[i] for i in shard],
                        cfg,
                        num_sims,
                        league,
                        keys,
                        slot.names,
                        recv_end,
                        requests,
                    ),
                    name=f"taz-actor-{w}",
                    daemon=True,
                )
                proc.start()
                recv_end.close()  # the actor holds the only receiving end now
                procs.append(proc)
                conns.append(send_end)

        results: list[GameResult | None] = [None] * len(specs)
        sims = forwards = 0
        done: set[int] = set()
        while len(done) < k:
            try:
                batch = [requests.get(timeout=poll_s)]
            except queue_mod.Empty:
                for w, proc in enumerate(procs):
                    if w not in done and not proc.is_alive():
                        raise RuntimeError(
                            f"self-play actor {w} exited (code {proc.exitcode}) without reporting"
                        ) from None
                continue
            while True:
                try:
                    batch.append(requests.get_nowait())
                except queue_mod.Empty:
                    break

            pending: dict[str, list[tuple[int, int]]] = {}
            for msg in batch:
                if msg[0] == "eval":
                    _, w, n, key = msg
                    pending.setdefault(key, []).append((w, n))
                elif msg[0] == "done":
                    _, w, shard_results, runner_stats = msg
                    for local, i in enumerate(shards[w]):
                        results[i] = shard_results[local]
                    sims += runner_stats["sims"]
                    done.add(w)
                else:
                    raise RuntimeError(f"self-play actor {msg[1]} failed:\n{msg[2]}")

            for key, reqs in pending.items():
                tokens = np.concatenate([slots[w].arr["tokens"][:n] for w, n in reqs])
                policy, promo, value = evaluators[key](tokens)
                forwards += 1
                total = len(tokens)
                policy = np.asarray(policy, dtype=np.float32).reshape(total, _POLICY)
                promo = np.asarray(promo, dtype=np.float32).reshape(total, _PROMO)
                value = np.asarray(value, dtype=np.float32).reshape(total)
                off = 0
                for w, n in reqs:
                    a = slots[w].arr
                    a["policy"][:n] = policy[off : off + n]
                    a["promo"][:n] = promo[off : off + n]
                    a["value"][:n] = value[off : off + n]
                    off += n
                    conns[w].send(None)

        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            # Dropping them would shift every later game onto the wrong spec.
            raise RuntimeError(f"self-play actors returned no result for games {missing[:8]}")
        return list(results), {"sims": sims, "forwards": forwards}  # type: ignore[arg-type]
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=10)
        for conn in conns:
            conn.close()
        for slot in slots:
            slot.close()
            slot.unlink()
        requests.cancel_join_thread()
        requests.close()
