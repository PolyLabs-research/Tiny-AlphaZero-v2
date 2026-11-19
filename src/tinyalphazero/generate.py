"""Self-play from one fixed network, for the teacher to label (PREREG-p2_ae).

A training run's self-play (``phase2``) plays with the network it is training, into the
ring it trains from. This plays with one checkpoint that never changes, in chunks of
``chunk`` games, into a ring sized so it cannot wrap; the rows are pooled and labelled
later (``pool_positions``, ``label_teacher``). Chunk ``c``'s games are
``wave_specs(chunk_seed(seed, c), chunk, cfg.selfplay)``: the network against itself, no
league, under the config's resignation, random-opening and ply-cap rules; each row's
``iteration`` is its chunk. So the data are a function of (checkpoint, config, seed, the
chunk numbers) and not of which machine played which chunks, nor of how many actor
processes played them (``actors``' contract: the same games, stored in the same order).

Resumable: after each chunk the ring is flushed, then ``state.json`` (temp file and
``os.replace``) records the chunks done and the ring's head; a restart checks the
parameters match, rewinds the ring to that head and plays on from the next chunk.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tinyalphazero.actors import play_specs_parallel
from tinyalphazero.config import Config, config_fingerprint
from tinyalphazero.phase2 import iteration_seed
from tinyalphazero.replay import ReplayBuffer
from tinyalphazero.selfplay import LATEST, SelfPlayStats, play_specs, record_games, wave_specs

#: Seed stream of a generation chunk; phase 2 uses 1-3 and 100 + wave.
STREAM_GENERATE = 7


def chunk_seed(seed: int, chunk: int) -> int:
    """The wave seed of chunk ``chunk`` of a generation run seeded ``seed``."""
    return iteration_seed(seed, chunk, STREAM_GENERATE)


def generate(
    evaluate_batch: Any,
    cfg: Config,
    out: Path,
    seed: int,
    chunk_start: int,
    chunks: int,
    chunk: int = 512,
    processes: int = 1,
    num_sims: int | None = None,
    device: str = "cpu",
    on_chunk: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Play chunks ``chunk_start .. chunk_start + chunks - 1`` into ``out/replay``.

    ``evaluate_batch`` answers leaves (``torch_evaluate_batch``; float32). Returns the
    final state. ``on_chunk`` gets each chunk's log row after it is committed.
    """
    if chunks < 1 or chunk < 1 or chunk_start < 0:
        raise ValueError("chunks and chunk must be >= 1, chunk_start >= 0")
    sp = cfg.selfplay
    sims = num_sims if num_sims is not None else sp.sims_per_move
    params = {
        "seed": seed,
        "chunk_start": chunk_start,
        "chunks": chunks,
        "chunk": chunk,
        "sims": sims,
        "config_fingerprint": config_fingerprint(cfg),
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if {k: state.get(k) for k in params} != params:
            raise ValueError(f"{state_path} was written for other parameters: {state}")
        ring = ReplayBuffer.open(out / "replay")
        ring.rewind_to(int(state["total_written"]))
    else:
        # A game stores at most one row per ply, so this capacity never wraps.
        ring = ReplayBuffer.create(out / "replay", capacity=chunks * chunk * sp.max_plies)
        state = {**params, "chunks_done": 0, "total_written": 0}
    for c in range(chunk_start + int(state["chunks_done"]), chunk_start + chunks):
        specs = wave_specs(chunk_seed(seed, c), chunk, sp)
        t0 = time.perf_counter()
        if processes > 1:
            results, runner = play_specs_parallel(
                {LATEST: evaluate_batch}, specs, cfg, sims, processes=processes, league=False
            )
        else:
            results, runner = play_specs(evaluate_batch, specs, cfg, sims, device=device)
        stats = SelfPlayStats()
        record_games(results, specs, stats, ring, iteration=c)
        ring.flush()
        stats.seconds = time.perf_counter() - t0
        stats.sims, stats.forwards = runner["sims"], runner["forwards"]
        state.update(chunks_done=c - chunk_start + 1, total_written=ring.total_written)
        tmp = out / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(tmp, state_path)
        row = {"chunk": c, "rows": ring.total_written, **stats.to_dict()}
        with (out / "generate.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if on_chunk is not None:
            on_chunk(row)
    return state
