"""Self-play from one fixed network (tinyalphazero.generate, PREREG-p2_ae).

The data must be a function of (network, config, seed, chunk numbers) alone: resuming
after a kill, splitting the chunks between machines, or changing the number of actor
processes must store the same rows. Uses ``test_actors``' batch-independent evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_actors import _evaluator, _ring_bytes
from tinyalphazero.config import Config
from tinyalphazero.generate import chunk_seed, generate
from tinyalphazero.replay import ReplayBuffer


def _cfg() -> Config:
    cfg = Config()
    sp = cfg.selfplay
    sp.sims_per_move, sp.max_plies, sp.opening_random_plies = 8, 30, 6
    sp.resign_threshold, sp.resign_consecutive, sp.resign_disabled_frac = -0.6, 3, 0.25
    return cfg


def _rows(path: Path) -> dict[str, bytes]:
    return _ring_bytes(ReplayBuffer.open(path / "replay", readonly=True))


def test_chunks_resume_and_split_store_the_same_rows(tmp_path: Path) -> None:
    ev, cfg = _evaluator(31, 17), _cfg()
    whole = generate(ev, cfg, tmp_path / "whole", seed=5, chunk_start=0, chunks=4, chunk=6)
    assert whole["chunks_done"] == 4

    class Killed(Exception):
        pass

    def die_after_two(row: dict) -> None:
        if row["chunk"] == 1:
            raise Killed

    with pytest.raises(Killed):
        generate(ev, cfg, tmp_path / "resumed", 5, 0, 4, chunk=6, on_chunk=die_after_two)
    assert json.loads((tmp_path / "resumed" / "state.json").read_text())["chunks_done"] == 2
    generate(ev, cfg, tmp_path / "resumed", 5, 0, 4, chunk=6)
    assert _rows(tmp_path / "resumed") == _rows(tmp_path / "whole")

    generate(ev, cfg, tmp_path / "a", 5, 0, 2, chunk=6)
    generate(ev, cfg, tmp_path / "b", 5, 2, 2, chunk=6)
    a, b, w = _rows(tmp_path / "a"), _rows(tmp_path / "b"), _rows(tmp_path / "whole")
    assert all(a[k] + b[k] == w[k] for k in w)
    ring_b = ReplayBuffer.open(tmp_path / "b" / "replay", readonly=True)
    assert set(np.asarray(ring_b._arr["iteration"][: ring_b.total_written]).tolist()) == {2, 3}


def test_actor_processes_change_throughput_not_rows(tmp_path: Path) -> None:
    ev, cfg = _evaluator(31, 17), _cfg()
    generate(ev, cfg, tmp_path / "one", 5, 0, 1, chunk=8, processes=1)
    generate(ev, cfg, tmp_path / "three", 5, 0, 1, chunk=8, processes=3)
    assert _rows(tmp_path / "three") == _rows(tmp_path / "one")


def test_a_resume_with_other_parameters_is_refused(tmp_path: Path) -> None:
    ev, cfg = _evaluator(31, 17), _cfg()
    generate(ev, cfg, tmp_path / "g", 5, 0, 1, chunk=4)
    with pytest.raises(ValueError):
        generate(ev, cfg, tmp_path / "g", 6, 0, 1, chunk=4)


def test_chunk_seeds_are_distinct_and_fixed() -> None:
    seeds = [chunk_seed(5, c) for c in range(200)]
    assert len(set(seeds)) == 200
    assert chunk_seed(5, 0) == chunk_seed(5, 0) != chunk_seed(6, 0)
