"""Parallel self-play actors (``tinyalphazero.actors``).

The claim the module makes is strong — *the same games, stored in the same order*,
whatever the number of processes — so the tests are exact: every ring field is
compared byte for byte against the in-process path. That needs an evaluator whose
answer for a row cannot depend on the batch the row arrives in. A torch net on the
CPU does not promise that (BLAS blocking varies with batch size), so the
evaluators here are integer arithmetic with one final IEEE division: row-wise, and
the same in any SIMD lane.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from tinyalphazero.actors import SHM_PREFIX, _actor_environment, play_specs_parallel
from tinyalphazero.config import Config, load_config, save_config
from tinyalphazero.phase2 import Phase2Run
from tinyalphazero.replay import ReplayBuffer
from tinyalphazero.selfplay import (
    LATEST,
    GameAssignment,
    SelfPlayStats,
    play_games,
    record_games,
    wave_specs,
)

REPO = Path(__file__).resolve().parents[1]
MINI = REPO / "configs" / "mini.yaml"

_GRID = np.arange(4096, dtype=np.int64)
_PRIMES = np.array([(i * 7919) % 104729 + 1 for i in range(72)], dtype=np.int64)


def _evaluator(a: int, b: int):
    """Deterministic and batch-independent, with a material term so games resign.

    A side two pieces down sees every reply as bad, which is what makes
    resignation (and the resign-disabled bookkeeping) fire at varied plies.
    """

    def evaluate(tokens: np.ndarray):
        t = np.asarray(tokens, dtype=np.int64)
        h = (t * _PRIMES).sum(axis=1) % 1009
        logits = ((h[:, None] * a + _GRID[None, :] * b) % 101).astype(np.float32) / np.float32(10)
        sq = t[:, :64]
        white = ((sq >= 1) & (sq <= 6)).sum(axis=1)
        black = ((sq >= 7) & (sq <= 12)).sum(axis=1)
        diff = np.where(t[:, 64] == 13, white - black, black - white)
        v = np.clip(diff * 45 + ((h * a) % 21) - 10, -99, 99)
        promo = np.full((len(t), 64, 64, 4), 0.25, dtype=np.float32)
        return logits.reshape(len(t), 64, 64), promo, v.astype(np.float32) / np.float32(100)

    return evaluate


def _scenario(openings: int = 0) -> tuple[Config, dict, list[GameAssignment]]:
    cfg = Config()
    sp = cfg.selfplay
    sp.opening_random_plies = openings
    sp.games_per_iteration, sp.concurrent_games, sp.sims_per_move, sp.max_plies = 16, 8, 8, 40
    sp.resign_threshold, sp.resign_consecutive, sp.resign_disabled_frac = -0.6, 3, 0.25
    evaluators = {LATEST: _evaluator(31, 17), "snap": _evaluator(13, 29)}
    assignments = [
        GameAssignment(white=LATEST, black="snap")
        if i % 4 == 1
        else GameAssignment(white="snap", black=LATEST)
        if i % 4 == 3
        else GameAssignment()
        for i in range(sp.games_per_iteration)
    ]
    return cfg, evaluators, assignments


def _ring_bytes(buf: ReplayBuffer) -> dict[str, bytes]:
    n = buf.total_written
    return {k: np.ascontiguousarray(v[:n]).tobytes() for k, v in buf._arr.items()}


def _stats_view(stats: SelfPlayStats) -> dict:
    s = stats.to_dict()
    for volatile in ("seconds", "sims_per_s", "games_per_hour", "forwards"):
        s.pop(volatile)
    s.update(
        pool_positions=stats.pool_positions,
        plies=stats.plies,
        decisive_plies=stats.decisive_plies,
        move1_entropies=stats.move1_entropies,
        move1_prior_entropies=stats.move1_prior_entropies,
    )
    return s


def _shm_segments() -> set[str]:
    shm = Path("/dev/shm")
    return (
        {p.name for p in shm.iterdir() if p.name.startswith(SHM_PREFIX)} if shm.exists() else set()
    )


def _in_process(tmp_path: Path, openings: int = 0) -> tuple[ReplayBuffer, SelfPlayStats]:
    """The reference: play_games wave by wave, exactly as Phase2Run._generate does."""
    cfg, evaluators, assignments = _scenario(openings)
    sp = cfg.selfplay
    buf = ReplayBuffer.create(tmp_path / "in_process", capacity=4096)
    merged = SelfPlayStats()
    for w, start in enumerate(range(0, sp.games_per_iteration, sp.concurrent_games)):
        n = min(sp.concurrent_games, sp.games_per_iteration - start)
        _traces, stats = play_games(
            evaluators[LATEST],
            n,
            cfg=cfg,
            seed=1000 + w,
            num_sims=sp.sims_per_move,
            iteration=3,
            buffer=buf,
            evaluators=evaluators,
            assignments=assignments[start : start + n],
        )
        merged.absorb(stats)
    return buf, merged


@pytest.mark.parametrize(("processes", "openings"), [(1, 0), (3, 0), (3, 6)])
def test_actors_store_exactly_the_in_process_ring(
    tmp_path: Path, processes: int, openings: int
) -> None:
    """Same specs, same games, same order: every ring field byte-identical.

    The scenario is built to reach the branches a quiet run would not: league games
    on both colours, resignations, resign-disabled games that would have resigned,
    and games ending at the ply cap; and, with ``openings``, random-opening starts.
    """
    ref_buf, ref_stats = _in_process(tmp_path, openings)
    assert ref_stats.results.get("resign", 0) > 0 and ref_stats.would_have_resigned > 0
    assert ref_stats.pool_positions > 0

    cfg, evaluators, assignments = _scenario(openings)
    sp = cfg.selfplay
    specs = []
    for w, start in enumerate(range(0, sp.games_per_iteration, sp.concurrent_games)):
        n = min(sp.concurrent_games, sp.games_per_iteration - start)
        specs += wave_specs(1000 + w, n, sp, assignments[start : start + n])

    buf = ReplayBuffer.create(tmp_path / f"actors_{processes}", capacity=4096)
    stats = SelfPlayStats()
    results, runner_stats = play_specs_parallel(
        evaluators, specs, cfg, sp.sims_per_move, processes=processes
    )
    record_games(results, specs, stats, buf, iteration=3)
    stats.sims = runner_stats["sims"]

    assert buf.total_written == ref_buf.total_written
    assert _ring_bytes(buf) == _ring_bytes(ref_buf)
    assert _stats_view(stats) == _stats_view(ref_stats)


def test_phase2_generate_with_actors_matches_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real ``_generate``, league on: the wiring, not just the function.

    A second pool member makes ``assign_league`` hand out opponent games; the
    networks are swapped for the batch-independent stubs so the comparison can be
    exact.
    """
    stubs = {LATEST: _evaluator(31, 17)}

    def fake_build(self, assignments):
        keys = {a.white for a in assignments} | {a.black for a in assignments}
        return {k: stubs.get(k) or _evaluator(13 + len(k), 29) for k in keys | {LATEST}}

    monkeypatch.setattr(Phase2Run, "_build_evaluators", fake_build)
    rings = {}
    for processes in (1, 3):
        cfg = load_config(str(MINI))
        cfg.selfplay.actor_processes = processes
        cfg.selfplay.games_per_iteration, cfg.selfplay.max_plies = 12, 30
        run = Phase2Run.create(tmp_path / f"run{processes}", cfg, torch.device("cpu"))
        run.pool.snapshots.insert(0, "opponent.pt")  # an older snapshot the league can draw
        stats = run._generate(1)
        assert stats.games == 12 and stats.pool_positions > 0
        rings[processes] = _ring_bytes(run.buffer)
    assert rings[1] == rings[3]


def test_an_actor_that_raises_fails_the_call_loudly(tmp_path: Path) -> None:
    """A spec naming a network the server does not have: the actor's own check fires."""
    cfg, evaluators, _ = _scenario()
    specs = wave_specs(7, 4, cfg.selfplay, [GameAssignment(white="missing")] * 4)
    before = _shm_segments()
    with pytest.raises(RuntimeError, match="no evaluator supplied"):
        play_specs_parallel(evaluators, specs, cfg, 4, processes=2)
    assert _shm_segments() == before, "a failed call leaked shared memory"


def test_a_failing_network_propagates_and_releases_everything(tmp_path: Path) -> None:
    cfg, evaluators, _ = _scenario()

    def broken(tokens):
        raise FloatingPointError("the GPU said no")

    specs = wave_specs(7, 4, cfg.selfplay)
    before = _shm_segments()
    with pytest.raises(FloatingPointError, match="the GPU said no"):
        play_specs_parallel({LATEST: broken}, specs, cfg, 4, processes=2)
    assert _shm_segments() == before


def test_a_normal_call_leaves_no_shared_memory_behind(tmp_path: Path) -> None:
    cfg, evaluators, _ = _scenario()
    before = _shm_segments()
    results, _stats = play_specs_parallel(
        evaluators, wave_specs(3, 6, cfg.selfplay), cfg, 4, processes=3, league=False
    )
    assert len(results) == 6
    assert _shm_segments() == before


def test_the_actor_import_path_never_loads_torch() -> None:
    """Actors import this module and its dependencies only — never torch, never CUDA."""
    out = subprocess.run(
        [sys.executable, "-c", "import sys, tinyalphazero.actors; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


def test_actor_environment_is_restored_after_start() -> None:
    main = sys.modules["__main__"]
    before_env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    before_main = {k: main.__dict__.get(k, "<absent>") for k in ("__file__", "__spec__")}
    with _actor_environment():
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert main.__dict__.get("__spec__") is None and "__file__" not in main.__dict__
    assert {k: os.environ.get(k) for k in before_env} == before_env
    assert {k: main.__dict__.get(k, "<absent>") for k in before_main} == before_main


# ---------------------------------------------------------------------------
# SIGKILL with actors live (AGENTS.md §6)
# ---------------------------------------------------------------------------


def _children(pid: int) -> dict[int, str]:
    """Direct children of ``pid`` with their start times, read from /proc."""
    kids: dict[int, str] = {}
    for task in Path(f"/proc/{pid}/task").iterdir():
        for child in (task / "children").read_text().split():
            stat = Path(f"/proc/{child}/stat").read_text()
            kids[int(child)] = stat.rsplit(")", 1)[1].split()[19]
    return kids


def _alive(pid: int, start: str) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19] == start
    except FileNotFoundError:
        return False


@pytest.mark.slow
def test_sigkill_with_actors_live_leaves_no_orphans_and_resumes(tmp_path: Path) -> None:
    """Kill the training process mid-generation while its actors are playing.

    The actors must notice and exit (their next wait finds the parent gone), the
    resource tracker must reclaim the slots, and the run must resume as any killed
    run does — the actors never wrote the ring, so there is nothing of theirs to
    repair.
    """
    cfg = load_config(str(MINI))
    cfg.selfplay.actor_processes = 2
    cfg.selfplay.games_per_iteration, cfg.selfplay.sims_per_move = 24, 24
    config = tmp_path / "actors.yaml"
    save_config(cfg, str(config))
    out = tmp_path / "run"
    before = _shm_segments()

    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(REPO / "scripts" / "train_phase2.py"),
            "--config",
            str(config),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--iterations",
            "2",
        ],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    deadline = time.time() + 300
    while time.time() < deadline:
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line or "[stage] selfplay iteration=2" in line:
            break
    time.sleep(4.0)  # past spawn: the actors are mid-game
    kids = _children(proc.pid)
    # This run's children, read from its own /proc entry: the two actors are the
    # spawn_main ones (the third child, if present, is the resource tracker).
    actors = [p for p in kids if b"spawn_main" in Path(f"/proc/{p}/cmdline").read_bytes()]
    assert len(actors) == 2, f"expected both actors live at the kill, found {kids}"
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=30)

    deadline = time.time() + 30
    while time.time() < deadline and any(_alive(p, s) for p, s in kids.items()):
        time.sleep(0.5)
    assert not any(_alive(p, s) for p, s in kids.items()), "actors outlived the killed run"
    deadline = time.time() + 30
    while time.time() < deadline and _shm_segments() != before:
        time.sleep(0.5)
    assert _shm_segments() == before, "the killed run's slots were never reclaimed"

    state = json.loads((out / "state.json").read_text())
    assert state["iteration"] == 1
    resumed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(REPO / "scripts" / "train_phase2.py"),
            "--resume",
            str(out),
            "--iterations",
            "1",
            "--device",
            "cpu",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert resumed.returncode == 0, resumed.stderr[-3000:]
    assert json.loads((out / "state.json").read_text())["iteration"] == 2
