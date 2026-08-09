"""Chunk 9: the Phase 2 loop (spec §7).

Three kinds of test live here, and only the first kind is cheap.

The unit tests cover the pieces where a mistake is silent: a log row that a
dashboard will read months from now, a league draw whose marginals nobody checks,
and the batched runner's new ability to answer one buffer with several networks.

The **golden mini-run** (``-m golden``) is the loop itself, end to end, in under
three minutes — it is the test that says the thing runs at all.

The **crash-resume** tests (``-m slow``) kill a real subprocess with SIGKILL at two
different points inside an iteration and require the run to come back. Spec §9.2
and AGENTS.md §6 both ask for resumability; a 24-hour unattended campaign is a plan
rather than a hope only if the kill has actually been performed.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from tinyalphazero import logschema
from tinyalphazero.config import (
    Config,
    SelfPlayConfig,
    config_fingerprint,
    load_config,
    save_config,
)
from tinyalphazero.mcts.batch_runner import BatchRunner, uniform_stub_batch
from tinyalphazero.phase2 import Phase2Run, iteration_seed, run_phase2
from tinyalphazero.replay import ReplayBuffer
from tinyalphazero.selfplay import LATEST, CheckpointPool, GameAssignment, assign_league
from tinyalphazero.train import load_checkpoint

REPO = Path(__file__).resolve().parents[1]
MINI = REPO / "configs" / "mini.yaml"


# ---------------------------------------------------------------------------
# The log schema — chunk 11's dashboard reads this and nothing else
# ---------------------------------------------------------------------------


def _row(**over) -> dict:
    base = {
        "iteration": 1,
        "git_sha": "abc1234",
        "games": 8,
        "positions": 640,
        "mean_plies": 80.0,
        "mean_plies_decisive": None,
        "decisive_rate": 0.0,
        "draw_rate": 1.0,
        "draw_rate_ply_cap": 1.0,
        "draw_rate_draw_claim": 0.0,
        "draw_rate_auto": 0.0,
        "resign_rate": 0.0,
        "false_resign_rate": None,
        "move1_policy_entropy": 2.9,
        "move1_prior_entropy": 2.9,
        "pool_moves_frac": 0.0,
        "train_steps": 20,
        "skipped_steps": 0,
        "loss": 4.5,
        "policy_loss": 3.4,
        "promo_loss": 0.1,
        "value_loss": 1.0,
        "legality_loss": 0.0,
        "grad_norm": 1.4,
        "lr": 2e-4,
        "replay_size": 640,
        "pool_size": 2,
        "sims": 10240,
        "sims_per_s": 1300.0,
        "selfplay_s": 7.0,
        "train_s": 0.9,
        "snapshot_s": 0.05,
        "total_s": 8.0,
    }
    base.update(over)
    return logschema.iteration_row(**base)


def test_row_derives_the_residual_so_the_split_always_sums() -> None:
    """Erratum 6's ~35 min/iteration budget is only auditable if minutes add up.

    ``other_s`` is computed, never passed: a stage that under-reports itself shows
    up as unexplained time rather than disappearing into a rounding tolerance.
    """
    r = _row(selfplay_s=7.0, train_s=0.9, snapshot_s=0.05, total_s=8.0)
    assert r["other_s"] == pytest.approx(0.05)
    assert r["selfplay_s"] + r["train_s"] + r["snapshot_s"] + r["other_s"] == pytest.approx(
        r["total_s"]
    )


def test_validate_rejects_a_split_that_does_not_account_for_the_iteration() -> None:
    r = _row()
    r["train_s"] = 99.0
    with pytest.raises(ValueError, match="wall-clock split"):
        logschema.validate(r)


def test_validate_rejects_unknown_and_missing_fields() -> None:
    r = _row()
    r["experiment_note"] = "oops"
    with pytest.raises(ValueError, match="unknown"):
        logschema.validate(r)

    r = _row()
    del r["sims_per_s"]
    with pytest.raises(ValueError, match="missing"):
        logschema.validate(r)


def test_only_the_declared_fields_may_be_null() -> None:
    """None means "not measured"; a 0.0 in these slots would read as calibrated."""
    assert _row(false_resign_rate=None, mean_plies_decisive=None)  # both allowed
    with pytest.raises(ValueError, match="null"):
        logschema.validate(_row() | {"decisive_rate": None})


def test_read_jsonl_survives_a_torn_final_line(tmp_path: Path) -> None:
    """A SIGKILL mid-write must not cost the dashboard the whole log."""
    path = tmp_path / "iterations.jsonl"
    path.write_text(json.dumps(_row(iteration=1)) + "\n" + '{"iteration": 2, "gam')
    rows = logschema.read_jsonl(path)
    assert [r["iteration"] for r in rows] == [1]


def test_read_jsonl_still_raises_on_corruption_that_is_not_the_tail(tmp_path: Path) -> None:
    path = tmp_path / "iterations.jsonl"
    path.write_text("{not json\n" + json.dumps(_row(iteration=2)) + "\n")
    with pytest.raises(json.JSONDecodeError):
        logschema.read_jsonl(path)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def test_the_two_move1_entropies_measure_different_things() -> None:
    """The prior and the improved policy diverge by design, and by a lot.

    Measured on the GPU smoke: between two iterations the improved policy's
    entropy fell 2.909 → 0.226 while the prior went 2.972 → 2.778 of a 2.996
    maximum — because σ scales q̂ by (c_visit + max_N) ≈ 70 and the value head's
    child-value spread grew from 0.010 to 0.885. A single logged "entropy" would
    have reported a working value head as a collapsing policy.
    """
    from tinyalphazero.mcts import gumbel

    cfg = Config().mcts
    logits = np.log(np.full(20, 1 / 20, dtype=np.float32))
    flat_q = np.full(20, -0.01, dtype=np.float32)
    spread_q = np.linspace(-0.5, 0.4, 20).astype(np.float32)

    prior_h = gumbel.entropy(np.full(20, 1 / 20))
    flat_h = gumbel.entropy(gumbel.improved_policy(logits, flat_q, 20, cfg))
    spread_h = gumbel.entropy(gumbel.improved_policy(logits, spread_q, 20, cfg))

    assert prior_h == pytest.approx(np.log(20), abs=1e-6)
    assert flat_h > 2.9, flat_h
    assert spread_h < 0.5, spread_h


def test_iteration_seed_separates_run_iteration_and_stream() -> None:
    """(run, iteration, stream) must be an injective key, or two streams correlate."""
    keys = [(s, i, st) for s in (1, 2) for i in range(20) for st in (1, 2, 100, 101)]
    seeds = [iteration_seed(*k) for k in keys]
    assert len(set(seeds)) == len(seeds), "seed fan-out collided"
    assert seeds == [iteration_seed(*k) for k in keys], "not reproducible"


# ---------------------------------------------------------------------------
# League (§7: 80% latest-vs-latest, 20% latest-vs-pool)
# ---------------------------------------------------------------------------


def _pool(n: int, latest_frac: float = 0.8, size: int = 20) -> CheckpointPool:
    pool = CheckpointPool(SelfPlayConfig(league_latest_frac=latest_frac, league_pool_size=size))
    for i in range(n):
        pool.add(f"ckpt_iter{i:04d}.pt")
    return pool


def test_league_keeps_the_spec_s_20_percent_after_coalescing() -> None:
    """Coalescing trades opponent *diversity* for batch width, never the fraction.

    If it moved the count, it would silently retune the staleness of ~20% of the
    training data — the exact quantity `train_on_pool_moves` exists to ablate.
    """
    pool = _pool(20)
    rng = np.random.default_rng(0)
    n = 4000
    assignments = assign_league(pool, n, rng, max_opponents=4)
    league = [a for a in assignments if a.white != LATEST or a.black != LATEST]
    frac = len(league) / n
    # 20% by the pool's own draw, minus the 1/20 of pool draws that land on latest.
    assert 0.17 < frac < 0.21, frac


def test_league_spans_at_most_max_opponents_snapshots() -> None:
    pool = _pool(20)
    assignments = assign_league(pool, 500, np.random.default_rng(3), max_opponents=4)
    opponents = {k for a in assignments for k in (a.white, a.black)} - {LATEST}
    assert 1 <= len(opponents) <= 4, opponents


def test_league_gives_the_learner_both_colours() -> None:
    """A fixed colour for the learner would bake first-move advantage into 20% of the data."""
    pool = _pool(20)
    assignments = assign_league(pool, 800, np.random.default_rng(5), max_opponents=4)
    league = [a for a in assignments if a.white != LATEST or a.black != LATEST]
    as_white = sum(a.white == LATEST for a in league)
    assert 0.4 < as_white / len(league) < 0.6, as_white / len(league)


def test_league_is_inert_while_the_pool_holds_only_the_initial_weights() -> None:
    """Iteration 1's pool is {anchor} and latest *is* the anchor: no opponent exists."""
    assignments = assign_league(_pool(1), 64, np.random.default_rng(0), max_opponents=4)
    assert all(a == GameAssignment() for a in assignments)


def test_assignment_maps_colour_to_the_right_weights() -> None:
    a = GameAssignment(white=LATEST, black="ckpt_iter0003.pt")
    assert a.key_for(chess.WHITE) == LATEST
    assert a.key_for(chess.BLACK) == "ckpt_iter0003.pt"


def test_checkpoint_pool_has_exactly_one_implementation() -> None:
    """AGENTS.md §5: "if a component exists twice, that's a defect".

    v1 carried two copies of the league in two modules and they drifted; the spec's
    post-mortem lists it as defect 7. This is the grep that keeps chunk 9 from
    reintroducing it now that a second module wants opponent sampling.
    """
    defs = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-n", "^class CheckpointPool", "--", "src", "scripts"],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert len(defs) == 1, f"CheckpointPool defined {len(defs)} times: {defs}"
    assert defs[0].startswith("src/tinyalphazero/selfplay.py:")

    phase2_src = (REPO / "src" / "tinyalphazero" / "phase2.py").read_text()
    assert "CheckpointPool" in phase2_src
    assert "from tinyalphazero.selfplay import" in phase2_src


# ---------------------------------------------------------------------------
# Per-network leaf routing in the batched runner
# ---------------------------------------------------------------------------


def _tagged_stub(tag: float):
    """A stub evaluator that stamps its identity into the value it returns."""

    def evaluate(tokens: np.ndarray):
        policy, promo, _values = uniform_stub_batch(tokens)
        return policy, promo, np.full(tokens.shape[0], tag, dtype=np.float32)

    return evaluate


def test_default_path_still_issues_exactly_one_forward_per_flush() -> None:
    """The equivalence gate rests on the single-evaluator path being untouched."""
    runner = BatchRunner(uniform_stub_batch, Config().mcts, max_plies=2)
    list(runner.play_games(4, num_sims=8))
    single = runner.stats["forwards"]

    runner2 = BatchRunner(uniform_stub_batch, Config().mcts, max_plies=2)
    list(runner2.play_games(4, num_sims=8, evaluator_for=lambda _gi, _side: uniform_stub_batch))
    assert runner2.stats["forwards"] == single, "grouping changed the default forward count"
    assert runner2.stats["sims"] == runner.stats["sims"]


def test_leaves_are_routed_to_the_searching_side_s_network() -> None:
    """Each network must answer exactly the leaves its own search produced.

    The board is pushed to the *leaf* when a simulation yields, so reading
    ``board.turn`` at request time would route by the leaf's mover rather than the
    searcher's — a half-game-off error that no gate reads and that would train the
    learner on the opponent's search.
    """
    seen: dict[str, list[bool]] = {"white": [], "black": []}

    def spy(name: str, side: bool):
        def evaluate(tokens: np.ndarray):
            seen[name].append(side)
            return uniform_stub_batch(tokens)

        return evaluate

    white_net, black_net = spy("white", chess.WHITE), spy("black", chess.BLACK)
    runner = BatchRunner(uniform_stub_batch, Config().mcts, max_plies=6)
    list(
        runner.play_games(
            2,
            num_sims=8,
            evaluator_for=lambda _gi, side: white_net if side == chess.WHITE else black_net,
        )
    )
    assert seen["white"] and seen["black"], "one side's network never ran"
    assert runner.stats["forwards"] >= 2, "two networks cannot share one forward"


def test_two_networks_are_two_forwards_over_one_concurrent_pool() -> None:
    """League games stay in the same batch pool; only the forward is split."""
    a, b = _tagged_stub(0.5), _tagged_stub(-0.5)
    runner = BatchRunner(a, Config().mcts, max_plies=4)
    list(runner.play_games(4, num_sims=8, evaluator_for=lambda gi, _s: a if gi % 2 == 0 else b))
    # 4 games advance together; each flush answers both networks.
    assert runner.stats["leaves"] > 0
    assert runner.stats["forwards"] >= 2


# ---------------------------------------------------------------------------
# Replay: the mover tag, the filter, and the write-ordering contract
# ---------------------------------------------------------------------------


def _replay_row(i: int, mover_is_latest: bool = True) -> dict:
    return {
        "tokens": np.full(72, i % 43, dtype=np.int8),
        "legals": np.full(512, 255, dtype=np.uint8),
        "move_ft": (i * 7) % 4096,
        "promo": 0,
        "outcome": (i % 3) - 1,
        "improved_policy": [((i * 7) % 4096, 0, 1.0)],
        "iteration": i,
        "mover_is_latest": mover_is_latest,
    }


def test_mover_tag_round_trips_and_filters_the_sample(tmp_path: Path) -> None:
    """The ablation must be a filter over stored data, not a regenerated buffer."""
    buf = ReplayBuffer.create(tmp_path / "r", capacity=64)
    for i in range(20):
        buf.add(**_replay_row(i, mover_is_latest=i % 4 != 0))
    buf.flush()

    everything = buf.sample(200, np.random.default_rng(0))
    assert 0 in set(everything["mover_is_latest"].tolist()), "pool moves vanished from the buffer"

    latest_only = buf.sample(200, np.random.default_rng(0), latest_only=True)
    assert set(latest_only["mover_is_latest"].tolist()) == {1}


def test_latest_only_on_a_pool_only_buffer_raises_rather_than_returning_nothing(
    tmp_path: Path,
) -> None:
    buf = ReplayBuffer.create(tmp_path / "r", capacity=8)
    buf.add(**_replay_row(0, mover_is_latest=False))
    buf.flush()
    with pytest.raises(ValueError, match="no latest-weights positions"):
        buf.sample(4, np.random.default_rng(0), latest_only=True)


def test_the_durable_head_moves_only_on_flush(tmp_path: Path) -> None:
    """Write-ordering contract, clause 2: data lands before anything points at it.

    A process killed between ``add`` and ``flush`` must be indistinguishable from
    one that never wrote the rows — otherwise a torn iteration is not "lost", it is
    half-committed, and no reader can tell which rows are real.
    """
    buf = ReplayBuffer.create(tmp_path / "r", capacity=32)
    for i in range(5):
        buf.add(**_replay_row(i))
    buf.flush()

    for i in range(100, 105):
        buf.add(**_replay_row(i))  # written to the pages, never flushed

    reopened = ReplayBuffer.open(tmp_path / "r")
    assert len(reopened) == 5, "the head advanced without a flush"
    assert [int(reopened._arr["iteration"][i]) for i in range(5)] == [0, 1, 2, 3, 4]


def test_rewind_drops_a_flushed_but_uncommitted_iteration(tmp_path: Path) -> None:
    buf = ReplayBuffer.create(tmp_path / "r", capacity=32)
    for i in range(5):
        buf.add(**_replay_row(i))
    buf.flush()
    committed = buf.total_written

    for i in range(100, 108):  # a generation that flushed, then died before its snapshot
        buf.add(**_replay_row(i))
    buf.flush()
    assert len(buf) == 13

    buf.rewind_to(committed)
    assert len(buf) == 5
    assert ReplayBuffer.open(tmp_path / "r").total_written == committed


def test_rewind_refuses_to_move_the_head_forward(tmp_path: Path) -> None:
    buf = ReplayBuffer.create(tmp_path / "r", capacity=8)
    buf.add(**_replay_row(0))
    buf.flush()
    with pytest.raises(ValueError, match="ahead of"):
        buf.rewind_to(99)


def test_a_chunk_8_ring_is_rejected_rather_than_read_as_all_pool(tmp_path: Path) -> None:
    """Format drift across a multi-day run is worse than a crash (chunk 8's rule)."""
    buf = ReplayBuffer.create(tmp_path / "r", capacity=4)
    meta = json.loads((buf.path / "meta.json").read_text())
    meta["version"] = 1
    (buf.path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="version"):
        ReplayBuffer.open(tmp_path / "r")


# ---------------------------------------------------------------------------
# Run directory contract
# ---------------------------------------------------------------------------


def test_run_directory_is_self_describing_and_relocatable(tmp_path: Path) -> None:
    """The run directory *is* the state, so it must survive being moved.

    Absolute paths in state.json would have made a renamed run unresumable — which
    is exactly the kind of breakage a 24-hour campaign discovers at hour 19.
    """
    cfg = load_config(str(MINI))
    cfg.phase2.iterations = 1
    run = Phase2Run.create(tmp_path / "run", cfg, torch.device("cpu"))
    run.iterate()

    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["pool"] == ["ckpt_iter0000.pt", "ckpt_iter0001.pt"]
    assert all("/" not in p for p in state["pool"])

    moved = tmp_path / "renamed"
    (tmp_path / "run").rename(moved)
    resumed = Phase2Run.resume(moved, torch.device("cpu"))
    assert resumed.state.iteration == 1
    resumed.iterate()
    assert resumed.state.iteration == 2


def test_resume_drops_a_log_row_the_state_never_committed(tmp_path: Path) -> None:
    """The row is written a millisecond before the state commit; a kill fits there.

    Without truncation the re-run appends a *second* row for the same iteration,
    and `iterations.jsonl` stops being the sequence `state.json` asserts — which is
    precisely what the dashboard is entitled to assume about it.
    """
    cfg = load_config(str(MINI))
    run = Phase2Run.create(tmp_path / "run", cfg, torch.device("cpu"))
    run.iterate()

    log = tmp_path / "run" / "iterations.jsonl"
    with log.open("a") as fh:
        fh.write(json.dumps(_row(iteration=2)) + "\n")
    assert len(logschema.read_jsonl(log)) == 2

    resumed = Phase2Run.resume(tmp_path / "run", torch.device("cpu"))
    assert [r["iteration"] for r in logschema.read_jsonl(log)] == [1]
    resumed.iterate()
    assert [r["iteration"] for r in logschema.read_jsonl(log)] == [1, 2]


def test_iteration_zero_pool_is_the_initial_weights(tmp_path: Path) -> None:
    """§7's "iteration 0 pool = {anchor}" — copied in, so the run is self-contained."""
    cfg = load_config(str(MINI))
    run = Phase2Run.create(tmp_path / "run", cfg, torch.device("cpu"))
    assert run.pool.snapshots == ["ckpt_iter0000.pt"]
    assert (tmp_path / "run" / "ckpt_iter0000.pt").exists()
    _net, meta = load_checkpoint(tmp_path / "run" / "ckpt_iter0000.pt")
    assert meta["phase"] == "2-init"


# ---------------------------------------------------------------------------
# The golden mini-run (spec §9.4)
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_mini_run(tmp_path: Path) -> None:
    """Two iterations of the real loop on CPU, under three minutes.

    What it asserts is deliberately about *mechanism*, not strength: two iterations
    of RL say nothing about whether the agent is learning chess, but they say
    everything about whether the loss is finite and moving, whether search is still
    exploring, and whether the checkpoint that comes out is the one the config
    describes.
    """
    cfg = load_config(str(MINI))
    t0 = time.perf_counter()
    outcomes = run_phase2(tmp_path / "golden", cfg, torch.device("cpu"), iterations=2)
    elapsed = time.perf_counter() - t0
    assert elapsed < 180, f"golden run took {elapsed:.1f}s; the CI budget is 180s"

    # -- finite, window-averaged decreasing loss --
    losses = [x for oc in outcomes for x in oc.step_losses]
    assert len(losses) == 2 * cfg.train.train_steps_per_iter
    assert all(np.isfinite(losses)), "non-finite loss in the golden run"
    w = len(losses) // 4
    assert np.mean(losses[-w:]) < np.mean(losses[:w]), (
        f"loss did not fall: first window {np.mean(losses[:w]):.4f} "
        f"vs last {np.mean(losses[-w:]):.4f}"
    )

    # -- search is still exploring at move 1 (§8 collapse detector) --
    # Asserted on the prior, not the improved policy: at 48 sims σ multiplies q̂ by
    # ~70, so the improved policy's entropy is a reading on the value head and
    # legitimately collapses once it can separate two moves. Only the prior
    # answers "has the policy stopped considering alternatives".
    for oc in outcomes:
        assert oc.row["move1_prior_entropy"] > 0.5, oc.row["move1_prior_entropy"]
        assert oc.row["move1_policy_entropy"] > 0.0, oc.row["move1_policy_entropy"]

    # -- every row is a schema row, and the split adds up --
    rows = logschema.read_jsonl(tmp_path / "golden" / "iterations.jsonl")
    assert [r["iteration"] for r in rows] == [1, 2]
    for r in rows:
        logschema.validate(r)

    # -- the final checkpoint loads and describes this config --
    final = tmp_path / "golden" / "ckpt_iter0002.pt"
    net, meta = load_checkpoint(final, reference_model_cfg=cfg.model)
    assert config_fingerprint(meta["config_obj"]) == config_fingerprint(cfg)
    assert meta["phase"] == "2"
    with torch.no_grad():
        out = net(torch.zeros(1, 72, dtype=torch.long))
    assert torch.isfinite(out.policy_ft).all()


# ---------------------------------------------------------------------------
# Crash-resume, proven with a real SIGKILL (AGENTS.md §6)
# ---------------------------------------------------------------------------


def _crash_config(tmp_path: Path) -> Path:
    """Mini, but with a training stage long enough to be killed *inside*.

    Mini's 20 steps take ~0.4s, which is not a window — a kill aimed at training
    would land in the snapshot as often as not, and the test would be measuring
    scheduler luck rather than the write-ordering contract.
    """
    cfg = load_config(str(MINI))
    cfg.train.train_steps_per_iter = 300
    path = tmp_path / "crash.yaml"
    save_config(cfg, str(path))
    return path


def _launch(config: Path, out: Path, iterations: int = 3) -> subprocess.Popen:
    return subprocess.Popen(
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
            str(iterations),
        ],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _kill_at_stage(proc: subprocess.Popen, stage: str, iteration: int, timeout: float = 300) -> None:
    """SIGKILL the process the moment it announces it has entered ``stage``.

    SIGKILL, not SIGTERM: the point is the case with no cleanup, no atexit, no
    flush-on-the-way-out. A run that only survives a polite shutdown has not been
    shown to survive a power cut or an OOM kill.
    """
    marker = f"[stage] {stage} iteration={iteration}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            break
        if marker in line:
            time.sleep(0.8)  # be genuinely *inside* the stage, not at its doorstep
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
            return
    proc.kill()
    raise AssertionError(f"never reached stage marker {marker!r}")


@pytest.mark.slow
@pytest.mark.parametrize(("stage", "iteration"), [("selfplay", 1), ("train", 2)])
def test_sigkill_mid_iteration_then_resume(tmp_path: Path, stage: str, iteration: int) -> None:
    """Kill inside generation and inside training; both must resume and continue.

    The two points fail differently. Mid-generation, nothing was flushed, so resume
    must find the previous iteration intact and simply redo the work. Mid-training,
    generation *did* flush, so the durable replay head is ahead of the last
    committed iteration and resume must rewind it — otherwise the re-run stores
    those games a second time and ``replay_size`` stops meaning anything.
    """
    out = tmp_path / "run"
    proc = _launch(_crash_config(tmp_path), out, iterations=3)
    _kill_at_stage(proc, stage, iteration)
    assert proc.returncode is not None and proc.returncode != 0

    # -- the run directory survived the kill --
    state = json.loads((out / "state.json").read_text())
    assert state["iteration"] == iteration - 1, state
    buf = ReplayBuffer.open(out / "replay")
    assert 0 <= len(buf) <= buf.capacity
    committed = state["replay_total_written"]
    if stage == "train":
        assert buf.total_written > committed, "generation flushed; the head should be ahead"

    # -- and one further iteration completes from it --
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

    after = json.loads((out / "state.json").read_text())
    assert after["iteration"] == iteration
    assert (out / f"ckpt_iter{iteration:04d}.pt").exists()

    rows = logschema.read_jsonl(out / "iterations.jsonl")
    assert [r["iteration"] for r in rows] == list(range(1, iteration + 1)), rows
    for r in rows:
        logschema.validate(r)

    # The torn iteration's games were dropped, not counted twice: the ring holds
    # exactly the positions the committed rows claim to have produced.
    reopened = ReplayBuffer.open(out / "replay")
    assert reopened.total_written == sum(r["positions"] for r in rows)
    assert reopened.total_written == after["replay_total_written"]
    assert rows[-1]["replay_size"] == len(reopened)
