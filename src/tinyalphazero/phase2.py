"""Phase 2: the alternating self-play loop (spec §7, chunk 9).

One process, one GPU, three components in strict sequence per iteration:
generate → push traces to replay → train → snapshot → append to the league pool →
write one JSONL row. §7 chose alternating over asynchronous deliberately: async
buys staleness bookkeeping and nothing else at this scale.

**No gating.** The learner always generates from and trains on the latest weights.
v1's arena gate is deleted rather than fixed — at 4–100 games per iteration it
either does nothing or rejects real progress at random — and the anchor ladder
(§8, chunk 10) is the progress signal instead.

**Which search profile runs where, stated once, here, because this is where both
appear in the same process.** Self-play inside the loop runs the *self-play*
profile: ``mcts.root_noise=True``, paper σ constants, Gumbel noise at the root,
because its output is training data and its job is diversity. Every evaluation —
suites, the anchor ladder, match play — runs the *evaluation* profile
(``root_noise=False``, deterministic top-m, acting argmax on ``logits + σ(q̂)``).
The failure mode is silent in both directions: noise in a measurement reports
exploration as weakness, and determinism in generation produces a corpus with no
exploration in it. Chunk 10's ladder is a separate process reading checkpoints, so
nothing here ever needs to switch profiles mid-run.

**Resume is reconstruction, not restoration.** ``--resume`` rebuilds the iteration
counter, the pool, the replay ring, and the optimizer from the run directory; RNG
state is *derived* from (run seed, iteration, stream) rather than serialised, so a
resumed iteration draws exactly what an uninterrupted one would have. See
``replay.py``'s write-ordering contract for what a torn iteration costs.

    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python scripts/train_phase2.py \\
        --config configs/default.yaml --init runs/phase1_anchor.pt \\
        --out runs/p2_a --iterations 3
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyalphazero import logschema
from tinyalphazero.actors import play_specs_parallel
from tinyalphazero.config import Config, config_fingerprint, load_config, save_config
from tinyalphazero.envinfo import environment_report
from tinyalphazero.mcts.batch_runner import torch_evaluate_batch
from tinyalphazero.model import ChessNet
from tinyalphazero.provenance import git_dirty, git_sha
from tinyalphazero.rehearsal import RehearsalSampler
from tinyalphazero.replay import ReplayBuffer
from tinyalphazero.selfplay import (
    LATEST,
    CheckpointPool,
    GameAssignment,
    GameSpec,
    SelfPlayStats,
    assign_league,
    play_games,
    record_games,
    wave_specs,
)
from tinyalphazero.train import load_checkpoint, save_checkpoint, train_step_phase2

STATE_FILE = "state.json"
LOG_FILE = "iterations.jsonl"
CONFIG_FILE = "config.yaml"
PROVENANCE_FILE = "provenance.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_RUNS = REPO_ROOT / "runs"

# Independent RNG streams within one (run seed, iteration). Named so a future
# stream cannot silently collide with an existing one.
_STREAM_LEAGUE = 1
_STREAM_TRAIN = 2
_STREAM_INIT = 3  # fresh-weight initialisation at iteration 0 (REVIEW-2026-10-21 C5)
_STREAM_WAVE = 100  # + wave index


def iteration_seed(seed: int, iteration: int, stream: int) -> int:
    """Reproducible seed for one (run, iteration, stream).

    Composed with ``selfplay.game_seed``, this is spec §9.2's fan-out to the
    individual game: any game of a 24-hour run is replayable on its own from
    (run seed, iteration, wave, game index) without replaying anything before it.
    """
    return int(np.random.SeedSequence([seed, iteration, stream]).generate_state(1)[0])


def _copy_seed_pool(donor: Path, dest: Path, pool_size: int) -> list[str]:
    """Copy the donor run's league pool into ``dest``; return the new file names.

    Read from the donor's ``state.json`` rather than globbing its directory: the
    pool is what the donor's league actually held, and a glob would also pick up
    snapshots it had already evicted.

    Copied, not referenced, for the reason ``create`` copies the anchor — a run
    directory has to survive the donor being moved or deleted. Renamed with a
    ``seed_`` prefix because the donor's names are ``ckpt_iterNNNN.pt`` and so are
    this run's: at 20 seeded snapshots the two numberings overlap for any run
    longer than ``pool_size - 1`` iterations, and the collision would silently
    replace a league opponent with a checkpoint of the model playing it.
    """
    state_path = donor / STATE_FILE
    if not state_path.exists():
        raise FileNotFoundError(f"--seed-pool {donor} is not a run directory (no {STATE_FILE})")
    snapshots = list(json.loads(state_path.read_text()).get("pool", []))[-pool_size:]
    if not snapshots:
        raise ValueError(f"--seed-pool {donor} has an empty league pool; nothing to continue")

    names: list[str] = []
    for snap in snapshots:
        src = donor / snap
        if not src.exists():
            raise FileNotFoundError(
                f"{donor}'s pool names {snap} but the file is gone; the league "
                "cannot be continued from an incomplete donor"
            )
        name = f"seed_{donor.name}_{snap}"
        shutil.copy2(src, dest / name)
        names.append(name)
    return names


def _copy_seed_replay(donor: Path, buffer: ReplayBuffer) -> dict[str, Any]:
    """Continue the donor run's replay ring in ``buffer``; return what was carried.

    The committed head comes from the donor's ``state.json``, not its ring's
    ``meta.json``: the ring can hold a torn iteration past the commit (p2_e's held
    55,259 rows of an iteration that never happened), and ``seed_from`` carries only
    rows that are committed and intact. Copied rather than referenced, like the pool.
    """
    state_path = donor / STATE_FILE
    if not state_path.exists():
        raise FileNotFoundError(f"--seed-replay {donor} is not a run directory (no {STATE_FILE})")
    state = json.loads(state_path.read_text())
    committed = int(state["replay_total_written"])
    donor_iteration = int(state["iteration"])
    source = ReplayBuffer.open(donor / "replay", readonly=True)
    rows = buffer.seed_from(source, committed=committed, iteration_offset=-donor_iteration)
    buffer.flush()
    return {
        "rows": rows,
        "donor_iteration": donor_iteration,
        "donor_committed": committed,
        "donor_head": source.total_written,
        "iteration_offset": -donor_iteration,
    }


@dataclass
class RunState:
    """Everything ``--resume`` needs that is not already in a checkpoint."""

    iteration: int = 0
    step: int = 0
    skipped_steps_total: int = 0
    pool: list[str] = field(default_factory=list)
    replay_total_written: int = 0
    init_from: str = ""
    seed: int = 42
    # The SHA of the code this run was *launched* from, read once in ``create``
    # and stamped onto every row and checkpoint after that. Reading HEAD at write
    # time stamped whatever was checked out at the moment (REVIEW-2026-10-21 C2).
    # Empty on a state.json written before the field existed; ``resume`` fills it
    # from provenance.json.
    git_sha: str = ""
    # SHAs of the code that *resumed* this run, in order, when they differed from
    # the launch SHA. The rows still carry the launch SHA — the record of what
    # produced them is the one the reader needs — and this is where the fact
    # that something else continued the run lives.
    resumed_with: list[str] = field(default_factory=list)


@dataclass
class IterationOutcome:
    """One iteration's schema row, plus what only a test wants."""

    row: dict[str, Any]
    step_losses: list[float]
    stats: SelfPlayStats


class Phase2Run:
    """A run directory, opened for writing. Create with :meth:`create` or :meth:`resume`."""

    def __init__(
        self,
        path: Path,
        cfg: Config,
        net: ChessNet,
        optimizer: torch.optim.Optimizer,
        buffer: ReplayBuffer,
        state: RunState,
        device: torch.device,
    ) -> None:
        self.path = path
        self.cfg = cfg
        self.net = net
        self.optimizer = optimizer
        self.buffer = buffer
        self.state = state
        self.device = device
        self.use_bf16 = device.type == "cuda"
        self.pool = CheckpointPool(cfg.selfplay)
        for snap in state.pool:
            self.pool.add(snap)
        # Anchor rehearsal (p2_b). A no-op unless the rehearsal fractions are set,
        # and a pass-through wrapper when it is off, so runs that predate the lever
        # draw exactly the batches they drew before.
        self.rehearsal = RehearsalSampler(buffer, cfg.train, state.seed)
        self.on_stage: Any = None  # set by the CLI; see _stage()
        # Ladder cadence and rungs (spec §8), from the run's own config so that a
        # resumed run knows them — p2_a lost three protocol passes because they
        # lived only in a CLI flag (REVIEW-2026-10-21 C3). 0 disables the ladder;
        # configs/mini.yaml sets 0, and a test that wants a pass asks for one.
        self.ladder_every: int = cfg.eval.eval_every_n_iters
        self.ladder_anchors: str = cfg.eval.ladder_anchors
        self.ladder_anchor_ckpt: Path = REPO_ROOT / cfg.eval.anchor_ckpt
        self.on_ladder_event: Any = None  # the CLI's printer, same as on_stage

    # -- lifecycle -------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | Path,
        cfg: Config,
        device: torch.device,
        init_ckpt: str | Path | None = None,
        seed: int | None = None,
        seed_pool: str | Path | None = None,
        seed_replay: str | Path | None = None,
    ) -> Phase2Run:
        """Start a run: iteration 0 is the initial weights, and the pool is {them}.

        ``seed_pool`` continues another run's league instead of starting from a
        one-member pool. A warm-started run whose pool resets would spend its first
        20 iterations playing an opponent distribution that no longer resembles the
        weights it inherited — a confound, and a large one, in a run whose whole
        purpose is attributing a change to one lever. Continuity is not a lever.

        ``seed_replay`` continues another run's replay ring the same way, so the
        first iterations do not fit a near-empty ring many times over. Unlike the
        pool, it is being *tested* as a lever (PREREG-p2_f), because p2_b..p2_e all
        ran without it and their mid-round dips sit exactly in the fill phase.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # The effective seed goes *into* the config before anything is saved, so
        # config.yaml, provenance.json and every checkpoint's embedded config say
        # the seed the run actually used — not the YAML's value with the override
        # living only in a gitignored state.json (REVIEW-2026-10-21 C5/E3).
        seed = cfg.seed if seed is None else seed
        cfg = replace(cfg, seed=seed)
        sha = git_sha()

        # Fresh weights are drawn from a stream of the run seed, so a run without
        # --init (the golden mini-run) is reproducible from its config alone
        # (spec §9.2). An --init checkpoint overwrites them below.
        torch.manual_seed(iteration_seed(seed, 0, _STREAM_INIT))
        net = ChessNet(cfg.model).to(device)
        if init_ckpt is not None:
            # Validate shape against *our* config for the readable error, then load
            # the weights into a model built from our config — so Phase 2's dropout
            # and head selection come from the campaign config, not from whatever
            # the donor checkpoint was trained with.
            donor, _meta = load_checkpoint(init_ckpt, device=device, reference_model_cfg=cfg.model)
            net.load_state_dict(donor.state_dict())
            del donor

        optimizer = torch.optim.AdamW(
            net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        buffer = ReplayBuffer.open_or_create(path / "replay", cfg.train.replay_capacity)
        seeded = _copy_seed_replay(Path(seed_replay), buffer) if seed_replay is not None else None
        state = RunState(
            iteration=0,
            replay_total_written=buffer.total_written,
            init_from=str(init_ckpt or ""),
            seed=seed,
            git_sha=sha,
        )

        save_config(cfg, str(path / CONFIG_FILE))
        (path / PROVENANCE_FILE).write_text(
            json.dumps(
                {
                    "git_sha": sha,
                    "git_dirty": git_dirty(),
                    "config_fingerprint": config_fingerprint(cfg),
                    "seed": seed,
                    "torch": torch.__version__,
                    "device": str(device),
                    "device_name": (
                        torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
                    ),
                    "init_from": state.init_from,
                    "seed_pool_from": str(seed_pool or ""),
                    "seed_replay_from": str(seed_replay or ""),
                    "seed_replay": seeded,
                    # The check_env report AGENTS §4 asks every run dir to carry;
                    # the matmul probe only where it is cheap (REVIEW-2026-10-21 E3).
                    "environment": environment_report(matmul=device.type == "cuda"),
                },
                indent=2,
            )
            + "\n"
        )

        run = cls(path, cfg, net, optimizer, buffer, state, device)
        # Seeded opponents go in *before* iteration 0's own snapshot, so the pool
        # ends up holding the donor's most recent (pool_size - 1) plus this run's
        # initial weights — exactly the pool the donor would have had after one
        # more iteration.
        if seed_pool is not None:
            for name in _copy_seed_pool(Path(seed_pool), path, cfg.selfplay.league_pool_size):
                run.pool.add(name)
        # Iteration 0 is a real, resumable state, and its checkpoint is the pool's
        # first member (§7: "iteration 0 pool = {anchor}"). Copying the anchor into
        # the run directory rather than pointing at it keeps the run self-contained
        # — a 24-hour campaign should not break because a path outside it moved.
        # The pool holds *file names*, not paths: a run directory that is renamed
        # or moved between a crash and its resume must still find its own
        # snapshots, and absolute paths baked into state.json would not.
        run.pool.add(run._snapshot(0, phase="2-init").name)
        run.state.pool = list(run.pool.snapshots)
        run._write_readme()
        run._commit_state()
        return run

    @classmethod
    def resume(cls, path: str | Path, device: torch.device) -> Phase2Run:
        """Reopen a run at its last committed iteration."""
        path = Path(path)
        state = RunState(**json.loads((path / STATE_FILE).read_text()))
        cfg = load_config(str(path / CONFIG_FILE))
        if not state.git_sha:
            # A run created before the field existed: the launch SHA was written
            # to provenance.json and nowhere else.
            prov_path = path / PROVENANCE_FILE
            if prov_path.exists():
                state.git_sha = str(json.loads(prov_path.read_text()).get("git_sha", ""))
        now = git_sha()
        if now != state.git_sha and (not state.resumed_with or state.resumed_with[-1] != now):
            state.resumed_with.append(now)

        ckpt = path / f"ckpt_iter{state.iteration:04d}.pt"
        net, meta = load_checkpoint(ckpt, device=device, reference_model_cfg=cfg.model)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        if "optimizer" in meta:
            optimizer.load_state_dict(meta["optimizer"])

        cls._truncate_log(path, state.iteration)
        buffer = ReplayBuffer.open(path / "replay")
        # The durable head can be *ahead* of the last committed iteration: a kill
        # between generation's flush and the snapshot leaves games belonging to an
        # iteration that never happened. Dropping them is what makes "a torn
        # iteration loses at most its own partial games" true rather than
        # aspirational — otherwise the re-run stores them a second time.
        buffer.rewind_to(state.replay_total_written)

        return cls(path, cfg, net, optimizer, buffer, state, device)

    @staticmethod
    def _truncate_log(path: Path, iteration: int) -> None:
        """Drop log rows for iterations ``state.json`` does not claim happened.

        The row is written just before the state commit, so a kill in that
        millisecond leaves a row for an iteration that will now be re-run — and
        the log would end up describing it twice. Truncating on resume keeps
        ``iterations.jsonl`` exactly the sequence ``state.json`` asserts, which is
        the property chunk 11's dashboard is entitled to assume. It also rewrites
        away a torn final line.
        """
        log = path / LOG_FILE
        if not log.exists():
            return
        rows = [r for r in logschema.read_jsonl(log) if r["iteration"] <= iteration]
        # Rewritten whole, so written to a temp file and renamed: a kill inside a
        # plain write_text leaves a half-written log and the next resume cannot
        # read it (REVIEW-2026-10-21 C9) — the same contract as _commit_state.
        tmp = log.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        os.replace(tmp, log)

    @classmethod
    def open_or_create(
        cls,
        path: str | Path,
        cfg: Config,
        device: torch.device,
        init_ckpt: str | Path | None = None,
        seed: int | None = None,
        seed_pool: str | Path | None = None,
        seed_replay: str | Path | None = None,
    ) -> Phase2Run:
        if (Path(path) / STATE_FILE).exists():
            return cls.resume(path, device)
        return cls.create(
            path,
            cfg,
            device,
            init_ckpt=init_ckpt,
            seed=seed,
            seed_pool=seed_pool,
            seed_replay=seed_replay,
        )

    # -- one iteration ---------------------------------------------------

    def iterate(self) -> IterationOutcome:
        """Generate, train, snapshot, log, commit. Returns the row it wrote."""
        i = self.state.iteration + 1
        t0 = time.perf_counter()

        self._stage("selfplay", i)
        stats = self._generate(i)
        t_gen = time.perf_counter()

        self._stage("train", i)
        losses, step_losses, skipped = self._train(i)
        t_train = time.perf_counter()

        self._stage("snapshot", i)
        self.pool.add(self._snapshot(i).name)
        t_snap = time.perf_counter()

        row = self._row(i, stats, losses, skipped, t0, t_gen, t_train, t_snap)
        logschema.append_row(self.path / LOG_FILE, row)

        self.state.iteration = i
        self.state.pool = list(self.pool.snapshots)
        self.state.replay_total_written = self.buffer.total_written
        self._commit_state()

        # Ladder *after* the commit, never before: a pass that dies takes nothing
        # with it, and its own log is separately resumable. An eval folded into
        # the iteration's critical path would make a 45-minute measurement able to
        # lose 27 minutes of generation.
        if self.ladder_every and i % self.ladder_every == 0:
            self._stage("ladder", i)
            self._run_ladder(i)

        return IterationOutcome(row=row, step_losses=step_losses, stats=stats)

    def _run_ladder(self, iteration: int) -> None:
        """Run one anchor-ladder pass inline, sequentially (spec §8, chunk 10).

        Sequential on purpose: the ladder searches on the same GPU the loop does,
        and two processes sharing it would make both timings meaningless — the
        wall-clock split in the iteration row is only auditable if nothing else is
        running.

        **Budget, measured (2026-08-10), not estimated.** The pre-registered guess
        was 30–45 min per pass; the dominant rung — 200 games against the frozen
        anchor, both sides searching, at the 300-ply ladder cap — came in at
        **8.4 min**. Adding the cheaper rungs at their measured rates gives a full
        pass of **≈19 min**, and that is an upper bound because 200 concurrent
        games batch better than the 20 the small rungs were measured on. Against a
        ≈27 min iteration, ``--ladder-every 10`` therefore costs **≈7%** of
        campaign wall clock. The 300-ply cap did not make the ladder expensive:
        games average ~102 plies, so the cap is a safety net, not a cost.
        """
        from tinyalphazero.evaluate import run_ladder_pass

        run_ladder_pass(
            self.path / f"ckpt_iter{iteration:04d}.pt",
            self.cfg,
            self.device,
            out=self.path / "ladder.jsonl",
            anchors=self.ladder_anchors,
            anchor_ckpt=self.ladder_anchor_ckpt,
            iteration=iteration,
            seed=self.state.seed,
            role=logschema.ROLE_PROTOCOL,
            produced=logschema.PRODUCED_INLINE,
            on_event=self.on_ladder_event,
            git_sha=self.state.git_sha,
        )

    # -- stages ----------------------------------------------------------

    def _generate(self, iteration: int) -> SelfPlayStats:
        """Play one iteration's games, in waves of ``selfplay.concurrent_games``."""
        sp = self.cfg.selfplay
        total = sp.games_per_iteration
        wave_size = max(1, min(sp.concurrent_games, total))

        league_rng = np.random.default_rng(
            iteration_seed(self.state.seed, iteration, _STREAM_LEAGUE)
        )
        assignments = assign_league(
            self.pool, total, league_rng, self.cfg.phase2.league_opponents_per_iter
        )
        evaluators = self._build_evaluators(assignments)

        if sp.actor_processes > 1:
            return self._generate_parallel(iteration, assignments, evaluators, wave_size)

        merged = SelfPlayStats()
        for w, start in enumerate(range(0, total, wave_size)):
            n = min(wave_size, total - start)
            wave_assignments = assignments[start : start + n]
            _traces, stats = play_games(
                evaluators[LATEST],
                n,
                cfg=self.cfg,
                # A distinct seed per wave, so wave 2 does not inherit wave 1's
                # resign-disabled pattern; (iteration, wave, index) still names a
                # unique game.
                seed=iteration_seed(self.state.seed, iteration, _STREAM_WAVE + w),
                num_sims=sp.sims_per_move,
                device=str(self.device),
                iteration=iteration,
                buffer=self.buffer,
                evaluators=evaluators,
                assignments=wave_assignments,
            )
            merged.absorb(stats)
        return merged

    def _generate_parallel(
        self,
        iteration: int,
        assignments: list[GameAssignment],
        evaluators: dict[str, Any],
        wave_size: int,
    ) -> SelfPlayStats:
        """The same games as the wave loop above, played by ``actor_processes`` actors.

        Specs are drawn wave by wave with the wave loop's seeds, so each game is the
        game the in-process path would have played; all of them then play at once,
        dealt across the actors, and are stored in the wave loop's order.
        """
        sp = self.cfg.selfplay
        specs: list[GameSpec] = []
        for w, start in enumerate(range(0, sp.games_per_iteration, wave_size)):
            n = min(wave_size, sp.games_per_iteration - start)
            seed = iteration_seed(self.state.seed, iteration, _STREAM_WAVE + w)
            specs += wave_specs(seed, n, sp, assignments[start : start + n])

        stats = SelfPlayStats()
        t0 = time.perf_counter()
        results, runner_stats = play_specs_parallel(
            evaluators, specs, self.cfg, sp.sims_per_move, processes=sp.actor_processes
        )
        record_games(results, specs, stats, self.buffer, iteration)
        self.buffer.flush()
        stats.seconds = time.perf_counter() - t0
        stats.sims = runner_stats["sims"]
        stats.forwards = runner_stats["forwards"]
        return stats

    def _build_evaluators(self, assignments: list[GameAssignment]) -> dict[str, Any]:
        """One ``evaluate_batch`` per distinct opponent this iteration names.

        Loaded once per iteration and dropped at the end of it: the pool holds 20
        snapshots but an iteration names at most ``league_opponents_per_iter`` of
        them, so at most a handful of 26 MB models are resident at a time.
        """
        self.net.eval()
        evaluators: dict[str, Any] = {LATEST: torch_evaluate_batch(self.net, self.device)}
        keys = {a.white for a in assignments} | {a.black for a in assignments}
        for key in sorted(keys - {LATEST}):
            opponent, _meta = load_checkpoint(self.path / key, device=self.device)
            opponent.eval()
            evaluators[key] = torch_evaluate_batch(opponent, self.device)
        return evaluators

    def _train(self, iteration: int) -> tuple[dict[str, float], list[float], int]:
        cfg_t = self.cfg.train
        rng = np.random.default_rng(iteration_seed(self.state.seed, iteration, _STREAM_TRAIN))
        self.net.train()

        keys = ("loss", "policy_loss", "promo_loss", "value_loss", "legality_loss", "grad_norm")
        acc: dict[str, list[float]] = {k: [] for k in keys}
        step_losses: list[float] = []
        skipped_here = 0

        for step in range(cfg_t.train_steps_per_iter):
            batch = self.rehearsal.sample(
                cfg_t.batch_size,
                rng,
                iteration=iteration,
                step=step,
                latest_only=not self.cfg.phase2.train_on_pool_moves,
            )
            out = train_step_phase2(
                batch,
                self.net,
                self.optimizer,
                self.device,
                cfg_t,
                use_bf16=self.use_bf16,
                masked_policy=self.cfg.phase2.masked_policy,
            )
            self.state.step += 1
            if out["skipped"]:
                skipped_here += 1
                self.state.skipped_steps_total += 1
                self._assert_skip_rate()
                continue
            for k in keys:
                acc[k].append(out[k])
            step_losses.append(out["loss"])

        means = {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}
        return means, step_losses, skipped_here

    def _assert_skip_rate(self) -> None:
        """Phase 1's abort rule, carried over verbatim (see ``train._apply_update``)."""
        p2 = self.cfg.phase2
        total, step = self.state.skipped_steps_total, self.state.step
        if total > p2.nan_abort_min_steps and total > p2.nan_abort_frac * step:
            raise RuntimeError(
                f"Aborting: {total} non-finite gradients in {step} steps "
                f"({100 * total / step:.2f}%). Investigate before continuing — see the "
                f"ROCm SDPA note in PROGRESS.md; the first thing to try is dropping "
                f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL and rerunning once."
            )

    def _snapshot(self, iteration: int, phase: str = "2") -> Path:
        path = self.path / f"ckpt_iter{iteration:04d}.pt"
        save_checkpoint(
            path,
            self.net,
            self.optimizer,
            self.cfg,
            step=self.state.step,
            phase=phase,
            git_sha=self.state.git_sha,
        )
        return path

    # -- bookkeeping -----------------------------------------------------

    def _row(
        self,
        iteration: int,
        stats: SelfPlayStats,
        losses: dict[str, float],
        skipped: int,
        t0: float,
        t_gen: float,
        t_train: float,
        t_snap: float,
    ) -> dict[str, Any]:
        res = stats.results
        n = max(1, stats.games)
        decisive = res.get("checkmate", 0) + res.get("resign", 0)
        return logschema.iteration_row(
            iteration=iteration,
            git_sha=self.state.git_sha,
            games=stats.games,
            positions=stats.positions,
            mean_plies=round(stats.mean_plies, 2),
            mean_plies_decisive=(
                None if stats.mean_plies_decisive is None else round(stats.mean_plies_decisive, 2)
            ),
            decisive_rate=decisive / n,
            draw_rate=(n - decisive) / n,
            draw_rate_ply_cap=res.get("ply_cap", 0) / n,
            draw_rate_draw_claim=res.get("draw_claim", 0) / n,
            draw_rate_auto=res.get("draw", 0) / n,
            resign_rate=res.get("resign", 0) / n,
            false_resign_rate=stats.false_resign_rate,
            move1_policy_entropy=round(stats.move1_policy_entropy, 4),
            move1_prior_entropy=round(stats.move1_prior_entropy, 4),
            pool_moves_frac=(stats.pool_positions / stats.positions if stats.positions else 0.0),
            train_steps=self.cfg.train.train_steps_per_iter,
            skipped_steps=skipped,
            loss=losses["loss"],
            policy_loss=losses["policy_loss"],
            promo_loss=losses["promo_loss"],
            value_loss=losses["value_loss"],
            legality_loss=losses["legality_loss"],
            grad_norm=losses["grad_norm"],
            lr=self.optimizer.param_groups[0]["lr"],
            replay_size=len(self.buffer),
            pool_size=len(self.pool.snapshots),
            sims=stats.sims,
            sims_per_s=round(stats.sims_per_s, 1),
            selfplay_s=round(t_gen - t0, 3),
            train_s=round(t_train - t_gen, 3),
            snapshot_s=round(t_snap - t_train, 3),
            total_s=round(time.perf_counter() - t0, 3),
        )

    def _commit_state(self) -> None:
        """Write ``state.json`` last, and atomically.

        Ordering is the whole contract: replay is flushed, then the checkpoint is
        written, then the log row, and only then does ``state.json`` say the
        iteration happened. A crash at any earlier point leaves a run that resumes
        at the previous iteration and redoes work — never one that skips an
        iteration or half-commits it.
        """
        tmp = self.path / (STATE_FILE + ".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=2) + "\n")
        os.replace(tmp, self.path / STATE_FILE)

    def _write_readme(self) -> None:
        cfg = self.cfg
        per_ckpt_mb = 78
        per_day = 50
        (self.path / "README.md").write_text(
            f"""# Phase 2 run: {self.path.name}

Written by `scripts/train_phase2.py` (chunk 9). Regenerate nothing by hand — the
run directory is the state, and `--resume {self.path}` reconstructs from it.

| File | What it is |
|------|------------|
| `config.yaml` | resolved config, fingerprint `{config_fingerprint(cfg)[:12]}` |
| `provenance.json` | git SHA, torch/device, init checkpoint |
| `state.json` | iteration counter, pool, replay head — written last, atomically |
| `iterations.jsonl` | one `logschema.IterationRow` per iteration |
| `ckpt_iter####.pt` | snapshot per iteration, keep-all; `0000` is the initial weights |
| `replay/` | memmap ring, capacity {cfg.train.replay_capacity:,} positions |

**Disk.** Snapshots are kept, not rotated: ~{per_ckpt_mb} MB each (weights +
AdamW moments) at ~{per_day} iterations/day is ~{per_ckpt_mb * per_day / 1000:.1f} GB
per day. The replay ring is fixed-size and allocated up front
(~{cfg.train.replay_capacity * 1300 / 1e9:.1f} GB at capacity). A 24-hour M3
campaign therefore needs roughly
{per_ckpt_mb * per_day / 1000 + cfg.train.replay_capacity * 1300 / 1e9:.0f} GB free;
budget for it before starting, because running out mid-run costs the campaign, not
the iteration.
"""
        )

    def _stage(self, name: str, iteration: int) -> None:
        if self.on_stage is not None:
            self.on_stage(name, iteration)


def run_phase2(
    out: str | Path,
    cfg: Config,
    device: torch.device,
    iterations: int,
    init_ckpt: str | Path | None = None,
    seed: int | None = None,
    seed_pool: str | Path | None = None,
    seed_replay: str | Path | None = None,
    resume: bool = False,
    on_stage: Any = None,
    ladder_every: int | None = None,
    ladder_anchors: str | None = None,
    ladder_anchor_ckpt: str | Path | None = None,
    on_ladder_event: Any = None,
) -> list[IterationOutcome]:
    """Run (or continue) ``iterations`` iterations and return their outcomes.

    The ladder protocol lives in ``cfg.eval``. For a *new* run the three
    ``ladder_*`` arguments, when given, are written into the config before it is
    saved, so the run directory records the cadence and rungs it actually used
    and ``--resume`` rebuilds them (REVIEW-2026-10-21 C3). For a resumed run they
    apply to this invocation only and the saved config is left as the record of
    the launch; the CLI says so when it passes them.
    """
    if not resume:
        ev = cfg.eval
        if ladder_every is not None:
            ev = replace(ev, eval_every_n_iters=ladder_every)
        if ladder_anchors is not None:
            ev = replace(ev, ladder_anchors=ladder_anchors)
        if ladder_anchor_ckpt is not None:
            ev = replace(ev, anchor_ckpt=_repo_relative(ladder_anchor_ckpt))
        cfg = replace(cfg, eval=ev)
    run = (
        Phase2Run.resume(out, device)
        if resume
        else Phase2Run.open_or_create(
            out,
            cfg,
            device,
            init_ckpt=init_ckpt,
            seed=seed,
            seed_pool=seed_pool,
            seed_replay=seed_replay,
        )
    )
    run.on_stage = on_stage
    run.on_ladder_event = on_ladder_event
    if ladder_every is not None:
        run.ladder_every = ladder_every
    if ladder_anchors is not None:
        run.ladder_anchors = ladder_anchors
    if ladder_anchor_ckpt is not None:
        run.ladder_anchor_ckpt = Path(ladder_anchor_ckpt)
    # A pass killed mid-way left state.iteration at a ladder iteration with the
    # pass incomplete, and the next iterate() moved on without it (REVIEW-2026-10-21
    # C9). The pass skips every matchup already in its log, so re-running it here
    # costs only what the kill lost.
    s = run.state.iteration
    if resume and run.ladder_every and s > 0 and s % run.ladder_every == 0:
        run._stage("ladder", s)
        run._run_ladder(s)
    return [run.iterate() for _ in range(iterations)]


def _repo_relative(path: str | Path) -> str:
    """``eval.anchor_ckpt`` is stored relative to the repo when the path is inside it."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)
