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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyalphazero import logschema
from tinyalphazero.config import Config, config_fingerprint, load_config, save_config
from tinyalphazero.mcts.batch_runner import torch_evaluate_batch
from tinyalphazero.model import ChessNet
from tinyalphazero.replay import ReplayBuffer
from tinyalphazero.selfplay import (
    LATEST,
    CheckpointPool,
    GameAssignment,
    SelfPlayStats,
    assign_league,
    play_games,
)
from tinyalphazero.train import _git_sha, load_checkpoint, save_checkpoint, train_step_phase2

STATE_FILE = "state.json"
LOG_FILE = "iterations.jsonl"
CONFIG_FILE = "config.yaml"

# Independent RNG streams within one (run seed, iteration). Named so a future
# stream cannot silently collide with an existing one.
_STREAM_LEAGUE = 1
_STREAM_TRAIN = 2
_STREAM_WAVE = 100  # + wave index


def iteration_seed(seed: int, iteration: int, stream: int) -> int:
    """Reproducible seed for one (run, iteration, stream).

    Composed with ``selfplay.game_seed``, this is spec §9.2's fan-out to the
    individual game: any game of a 24-hour run is replayable on its own from
    (run seed, iteration, wave, game index) without replaying anything before it.
    """
    return int(np.random.SeedSequence([seed, iteration, stream]).generate_state(1)[0])


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
        self.on_stage: Any = None  # set by the CLI; see _stage()

    # -- lifecycle -------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | Path,
        cfg: Config,
        device: torch.device,
        init_ckpt: str | Path | None = None,
        seed: int | None = None,
    ) -> Phase2Run:
        """Start a run: iteration 0 is the initial weights, and the pool is {them}."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        seed = cfg.seed if seed is None else seed

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
        state = RunState(
            iteration=0,
            replay_total_written=buffer.total_written,
            init_from=str(init_ckpt or ""),
            seed=seed,
        )

        save_config(cfg, str(path / CONFIG_FILE))
        (path / "provenance.json").write_text(
            json.dumps(
                {
                    "git_sha": _git_sha(),
                    "config_fingerprint": config_fingerprint(cfg),
                    "torch": torch.__version__,
                    "device": str(device),
                    "device_name": (
                        torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
                    ),
                    "init_from": state.init_from,
                },
                indent=2,
            )
            + "\n"
        )

        run = cls(path, cfg, net, optimizer, buffer, state, device)
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
        log.write_text("".join(json.dumps(r) + "\n" for r in rows))

    @classmethod
    def open_or_create(
        cls,
        path: str | Path,
        cfg: Config,
        device: torch.device,
        init_ckpt: str | Path | None = None,
        seed: int | None = None,
    ) -> Phase2Run:
        if (Path(path) / STATE_FILE).exists():
            return cls.resume(path, device)
        return cls.create(path, cfg, device, init_ckpt=init_ckpt, seed=seed)

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

        return IterationOutcome(row=row, step_losses=step_losses, stats=stats)

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

        for _ in range(cfg_t.train_steps_per_iter):
            batch = self.buffer.sample(
                cfg_t.batch_size, rng, latest_only=not self.cfg.phase2.train_on_pool_moves
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
        save_checkpoint(path, self.net, self.optimizer, self.cfg, step=self.state.step, phase=phase)
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
            git_sha=_git_sha(),
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
    resume: bool = False,
    on_stage: Any = None,
) -> list[IterationOutcome]:
    """Run (or continue) ``iterations`` iterations and return their outcomes."""
    run = (
        Phase2Run.resume(out, device)
        if resume
        else Phase2Run.open_or_create(out, cfg, device, init_ckpt=init_ckpt, seed=seed)
    )
    run.on_stage = on_stage
    return [run.iterate() for _ in range(iterations)]
