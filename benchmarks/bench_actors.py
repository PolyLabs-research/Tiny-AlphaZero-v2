#!/usr/bin/env python3
"""Self-play throughput, in-process vs N actor processes (``tinyalphazero.actors``).

    gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \\
        .venv/bin/python benchmarks/bench_actors.py --ckpt runs/p2_f/ckpt_iter0020.pt \\
        --opponent runs/p2_f/ckpt_iter0015.pt --processes 0,2,4,8

Every configuration plays the *same* specs (``wave_specs``, fixed seeds), with one
game in five a league game, so the rows differ only in how the games are played:
``0`` is the in-process path (``play_games`` wave by wave, as ``Phase2Run`` has
always run it), ``N`` is ``play_specs_parallel`` with N actors. ``--max-plies``
truncates games so a sweep fits in minutes; per-simulation cost is what is being
compared, and truncation changes it for every row alike.

Each row records sims/s, forwards (and so the mean leaves per forward), mean GPU
busy (amdgpu's ``gpu_busy_percent`` or ``nvidia-smi``, sampled every 0.5 s), mean
host CPU busy (``/proc/stat``) and the cores this cgroup actually used — inside a
container the host figure counts every tenant — so a plateau can be attributed:
GPU-bound, CPU-bound, or neither. ``--note`` says where the row was measured.
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import threading
import time
from pathlib import Path

import torch

from tinyalphazero.actors import play_specs_parallel
from tinyalphazero.config import load_config
from tinyalphazero.envinfo import append_benchmark_row, benchmark_row
from tinyalphazero.mcts.batch_runner import torch_evaluate_batch
from tinyalphazero.selfplay import (
    LATEST,
    GameAssignment,
    SelfPlayStats,
    play_games,
    record_games,
    wave_specs,
)
from tinyalphazero.train import load_checkpoint


class _Sampler:
    """Mean GPU busy % over a window: amdgpu's sysfs counter, or ``nvidia-smi``."""

    def __init__(self) -> None:
        paths = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
        self.path = Path(paths[0]) if paths else None
        self.nvidia = self.path is None and shutil.which("nvidia-smi") is not None
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read(self) -> int | None:
        if self.path is not None:
            return int(self.path.read_text())
        if self.nvidia:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            return (
                int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else None
            )
        return None

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            value = self._read()
            if value is not None:
                self.samples.append(value)

    def __enter__(self) -> _Sampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def mean(self) -> float | None:
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else None


def _cpu_times() -> tuple[int, int]:
    """(busy, total) jiffies summed over all CPUs."""
    fields = [int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
    idle = fields[3] + fields[4]  # idle + iowait
    return sum(fields) - idle, sum(fields)


def _cgroup_cpu_s() -> float | None:
    """CPU seconds used by this cgroup so far (v2 or v1), or None.

    Inside a container ``/proc/stat`` counts the whole host — a RunPod pod reported
    96 CPUs against a 7.65-CPU quota — so cores actually used come from the cgroup.
    """
    v2 = Path("/sys/fs/cgroup/cpu.stat")
    if v2.exists():
        for line in v2.read_text().splitlines():
            if line.startswith("usage_usec "):
                return int(line.split()[1]) / 1e6
    v1 = Path("/sys/fs/cgroup/cpuacct/cpuacct.usage")
    if v1.exists():
        return int(v1.read_text()) / 1e9
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--opponent", type=Path, default=None, help="League opponent for 1 game in 5")
    ap.add_argument("--config", default="configs/p2_f.yaml")
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--max-plies", type=int, default=40)
    ap.add_argument("--processes", default="0,2,4,8", help="0 = in-process")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results.jsonl"))
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--note", default="", help="where this ran, e.g. the pod and its quota")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sp = cfg.selfplay
    sp.games_per_iteration, sp.sims_per_move, sp.max_plies = args.games, args.sims, args.max_plies
    device = torch.device(args.device)
    net, _ = load_checkpoint(args.ckpt, device=device)
    net.eval()
    evaluators = {LATEST: torch_evaluate_batch(net, device)}
    assignments = [GameAssignment()] * args.games
    if args.opponent is not None:
        opp, _ = load_checkpoint(args.opponent, device=device)
        opp.eval()
        evaluators["opponent"] = torch_evaluate_batch(opp, device)
        assignments = [
            GameAssignment()
            if i % 5
            else GameAssignment(white=LATEST, black="opponent")
            if (i // 5) % 2
            else GameAssignment(white="opponent", black=LATEST)
            for i in range(args.games)
        ]

    wave = sp.concurrent_games
    waves = [(w, s, min(wave, args.games - s)) for w, s in enumerate(range(0, args.games, wave))]
    specs = [sp_ for w, s, n in waves for sp_ in wave_specs(500 + w, n, sp, assignments[s : s + n])]

    base = None
    for processes in [int(p) for p in args.processes.split(",")]:
        stats = SelfPlayStats()
        busy0, total0 = _cpu_times()
        cg0 = _cgroup_cpu_s()
        t0 = time.perf_counter()
        with _Sampler() as gpu:
            if processes == 0:
                for w, s, n in waves:
                    _t, st = play_games(
                        evaluators[LATEST],
                        n,
                        cfg=cfg,
                        seed=500 + w,
                        num_sims=args.sims,
                        device=str(device),
                        evaluators=evaluators,
                        assignments=assignments[s : s + n],
                    )
                    stats.absorb(st)
            else:
                results, rs = play_specs_parallel(
                    evaluators, specs, cfg, args.sims, processes=processes
                )
                record_games(results, specs, stats, None, iteration=0)
                stats.sims, stats.forwards = rs["sims"], rs["forwards"]
        elapsed = time.perf_counter() - t0
        busy1, total1 = _cpu_times()
        cg1 = _cgroup_cpu_s()
        sims_per_s = stats.sims / elapsed
        base = base or sims_per_s
        row = benchmark_row(
            "selfplay_actors",
            device=str(device),
            ckpt=str(args.ckpt),
            opponent=str(args.opponent or ""),
            processes=processes,
            games=stats.games,
            positions=stats.positions,
            sims=stats.sims,
            sims_per_move=args.sims,
            max_plies=args.max_plies,
            elapsed_s=round(elapsed, 2),
            sims_per_s=round(sims_per_s, 1),
            speedup_vs_first=round(sims_per_s / base, 2),
            forwards=stats.forwards,
            leaves_per_forward=round(stats.sims / max(stats.forwards, 1), 1),
            gpu_busy_mean=gpu.mean,
            cpu_busy_mean=round(100 * (busy1 - busy0) / max(total1 - total0, 1), 1),
            cpu_cores_used=round((cg1 - cg0) / elapsed, 2)
            if cg0 is not None and cg1 is not None
            else None,
            gpu_name=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            note=args.note,
            results=stats.results,
        )
        print(
            f"processes={processes:>2}  {row['sims_per_s']:>8.0f} sims/s  "
            f"x{row['speedup_vs_first']:<5}  {row['elapsed_s']:>7.1f}s  "
            f"gpu {row['gpu_busy_mean']}%  cores {row['cpu_cores_used']}  "
            f"leaves/forward {row['leaves_per_forward']}",
            flush=True,
        )
        if not args.no_record:
            append_benchmark_row(args.out, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
