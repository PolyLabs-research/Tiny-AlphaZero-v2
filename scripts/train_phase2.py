#!/usr/bin/env python3
"""Phase 2 self-play training loop (spec §7).

Usage:
    # start a campaign from the Phase 1 anchor
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python scripts/train_phase2.py \\
        --config configs/default.yaml --init runs/phase1_anchor.pt \\
        --out runs/p2_a --iterations 3

    # continue it after a crash, a reboot, or a deliberate stop
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python scripts/train_phase2.py \\
        --resume runs/p2_a --iterations 50

    # the golden mini-run (CPU, < 3 min)
    python scripts/train_phase2.py --config configs/mini.yaml --out runs/golden --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tinyalphazero.config import load_config
from tinyalphazero.evaluate import format_ladder_event
from tinyalphazero.phase2 import STATE_FILE, run_phase2


def _print_ladder(kind: str, payload: dict) -> None:
    """The ladder's own reporting, rendered by the library so the inline pass
    and `scripts/eval_ladder.py` produce byte-identical output."""
    line = format_ladder_event(kind, payload)
    if line is not None:
        print(line, flush=True)


def _print_stage(name: str, iteration: int) -> None:
    # Machine-readable and line-buffered: the crash-resume test reads these to know
    # when the process is inside generation versus inside training, so that a
    # SIGKILL lands where the test says it lands rather than wherever it happens to.
    print(f"[stage] {name} iteration={iteration}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="YAML config (default: spec defaults)")
    ap.add_argument("--out", type=Path, default=None, help="Run directory to create")
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue an existing run directory: iteration counter, league pool, "
        "replay ring and optimizer are reconstructed from it. Mutually exclusive "
        "with --out.",
    )
    ap.add_argument(
        "--init",
        type=Path,
        default=None,
        help="Checkpoint to start from (the Phase 1 anchor). Omitted means random "
        "init, which is the golden mini-run's case and nobody else's.",
    )
    ap.add_argument(
        "--iterations", type=int, default=None, help="Iterations to run this invocation"
    )
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=None, help="Run seed (default: config seed)")
    ap.add_argument(
        "--ladder-every",
        type=int,
        default=0,
        help="Run an anchor-ladder pass (spec §8) every N iterations, inline and "
        "sequentially — the GPU is never shared. Measured 2026-08-10: a full pass "
        "is ~19 min (the 200-game anchor rung alone is 8.4 min) against a ~27 min "
        "iteration, so --ladder-every 10 costs about 7%% of campaign wall clock. "
        "0 (default) disables it.",
    )
    ap.add_argument(
        "--ladder-anchors",
        default="random,greedy,phase1",
        help="Anchor rungs for --ladder-every. Stockfish rungs skip cleanly when "
        "the binary is absent.",
    )
    ap.add_argument(
        "--ladder-anchor-ckpt",
        type=Path,
        default=None,
        help="Frozen Phase 1 opponent (default: runs/phase1_anchor.pt, anchor v2).",
    )
    args = ap.parse_args(argv)

    if (args.out is None) == (args.resume is None):
        ap.error("pass exactly one of --out (new run) or --resume (existing run)")
    if args.resume is not None and args.init is not None:
        ap.error("--init is meaningless with --resume: a resumed run starts from its own snapshot")
    if args.out is not None and (args.out / STATE_FILE).exists():
        # Silently continuing would be the safe *implementation* and the wrong
        # *behaviour*: "start a campaign" and "continue one" are different
        # intentions, and a run directory is 24 hours of work to reuse by accident.
        ap.error(f"{args.out} is an existing run — pass --resume {args.out} to continue it")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.resume is not None:
        cfg = load_config(str(args.resume / "config.yaml"))
        out = args.resume
    else:
        cfg = load_config(args.config)
        out = args.out

    iterations = args.iterations if args.iterations is not None else cfg.phase2.iterations
    print(
        f"Phase 2 | device={device} | iterations={iterations} | "
        f"games/iter={cfg.selfplay.games_per_iteration} @ {cfg.selfplay.concurrent_games} concurrent "
        f"| sims={cfg.selfplay.sims_per_move} | steps/iter={cfg.train.train_steps_per_iter} "
        f"| batch={cfg.train.batch_size} | out={out}"
    )

    outcomes = run_phase2(
        out,
        cfg,
        device,
        iterations,
        init_ckpt=args.init,
        seed=args.seed,
        resume=args.resume is not None,
        on_stage=_print_stage,
        ladder_every=args.ladder_every,
        ladder_anchors=args.ladder_anchors,
        ladder_anchor_ckpt=args.ladder_anchor_ckpt,
        on_ladder_event=_print_ladder,
    )

    for oc in outcomes:
        r = oc.row
        print(
            f"iter {r['iteration']:4d} | loss={r['loss']:.4f} "
            f"(policy={r['policy_loss']:.4f} value={r['value_loss']:.4f}) | "
            f"decisive={r['decisive_rate']:.3f} draw={r['draw_rate']:.3f} "
            f"(cap={r['draw_rate_ply_cap']:.3f} claim={r['draw_rate_draw_claim']:.3f}) | "
            f"resign={r['resign_rate']:.3f} false={r['false_resign_rate']} | "
            f"plies={r['mean_plies']:.1f} H_prior={r['move1_prior_entropy']:.3f} "
            f"H_target={r['move1_policy_entropy']:.3f} | "
            f"{r['sims_per_s']:.0f} sims/s | "
            f"{r['total_s'] / 60:.1f} min "
            f"(gen {r['selfplay_s'] / 60:.1f} + train {r['train_s'] / 60:.1f} "
            f"+ snap {r['snapshot_s'] / 60:.2f} + other {r['other_s'] / 60:.2f})"
            + (f" | SKIPPED {r['skipped_steps']}" if r["skipped_steps"] else "")
        )

    last = json.loads((Path(out) / STATE_FILE).read_text())["iteration"]
    print(f"Done. Latest checkpoint: {Path(out) / f'ckpt_iter{last:04d}.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
