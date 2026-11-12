"""Run a checkpoint as a UCI engine on stdin/stdout (the public benchmark).

    .venv/bin/python scripts/uci_engine.py --ckpt runs/<run>/ckpt_iter0010.pt
    .venv/bin/python scripts/uci_engine.py --ckpt ... --device cuda --sims 48

The engine plays the evaluation profile at ``eval.eval_match_sims`` from the run's
``config.yaml`` (beside ``--ckpt`` unless ``--config`` says otherwise) -- the
configuration every ladder and verdict number was measured in. See
``tinyalphazero.uci`` for what is held equal to the ladder and how.

stdout carries the protocol and nothing else; diagnostics go to stderr. On cpu,
torch runs ``--threads`` intra-op threads (default 1): an engine that spreads over
every core under a gauntlet is the all-core load that has tripped this box's CPU
thermal limit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tinyalphazero.config import config_fingerprint, load_config
from tinyalphazero.uci import UciEngine, UciSession, single_from_batch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, type=Path, help="Checkpoint to play")
    ap.add_argument(
        "--config",
        default=None,
        help="YAML config. Default: the config.yaml beside --ckpt when there is one, "
        "else the dataclass defaults (as scripts/eval_ladder.py).",
    )
    ap.add_argument("--sims", type=int, default=None, help="Default: eval.eval_match_sims")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (default cpu)")
    ap.add_argument("--threads", type=int, default=1, help="torch intra-op threads on cpu")
    ap.add_argument("--name", default=None, help="UCI id name (default: TinyAlphaZero-v2 <ckpt>)")
    args = ap.parse_args(argv)

    config_path = args.config
    if config_path is None and (args.ckpt.parent / "config.yaml").exists():
        config_path = str(args.ckpt.parent / "config.yaml")
    cfg = load_config(config_path)
    sims = args.sims if args.sims is not None else cfg.eval.eval_match_sims
    name = args.name or f"TinyAlphaZero-v2 {args.ckpt.parent.name}/{args.ckpt.stem}"
    print(
        f"uci_engine: {args.ckpt} | config {config_path or 'dataclass defaults'} "
        f"fingerprint {config_fingerprint(cfg)[:12]} | sims {sims} | device {args.device}",
        file=sys.stderr,
        flush=True,
    )

    def make_engine(engine_sims: int) -> UciEngine:
        import torch

        from tinyalphazero.mcts.batch_runner import torch_evaluate_batch
        from tinyalphazero.train import load_checkpoint

        torch.set_num_threads(max(1, args.threads))
        device = torch.device(args.device)
        net, _meta = load_checkpoint(args.ckpt, device=device)
        net.eval()
        return UciEngine(
            single_from_batch(torch_evaluate_batch(net, device)), cfg.mcts, engine_sims
        )

    def write(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    session = UciSession(make_engine, write, default_sims=sims, name=name)
    for line in sys.stdin:
        if not session.handle(line):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
