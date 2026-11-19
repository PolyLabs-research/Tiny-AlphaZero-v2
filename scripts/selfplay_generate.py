#!/usr/bin/env python3
"""Self-play from one fixed checkpoint into a ring, for labelling (PREREG-p2_ae).

    python scripts/selfplay_generate.py --ckpt runs/p2_ac/e32/size_31892957/best.pt \\
        --config runs/p2_w_quarter_down/config.yaml --seed 5 --chunk-start 0 --chunks 30 \\
        --out runs/p2_ae/gen0 --device cuda --actors 40

Chunks of ``--chunk`` games (``tinyalphazero.generate``): the network against itself
under the config's self-play rules (simulations, Gumbel root, random-opening plies, ply
cap, resignation); chunk ``c`` is the same games wherever and by however many actor
processes it is played (``--actors`` is throughput only). Resumable: rerun the same
command. Writes ``<out>/replay`` (the ring), ``state.json``, ``generate.jsonl`` (one row
per chunk) and ``provenance.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", required=True, help="the config whose self-play rules apply")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--chunk-start", required=True, type=int)
    ap.add_argument("--chunks", required=True, type=int)
    ap.add_argument("--chunk", type=int, default=512, help="games a chunk")
    ap.add_argument("--actors", type=int, default=None, help="default: selfplay.actor_processes")
    ap.add_argument("--sims", type=int, default=None, help="default: selfplay.sims_per_move")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import torch

    from tinyalphazero.config import config_fingerprint, load_config
    from tinyalphazero.generate import generate
    from tinyalphazero.mcts.batch_runner import torch_evaluate_batch
    from tinyalphazero.provenance import git_sha
    from tinyalphazero.train import load_checkpoint

    cfg = load_config(args.config)
    actors = args.actors if args.actors is not None else cfg.selfplay.actor_processes
    device = torch.device(args.device)
    net, _meta = load_checkpoint(args.ckpt, device=device)
    net.eval()
    args.out.mkdir(parents=True, exist_ok=True)
    prov = args.out / "provenance.json"
    record = {
        "ckpt": str(args.ckpt),
        "ckpt_sha256": _sha256(args.ckpt),
        "config": args.config,
        "config_fingerprint": config_fingerprint(cfg),
        "seed": args.seed,
        "chunk_start": args.chunk_start,
        "chunks": args.chunks,
        "chunk": args.chunk,
        "git_sha": git_sha(),
    }
    if prov.exists():
        before = json.loads(prov.read_text())
        same = {k: before.get(k) for k in record if k != "git_sha"}
        if same != {k: v for k, v in record.items() if k != "git_sha"}:
            print(f"{prov} names another run: {before}", file=sys.stderr)
            return 1
    else:
        prov.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"selfplay_generate: chunks {args.chunk_start}..{args.chunk_start + args.chunks - 1} "
        f"of {args.chunk} games, seed {args.seed}, {actors} actors, {device}",
        flush=True,
    )

    def report(row: dict) -> None:
        print(
            f"chunk {row['chunk']}: {row['games']} games, {row['positions']} positions, "
            f"{row['rows']} rows; {row['sims_per_s']} sims/s, mean plies {row['mean_plies']}",
            flush=True,
        )

    state = generate(
        torch_evaluate_batch(net, device), cfg, args.out, args.seed, args.chunk_start,
        args.chunks, chunk=args.chunk, processes=actors, num_sims=args.sims,
        device=str(device), on_chunk=report,
    )  # fmt: skip
    print(json.dumps(state))
    return 0


if __name__ == "__main__":
    sys.exit(main())
