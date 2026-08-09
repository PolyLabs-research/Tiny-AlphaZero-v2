"""Diagnose the phase1_c residual: what ARE the 2.4% illegal argmaxes, and does
top-k masked policy cover the mates that top-1 misses?

Standing tooling, not a one-off: this is the instrument that produced
BLOCKED-2026-08-06-phase1-gates.md (check-legality tail vs. geometry, failure-set
overlap across checkpoints, masked top-k anchor coverage). Reach for it whenever a
policy-head metric plateaus and the question is *which* positions are failing.

Usage:
  VIRTUAL_ENV=.venv .venv/bin/python scripts/diagnose_phase1.py \
      --ckpt runs/phase1_anchor.pt --ckpt2 runs/phase1_c/epoch_04.pt \
      --eval-rand runs/data/eval_rand --eval-mate1 runs/data/eval_mate1_1k
"""

from __future__ import annotations

import argparse
from collections import Counter

import chess
import torch

from tinyalphazero.data import Phase1Dataset
from tinyalphazero.encoding import decode_board
from tinyalphazero.model import ChessNet
from tinyalphazero.train import unpack_legal_mask


def load_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]["model"] if isinstance(ck["config"], dict) else ck["config"].model
    from tinyalphazero.config import load_config

    base = load_config()
    for k, v in cfg.items() if isinstance(cfg, dict) else vars(cfg).items():
        if hasattr(base.model, k):
            setattr(base.model, k, v)
    net = ChessNet(base.model).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


@torch.no_grad()
def forward_all(net, tokens, device, bs=256):
    outs = []
    for i in range(0, len(tokens), bs):
        o = net(tokens[i : i + bs].to(device))
        outs.append(o.policy_ft.float().cpu())
    return torch.cat(outs)


def classify_failure(board: chess.Board, f: int, t: int) -> str:
    piece = board.piece_at(f)
    if piece is None:
        return "EMPTY_FROM_SQUARE"
    if piece.color != board.turn:
        return "OPPONENT_PIECE_AT_FROM"
    promo = None
    if piece.piece_type == chess.PAWN and chess.square_rank(t) in (0, 7):
        promo = chess.QUEEN
    mv = chess.Move(f, t, promotion=promo)
    if board.is_pseudo_legal(mv):
        return "PSEUDO_LEGAL__LEAVES_KING_IN_CHECK"  # pins / moving into or ignoring check
    if piece.piece_type == chess.KING and abs(chess.square_file(t) - chess.square_file(f)) == 2:
        return "CASTLING_RELATED"
    return "BAD_GEOMETRY_OR_BLOCKED"


def illegal_argmax_report(name, ds, net, device):
    tokens = torch.stack([ds[i]["tokens"] for i in range(len(ds))])
    legals = torch.stack([ds[i]["legals"] for i in range(len(ds))])
    mask = unpack_legal_mask(legals).reshape(len(ds), 4096)
    pol = forward_all(net, tokens, device).reshape(len(ds), 4096)
    top1 = pol.argmax(1)
    illegal_idx = (~mask.gather(1, top1[:, None]).squeeze(1)).nonzero().squeeze(1).tolist()
    print(
        f"\n[{name}] illegal unmasked argmax: {len(illegal_idx)}/{len(ds)} "
        f"({len(illegal_idx) / len(ds):.3%})"
    )
    cats = Counter()
    in_check = 0
    for i in illegal_idx:
        board = decode_board(tokens[i].numpy())
        f, t = divmod(int(top1[i]), 64)
        cats[classify_failure(board, f, t)] += 1
        if board.is_check():
            in_check += 1
    for k, v in cats.most_common():
        print(f"    {k:38} {v:4}  ({v / max(len(illegal_idx), 1):.1%})")
    if illegal_idx:
        print(f"    (position was in check for {in_check}/{len(illegal_idx)} failures)")
    return set(illegal_idx)


@torch.no_grad()
def mate_topk(ds, net, device, ks=(1, 3, 8, 16)):
    tokens = torch.stack([ds[i]["tokens"] for i in range(len(ds))])
    legals = torch.stack([ds[i]["legals"] for i in range(len(ds))])
    target = torch.tensor(
        [int(ds[i]["from_sq"]) * 64 + int(ds[i]["to_sq"]) for i in range(len(ds))]
    )
    mask = unpack_legal_mask(legals).reshape(len(ds), 4096)
    pol = forward_all(net, tokens, device).reshape(len(ds), 4096)
    pol = pol.masked_fill(~mask, float("-inf"))
    order = pol.argsort(dim=1, descending=True)
    print(f"\n[mate1 masked top-k coverage over {len(ds)} anchors]")
    for k in ks:
        hit = (order[:, :k] == target[:, None]).any(1).float().mean().item()
        print(f"    top-{k:<3} {hit:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt2", default=None, help="second checkpoint for failure-set overlap")
    ap.add_argument("--eval-rand", required=True)
    ap.add_argument("--eval-mate1", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rand = Phase1Dataset(args.eval_rand)
    mate = Phase1Dataset(args.eval_mate1)
    net = load_net(args.ckpt, device)

    fail_final = illegal_argmax_report(f"{args.ckpt}", rand, net, device)
    mate_topk(mate, net, device)

    if args.ckpt2:
        net2 = load_net(args.ckpt2, device)
        fail_e4 = illegal_argmax_report(f"{args.ckpt2}", rand, net2, device)
        inter = len(fail_final & fail_e4)
        uni = len(fail_final | fail_e4)
        print(
            f"\n[failure-set overlap] |final ∩ e4|={inter}  |union|={uni}  "
            f"Jaccard={inter / max(uni, 1):.2f}  (high overlap ⇒ persistent hard class, not noise)"
        )


if __name__ == "__main__":
    main()
