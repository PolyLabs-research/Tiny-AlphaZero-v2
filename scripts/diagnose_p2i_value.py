#!/usr/bin/env python3
"""The p2_i value-head probe (RUNLOG-p2_i §6: diagnose the per_row cost before p2_j).

Loads checkpoints and reads their value head on four position sets, with no search:
the frozen book (plus the initial position); samples of the two rehearsal sets
(mate-in-1, mate-in-2: the rows whose value weight the lever changed); and samples
of each arm's final replay ring, where the stored outcome and root_q give a
calibration and a search-agreement reading on the arms' own self-play. Nothing here
is a registered statistic; it is a diagnosis of the two iteration-20 checkpoints.

    gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \\
      .venv/bin/python scripts/diagnose_p2i_value.py \\
        --net cont=runs/p2_i_cont/ckpt_iter0020.pt --net row=runs/p2_i_row/ckpt_iter0020.pt \\
        --net start=runs/p2_h/ckpt_iter0060.pt \\
        --ring cont=runs/p2_i_cont/replay --ring row=runs/p2_i_row/replay \\
        --reps cont='runs/p2_i_cont/reps_iter0020_r*.jsonl' \\
        --reps row='runs/p2_i_row/reps_iter0020_r*.jsonl' \\
        --out runs/p2_i_row/diag_value_probe.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import chess
import numpy as np
import torch

from tinyalphazero import logschema
from tinyalphazero.encoding import encode_board
from tinyalphazero.evaluate import load_openings
from tinyalphazero.model import value_scalar
from tinyalphazero.replay import ReplayBuffer
from tinyalphazero.train import load_checkpoint


def _values(net: torch.nn.Module, tokens: np.ndarray, device: torch.device) -> np.ndarray:
    """P(win) - P(loss) per row, side-to-move perspective, plus log P(win)."""
    out: list[np.ndarray] = []
    t = torch.from_numpy(tokens.astype(np.int64))
    with torch.inference_mode():
        for i in range(0, len(t), 512):
            wdl = net(t[i : i + 512].to(device)).value_wdl.float()
            v = value_scalar(wdl)
            logp_win = torch.log_softmax(wdl, dim=-1)[:, 2]
            out.append(torch.stack([v, logp_win], dim=1).cpu().numpy())
    return np.concatenate(out)


def _ece(v: np.ndarray, outcome: np.ndarray, bins: int = 10) -> dict:
    predicted = (v + 1) / 2
    realized = (outcome.astype(np.float64) + 1) / 2
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(predicted, edges) - 1, 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = which == b
        if not m.any():
            table.append({"bin": b, "n": 0})
            continue
        p, r = float(predicted[m].mean()), float(realized[m].mean())
        ece += (m.sum() / len(predicted)) * abs(p - r)
        table.append(
            {"bin": b, "n": int(m.sum()), "predicted": round(p, 4), "realized": round(r, 4)}
        )
    return {"ece": round(float(ece), 4), "bins": table}


def _summary(v: np.ndarray) -> dict:
    return {
        "mean_v": round(float(v.mean()), 4),
        "mean_abs_v": round(float(np.abs(v).mean()), 4),
        "frac_abs_v_gt_0_9": round(float((np.abs(v) > 0.9).mean()), 4),
        "frac_abs_v_lt_0_2": round(float((np.abs(v) < 0.2).mean()), 4),
        "hist_v": [int(c) for c in np.histogram(v, bins=np.linspace(-1, 1, 11))[0]],
    }


def _kv(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs:
        k, _, val = p.partition("=")
        if not k or not val:
            raise SystemExit(f"expected NAME=PATH, got {p!r}")
        out[k] = val
    return out


def _mate_sample(d: Path, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads((d / "meta.json").read_text())
    n = int(meta["n"])
    boards = np.memmap(d / "boards.dat", dtype=np.int8, mode="r", shape=(n, 72))
    outcomes = np.memmap(d / "outcomes.dat", dtype=np.int8, mode="r", shape=(n,))
    idx = np.sort(rng.choice(n, size=min(k, n), replace=False))
    return np.asarray(boards[idx]), np.asarray(outcomes[idx])


def _ring_sample(d: Path, k: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    buf = ReplayBuffer.open(d, readonly=True)
    n = len(buf)
    idx = np.sort(rng.choice(n, size=min(k, n), replace=False))
    return {
        "tokens": np.asarray(buf._arr["boards"][idx]),
        "outcome": np.asarray(buf._arr["outcomes"][idx]).astype(np.int8),
        "root_q": np.asarray(buf._arr["root_q"][idx]).astype(np.float64),
        "iteration": np.asarray(buf._arr["iteration"][idx]),
    }


def _per_opening_rep_means(pattern: str) -> np.ndarray:
    rows = [
        r
        for p in sorted(glob.glob(pattern))
        for r in logschema.read_jsonl(p)
        if r.get("row_type") == logschema.MATCH_ROW
    ]
    if not rows:
        raise SystemExit(f"no rep rows match {pattern!r}")
    return np.array([r["pair_scores"] for r in rows], dtype=np.float64).mean(axis=0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--net", action="append", required=True, metavar="NAME=CKPT")
    ap.add_argument("--ring", action="append", default=[], metavar="NAME=REPLAY_DIR")
    ap.add_argument("--reps", action="append", default=[], metavar="NAME=GLOB")
    ap.add_argument("--mate1", type=Path, default=Path("runs/data/mate1_100k_clean"))
    ap.add_argument("--mate2", type=Path, default=Path("runs/data/mate2_50k_clean"))
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(args.seed)
    nets = {}
    for name, path in _kv(args.net).items():
        net, _meta = load_checkpoint(path, device=device)
        nets[name] = net.eval()

    # Position sets, all shared across nets so the comparison is paired.
    book = load_openings()
    book_tokens = []
    for epd, _ident in book:
        b = chess.Board()
        b.set_epd(epd)
        book_tokens.append(encode_board(b, 0))
    sets: dict[str, dict] = {
        "book": {"tokens": np.stack(book_tokens), "ids": [i for _, i in book]},
        "initial_position": {"tokens": encode_board(chess.Board(), 0)[None, :]},
    }
    for label, d in (("mate1", args.mate1), ("mate2", args.mate2)):
        tok, outc = _mate_sample(d, args.sample, rng)
        sets[label] = {"tokens": tok, "outcome": outc}
    for name, d in _kv(args.ring).items():
        sets[f"ring_{name}"] = _ring_sample(Path(d), args.sample, rng)

    result: dict = {"seed": args.seed, "sample": args.sample, "nets": _kv(args.net), "sets": {}}
    values: dict[tuple[str, str], np.ndarray] = {}
    for sname, sdata in sets.items():
        entry: dict = {"n": int(len(sdata["tokens"]))}
        for nname, net in nets.items():
            vl = _values(net, sdata["tokens"], device)
            v, logp_win = vl[:, 0], vl[:, 1]
            values[(sname, nname)] = v
            e = _summary(v)
            if "outcome" in sdata:
                outc = sdata["outcome"]
                e["calibration"] = _ece(v, outc)
                win = outc == 1
                if win.any():
                    e["mean_ce_on_wins"] = round(float(-logp_win[win].mean()), 4)
                    e["mean_v_on_wins"] = round(float(v[win].mean()), 4)
                e["outcome_mix"] = {
                    "win": int((outc == 1).sum()),
                    "draw": int((outc == 0).sum()),
                    "loss": int((outc == -1).sum()),
                }
            if "root_q" in sdata:
                q = sdata["root_q"]
                e["mse_vs_root_q"] = round(float(((v - q) ** 2).mean()), 4)
                e["mean_abs_root_q"] = round(float(np.abs(q).mean()), 4)
                e["corr_v_root_q"] = round(float(np.corrcoef(v, q)[0, 1]), 4)
            entry[nname] = e
        result["sets"][sname] = entry

    # Pairwise differences between nets on the shared sets.
    names = list(nets)
    result["pairs"] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pair: dict = {}
            for sname in sets:
                d = values[(sname, b)] - values[(sname, a)]
                pair[sname] = {
                    "mean_diff_v": round(float(d.mean()), 4),
                    "mean_abs_diff_v": round(float(np.abs(d).mean()), 4),
                    "corr_v": round(
                        float(np.corrcoef(values[(sname, a)], values[(sname, b)])[0, 1]), 4
                    )
                    if len(d) > 2
                    else None,
                }
            result["pairs"][f"{b}-{a}"] = pair

    # The book: per-opening values against the per-opening K1 difference, if reps are given.
    reps = _kv(args.reps)
    if len(reps) == 2 and all(k in nets for k in reps):
        a, b = list(reps)
        score_diff = _per_opening_rep_means(reps[b]) - _per_opening_rep_means(reps[a])
        va, vb = values[("book", a)], values[("book", b)]
        dv = vb - va
        result["book_vs_k1"] = {
            "pair": f"{b}-{a}",
            "corr_dv_scorediff": round(float(np.corrcoef(dv, score_diff)[0, 1]), 4),
            "corr_absdv_scorediff": round(float(np.corrcoef(np.abs(dv), score_diff)[0, 1]), 4),
            "corr_va_scorediff": round(float(np.corrcoef(va, score_diff)[0, 1]), 4),
            "corr_vb_scorediff": round(float(np.corrcoef(vb, score_diff)[0, 1]), 4),
            "mean_score_diff": round(float(score_diff.mean()), 4),
            "per_opening": [
                {
                    "id": sets["book"]["ids"][i],
                    f"v_{a}": round(float(va[i]), 4),
                    f"v_{b}": round(float(vb[i]), 4),
                    "score_diff": round(float(score_diff[i]), 4),
                }
                for i in range(len(dv))
            ],
        }

    text = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    # A short console table.
    for sname, entry in result["sets"].items():
        print(f"== {sname} (n={entry['n']})")
        for nname in nets:
            e = entry[nname]
            line = f"   {nname:6s} mean_v {e['mean_v']:+.3f} |v| {e['mean_abs_v']:.3f} |v|>0.9 {e['frac_abs_v_gt_0_9']:.3f}"
            if "calibration" in e:
                line += f" ece {e['calibration']['ece']:.4f}"
            if "mean_ce_on_wins" in e:
                line += f" ce_wins {e['mean_ce_on_wins']:.4f} v_wins {e['mean_v_on_wins']:+.3f}"
            if "mse_vs_root_q" in e:
                line += f" mse_q {e['mse_vs_root_q']:.4f} corr_q {e['corr_v_root_q']:.3f}"
            print(line)
    if "book_vs_k1" in result:
        bk = result["book_vs_k1"]
        print(
            f"== book vs K1 ({bk['pair']}): corr(dv, score_diff) {bk['corr_dv_scorediff']:+.3f}, corr(|dv|, score_diff) {bk['corr_absdv_scorediff']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
