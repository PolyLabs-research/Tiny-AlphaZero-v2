"""Stockfish as teacher and as annotator (PREREG-p2_t, DIAG-2026-11-07-where-points-go).

One reading of a position, two consumers:

* **the error analysis** grades each move of a recorded game by the expected score
  Stockfish gives the position before and after it;
* **distillation** relabels positions from a self-play replay ring with Stockfish's
  move preferences and evaluation, and writes them back as ordinary ring rows
  (:func:`build_teacher_ring`), which the rehearsal sampler mixes into batches.

**What a teacher row is.** The same position, legal mask and game outcome ``z`` as
the ring row it came from. Two targets change: the policy becomes a softmax over
Stockfish's ``multipv`` lines (:func:`policy_from_lines`), and ``root_q``, the
search value the p2_c blend trains toward, becomes Stockfish's evaluation turned
into an expected result *at our level of play* (:func:`fit_outcome_model`). The
second matters because Stockfish's own WDL model is fitted to engine games: it
calls a 45-centipawn deficit 92% drawn, where our self-play games are about 80%
decisive. Fitting the same two-parameter model to the ring's own outcomes keeps
the value head on the scale the search already reads.

The engine is reached through an ``analyse`` callable, so everything but the
subprocess is tested without a Stockfish binary. Library code: no printing.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np

MATE_CP = 10_000  # python-chess ``mate_score``: a mate in n moves scores MATE_CP - n
FIT_CLIP_CP = 2_000  # centipawns beyond which the outcome model is flat; mates land here
TEACHER_ITERATION = -2  # the ring ``iteration`` of a teacher row (rehearsal anchors are -1)
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


@dataclass(frozen=True)
class Line:
    """One ``multipv`` line: its first move and the score from the side to move."""

    move: str  # UCI
    cp: int  # mates mapped to +-(MATE_CP - moves)
    pv: tuple[str, ...] = ()


@dataclass(frozen=True)
class Reading:
    """Stockfish's view of one position, from the side to move; ``lines`` best first."""

    lines: tuple[Line, ...]
    wdl: tuple[float, float, float]  # engine's (win, draw, loss), side to move, sums to 1
    depth: int
    nodes: int

    @property
    def cp(self) -> int:
        return self.lines[0].cp

    @property
    def mate_for_mover(self) -> bool:
        return self.cp >= MATE_CP - 500

    @property
    def mate_against_mover(self) -> bool:
        return self.cp <= -(MATE_CP - 500)

    @property
    def expected(self) -> float:
        """The engine's expected score for the side to move, ``win + draw / 2``."""
        w, d, _ = self.wdl
        return w + d / 2


Analyse = Callable[[chess.Board, int, int], Reading]  # (board, nodes, multipv) -> Reading


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def open_engine(path: str, hash_mb: int = 64) -> Any:
    """One single-threaded Stockfish with WDL output: deterministic at fixed nodes."""
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(path)
    engine.configure({"Threads": 1, "Hash": hash_mb, "UCI_ShowWDL": True})
    return engine


def engine_analyse(engine: Any) -> Analyse:
    """Wrap an open engine as an :data:`Analyse`.

    ``game=object()`` makes python-chess send ``ucinewgame`` before every position,
    which clears the hash: a reading depends on the position and its move history
    only, not on what the same process searched before, so labels do not depend on
    how positions were dealt to workers.
    """
    import chess.engine

    def analyse(board: chess.Board, nodes: int, multipv: int) -> Reading:
        infos = engine.analyse(
            board, chess.engine.Limit(nodes=nodes), multipv=multipv, game=object()
        )
        if isinstance(infos, dict):
            infos = [infos]
        lines = []
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            score = info["score"].pov(board.turn).score(mate_score=MATE_CP)
            lines.append(Line(pv[0].uci(), int(score), tuple(m.uci() for m in pv)))
        if not lines:
            raise RuntimeError(f"Stockfish returned no line for {board.fen()}")
        wdl = infos[0].get("wdl")
        if wdl is None:
            raise RuntimeError("Stockfish returned no WDL: is UCI_ShowWDL set?")
        w = wdl.pov(board.turn)
        total = float(w.wins + w.draws + w.losses)
        return Reading(
            lines=tuple(lines),
            wdl=(w.wins / total, w.draws / total, w.losses / total),
            depth=int(infos[0].get("depth", 0)),
            nodes=int(infos[0].get("nodes", 0)),
        )

    return analyse


def available_cpus() -> int:
    """CPUs this process may use: the cgroup quota, not the host's count.

    On RunPod ``nproc`` reports the host (96) while the pod's quota was 7.65 for a pod
    RunPod called 9 vCPU (cloud/README.md); sizing a worker pool from the host count
    oversubscribes it. The smallest of the affinity mask, ``RUNPOD_CPU_COUNT`` and the
    cgroup quota (v2 ``cpu.max``, else v1) wins, as ``cloud/setup_pod.sh`` reads it.
    """
    counts = [len(os.sched_getaffinity(0))]
    env = os.environ.get("RUNPOD_CPU_COUNT", "")
    if env.isdigit() and int(env) > 0:
        counts.append(int(env))
    for quota_file, period_file in (
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            if period_file is None:
                quota, period = Path(quota_file).read_text().split()[:2]
            else:
                quota = Path(quota_file).read_text().strip()
                period = Path(period_file).read_text().strip()
            if quota not in ("max", "-1"):
                counts.append(int(int(quota) / int(period)))
            break
        except (OSError, ValueError):
            continue
    return max(1, min(counts))


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def policy_from_lines(lines: Sequence[Line], temperature_cp: float) -> list[tuple[str, float]]:
    """Softmax over the lines' scores: ``p_i ~ exp((cp_i - cp_best) / T)``.

    Lines are the engine's ``multipv`` set, so moves outside it get no mass: the
    engine ranked them below every move it reports. A duplicated move (an engine
    reporting the same first move twice) keeps its best score once.
    """
    if temperature_cp <= 0:
        raise ValueError(f"temperature must be positive, got {temperature_cp}")
    best: dict[str, int] = {}
    for line in lines:
        if line.move not in best or line.cp > best[line.move]:
            best[line.move] = line.cp
    if not best:
        raise ValueError("no lines to make a policy from")
    top = max(best.values())
    weights = {m: math.exp((cp - top) / temperature_cp) for m, cp in best.items()}
    total = sum(weights.values())
    return sorted(((m, w / total) for m, w in weights.items()), key=lambda x: -x[1])


def outcome_probs(cp: np.ndarray | float, a: float, s: float) -> np.ndarray:
    """``(win, draw, loss)`` for the side to move under the ordinal model.

    ``P(win) = sigmoid((cp - a) / s)``, ``P(loss) = sigmoid((-cp - a) / s)``, draw
    the rest: Stockfish's own WDL form, with ``a >= 0`` the half-width of the draw
    band and ``s`` its slope, here fitted to our games rather than to the engine's.
    """
    x = np.clip(np.asarray(cp, dtype=np.float64), -FIT_CLIP_CP, FIT_CLIP_CP)
    win = 1.0 / (1.0 + np.exp(-(x - a) / s))
    loss = 1.0 / (1.0 + np.exp(-(-x - a) / s))
    draw = np.clip(1.0 - win - loss, 0.0, 1.0)
    return np.stack([win, draw, loss], axis=-1)


def fit_outcome_model(cp: np.ndarray, z: np.ndarray, iters: int = 200) -> tuple[float, float]:
    """Maximum-likelihood ``(a, s)`` of :func:`outcome_probs` for outcomes ``z``.

    ``cp`` is the engine's score and ``z`` the game result (+1 / 0 / -1), both from
    the side to move. Two parameters, a few hundred thousand rows: Newton steps on
    ``(log a, log s)`` would do, but a damped L-BFGS in torch is shorter and the
    dependency is already here.
    """
    import torch

    x = torch.tensor(np.clip(cp, -FIT_CLIP_CP, FIT_CLIP_CP), dtype=torch.float64)
    cls = torch.tensor(np.asarray(z, dtype=np.int64) + 1)  # 0 loss, 1 draw, 2 win
    params = torch.tensor(
        [math.log(50.0), math.log(200.0)], dtype=torch.float64, requires_grad=True
    )
    opt = torch.optim.LBFGS([params], max_iter=iters, line_search_fn="strong_wolfe")

    def nll() -> torch.Tensor:
        opt.zero_grad()
        a, s = params.exp()
        win = torch.sigmoid((x - a) / s)
        loss = torch.sigmoid((-x - a) / s)
        draw = (1 - win - loss).clamp_min(1e-12)
        probs = torch.stack([loss, draw, win], dim=-1).clamp_min(1e-12)
        value = -probs.gather(1, cls[:, None]).log().mean()
        value.backward()
        return value

    opt.step(nll)
    a, s = (float(v) for v in params.detach().exp())
    return a, s


def material(board: chess.Board, color: chess.Color) -> int:
    """Material of ``color`` minus the other side's, in pawns (P1 N3 B3 R5 Q9)."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        total += value * (
            len(board.pieces(piece_type, color)) - len(board.pieces(piece_type, not color))
        )
    return total


def material_after_line(
    board: chess.Board, pv: Iterable[str], color: chess.Color, plies: int
) -> int:
    """``color``'s material once up to ``plies`` moves of ``pv`` are played on a copy."""
    b = board.copy(stack=False)
    for i, uci in enumerate(pv):
        if i >= plies:
            break
        move = chess.Move.from_uci(uci)
        if move not in b.legal_moves:
            break
        b.push(move)
    return material(b, color)


def non_pawn_material(board: chess.Board) -> int:
    """Both sides' knights, bishops, rooks and queens, in pawns."""
    return sum(
        value * len(board.pieces(pt, c))
        for pt, value in PIECE_VALUES.items()
        if pt != chess.PAWN
        for c in (chess.WHITE, chess.BLACK)
    )


# ---------------------------------------------------------------------------
# The teacher ring
# ---------------------------------------------------------------------------


@dataclass
class TeacherSettings:
    """Everything a teacher ring's labels depend on besides the positions; in teacher.json."""

    nodes: int = 200_000
    multipv: int = 5
    temperature_cp: float = 50.0
    hash_mb: int = 64
    stockfish: str = ""  # path; its version line is recorded beside it
    stockfish_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def position_key(tokens: np.ndarray) -> bytes:
    """The position proper: squares, side, castling, en passant (``book_rows_in_ring``'s key)."""
    return np.asarray(tokens, dtype=np.int8)[:70].tobytes()


def exclusion_keys(epd_paths: Sequence[str | Path]) -> set[bytes]:
    """Keys of every position in the given EPD suites: the gates, the book, the start."""
    from tinyalphazero.encoding import encode_board

    keys: set[bytes] = set()
    for path in epd_paths:
        for raw in Path(path).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            board = chess.Board()
            board.set_epd(line)
            keys.add(position_key(encode_board(board, 0)))
    return keys


def select_rows(
    boards: np.ndarray, excluded: set[bytes], seen: set[bytes] | None = None
) -> tuple[np.ndarray, dict[str, int]]:
    """Ring rows to label: first occurrence of each position, none excluded.

    Rows whose position had already occurred in its game (repetition token > 0) are
    dropped too: their right value depends on a history a FEN cannot carry, and the
    engine would score them blind to the repetition.

    ``seen`` carries "already taken" across calls: :func:`pool_rows` passes one set
    through several rings, so a position kept from an earlier ring counts as a
    duplicate in every later one. It is updated in place.
    """
    from tinyalphazero.encoding import REP_0

    seen = set() if seen is None else seen
    keep: list[int] = []
    counts = {"rows": int(len(boards)), "repeated": 0, "excluded": 0, "duplicate": 0}
    for i in range(len(boards)):
        row = np.asarray(boards[i])
        if int(row[71]) != REP_0:
            counts["repeated"] += 1
            continue
        key = position_key(row)
        if key in excluded:
            counts["excluded"] += 1
            continue
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        keep.append(i)
    counts["selected"] = len(keep)
    return np.asarray(keep, dtype=np.int64), counts


def reading_to_json(row: int, reading: Reading) -> dict[str, Any]:
    return {
        "row": int(row),
        "lines": [[ln.move, ln.cp] for ln in reading.lines],
        "wdl": [round(v, 4) for v in reading.wdl],
        "depth": reading.depth,
        "nodes": reading.nodes,
    }


def label_rows(
    boards: np.ndarray,
    rows: Sequence[int],
    analyse: Analyse,
    settings: TeacherSettings,
    out: Path,
) -> int:
    """Append one JSON line per ring row to ``out``; rows already there are skipped.

    Restartable: a worker killed mid-shard resumes after the last complete line.
    Returns the number of rows labelled by this call.
    """
    from tinyalphazero.encoding import decode_board

    done: set[int] = set()
    if out.exists():
        data = out.read_bytes()
        complete = data[: data.rfind(b"\n") + 1]  # a torn last line is cut, then relabelled
        if len(complete) != len(data):
            with out.open("r+b") as fh:
                fh.truncate(len(complete))
        for raw in complete.decode().splitlines():
            done.add(int(json.loads(raw)["row"]))
    n = 0
    with out.open("a") as fh:
        for row in rows:
            if int(row) in done:
                continue
            board = decode_board(np.asarray(boards[int(row)]))
            reading = analyse(board, settings.nodes, settings.multipv)
            fh.write(json.dumps(reading_to_json(int(row), reading)) + "\n")
            fh.flush()
            n += 1
    return n


def read_labels(paths: Iterable[Path]) -> dict[int, dict[str, Any]]:
    """Every complete label line across the shards, keyed by ring row."""
    labels: dict[int, dict[str, Any]] = {}
    for path in paths:
        with Path(path).open() as fh:
            for raw in fh:
                try:
                    item = json.loads(raw)
                except ValueError:
                    continue
                labels[int(item["row"])] = item
    return labels


def labelled_rows(paths: Iterable[Path]) -> set[int]:
    """The rows any shard already holds a complete label for (``read_labels``'s keys, cheaply).

    The labeller hands out only the rest, so shards written by any number of workers -- a
    pod's 54, say -- are finished by a rerun with any ``--workers``, which then labels nothing.
    """
    rows: set[int] = set()
    for path in paths:
        with Path(path).open() as fh:
            for raw in fh:
                try:
                    rows.add(int(json.loads(raw)["row"]))
                except ValueError:
                    continue
    return rows


def build_teacher_ring(
    source: Any,
    labels: dict[int, dict[str, Any]],
    rows: Sequence[int],
    out_dir: Path,
    settings: TeacherSettings,
    provenance: dict[str, Any],
    model: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Write the teacher ring: one row per labelled source row, in ``rows`` order.

    ``source`` is the self-play ring the positions came from (read-only). Unless
    ``model`` gives ``(a, s)``, the outcome model is fitted first, on every labelled
    row's (engine score, game outcome); each row's ``root_q`` is then
    ``P(win) - P(loss)`` under it. A held-out ring passes the training ring's fit.
    """
    from tinyalphazero.encoding import encode_move
    from tinyalphazero.replay import ReplayBuffer

    rows = [int(r) for r in rows if int(r) in labels]
    if not rows:
        raise ValueError("no labelled rows to write")
    arr = source._arr
    cp = np.array([labels[r]["lines"][0][1] for r in rows], dtype=np.float64)
    z = np.array([int(arr["outcomes"][r]) for r in rows], dtype=np.int64)
    a, s = model if model is not None else fit_outcome_model(cp, z)
    probs = outcome_probs(cp, a, s)

    ring = ReplayBuffer.create(out_dir, capacity=len(rows))
    for k, r in enumerate(rows):
        lines = [Line(m, int(c)) for m, c in labels[r]["lines"]]
        policy = []
        for uci, p in policy_from_lines(lines, settings.temperature_cp):
            f, t, promo = encode_move(chess.Move.from_uci(uci))
            policy.append((f * 64 + t, promo, float(p)))
        best = chess.Move.from_uci(lines[0].move)
        f, t, promo = encode_move(best)
        ring.add(
            tokens=np.asarray(arr["boards"][r]),
            legals=np.asarray(arr["legals"][r]),
            move_ft=f * 64 + t,
            promo=promo,
            outcome=int(z[k]),
            improved_policy=policy,
            iteration=TEACHER_ITERATION,
            resign_disabled=False,
            mover_is_latest=False,
            root_q=float(probs[k, 0] - probs[k, 2]),
        )
    ring.flush()

    draws = float(np.mean(z == 0))
    meta = {
        "settings": asdict(settings),
        "outcome_model": {
            "a": a,
            "s": s,
            "fitted_here": model is None,
            "rows": len(rows),
            "draw_rate": draws,
        },  # fmt: skip
        "rows": len(rows),
        "root_q_mean_abs": float(np.mean(np.abs(probs[:, 0] - probs[:, 2]))),
        **provenance,
    }
    (Path(out_dir) / "teacher.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


# ---------------------------------------------------------------------------
# Pools across rings and agreement with the teacher (DIAG-2026-11-13 capacity-or-data)
# ---------------------------------------------------------------------------


def pool_rows(
    boards: Sequence[np.ndarray],
    excluded: set[bytes],
    seed: int,
    limit: int | None = None,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Positions from several rings, each once, as ``(ring, row)`` pairs in a seeded order.

    Each ring's boards go through :func:`select_rows` with one ``seen`` set shared
    across rings, taken in the order given, so a position kept from an earlier ring is a
    duplicate in every later one. The kept pairs are then shuffled with ``seed`` and cut
    to ``limit``: every prefix of the result is a uniform sample of the pool, which is
    what lets one labelled ring serve as nested training sets.
    """
    seen: set[bytes] = set()
    picks: list[np.ndarray] = []
    counts: list[dict[str, int]] = []
    for k, ring_boards in enumerate(boards):
        rows, c = select_rows(ring_boards, excluded, seen=seen)
        picks.append(np.stack([np.full(len(rows), k, dtype=np.int64), rows], axis=1))
        counts.append(c)
    pairs = np.concatenate(picks) if picks else np.zeros((0, 2), dtype=np.int64)
    pairs = pairs[np.random.default_rng(seed).permutation(len(pairs))]
    if limit is not None:
        pairs = pairs[:limit]
    return pairs, counts


def write_pool(rings: Sequence[Any], pairs: np.ndarray, out_dir: Path, chunk: int = 100_000) -> Any:
    """Copy the picked rows, every stored field as it is, into a new ring in ``pairs`` order.

    The pool is a source ring for ``scripts/label_teacher.py``: it reads boards, legal
    masks and game outcomes from it, so those must be the source rows' own, bit for bit.
    Every source must carry every field the new ring has.
    """
    from tinyalphazero.replay import ReplayBuffer

    out = ReplayBuffer.create(out_dir, capacity=max(1, len(pairs)))
    for k, ring in enumerate(rings):
        missing = [f for f in out._arr if f not in ring._arr]
        if missing and np.any(pairs[:, 0] == k):
            raise ValueError(f"source ring {k} lacks fields {missing}")
    for lo in range(0, len(pairs), chunk):
        part = pairs[lo : lo + chunk]
        for k in np.unique(part[:, 0]):
            sel = np.flatnonzero(part[:, 0] == k)
            src = part[sel, 1]
            for name, arr in out._arr.items():
                arr[lo + sel] = rings[int(k)]._arr[name][src]
    out.total_written = len(pairs)
    out.flush()
    return out


def agreement(
    net: Any, ring: Any, device: Any, batch: int = 512, rows: np.ndarray | None = None
) -> dict[str, Any]:
    """How close ``net``'s prior is to the teacher on ``ring``'s rows (no search).

    The summary (``top1_agreement``, ``teacher_policy_ce``, ``value_mse_vs_teacher``) is
    computed exactly as ``scripts/teacher_agreement.py`` always has: rows in order,
    batch sums accumulated in that order, so a T1 reading reproduces. Added for the
    capacity-or-data diagnostic: ``teacher_entropy`` (the teacher's own entropy, the
    floor no prior can pass) and, per row, ``ce_rows`` and ``entropy_rows`` for paired
    comparisons between checkpoints. ``rows`` restricts to those ring rows, in order.
    """
    import torch

    from tinyalphazero.model import masked_policy, value_scalar
    from tinyalphazero.train import unpack_legal_mask

    arr = ring._arr
    idx_all = np.arange(len(ring)) if rows is None else np.asarray(rows, dtype=np.int64)
    n = len(idx_all)
    top1 = 0
    ce_sum = 0.0
    se_sum = 0.0
    ce_rows = np.zeros(n, dtype=np.float64)
    ent_rows = np.zeros(n, dtype=np.float64)
    was_training = net.training
    net.eval()
    for lo in range(0, n, batch):
        idx = idx_all[lo : lo + batch]
        tokens = torch.from_numpy(arr["boards"][idx].astype(np.int64)).to(device)
        legal = unpack_legal_mask(torch.from_numpy(np.asarray(arr["legals"][idx]))).to(device)
        target = torch.from_numpy(np.stack([ring.densify_one(int(i)) for i in idx])).to(device)
        best = torch.from_numpy(arr["moves"][idx].astype(np.int64)).to(device)
        with torch.no_grad():
            out = net(tokens)
            probs = masked_policy(out.policy_ft.float(), legal)
            value = value_scalar(out.value_wdl.float())
        top1 += int((probs.argmax(dim=1) == best[:, 0] * 64 + best[:, 1]).sum())
        per_row = -(target * probs.clamp_min(1e-12).log()).sum(dim=1)
        ce_sum += float(per_row.sum())
        root_q = torch.from_numpy(arr["root_q"][idx].astype(np.float32)).to(device)
        se_sum += float(((value - root_q) ** 2).sum())
        ent = -(target * target.clamp_min(1e-12).log()).sum(dim=1)
        ce_rows[lo : lo + len(idx)] = per_row.double().cpu().numpy()
        ent_rows[lo : lo + len(idx)] = ent.double().cpu().numpy()
    if was_training:
        net.train()
    return {
        "rows": n,
        "top1_agreement": top1 / n,
        "teacher_policy_ce": ce_sum / n,
        "value_mse_vs_teacher": se_sum / n,
        "teacher_entropy": float(ent_rows.mean()),
        "ce_rows": ce_rows,
        "entropy_rows": ent_rows,
    }
