"""Stockfish as teacher and annotator: the parts that need no binary (tinyalphazero.teacher).

The engine is a fake ``analyse`` here; what is tested is everything around it: the
targets made from a reading, the outcome model, which ring rows are labelled, the
teacher ring's rows, and the error analysis's grading of a game.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest

from tinyalphazero.encoding import REP_0, encode_board, legal_moves_packed
from tinyalphazero.replay import ReplayBuffer
from tinyalphazero.teacher import (
    MATE_CP,
    TEACHER_ITERATION,
    Line,
    Reading,
    TeacherSettings,
    build_teacher_ring,
    exclusion_keys,
    fit_outcome_model,
    label_rows,
    material,
    material_after_line,
    outcome_probs,
    policy_from_lines,
    position_key,
    read_labels,
    select_rows,
)

REPO = Path(__file__).resolve().parents[1]


def test_policy_is_a_softmax_over_the_lines_in_centipawns() -> None:
    lines = [Line("e2e4", 30), Line("d2d4", 30), Line("g1f3", -20)]
    policy = dict(policy_from_lines(lines, temperature_cp=50.0))
    assert policy["e2e4"] == pytest.approx(policy["d2d4"])
    assert policy["g1f3"] / policy["e2e4"] == pytest.approx(np.exp(-1.0))
    assert sum(policy.values()) == pytest.approx(1.0)


def test_policy_keeps_a_repeated_move_once_at_its_best_score() -> None:
    policy = policy_from_lines([Line("e2e4", 10), Line("e2e4", -500), Line("d2d4", 10)], 50.0)
    assert dict(policy) == pytest.approx({"e2e4": 0.5, "d2d4": 0.5})


def test_policy_puts_the_mates_ahead_of_any_material() -> None:
    lines = [Line("h5f7", MATE_CP - 1), Line("d1h5", MATE_CP - 3), Line("e1g1", 900)]
    policy = dict(policy_from_lines(lines, 50.0))
    assert policy["e1g1"] < 1e-50
    assert policy["h5f7"] > policy["d1h5"] > 0.4


def test_policy_refuses_a_zero_temperature() -> None:
    with pytest.raises(ValueError):
        policy_from_lines([Line("e2e4", 0)], 0.0)


def test_outcome_model_is_a_distribution_and_symmetric() -> None:
    cp = np.array([-3000.0, -150.0, 0.0, 150.0, MATE_CP])
    p = outcome_probs(cp, a=60.0, s=120.0)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[1], p[3][::-1])  # (w, d, l) at -150 is (l, d, w) at +150
    assert p[2, 0] == pytest.approx(p[2, 2]) and p[2, 1] > 0
    assert p[4, 0] > 0.99  # a mate is clipped to the model's flat end, a near-certain win


def test_outcome_model_fit_recovers_its_parameters() -> None:
    rng = np.random.default_rng(0)
    cp = rng.normal(0, 300, size=40_000)
    probs = outcome_probs(cp, a=80.0, s=150.0)
    z = np.array([2 - rng.choice(3, p=row) for row in probs]) - 1  # (w, d, l) -> +1, 0, -1
    a, s = fit_outcome_model(cp, z)
    assert a == pytest.approx(80.0, rel=0.08)
    assert s == pytest.approx(150.0, rel=0.08)


def test_material_and_the_line_it_ends_on() -> None:
    board = chess.Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")
    assert material(board, chess.WHITE) == 0
    # Qxd5 wins the queen; the line stops at its first illegal move rather than raising.
    assert material_after_line(board, ["d1d5", "e8e7", "a1a2"], chess.WHITE, 8) == 9
    assert material_after_line(board, ["d1d5"], chess.BLACK, 0) == 0


def _ring_with(tmp: Path, boards: list[chess.Board], reps: list[int], z: list[int]) -> ReplayBuffer:
    ring = ReplayBuffer.create(tmp, capacity=len(boards))
    for b, r, o in zip(boards, reps, z, strict=True):
        move = next(iter(b.legal_moves))
        ring.add(
            tokens=encode_board(b, r),
            legals=legal_moves_packed(b),
            move_ft=move.from_square * 64 + move.to_square,
            promo=0,
            outcome=o,
            improved_policy=[(move.from_square * 64 + move.to_square, 0, 1.0)],
            iteration=7,
            root_q=0.25,
        )
    ring.flush()
    return ring


def test_rows_are_first_occurrences_off_the_gates_and_never_repeated(tmp_path: Path) -> None:
    start = chess.Board()
    e4 = chess.Board()
    e4.push_uci("e2e4")
    book = chess.Board()
    book.set_epd(
        (REPO / "tests/suites/openings_4ply.epd").read_text().splitlines()[0].split(" bm")[0]
    )
    ring = _ring_with(
        tmp_path / "ring", [e4, e4, start, book, e4], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]
    )
    excluded = exclusion_keys(
        [REPO / "tests/suites/openings_4ply.epd", REPO / "tests/suites/openings_startpos.epd"]
    )
    assert position_key(encode_board(book, 0)) in excluded
    rows, counts = select_rows(ring._arr["boards"][: len(ring)], excluded)
    assert rows.tolist() == [0]  # e4 once; start and the book position excluded; rep 1 dropped
    assert counts == {
        "rows": 5, "repeated": 1, "excluded": 2, "duplicate": 1, "selected": 1,
    }  # fmt: skip
    assert int(ring._arr["boards"][4][71]) != REP_0


def _fake_analyse(board: chess.Board, nodes: int, multipv: int) -> Reading:
    moves = sorted(board.legal_moves, key=lambda m: m.uci())[:multipv]
    return Reading(
        lines=tuple(Line(m.uci(), 40 - 20 * i) for i, m in enumerate(moves)),
        wdl=(0.2, 0.7, 0.1),
        depth=12,
        nodes=nodes,
    )


def test_labelling_resumes_after_the_last_complete_line(tmp_path: Path) -> None:
    boards = [chess.Board(), chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")]
    ring = _ring_with(tmp_path / "ring", boards, [0, 0], [1, -1])
    out = tmp_path / "labels_0.jsonl"
    settings = TeacherSettings(nodes=1000, multipv=3)
    arr = ring._arr["boards"]
    assert label_rows(arr, [0], _fake_analyse, settings, out) == 1
    with out.open("a") as fh:
        fh.write('{"row": 1, "lin')  # a worker killed mid-write
    assert label_rows(arr, [0, 1], _fake_analyse, settings, out) == 1
    labels = read_labels([out])
    assert sorted(labels) == [0, 1]
    assert len(labels[0]["lines"]) == 3 and labels[0]["nodes"] == 1000


def test_teacher_ring_rows_keep_the_position_and_outcome_and_take_the_engine_targets(
    tmp_path: Path,
) -> None:
    boards = []
    rng = np.random.default_rng(1)
    for _ in range(60):
        b = chess.Board()
        for _ in range(int(rng.integers(2, 12))):
            b.push(list(b.legal_moves)[int(rng.integers(0, b.legal_moves.count()))])
        boards.append(b)
    z = [int(v) for v in rng.choice([-1, 0, 1], size=len(boards))]
    ring = _ring_with(tmp_path / "ring", boards, [0] * len(boards), z)
    settings = TeacherSettings(nodes=1000, multipv=3, temperature_cp=50.0)
    label_rows(
        ring._arr["boards"], range(len(boards)), _fake_analyse, settings, tmp_path / "l.jsonl"
    )
    labels = read_labels([tmp_path / "l.jsonl"])
    meta = build_teacher_ring(
        ring, labels, list(range(len(boards))), tmp_path / "teacher", settings, {"source": "test"}
    )
    teacher = ReplayBuffer.open(tmp_path / "teacher", readonly=True)
    assert len(teacher) == len(boards) == meta["rows"]
    assert json.loads((tmp_path / "teacher" / "teacher.json").read_text())["source"] == "test"
    a, s = meta["outcome_model"]["a"], meta["outcome_model"]["s"]
    for k in (0, 17, 59):
        assert np.array_equal(teacher._arr["boards"][k], ring._arr["boards"][k])
        assert np.array_equal(teacher._arr["legals"][k], ring._arr["legals"][k])
        assert int(teacher._arr["outcomes"][k]) == z[k]
        assert int(teacher._arr["iteration"][k]) == TEACHER_ITERATION
        assert int(teacher._arr["mover_latest"][k]) == 0
        lines = [Line(m, c) for m, c in labels[k]["lines"]]
        want = dict(policy_from_lines(lines, 50.0))
        dense = teacher.densify_one(k)
        for uci, p in want.items():
            mv = chess.Move.from_uci(uci)
            assert dense[mv.from_square * 64 + mv.to_square] == pytest.approx(p, abs=2e-3)
        win, _, loss = outcome_probs(lines[0].cp, a, s)
        assert float(teacher._arr["root_q"][k]) == pytest.approx(win - loss, abs=2e-3)
    batch = teacher.sample(8, np.random.default_rng(0))
    assert bool(batch["root_q_mask"].all())


def test_a_phase2_iteration_trains_on_teacher_rows(tmp_path: Path) -> None:
    """The lever end to end: a mini run with a teacher ring takes its steps and logs them."""
    import math

    import torch

    from tinyalphazero import logschema
    from tinyalphazero.config import load_config
    from tinyalphazero.phase2 import Phase2Run

    rng = np.random.default_rng(3)
    boards = []
    for _ in range(40):
        b = chess.Board()
        for _ in range(int(rng.integers(1, 10))):
            b.push(list(b.legal_moves)[int(rng.integers(0, b.legal_moves.count()))])
        boards.append(b)
    ring = _ring_with(tmp_path / "src", boards, [0] * len(boards), [1, -1, 0, 1] * 10)
    settings = TeacherSettings(nodes=1000, multipv=3)
    label_rows(ring._arr["boards"], range(40), _fake_analyse, settings, tmp_path / "l.jsonl")
    build_teacher_ring(
        ring, read_labels([tmp_path / "l.jsonl"]), list(range(40)), tmp_path / "teacher",
        settings, {}, model=(60.0, 150.0),
    )  # fmt: skip

    cfg = load_config(str(REPO / "configs" / "mini.yaml"))
    cfg.phase2.iterations = 1
    cfg.train.teacher_frac = 0.25
    cfg.train.teacher_dir = str(tmp_path / "teacher")
    run = Phase2Run.create(tmp_path / "run", cfg, torch.device("cpu"))
    assert run.rehearsal.mix_for(cfg.train.batch_size).teacher == 16
    run.iterate()
    rows = logschema.read_jsonl(tmp_path / "run" / "iterations.jsonl")
    assert len(rows) == 1 and math.isfinite(rows[0]["loss"])


def test_the_labeller_excludes_every_measured_suite() -> None:
    """Every EPD the instrument reads is kept out of the teacher ring, including suites added later."""
    from scripts.label_teacher import EXCLUDE

    suites = {str(p.relative_to(REPO)) for p in (REPO / "tests" / "suites").glob("*.epd")}
    assert set(EXCLUDE) == suites


# --- pools across rings and agreement (DIAG-2026-11-13 capacity-or-data) ----------------


def _walks(n: int, seed: int) -> list[chess.Board]:
    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(n):
        b = chess.Board()
        for _ in range(int(rng.integers(2, 14))):
            b.push(list(b.legal_moves)[int(rng.integers(0, b.legal_moves.count()))])
        boards.append(b)
    return boards


def test_select_rows_shares_seen_across_calls(tmp_path: Path) -> None:
    e4 = chess.Board()
    e4.push_uci("e2e4")
    d4 = chess.Board()
    d4.push_uci("d2d4")
    a = _ring_with(tmp_path / "a", [e4, d4], [0, 0], [0, 0])
    b = _ring_with(
        tmp_path / "b", [d4, e4, chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")], [0] * 3, [0] * 3
    )
    seen: set[bytes] = set()
    rows_a, _ = select_rows(a._arr["boards"][:2], set(), seen=seen)
    rows_b, counts_b = select_rows(b._arr["boards"][:3], set(), seen=seen)
    assert rows_a.tolist() == [0, 1]
    assert rows_b.tolist() == [2] and counts_b["duplicate"] == 2
    # Without a shared set each ring stands alone, as before.
    assert select_rows(b._arr["boards"][:3], set())[0].tolist() == [0, 1, 2]


def test_pool_rows_dedupes_across_rings_and_every_prefix_is_the_same_order(tmp_path: Path) -> None:
    from tinyalphazero.teacher import pool_rows

    boards = _walks(50, 11)
    a = _ring_with(tmp_path / "a", boards[:30], [0] * 30, [0] * 30)
    b = _ring_with(tmp_path / "b", boards[20:], [0] * 30, [0] * 30)  # 10 positions shared
    excluded = {position_key(encode_board(boards[45], 0))}
    srcs = [a._arr["boards"][:30], b._arr["boards"][:30]]
    pairs, counts = pool_rows(srcs, excluded, seed=0)
    keys = [position_key(srcs[k][r]) for k, r in pairs]
    want = {position_key(encode_board(x, 0)) for x in boards} - excluded
    assert len(keys) == len(set(keys)) == len(want) and set(keys) == want
    assert counts[1]["duplicate"] >= 10 - 0 and counts[1]["excluded"] == 1
    again, _ = pool_rows(srcs, excluded, seed=0, limit=17)
    assert np.array_equal(again, pairs[:17])  # the limit is a prefix of the seeded order
    assert not np.array_equal(pool_rows(srcs, excluded, seed=1)[0], pairs)


def test_write_pool_copies_every_field_in_pair_order(tmp_path: Path) -> None:
    from tinyalphazero.teacher import write_pool

    boards = _walks(20, 5)
    a = _ring_with(tmp_path / "a", boards[:12], [0] * 12, [1, 0, -1] * 4)
    b = _ring_with(tmp_path / "b", boards[12:], [0] * 8, [-1, 1] * 4)
    pairs = np.array([[1, 3], [0, 0], [0, 11], [1, 7], [0, 5]], dtype=np.int64)
    out = write_pool([a, b], pairs, tmp_path / "pool", chunk=2)
    reopened = ReplayBuffer.open(tmp_path / "pool", readonly=True)
    assert len(reopened) == len(out) == 5
    for j, (k, r) in enumerate(pairs):
        src = (a, b)[k]
        for name in reopened._arr:
            assert np.array_equal(reopened._arr[name][j], src._arr[name][r]), (name, j)


def _teacher_ring(tmp: Path, boards: list[chess.Board], name: str) -> ReplayBuffer:
    src = _ring_with(
        tmp / f"{name}_src",
        boards,
        [0] * len(boards),
        [1, -1, 0] * (len(boards) // 3) + [0] * (len(boards) % 3),
    )
    settings = TeacherSettings(nodes=1000, multipv=3)
    label_rows(
        src._arr["boards"], range(len(boards)), _fake_analyse, settings, tmp / f"{name}.jsonl"
    )
    build_teacher_ring(
        src, read_labels([tmp / f"{name}.jsonl"]), list(range(len(boards))), tmp / name,
        settings, {}, model=(60.0, 150.0),
    )  # fmt: skip
    return ReplayBuffer.open(tmp / name, readonly=True)


def test_agreement_summary_is_the_scripts_arithmetic(tmp_path: Path) -> None:
    """The T1 numbers must not move: the summary is computed as the script always did."""
    import torch

    from tinyalphazero.config import ModelConfig
    from tinyalphazero.model import ChessNet, masked_policy, value_scalar
    from tinyalphazero.teacher import agreement
    from tinyalphazero.train import unpack_legal_mask

    ring = _teacher_ring(tmp_path, _walks(23, 2), "t")
    torch.manual_seed(0)
    net = ChessNet(
        ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, d_policy=16, dropout=0.0)
    ).eval()
    dev = torch.device("cpu")
    got = agreement(net, ring, dev, batch=8)

    arr, n = ring._arr, len(ring)
    top1, ce_sum, se_sum = 0, 0.0, 0.0
    for lo in range(0, n, 8):  # the loop scripts/teacher_agreement.py ran before
        idx = np.arange(lo, min(n, lo + 8))
        tokens = torch.from_numpy(arr["boards"][idx].astype(np.int64))
        legal = unpack_legal_mask(torch.from_numpy(np.asarray(arr["legals"][idx])))
        target = torch.from_numpy(np.stack([ring.densify_one(int(i)) for i in idx]))
        best = torch.from_numpy(arr["moves"][idx].astype(np.int64))
        with torch.no_grad():
            out = net(tokens)
            probs = masked_policy(out.policy_ft.float(), legal)
            value = value_scalar(out.value_wdl.float())
        top1 += int((probs.argmax(dim=1) == best[:, 0] * 64 + best[:, 1]).sum())
        ce_sum += float(-(target * probs.clamp_min(1e-12).log()).sum(dim=1).sum())
        root_q = torch.from_numpy(arr["root_q"][idx].astype(np.float32))
        se_sum += float(((value - root_q) ** 2).sum())
    assert got["top1_agreement"] == top1 / n
    assert got["teacher_policy_ce"] == ce_sum / n
    assert got["value_mse_vs_teacher"] == se_sum / n
    assert got["ce_rows"].shape == (n,) and got["ce_rows"].mean() == pytest.approx(
        ce_sum / n, rel=1e-6
    )
    assert np.all(got["entropy_rows"] <= got["ce_rows"] + 1e-6)  # Gibbs: CE >= the target's entropy
    sub = agreement(net, ring, dev, batch=8, rows=np.array([3, 9, 20]))
    assert sub["rows"] == 3 and np.allclose(sub["ce_rows"], got["ce_rows"][[3, 9, 20]])


def test_labelled_rows_reads_every_shard_and_skips_a_torn_line(tmp_path: Path) -> None:
    from tinyalphazero.teacher import labelled_rows

    (tmp_path / "shard_000.jsonl").write_text('{"row": 3, "lines": []}\n{"row": 9, "lines": []}\n')
    (tmp_path / "shard_001.jsonl").write_text('{"row": 4, "lines": []}\n{"row": 1')  # torn
    assert labelled_rows(sorted(tmp_path.glob("shard_*.jsonl"))) == {3, 4, 9}
