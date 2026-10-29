"""Anchor rehearsal (p2_b's lever): mix ratio, schema, and reproducibility.

The lever's failure modes are all quiet ones. A mix that is not the configured
mix, a batch that trains fine but carries a malformed anchor target, or draws
that are not reproducible from (iteration, step) all produce a run that finishes
and reports numbers — the run just is not the experiment anyone designed. These
tests exist to make each of those loud.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tinyalphazero.config import Config, load_config
from tinyalphazero.rehearsal import ANCHOR_ITERATION, RehearsalSampler, plan_mix

REPO = Path(__file__).resolve().parents[1]
POLICY_DIM = 4096


# ---------------------------------------------------------------------------
# Fixtures: a synthetic anchor set, and a stand-in for the replay ring
# ---------------------------------------------------------------------------


def _write_dataset(path: Path, n: int, with_keymoves: bool = False) -> None:
    """A real on-disk Phase 1 dataset, small, in the real format.

    Row i is tagged in tokens 0 and 1 as (i // 8, i % 8) so a drawn row is
    identifiable in the batch — which is what lets the uniformity test see
    *which* rows the sampler took rather than only how many. Base 8 rather than
    the index itself because tokens are embedding indices: anything >= VOCAB_SIZE
    (43) is out of range for the model these batches are fed to.
    """
    path.mkdir(parents=True, exist_ok=True)
    boards = np.zeros((n, 72), dtype=np.int8)
    boards[:, 0] = np.arange(n) // 8
    boards[:, 1] = np.arange(n) % 8
    boards.tofile(path / "boards.dat")

    moves = np.zeros((n, 2), dtype=np.int16)
    moves[:, 0] = np.arange(n) % 64  # from
    moves[:, 1] = (np.arange(n) * 7 + 3) % 64  # to
    moves.tofile(path / "moves.dat")

    # Every fourth row promotes, so promo_mask has both states in any real draw.
    promos = np.where(np.arange(n) % 4 == 0, 4, 0).astype(np.int8)  # 4 = queen
    promos.tofile(path / "promos.dat")

    legals = np.full((n, 512), 0xFF, dtype=np.uint8)  # everything "legal"
    legals.tofile(path / "legals.dat")

    outcomes = np.ones(n, dtype=np.int8)  # mate anchors are all +1
    outcomes.tofile(path / "outcomes.dat")

    if with_keymoves:
        # Two key moves per row: the stored one and one other. Uniform over both
        # is the target the sampler must build.
        km = np.zeros((n, 512), dtype=np.uint8)
        for i in range(n):
            for cell in (int(moves[i, 0]) * 64 + int(moves[i, 1]), (i * 13 + 5) % POLICY_DIM):
                km[i, cell // 8] |= 1 << (cell % 8)
        km.tofile(path / "keymoves.dat")

    (path / "meta.json").write_text(
        json.dumps(
            {
                "n": n,
                "version": "1",
                "git_sha": "test",
                "dtype_map": {
                    "boards": "int8",
                    "moves": "int16",
                    "promos": "int8",
                    "legals": "uint8",
                    "outcomes": "int8",
                },
                "generation_config": {"mode": "synthetic"},
            }
        )
    )


class _FakeRing:
    """Stands in for ReplayBuffer.sample with the same keys and dtypes.

    A real ring would need self-play to fill it; what these tests are about is
    the mixing, so the ring only has to be schema-correct and identifiable
    (``iteration`` is 0 for ring rows, which anchors never are).
    """

    def __init__(self) -> None:
        self.last_n: int | None = None

    def sample(self, batch_size, rng, latest_only=False):  # noqa: ANN001, ARG002
        self.last_n = batch_size
        return {
            "tokens": torch.zeros((batch_size, 72), dtype=torch.long),
            "legals": torch.full((batch_size, 512), 255, dtype=torch.uint8),
            "outcome": torch.zeros(batch_size, dtype=torch.long),
            "policy": torch.full((batch_size, POLICY_DIM), 1.0 / POLICY_DIM),
            "promo_target": torch.zeros((batch_size, 4), dtype=torch.float32),
            "promo_mask": torch.zeros(batch_size, dtype=torch.bool),
            "from_sq": torch.zeros(batch_size, dtype=torch.long),
            "to_sq": torch.ones(batch_size, dtype=torch.long),
            "iteration": torch.zeros(batch_size, dtype=torch.long),
            "mover_is_latest": torch.ones(batch_size, dtype=torch.long),
            "root_q": torch.full((batch_size,), 0.5),
            "root_q_mask": torch.ones(batch_size, dtype=torch.bool),
        }


@pytest.fixture
def anchors(tmp_path: Path) -> Path:
    _write_dataset(tmp_path / "mate1", n=64)
    _write_dataset(tmp_path / "mate2", n=64, with_keymoves=True)
    return tmp_path


def _cfg_train(anchors: Path, frac1: float = 0.02, frac2: float = 0.03):
    cfg = Config()
    t = cfg.train
    t.rehearsal_frac = frac1 + frac2
    t.rehearsal_mate1_frac = frac1
    t.rehearsal_mate2_frac = frac2
    t.rehearsal_mate1_dir = str(anchors / "mate1")
    t.rehearsal_mate2_dir = str(anchors / "mate2")
    return t


# ---------------------------------------------------------------------------
# Mix ratio
# ---------------------------------------------------------------------------


def test_plan_mix_preserves_batch_size() -> None:
    """The optimiser must take identically-shaped steps with the lever on or off."""
    for bs in (1, 7, 64, 1024, 4096):
        for f1, f2 in ((0.0, 0.0), (0.02, 0.03), (0.1, 0.1), (0.5, 0.49)):
            mix = plan_mix(bs, f1, f2)
            assert mix.total == bs
            assert mix.ring >= 0 and mix.mate1 >= 0 and mix.mate2 >= 0


def test_mix_ratio_is_the_configured_ratio(anchors: Path) -> None:
    """The brief's 0.05 = 0.02/0.03 must be what the batch actually contains."""
    sampler = RehearsalSampler(_FakeRing(), _cfg_train(anchors), seed=42)
    batch = sampler.sample(1024, np.random.default_rng(0), iteration=1, step=0)

    assert batch["tokens"].shape[0] == 1024
    n_anchor = int((batch["iteration"] == ANCHOR_ITERATION).sum())
    assert n_anchor == round(1024 * 0.02) + round(1024 * 0.03) == 51
    assert int((batch["iteration"] == 0).sum()) == 1024 - 51


def test_anchor_draws_are_uniform_over_the_set(anchors: Path) -> None:
    """chi-square goodness-of-fit on which anchor rows get drawn.

    A sampler biased toward the front of the memmap would rehearse a fixed
    thousandth of the anchor set for the whole campaign and still look right in
    every ratio check.
    """
    n_rows = 64
    sampler = RehearsalSampler(_FakeRing(), _cfg_train(anchors, 0.5, 0.0), seed=42)
    ring = np.random.default_rng(0)

    counts = np.zeros(n_rows, dtype=float)
    draws = 0
    for step in range(200):  # 200 * 50 = 10,000 anchor rows
        batch = sampler.sample(100, ring, iteration=3, step=step)
        anchor_rows = batch["tokens"][batch["iteration"] == ANCHOR_ITERATION]
        for hi, lo in anchor_rows[:, :2].tolist():
            counts[int(hi) * 8 + int(lo)] += 1
            draws += 1

    assert draws == 10_000
    expected = draws / n_rows
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    # df = 63; the 99.9th percentile is ~103.4. A uniform sampler clears this
    # essentially always, and a front-biased one fails by orders of magnitude.
    assert chi2 < 103.4, f"anchor draws are not uniform over the set: chi2={chi2:.1f}"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_mixed_batch_trains(anchors: Path) -> None:
    """One mixed batch through the real train_step_phase2, on CPU."""
    from tinyalphazero.model import ChessNet
    from tinyalphazero.train import train_step_phase2

    cfg = Config()
    cfg.model.n_layers, cfg.model.d_model, cfg.model.d_ff, cfg.model.n_heads = 1, 32, 64, 2
    net = ChessNet(cfg.model)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)

    sampler = RehearsalSampler(_FakeRing(), _cfg_train(anchors, 0.25, 0.25), seed=42)
    batch = sampler.sample(16, np.random.default_rng(0), iteration=1, step=0)
    out = train_step_phase2(batch, net, opt, torch.device("cpu"), cfg.train, masked_policy=True)

    assert not out["skipped"]
    for key in ("loss", "policy_loss", "value_loss"):
        assert np.isfinite(out[key]), f"{key} is not finite on a mixed batch"


def test_anchor_policy_targets_are_normalised(anchors: Path) -> None:
    """Every anchor row's policy target is a distribution, and a correct one.

    mate-in-1 has no keymoves: one-hot on the stored move. mate-in-2 does: mass
    spread uniformly over the key set, because the stored move is an arbitrary
    tie-break among several moves that all force mate, and training toward the
    tie-break would fight the coverage objective the lever exists to fix.
    """
    for frac1, frac2, expected_support in ((0.5, 0.0, 1), (0.0, 0.5, 2)):
        sampler = RehearsalSampler(_FakeRing(), _cfg_train(anchors, frac1, frac2), seed=42)
        batch = sampler.sample(32, np.random.default_rng(0), iteration=1, step=0)
        rows = batch["policy"][batch["iteration"] == ANCHOR_ITERATION]
        assert rows.shape[0] > 0
        torch.testing.assert_close(rows.sum(dim=1), torch.ones(rows.shape[0]))
        support = (rows > 0).sum(dim=1)
        assert torch.all(support == expected_support), (
            f"expected {expected_support} supported move(s) per anchor row, got "
            f"{sorted(set(support.tolist()))}"
        )


def test_promo_targets_follow_the_replay_convention(anchors: Path) -> None:
    """promo_mask marks promoting rows; the target is one-hot at class-1."""
    sampler = RehearsalSampler(_FakeRing(), _cfg_train(anchors, 0.5, 0.0), seed=42)
    batch = sampler.sample(64, np.random.default_rng(0), iteration=1, step=0)
    is_anchor = batch["iteration"] == ANCHOR_ITERATION
    masked = batch["promo_mask"][is_anchor]
    targets = batch["promo_target"][is_anchor]
    assert masked.any(), "the synthetic set promotes every fourth row; none were drawn"
    torch.testing.assert_close(targets[masked].sum(dim=1), torch.ones(int(masked.sum())))
    assert torch.all(targets[~masked] == 0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_anchor_draws_are_seeded_per_iteration_and_step(anchors: Path) -> None:
    """(iteration, step) determines the anchor rows, independent of ring history.

    Reproducing iteration 12 step 300 must not require replaying the 4,700 ring
    draws before it — the same property §9.2 gives self-play games.
    """
    train = _cfg_train(anchors)
    a = RehearsalSampler(_FakeRing(), train, seed=42)
    b = RehearsalSampler(_FakeRing(), train, seed=42)

    # Different ring generators, and b is asked out of order: only (iteration,
    # step) should matter to the anchor half.
    ba = a.sample(1024, np.random.default_rng(1), iteration=12, step=300)
    for step in (7, 99, 300):
        bb = b.sample(1024, np.random.default_rng(999), iteration=12, step=step)
        rows_a = ba["tokens"][ba["iteration"] == ANCHOR_ITERATION]
        rows_b = bb["tokens"][bb["iteration"] == ANCHOR_ITERATION]
        if step == 300:
            assert torch.equal(rows_a, rows_b), "same (iteration, step) drew different anchors"
        else:
            assert not torch.equal(rows_a, rows_b), "different steps drew identical anchors"

    c = RehearsalSampler(_FakeRing(), train, seed=43)
    bc = c.sample(1024, np.random.default_rng(1), iteration=12, step=300)
    rows_c = bc["tokens"][bc["iteration"] == ANCHOR_ITERATION]
    assert not torch.equal(ba["tokens"][ba["iteration"] == ANCHOR_ITERATION], rows_c), (
        "the run seed does not reach the anchor draw"
    )


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------


def test_disabled_is_a_pass_through() -> None:
    """With no rehearsal configured the ring batch is returned untouched.

    This is what keeps every run predating the lever bit-identical: the wrapper
    must not perturb the ring's draw sequence merely by existing.
    """
    ring = _FakeRing()
    sampler = RehearsalSampler(ring, Config().train, seed=42)
    assert not sampler.enabled
    batch = sampler.sample(256, np.random.default_rng(0), iteration=1, step=0)
    assert ring.last_n == 256
    assert int((batch["iteration"] == ANCHOR_ITERATION).sum()) == 0


def test_missing_anchor_dir_raises_rather_than_running_the_control(tmp_path: Path) -> None:
    """A configured lever whose data is absent must not silently become a no-op."""
    t = Config().train
    t.rehearsal_frac = t.rehearsal_mate1_frac = 0.05
    t.rehearsal_mate1_dir = str(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError, match="anchor set is missing"):
        RehearsalSampler(_FakeRing(), t, seed=42)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_split_must_agree_with_total(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        "train:\n  rehearsal_frac: 0.05\n"
        "  rehearsal_mate1_frac: 0.02\n  rehearsal_mate2_frac: 0.02\n"
    )
    with pytest.raises(ValueError, match="must agree"):
        load_config(str(cfg_path))


def _flat(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def _config_deltas(a_path: Path, b_path: Path, a: str, b: str, allowed: set[str]) -> list[str]:
    """Every field that differs between two configs, *including* top-level scalars.

    The previous comparison iterated the dict-valued sections only, so `seed`
    sat outside the "zero deltas" assertion: a p2_d with `seed: 43` against
    p2_c's 42 would have read as the null lever while re-seeding self-play, the
    opening draw and the bootstrap (REVIEW-2026-10-21 P11).
    """
    import dataclasses

    fa = _flat(dataclasses.asdict(load_config(str(a_path))))
    fb = _flat(dataclasses.asdict(load_config(str(b_path))))
    assert fa.keys() == fb.keys()
    return [
        f"{key}: {a}={fa[key]} {b}={fb[key]}"
        for key in fa
        if fa[key] != fb[key] and key.rsplit(".", 1)[-1] not in allowed
    ]


def test_config_deltas_see_the_top_level_seed(tmp_path: Path) -> None:
    """The guard the three brief tests share must notice a changed `seed`."""
    from tinyalphazero.config import Config, save_config

    base = Config()
    save_config(base, str(tmp_path / "a.yaml"))
    base.seed = 43
    save_config(base, str(tmp_path / "b.yaml"))
    assert _config_deltas(tmp_path / "a.yaml", tmp_path / "b.yaml", "a", "b", set()) == [
        "seed: a=42 b=43"
    ]


def test_p2d_config_is_the_brief() -> None:
    """configs/p2_d.yaml differs from p2_c's resolved config in NOTHING.

    Track 2 is the null lever: twenty more iterations at an identical
    configuration, asking whether the flywheel continues with nothing added. The
    claim "zero new levers" has exactly one form of evidence — that zero fields
    moved — and an accidental delta would turn a control into an uncontrolled
    experiment while still looking like a clean result.
    """
    import dataclasses

    p2d = REPO / "configs" / "p2_d.yaml"
    p2c = REPO / "runs" / "p2_c" / "config.yaml"
    if not (p2d.exists() and p2c.exists()):
        pytest.skip("p2_d config or p2_c run config not present")

    b = dataclasses.asdict(load_config(str(p2d)))
    differences = _config_deltas(p2c, p2d, "p2_c", "p2_d", allowed=set())
    assert not differences, "the null lever is not null:\n  " + "\n  ".join(differences)
    # The two standing settings and the flag Track 1 adds, asserted by name so a
    # future default change cannot silently redefine "identical".
    assert b["train"]["value_q_mse_weight"] == 0.5
    assert b["train"]["rehearsal_frac"] == 0.05
    assert b["mcts"].get("proof_directed_expansion", False) is False


def test_p2e_config_is_the_brief() -> None:
    """configs/p2_e.yaml resolves to exactly p2_d's run config: the null re-baseline.

    p2_e is the first round on the REVIEW-2026-10-21 code, which changed what a
    fixed config trains (C4) and searches (A1-A6, D15). Its claim is "the code
    moved and nothing else did", and an accidental config delta would leave that
    claim unreadable while still producing a clean-looking run.
    """
    import dataclasses

    p2e = REPO / "configs" / "p2_e.yaml"
    p2d = REPO / "runs" / "p2_d" / "config.yaml"
    if not (p2e.exists() and p2d.exists()):
        pytest.skip("p2_e config or p2_d run config not present")

    differences = _config_deltas(p2d, p2e, "p2_d", "p2_e", allowed=set())
    assert not differences, "the null re-baseline is not null:\n  " + "\n  ".join(differences)
    # The fields the fix pass added, asserted by name: each defaults to the legacy
    # behaviour today, and a future default change must not redefine this round.
    b = dataclasses.asdict(load_config(str(p2e)))
    assert b["train"]["value_q_blend"] == "batch_mean"
    assert b["model"]["attn_init"] == "torch_default"
    assert b["train"]["phase1_anchor_weighting"] == "legacy"
    assert b["mcts"]["proof_directed_expansion"] is False
    raw = (REPO / "configs" / "p2_e.yaml").read_text()
    for pinned in (
        "value_q_blend:",
        "attn_init:",
        "phase1_anchor_weighting:",
        "proof_directed_expansion:",
    ):
        assert f"  {pinned}" in raw, f"{pinned} must be written in p2_e.yaml, not inherited"


def test_p2f_config_is_the_brief() -> None:
    """configs/p2_f.yaml resolves to exactly p2_e's run config: the lever is a flag.

    p2_f's one change is `--seed-replay runs/p2_e` at launch, recorded in its
    provenance.json. A config delta on top of it would make the ring carry
    unattributable while the run still looked clean. Compared against p2_e's
    *resolved* run config, which carries the ladder cadence and rungs p2_e ran.
    """
    p2f = REPO / "configs" / "p2_f.yaml"
    p2e = REPO / "runs" / "p2_e" / "config.yaml"
    if not (p2f.exists() and p2e.exists()):
        pytest.skip("p2_f config or p2_e run config not present")

    differences = _config_deltas(p2e, p2f, "p2_e", "p2_f", allowed=set())
    assert not differences, "p2_f moves more than the ring:\n  " + "\n  ".join(differences)


def test_p2g_config_is_the_brief() -> None:
    """configs/p2_g.yaml is p2_f's run config plus the lever and three run settings.

    The lever is ``opening_random_plies``. ``actor_processes`` moves wall clock only
    (tests/test_actors.py), ``eval_every_n_iters: 0`` moves the ladder off the pod to
    the local instrument, and ``iterations`` is the protocol's length. Anything else
    moving would be a second lever.
    """
    import dataclasses

    p2g = REPO / "configs" / "p2_g.yaml"
    p2f = REPO / "runs" / "p2_f" / "config.yaml"
    if not (p2g.exists() and p2f.exists()):
        pytest.skip("p2_g config or p2_f run config not present")

    allowed = {"opening_random_plies", "actor_processes", "eval_every_n_iters", "iterations"}
    differences = _config_deltas(p2f, p2g, "p2_f", "p2_g", allowed=allowed)
    assert not differences, "p2_g moves more than registered:\n  " + "\n  ".join(differences)
    g = dataclasses.asdict(load_config(str(p2g)))
    assert g["selfplay"]["opening_random_plies"] == 8
    assert g["selfplay"]["opening_exclude"] == "tests/suites/openings_4ply.epd"
    assert g["selfplay"]["actor_processes"] == 20
    assert g["eval"]["eval_every_n_iters"] == 0
    assert g["phase2"]["iterations"] == 60


def test_p2i_row_config_is_the_brief() -> None:
    """configs/p2_i_row.yaml is configs/p2_g.yaml with ``value_q_blend: per_row`` and nothing else (PREREG-p2_i).

    The continuation arm runs configs/p2_g.yaml unchanged, so the A/B's two arms differ in this one field; the round
    length is the invocation's ``--iterations``, not the file's.
    """
    import dataclasses

    p2g = REPO / "configs" / "p2_g.yaml"
    p2i = REPO / "configs" / "p2_i_row.yaml"
    if not (p2g.exists() and p2i.exists()):
        pytest.skip("p2_g or p2_i_row config not present")
    differences = _config_deltas(p2g, p2i, "p2_g", "p2_i_row", allowed={"value_q_blend"})
    assert not differences, "p2_i_row moves more than registered:\n  " + "\n  ".join(differences)
    row = dataclasses.asdict(load_config(str(p2i)))
    base = dataclasses.asdict(load_config(str(p2g)))
    assert (
        base["train"]["value_q_blend"] == "batch_mean"
        and row["train"]["value_q_blend"] == "per_row"
    )
    assert row["train"]["value_q_mse_weight"] == base["train"]["value_q_mse_weight"] == 0.5
    assert row["train"]["rehearsal_frac"] == base["train"]["rehearsal_frac"] == 0.05


def test_p2k_lr_config_is_the_brief() -> None:
    """configs/p2_k_lr.yaml is configs/p2_g.yaml with ``lr`` stepped down tenfold and nothing else (PREREG-p2_k).

    The continuation arm runs configs/p2_g.yaml unchanged, so the A/B's two arms differ in this one field; the
    phase 1 schedule's fields (``phase1_lr_start``/``_end``) are other leaves and must not move.
    """
    import dataclasses

    p2g = REPO / "configs" / "p2_g.yaml"
    p2k = REPO / "configs" / "p2_k_lr.yaml"
    if not (p2g.exists() and p2k.exists()):
        pytest.skip("p2_g or p2_k_lr config not present")
    differences = _config_deltas(p2g, p2k, "p2_g", "p2_k_lr", allowed={"lr"})
    assert not differences, "p2_k_lr moves more than registered:\n  " + "\n  ".join(differences)
    lr = dataclasses.asdict(load_config(str(p2k)))
    base = dataclasses.asdict(load_config(str(p2g)))
    assert base["train"]["lr"] == 2e-4 and lr["train"]["lr"] == 2e-5
    assert lr["train"]["weight_decay"] == base["train"]["weight_decay"]


def test_p2c_config_is_the_brief() -> None:
    """configs/p2_c.yaml differs from p2_b's resolved config in exactly one field.

    Same guard as p2_b's, one round on: the claim "one lever" has no evidence
    except that nothing else moved, and rehearsal staying at 0.05 is part of what
    must be verified — it is standing configuration this round, not a lever.
    """
    import dataclasses

    p2c = REPO / "configs" / "p2_c.yaml"
    p2b = REPO / "runs" / "p2_b" / "config.yaml"
    if not (p2c.exists() and p2b.exists()):
        pytest.skip("p2_c config or p2_b run config not present")

    b = dataclasses.asdict(load_config(str(p2c)))
    differences = _config_deltas(p2b, p2c, "p2_b", "p2_c", allowed={"value_q_mse_weight"})
    assert not differences, "p2_c changes more than the lever:\n  " + "\n  ".join(differences)
    assert b["train"]["value_q_mse_weight"] == 0.5
    assert b["train"]["rehearsal_frac"] == 0.05, "rehearsal must stay at p2_b's setting"


def test_p2b_config_is_the_brief(tmp_path: Path) -> None:
    """configs/p2_b.yaml is the lever and nothing else.

    Guards the experiment's design, not the code: p2_b's claim is "one lever",
    and the only evidence for that claim is that no other field differs from the
    campaign p2_a actually ran.
    """
    p2b_path = REPO / "configs" / "p2_b.yaml"
    if not p2b_path.exists():
        pytest.skip("configs/p2_b.yaml not present")
    p2a_path = REPO / "runs" / "p2_a" / "config.yaml"
    if not p2a_path.exists():
        pytest.skip("runs/p2_a/config.yaml not present")

    allowed = {"rehearsal_frac", "rehearsal_mate1_frac", "rehearsal_mate2_frac"}
    differences = _config_deltas(p2a_path, p2b_path, "p2_a", "p2_b", allowed=allowed)
    assert not differences, "p2_b changes more than the lever:\n  " + "\n  ".join(differences)


def test_rehearsal_rows_declare_they_have_no_search_q(anchors: Path) -> None:
    """Anchor rows carry root_q_mask False: p2_c's blend must not touch them.

    A rehearsal row never had a tree. Blending it against a fabricated 0.0 would
    teach "this position was a draw" on positions that are all forced wins — the
    exact inversion of what the anchor sets are for.
    """
    sampler = RehearsalSampler(_FakeRing(), _cfg_train(anchors), seed=42)
    batch = sampler.sample(1024, np.random.default_rng(0), iteration=1, step=0)
    is_anchor = batch["iteration"] == ANCHOR_ITERATION
    assert not bool(batch["root_q_mask"][is_anchor].any()), "anchor rows claimed a search-Q"
    assert bool(batch["root_q_mask"][~is_anchor].all()), "ring rows lost their search-Q"
