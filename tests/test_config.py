"""Tests for config round-trip and fingerprint stability."""

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from tinyalphazero.config import (
    Config,
    EvalConfig,
    MCTSConfig,
    ModelConfig,
    SelfPlayConfig,
    TrainConfig,
    _dataclass_to_dict,
    config_fingerprint,
    load_config,
    save_config,
)

REPO = Path(__file__).resolve().parents[1]


def test_default_config_instantiates() -> None:
    cfg = Config()
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.mcts, MCTSConfig)
    assert isinstance(cfg.selfplay, SelfPlayConfig)
    assert isinstance(cfg.train, TrainConfig)
    assert isinstance(cfg.eval, EvalConfig)


def test_spec_defaults_model() -> None:
    m = ModelConfig()
    assert m.d_model == 256
    assert m.n_layers == 8
    assert m.n_heads == 8
    assert m.d_ff == 1024
    assert m.d_policy == 128
    assert m.seq_len == 72
    assert m.dropout == 0.1


def test_spec_defaults_mcts() -> None:
    m = MCTSConfig()
    assert m.sims == 48
    assert m.root_mode == "gumbel"
    assert m.gumbel_m == 16
    assert m.batch_leaves == 512


def test_spec_defaults_selfplay() -> None:
    s = SelfPlayConfig()
    assert s.games_per_iteration == 512
    assert s.concurrent_games == 256
    assert s.max_plies == 160
    assert s.resign_threshold == -0.90
    assert s.resign_consecutive == 8
    assert s.resign_disabled_frac == 0.10
    assert s.league_latest_frac == 0.80
    assert s.league_pool_size == 20


def test_spec_defaults_train() -> None:
    t = TrainConfig()
    assert t.batch_size == 1024
    assert t.lr == 2e-4
    assert t.weight_decay == 1e-4
    assert t.grad_clip == 1.0
    assert t.train_steps_per_iter == 400
    assert t.replay_capacity == 500_000


def test_yaml_round_trip() -> None:
    cfg = Config()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "config.yaml")
        save_config(cfg, path)
        cfg2 = load_config(path)
    assert cfg == cfg2


def test_yaml_round_trip_modified() -> None:
    cfg = Config()
    cfg.model.n_layers = 6
    cfg.mcts.sims = 32
    cfg.selfplay.max_plies = 200
    cfg.train.lr = 1e-3
    cfg.seed = 7

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "cfg.yaml")
        save_config(cfg, path)
        cfg2 = load_config(path)

    assert cfg == cfg2
    assert cfg2.model.n_layers == 6
    assert cfg2.mcts.sims == 32
    assert cfg2.selfplay.max_plies == 200
    assert cfg2.train.lr == 1e-3
    assert cfg2.seed == 7


def test_load_config_no_path_returns_defaults() -> None:
    cfg = load_config()
    assert cfg == Config()


def test_load_config_with_overrides() -> None:
    cfg = load_config(overrides={"model.n_layers": 4, "seed": 99})
    assert cfg.model.n_layers == 4
    assert cfg.seed == 99
    # Other fields untouched
    assert cfg.model.d_model == 256


def test_fingerprint_is_stable() -> None:
    cfg = Config()
    fp1 = config_fingerprint(cfg)
    fp2 = config_fingerprint(cfg)
    assert fp1 == fp2
    assert len(fp1) == 64  # sha256 hex


def test_fingerprint_changes_on_mutation() -> None:
    cfg1 = Config()
    cfg2 = Config()
    cfg2.model.n_layers = 6
    assert config_fingerprint(cfg1) != config_fingerprint(cfg2)


def test_fingerprint_is_deterministic_across_instances() -> None:
    fp1 = config_fingerprint(Config())
    fp2 = config_fingerprint(Config())
    assert fp1 == fp2


def test_dataclass_to_dict_nested() -> None:
    cfg = Config()
    d = _dataclass_to_dict(cfg)
    assert isinstance(d, dict)
    assert isinstance(d["model"], dict)
    assert d["model"]["d_model"] == 256


def test_save_creates_parent_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "subdir" / "config.yaml")
        save_config(Config(), path)
        assert Path(path).exists()


def test_load_config_overrides_on_file() -> None:
    cfg_base = Config()
    cfg_base.model.n_layers = 6
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "cfg.yaml")
        save_config(cfg_base, path)
        cfg = load_config(path, overrides={"model.n_layers": 10})
    assert cfg.model.n_layers == 10


# ---------------------------------------------------------------------------
# REVIEW-2026-10-21: the fingerprint contract, type checking, override typos,
# the eval-profile scoping of the Track 1 lever, and which configs opt in.
# ---------------------------------------------------------------------------


def test_fingerprint_is_the_sha256_of_the_saved_file() -> None:
    """The durable identity: `sha256sum config.yaml` == the recorded fingerprint.

    Re-loading under newer code is not reliable — adding a field with a default
    changes the dump of every unchanged YAML (REVIEW-2026-10-21 D11) — so the
    contract is pinned on the bytes save_config writes, and checked against the
    four committed campaigns' provenance records.
    """
    cfg = Config()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        save_config(cfg, str(path))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config_fingerprint(cfg)

    for run in ("p2_a", "p2_b", "p2_c", "p2_d"):
        recorded = json.loads((REPO / "runs" / run / "provenance.json").read_text())
        on_disk = hashlib.sha256((REPO / "runs" / run / "config.yaml").read_bytes()).hexdigest()
        assert on_disk == recorded["config_fingerprint"], run


def _write(tmp: str, text: str) -> str:
    path = Path(tmp) / "cfg.yaml"
    path.write_text(text)
    return str(path)


def test_load_config_rejects_values_of_the_wrong_type() -> None:
    """PyYAML's parse is not the dataclass's type (REVIEW-2026-10-21 D9).

    `'false'` is a truthy string that left root noise on while `resolved()` thought
    it was off; `1e-4` is a YAML 1.1 *string*; `mcts: 5` replaced a whole section.
    """
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(TypeError, match="root_noise.*bool"):
            load_config(_write(tmp, "mcts:\n  root_noise: 'false'\n"))
        with pytest.raises(TypeError, match=r"1\.0e-4"):
            load_config(_write(tmp, "train:\n  phase1_lr_start: 1e-4\n"))
        with pytest.raises(TypeError, match="section 'mcts'"):
            load_config(_write(tmp, "mcts: 5\n"))
        with pytest.raises(TypeError, match="n_layers.*int"):
            load_config(_write(tmp, "model:\n  n_layers: 6.5\n"))
        with pytest.raises(TypeError, match="n_layers.*int"):
            load_config(_write(tmp, "model:\n  n_layers: true\n"))
        # The one lossless coercion: an int where a float is declared.
        cfg = load_config(_write(tmp, "mcts:\n  c_scale: 1\n"))
        assert cfg.mcts.c_scale == 1.0 and isinstance(cfg.mcts.c_scale, float)
        # Optional fields accept None and the inner type.
        cfg = load_config(_write(tmp, "mcts:\n  eval_m: null\neval:\n  eval_seed: 3\n"))
        assert cfg.mcts.eval_m is None and cfg.eval.eval_seed == 3


def test_override_with_an_unknown_key_raises() -> None:
    """A typo in an override must not silently leave the default (REVIEW-2026-10-21 D10)."""
    with pytest.raises(ValueError, match="n_layer"):
        load_config(None, overrides={"model.n_layer": 4})
    with pytest.raises(ValueError, match="leaf"):
        load_config(None, overrides={"model.n_layers.deep": 4})
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(None, overrides={"modle.n_layers": 4})


def test_eval_profile_scopes_the_proof_directed_lever() -> None:
    """Measurement-side switch, like eval_m: never reaches self-play (REVIEW-2026-10-21 P2)."""
    m = MCTSConfig(eval_proof_directed_expansion=True)
    assert m.resolved().proof_directed_expansion is False, "self-play profile is untouched"
    assert replace(m, root_noise=False).resolved().proof_directed_expansion is True
    inherit = MCTSConfig(proof_directed_expansion=True, root_noise=False)
    assert inherit.resolved().proof_directed_expansion is True
    off = MCTSConfig(proof_directed_expansion=True, eval_proof_directed_expansion=False)
    assert replace(off, root_noise=False).resolved().proof_directed_expansion is False
    r = replace(m, root_noise=False).resolved()
    assert r.resolved() == r, "idempotent"


def test_corrected_behaviours_are_opt_in_and_the_record_stays_legacy() -> None:
    """Committed configs still describe what they ran; new configs opt in.

    The three training-affecting fixes (B3 attention init, E1 anchor weighting, C1
    blend weighting) default to the legacy behaviour so every checkpoint's embedded
    config and every runs/*/config.yaml means what it meant; configs/default.yaml
    carries the corrected values for the next round.
    """
    legacy = ("torch_default", "legacy", "batch_mean")
    for path in (None, "configs/mini.yaml", "runs/p2_c/config.yaml", "runs/p2_d/config.yaml"):
        cfg = load_config(None if path is None else str(REPO / path))
        got = (cfg.model.attn_init, cfg.train.phase1_anchor_weighting, cfg.train.value_q_blend)
        assert got == legacy, (path, got)
    cfg = load_config(str(REPO / "configs" / "default.yaml"))
    assert (cfg.model.attn_init, cfg.train.phase1_anchor_weighting, cfg.train.value_q_blend) == (
        "trunc_normal",
        "exact",
        "per_row",
    )
    # The anchor pin is the v2 digest from runs/ANCHORS.sha256, and the mini
    # config disables both the pin and the ladder.
    assert EvalConfig().anchor_digest.startswith("27dcdd05")
    mini = load_config(str(REPO / "configs" / "mini.yaml"))
    assert mini.eval.anchor_digest == "" and mini.eval.eval_every_n_iters == 0


def _flat(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def test_puct_ablation_is_the_brief() -> None:
    """`configs/puct_ablation.yaml` differs from `configs/default.yaml` in root_mode only.

    It had drifted (dropout 0.1, legality_loss_weight 1.0 from before default.yaml
    was retargeted), which would have confounded the M4 Gumbel-vs-PUCT comparison
    with two unrelated changes (REVIEW-2026-10-21 D8).
    """
    base = _flat(_dataclass_to_dict(load_config(str(REPO / "configs" / "default.yaml"))))
    puct = _flat(_dataclass_to_dict(load_config(str(REPO / "configs" / "puct_ablation.yaml"))))
    assert {k for k in base if base[k] != puct[k]} == {"mcts.root_mode"}
    assert puct["mcts.root_mode"] == "puct"
