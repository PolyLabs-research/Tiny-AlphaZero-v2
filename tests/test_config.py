"""Tests for config round-trip and fingerprint stability."""

import tempfile
from pathlib import Path

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
