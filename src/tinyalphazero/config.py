"""Single source of truth for all TinyAlphaZero configuration.

All spec defaults are encoded here; every checkpoint embeds the full config
so results are reproducible from the config alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Transformer trunk + head parameters (spec §4.1)."""

    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    d_policy: int = 128  # bilinear projection dim for policy head
    dropout: float = 0.1  # 0.1 Phase 1, 0.0 Phase 2 — set via config, not code
    seq_len: int = 72
    aux_legality: bool = True  # legality auxiliary head; off in Phase 2


@dataclass
class MCTSConfig:
    """MCTS and search parameters (spec §6.2)."""

    sims: int = 48  # simulations per move during self-play
    root_mode: str = "gumbel"  # "gumbel" | "puct"
    # Gumbel / Sequential Halving
    gumbel_m: int = 16  # top-m root actions to consider
    c_visit: float = 50.0  # sigma scaling: (c_visit + max_N) * c_scale * q
    c_scale: float = 1.0
    # PUCT (ablation control)
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    temperature_ply_cutoff: int = 30  # move visit sampling: T=1.0 before, 0.1 after
    # Batched leaf evaluation
    batch_leaves: int = 512


@dataclass
class SelfPlayConfig:
    """Self-play game generation parameters (spec §7)."""

    games_per_iteration: int = 512
    concurrent_games: int = 256
    sims_per_move: int = 48
    max_plies: int = 160  # adjudicate as draw beyond this
    resign_threshold: float = -0.90
    resign_consecutive: int = 8  # own-move streak before resigning
    resign_disabled_frac: float = 0.10  # fraction of games with resignation disabled
    league_latest_frac: float = 0.80  # fraction of opponents from latest weights
    league_pool_size: int = 20  # snapshots to keep in the league pool


@dataclass
class TrainConfig:
    """Optimizer and loss parameters shared across phases (spec §5, §7)."""

    # Phase 2 defaults; Phase 1 script overrides lr/schedule via CLI
    batch_size: int = 1024
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    train_steps_per_iter: int = 400  # Phase 2 steps per iteration
    replay_capacity: int = 500_000
    # Loss weights
    policy_loss_weight: float = 1.0
    promo_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    legality_loss_weight: float = 1.0
    # Phase 1 specific
    phase1_lr_start: float = 3e-4
    phase1_lr_end: float = 3e-5
    phase1_epochs: int = 3
    phase1_value_weight_random: float = 0.25  # down-weight random-play value labels
    phase1_value_weight_anchor: float = 1.0
    phase1_mate1_mix: float = 0.05  # fraction of mate-in-1 anchors per batch
    # Masking the policy CE zeroes the gradient on illegal logits, so the trunk never
    # learns legality — and on uniform-random-play data the masked loss is already at
    # its optimum at init. Phase 1 therefore trains unmasked; Phase 2 (search-improved
    # targets over the legal set) sets this back to True.
    phase1_policy_masked: bool = False


@dataclass
class EvalConfig:
    """Anchor ladder evaluation parameters (spec §8)."""

    anchor_games: int = 200  # games per (checkpoint, anchor) match
    bootstrap_resamples: int = 10_000
    suite_sims: int = 128  # sims for mate-in-N suites
    eval_every_n_iters: int = 10  # how often to run the ladder during Phase 2
    openings_path: str = "tests/suites/openings_4ply.epd"
    stockfish_path: str = ""  # empty = skip Stockfish anchors if binary absent


@dataclass
class Config:
    """Root configuration container. All defaults match spec values."""

    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to plain dicts (handles nested)."""
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


def _dict_to_dataclass(cls: type, d: dict) -> Any:
    """Recursively reconstruct a dataclass from a plain dict."""
    if not hasattr(cls, "__dataclass_fields__"):
        return d
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in d:
            val = d[f.name]
            if hasattr(f.type, "__dataclass_fields__") or (
                isinstance(f.type, type) and issubclass(f.type, object)
            ):
                # Resolve forward-ref strings vs actual type objects
                target_type = f.type if isinstance(f.type, type) else type(None)
                if hasattr(target_type, "__dataclass_fields__") and isinstance(val, dict):
                    val = _dict_to_dataclass(target_type, val)
            kwargs[f.name] = val
    return cls(**kwargs)


def _resolve_field_types(cfg_dict: dict, cls: type) -> Any:
    """Reconstruct dataclass hierarchy from nested dicts, resolving types from annotations."""
    import typing

    hints = {}
    for c in reversed(cls.__mro__):
        if hasattr(c, "__annotations__"):
            hints.update(c.__annotations__)

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in cfg_dict:
            continue
        val = cfg_dict[f.name]
        # Resolve string annotations
        annotation = hints.get(f.name, f.type)
        if isinstance(annotation, str):
            annotation = eval(annotation, {**vars(typing), **globals()})  # noqa: S307
        if isinstance(val, dict) and isinstance(annotation, type) and hasattr(annotation, "__dataclass_fields__"):
            val = _resolve_field_types(val, annotation)
        kwargs[f.name] = val
    return cls(**kwargs)


def load_config(path: str | None = None, overrides: dict | None = None) -> Config:
    """Load a Config from a YAML file, optionally applying flat key overrides.

    If path is None, returns a Config with all spec defaults.
    overrides keys use dot-notation: e.g. {"model.n_layers": 6}.
    """
    if path is not None:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        raw = {}

    cfg = _resolve_field_types(raw, Config) if raw else Config()

    if overrides:
        cfg_dict = _dataclass_to_dict(cfg)
        for key, value in overrides.items():
            parts = key.split(".")
            target = cfg_dict
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        cfg = _resolve_field_types(cfg_dict, Config)

    return cfg


def save_config(cfg: Config, path: str) -> None:
    """Serialise a Config to YAML. Round-trips exactly with load_config."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(_dataclass_to_dict(cfg), fh, default_flow_style=False, sort_keys=True)


def config_fingerprint(cfg: Config) -> str:
    """SHA-256 of the canonical (sorted-key) YAML representation."""
    canonical = yaml.safe_dump(_dataclass_to_dict(cfg), default_flow_style=False, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
