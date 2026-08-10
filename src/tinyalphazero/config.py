"""Single source of truth for all TinyAlphaZero configuration.

All spec defaults are encoded here; every checkpoint embeds the full config
so results are reproducible from the config alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields, replace
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
    # --- evaluation profile (BLOCKED-2026-08-08) ------------------------
    # Gumbel's root draw exists to diversify *self-play data*. Carrying it into
    # a suite, a ladder rung, or a match measures exploration noise on a
    # match-play question: a forcing move the prior ranks first can still be
    # knocked out of the considered set by an unlucky g. Deterministic acting at
    # evaluation is standard Gumbel-AZ practice (Danihelka et al. 2022 §5), and
    # chunk 10's ladder needs it regardless -- rungs that move by +/-2.5 points
    # under reseeding cannot resolve Elo.
    #
    # Rule: suites, ladder, and match play run the eval profile; self-play runs
    # noise. The default here is the *self-play* value, so anything that forgets
    # to choose keeps generating diverse data rather than silently freezing it.
    root_noise: bool = True
    # Split a promotion cell's prior by the promo head's distribution instead of
    # equally over the four pieces. OFF for Phase 1 checkpoints, ON from Phase 2.
    #
    # Not a preference — measured. Phase 1 trains on random play, where the four
    # promotion pieces are sampled uniformly, so the head has no promotion signal
    # and its deviations from uniform are fitted noise. On `phase1_anchor` it
    # ranks QUEEN last (0.165–0.217 vs 0.25 uniform) and turning this on costs
    # 2.5 points on the chunk 6 gate: 0.985 → 0.960, five new failures, zero
    # fixed, and on every one of the five the key move is a queen promotion.
    # Equal division is the correct prior for a head that knows nothing.
    #
    # Flip to True once self-play has trained the head on its own promotion
    # choices, and re-measure the gate when doing so.
    promo_prior_from_head: bool = False
    # Overrides applied only when root_noise is False. None means "inherit the
    # self-play value". Scoped deliberately: retuning search constants for a
    # measurement must not retune the constants that generate training data --
    # that is Phase 2 evidence territory, not a gate conversation.
    eval_m: int | None = None
    eval_c_visit: float | None = None
    eval_c_scale: float | None = None
    # PUCT (ablation control)
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    temperature_ply_cutoff: int = 30  # move visit sampling: T=1.0 before, 0.1 after
    # Batched leaf evaluation
    batch_leaves: int = 512

    def resolved(self) -> MCTSConfig:
        """The config search actually runs with, after applying the eval profile.

        Folding ``eval_*`` down onto ``gumbel_m`` / ``c_visit`` / ``c_scale``
        here — rather than teaching ``sigma()`` and ``improved_policy()`` to pick
        a branch — keeps exactly one definition of each constant in the search
        hot path. Two call sites consuming σ under two different rules is the
        shape of bug that produced ``IMMEDIATE_WIN_BONUS``.

        Idempotent: ``cfg.resolved().resolved() == cfg.resolved()``.
        """
        if self.root_noise:
            return self
        return replace(
            self,
            gumbel_m=self.gumbel_m if self.eval_m is None else self.eval_m,
            c_visit=self.c_visit if self.eval_c_visit is None else self.eval_c_visit,
            c_scale=self.c_scale if self.eval_c_scale is None else self.eval_c_scale,
        )


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
    phase1_mate2_mix: float = 0.0  # fraction of mate-in-2 anchors per batch (round d+)
    # Masking the policy CE zeroes the gradient on illegal logits, so the trunk never
    # learns legality — and on uniform-random-play data the masked loss is already at
    # its optimum at init. Phase 1 therefore trains unmasked; Phase 2 (search-improved
    # targets over the legal set) sets this back to True.
    phase1_policy_masked: bool = False


@dataclass
class Phase2Config:
    """Loop-level knobs for the Phase 2 orchestrator (spec §7, chunk 9).

    **What is deliberately *not* here.** Every per-iteration number in the §7
    table already has an owner: games/concurrency/sims/ply-cap/resignation/league
    live in ``selfplay``, optimiser and replay in ``train``, search constants in
    ``mcts``. Restating them under ``phase2`` would put two writable spellings of
    one number in the same file, which is the duplication rule in AGENTS.md §5
    and exactly how a run ends up honouring the copy nobody edited.
    ``configs/default.yaml`` carries the table verbatim as a comment with the
    owning key on each row, so the table is readable in one place and settable in
    exactly one.
    """

    iterations: int = 1000  # campaign length; --iterations overrides
    # Train on positions whose mover was a pool opponent rather than the latest
    # weights. Named default, pre-registered as the first ablation if M3 Elo
    # stalls: a search-improved target is a valid improvement target regardless of
    # which weights ran the search — it is merely ~20% staler, because ~20% of
    # plies come from pool opponents. ``ReplayBuffer`` tags every row with
    # ``mover_is_latest``, so flipping this filters the existing buffer instead of
    # regenerating it.
    train_on_pool_moves: bool = True
    # Distinct pool opponents drawn per iteration. §7 fixes the *fraction* of
    # league games (20%) and the pool they come from; it says nothing about how
    # many different snapshots one iteration must span. Coalescing onto a few
    # keeps the batched forward wide — an uncoalesced draw spreads ~102 games
    # over 20 snapshots, and a 5-row forward costs the same GPU latency as a
    # 256-row one. The marginal distribution over snapshots is unchanged across a
    # campaign; only within-iteration opponent diversity is traded away.
    league_opponents_per_iter: int = 4
    # Phase 2 targets are supported on the legal move set, so the policy CE is
    # taken over legal moves only (Phase 1 ran unmasked to *learn* legality).
    masked_policy: bool = True
    # NaN-skip guard, carried over verbatim from the Phase 1 trainer: skip the
    # update on a non-finite gradient, count the skips, abort if they become a
    # rate rather than a transient.
    nan_abort_frac: float = 0.01
    nan_abort_min_steps: int = 20


@dataclass
class EvalConfig:
    """Anchor ladder evaluation parameters (spec §8)."""

    anchor_games: int = 200  # games per (checkpoint, anchor) match
    bootstrap_resamples: int = 10_000
    # Two sim budgets, never conflated (chunk 10). 128 is the *frozen definition*
    # of the mate-in-N gates — every recorded gate number was measured at it, so
    # changing it does not re-measure the gate, it replaces the gate. 48 is match
    # play: self-play-comparable strength at a cost 200 games can afford.
    suite_sims: int = 128  # sims for mate-in-N suites
    eval_match_sims: int = 48  # sims per move for ladder match play
    # The ladder gets its own ply cap, longer than self-play's 160. §7's cap is a
    # *cost* bound on a 512-game iteration; a 200-game match runs once per ten
    # iterations and does not need it. Truncating a won game as a draw is not a
    # saving, it is a measurement error, and it errs toward 0.5 — i.e. it destroys
    # exactly the sensitivity M3's "non-overlapping CIs" criterion depends on.
    # Measured on the greedy-vs-random floor match: at 160 plies 28 of 200 games
    # were cut off mid-grind and the score read 0.880; at 300 that falls to 2 and
    # the score reads 0.938, with the residual draws structural (stalemate and
    # insufficient material) rather than truncation.
    eval_match_max_plies: int = 300
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
    phase2: Phase2Config = field(default_factory=Phase2Config)
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
        if (
            isinstance(val, dict)
            and isinstance(annotation, type)
            and hasattr(annotation, "__dataclass_fields__")
        ):
            val = _resolve_field_types(val, annotation)
        kwargs[f.name] = val
    return cls(**kwargs)


def _assert_known_keys(raw: dict, cls: type, source: str, prefix: str = "") -> None:
    """Reject config keys that do not exist on the dataclass tree.

    Silence here is the same failure family as a wrong git SHA: the run looks
    fine and the number is wrong. A near-miss on a real field is worse than a
    typo, because the default it leaves in place is plausible -- `phase1_lr`
    (not a field) next to `phase1_lr_end` (a field) would have left the LR at
    its 3e-4 default while reading as a request for 1e-4.
    """
    import dataclasses
    import typing

    valid = {f.name for f in dataclasses.fields(cls)}
    # `from __future__ import annotations` makes field.type a *string*, so the
    # nested dataclass would never be recognised and every nested key would pass.
    hints = typing.get_type_hints(cls)
    for key, value in raw.items():
        dotted = f"{prefix}{key}"
        if key not in valid:
            near = [n for n in valid if n.startswith(key[:8]) or key.startswith(n[:8])]
            hint = f" Did you mean {sorted(near)[0]!r}?" if near else ""
            raise ValueError(
                f"{source}: unknown config key {dotted!r} for {cls.__name__}.{hint} "
                f"Valid: {sorted(valid)}"
            )
        nested = hints.get(key)
        if isinstance(value, dict) and dataclasses.is_dataclass(nested):
            _assert_known_keys(value, nested, source, prefix=f"{dotted}.")


def load_config(path: str | None = None, overrides: dict | None = None) -> Config:
    """Load a Config from a YAML file, optionally applying flat key overrides.

    If path is None, returns a Config with all spec defaults.
    overrides keys use dot-notation: e.g. {"model.n_layers": 6}.
    """
    if path is not None:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        _assert_known_keys(raw, Config, str(path))
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
