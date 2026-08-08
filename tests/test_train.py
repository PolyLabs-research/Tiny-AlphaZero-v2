"""Tests for Phase 1 training primitives (chunk 4, spec §5)."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from tinyalphazero.config import Config, ModelConfig, TrainConfig
from tinyalphazero.data import (
    Phase1Dataset,
    generate_random_games,
    write_dataset,
)
from tinyalphazero.encoding import encode_board, encode_move, legal_moves_packed
from tinyalphazero.model import ChessNet
from tinyalphazero.train import (
    WeightedPhaseDataset,
    _binary_auc,
    _legality_loss,
    _policy_loss,
    _promo_loss,
    _value_loss,
    compute_metrics,
    load_checkpoint,
    save_checkpoint,
    train_step,
    unpack_legal_mask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mini_cfg(**overrides) -> ModelConfig:
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, d_policy=32, dropout=0.0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _mini_train_cfg(**overrides) -> TrainConfig:
    cfg = TrainConfig(
        batch_size=16,
        grad_clip=1.0,
        policy_loss_weight=1.0,
        promo_loss_weight=1.0,
        value_loss_weight=1.0,
        legality_loss_weight=1.0,
        phase1_value_weight_random=0.25,
        phase1_value_weight_anchor=1.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


_CPU = torch.device("cpu")


def _make_batch(n: int = 16, seed: int = 0, device: torch.device = _CPU) -> dict:
    """Generate a valid batch from actual chess positions."""
    rng = np.random.default_rng(seed)
    board = chess.Board()
    tokens_list, from_list, to_list, promo_list, legals_list, outcomes = [], [], [], [], [], []

    for _ in range(n):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over():
            board.reset()
            legal = list(board.legal_moves)
        move = legal[int(rng.integers(len(legal)))]
        tokens_list.append(encode_board(board, 0).astype(np.int64))
        f, t, p = encode_move(move)
        from_list.append(f)
        to_list.append(t)
        promo_list.append(p)
        legals_list.append(legal_moves_packed(board))
        outcomes.append(int(rng.choice([-1, 0, 1])))
        board.push(move)

    return {
        "tokens": torch.from_numpy(np.stack(tokens_list)).to(device),
        "from_sq": torch.tensor(from_list, dtype=torch.long, device=device),
        "to_sq": torch.tensor(to_list, dtype=torch.long, device=device),
        "promo": torch.tensor(promo_list, dtype=torch.long, device=device),
        "legals": torch.from_numpy(np.stack(legals_list)).to(device),
        "outcome": torch.tensor(outcomes, dtype=torch.long, device=device),
        "value_weight": torch.full((n,), 0.25, device=device),
    }


def _make_full_config(model_cfg: ModelConfig | None = None) -> Config:
    cfg = Config()
    if model_cfg is not None:
        cfg.model = model_cfg
    cfg.train = _mini_train_cfg()
    return cfg


# ---------------------------------------------------------------------------
# 1. unpack_legal_mask
# ---------------------------------------------------------------------------


def test_unpack_legal_mask_shape() -> None:
    packed = torch.zeros(4, 512, dtype=torch.uint8)
    out = unpack_legal_mask(packed)
    assert out.shape == (4, 64, 64)
    assert out.dtype == torch.bool


def test_unpack_legal_mask_bit_correctness() -> None:
    """Setting bit from_sq*64+to_sq should appear at position [from_sq, to_sq]."""
    packed = torch.zeros(1, 512, dtype=torch.uint8)
    from_sq, to_sq = 12, 28
    flat_idx = from_sq * 64 + to_sq
    packed[0, flat_idx // 8] |= 1 << (flat_idx % 8)

    mask = unpack_legal_mask(packed)
    assert mask[0, from_sq, to_sq].item() is True
    assert mask[0].sum().item() == 1


def test_unpack_round_trip_with_board() -> None:
    """unpack_legal_mask of legal_moves_packed must agree with legal_mask_ft."""
    from tinyalphazero.encoding import legal_mask_ft

    board = chess.Board()
    packed = torch.from_numpy(legal_moves_packed(board)).unsqueeze(0)
    expected = torch.from_numpy(legal_mask_ft(board)).unsqueeze(0)
    unpacked = unpack_legal_mask(packed)
    assert torch.equal(unpacked, expected)


# ---------------------------------------------------------------------------
# 2. Loss functions
# ---------------------------------------------------------------------------


def test_policy_loss_finite_on_valid_batch() -> None:
    batch = _make_batch(8)
    net = ChessNet(_mini_cfg()).eval()
    with torch.no_grad():
        out = net(batch["tokens"])
    legal_mask = unpack_legal_mask(batch["legals"])
    loss = _policy_loss(out.policy_ft, batch["from_sq"], batch["to_sq"], legal_mask)
    assert loss.isfinite(), f"Policy loss is not finite: {loss}"


def test_value_loss_weighted() -> None:
    """Uniform weights should give the same result as standard CE."""
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    outcomes = targets - 1  # shift back to -1..1
    w_uniform = torch.ones(8)
    loss = _value_loss(logits, outcomes, w_uniform)
    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_promo_loss_no_promo_returns_zero() -> None:
    promo_out = torch.randn(8, 64, 64, 4)
    from_sq = torch.zeros(8, dtype=torch.long)
    to_sq = torch.zeros(8, dtype=torch.long)
    promo_target = torch.zeros(8, dtype=torch.long)  # all 0 = no promo
    loss = _promo_loss(promo_out, from_sq, to_sq, promo_target)
    assert loss.item() == 0.0


def test_promo_loss_with_promo() -> None:
    promo_out = torch.randn(8, 64, 64, 4)
    from_sq = torch.zeros(8, dtype=torch.long)
    to_sq = torch.zeros(8, dtype=torch.long)
    promo_target = torch.tensor([0, 1, 0, 2, 0, 0, 3, 4], dtype=torch.long)
    loss = _promo_loss(promo_out, from_sq, to_sq, promo_target)
    assert loss.isfinite() and loss.item() > 0


def test_legality_loss_finite() -> None:
    B = 4
    legality = torch.randn(B, 64, 64)
    legal_mask = torch.randint(0, 2, (B, 64, 64), dtype=torch.bool)
    loss = _legality_loss(legality, legal_mask)
    assert loss.isfinite()


# ---------------------------------------------------------------------------
# 3. train_step
# ---------------------------------------------------------------------------


def test_train_step_returns_dict() -> None:
    model_cfg = _mini_cfg()
    net = ChessNet(model_cfg).train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    batch = _make_batch(8)
    losses = train_step(batch, net, optimizer, torch.device("cpu"), _mini_train_cfg())
    assert "loss" in losses
    assert "policy_loss" in losses
    assert "value_loss" in losses
    assert "legality_loss" in losses
    assert all(np.isfinite(v) for v in losses.values())


@pytest.mark.parametrize("masked", [True, False])
def test_train_step_loss_decreases(masked: bool) -> None:
    """Overfit a single batch: policy CE must drop 4× within 200 steps (mini model, 64 samples).

    Asserts on the policy component rather than the total. The aux legality BCE is
    4096 binary targets per sample and retains a floor on a single batch, so a
    total-loss ratio is both noisy across seeds and structurally capped; the policy
    term is what overfitting should actually drive toward zero. Single-step readings
    at lr=1e-2 bounce by 3× between adjacent steps, hence the smoothed windows.
    """
    model_cfg = _mini_cfg(dropout=0.0)
    net = ChessNet(model_cfg).train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2)
    batch = _make_batch(64, seed=7)
    cfg_train = _mini_train_cfg()

    history: list[dict[str, float]] = []
    for _step in range(200):
        history.append(
            train_step(
                batch, net, optimizer, torch.device("cpu"), cfg_train, masked_policy=masked
            )
        )

    def window(key: str, sl: slice) -> float:
        return float(np.mean([h[key] for h in history[sl]]))

    initial = window("policy_loss", slice(0, 5))
    final = window("policy_loss", slice(-20, None))
    ratio = initial / max(final, 1e-9)
    assert ratio >= 4, (
        f"Policy CE did not drop 4× (masked={masked}): "
        f"initial={initial:.4f}, final={final:.4f}, ratio={ratio:.2f}"
    )
    assert window("loss", slice(-20, None)) < window("loss", slice(0, 5))


def test_train_step_skips_update_on_non_finite_gradient() -> None:
    """A non-finite gradient must not reach the weights.

    On ROCm/gfx1151 the backward went NaN nondeterministically on a batch whose
    forward loss was finite, and the poisoned weights made every later step NaN —
    that is how a 600-step run came back all-NaN. Simulated here by injecting a NaN
    grad via a hook, since the real event is not reproducible on demand.
    """
    net = ChessNet(_mini_cfg(dropout=0.0)).train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2)
    batch = _make_batch(16, seed=11)
    cfg_train = _mini_train_cfg()

    before = {n: p.detach().clone() for n, p in net.named_parameters()}

    victim = next(p for p in net.parameters() if p.requires_grad)
    handle = victim.register_hook(lambda g: g * float("nan"))
    try:
        losses = train_step(batch, net, optimizer, torch.device("cpu"), cfg_train)
    finally:
        handle.remove()

    assert losses["skipped"] == 1.0, "non-finite gradient was not detected"
    for n, p in net.named_parameters():
        assert torch.equal(p, before[n]), f"parameter {n} was updated despite NaN gradient"
        assert torch.isfinite(p).all(), f"parameter {n} became non-finite"

    # And a clean step afterwards still updates, i.e. the guard is not sticky.
    losses = train_step(batch, net, optimizer, torch.device("cpu"), cfg_train)
    assert losses["skipped"] == 0.0
    assert any(not torch.equal(p, before[n]) for n, p in net.named_parameters())


def test_policy_loss_unmasked_starts_at_uniform_over_all_squares() -> None:
    """Regression guard for the runs/phase1_a failure.

    Masked CE on uniform-random-play data starts at E[ln n_legal] — already its
    optimum, so the run was flat for 9.5h. Unmasked CE starts at ln(4096) = 8.32
    and has somewhere to descend to. The gap between these two numbers is the bug.
    """
    net = ChessNet(_mini_cfg(dropout=0.0)).eval()
    batch = _make_batch(64, seed=7)
    with torch.no_grad():
        out = net(batch["tokens"])
    legal_mask = unpack_legal_mask(batch["legals"])

    masked = _policy_loss(
        out.policy_ft, batch["from_sq"], batch["to_sq"], legal_mask, masked=True
    ).item()
    unmasked = _policy_loss(
        out.policy_ft, batch["from_sq"], batch["to_sq"], legal_mask, masked=False
    ).item()

    assert unmasked > masked + 3.0, f"masked={masked:.3f} unmasked={unmasked:.3f}"
    assert abs(unmasked - math.log(4096)) < 1.0


def test_aux_legality_head_trains_the_trunk() -> None:
    """Regression guard: the aux head must not be detached from the trunk (spec §4.4).

    It shipped as `self._legality(h_sq.detach())`, which made it a linear probe and
    starved the trunk of its dense mobility signal.
    """
    net = ChessNet(_mini_cfg(dropout=0.0)).train()
    batch = _make_batch(8, seed=3)
    out = net(batch["tokens"])
    assert out.legality is not None

    legal_mask = unpack_legal_mask(batch["legals"])
    _legality_loss(out.legality, legal_mask).backward()

    trunk_grads = [
        p.grad.abs().sum().item()
        for name, p in net.named_parameters()
        if p.grad is not None and "aux" not in name and "_legality" not in name
    ]
    assert trunk_grads and max(trunk_grads) > 0, (
        "Legality BCE produced no gradient in the trunk — aux head is detached"
    )


def test_train_step_no_aux_legality() -> None:
    """train_step must work when aux_legality=False (legality head absent)."""
    model_cfg = _mini_cfg(aux_legality=False)
    net = ChessNet(model_cfg).train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    batch = _make_batch(8)
    losses = train_step(batch, net, optimizer, torch.device("cpu"), _mini_train_cfg())
    assert losses["legality_loss"] == 0.0


# ---------------------------------------------------------------------------
# 4. Checkpoint save/load round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_save_load_roundtrip() -> None:
    """Load a saved checkpoint and verify outputs are bit-exact in eval mode."""
    model_cfg = _mini_cfg()
    cfg = _make_full_config(model_cfg)
    net = ChessNet(model_cfg)
    optimizer = torch.optim.AdamW(net.parameters())

    batch = _make_batch(4)
    net.eval()
    with torch.no_grad():
        out_before = net(batch["tokens"])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, net, optimizer, cfg, step=42, phase="1")

        net2, meta = load_checkpoint(path, device="cpu")
        net2.eval()
        with torch.no_grad():
            out_after = net2(batch["tokens"])

    assert torch.equal(out_before.policy_ft, out_after.policy_ft), "policy_ft mismatch after reload"
    assert torch.equal(out_before.value_wdl, out_after.value_wdl), "value_wdl mismatch after reload"
    assert meta["step"] == 42
    assert meta["phase"] == "1"


def test_checkpoint_meta_fields() -> None:
    model_cfg = _mini_cfg()
    cfg = _make_full_config(model_cfg)
    net = ChessNet(model_cfg)
    optimizer = torch.optim.AdamW(net.parameters())

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, net, optimizer, cfg, step=100, phase="1")
        _, meta = load_checkpoint(path)

    assert "config" in meta
    assert "git_sha" in meta
    assert meta["step"] == 100
    assert meta["phase"] == "1"


def test_checkpoint_shape_mismatch_raises() -> None:
    """load_checkpoint must raise ValueError if model shape mismatches reference."""
    model_cfg = _mini_cfg()
    cfg = _make_full_config(model_cfg)
    net = ChessNet(model_cfg)
    optimizer = torch.optim.AdamW(net.parameters())

    # Reference with different d_model
    ref_cfg = _mini_cfg(d_model=128)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, net, optimizer, cfg, step=1, phase="1")
        with pytest.raises(ValueError, match="shape mismatch"):
            load_checkpoint(path, reference_model_cfg=ref_cfg)


# ---------------------------------------------------------------------------
# 5. WeightedPhaseDataset
# ---------------------------------------------------------------------------


def test_weighted_phase_dataset_adds_weight() -> None:
    rng = np.random.default_rng(0)
    games = list(generate_random_games(5, rng))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "d"
        write_dataset(iter(games), out)
        ds_raw = Phase1Dataset(out)
        ds = WeightedPhaseDataset(ds_raw, value_weight=0.25)
        item = ds[0]
        assert "value_weight" in item
        assert abs(item["value_weight"].item() - 0.25) < 1e-6


def test_weighted_phase_dataset_len() -> None:
    rng = np.random.default_rng(1)
    games = list(generate_random_games(5, rng))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "d"
        write_dataset(iter(games), out)
        ds = WeightedPhaseDataset(Phase1Dataset(out), value_weight=1.0)
        assert len(ds) == len(Phase1Dataset(out))


# ---------------------------------------------------------------------------
# 6. _binary_auc
# ---------------------------------------------------------------------------


def test_binary_auc_perfect() -> None:
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 1, 0, 0])
    assert abs(_binary_auc(scores, labels) - 1.0) < 1e-9


def test_binary_auc_random() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(100)
    labels = rng.integers(0, 2, 100).astype(float)
    auc = _binary_auc(scores, labels)
    assert 0.0 <= auc <= 1.0


def test_binary_auc_worst() -> None:
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([1, 1, 0, 0])
    assert abs(_binary_auc(scores, labels) - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# 7. compute_metrics (smoke test — correctness validated in acceptance run)
# ---------------------------------------------------------------------------


def test_compute_metrics_smoke() -> None:
    """compute_metrics must run and return the expected keys without crashing."""
    rng = np.random.default_rng(0)
    games = list(generate_random_games(20, rng))

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        write_dataset(iter(games), data_dir)

        model_cfg = _mini_cfg()
        net = ChessNet(model_cfg)
        metrics = compute_metrics(
            net,
            data_dir,
            device=torch.device("cpu"),
            mate1_dir=None,
            n_random=50,
            n_mate1=20,
        )

    expected_keys = {
        "top1_legal_rate",
        "top1_exact_rate",
        "mate1_solve_rate",
        "legality_auc",
        "wdl_anchor_mass",
    }
    assert expected_keys.issubset(metrics.keys())
    assert 0.0 <= metrics["top1_legal_rate"] <= 1.0
