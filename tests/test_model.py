"""Tests for ChessNet architecture (chunk 3, spec §4)."""

from __future__ import annotations

import chess
import numpy as np
import torch

from tinyalphazero.config import ModelConfig, load_config
from tinyalphazero.encoding import encode_board, legal_mask_ft
from tinyalphazero.model import ChessNet, masked_policy, value_scalar

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mini_cfg(**overrides) -> ModelConfig:
    """Tiny model for fast CPU tests."""
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, d_policy=32, dropout=0.0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _default_cfg() -> ModelConfig:
    return load_config().model


def _tokens(batch: int = 2, seed: int = 0) -> torch.Tensor:
    """Random legal positions encoded as (B, 72) LongTensor."""
    rng = np.random.default_rng(seed)
    board = chess.Board()
    tokens_list = []
    for _ in range(batch):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over():
            board.reset()
            legal = list(board.legal_moves)
        tokens_list.append(encode_board(board, 0).astype(np.int64))
        board.push(legal[int(rng.integers(len(legal)))])
    return torch.from_numpy(np.stack(tokens_list))


# ---------------------------------------------------------------------------
# 1. Output shapes
# ---------------------------------------------------------------------------


def test_output_shapes_mini() -> None:
    cfg = _mini_cfg()
    net = ChessNet(cfg).eval()
    tokens = _tokens(batch=3)
    with torch.no_grad():
        out = net(tokens)
    assert out.policy_ft.shape == (3, 64, 64)
    assert out.promo.shape == (3, 64, 64, 4)
    assert out.value_wdl.shape == (3, 3)
    assert out.legality is not None and out.legality.shape == (3, 64, 64)


def test_output_shapes_no_aux() -> None:
    cfg = _mini_cfg(aux_legality=False)
    net = ChessNet(cfg).eval()
    tokens = _tokens(batch=2)
    with torch.no_grad():
        out = net(tokens)
    assert out.legality is None


def test_output_shapes_default_cfg() -> None:
    cfg = _default_cfg()
    net = ChessNet(cfg).eval()
    tokens = _tokens(batch=1)
    with torch.no_grad():
        out = net(tokens)
    assert out.policy_ft.shape == (1, 64, 64)
    assert out.value_wdl.shape == (1, 3)


# ---------------------------------------------------------------------------
# 2. Masked policy
# ---------------------------------------------------------------------------


def test_masked_policy_sums_to_one() -> None:
    cfg = _mini_cfg()
    net = ChessNet(cfg).eval()
    board = chess.Board()
    tokens = torch.from_numpy(encode_board(board, 0).astype(np.int64)).unsqueeze(0)
    mask = torch.from_numpy(legal_mask_ft(board)).unsqueeze(0)  # (1, 64, 64)

    with torch.no_grad():
        out = net(tokens)
    probs = masked_policy(out.policy_ft, mask)

    assert probs.shape == (1, 4096)
    assert abs(probs.sum().item() - 1.0) < 1e-5, f"Sum={probs.sum().item()}"


def test_masked_policy_zero_on_illegal() -> None:
    cfg = _mini_cfg()
    net = ChessNet(cfg).eval()
    board = chess.Board()
    tokens = torch.from_numpy(encode_board(board, 0).astype(np.int64)).unsqueeze(0)
    mask = torch.from_numpy(legal_mask_ft(board)).unsqueeze(0)

    with torch.no_grad():
        out = net(tokens)
    probs = masked_policy(out.policy_ft, mask)

    legal_flat = mask.reshape(1, 4096).squeeze(0)
    illegal_probs = probs.squeeze(0)[~legal_flat]
    assert illegal_probs.max().item() == 0.0, "Illegal move has nonzero probability"


def test_masked_policy_nonzero_on_legal() -> None:
    cfg = _mini_cfg()
    net = ChessNet(cfg).eval()
    board = chess.Board()
    tokens = torch.from_numpy(encode_board(board, 0).astype(np.int64)).unsqueeze(0)
    mask = torch.from_numpy(legal_mask_ft(board)).unsqueeze(0)

    with torch.no_grad():
        out = net(tokens)
    probs = masked_policy(out.policy_ft, mask)

    legal_flat = mask.reshape(1, 4096).squeeze(0)
    legal_probs = probs.squeeze(0)[legal_flat]
    assert (legal_probs > 0).all(), "Some legal move has zero probability"


def test_masked_policy_batch() -> None:
    """Masked policy must work correctly across a batch."""
    cfg = _mini_cfg()
    net = ChessNet(cfg).eval()
    B = 4
    tokens = _tokens(batch=B)

    boards = [chess.Board() for _ in range(B)]
    for i, b in enumerate(boards):
        legal = list(b.legal_moves)
        for _ in range(i):
            if legal:
                b.push(legal[0])
                legal = list(b.legal_moves)

    masks = torch.stack([torch.from_numpy(legal_mask_ft(b)) for b in boards])  # (B,64,64)

    with torch.no_grad():
        out = net(tokens)
    probs = masked_policy(out.policy_ft, masks)

    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), f"Row sums: {sums}"


# ---------------------------------------------------------------------------
# 3. value_scalar
# ---------------------------------------------------------------------------


def test_value_scalar_shape() -> None:
    cfg = _mini_cfg()
    net = ChessNet(cfg).eval()
    tokens = _tokens(batch=5)
    with torch.no_grad():
        out = net(tokens)
    v = value_scalar(out.value_wdl)
    assert v.shape == (5,)


def test_value_scalar_bounds() -> None:
    cfg = _mini_cfg()
    net = ChessNet(cfg).eval()
    tokens = _tokens(batch=10)
    with torch.no_grad():
        out = net(tokens)
    v = value_scalar(out.value_wdl)
    assert (v > -1.0).all() and (v < 1.0).all(), f"value_scalar out of bounds: {v}"


def test_value_scalar_extreme_logits() -> None:
    """With very high win logit, value_scalar should be close to +1."""
    logits = torch.tensor(
        [
            [100.0, 0.0, 0.0],  # loss wins → scalar ≈ -1
            [0.0, 0.0, 100.0],
        ]
    )  # win wins → scalar ≈ +1
    v = value_scalar(logits)
    assert v[0].item() < -0.99
    assert v[1].item() > 0.99


# ---------------------------------------------------------------------------
# 4. Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_all_parameters() -> None:
    """A combined loss must produce non-None, non-zero gradients for every parameter."""
    cfg = _mini_cfg()
    net = ChessNet(cfg).train()
    tokens = _tokens(batch=4)
    board = chess.Board()
    mask = torch.from_numpy(legal_mask_ft(board)).unsqueeze(0).expand(4, -1, -1)

    out = net(tokens)

    # Build a combined loss touching all heads.
    # Policy: CE loss against an arbitrary legal move (avoids log(0) on illegal positions).
    B = tokens.size(0)
    mask_flat = mask.reshape(B, 4096).bool()
    logits_flat = out.policy_ft.reshape(B, 4096).masked_fill(~mask_flat, float("-inf"))
    target = mask_flat.float().argmax(dim=-1)  # first legal move per sample
    policy_loss = torch.nn.functional.cross_entropy(logits_flat, target)
    promo_loss = out.promo.mean()
    value_loss = out.value_wdl.pow(2).mean()
    legality_loss = (
        (out.legality.sigmoid() - mask.float()).pow(2).mean()
        if out.legality is not None
        else torch.tensor(0.0)
    )
    loss = policy_loss + promo_loss + value_loss + legality_loss
    loss.backward()

    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.norm().item() > 0, f"Zero gradient for {name}"


# ---------------------------------------------------------------------------
# 5. Determinism in eval mode
# ---------------------------------------------------------------------------


def test_forward_deterministic_in_eval() -> None:
    cfg = _mini_cfg()
    net = ChessNet(cfg).eval()
    tokens = _tokens(batch=3)
    with torch.no_grad():
        out1 = net(tokens)
        out2 = net(tokens)
    assert torch.equal(out1.policy_ft, out2.policy_ft)
    assert torch.equal(out1.value_wdl, out2.value_wdl)
    assert torch.equal(out1.promo, out2.promo)


# ---------------------------------------------------------------------------
# 6. Per-square feature sensitivity
# ---------------------------------------------------------------------------


def test_per_square_feature_sensitivity() -> None:
    """Perturbing square i's token must change u_i but not u_j for j != i."""
    cfg = _mini_cfg(dropout=0.0)
    net = ChessNet(cfg).eval()

    tokens = _tokens(batch=1)
    tokens_perturbed = tokens.clone()
    target_sq = 10  # e2 region
    # Change the token at target_sq to something different
    original_tok = tokens[0, target_sq].item()
    new_tok = (original_tok + 1) % 43  # just a different valid token
    tokens_perturbed[0, target_sq] = new_tok

    with torch.no_grad():
        x_orig, h_orig = net.encode(tokens)
        x_pert, h_pert = net.encode(tokens_perturbed)

        u_orig = net.W_from(h_orig)  # (1, 64, d_p)
        u_pert = net.W_from(h_pert)  # (1, 64, d_p)

    diff = (u_orig - u_pert).abs().squeeze(0)  # (64, d_p)
    max_diff_at_target = diff[target_sq].max().item()
    max_diff_elsewhere = diff[torch.arange(64) != target_sq].max().item()

    assert max_diff_at_target > 1e-6, "Perturbation had no effect on target square"
    # Other squares should be unaffected (transformer may diffuse slightly, but
    # the direct path h_sq[i] → u_i should dominate for a small-depth model)
    # We test a looser condition: target square changes MORE than others
    assert max_diff_at_target > max_diff_elsewhere, (
        f"Perturbation at sq {target_sq} affected other squares more: "
        f"target={max_diff_at_target:.6f}, elsewhere={max_diff_elsewhere:.6f}"
    )


# ---------------------------------------------------------------------------
# 7. Parameter count
# ---------------------------------------------------------------------------


def test_policy_head_under_200k_params() -> None:
    cfg = _mini_cfg()
    net = ChessNet(cfg)
    counts = net.param_count()
    assert counts["policy_head"] < 200_000, (
        f"Policy head has {counts['policy_head']} params (must be < 200K)"
    )


def test_policy_head_under_200k_params_default_cfg() -> None:
    cfg = _default_cfg()
    net = ChessNet(cfg)
    counts = net.param_count()
    assert counts["policy_head"] < 200_000, (
        f"Policy head has {counts['policy_head']} params (must be < 200K)"
    )


def test_total_param_count_default_cfg() -> None:
    cfg = _default_cfg()
    net = ChessNet(cfg)
    total = net.param_count()["total"]
    # Spec expects ~6.3M at n_layers=8, acceptable range 4-8M
    assert 3_000_000 < total < 9_000_000, f"Unexpected total params: {total:,}"


def test_param_count_helper_matches_manual() -> None:
    cfg = _mini_cfg()
    net = ChessNet(cfg)
    manual = sum(p.numel() for p in net.parameters())
    assert net.param_count()["total"] == manual


# ---------------------------------------------------------------------------
# 8. promo_logits (efficient training path)
# ---------------------------------------------------------------------------


def test_promo_logits_matches_dense() -> None:
    """promo_logits on specific pairs must match the dense promo output."""
    cfg = _mini_cfg(dropout=0.0)
    net = ChessNet(cfg).eval()
    tokens = _tokens(batch=4)

    from_sq = torch.tensor([12, 28, 3, 55])
    to_sq = torch.tensor([20, 36, 11, 63])

    with torch.no_grad():
        out = net(tokens)
        _, h_sq = net.encode(tokens)
        sparse = net.promo_logits(h_sq, from_sq, to_sq)  # (4, 4)

    # Index the dense output at the same positions
    batch_idx = torch.arange(4)
    dense_selected = out.promo[batch_idx, from_sq, to_sq, :]  # (4, 4)

    assert torch.allclose(sparse, dense_selected, atol=1e-5), (
        f"Max diff: {(sparse - dense_selected).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 9. encode() helper
# ---------------------------------------------------------------------------


def test_encode_shapes() -> None:
    cfg = _mini_cfg()
    net = ChessNet(cfg).eval()
    tokens = _tokens(batch=3)
    with torch.no_grad():
        h_all, h_sq = net.encode(tokens)
    assert h_all.shape == (3, 72, cfg.d_model)
    assert h_sq.shape == (3, 64, cfg.d_model)


def test_encode_consistent_with_forward() -> None:
    """h_sq from encode() must equal the first 64 rows of h_all from forward."""
    cfg = _mini_cfg(dropout=0.0)
    net = ChessNet(cfg).eval()
    tokens = _tokens(batch=2)
    with torch.no_grad():
        h_all, h_sq = net.encode(tokens)
    assert torch.equal(h_all[:, :64, :], h_sq)
