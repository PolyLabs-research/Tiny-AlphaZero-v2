"""Phase 1 training primitives: train_step, checkpoint I/O, eval metrics (spec §5)."""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tinyalphazero.config import Config, ModelConfig, _dataclass_to_dict, _resolve_field_types
from tinyalphazero.data import Phase1Dataset
from tinyalphazero.model import ChessNet

# ---------------------------------------------------------------------------
# Bitset helpers
# ---------------------------------------------------------------------------


def unpack_legal_mask(packed: torch.Tensor) -> torch.Tensor:
    """Unpack (B, 512) uint8 packed bitset → (B, 64, 64) bool.

    Bit i (= from_sq * 64 + to_sq) is in byte i//8, bit-position i%8 (LSB-first).
    """
    B = packed.shape[0]
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).bool()  # (B, 512, 8)
    return bits.reshape(B, 4096).reshape(B, 64, 64)


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------


def _policy_loss(
    policy_ft: torch.Tensor,    # (B, 64, 64) raw logits
    from_sq: torch.Tensor,      # (B,) long
    to_sq: torch.Tensor,        # (B,) long
    legal_mask: torch.Tensor,   # (B, 64, 64) bool
    masked: bool = True,
) -> torch.Tensor:
    """CE on the played (from, to) move.

    masked=True restricts the softmax to legal moves. That gives illegal logits
    exactly zero gradient, so nothing ever teaches the unmasked argmax to be legal
    — which is what the §5 top1_legal_rate gate measures. Phase 1 runs masked=False
    so legality is learned; Phase 2 runs masked=True over search-improved targets.
    """
    B = policy_ft.size(0)
    logits = policy_ft.reshape(B, 4096)
    if masked:
        logits = logits.masked_fill(~legal_mask.reshape(B, 4096), float("-inf"))
    targets = from_sq * 64 + to_sq  # (B,)
    return F.cross_entropy(logits, targets)


def _promo_loss(
    promo_out: torch.Tensor,      # (B, 64, 64, 4) logits
    from_sq: torch.Tensor,        # (B,) long
    to_sq: torch.Tensor,          # (B,) long
    promo_target: torch.Tensor,   # (B,) long — 0=none, 1=N, 2=B, 3=R, 4=Q
) -> torch.Tensor:
    mask = promo_target > 0
    if not mask.any():
        return torch.tensor(0.0, device=promo_out.device)
    idx = torch.where(mask)[0]
    logits = promo_out[idx, from_sq[idx], to_sq[idx], :]  # (K, 4)
    targets = promo_target[idx] - 1                         # shift to 0..3
    return F.cross_entropy(logits, targets)


def _value_loss(
    value_wdl: torch.Tensor,     # (B, 3) logits [loss, draw, win]
    outcome: torch.Tensor,        # (B,) long, values -1, 0, 1
    value_weight: torch.Tensor,  # (B,) float
) -> torch.Tensor:
    targets = outcome + 1              # shift to 0, 1, 2
    per_sample = F.cross_entropy(value_wdl, targets, reduction="none")  # (B,)
    return (per_sample * value_weight).mean()


def _legality_loss(
    legality: torch.Tensor,    # (B, 64, 64) logits
    legal_mask: torch.Tensor,  # (B, 64, 64) bool
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(legality, legal_mask.float())


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(
    batch: dict[str, torch.Tensor],
    net: ChessNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg_train: Any,
    use_bf16: bool = False,
    masked_policy: bool = True,
) -> dict[str, float]:
    """Single supervised training step. Modifies net and optimizer in place.

    batch must contain: tokens, from_sq, to_sq, promo, legals, outcome, value_weight.
    masked_policy selects the policy-CE variant (see _policy_loss); Phase 1 passes
    cfg.train.phase1_policy_masked, which defaults to False.
    Returns a dict of named scalar losses (all .item()).
    """
    tokens = batch["tokens"].to(device)
    from_sq = batch["from_sq"].to(device)
    to_sq = batch["to_sq"].to(device)
    promo = batch["promo"].to(device)
    legals = batch["legals"].to(device)
    outcome = batch["outcome"].to(device)
    value_weight = batch["value_weight"].to(device)

    legal_mask = unpack_legal_mask(legals)  # (B, 64, 64) bool

    optimizer.zero_grad()

    ctx = (
        torch.amp.autocast(device.type, dtype=torch.bfloat16)
        if use_bf16
        else contextlib.nullcontext()
    )
    with ctx:
        out = net(tokens)
        p_loss = _policy_loss(out.policy_ft, from_sq, to_sq, legal_mask, masked=masked_policy)
        pr_loss = _promo_loss(out.promo, from_sq, to_sq, promo)
        v_loss = _value_loss(out.value_wdl, outcome, value_weight)
        leg_loss = (
            _legality_loss(out.legality, legal_mask)
            if out.legality is not None
            else torch.tensor(0.0, device=device)
        )
        p_w = cfg_train.policy_loss_weight
        pr_w = cfg_train.promo_loss_weight
        v_w = cfg_train.value_loss_weight
        leg_w = cfg_train.legality_loss_weight
        total = p_w * p_loss + pr_w * pr_loss + v_w * v_loss + leg_w * leg_loss

    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg_train.grad_clip)

    # A single non-finite gradient permanently poisons the weights: every subsequent
    # step is then NaN. This was observed on ROCm/gfx1151, where the backward went
    # NaN nondeterministically on a batch whose forward loss was finite and
    # bit-identical to a clean run. Skipping the update costs one batch instead of
    # the whole run. Callers should track the skip rate — a rising one is a real
    # problem, not a transient.
    skipped = not (torch.isfinite(grad_norm) and torch.isfinite(total))
    if skipped:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.step()

    return {
        "loss": total.item(),
        "policy_loss": p_loss.item(),
        "promo_loss": pr_loss.item(),
        "value_loss": v_loss.item(),
        "legality_loss": leg_loss.item(),
        "grad_norm": grad_norm.item(),
        "skipped": float(skipped),
    }


# ---------------------------------------------------------------------------
# Dataset wrapper that adds per-sample value_weight
# ---------------------------------------------------------------------------


class WeightedPhaseDataset(Dataset):
    """Wraps Phase1Dataset and injects a fixed scalar value_weight per sample."""

    def __init__(self, ds: Phase1Dataset, value_weight: float) -> None:
        self.ds = ds
        self._weight = torch.tensor(value_weight, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.ds[idx]
        item["value_weight"] = self._weight.clone()
        return item


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


_MODEL_SHAPE_KEYS = {"d_model", "n_layers", "n_heads", "d_ff", "d_policy", "seq_len", "aux_legality"}


def save_checkpoint(
    path: str | Path,
    net: ChessNet,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    step: int,
    phase: str = "1",
) -> None:
    """Save a training checkpoint in the standard format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": _dataclass_to_dict(cfg),
            "git_sha": _git_sha(),
            "step": step,
            "phase": phase,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    reference_model_cfg: ModelConfig | None = None,
) -> tuple[ChessNet, dict]:
    """Load checkpoint, reconstruct ChessNet, optionally validate model shape.

    Returns (net, meta) where meta contains step, phase, config, git_sha.
    Raises ValueError if reference_model_cfg is supplied and shape fields differ.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = _resolve_field_types(ckpt["config"], Config)

    if reference_model_cfg is not None:
        saved = _dataclass_to_dict(cfg.model)
        ref = _dataclass_to_dict(reference_model_cfg)
        mismatches = {
            k: (saved[k], ref[k])
            for k in _MODEL_SHAPE_KEYS
            if k in saved and k in ref and saved[k] != ref[k]
        }
        if mismatches:
            raise ValueError(f"Model shape mismatch between checkpoint and reference: {mismatches}")

    net = ChessNet(cfg.model)
    net.load_state_dict(ckpt["model"])
    net.to(device)

    meta = {k: v for k, v in ckpt.items() if k != "model"}
    meta["config_obj"] = cfg
    return net, meta


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via Mann-Whitney rank-sum statistic (O(n log n), no sklearn)."""
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_pos, n_neg = len(pos), len(neg)
    combined = np.concatenate([pos, neg])
    is_pos = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(bool)
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)
    rank_sum = ranks[is_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# Masked top-k coverage of the mating move. k=8 is the gated one as of
# BLOCKED-2026-08-06-phase1-gates.md; the rest are reported for shape.
MATE1_TOPK = (1, 3, 8, 16)


@torch.no_grad()
def compute_metrics(
    net: ChessNet,
    data_dir: Path,
    device: torch.device,
    mate1_dir: Path | None = None,
    n_random: int = 2000,
    n_mate1: int = 500,
) -> dict[str, float]:
    """Compute §5 acceptance metrics on a sample of the dataset.

    Returns a dict with keys matching the JSONL log schema.
    """
    net.eval()
    rng = np.random.default_rng(0)

    # ---- random-play sample ----
    rand_ds = Phase1Dataset(data_dir)
    n_sample = min(n_random, len(rand_ds))
    idxs = rng.choice(len(rand_ds), size=n_sample, replace=False)

    legal_top1_correct = 0
    exact_top1_correct = 0
    leg_scores: list[np.ndarray] = []
    leg_labels: list[np.ndarray] = []

    for i in idxs:
        item = rand_ds[int(i)]
        tokens = item["tokens"].unsqueeze(0).to(device)
        from_sq = item["from_sq"].item()
        to_sq = item["to_sq"].item()
        legals_packed = item["legals"].unsqueeze(0).to(device)

        out = net(tokens)
        policy = out.policy_ft[0]  # (64, 64)

        # legal mask from packed bitset
        legal_mask_t = unpack_legal_mask(legals_packed)[0]  # (64, 64)

        # top-1 unmasked — is it legal?
        top1_flat = policy.reshape(4096).argmax().item()
        top1_from, top1_to = divmod(top1_flat, 64)
        legal_top1_correct += int(legal_mask_t[top1_from, top1_to].item())

        # top-1 masked — is it the played move?
        masked_logits = policy.reshape(4096).masked_fill(~legal_mask_t.reshape(4096), float("-inf"))
        top1_masked = masked_logits.argmax().item()
        played_idx = from_sq * 64 + to_sq
        exact_top1_correct += int(top1_masked == played_idx)

        # legality head AUC data
        if out.legality is not None:
            leg_logits = out.legality[0].reshape(4096).cpu().float().numpy()
            leg_gt = legal_mask_t.reshape(4096).cpu().numpy().astype(np.float32)
            leg_scores.append(leg_logits)
            leg_labels.append(leg_gt)

    top1_legal_rate = legal_top1_correct / n_sample
    top1_exact_rate = exact_top1_correct / n_sample

    legality_auc = float("nan")
    if leg_scores:
        all_scores = np.concatenate(leg_scores)
        all_labels = np.concatenate(leg_labels)
        legality_auc = _binary_auc(all_scores, all_labels)

    # ---- mate-in-1 metrics ----
    wdl_anchor_mass = float("nan")

    if mate1_dir is not None and Path(mate1_dir).exists():
        mate1_ds = Phase1Dataset(mate1_dir)
        n_m1 = min(n_mate1, len(mate1_ds))
        m1_idxs = rng.choice(len(mate1_ds), size=n_m1, replace=False)
    else:
        # Generate mate-in-1 positions on the fly
        from tinyalphazero.data import generate_mate_in_1
        m1_positions = list(generate_mate_in_1(min(n_mate1, 200), rng))

        class _InlineDS:
            def __len__(self):
                return len(m1_positions)
            def __getitem__(self, i):
                pos = m1_positions[i]
                return {
                    "tokens": torch.from_numpy(pos.tokens.astype(np.int64)),
                    "from_sq": torch.tensor(pos.from_sq, dtype=torch.long),
                    "to_sq": torch.tensor(pos.to_sq, dtype=torch.long),
                    "outcome": torch.tensor(pos.outcome, dtype=torch.long),
                    "legals": torch.from_numpy(pos.legals.copy()),
                }

        mate1_ds = _InlineDS()
        m1_idxs = np.arange(len(mate1_ds))
        n_m1 = len(m1_idxs)

    topk_hits = dict.fromkeys(MATE1_TOPK, 0)
    wdl_win_mass_total = 0.0

    for i in m1_idxs:
        item = mate1_ds[int(i)]
        tokens = item["tokens"].unsqueeze(0).to(device)
        from_sq = item["from_sq"].item()
        to_sq = item["to_sq"].item()

        legal_packed = item["legals"].unsqueeze(0).to(device) if "legals" in item else None

        out = net(tokens)
        policy = out.policy_ft[0]  # (64, 64)

        if legal_packed is not None:
            legal_mask_t = unpack_legal_mask(legal_packed)[0]
            masked_logits = policy.reshape(4096).masked_fill(~legal_mask_t.reshape(4096), float("-inf"))
        else:
            masked_logits = policy.reshape(4096)

        # Rank of the mating move among legal moves, then derive every k at once.
        # rank == 0 is exactly the old top-1 solve, so mate1_solve_rate stays
        # identical to the k=1 slice by construction rather than by convention.
        target = from_sq * 64 + to_sq
        rank = int((masked_logits > masked_logits[target]).sum().item())
        for k in MATE1_TOPK:
            topk_hits[k] += int(rank < k)

        # WDL mass on correct class (win = index 2 for outcome=+1)
        wdl_probs = torch.softmax(out.value_wdl[0], dim=-1)
        wdl_win_mass_total += wdl_probs[2].item()

    topk_rates = {
        k: (topk_hits[k] / n_m1 if n_m1 > 0 else float("nan")) for k in MATE1_TOPK
    }
    wdl_anchor_mass = wdl_win_mass_total / n_m1 if n_m1 > 0 else float("nan")

    return {
        "top1_legal_rate": top1_legal_rate,
        "top1_exact_rate": top1_exact_rate,
        "mate1_solve_rate": topk_rates[1],
        "legality_auc": legality_auc,
        "wdl_anchor_mass": wdl_anchor_mass,
        **{f"mate1_top{k}_masked": topk_rates[k] for k in MATE1_TOPK},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


NO_GIT_SENTINEL = "no-git"


def _git_sha() -> str:
    """Short HEAD SHA of the repository this package lives in, or ``"no-git"``.

    Anchored to the source tree (``git -C <repo_root>``), not the process cwd.
    The cwd-relative version silently stamped whatever repo the run happened to
    be launched from — and a *wrong* provenance record is worse than an absent
    one, because nothing downstream can tell the two apart. Absence is explicit.
    """
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (OSError, subprocess.SubprocessError):
        # git absent, not a repository, or no commits yet — all mean "no provenance".
        return NO_GIT_SENTINEL
