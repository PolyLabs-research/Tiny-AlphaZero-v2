# FINDING 2026-10-24 — ROCm's LayerNorm backward returns wrong affine gradients

**Stack:** torch 2.12.1+rocm7.2, HIP 7.2.53211, AMD Radeon 8060S (gfx1151) — this box.
**Reproducers:** `python scripts/check_layernorm_backward.py` (exit 1 when wrong);
`upstream/repro_layernorm_backward.py` (torch only).
**Status: a known upstream bug, fixed in torch ≥ 2.13** — see "Upstream" below.

## What is wrong

For **128 ≤ rows ≤ 65,536**, LayerNorm's backward **never computes the odd-indexed
elements of `weight.grad` and `bias.grad`**; the even-indexed ones are exact. fp32 on
the GPU against an fp64 CPU reference, random data:

| rows | width | even columns (rel. error) | odd columns: max \|value\| | odd columns (rel. error) |
|---|---|---|---|---|
| 64 | 256 | 1.8e-07 | (correct) | 0.00 |
| 128, 1,024 | 256 | ~1e-07 | **0** | **1.00** |
| 65,536 | 256 | 1.1e-06 | 3.3 (stale) | **1.00** |
| 73,728 and 131,072 | 256 | ~1.6e-07 | (correct) | 0.00 |
| 1,024 | 64 / 512 / 1,024 / 257 | ~1.7e-07 | 0 / 0.23 / 0.20 / 3.4 | **1.00** |
| 1,024 | 256, bf16 | 3e-03 | 0 | 1.00 |

Odd elements are left unwritten — zero in a fresh allocation, stale values otherwise —
so the whole-vector relative error is ≈ √½ ≈ 0.70. It is **deterministic** (identical
inputs give identical wrong output), unchanged by `AMD_SERIALIZE_KERNEL=3` /
`HIP_LAUNCH_BLOCKING=1`, the same against plain-op gradients on the same device, and
present in fp32, fp16 and bf16 at every width tried. The input gradient and the forward
are right. On the CPU, and on CUDA, nothing is wrong. Workaround, verified exact at
128 / 1,024 / 65,536 rows: `nn.LayerNorm(d, elementwise_affine=False)` followed by an
explicit `* weight + bias`.

## Upstream

- [pytorch/pytorch#199265](https://github.com/pytorch/pytorch/issues/199265) (gfx1201,
  closed 2026-10-05): the identical symptom on torch 2.12.1+rocm7.2 — "never writes
  odd-indexed weight/bias grad elements (M >= 128)". Root cause: the ROCm path of
  `aten/src/ATen/native/cuda/layer_norm_kernel.cu` assumed a 64-wide warp; on wave32
  RDNA GPUs half the feature values were dropped.
- [pytorch/pytorch#183864](https://github.com/pytorch/pytorch/pull/183864) (merged
  2026-05-27) queries the warp size at runtime and names "AMD Strix Halo and other RDNA
  GPUs"; it is in torch 2.13.0 and later. #199265 confirms 2.12.1 fails and
  2.14.1+rocm7.2 passes on gfx1201.
- [ROCm/TheRock#9064](https://github.com/ROCm/TheRock/issues/9064) (gfx1151, closed
  2026-10-16): the same on gfx1151 nightlies, resolved by a newer torch.
- `download.pytorch.org/whl/rocm7.2` carries torch 2.13.0, 2.14.0 and 2.14.1 for cp312.

**Verified on this box (2026-10-24):** torch 2.14.1+rocm7.2 from that index, in a
throwaway venv, passes `upstream/repro_layernorm_backward.py` at every row count from
64 to 131,072 and at widths 64, 256, 512 and 1,024 (worst relative error ~1e-6); 2.12.1
fails the same script. No new upstream report is needed; this box was simply on the
last affected release.

### Correction (2026-10-24 19:50)

The first version of this file said the error varied "differently on every run — the
signature of a race" and spanned 256–8,192 rows. Both were wrong: the "repeats" drew new
random data each time, and the row range was only what had been tested. Measured since:
deterministic, 128–65,536 rows, odd columns never written.

## Where it lands in this model

The only LayerNorm with that many rows is the value head's (`value_mlp.0`): it sees one
pooled row per position, 1,024 per training batch. The transformer's LayerNorms see
1,024 × 72 = 73,728 rows and are correct. `scripts/grad_diagnostic.py` on p2_f/0020 and
four of the bridge's fixed batches, raw (unclipped) gradients against CPU fp32:

| stack | `value_mlp.0.{weight,bias}` error | every other parameter | cosine of the whole gradient |
|---|---|---|---|
| ROCm bf16 + `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (**how every local round trained**) | up to 161× | ≤ 8% | 0.900, 0.991, 0.984, 0.906 |
| ROCm bf16, flag off | up to 5,141× | ≤ 6% | 0.073, 0.211, 0.067, 0.637 |
| ROCm bf16, SDPA forced to `math` | identical to flag off | | |
| ROCm fp32, flag on | up to 3,028× | 0.000 | 0.995, 0.989, 0.130, 0.103 |
| CUDA bf16 (RunPod RTX 4090) | — (norms within 1.3% of fp32 on all 8 bridge batches) | | |

Not bf16 (fp32 is wrong too) and not attention (the `math` backend changes nothing). The
error magnitude is large because the true gradient of those 512 parameters is small.

## What it did to training here

Every run on this box — Phase 1 and p2_a through p2_f, all at batch 1,024 — trained the
value head's LayerNorm affine parameters on noise. Where that noise is large it also
dominates the total gradient norm, and `grad_clip: 1.0` then rescales *every*
parameter's update down with it: in the training configuration, by ~10% on two of four
batches tested (whole-gradient norm ratio 1.10, cosine 0.90). Measurement is
unaffected: the ladder, the suites and the skill-0 rung run forward passes only.

## What follows

- **Training on CUDA is the correct stack**, not merely an equivalent one; PREREG-p2_g's
  Step 0b bridge failed because its reference (this box) is the side that is wrong.
- **Local training needs torch ≥ 2.13** (or the explicit-affine workaround) before it is
  trusted. Upgrading is a stack change, so not while p2_g is measured on this box
  (campaign rule 6); after it closes, with a bridge like p2_e's Step 0.
- The earlier note on non-finite gradients on ROCm (`HOW_TO_TRAIN.md`) may share a cause;
  not tested.
