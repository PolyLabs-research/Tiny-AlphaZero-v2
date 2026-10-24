# FINDING 2026-10-24 — ROCm's LayerNorm backward returns wrong affine gradients

**Stack:** torch 2.12.1+rocm7.2, HIP 7.2.53211, AMD Radeon 8060S (gfx1151) — this box.
**Reproducer:** `python scripts/check_layernorm_backward.py` (exit 1 when wrong).

## What is wrong

`nn.LayerNorm(256)` on the GPU, fp32, against an fp64 CPU reference on random data:

| rows | `weight.grad` | `bias.grad` | input grad |
|---|---|---|---|
| 64 | 1.8e-07 | 1.5e-07 | 6.6e-08 |
| 256 – 8192 | **0.64 – 0.74** | **0.67 – 0.74** | ~7e-08 |
| 73,728 | 1.6e-07 | 1.5e-07 | 6.8e-08 |

The affine gradients are ~70% wrong for every row count tested from 256 to 8,192,
and **differently on every run** (n = 1024 over four runs: 0.709, 0.745, 0.710, 0.734)
— the signature of a race in a reduction. The gradient passed to the input is always
right, and so is the forward. On the CPU, and on CUDA, nothing is wrong.

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
- Local training, if it resumes, needs a workaround before it is trusted — for example the
  value head's LayerNorm without fused affine (`elementwise_affine=False` plus an explicit
  scale and shift), checked with the reproducer — or a fixed ROCm build.
- The earlier note on non-finite gradients on ROCm (`HOW_TO_TRAIN.md`) may share a cause;
  not tested.
- Worth reporting upstream (PyTorch / ROCm), with `scripts/check_layernorm_backward.py`
  as the minimal reproducer.
