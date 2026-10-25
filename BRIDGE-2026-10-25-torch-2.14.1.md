# BRIDGE 2026-10-25 — this box's stack, torch 2.12.1 → 2.14.1 (+rocm7.2)

**Written before the upgrade. Not edited after, except by a dated results section.**

## Why

torch 2.12.1+rocm7.2 on this gfx1151 trains the value head's LayerNorm on half-missing
gradients (`FINDING-2026-10-24-rocm-layernorm-backward.md`, pytorch/pytorch#199265);
2.14.1+rocm7.2 from the same index is verified correct on the reproducer. Upgrading
changes the stack every local measurement runs on (campaign rule 6), so before a local
number is compared with p2_g's, the instrument is bridged: the same checkpoint,
measured on both stacks.

## State at writing

p2_g is closed (`672c444`); nothing is running. The venv holds torch 2.12.1+rocm7.2,
triton-rocm 3.7.1, numpy 2.5.0 (full freeze kept in the session scratchpad for
rollback). Measured on 2.12.1 already: p2_g/0060's ladder pass and three skill-0 reps
(`runs/p2_g/`), and its forward outputs on 2,048 positions drawn with seed 0 from p2_g's
ring — **bit-identical across two runs** (`scripts/forward_dump.py`), so any difference
after the upgrade is the stack's.

## Registered expectations (2.14.1, this box, under `gpu hold`)

| | measurement | expected |
|---|---|---|
| B0 | forward outputs of p2_g/0060 on the same 2,048 positions | legal-move argmax agreement ≥ 0.99; value max \|diff\| ≤ 0.05 |
| B1 | `scripts/check_layernorm_backward.py` | exit 0 (every row count correct) |
| B2 | `scripts/stack_bridge.py` on p2_f/0020, the 8 bridge batches | every batch's grad_norm within 2% of CPU fp32 (`runs/p2_g_bridge/bridge_cpu.json`; the CUDA pod was ≤ 1.3%); losses within 0.1% |
| B3 | ladder pass on p2_g/0060 (`role=diagnostic`) | mate-in-2 within ±3 of 184/200; mate-in-3 within ±4 of 139/200; random, greedy, phase1 each within ±2 games of 2.12.1's 200/200 |
| B4 | three skill-0 reps of p2_g/0060 (frozen book) | paired CI against 2.12.1's reps (0.745, 0.75, 0.7225) includes zero |
| B5 | ECE on that pass | reported; 2.12.1 read 0.0593 |

Plus the test suite (CPU) green and `scripts/check_env.py` reporting the new version.

## Decision rules

- **All pass:** 2.14.1 is this box's stack. `pyproject.toml` raises the floor to
  `torch>=2.13` with the reason; local measurements stay comparable with p2_g's.
- **B1 or B2 fails:** the upgrade did not fix training here; record it and keep local
  training off until it does. Measurement continues to be judged on B0/B3/B4.
- **B3 or B4 fails:** the instrument moved. The size is recorded, and every later local
  comparison is made against references re-measured on 2.14.1 — as p2_e did when the
  skill-0 instrument moved (PREREG-p2_e Addendum 1).
- **Rollback** (the stack misbehaves outright): reinstall torch 2.12.1+rocm7.2 from the
  freeze.

## Results (2026-10-25, measured after the upgrade)

Upgraded in place: torch 2.14.1+rocm7.2, triton-rocm 3.8.0 (from 2.12.1 / 3.7.1), same
index, same HIP 7.2.53211. `check_env.py` reports 2.14.1+rocm7.2 on the Radeon 8060S (bf16
matmul 28.9 TFLOPS); CPU suite 773 passed, 2 skipped. Every GPU measurement ran under
`gpu hold`, code `3ebd31e`.

| | expected | measured | |
|---|---|---|---|
| B0 | argmax agreement ≥ 0.99; value max \|diff\| ≤ 0.05 | agreement **0.9868** (27 of 2,048 differ); value max \|diff\| 0.035 (mean 0.0023); policy max 0.28 (mean 0.017); promotion max 0.041 | **miss** (agreement) |
| B1 | exit 0 | exit 0 | pass |
| B2 | grad_norm within 2% of CPU fp32 on every batch; losses within 0.1% | grad_norm 0.14, 0.50, 0.32, 0.69, 0.26, 1.16, 0.10, 0.11% (2.12.1: 122, 2.5, 3.1, 6.5, 6.5, 0.9, 13.9, 11.1%); total loss worst 0.082% | pass |
| B3 | mate-in-2 184 ± 3; mate-in-3 139 ± 4; rungs ± 2 games of 200/200 | mate-in-2 **184**, mate-in-3 **139**; random 200/200 and greedy 200/200; phase1 199 W, 1 D, 0 L | pass |
| B4 | paired CI against (0.745, 0.75, 0.7225) includes zero | reps 0.7375, 0.7725, 0.715 (mean 0.742 against 0.739): **+0.0025 [−0.0375, +0.0433]** | pass |
| B5 | reported | ECE 0.0402 (2.12.1: 0.0593); book prior entropy 0.9025, identical | — |

**B0's miss, characterized after the fact:** every one of the 27 positions whose legal
argmax changed was a near-tie on 2.12.1 — top-two legal logit gap median 0.012, max 0.031
(two bf16 steps at these logits' magnitude), against a median gap of 0.77 over all 2,048;
4.4% of positions have a gap that small. In 20 of the 27 the new choice is 2.12.1's second
choice. 2.12.1 was bit-identical across two runs, so this is the new stack's kernels and
not noise. The 0.99 bar was set without knowing how many near-ties the sample held.

**Notes on B2 and B5, so they are not read as more than they are.** B2's "losses" is read
as the total loss, as Step 0b.1 read it (`PREREG-p2_g` Addendum 2, 0b.1). Component-wise, one entry is
past 0.1%: promotion loss on batch 6, 1.24% off — 2.4e-5 absolute on a loss of 0.0019. The
CUDA pod, the stack verified correct, is 0.41% off on that same entry. B5's ECE is
measured on each pass's own phase1 and skill-0 games, so the two numbers come from
different games. 0.040 lies inside the 0.040–0.059 band p2_g's checkpoints spanned.

### Decision

B1–B4 pass and B0 misses. No row covers that case: "All pass" is not literally met, and
B0 is in neither failure row. The reading taken here is recorded as such, not as a rule
change:

- **2.14.1 is this box's stack, and local training is trusted again** (B1 and B2: the
  defect is gone, and the gradients match CPU fp32 better than the CUDA pod did).
  `pyproject.toml` raises the floor to `torch>=2.13`.
- **The instrument did not move measurably** (B3 and B4, the measurements B0 exists to
  predict). B4's resolution is about ±0.04, so where B0 counsels caution, the "instrument
  moved" row's remedy is followed wherever it costs nothing: **later local comparisons
  against p2_g/0060 use the 2.14.1 reps** (`runs/p2_g_bridge214/`). A cross-stack
  comparison against an older 2.12.1 reference whose margin is under ~0.04 is
  re-measured on 2.14.1 before it is read.
- **Rollback** is not needed. The 2.12.1 freeze stays in the uv cache until this is
  accepted.

Records: `runs/p2_g_bridge/bridge_local_2141.json` (B2); `runs/p2_g_bridge214/ladder.jsonl`
(B3, B5), `reps_iter0060_r{1,2,3}.jsonl` and `p2_verdict_reps.json` (B4). The forward
dumps (B0, 168 MB each) stayed outside the repo; `scripts/forward_dump.py` regenerates them.
