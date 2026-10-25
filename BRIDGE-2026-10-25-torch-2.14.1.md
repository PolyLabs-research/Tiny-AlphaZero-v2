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
