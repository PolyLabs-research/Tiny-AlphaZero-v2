# How to Train TinyAlphaZero v2

This guide covers the Phase 1 supervised bootstrap pipeline from raw data generation through model evaluation.

---

## Prerequisites

```bash
# Install dependencies (run once)
~/.local/bin/uv pip install -e ".[dev]"

# Verify environment
VIRTUAL_ENV=.venv .venv/bin/python scripts/check_env.py
```

Expected output shows torch version, device (CPU or ROCm/CUDA), and bf16 matmul throughput. Phase 1 runs on CPU but GPU is ~10–20× faster.

---

## Step 1 — Generate training data

Two datasets are needed: random-play games and mate-in-1 anchors.

```bash
# ~2M positions from 50K random games (~5–10 min on CPU, ~1 min on GPU)
VIRTUAL_ENV=.venv .venv/bin/python scripts/generate_phase1.py \
    --games 50000 \
    --out runs/data/phase1 \
    --seed 1

# 20K mate-in-1 anchor positions (~4 min on CPU)
VIRTUAL_ENV=.venv .venv/bin/python scripts/generate_phase1.py \
    --mate1 20000 \
    --out runs/data/mate1 \
    --seed 2
```

Also generate the held-out evaluation sets. Metrics must never be computed on the
training dirs:

```bash
VIRTUAL_ENV=.venv .venv/bin/python scripts/generate_phase1.py \
    --games 500 --out runs/data/eval_rand --seed 99
VIRTUAL_ENV=.venv .venv/bin/python scripts/generate_phase1.py \
    --mate1 500 --out runs/data/eval_mate1 --seed 100
```

Both write memmap files + a `meta.json` sidecar. Expected sizes:

| Dataset | Positions | Disk |
|---------|-----------|------|
| `runs/data/phase1` | ~9.5M | ~5.3 GB |
| `runs/data/mate1` | 20K | ~11 MB |
| `runs/data/eval_rand` | ~95K | ~53 MB |
| `runs/data/eval_mate1` | 500 | ~300 KB |

> 50K games average ~191 plies, so `phase1` is ~9.5M positions — not the ~2M this
> guide used to claim. Budget disk and epoch time accordingly.

**Check the mode, not just the directory name.** `runs/data/mate1` was once
generated with `--games` instead of `--mate1`, producing 3.8M random positions
under the anchor name; that alone cost a 9.5-hour training run. `train_phase1.py`
and `eval_phase1.py` now call `assert_dataset_mode()` and refuse to start on a
mismatch, but verify by hand when in doubt:

```bash
VIRTUAL_ENV=.venv .venv/bin/python -c "
import json; from pathlib import Path
for d in ['runs/data/phase1','runs/data/mate1','runs/data/eval_rand','runs/data/eval_mate1']:
    m = json.loads(Path(d,'meta.json').read_text())
    g = m['generation_config']
    print(f\"{d:24} n={m['n']:>9}  mode={g['mode']:<12} argv={' '.join(g.get('argv',[])[1:])}\")
"
```

To inspect:

```bash
VIRTUAL_ENV=.venv .venv/bin/python -c "
import json; from pathlib import Path
for d in ['runs/data/phase1', 'runs/data/mate1']:
    m = json.loads(Path(d, 'meta.json').read_text())
    print(d, '->', m['n'], 'positions')
"
```

---

## Step 2 — Train (Phase 1)

```bash
VIRTUAL_ENV=.venv .venv/bin/python scripts/train_phase1.py \
    --data runs/data/phase1 \
    --mate1 runs/data/mate1 \
    --eval-data runs/data/eval_rand \
    --eval-mate1 runs/data/eval_mate1 \
    --out runs/phase1_a \
    --epochs 3 \
    --seed 42
```

Add `--max-steps 600` for a timing/smoke slice that does not commit to the full run.

### What happens

- Loads a 6.5M-param `ChessNet` with the default config (`configs/default.yaml`).
- Builds a mixed DataLoader: 95% random-play (value weight 0.25) + 5% mate-in-1 (value weight 1.0), sampled via `WeightedRandomSampler`.
- Trains with AdamW, lr cosine 3e-4 → 3e-5, batch 1024, bf16 autocast on GPU.
- Uses **unmasked** policy CE (`phase1_policy_masked=False`). Masking the CE gives
  illegal logits zero gradient, so `top1_legal_rate` can never train; see the
  `runs/phase1_a` post-mortem in `PROGRESS.md`.
- Logs a progress line every 200 steps with **per-component** losses, and per-epoch
  metrics to `runs/phase1_a/metrics.jsonl`.
- Saves epoch checkpoints (`epoch_01.pt`, `epoch_02.pt`, `epoch_03.pt`) and `final.pt`.

### Expected console output

```
Phase 1 training | device=cuda | bf16=True | steps=28008/28008 | batch=1024 | workers=8 | masked_policy=False
  step    200/28008  loss=9.4715  policy=8.2717  promo=0.9710  value=0.1882  leg=0.0406  lr=3.00e-04
  step    400/28008  loss=8.2164  policy=6.8483  promo=1.1518  value=0.1745  leg=0.0419  lr=2.99e-04
  ...
Epoch 1 | loss=5.4120 (policy=3.6210) | top1_legal=0.9123 | mate1_solve=0.4210 | auc=0.9812 | wdl_anchor=0.7431
Saved runs/phase1_a/epoch_01.pt
```

**Read the components, not the total.** The totals here are not comparable to the
old guide's numbers:

- Initial total ≈ **10.6**, with policy at ln(4096) ≈ **8.32**, falling steeply.
  A flat policy term from step 1 is the signature of the `runs/phase1_a` bug.
- Converged policy CE returns to ≈ **3.1–3.3** = E[ln n_legal]. That is the *same*
  number the failed run showed — but reached by learning rather than granted by the
  mask. Judge by `top1_legal_rate`, not by the policy value.
- Promo CE has an irreducible **ln(4) ≈ 1.39** floor: random play promotes uniformly
  over four pieces.
- Converged total ≈ **4.8–5.5** is expected and fine.

By epoch 3, `top1_legal` should approach or exceed 0.99.

### Non-finite gradients

If you see lines like:

```
  step   1423: non-finite gradient (norm=nan), update skipped (1 total)
```

that is the NaN guard. On ROCm/gfx1151 the backward pass has gone non-finite
nondeterministically while the forward loss was still finite; a single such update
poisons the weights and makes every later step NaN. `train_step` now skips the update
instead, so a transient costs one batch. Skips are logged per epoch as
`skipped_steps` in `metrics.jsonl`, and the run aborts if they exceed 1% of steps.

An occasional skip is tolerable. A steady trickle is not — see the ROCm SDPA note in
`PROGRESS.md` and try math-only attention:

```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
```

### Wall-clock targets

| Hardware | Expected time |
|----------|--------------|
| ROCm/gfx1151 (Strix Halo) | < 1 hour |
| NVIDIA RTX 3090 | ~45 min |
| CPU (16-core) | ~6–8 hours |

The first real run took **9.5 hours** against the "< 1 hour" target. Two reasons:
the dataset is 9.5M positions (not the ~2M assumed above), and the loader was
`num_workers=2` with per-sample random memmap reads. The default is now
`--workers 8` with `prefetch_factor=4`. Time a slice before committing:

```bash
VIRTUAL_ENV=.venv .venv/bin/python scripts/train_phase1.py \
    --data runs/data/phase1 --mate1 runs/data/mate1 \
    --eval-data runs/data/eval_rand --eval-mate1 runs/data/eval_mate1 \
    --out runs/_timing --max-steps 600 --workers 8
```

If the step rate is far off GPU-bound for a 6.5M-param model, it is the input
pipeline, not the GPU — raise `--workers` before anything else.

### Resuming from a checkpoint

```bash
VIRTUAL_ENV=.venv .venv/bin/python scripts/train_phase1.py \
    --data runs/data/phase1 \
    --mate1 runs/data/mate1 \
    --eval-data runs/data/eval_rand \
    --eval-mate1 runs/data/eval_mate1 \
    --out runs/phase1_a \
    --ckpt runs/phase1_a/epoch_02.pt
```

The optimizer state and LR schedule are restored automatically.

### Custom config

```bash
# Use mini config for a quick smoke run (< 5 min on CPU)
VIRTUAL_ENV=.venv .venv/bin/python scripts/train_phase1.py \
    --data runs/data/phase1 \
    --out runs/smoke \
    --config configs/mini.yaml \
    --epochs 1
```

---

## Step 3 — Evaluate

```bash
VIRTUAL_ENV=.venv .venv/bin/python scripts/eval_phase1.py \
    --ckpt runs/phase1_a/final.pt \
    --data runs/data/phase1 \
    --mate1 runs/data/mate1
```

### Acceptance table (§5 gates)

```
==================================================================
Metric                        Value          Gate    Pass
------------------------------------------------------------------
  top1_legal_rate             0.9934      ≥ 0.99    PASS
  mate1_solve_rate            0.9210      ≥ 0.90    PASS
  legality_auc                0.9967     ≥ 0.995    PASS
  wdl_anchor_mass             0.8340      ≥ 0.80    PASS
------------------------------------------------------------------
  top1_exact_rate             0.2810          (informational)
==================================================================

All gates PASSED — ready for Phase 2.
```

The script exits 0 on pass, 1 on any failure.

### Metric definitions

| Metric | Definition |
|--------|-----------|
| `top1_legal_rate` | Fraction of positions where the raw unmasked policy argmax is a legal move — measures whether the model has learned piece mobility without relying on the mask |
| `top1_exact_rate` | Fraction where the masked policy argmax matches the played move (supervised accuracy) |
| `mate1_solve_rate` | Fraction of mate-in-1 positions where the masked policy argmax is the mating move — no search |
| `legality_auc` | ROC AUC of the auxiliary legality head's per-square predictions against the true legal mask |
| `wdl_anchor_mass` | Mean P(win) assigned by the value head on mate-in-1 positions (side to move wins) |

---

## Troubleshooting

**`top1_legal_rate` plateaus below 0.98**
The most likely cause is an encoding bug. Re-run the Markov fuzz:
```bash
VIRTUAL_ENV=.venv .venv/bin/python -m tinyalphazero.encoding --fuzz 1000000
```
Any violation printed here invalidates the dataset — regenerate from scratch with a fresh seed.

**`mate1_solve_rate` below 0.85 after 3 epochs**
Check the mate1 mixture is actually being sampled: inspect `runs/phase1_a/metrics.jsonl` to confirm `mate1_solve_rate` rises each epoch. If it's flat, the mate1 DataLoader path may not be loading correctly. Also confirm the mate1 dataset has genuine checkmates:
```bash
VIRTUAL_ENV=.venv .venv/bin/python -m pytest tests/test_data.py::test_mate_in_1_move_is_checkmate -v
```

**`legality_auc` below 0.99**
The auxiliary legality head is not converging. Verify `cfg.model.aux_legality=True` in the config and that `legality_loss_weight` is not accidentally 0.

**Loss is NaN from step 1**
Almost always a data issue. Check that the played move for every position is actually in the legal mask:
```bash
VIRTUAL_ENV=.venv .venv/bin/python -m pytest tests/test_data.py::test_stored_legal_bitset_matches_board_reconstruction -v
```

**Out of GPU memory**
Reduce `--batch` (e.g. `--batch 512`). The default 1024 requires ~6 GB VRAM; 512 requires ~3 GB.

---

## Step 4 — Phase 2 self-play loop (spec §7)

Phase 1 is a supervised bootstrap; Phase 2 is the AlphaZero loop proper. One
process alternates generate → train → snapshot, forever, from the Phase 1 anchor.

```bash
# The golden mini-run first — 2 iterations, CPU, under 3 minutes. If this fails,
# nothing below is worth starting.
make golden

# A campaign. TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 is the blessed stack on
# this box (see "Non-finite gradients" above); nothing else should hold the GPU.
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python scripts/train_phase2.py \
    --config configs/default.yaml \
    --init runs/phase1_anchor.pt \
    --out runs/p2_a \
    --iterations 50
```

### Resuming

Killing the process at any point is safe, and `--resume` is how you come back:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python scripts/train_phase2.py \
    --resume runs/p2_a --iterations 50
```

The iteration counter, league pool, replay ring and optimizer state are
reconstructed from the run directory; the config is re-read from
`runs/p2_a/config.yaml`, not from `--config`, so a resumed run cannot silently
change what it is. RNG is *derived* from (run seed, iteration, stream) rather than
saved, so a resumed iteration draws exactly what an uninterrupted one would have.

At most one iteration's work is lost — never more, and never a corrupted buffer.
That is a tested property, not a claim: `pytest tests/test_phase2.py -m slow`
SIGKILLs a real run inside generation and inside training and requires it back.

### Reading the log

`runs/p2_a/iterations.jsonl` has one row per iteration, defined once in
`src/tinyalphazero/logschema.py`.

```bash
python -c "
from tinyalphazero.logschema import read_jsonl
for r in read_jsonl('runs/p2_a/iterations.jsonl'):
    print(f\"iter {r['iteration']:4d}  loss={r['loss']:.4f}  decisive={r['decisive_rate']:.3f}\"
          f\"  plies={r['mean_plies']:.0f}  H_prior={r['move1_prior_entropy']:.2f}\"
          f\"  {r['total_s']/60:.1f} min (gen {r['selfplay_s']/60:.1f} + train {r['train_s']/60:.1f})\")
"
```

The numbers to watch, in order of how early they warn you:

| Column | What it means when it moves |
|--------|-----------------------------|
| `total_s` split | Erratum 6's ≤ ~35 min budget, per stage. Over budget → profile the stage the row names, do not guess |
| `move1_prior_entropy` | §8's collapse detector. Falling toward 0 means the policy has stopped considering alternatives, and nothing outside its top-m is ever simulated again |
| `move1_policy_entropy` | Sharpness of the *improved policy*, i.e. of the training target. **This one is expected to fall fast and is not a collapse.** σ scales q̂ by `(c_visit + max_N)` ≈ 70, so the target sharpens the moment the value head can separate two moves — measured 2.909 → 0.226 across two iterations while the prior barely moved |
| `false_resign_rate` | `null` until a resign-disabled game crosses the threshold. Its first non-null value is a calibration reading, not a fault; §7 wants it under 5% |

### What is deliberately absent

**No gating.** The learner always trains on and generates from the latest weights.
v1's arena gate is deleted rather than fixed: at 4–100 games per iteration it
either does nothing or rejects real progress at random. The anchor ladder (§8) is
the progress signal instead.

---

## Step 5 — The anchor ladder (spec §8)

The progress signal. Offline, read-only, and re-runnable against any past
checkpoint — nothing here writes to a replay buffer or a checkpoint.

```bash
# standalone, against any checkpoint
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python scripts/eval_ladder.py \
    --ckpt runs/p2_a/ckpt_iter0010.pt --anchors random,greedy,phase1 \
    --out runs/p2_a/ladder.jsonl
```

### Running the M3 campaign

This is the whole thing, in one command:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python scripts/train_phase2.py \
    --config configs/default.yaml --init runs/phase1_anchor.pt \
    --out runs/p2_a --iterations 50 --ladder-every 10
```

~27 min per iteration plus a ~19 min ladder pass every tenth, so 50 iterations is
roughly 24 hours. Killing it is safe at any point; `--resume runs/p2_a` continues,
and a ladder pass is separately resumable — a matchup already in `ladder.jsonl` is
never recomputed.

### What the ladder measures, and what it deliberately does not

| Rung | Question | On the M3 go/no-go? |
|------|----------|---------------------|
| `random` | floor sanity — is the net broken in a way nothing else would catch | no |
| `greedy` | basic tactics against a 1-ply material grabber | no |
| `phase1` | **Elo vs the frozen anchor — this is M3's criterion** | **yes** |
| `stockfish:N` | external calibrated scale | no — M4, and skipped when the binary is absent |
| mate-in-2 suite | no-regress against the pinned 0.985 baseline, and ≥ 0.95 absolute | **yes** |
| mate-in-3 suite | tactical depth trajectory | no — informational, no gate |
| watchlist | the three residual FENs, by name, with the value head's estimate on each key child | no — the mechanism behind the gate |
| ECE, book entropy | value calibration and the collapse detector | no |

Two sim budgets, never conflated: **matches at 48** (`eval.eval_match_sims`),
**suites at 128** (`eval.suite_sims`). 128 is the frozen definition of the
mate-in-2 gate — every recorded gate number was measured at it — so running the
gate at 48 would not re-measure it, it would replace it.

### Reading a ladder row

```bash
python -c "
from tinyalphazero.logschema import read_jsonl
for r in read_jsonl('runs/p2_a/ladder.jsonl'):
    if r['row_type'] == 'match':
        print(f\"iter {r['iteration']} vs {r['anchor']:8s} score={r['score']:.3f} \"
              f\"Elo {r['elo']:+.1f} [{r['elo_lo']:+.1f}, {r['elo_hi']:+.1f}]\")
"
```

Three things that look like problems and are not:

- **`Elo ≥ +636 (saturated)`** — a clean sweep has unbounded implied Elo. The
  score is clamped to half a game, the resolution of the measurement, and
  `elo_clamped` says so. Expect it against `random` forever.
- **A degenerate CI on a self-match.** A checkpoint against itself scores exactly
  0.500 on every opening pair by construction — the colour-swapped games are the
  same game with the labels swapped — so the interval collapses onto zero. That is
  the null working, not the bootstrap failing.
- **`ece: null`** — calibration samples the `phase1` matchup, so a pass without it
  measures nothing. `calibration_note` says which.

---

## Inspecting metrics

```bash
# Print the metrics log as a table
VIRTUAL_ENV=.venv .venv/bin/python -c "
import json
from pathlib import Path
lines = Path('runs/phase1_a/metrics.jsonl').read_text().strip().split('\n')
for line in lines:
    m = json.loads(line)
    print(f\"step={m['step']:5d}  loss={m['loss']:.4f}  legal={m['top1_legal_rate']:.4f}  mate1={m['mate1_solve_rate']:.4f}  auc={m['legality_auc']:.4f}\")
"
```

---

## File layout after a complete run

```
runs/
  data/
    phase1/          # random-play memmap dataset (~1.3 GB)
    mate1/           # mate-in-1 anchor dataset (~13 MB)
  phase1_a/
    epoch_01.pt      # checkpoint after epoch 1
    epoch_02.pt
    epoch_03.pt
    final.pt         # same as epoch_03.pt, canonical name for downstream chunks
    metrics.jsonl    # one JSON line per epoch
  p2_a/              # a Phase 2 campaign (chunk 9)
    config.yaml      # resolved config — the resume reads THIS, not --config
    provenance.json  # git SHA, torch/device, init checkpoint
    state.json       # iteration counter, league pool, replay head; written last
    iterations.jsonl # one logschema.IterationRow per iteration
    ckpt_iter0000.pt # the initial weights; iteration 0 pool = {this}
    ckpt_iter0001.pt # ~78 MB each (weights + AdamW moments), keep-all
    replay/          # memmap ring, ~540 MB at the 500K default capacity
```

The `final.pt` checkpoint is the input to Chunk 5 (MCTS) and all subsequent phases.
