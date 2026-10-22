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

- Loads a 6.5M-param `ChessNet` with the **dataclass defaults** in
  `src/tinyalphazero/config.py` (dropout 0.1, legality auxiliary on), or with the
  Phase 1 YAML named by `--config` (`configs/phase1_c.yaml`, `configs/phase1_d.yaml`
  are the pinned rounds). Do **not** pass `configs/default.yaml` here: it is the
  Phase 2 campaign config (`dropout 0.0`, `legality_loss_weight 0.0`) and would train
  Phase 1 with the legality auxiliary off, making the `legality_auc ≥ 0.995` gate
  unmeetable.
- Builds a mixed DataLoader over random play (value weight 0.25) and the mate-in-1
  anchors (value weight 1.0), sampled with per-sample weights. How the configured
  mix `train.phase1_mate1_mix` becomes a batch fraction is set by
  `train.phase1_anchor_weighting`: `exact` makes a mix of 0.05 **5.00 %** of every
  batch; the dataclass default `legacy` reproduces the double normalisation every
  round up to the anchor v2 trained under, which realises 0.05 as **5.25 %**
  (0.08/0.07 as 9.17 %/8.02 %). The run log prints the realised fraction
  (`anchor weighting=... | realised fractions: mate1=...`), and the embedded
  config of a checkpoint says which weighting produced it.
- Trains with AdamW, lr cosine 3e-4 → 3e-5, batch 1024, bf16 autocast on GPU.
- Uses **unmasked** policy CE (`phase1_policy_masked=False`). Masking the CE gives
  illegal logits zero gradient, so `top1_legal_rate` can never train; see the
  `runs/phase1_a` post-mortem in `PROGRESS.md`.
- Writes the run's provenance first: `runs/phase1_a/config.yaml` (the resolved
  config, seed included) and `runs/phase1_a/provenance.json` (git SHA, config
  fingerprint, torch version, device, the `check_env` report).
- Logs a progress line every 200 steps with **per-component** losses, and per-epoch
  metrics to `runs/phase1_a/metrics.jsonl`. The per-epoch metrics are computed on
  `--eval-data` / `--eval-mate1`; if either is omitted the script falls back to the
  training dir **and says so** — treat that warning as a reason to stop.
- Saves epoch checkpoints (`epoch_01.pt`, `epoch_02.pt`, `epoch_03.pt`) and `final.pt`.

### Expected console output

```
Phase 1 training | device=cuda | bf16=True | steps=28008/28008 | batch=1024 | workers=8 | masked_policy=False | compile=False
  anchor weighting=exact | realised fractions: mate1=0.0500
  step    200/28008  loss=9.4715  policy=8.2717  promo=0.9710  value=0.1882  leg=0.0406  lr=3.00e-04
  step    400/28008  loss=8.2164  policy=6.8483  promo=1.1518  value=0.1745  leg=0.0419  lr=2.99e-04
  ...
Epoch 1 | loss=5.4120 (policy=3.6210) | mate2_top16=nan | mate1_top8=0.8210 | auc=0.9812 | wdl_anchor=0.7431 | top1_legal=0.9123ⁱ | mate1_solve=0.4210ⁱ
Saved runs/phase1_a/epoch_01.pt
```

The `ⁱ` marks informational metrics (retired gates, see Step 3); `mate2_top16` is
`nan` unless `--eval-mate2` is given.

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

By epoch 3 the gated metrics should be at or near their gates: `auc` ≥ 0.995,
`mate1_top8` ≥ 0.97, `wdl_anchor` ≥ 0.80. `top1_legal` and `mate1_solve` are
informational since BLOCKED-2026-08-06.

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

A resumed run is the run the checkpoint describes:

- **Config comes from the checkpoint**, not from `--config` or the dataclass
  defaults: epochs (and so the cosine `T_max`), batch size, LR start/end, the anchor
  mixes and weighting, and the seed. `--config`, `--epochs`, `--batch` or `--seed`
  that disagree with it are refused with an error naming both values; drop them, or
  start a new run with `--init-weights` if a different config is the intention.
- **Data order is the uninterrupted run's.** Each epoch's sample multiset is derived
  from `(seed, epoch)`, so resumed epoch 3 draws exactly what uninterrupted epoch 3
  would have, not a replay of epoch 1.
- **Optimizer state is restored and the LR schedule is replayed** from step 0 to
  the checkpoint's step, so the resumed trajectory matches the uninterrupted one
  (verified bit-identical on the mini config, CPU).
- `metrics.jsonl` is appended to; `provenance.json` gains a `resumes` entry with the
  checkpoint, step and the SHA of the resuming code; a run dir whose `config.yaml`
  disagrees with the checkpoint is refused.

### Custom config

```bash
# Use mini config for a quick smoke run (< 5 min on CPU). Pass the held-out dirs:
# without --eval-mate1 the gated mate-in-1 metrics are measured on the training
# anchors (the script warns), and without any mate-in-1 dir they are measured on
# positions the metric code generates itself.
VIRTUAL_ENV=.venv .venv/bin/python scripts/train_phase1.py \
    --data runs/data/phase1 \
    --mate1 runs/data/mate1 \
    --eval-data runs/data/eval_rand \
    --eval-mate1 runs/data/eval_mate1 \
    --out runs/smoke \
    --config configs/mini.yaml \
    --epochs 1
```

---

## Step 3 — Evaluate

Evaluate on the **held-out** sets from Step 1, never on the training dirs:

```bash
VIRTUAL_ENV=.venv .venv/bin/python scripts/eval_phase1.py \
    --ckpt runs/phase1_a/final.pt \
    --data runs/data/eval_rand \
    --mate1 runs/data/eval_mate1 \
    [--eval-mate2 runs/data/eval_mate2_1k]
```

`--mate1` is required for a verdict: two of the three gates are anchor-set
metrics, and without it they print `not measured` and the script exits 1 rather
than scoring positions it generated itself. `--eval-mate2` adds the round-d gate.
The header reports the denominators actually used (`min(n, len(dataset))`).

### Acceptance table (§5 gates, as revised by BLOCKED-2026-08-06 and -08-08)

```
========================================================================
Metric                            Value          Gate    Pass
------------------------------------------------------------------------
  legality_auc                     0.9967       ≥ 0.995    PASS
  wdl_anchor_mass                  0.8340        ≥ 0.80    PASS
  mate1_top8_masked                0.9850        ≥ 0.97    PASS
  mate2_top16_coverage             0.9620        ≥ 0.95    PASS   (only with --eval-mate2)
------------------------------------------------------------------------
  mate2_solve_rate                 0.6100  informational (top-1)
  top1_legal_rate                  0.9934  retired 2026-08-06 (was ≥ 0.99)
  mate1_solve_rate                 0.9210  retired 2026-08-06 (was ≥ 0.90)
  top1_exact_rate                  0.2810  informational
  mate1_top3_masked                0.9700  informational
  mate1_top16_masked               0.9990  informational
========================================================================

All gates PASSED — ready for Phase 2.
```

The script exits 0 on pass, 1 on any failure (including "not measured").

### Metric definitions

| Metric | Definition |
|--------|-----------|
| `legality_auc` | ROC AUC of the auxiliary legality head's per-square predictions against the true legal mask |
| `wdl_anchor_mass` | Mean P(win) assigned by the value head on mate-in-1 positions (side to move wins) |
| `mate1_top8_masked` | Fraction of mate-in-1 positions whose mating move is in the masked policy's top 8 — the considered-set question search actually asks |
| `mate2_top16_coverage` | Fraction of mate-in-2 positions with **any** key move in the masked top 16 (Gumbel's considered-set size) |
| `top1_legal_rate` | Retired. Fraction of positions where the raw unmasked policy argmax is a legal move |
| `top1_exact_rate` | Informational. Fraction where the masked policy argmax matches the played move (supervised accuracy) |
| `mate1_solve_rate` | Retired. Fraction of mate-in-1 positions where the masked policy argmax is the mating move — equals `mate1_top1_masked` |

---

## Troubleshooting

**`top1_legal_rate` plateaus below 0.98**
The most likely cause is an encoding bug. Re-run the Markov fuzz:
```bash
VIRTUAL_ENV=.venv .venv/bin/python scripts/fuzz_encoding.py --fuzz 1000000
```
Any violation printed here invalidates the dataset — regenerate from scratch with a fresh seed.

**`mate1_top8_masked` below 0.97 after 3 epochs**
Check the mate1 mixture is actually being sampled: the run log's `realised fractions:
mate1=...` line says what share of each batch came from the anchors, and
`runs/phase1_a/metrics.jsonl` should show `mate1_top8_masked` rising each epoch. If it's
flat, the mate1 DataLoader path may not be loading correctly. Also confirm the mate1
dataset has genuine checkmates:
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
# this box (see "Non-finite gradients" above); nothing else should hold the GPU,
# so the run goes under `gpu hold` (below).
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/train_phase2.py \
    --config configs/default.yaml \
    --init runs/phase1_anchor.pt \
    --out runs/p2_a \
    --iterations 50
```

### Sharing the GPU: `gpu hold`

The box's GPU is shared with a local LLM (`llama-server.service`, ~45 GB, Vulkan)
and ComfyUI, both managed by `~/.local/bin/gpu`. Memory is not the conflict — the
network is small — compute is: a co-tenant makes the logged throughput and wall
clock incomparable with earlier rounds (campaign rule 6). `gpu hold --by WHO --
CMD...` runs CMD with the GPU to itself:

| When | What `gpu hold` does |
|---|---|
| ComfyUI has queued jobs | refuses (exit 1); run it again when they finish |
| another hold is live | refuses (exit 1) |
| start | asks an idle ComfyUI to unload; stops llama-server (and anything else in the tool's `DROPPABLE`) whatever VRAM is free; writes a lease bound to CMD's pid |
| while CMD runs | `gpu restore`, Comfy Studio's Resume button and the 5-minute `comfyui-idle-unload` timer all leave the lease alone |
| CMD exits (any status) | restarts llama-server through its own `gpu guard` and waits for `/health` (~45 s); exits with CMD's status. If ComfyUI is mid-job at that moment, the restart is left to the timer, which does it once ComfyUI is idle |
| the wrapper is killed but CMD runs on | the lease stays bound to CMD |
| CMD dies without the wrapper restoring | the timer gives the GPU back on its next tick (ComfyUI idle ≥ 10 min) |

`gpu status` shows a live hold as `held … by tinyalphazero (pid N)`. Ctrl-C reaches
the training process directly; it is not intercepted. ComfyUI *can* still be used
during a run — the hold does not block it — and if it is, that stretch of the log
is a shared-GPU stretch. On a machine without the tool, drop the `gpu hold … --`
prefix.

### Resuming

Killing the process at any point is safe, and `--resume` is how you come back:

```bash
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/train_phase2.py --resume runs/p2_a --iterations 50
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
    config.yaml      # resolved config, seed included — what --ckpt resumes
    provenance.json  # git SHA, config fingerprint, torch/device, check_env report
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
