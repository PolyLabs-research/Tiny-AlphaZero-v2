# TinyAlphaZero v2 — Build Progress

## Status: Phase 1 CLOSED (2026-08-06, tag `phase1-closed`); Chunk 5 is next

Chunks 0–4 are **done and signed off**. Phase 1 closed on `runs/phase1_anchor.pt`
(= `runs/phase1_c/final.pt`, step 56,484) after three rounds; two of the four original
§5 gates were retired to informational rather than lowered — reasoning in
`BLOCKED-2026-08-06-phase1-gates.md`, campaign log in `TRAINING_STRATEGY.md`. Chunks
5–12 are not started. See "Phase 1 post-mortem" below before touching training code,
and "Readiness 2026-08-06" at the end for the close-out audit.

---

## Completed Chunks

### Chunk 0 — Repository skeleton and configuration ✅

**Files created:**
- `pyproject.toml` — uv-managed package; deps: torch, numpy, python-chess, pyyaml, pytest, hypothesis, ruff
- `src/tinyalphazero/__init__.py` — version 0.1.0
- `src/tinyalphazero/config.py` — full dataclass config tree
- `tests/test_config.py` — 15 tests, all passing
- `scripts/check_env.py` — env/device/VRAM/bf16 throughput check
- `.gitignore`, `Makefile`
- `configs/default.yaml`, `configs/mini.yaml`, `configs/puct_ablation.yaml`

**Config structure:** `Config` → `ModelConfig`, `MCTSConfig`, `SelfPlayConfig`, `TrainConfig`, `EvalConfig`. All spec defaults from §4.1, §6.2, §7 are encoded. `load_config` / `save_config` round-trip exactly; `config_fingerprint` is a SHA-256 of canonical YAML.

**DONE-WHEN gates:** all green.

---

### Chunk 1 — Encoding (spec §3) ✅

**Files created:**
- `src/tinyalphazero/encoding.py`
- `tests/test_encoding.py`

**What's in encoding.py:**
- All vocabulary constants: `ES`, `WP`–`WK`, `BP`–`BK`, `TURN_W/B`, `CWK/CWQ/CBK/CBQ Y/N`, `EP_NONE`/`EP_A`–`EP_H`, `HM_0`–`HM_7`, `REP_0`–`REP_2`; `VOCAB_SIZE=43`, `SEQ_LEN=72`
- `encode_board(board, repetitions) → (72,) int8` — EP token only set when `has_legal_en_passant()` is True (the load-bearing Markov fix)
- `decode_board(tokens) → chess.Board` — halfmove clock set to bucket lower bound
- `halfmove_bucket(clock) → int`
- `encode_move(move) → (from, to, promo)` — promo: 0=none, 1=N, 2=B, 3=R, 4=Q
- `decode_move(from_sq, to_sq, promo) → chess.Move`
- `legal_mask_ft(board) → (64,64) bool`
- `legal_moves_packed(board) → (512,) uint8` bitset over from\*64+to
- `_repetitions_from_board(board)` helper using `board.is_repetition()`
- `__main__` fuzz harness (`--fuzz N`) for Markov invariant checking

**Test results:** 25 tests passing (~40s due to 200K-position Markov fuzz in test suite).

**Fuzz result:** 1M positions, 976,699 unique encodings, 23,301 collisions — all with identical legal move sets. No invariant violations.

**One spec deviation noted:** `decode_move` does not raise `ValueError` for promo=0 on back-rank moves. The spec says "raise for pawn-to-last-rank" but this cannot be implemented geometrically without knowing the piece type — a bishop on d2 capturing e1 is geometrically identical to a pawn promotion. The round-trip guarantee (`decode_move(*encode_move(m)) == m` for all legal moves) takes precedence. Documented in the docstring.

**DONE-WHEN gates:** all green.

---

## Environment

- Python 3.12.3, PyTorch 2.12.1+cu130
- uv at `~/.local/bin/uv`; venv at `.venv/`
- Install: `VIRTUAL_ENV=/home/michael/Documents/dev/chess/.venv ~/.local/bin/uv pip install -e ".[dev]"`
- Tests: `VIRTUAL_ENV=.venv .venv/bin/python -m pytest tests/ -q`
- Lint: `VIRTUAL_ENV=.venv .venv/bin/python -m ruff check src/ tests/ scripts/`

---

### Chunk 2 — Phase 1 data generation (spec §5) ✅

**Files created:**
- `src/tinyalphazero/data.py`
- `scripts/generate_phase1.py`
- `tests/test_data.py`

**What's in data.py:**
- `PositionRecord` / `GameRecord` / `DatasetMeta` dataclasses
- `generate_random_games(num_games, rng, max_plies=200, min_plies=10)` — yields `GameRecord`; records tokens, move, legals, and outcome from side-to-move perspective; repetition count tracked via `board.is_repetition()`
- `generate_mate_in_1(num_positions, rng)` — harvests mate-in-1 positions as a by-product of random play (no rejection sampling); always outcome=+1
- `write_dataset(records, out_dir)` — accepts `Iterator[GameRecord | PositionRecord]`; writes five memmaps + `meta.json`; memmap layout: `boards.dat` (int8, N×72), `moves.dat` (int16, N×2), `promos.dat` (int8, N), `legals.dat` (uint8, N×512), `outcomes.dat` (int8, N)
- `Phase1Dataset(torch.utils.data.Dataset)` — opens memmaps read-only; `__getitem__` returns `dict[str, Tensor]` with keys `tokens, from_sq, to_sq, promo, legals, outcome`

**Test results:** 18/18 passing in ~4s (data tests don't need the heavy encoding fuzz).

**Performance:** 200 games → 37,990 positions in 3.1s; 500 mate-in-1 positions in 4.0s. Well within 60s budget.

**DONE-WHEN gates:** all green.

---

### Chunk 3 — Model (spec §4) ✅

**Files created:**
- `src/tinyalphazero/model.py`
- `tests/test_model.py`

**What's in model.py:**
- `ModelOutput` dataclass: `policy_ft (B,64,64)`, `promo (B,64,64,4)`, `value_wdl (B,3)`, `legality (B,64,64)|None`
- `masked_policy(policy_ft, legal_mask) → (B,4096) probs` — -inf on illegal moves before softmax
- `value_scalar(value_wdl) → (B,)` = P(win)−P(loss) from WDL logits
- `ChessNet(cfg: ModelConfig)` — 8-layer pre-norm transformer, GELU, no causal mask
  - Token embedding + factored rank+file positional encoding for squares 0–63 (rank_emb + file_emb, both (8, d_model)); 8-slot flag embedding for state tokens 64–71; table computed once per forward from current embedding parameters
  - Bilinear policy head: `W_from · h_i` dot `W_to · h_j` / sqrt(d_p) → 65K params vs 2.2M v1 cost
  - Promotion sub-head: split Linear(d,4)+Linear(d,4) summed, equivalent to Linear(2d,4) on [h_i;h_j] but avoids materialising (B,64,64,2d); broadcast to (B,64,64,4)
  - Value head: single learned query vector attends over all 72 tokens (softmax attention pool) → LayerNorm+Linear(d,d)+GELU+Linear(d,3)
  - Optional aux legality head (same bilinear, separate weights; None when `cfg.aux_legality=False`)
  - `encode(tokens) → (h_all, h_sq)` for training code needing raw features without head overhead
  - `promo_logits(h_sq, from_sq, to_sq) → (B,4)` — memory-efficient per-pair promo for training
  - `param_count() → {total, policy_head}`

**Test results:** 20/20 passing (78/78 total across all chunks).

**Param counts (default cfg):** total=6,536,199; policy_head=67,588 (< 200K gate ✅).

**DONE-WHEN gates:** all green.

---

### Chunk 4 — Phase 1 training (spec §5) ✅

**Files created:**
- `src/tinyalphazero/train.py`
- `scripts/train_phase1.py`
- `scripts/eval_phase1.py`
- `tests/test_train.py`

**What's in train.py:**
- `unpack_legal_mask(packed (B,512) uint8) → (B,64,64) bool` — LSB-first bitset unpacking, verified against `legal_mask_ft`
- `_policy_loss`, `_promo_loss`, `_value_loss`, `_legality_loss` — individual loss components
  - Policy: CE on played (from,to) under legal mask
  - Promo: CE on promotion sub-class (0-3), only on positions where promo > 0
  - Value: CE with per-sample weight (0.25 random-play / 1.0 mate anchors)
  - Legality: BCE on full 64×64 grid from aux head
- `train_step(batch, net, optimizer, device, cfg_train, use_bf16)` — single supervised step with grad clip, optional bf16 autocast; returns named loss dict
- `WeightedPhaseDataset` — wraps Phase1Dataset, injects `value_weight` field per sample
- `save_checkpoint` / `load_checkpoint` — standard format: `{model, optimizer, config, git_sha, step, phase}`; `load_checkpoint` validates model shape fields against an optional reference config
- `compute_metrics(net, data_dir, device, ...)` → `{top1_legal_rate, top1_exact_rate, mate1_solve_rate, legality_auc, wdl_anchor_mass}` — mate1 generated on-the-fly if no mate1_dir provided
- `_binary_auc` — Mann-Whitney AUC (O(n log n), no sklearn)

**What's in train_phase1.py:**
- `WeightedRandomSampler`-based mixed DataLoader (cfg.train.phase1_mate1_mix fraction of mate anchors)
- AdamW + CosineAnnealingLR from phase1_lr_start → phase1_lr_end over total steps
- Per-epoch checkpointing + JSONL metrics log
- Resume from `--ckpt` flag

**What's in eval_phase1.py:**
- §5 acceptance table: top1_legal ≥ 0.99, mate1_solve ≥ 0.90, AUC ≥ 0.995, wdl_anchor_mass ≥ 0.80
- Exit 0 if all gates pass, exit 1 otherwise

**Test results:** 20/20 (98/98 total across all chunks), lint clean.

**DONE-WHEN gates:** `pytest tests/test_train.py -q` — green. Full-run gates require the data generation step (50K games + 20K mate1).

---

---

## Phase 1 post-mortem — `runs/phase1_a` (39,135 steps, ~9.5h, all four §5 gates missed)

| metric | achieved | gate |
|---|---|---|
| top1_legal_rate | 0.000 | ≥ 0.99 |
| mate1_solve_rate | 0.065 | ≥ 0.90 |
| legality_auc | 0.981 | ≥ 0.995 |
| wdl_anchor_mass | 0.034 | ≥ 0.80 |

Total loss was flat across 39K steps (4.697 → 4.644). Three stacked causes; fixing
any one alone would not have passed.

### Cause 1 — the mate-in-1 anchor set contained random games

`runs/data/mate1/meta.json` read `mode: random_games, num_games: 20000`: the
generator was invoked with `--games` instead of `--mate1`, yielding 3.8M random
positions. `train_phase1.py` used that dir both as the training anchor mix and as
`mate1_dir` for metrics, so anchor supervision *and* the mate1/WDL metrics were
measuring noise. The generator CLI was never at fault — nothing downstream validated
the result.

**Fixed:** regenerated (`--mate1 20000 --seed 2`, 20,000 positions, all outcome=+1).
`write_dataset` now records full `sys.argv` in `meta.json`, and
`assert_dataset_mode()` makes `train_phase1.py` / `eval_phase1.py` refuse to start on
a mode mismatch.

### Cause 2 — the masked policy CE was a structural no-op on this data

`_policy_loss` masked illegal logits to −inf before the softmax, so illegal logits
received **exactly zero gradient** — while the §5 `top1_legal_rate` gate measures the
*unmasked* argmax. The two were incompatible as specified. Worse: on uniform-random
play the target is uniform-over-legal, and a masked softmax at init is already
uniform-over-legal, so the loss started at its own optimum (E[ln n_legal] ≈ 3.2) and
never moved. That also explains the flat total: policy 3.2 (at floor) + promo ln(4) =
1.39 (irreducible) + value 0.1 + legality 0.03 ≈ 4.7, matching the observed 4.68–4.70.

This was a defect in the plan/spec wording, not the implementation — chunk 4
specified "masked policy CE" while §5 gated unmasked legality.

**Fixed:** `TrainConfig.phase1_policy_masked = False`; `_policy_loss` and `train_step`
take a `masked` flag. Phase 2 will pass `masked=True`, where search-improved targets
over the legal set make masking correct. Plan wording corrected.

### Cause 3 — the aux legality head was detached from the trunk

`model.py` computed `self._legality(h_sq.detach())`, commented "so legality BCE cannot
corrupt trunk features". That inverts spec §4.4, where the head exists precisely to
give the trunk a dense supervised mobility signal. As shipped it was a linear probe:
BCE trained only `W_from_aux`/`W_to_aux`. Combined with cause 2, the trunk's only real
gradient for 9.5 hours was the 0.25-weighted value signal — which is also why AUC
plateaued at 0.981 under the 0.995 gate.

**Fixed:** `.detach()` removed.

### Minor

- **Step accounting:** `steps_per_epoch = ceil(n/batch) = 13045` but `drop_last=True`
  yields 13044, leaving a 3-step orphan "epoch 4". Now `len(loader)`.
- **Eval was not held out:** metrics were computed on the training set. Added
  `runs/data/eval_rand` (500 games, seed 99) and `runs/data/eval_mate1` (500, seed
  100); `--eval-data` / `--eval-mate1` wire them in.
- **Component logging:** only the total was logged. Had components been visible, a
  policy term pinned at 3.2 from step 1 would have made cause 2 obvious immediately.
  Now logged per 200 steps and per epoch.
- **Dataset size erratum:** 50K games average ~191 plies → **9,540,953** positions,
  not the ~2M the spec and runbook claimed.
- **Throughput — the input pipeline was NOT the bottleneck.** Measured on the real
  data at batch 1024:

  | stage | ms/step |
  |---|---|
  | data-only iterate (workers=2) | 19.2 |
  | data-only iterate (workers=8) | 2.2 |
  | H2D transfer | 0.1 |
  | **compute (fwd+bwd+step)** | **883.3** |
  | (forward alone) | 368.0 |

  883 of 899 ms is GPU compute, so `--workers 8` changed end-to-end throughput by
  nothing (0.899 vs 0.875 s/step — marginally worse). Routing the promo head through
  the unused memory-efficient `promo_logits()` instead of the dense `(B,64,64,4)`
  tensor also measured at 1.00×. **Projected full 3-epoch run: ~6.9 hours**, against
  the spec's "< 1 hour". The remaining lever is the trunk itself (d_model=256,
  8 layers, 72 tokens) on this iGPU, at roughly 8% of theoretical bf16 throughput —
  not the loader. `--workers 8`, `prefetch_factor=4` and `--max-steps` are kept
  regardless; they are correct, just not where the time was.

### Open issue — nondeterministic NaN gradient on ROCm

A 600-step slice after the fixes came back **all-NaN**. Diagnosis: the gradient goes
non-finite in the **backward** pass while the forward loss is still finite
(`grad_norm=nan` at step 0 with `loss=10.7573`). Once a NaN update lands, the weights
are poisoned and every subsequent step is NaN — which is why the whole run reported
NaN from step 200 on.

It is **not reproducible on demand**, and not data-dependent: the failing step and a
clean step produced a bit-identical forward loss (10.7573) on the same batch and seed,
then differed in the backward. Each loss term backwards fine alone and summed. A probe
of 8 × 50-step runs found zero recurrences:

| config | trials | NaN |
|---|---|---|
| bf16 + flash/mem SDPA | 3 | none |
| bf16 + math SDPA only | 3 | none |
| fp32 + flash/mem SDPA | 2 | none |

So the backend split is **not** established — the probe simply never reproduced it.
PyTorch warns that flash and mem-efficient SDPA are experimental on this AMD GPU,
which is the leading suspicion but is unproven.

**`torch.compile` makes it reproducible — and is unusable here.** Under
`torch.compile` the backward returns non-finite gradients on **~95% of steps**
(21 of 22 before the abort guard fired), from the very first step, with weights still
finite — so it is the compiled backward itself, not weight poisoning. That is strong
evidence the NaN is a ROCm kernel defect rather than anything in the loss or data.

This also invalidates a tempting optimisation: `torch.compile` benchmarks at 533 vs
880 ms/step (1.65×), but that measurement is meaningless — the compiled model was
skipping nearly every optimiser update. **Speed was benchmarked without checking
correctness.** The `--compile` flag is retained but documented as broken on this
platform; retest on newer ROCm.

**Mitigated, not root-caused.** `train_step` now checks the grad norm and the total
loss, skips `optimizer.step()` when either is non-finite, and reports
`skipped`/`grad_norm`. `train_phase1.py` counts skips, logs them per epoch, and aborts
if they exceed 1% of steps (min 20) rather than quietly training on a fraction of its
batches. A transient now costs one batch instead of the run. If skips are frequent in
the real run, try math-only SDPA
(`torch.backends.cuda.enable_flash_sdp(False)`, `enable_mem_efficient_sdp(False)`) or
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.

### Verified after the fix

A 600-step slice on the real data reproduces the predicted healthy signature: initial
total **10.76**, policy at **8.3196** ≈ ln(4096) = 8.3178, descending to ~6.85 by step
24 and ~6.3 by step 50. Regression tests added for both causes 2 and 3.

**How to read the corrected loss:** components, never the total. Converged policy CE
returns to ≈3.1–3.3 — the same number the failed run showed, but reached by learning
rather than granted by the mask. Judge by `top1_legal_rate`. Converged total ≈4.8–5.5
is expected.

---

## Readiness 2026-08-06 — Phase 1 close-out audit

**Gate:** `pytest -q` → **108 passed**; `ruff check src/ tests/ scripts/` → **All checks
passed**; tag `phase1-closed` present on `0c0cebd`.

| Task | Status |
|---|---|
| A1 tag closure point | ✅ `phase1-closed` on `0c0cebd`, tree clean |
| A2 checkpoint SHA audit | ✅ all three read `unknown`; helper had 2 defects, both fixed, 4 tests |
| A3 top-k into `compute_metrics` | ✅ one implementation; eval reproduces the BLOCKED table exactly |
| A4 anchor identity | ✅ `runs/ANCHORS.sha256` verifies — ⚠️ NAS copy **deferred**, no mount exists |
| A5 remote | ⚠️ **deferred** — `gh` not installed; commands printed below, nothing pushed |
| A6 benchmark record | ✅ 5 rows backfilled and parse; stale `--compile` version string fixed |
| A7 readiness gate | ✅ this section |

**Two open items carry real risk and both need Tom:** the anchor exists only on the same
physical device as the OS and the repo (A4), and there is no remote (A5). Neither is
fixable from inside this session without credentials.


### A2 — checkpoint git-SHA audit (verbatim)

```
runs/phase1_b/final.pt       git_sha='unknown'   step=28008 phase=1
runs/phase1_c/final.pt       git_sha='unknown'   step=56484 phase=1
runs/phase1_anchor.pt        git_sha='unknown'   step=56484 phase=1
```

All three carry the literal string `unknown`: chunks 0–4 ran with no `.git`, so no run
in the campaign has a real SHA. Provenance for rounds a–c is the campaign rows plus
dataset `meta.json` `argv`, not the checkpoint field.

`_git_sha()` had **two** defects, both fixed:

1. Sentinel was `"unknown"`, not the specified explicit `"no-git"`.
2. **It was cwd-dependent.** `git rev-parse` ran in the process working directory, so a
   run launched from inside a *different* repository would have silently stamped that
   repository's SHA onto the checkpoint. A wrong provenance record is worse than an
   absent one — nothing downstream can distinguish them. Now anchored to the package
   source tree via `git -C <repo_root>`.

Verified after fix — same answer from both locations, where it previously disagreed:

```
in-repo : '0c0cebd'
from /tmp: '0c0cebd'     (was 'unknown')
```

Tests added (4): success path, both unavailability branches (git binary missing /
not-a-repo) returning `NO_GIT_SENTINEL`, and a cwd-independence regression test.

### A4 — anchor identity and durability

```
276cfea67ed0c2578712c31edea9b7a75e5b1337cc67ae7f1326382b067ab5b1  runs/phase1_anchor.pt
```

`runs/ANCHORS.sha256` committed; `sha256sum -c runs/ANCHORS.sha256` → OK. The digest is
identical to `runs/phase1_c/final.pt`, confirming the anchor is a byte-exact copy and
not a re-save. The bytes stay untracked (75 MB); their *identity* is versioned. ROCm
nondeterminism means these bytes are **not** reproducible from the seed — if they are
lost, the Phase 2 ladder loses its anchor permanently.

**NAS copy: DEFERRED — no mount exists.** `mount | grep -i 8tb` → no match; no
cifs/nfs/smb mounts; nothing in `/etc/fstab`. Mounted volumes are `/media/michael/disk2`
(1.9T local NVMe, `nvme0n1p5`), `/media/michael/SCARLETT` (160 K), and `/mnt/roms` —
none is the Jetson share. Per the brief, Samba credentials were **not** configured.
Awaiting Tom. **The anchor currently exists only on `nvme1n1` — the same device as the
OS and the repo.** A second copy on `disk2` would at least survive a single-device
failure and needs no credentials.

### A5 — remote

**DEFERRED — `gh` is not installed** (`gh: command not found`), so there is no
authenticated account to confirm against. Nothing was pushed. Manual commands for Tom:

```bash
# after: sudo apt install gh   (or https://cli.github.com)
gh auth login
gh repo create Tiny-AlphaZero-v2 --private --source=. --remote=origin
git push -u origin main --tags
git ls-remote origin        # expect refs/tags/phase1-closed
```

### A6 — benchmark record

`benchmarks/results.jsonl` did not exist; backfilled with 5 dated rows from
TRAINING_STRATEGY.md, each stamped with the stack (`torch 2.12.1+rocm7.2`, HIP
7.2.53211) and flagged `"backfilled": true` — the `git_sha` on those rows is the
backfill commit, not the measurement commit, since the measurements predate the repo.

| config | result |
|---|---|
| eager, batch 1024 | 870 ms/step (1,177 samples/s) — baseline |
| eager + experimental SDPA | **564 ms/step, 1.54×, 0 NaN skips over 56,484 steps** |
| `torch.compile` | 100% non-finite gradients — unusable |
| eager, batch 2048 | 1,098 samples/s — *below* the 1024 baseline; a cliff, not a win |
| loader isolation | synthetic 870 vs real 902 ms — ~3.5% overhead, GPU-bound |

The experimental-SDPA row is the one that matters: it is both the fastest configuration
and the one with stability demonstrated at full scale, and it is what round `c` ran
under. `--compile` help text corrected — it cited `2.12.1+cu130`, a CUDA wheel that is
not what is installed; now cites the real ROCm build so "retest on newer ROCm" is
actionable by comparing `torch.__version__` / `torch.version.hip`.

`benchmarks/results.jsonl` was in `.gitignore`; removed, for the same reason `BLOCKED.md`
was removed from it — a dated, SHA-stamped record is worthless if it is not versioned.

---

## Next: Chunk 5 — Array-tree MCTS
