# TinyAlphaZero v2 — Build Progress

## Status: Phase 1 CLOSED (r2) on anchor v2 (2026-08-09, tag `phase1-closed-r2`); chunks 0–8 done; **chunk 9 is next**

Chunks **0–7** are done, reviewed and merged to `main`. Phase 1 first closed 2026-08-06 on
`phase1_c/final.pt`, was reopened 2026-08-08 by the chunk 6 mate-in-2 gate, and re-closed
2026-08-09 on **anchor v2 = `runs/phase1_d/final.pt`** with that gate at **98.5%** against its
original, never-moved 0.95 threshold. Two of the four original §5 gates remain retired to
informational rather than lowered — reasoning in `BLOCKED-2026-08-06-phase1-gates.md`, campaign
log in `TRAINING_STRATEGY.md`, gate story in `BLOCKED-2026-08-08-mate2-gate.md`.

Chunks 9–12 are not started; chunk 8's intake register is near the end of this file (S4 delivered, repetition ownership delivered). See
"Phase 1 post-mortem" below before touching training code, and "Readiness 2026-08-06" for the
first close-out audit.

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
| A4 anchor identity | ✅ `runs/ANCHORS.sha256` verifies; second copy on `disk2` verified — ⚠️ NAS copy **deferred**, no mount exists |
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
Awaiting Tom for the offsite copy.

**Interim second copy made** (Tom's call, 2026-08-08) —
`/media/michael/disk2/tinyalphazero-v2-anchors/phase1_anchor.pt`, on `nvme0n1`, a
different physical device from the repo's `nvme1n1`. Digest verified identical:

```
276cfea67ed0c2578712c31edea9b7a75e5b1337cc67ae7f1326382b067ab5b1
```

This survives a single-device failure but is **still not offsite** — same machine, same
room. The NAS copy remains open.

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

## Chunk 5 — Array-tree MCTS, synchronous single game ✅ (branch `chunk-05-mcts`, 2026-08-08)

Created `src/tinyalphazero/mcts/{tree,search}.py`, `scripts/generate_suites.py`,
`tests/test_mcts.py`, `tests/suites/mate_in_{1,2}.epd`. All four §6.4 gates pass.

### Gate results (verbatim)

```
$ pytest tests/test_mcts.py -q
40 passed in 2.07s

$ python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_1.epd --sims 16 --stub
mate_in_1.epd: 100/100 solved at 16 sims (100.0%)
exit=0

$ pytest -q
159 passed, 36 warnings in 57.21s

$ ruff check src/ tests/ scripts/
All checks passed!
```

| Gate | Result |
|---|---|
| 1. Mate-in-1, 16 sims, stub evaluator | **100/100** |
| 2. Invariant fuzz (`1 + ΣN(root) == sims+1`, `vloss == 0`, `W` finite) | pass, 85 random positions × {1,7,32} sims |
| 3. Pinned visit counts, 64 sims | pass, 2 positions |
| 4. `Board.copy` ban | pass, incl. the suite path |

### The design decision that makes gate 1 meaningful

The mate-in-1 suite has **min 23, median 43, max 64 legal moves**. PUCT with uniform
priors visits one fresh root edge per simulation, so at 16 sims a lazily-expanding
search touches under 40% of root edges and *cannot* see the mate — it would be
measuring branching factor, not search. Measured directly: with terminal detection
deferred to first traversal, the gate scored **49/100**.

So expansion resolves each child's terminal status eagerly (one push/pop per legal
move), and `_select_edge` uses that exact value in place of the completed-Q estimate
where it exists. Mates are then found on simulation 1 — `test_mate_in_1_is_found_on_the_first_simulation`
pins that. Cost is O(branching) push/pops per expansion; chunk 5 is explicitly
correctness-before-speed and chunk 7 can revisit after profiling.
`test_mate_in_1_suite_positions_exceed_the_sim_budget` asserts min branching > 16 so
the gate can never quietly become passable-by-accident.

### The pinned-visit gate needed two positions, not one

Under a uniform-policy zero-value stub, a 47-move midgame position at 64 sims produces
visit counts that are **all 1s and 2s** — correct behaviour (there is no signal to
concentrate on) but a weak canary that most math changes would not move. Added a
second pin: `6k1/5ppp/8/8/8/8/5PPP/R5K1 w` where `Ra8#` is a proven win, concentrating
**45 of 64 visits** with Q=+1.0. That vector moves if terminal detection, terminal-value
propagation, the backup sign, completed-Q, or `c_puct` changes. Both are asserted; the
near-uniformity of the first is itself asserted, so signal leaking into it also fails.

### Pre-existing flaky test, found and fixed

`test_train_step_loss_decreases[False]` failed once in four full-suite runs, then passed
5/5 in isolation. Cause: it seeded the *batch* but not the *model init*, so it consumed
whatever global torch RNG state preceding tests left behind — adding any test earlier in
the file could flip it, which is what the A2/A3 additions did. Margin measured over 10
inits: policy-CE ratio **5.6×–25.6× unmasked** vs **12.7×–39.8× masked** against a 4×
gate, so the unmasked path sits close enough to lose occasionally. Fixed with
`torch.manual_seed(0)`; **the 4× threshold is unchanged** — only the uncontrolled
variable was removed. Verified 131 passed × 4 consecutive full runs.

### Repetition off-by-one — found by the review's question 1, fixed

Eager terminal resolution creates a repetition hazard, and reviewing for it turned up a
live bug. `encode_board(board, repetitions)` documents *"occurred **earlier**"* (a first
visit is 0) while the threefold claim needs the total **including now**. Both were read
off one counter, so **every non-root expansion passed a count one too high** — measured
44 of 45 expansions wrong on a king-shuffle position, root the only correct one.

No gate could have caught it: the stub evaluator ignores that token, so it is invisible
until the real network arrives. It would then have fed a corrupted repetition signal to
every non-root position — the exact input spec §12.1 flags as the thing to watch for
shuffle-into-draw pathologies.

Fixed by naming the distinction instead of computing it inline: `_total_occurrences`
(counts now) and `_prior_occurrences` (excludes now), both derived from one stated
invariant — *`path_reps` always includes the position the board is standing on, root
included*. Establishing that for the root also corrected threefold detection on paths
that return to the root position, which previously under-counted.

Verified with python-chess as an independent oracle (the search pushes onto the live
board, so `_repetitions_from_board` sees the true path): **0 mismatches over 207
expansions** across three positions, from 44/45 wrong before.

**The oracle is now a permanent gate, not a one-time check** — "no gate could have caught
this" must not stay true. `test_repetition_token_fuzz_shuffle_heavy` fuzzes 7
shuffle-heavy positions (sparse material, few captures, so paths revisit constantly)
× {16, 64} sims, and `test_repetition_token_fuzz_with_real_move_history` runs the same
assertion on boards carrying genuine history, since a self-play board's prior-occurrence
count spans both the pre-search stack and the in-search path. Mutation-tested: with the
off-by-one reintroduced, **16 of the repetition tests fail**, so the gate has teeth.

Oracle validity condition, stated because it is load-bearing: it holds only when history
lives in the move stack, i.e. `repetitions_fn` contributes 0. A non-zero `repetitions_fn`
describes occurrences python-chess cannot see, and the assertion would be checking the
wrong thing.

### Chunk 6 required reading — the four terminal-interaction questions

Raised at the chunk 5 review. Two are answered by the chunk 5 implementation; two are
open and belong to chunk 6/7.

1. **Repetition-aware child terminality — ANSWERED.** The eager check consults the
   path-local counter, so terminality is path-aware, not a property of the position
   alone. Caching it on the edge is sound *because a tree path from root to any node is
   unique*, so the count that decided the flag cannot later change — that reasoning is
   the load-bearing part and it breaks the moment the tree becomes a DAG (transposition
   table), which chunk 7 should treat as a blocker, not an optimisation.
   Pinned by `test_child_terminality_accounts_for_the_path_not_just_the_position`.
2. **Budget accounting — ANSWERED: terminals DO consume budget.** A simulation ending on
   a terminal edge still increments that edge's `N` and counts as one simulation, so the
   identity is `1 + Σ N(root edges) == sims_run + 1` with no special case, and
   `sims_run == num_sims` whenever the root has edges. Sequential Halving in chunk 6
   must be written against *this* convention: its exact-budget arithmetic will not
   balance if it assumes terminal hits are free.
3. **Gumbel root interaction — OPEN.** Root terminal children need the same exact-value
   substitution inside σ(q̂) and the improved-policy completion that `_select_edge` does.
   A mate scored by completed-Q estimate at the root while non-root selection uses exact
   values is an inconsistency Gumbel absorbs silently into worse policy targets.
4. **Chunk 7 equivalence — OPEN.** Terminal short-circuits never reach the evaluator, so
   the batched runner must not count them as leaves awaiting a forward pass, or the
   batch-1 equivalence gate fails late and confusingly.

### `.gitignore` must-track test

Added `tests/test_gitignore_musttrack.py` (11 cases) after four separate incidents across
two sessions — `BLOCKED.md` ignored outright, bare `data/` blocking descent into
`runs/data/`, `benchmarks/results.jsonl` ignored, and `runs/**` swallowing
`runs/ANCHORS.sha256` on the very commit that added it. It runs `git check-ignore` over a
must-track list and, in the other direction, asserts checkpoints and `.venv` stay ignored
so a too-wide negation cannot start committing 75 MB blobs.

### Not done in this chunk (by instruction)

Chunk 6 was not started. `runs/phase1_anchor.pt` is untouched — every gate above runs
against the stub evaluator, and the network is first consumed by chunk 6's mate-in-2
gate. The branch is **not merged**; it awaits the chunks 5–7 review checkpoint.

---

## Chunk 6 — Gumbel root selection ⚠️ BLOCKED (branch `chunk-06-gumbel`, 2026-08-08)

Implemented in full: `mcts/gumbel.py`, Gumbel top-m + Sequential Halving, `improved_policy`,
PUCT+Dirichlet ablation control, `--ckpt` network evaluator. **185 tests pass, ruff clean.**

### Gate results (verbatim)

```
$ pytest tests/test_mcts.py -q
66 passed in 5.51s

$ python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_1.epd --sims 16 --stub
mate_in_1.epd: 100/100 solved at 16 sims (100.0%) | eval=stub root=gumbel | gate >= 100%

$ python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_1.epd --sims 48 --stub
mate_in_1.epd: 100/100 solved at 48 sims (100.0%) | eval=stub root=gumbel | gate >= 100%

$ python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_2.epd --sims 128 --ckpt runs/phase1_anchor.pt
mate_in_2.epd: 154/200 solved at 128 sims (77.0%) | eval=runs/phase1_anchor.pt root=gumbel | gate >= 95%

$ pytest -q
185 passed, 36 warnings in 59.91s
```

| Gate | Result |
|---|---|
| Gumbel mate-in-1, 48 sims, stub | **100/100** |
| Sequential Halving exact budget, 8 (n,m) combos | pass |
| `improved_policy` well-formed; >0.9 on a proven mate | pass |
| PUCT ablation control still 100/100 | pass |
| **mate-in-2 ≥95% @128 sims with the anchor** | **77.0% — FAILS** |

**BLOCKED after 3 fix attempts: `BLOCKED-2026-08-08-mate2-gate.md`.** Chunk 7 not started.

### This fires a return condition of the Phase 1 closure

`BLOCKED-2026-08-06` named "mate-in-2 < 95% @128 sims with this checkpoint" as return
condition 1. It has fired. The mate-in-1 half did not (100/100, stub, no network).

Diagnosis order was search-first as instructed. Stub 30.5% → anchor 77.0%, and the decisive
split: **95.4% solved *given* the forcing move was in the considered set**, which is only
75.5% of the time at m=16. Search is at the gate; *consideration* is not. Gumbel takes top-m
by `g + logits`, so a move the prior ranks below ~16th is never simulated — network prior
`top16 = 83%` on these positions is a hard ceiling on a 95% gate.

Neither knob rescues it: m ∈ {16,24,32,48,128} → 76/73.5/80/81/81%; sims ∈ {128,256,512,1024}
→ 81.7/81.7/91.7/86.7% (60-position subset). Full evidence and the three attempts in the
BLOCKED doc.

The tension is real: the Phase 1 retirement argued mate-in-1 is "search's job", which holds
(100% with no network). It does not extend to mate-in-2, where the first move must be
*considered*, and consideration is the policy prior's job — the exact quality Phase 1 stopped
gating. Both decisions are defensible alone; this gate is where they collide.

### Three chunk-5 tests were silently re-pointed and had to be pinned

`MCTSConfig.root_mode` defaults to `"gumbel"`, and chunk 6 added dispatch on it — so the
chunk 5 pinned-visit and backup tests, written against PUCT, began measuring Sequential
Halving instead. They now set `root_mode="puct"` explicitly (and `dirichlet_frac=0` for
determinism). Worth noting as a pattern: adding a dispatch on a defaulted config field
silently changes what every existing test of the old path measures.

---

## Chunk 6 — evaluation profile + bounded tuning (2026-08-09, superseded same day)

> **Superseded by "Gate RESOLVED" below.** This section records the state at 0.940, when the
> residual was believed to be a capability limit. It was a bug. Kept unedited because the
> reasoning that reached the wrong conclusion is the useful part.

Tom's decision on the round-`d` split result: **the gate number does not move.** Two staged
interventions, pre-registered in `BLOCKED-2026-08-08-mate2-gate.md` *before* any measurement.
Both were run. **The gate is 0.940 and the ≥0.95 threshold is not met.** Nothing was promoted,
merged or tagged; the gate-number conversation goes back to Tom.

Authoritative write-up (with the argument, not just the numbers): `BLOCKED-2026-08-08` and the
round-`d` result block in `TRAINING_STRATEGY.md`, both on branch `round-d`.

### Stage 1 — the evaluation profile

`MCTSConfig.root_noise` (default `true` = self-play). At `false`: root candidates are the
deterministic top-m by raw logits, and the acting argmax is `logits + σ(q̂)` — Sequential
Halving, σ, the solver and the budget convention unchanged. Implemented as `g ≡ 0` rather than
a second selection path. Suites, ladder and match play run this profile; self-play runs noise.

```
$ python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_2.epd --sims 128 \
      --ckpt runs/phase1_d/final.pt --no-root-noise --seed 1
mate_in_2.epd: 188/200 solved at 128 sims (94.0%) | eval=runs/phase1_d/final.pt root=gumbel profile=eval(deterministic) m=16 c_visit=50 c_scale=1 | gate >= 95%

$ ... --seed 2
mate_in_2.epd: 188/200 solved at 128 sims (94.0%) | ... (identical, including the failure list)

$ ... --suite tests/suites/mate_in_1.epd --sims 16 --ckpt runs/phase1_d/final.pt --no-root-noise
mate_in_1.epd: 100/100 solved at 16 sims (100.0%) | gate >= 100%
```

0.890 → **0.940**, and the ±2.5-point seed spread is gone: both seeds agree exactly, on the
score *and* on which twelve positions failed. **Pre-registered projection was 0.93 ± 0.02**;
0.940 sits at the top edge of that window, so the three-factor decomposition was right and
stage 2 was authorised rather than triggered by surprise.

### Stage 2 — one bounded sweep, on a disjoint 500-position tuning suite

`runs/data/mate2_tune_500` (seed 102), verified disjoint from both gate suites, all four
held-out eval sets and all five anchor sets — including the superseded ones, since `phase1_d`
warm-started from `phase1_c` (`tests/test_contamination.py`, 29 pass). Full grid, 128 sims,
eval profile, ~98 s per config:

| c_visit | c_scale | m=16 | m=32 |
|---|---|---|---|
| 25 | 0.5 | 91.2 | 92.6 |
| 25 | 1.0 | 91.8 | 92.8 |
| 25 | 2.0 | 92.2 | **93.2** |
| 50 | 0.5 | 91.8 | 92.8 |
| 50 | 1.0 | 92.2ᵈ | **93.2** |
| 50 | 2.0 | 92.4 | 92.2 |
| 100 | 0.5 | 92.2 | **93.4** ← selected |
| 100 | 1.0 | 92.4 | 92.2 |
| 100 | 2.0 | 92.6 | 90.6 |

ᵈ = paper defaults. Selection by tuning-suite solve alone, ties toward defaults; 93.4% was a
unique maximum. **Single-shot gate with the chosen config, the one permitted touch:**

```
$ python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_2.epd --sims 128 \
      --ckpt runs/phase1_d/final.pt --no-root-noise --eval-m 32 --eval-c-visit 100 --eval-c-scale 0.5 --seed 1
mate_in_2.epd: 188/200 solved at 128 sims (94.0%) | ... m=32 c_visit=100 c_scale=0.5 | gate >= 95%
   (seed 2 identical)
```

**188/200 — the same score as the untuned profile, on a different set of twelve failures.**
The +1.2 points the sweep bought on the tuning suite transferred as exactly zero. That is the
firewall doing its job: the whole 18-config grid spans 90.6–93.4%, and the binomial standard
error at n=500, p≈0.93 is ±1.14 points, so the grid's spread is ~2.5 SE and picking its
maximum is mostly picking noise. Had the sweep been run on the gate suite, 93.4% would have
been reported as a tuning win.

### The decomposition now closes exactly, and names what is left

`scripts/diagnose_mate2.py` against the gate suite, eval profile:

| Factor | m=16 (defaults) | m=32 (selected) |
|---|---|---|
| prior coverage @ top-m | 0.990 | 0.995 |
| solved given considered | 0.949 | 0.945 |
| **product** | **0.940** | **0.940** |
| measured gate | 0.940 | 0.940 |

Under noise the chain was `0.990 × 0.925 × 0.941`; the middle factor is now 1.000 by
construction. Prior coverage is at 0.990 (top-1 0.660, top-8 0.975, top-32 0.995, median
forcing-move rank 0). **The entire residual is conversion — solved *given* the winning move
was simulated — and raising m trades against it almost exactly: +0.005 coverage, −0.004
conversion, product unchanged to three places.**

Twelve failures = 2 never considered + 10 considered but not converted. To reach 0.95 at
coverage 0.990, conversion must rise from 0.949 to ≥0.960.

**What is now known that was not on 2026-08-09 morning:** the gate is not prior-limited (round
`d` settled that), not noise-limited (stage 1 settled that), and not σ/m-limited (stage 2
settled that, at the cost of one burned tuning suite). It is limited by what the search does
with 128 simulations once the right move is already in front of it.

---

## Chunk 7 — Batched leaf evaluation ✅ code + gates; ⚠️ throughput target missed (2026-08-09)

Both DONE-WHENs green. The deferred GPU benchmark has now been run at the spec §10
configuration. **Recorded as measurements, not as pass/fail** — the escalation rule keys off
1,500 sims/s, which was not crossed.

```
$ pytest tests/test_batch_runner.py -q
30 passed in 1.87s

$ python benchmarks/bench_selfplay.py --games 32 --sims 16 --device cpu
sims/s 2847.6 | leaves/s 3021.6 | mean fill 31.36/512   # DONE-WHEN: runs to completion ✅

$ python benchmarks/bench_selfplay.py --games 256 --sims 48 --device cuda --ckpt runs/phase1_anchor.pt
games=256 plies=5120 in 128.21s
  sims/s 1916.9 | leaves/s 1955.2 | forwards/s 7.64 | mean fill 255.79/512 | terminal s-c 201

$ python benchmarks/bench_selfplay.py --games 256 --sims 48 --device cpu     # stub evaluator
games=256 plies=5080 in 91.13s
  sims/s 2675.7 | leaves/s 2726.5 | mean fill 253.54/512

$ python benchmarks/bench_inference.py --ckpt runs/phase1_anchor.pt --device cuda
batch   64: median   7.195 ms  p95   7.503 ms   8895.5 pos/s
batch  256: median  37.980 ms  p95  38.403 ms   6740.4 pos/s
batch  512: median  81.997 ms  p95  83.085 ms   6244.1 pos/s
```

| §10 target | Measured | |
|---|---|---|
| ≥ 3,000 sims/s aggregate | **1,917** | misses (above the 1,500 escalation floor) |
| < 8 ms forward at batch 512 | **82.0 ms** | misses by ~10× |

### The 3,000 sims/s target is unreachable on this box, and not because of the GPU

The stub run is the decisive one: **with a zero-cost evaluator the runner still only reaches
2,676 sims/s.** The §10 aggregate target sits above the ceiling of the runner with no network
in the loop at all, so no amount of GPU or batching work can reach it — python-chess move
generation and tree ops are the binding constraint. That is exactly the bottleneck ordering
§10 predicted ("movegen first, tree ops second, GPU last"); it is now measured rather than
assumed. The network costs the difference: 1/1917 − 1/2676 ≈ 0.15 ms per simulation, ~28% on
top of runner overhead.

**No py-spy profile was run** — chunk 7's escalation path triggers below 1,500 sims/s and 1,917
is above it. The stub decomposition already localises the cost to the same place a flame graph
would, and the plan is explicit that a movegen backend swap is a *new negotiated chunk*, not an
improvisation. Flagging for Tom rather than acting: **§10's 3,000 sims/s needs either a
renegotiated target or that new chunk.**

### The forward-latency target was calibrated for different hardware

82 ms at batch 512 against an 8 ms target is not a tuning gap. Per-position throughput *falls*
with batch size — 8,896 pos/s at 64, 6,244 at 512 — so the trunk is already compute-bound by
batch 64 and there is no batching headroom left to find. Same iGPU characteristic the Phase 1
record notes (batch 2048 is a training throughput *cliff*; the trunk runs at ~8% of theoretical
bf16). Measured with `inference_mode` + bf16 autocast, matching `torch_evaluate_batch` exactly;
the ~8 MB device→host copy the real path also pays is *excluded*, since §10 asks about the
forward.

Practical consequence for chunk 8/12, not a defect: `batch_leaves=512` never fills. Mean fill
is 255.8/512 with 256 concurrent games, because the buffer flushes when every game is blocked —
one leaf per game. Fill tracks `concurrent_games`, not `batch_leaves`, so the former is the
knob that matters for batch efficiency.

### The eval profile had to be wired into the batched path too

`BatchRunner` reimplements the Gumbel root rather than calling `Search.run_gumbel`, and it drew
Gumbel noise unconditionally — so `root_noise=False` was **silently ignored here**. Chunk 10's
ladder plays its matches through this path, so suite numbers would have been deterministic
while ladder numbers stayed noisy, with nothing in the code to say the two disagreed. Fixed by
resolving the profile in `__init__` (as `Search` does) and repeating the `g ≡ 0` substitution.

The equivalence gate is now parametrized over **both** profiles, which is the only place the
two independent root implementations are compared directly. 30 tests pass; 211 pass overall.

Third instance of one failure family, after `root_mode` re-pointing three chunk-5 tests and
`phase1_lr` silently not existing: **a defaulted config field with more than one consumer fails
silently on whichever consumer forgot to read it.**

---

## Gate RESOLVED — review fixes, then 0.940 → 0.985 ✅ (2026-08-09)

Review conditions S1–S3 accepted as merge conditions (S4 deferred), then the gate re-measured
and its failures triaged. **The chunk 6 mate-in-2 gate passes at 98.5% on its original 0.95
threshold and original frozen suite.** Phase 1 re-closed on **anchor v2 = `phase1_d/final.pt`**.

### Review fixes (S1, S2, S3, N3)

The reviewed defect was **verified before being fixed**. `scripts/parity_probe.py` sweeps
m × sims comparing `BatchRunner` (batch_leaves=1, one game) against `Search`:

```
BEFORE:  8 disagreement(s)   — all at sims=6, m ∈ {3,5}
AFTER:   0 disagreement(s)   — 4 fixtures × m ∈ {3,5,12,16} × sims ∈ {6,16,31,48}
```

The batched driver picked its forced edge with `survivors[sims_run % len(survivors)]` — a
round-robin — where the synchronous one uses `allocate_within_phase`, which gives a phase's
remainder to the **best-ranked** survivors. At `m=5, sims=6` the schedule is `[5, 1, 0]` and
phase 2's single visit belongs to `survivors[0]`; the round-robin spent it on `survivors[5 % 3]`,
the worst of the three. A second divergence surfaced while fixing it: the old loop halved
survivors on zero-budget phases, which the synchronous `continue` skips.

**Why it hid for a whole chunk:** the equivalence gate only ran m=16 × sims ∈ {16,48}. At
(128,16) *and* at those pairs, every phase budget divides evenly across its survivors —
remainder 0 throughout — so a round-robin and a correct allocation agree by accident. The gate
was real but its parametrization was the comfortable case.

S1 moved `form_considered_set`, `halve_survivors`, `force_include_proven_wins` and
`phase_visit_queue` into `gumbel.py`; both drivers consume them and neither re-derives any
part. S3 gave `_commit_move` the real `base_reps` — every trace row previously claimed to be a
first occurrence, which would have entered Phase 2 as silently corrupt training data rather
than as a crash. N3 zeroes `_grow_nodes`' tiled region.

### B1 — post-fix gate, and why the number did not move

```
mate_in_2.epd: 188/200 solved at 128 sims (94.0%)   (seeds 1 and 2 identical)
```

Unchanged from before the fix, and that is the expected result rather than a disappointment:
at (128, 16) `halving_phase_budgets` returns `[32,32,32,32]` against 16/8/4/2 survivors, so
every phase divides evenly and the allocation fix **provably cannot** change this
configuration — and the suite runner drives `Search`, not `BatchRunner`, in any case. What the
re-measurement actually established is that the S1 refactor is behaviour-preserving on the
synchronous path: identical score *and* identical failure list. 0.940 ∈ [0.93, 0.95) → the
pre-registered "proceed as planned" branch.

### B2 — triage: 9 BUG, 2 COVERAGE, 1 CAPABILITY

`scripts/triage_mate2.py`, reproduction check **0 mismatches vs `Search.run`**:

| id | verdict | key-edge visits | replies expanded | sims for proof | value head | reason |
|---|---|---|---|---|---|---|
| mate2_004 | BUG | 30 | — | — | +0.030 | key edge proven but not selected |
| mate2_006 | BUG | 30 | — | — | +0.039 | key edge proven but not selected |
| mate2_008 | BUG | 30 | — | — | +0.032 | key edge proven but not selected |
| mate2_054 | BUG | 30 | — | — | +0.046 | key edge proven but not selected |
| mate2_055 | BUG | 30 | — | — | +0.088 | key edge proven but not selected |
| mate2_056 | BUG | 30 | — | — | +0.082 | key edge proven but not selected |
| mate2_070 | BUG | 30 | — | — | +0.079 | key edge proven but not selected |
| mate2_129 | CAPABILITY | 2 | 1/16 | 17 | +0.036 | reply set never fully expanded |
| mate2_139 | BUG | 30 | — | — | −0.004 | key edge proven but not selected |
| mate2_165 | COVERAGE | 0 | — | — | +0.030 | no key move in the considered set |
| mate2_166 | COVERAGE | 0 | — | — | +0.032 | no key move in the considered set |
| mate2_196 | BUG | 30 | — | — | +0.092 | key edge proven but not selected |

**The nine BUGs were one defect, and it was not the solver.** The proof chain for `mate2_006`
was walked to its leaf and is sound — nine plies, every defender node holding exactly one legal
reply, tree edge counts matching python-chess at every step, bottoming out on a real
checkmate. The search's proofs were correct. What was missing was an **order over them**:

```
ply  1: b3e3  prov pv=-1.0 | truth: legal_replies=1
ply  3: g1g5  prov pv=-1.0 | truth: legal_replies=1
ply  5: e3e5  prov pv=-1.0 | truth: legal_replies=1
ply  7: g5g7  prov pv=-1.0 | truth: legal_replies=1
ply  9: e5e7  TERM tv=-1.0 | truth: checkmate=True   <- proof bottoms out correctly
```

Every proven win sits at q̂ = +1, so σ(q̂) ties them and the acting argmax fell through to
`g + logits` — noise under self-play, the raw **policy prior** under the evaluation profile.
The search played a proven mate-in-5 over a proven mate-in-2 because the network's prior
preferred it. `IMMEDIATE_WIN_BONUS` already encoded the right principle but only for depth 1.

Fixed by inducting mate distance through the solver (`Tree.proven_depth`; winner takes the
fastest proved line, defender the slowest) and scoring `EXACT_WIN_BONUS − depth × WIN_DEPTH_STEP`.

### B1 re-run (the one permitted) — PASS

```
mate_in_2.epd: 197/200 solved at 128 sims (98.5%) | profile=eval(deterministic) | gate >= 95%
  failed: mate2_129, mate2_165, mate2_166        (identical under seeds 1 and 2)
mate_in_1.epd: 100/100 at 16 sims and at 48 sims (stub) — no regression
```

The three survivors are exactly the three non-BUG triage rows — the consistency check that the
classification was real, not a story fitted to the outcome. **340 passed, ruff clean, parity
probe 0 disagreements.**

**What made this findable:** the bug was invisible while the prior was the binding constraint —
at round `c`'s 0.83 coverage the search rarely proved two wins in the same position, so there
was nothing to mis-order. Each earlier intervention that removed a *different* limitation moved
this one closer to the surface, and the evaluation profile made the tie-break fall through to
the prior *deterministically* rather than sometimes. The three "failed" interventions were what
made the real defect legible. Generalisable: **an aggregate residual is not a diagnosis** —
three rounds treated "conversion" as one quantity; one script that classified each failure
separately produced a nine-position bug with a one-line remedy.

---

## Chunks 5–7 external review — conditions and outcomes (2026-08-09)

**The review document itself (`REVIEW-2026-08-09-chunks-5-7.md`) was never delivered to this
machine** — it is absent from the repo, both worktrees, `~/Downloads` and any `outputs`
directory, and two successive briefs have asked for it to be committed. This section is
therefore *this repo's* record of the conditions **as relayed in the briefs**, plus what was
actually done about each. It is not a substitute for the review and does not reproduce its
reasoning; commit the real document alongside this when it turns up.

| # | Condition | Status | Where |
|---|---|---|---|
| S1 | One Gumbel root implementation; both drivers consume it | **Applied** | `gumbel.form_considered_set` / `halve_survivors` / `force_include_proven_wins` / `phase_visit_queue` |
| S2 | Batched driver must allocate phase visits, not round-robin | **Applied** — defect reproduced first (8 → 0 probe disagreements) | `BatchRunner._next_forced_edge`, `scripts/parity_probe.py` |
| S3 | Trace tokens must encode the real repetition count | **Applied** (chunk 7) + ownership completed in chunk 8 | `_commit_move`, per-game history counter |
| S4 | Promotion priors via the promo sub-head | **Deferred at review time; delivered in chunk 8** | evaluator interface v2 |
| N3 | Zero the tiled region in `_grow_nodes` | **Applied** | `Tree._grow_nodes` |
| N1/N2 | Perf notes | **Open**, awaiting profile evidence | — |

**What the review did not catch**, recorded because it bounds how much assurance a review of
this kind buys: the nine-position **proven-win ordering** defect (search proved a mate-in-2 and
played a proven mate-in-5). It was not visible in the code under review — every line was
individually correct, `IMMEDIATE_WIN_BONUS` looked like it covered the case, and the failure
only appears when two proven wins exist in one position, which needed round `d`'s coverage
improvement to become common. It was found by *triaging gate failures*, not by reading code.
Reviews catch defects of construction; measurement catches defects of interaction.

---

## Chunk 8 intake — carried forward, not actioned (2026-08-09)

Register for the next brief. Nothing here is a defect in merged code.

- **S4 — promotion priors via the promo sub-head.** Deferred because it is an *interface*
  change, not a tweak: `EvaluateFn` returns no promo distribution, so wiring it touches both
  evaluators, the batch adapter and `_build_node`. Bundle with chunk 8's evaluator work.
- **Repetition / adjudication ownership** (from S3). Per-game history `repetitions_fn` built
  from the move stack; claim-draw (threefold, 50-move) adjudication at game level. The
  `REP_1` trace test armed in this session (`test_trace_tokens_carry_the_repetition_count`)
  passes today via an injected `repetitions_fn` and goes green natively here.
- **M3 watchlist** — `benchmarks/m3_watchlist.json`: `mate2_129`, `_165`, `_166`. The value
  head reads ≈ +0.03 on positions that are in fact won: not merely imprecise but nearly
  indifferent, which is what self-play value training is for. `mate2_129` needs 17 sims on the
  key edge to reach a proof and got 2.
- **`improved_policy` does not know about mate distance.** The acting argmax now prefers the
  shorter mate; the *policy target* still spreads equally across proven wins, because it is
  built from completed-Q (±1) alone. Harmless in Phase 1, but in Phase 2 it teaches the network
  that a mate-in-5 is as good as a mate-in-2. Decide deliberately in chunk 8/9 — it is a
  training-target question, not a search one.
- **Resignation calibration** per spec §7; **`CheckpointPool` single-implementation rule**;
  **N1/N2 perf notes** awaiting profile evidence.
- **§10 erratum 6** — targets replaced by "Phase 2 iteration ≤ ~35 min at default config",
  which the current build meets with ~no margin. Escalate on a *regression* in sims/sec, not
  on the absolute number.

---

## Session 2026-08-09 — tree state, and the push finally happened

Nothing merged, nothing tagged. That is the pre-registered consequence of the gate landing at
0.940: promotion, the `round-d` → `main` merge and the `phase1-closed-r2` tag were all
conditional on ≥ 0.95.

| Branch | Tip | vs `main` | State |
|---|---|---|---|
| `main` | `29b2294` | — | untouched all session |
| `round-d` | `17625b0` | +7 | Phase 1 docs + contamination gate. **Unmerged, untagged** |
| `chunk-05-mcts` | `73abbf9` | +3 | unmerged, pending review |
| `chunk-06-gumbel` | `67e6025` | +7 | eval profile, sweep, diagnosis. Unmerged |
| `chunk-07-batched` | `e7a5c8d` | +10 | contains `chunk-06-gumbel`. Unmerged |

`runs/phase1_anchor.pt` is byte-identical to `runs/phase1_c/final.pt` (md5
`d58045a6865bd5c973336db21296b8ee`) — **not promoted**, and `ANCHORS.sha256` is unchanged with
its single dated v1 line. `runs/phase1_d/final.pt` remains the strictly better checkpoint on
every held-out gate, available to promote whenever the gate conversation resolves.

Only pre-existing tag is `phase1-closed`. Both worktrees clean (`/chess` on `round-d`,
`/chess-c7` on `chunk-07-batched`).

**Where the work landed, and why not all on one branch.** `round-d` has no
`src/tinyalphazero/mcts/` at all — the search package lives only on the `chunk-05/06/07` chain,
which is a diverged sibling off `main` — so the evaluation profile could not physically be
built there. It is a chunk 6 feature and went to `chunk-06-gumbel`, then forward into
`chunk-07-batched`; the Phase 1 documents and the contamination gate stayed on `round-d`. The
`position_key` helper and `test_contamination.py` exist only on `round-d`, which is why the
tuning suite's disjointness was verified there and the sweep was run on `chunk-06-gumbel`. The
tuning suite itself lives under `runs/` (gitignored), so it survives checkouts; only its
`meta.json` is tracked, on both branches.

**Pushed — the four-session backlog is cleared.** Tom generated an ed25519 keypair and pushed
during this session. `origin` (`git@github.com:tripptytrip/Tiny-AlphaZero-v2.git`) now carries
all five branches and the `phase1-closed` tag, each remote ref identical to its local tip;
`git log --branches --tags --not --remotes` is empty. Verified after the fact rather than
assumed, because "I pushed" and "everything is on the remote" are different claims — a bare
`git push` from a branch with no upstream is the usual way they come apart.

Still local-only and unprotected by this: the ~6.8 G of untracked checkpoints and datasets,
including `runs/phase1_d/final.pt` (the strictly-better-but-unpromoted checkpoint) and
`runs/phase1_anchor.pt` itself. `runs/ANCHORS.sha256` is committed, so the anchor's *identity*
survives a disk loss — its *bytes* do not. Regenerating `phase1_d` costs ~3 h of GPU time and
`phase1_c` considerably more.

Note for the next session: only `main` has an upstream set. The other four branches were pushed
by explicit refspec, so a bare `git push` from them will not do what it looks like it does.

---

## Chunk 8 — Self-play actor + replay ring ✅ (2026-08-09)

```
$ pytest -q                                      442 passed
$ ruff check src/ tests/ scripts/ benchmarks/    All checks passed
$ python -m tinyalphazero.selfplay --ckpt runs/phase1_anchor.pt --games 8 --sims 16 \
      --out runs/sp_smoke --device cpu
8 games, 1280 positions in 42.7s (674.3 games/h) | eval=runs/phase1_anchor.pt
  results: {'ply_cap': 8}
  resign-disabled 2, would-have-resigned 0, false-resign rate None
$ python -c "...ReplayBuffer.open('runs/sp_smoke/replay')..."     1280
```

### Throughput: erratum #6 met on the real self-play path

```
64 games, 7901 positions in 195.2s (1180.3 games/h) | 48 sims | device=cuda
  results: {'checkmate': 40, 'ply_cap': 24}
```

**1,943 sims/s** end-to-end — within 1.4% of the bench harness's 1,917, which is
the cross-check that the harness was measuring the real thing. At 512 games per
iteration that projects to **26.0 min**, inside erratum #6's ≤ ~35 min budget
with about 26% headroom. First number that tests the erratum against the actual
actor rather than the benchmark.

**One runner drives all games.** The first version used a runner per game for a
simpler resignation hook, which made every forward batch-1 — the exact failure
chunk 7 exists to prevent. Measured on the smoke: **80.1s → 42.7s**. Worth
recording because the mistake was invisible in the tests; only the clock caught it.

### Resignation never fired, and that is the value head again

Across 64 games and 9 resign-disabled samples, `would_have_resigned` is **0**.
The rule needs best root q < −0.90 for eight of a side's own moves, and the
Phase 1 value head does not produce numbers that negative — the same indifference
the M3 watchlist records (≈ +0.03 on positions that are *won*). `false_resign_rate`
reports `None` rather than 0.0: nothing has been measured yet, and a 0.0 there
would read as "calibrated". Expect this to start firing once Phase 2 sharpens
value, and treat the first non-null false-resign rate as the calibration signal.

### What Part A found, in one line

The promo-prior change (review S4) is **behind a flag, default off**, because
turning it on costs the chunk 6 gate 0.985 → 0.960. The Phase 1 promo head ranks
**queen last** — it trained on random play, where the four promotion pieces are
sampled uniformly, so it has no promotion signal at all. See the Part A commit
and `MCTSConfig.promo_prior_from_head`.

---

## Next: **chunk 9** — the Phase 2 loop and the golden mini-run, then the 24h M3 campaign with the watchlist riding along.
