# TinyAlphaZero v2 — Training Strategy

This is the **campaign document**: which runs exist, why each was launched, the exact call that launched it, what it produced, and what gets launched next. `HOW_TO_TRAIN.md` explains *how the machinery works*; this file records *what we are doing with it and why*. One round = one entry here, written **before** the run starts (pre-registered expectations) and completed after it ends (results + decision).

**Current status:** **Phase 1 CLOSED (r2)** on **anchor v2 = `runs/phase1_d/final.pt`**, 2026-08-09, tag `phase1-closed-r2`. All four held-out gates pass, and the chunk 6 mate-in-2 gate that reopened the phase now reads **98.5%** against its original, never-moved 0.95 threshold. Next: **chunk 8** (self-play actor + replay); chunks 5–7 are merged and reviewed.

**How the reopening resolved.** 0.770 (round `c` anchor) → 0.890 (round `d`: prior coverage 0.83 → 0.990) → 0.940 (evaluation profile: stop measuring exploration noise on a match-play question) → 0.940 (bounded σ/m tuning: null result, firewall held) → **0.985** (mate-distance ordering over proven wins — a search bug that only became visible once the prior stopped being the binding constraint). Two of the three causes behind round `d`'s shortfall were search defects, not model ones. Full chain: `BLOCKED-2026-08-08-mate2-gate.md`.

---

## Campaign rules

1. **Every run gets a fresh letter and a fresh `--out` dir** (`runs/phase1_c`, `runs/phase1_d`, …). Never resume into or overwrite a previous round's directory; `--ckpt` resumption is for crash recovery within a round only.
2. **One lever per round where possible.** When a round must change several things (as `c` does), list every delta explicitly so the attribution ambiguity is at least recorded.
3. **Pre-register expectations.** Each round's entry states predicted gate outcomes before launch. A run can't be judged honestly against expectations written after reading its results.
4. **Eval sets are frozen for the whole of Phase 1** (one sanctioned upgrade in round `c` pre-flight, below, with re-baselining). Rounds are only comparable if measured on the same held-out data. Evaluation never touches training dirs — note `HOW_TO_TRAIN.md` Step 3's example still points `eval_phase1.py` at `runs/data/phase1`/`mate1`; use the `eval_*` dirs.
5. **Gates don't move.** A failed gate triggers either another round or a `BLOCKED.md` conversation — never a threshold edit in passing. Exercised once: `BLOCKED-2026-08-06-phase1-gates.md` retired two gates after three rounds and a 95K-position diagnosis, replaced them with a metric the pipeline actually consumes, kept the retired numbers reported, and named three downstream return conditions. Retiring a proxy that measures the wrong thing is in bounds; relaxing a threshold to fit a result is not.
6. **Record the environment.** Each entry notes git SHA and any stack change (torch/ROCm version, SDPA backend). A run after a stack change is not directly comparable to one before it.

---

## Campaign log

| Round | Date | Steps | top1_legal | mate1 top-1 | auc | wdl | Verdict |
|---|---|---|---|---|---|---|---|
| `phase1_a` | Jul 3–4 | 39,135 | 0.000 | 0.065 | 0.981 | 0.034 | invalid run — see post-mortem |
| `phase1_b` | Aug 4 | 28,008 | 0.953 | 0.527 | **0.9998 ✓** | 0.683 | learning; schedule + anchor signal insufficient |
| `phase1_c` | Aug 6 | 56,484 | 0.976ᵢ | 0.802ᵢ | **1.000 ✓** | **0.892 ✓** | Phase 1 CLOSED — BLOCKED-2026-08-06 (ᵢ = retired→informational) |
| `phase1_d` | Aug 9 | 18,926 | 0.983ⁱ | 0.835ⁱ | **1.000 ✓** | **0.933 ✓** | held-out gates ALL PASS (mate2_top16 **0.977 ✓**); chunk-6 re-gate **0.890 ✗** — see result block |

**Measurement basis.** All rows are `scripts/eval_phase1.py` against the frozen held-out sets (`eval_rand` n_random=2000; `eval_mate1_1k` full 1,000 anchors) — *not* the in-training eval, which samples 1000/200 and reads a point or two higher on the noisy metrics. Round `b`'s row is the re-baselined measurement from round `c`'s pre-flight step 3 (it read 0.967 / 0.480 / 0.685 on the retired 500-anchor set and the smaller in-training sample); rounds are only comparable on the same basis, per rule 4.

**Provenance.** Runs a–c predate the repository — provenance for those rounds is the campaign rows + dataset `meta.json` `argv`, not checkpoint SHA (all three read `git_sha='unknown'`; see PROGRESS.md "Readiness 2026-08-06"). SHA embedding is live from `phase1-closed` onward.

**Gates as of 2026-08-06** (`BLOCKED-2026-08-06-phase1-gates.md`): gated = `legality_auc` ≥0.995, `wdl_anchor_mass` ≥0.80, **`mate1_top8_masked` ≥0.97** (new). `top1_legal_rate` (was ≥0.99) and `mate1_solve_rate` (was ≥0.90, = masked top-1) are retired to informational, marked ᵢ. The new gate is not a rubber stamp: `b` scores 0.820 on it, `c` 0.980.

### Round `phase1_a` — invalid (all gates missed)

Launched with three stacked defects: `runs/data/mate1` was random games (`--games` typo, no meta validation), masked policy CE (structural no-op on uniform-random targets; unmasked legality untrainable), aux legality head detached from the trunk. Root-cause analysis: `PHASE1_FINDINGS_AND_FIXES.md`. Value of the round: the post-mortem, the fail-fast dataset validation, component logging, and the NaN-skip guard all came out of it. Its numbers carry no information about the architecture.

### Round `phase1_b` — first valid run

```bash
VIRTUAL_ENV=.venv .venv/bin/python scripts/train_phase1.py \
    --data runs/data/phase1 --mate1 runs/data/mate1 \
    --eval-data runs/data/eval_rand --eval-mate1 runs/data/eval_mate1 \
    --out runs/phase1_b --epochs 3 --seed 42
```

Deltas vs `a`: genuine 20K anchors; unmasked CE; attached aux head; held-out eval; `workers=8`; component logging; NaN-skip guard.

Result: healthy learning curve (policy 8.32 → 3.48, near the E[ln n_legal] floor). AUC gate passed epoch 1. `top1_legal` gap halved each epoch (0.868 → 0.940 → 0.967) and was still closing when the cosine floor (3e-5) ended learning — a schedule limit, not a model limit. `mate1_solve` and `wdl_anchor` rose monotonically but landed far short; both are driven solely by the 5% anchor mixture fighting 95% "be uniform" gradient. Promo CE below its ln 4 floor (1.13) = mild anchor-adjacent memorization ⇒ spare capacity ⇒ the shortfall is signal, not capacity. 5 non-finite-gradient skips in 28K steps (0.02%) — benign, watch the rate.

**Decision:** no model changes. Round `c` changes data volume/mix and schedule length only.

**Re-baseline (pre-flight step 3, run 2026-08-06):** on the upgraded 1K anchor set, `b` reads 0.953 / 0.527 / 0.9998 / 0.683 (table row updated). The `top1_legal` shift 0.967 → 0.953 is the sample change, not a regression: in-training eval draws 1000 random positions, `eval_phase1.py` draws 2000. One premise from this round did not survive `c` — see below.

### Round `phase1_c` — more anchor signal, longer schedule *(ran Aug 5–6; result at the end of this section)*

**Deltas vs `b` (three, recorded per rule 2):** anchors 20K → 100K (new dataset, more *diverse* mates — solve rate is measured held-out, so diversity is what generalizes); mixture 0.05 → 0.10; epochs 3 → 6 with a fresh cosine over the full ~56.5K steps (not a resume — resuming `b` would continue at the LR floor).

**Pre-flight (run in order):**

```bash
# 1. Generate the 100K anchor set (~20 min CPU; verify mode line prints mate_in_1)
VIRTUAL_ENV=.venv .venv/bin/python scripts/generate_phase1.py \
    --mate1 100000 --out runs/data/mate1_100k --seed 3

# 2. One-time eval upgrade: 500 → 1000 held-out mates (±4.4% → ±3.1% at the gate).
#    Same seed lineage; frozen thereafter for all of Phase 1.
VIRTUAL_ENV=.venv .venv/bin/python scripts/generate_phase1.py \
    --mate1 1000 --out runs/data/eval_mate1_1k --seed 100

# 3. Re-baseline round b on the upgraded eval set so rounds stay comparable
VIRTUAL_ENV=.venv .venv/bin/python scripts/eval_phase1.py \
    --ckpt runs/phase1_b/final.pt \
    --data runs/data/eval_rand --mate1 runs/data/eval_mate1_1k
#    → record the re-baselined mate1/wdl numbers in this file next to round b

# 4. Mode audit (refuse to launch on any mismatch)
VIRTUAL_ENV=.venv .venv/bin/python - <<'EOF'
import json; from pathlib import Path
for d in ["runs/data/phase1","runs/data/mate1_100k","runs/data/eval_rand","runs/data/eval_mate1_1k"]:
    g = json.loads(Path(d,"meta.json").read_text())["generation_config"]
    print(f"{d:28} mode={g['mode']:<12} argv={' '.join(g.get('argv',[])[1:])}")
EOF

# 5. Timing slice before committing ~5-8h (abort if ms/step regressed vs b)
VIRTUAL_ENV=.venv .venv/bin/python scripts/train_phase1.py \
    --data runs/data/phase1 --mate1 runs/data/mate1_100k \
    --eval-data runs/data/eval_rand --eval-mate1 runs/data/eval_mate1_1k \
    --config configs/phase1_c.yaml \
    --out runs/_timing_c --max-steps 600
```

**Config** — create `configs/phase1_c.yaml` (defaults inherited; only the delta):

```yaml
train:
  phase1_mate1_mix: 0.10
  phase1_epochs: 6
```

**The run:**

```bash
VIRTUAL_ENV=.venv .venv/bin/python scripts/train_phase1.py \
    --data runs/data/phase1 --mate1 runs/data/mate1_100k \
    --eval-data runs/data/eval_rand --eval-mate1 runs/data/eval_mate1_1k \
    --config configs/phase1_c.yaml \
    --out runs/phase1_c --epochs 6 --seed 42
```

**Post-run:**

```bash
VIRTUAL_ENV=.venv .venv/bin/python scripts/eval_phase1.py \
    --ckpt runs/phase1_c/final.pt \
    --data runs/data/eval_rand --mate1 runs/data/eval_mate1_1k
```

**Pre-registered expectations:** `top1_legal` ≥ 0.99 (it was 0.967 with the gap halving per epoch and LR out of runway); `wdl_anchor` ≥ 0.80 (2× mix + 5× anchor diversity directly feed it); `auc` holds ≥ 0.999; `mate1_solve` lands 0.70–0.90 — the honest uncertainty of the round. Skipped-step count stays < 20.

**Decision rules (write the outcome into the log, then act):**
- **All four gates pass** → tag `runs/phase1_c/final.pt` as the Phase 1 anchor checkpoint; Phase 1 closed; proceed to Chunk 5.
- **Only `mate1_solve` fails, still rising at epoch 6** → round `d`: epochs 6 → 10, nothing else. Schedule was the binding constraint.
- **Only `mate1_solve` fails, plateaued (< 0.01/epoch over the last two)** → BLOCKED.md conversation: with diverse anchors, adequate capacity, and a converged schedule, the question becomes whether 0.90 *without search* was the right gate — mate-in-1 under 16-sim MCTS (the Chunk 6 gate) is the number that actually matters downstream. Resolve deliberately, not by edit.
- **`top1_legal` fails** → round `d`: raise the LR floor (`phase1_lr_end: 5e-5`) and extend epochs before touching anything else; a 0.98x plateau *across* rounds despite that escalates to the encoding fuzz per `HOW_TO_TRAIN.md` troubleshooting.
- **Skip rate grows past ~0.1%** → pause the campaign, pin math-only SDPA (`PROGRESS.md` ROCm note), re-run the timing slice, and note the stack change per rule 6 before the next round.

**Result (2026-08-06):** 56,484 steps, 6 epochs, ~9.3 h wall-clock, **0 skipped steps** (down from 5 in `b` — the guard never fired). Held-out: `top1_legal` 0.976, `mate1_solve` (masked top-1) 0.802, `legality_auc` 1.0000, `wdl_anchor_mass` 0.892, `mate1_top8_masked` 0.980, `top16` 0.988.

Against pre-registration: `wdl_anchor` hit (0.892 ≥ 0.80 — 2× mix + 5× anchor diversity did what was predicted); `auc` hit (1.0000); `mate1_solve` landed 0.802, just under the pre-registered 0.70–0.90 band's midpoint and short of the gate; `top1_legal` **missed** at 0.976, and the reasoning behind that prediction was wrong. The prediction assumed the `b` plateau was schedule-bound; `c` flatlined at 0.981 (in-training basis) from epoch 4 with LR still ≈1e-4, falsifying the premise before the planned round `d` (epochs-only) could spend compute on it. Both misses were then diagnosed rather than re-run.

**Diagnosis** (`scripts/diagnose_phase1.py`, full 95,465-position held-out set): 79.5% of the residual illegal argmaxes are pseudo-legal moves that leave the king in check; positions in check are ~6× overrepresented. Epoch 4 → 6, geometry failures fell 40% (669 → 403) while the check class was static (1576 → 1566), and failure-set Jaccard(e4, final) = 0.36 — a persistent class, not stuck positions. The aux legality head scores AUC 1.0000 on the same positions: the trunk encodes full legality; the bilinear from→to policy head can't route it. The plateau *is* the check-legality tail.

**Decision taken:** none of the pre-registered branches — the plateau was neither "still rising" nor a capacity limit, so it went to the `BLOCKED.md` conversation rule 5 reserves. Outcome: **Phase 1 CLOSED**, `runs/phase1_c/final.pt` copied to **`runs/phase1_anchor.pt`** (frozen ladder anchor, spec §8). The two unmasked-top-1 gates were retired to informational and replaced by `mate1_top8_masked` ≥ 0.97 — downstream consumption is legality-masked only (spec §6), and chunk 5's own gate requires 100% mate-in-1 at 16 sims *with a uniform stub evaluator*, so top-1 was never load-bearing for anchor correctness. Full argument, revised acceptance basis, and the three return conditions that reopen this: `BLOCKED-2026-08-06-phase1-gates.md`. `scripts/eval_phase1.py` updated to gate on the revised set and report the retired metrics.

**Carried forward (write-up note):** a 6.5M-param model on pure random play drives *geometric* legality toward zero and was still falling 40% per two epochs at the end, while *check-legality* — the one global constraint in the rules — sits as a static ~1.6% tail the factorized head cannot absorb, even though a linear probe on the same trunk reads legality at AUC 1.0. The information is present; the head's inductive bias can't route it. This is the §4.2 attention-bias escalation's trigger condition, should Phase 2 ever need it.

---

### Round `phase1_d` — planned: mate-in-2 anchor coverage (REOPENING)

**Why this round exists.** The 2026-08-06 closure was reopened via its own return condition 1:
the chunk 6 gate `mate-in-2 ≥ 95% @128 sims` scored **77.0%** with `runs/phase1_anchor.pt`.
Diagnosis (`BLOCKED-2026-08-08-mate2-gate.md`): search is at its ceiling — **95.4% solved
*given* the forcing move was in Gumbel's considered set** — but the prior only puts it there
75.5% of the time (network `top16` rank coverage 83%). Gumbel's top-m by `g + logits` is the
Phase 2 root mechanism, so this is a property the pipeline consumes directly, not a proxy.
The gate does not move; the model gets fixed.

**Intent (one line):** raise the prior's rank coverage of mate-in-2 key moves so Gumbel's
considered set contains them — 83% top-16 → target ≥ 95%.

**Deltas vs `c` (recorded per rule 2):** new `mate_in_2` anchor set (50K) at mix 0.07;
mate-in-1 mix 0.10 → 0.08; **warm restart** from `runs/phase1_anchor.pt` via `--init-weights`
(model weights only — fresh optimizer, fresh schedule) rather than training from scratch;
constant LR 1e-4 (no cosine) for 2 epochs. Attribution is deliberately impure: this is a
targeted repair of one measured deficiency, not a clean single-lever comparison, and the
gate that judges it (`mate2_top16_coverage`) did not exist for rounds a–c.

**Pre-flight:** mode audit over all six data dirs; `test_no_suite_contamination` green
(zero overlap between anchor sets and {suite, both held-out eval sets}); 300-step timing
slice with `--init-weights` (expect ≈564 ms/step → ≈3 h for ≈19K steps); delete `runs/_timing_d`.

**Config** — `configs/phase1_d.yaml`:

```yaml
train:
  phase1_epochs: 2
  phase1_lr: 1.0e-4
  phase1_lr_end: 1.0e-4
  phase1_mate1_mix: 0.08
  phase1_mate2_mix: 0.07
```

**The run:**

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 VIRTUAL_ENV=.venv .venv/bin/python scripts/train_phase1.py \
    --data runs/data/phase1 --mate1 runs/data/mate1_100k --mate2 runs/data/mate2_50k \
    --eval-data runs/data/eval_rand --eval-mate1 runs/data/eval_mate1_1k --eval-mate2 runs/data/eval_mate2_1k \
    --config configs/phase1_d.yaml --init-weights runs/phase1_anchor.pt \
    --out runs/phase1_d --epochs 2 --seed 42
```

**Pre-registered expectations (written before launch, rule 3).**

*Gated:* `mate2_top16_coverage ≥ 0.95` (new, held-out); `mate1_top8_masked ≥ 0.97`
(no-regress); `legality_auc ≥ 0.995`; `wdl_anchor_mass ≥ 0.80` — note this **may rise**
simply because the anchor mixture now carries more outcome-+1 positions; that is expected
arithmetic, not evidence of improvement, and must not be read as one.
*Informational:* `mate2_solve_rate` (top-1), `top1_legal_rate`, `mate1_solve_rate`.
*Operative re-gate:* chunk 6 mate-in-2 ≥ 0.95 at 128 sims with the new checkpoint — this is
the number that actually decides the round; `mate2_top16_coverage` is its held-out proxy and
the two are expected to agree. If they disagree, the disagreement is the finding.
*Stability:* non-finite-gradient skips < 10.

**Decision rules (write the outcome into the log, then act):**
- **All pass** → re-close: `runs/phase1_anchor.pt` ← `phase1_d/final.pt`; append the v2 digest
  to `runs/ANCHORS.sha256` keeping the dated v1 line; merge `round-d` → `main`; annotated tag
  `phase1-closed-r2`; amend both BLOCKED docs to closed; row + result block here.
- **`mate2_top16` or the chunk 6 re-gate lands 0.90–0.95** → exactly one escalation, single
  lever: `phase1_epochs: 3` *or* mixes 0.10/0.10 — **not both**. Re-gate.
- **< 0.90 after that escalation** → stop. Update `BLOCKED-2026-08-08` with the evidence that
  supervised anchoring cannot buy this coverage at this scale, and hand the gate conversation
  back. Do not iterate further.

---

**Result (2026-08-09):** 18,926 steps, 2 epochs, 3.05 h wall-clock, **1 skipped step**, constant LR 1e-4 throughout, warm-started from `runs/phase1_anchor.pt` (donor step 56,484).

**Held-out gates — all four PASS:**

| Metric | Round `c` | Round `d` | Gate |
|---|---|---|---|
| `mate2_top16_coverage` | 0.8100 | **0.9770** | ≥0.95 ✓ |
| `mate1_top8_masked` | 0.9800 | **0.9940** | ≥0.97 ✓ |
| `legality_auc` | 1.0000 | 1.0000 | ≥0.995 ✓ |
| `wdl_anchor_mass` | 0.8916 | 0.9333 | ≥0.80 ✓ |

Informational: `mate2_solve_rate` 0.671, `mate1_solve_rate` 0.835, `top1_legal` 0.983.

**Trajectory:** epoch 1 → 2 on `mate2_top16_coverage` was 0.9760 → 0.9770, **+0.001 — saturated by epoch 1**. The 0.07 mix was generous, not tight; the second epoch bought essentially nothing on the target metric. The no-regress gates did not erode under two full epochs of un-annealed 1e-4 — `mate1_top8` rose 0.980 → 0.994 and `wdl` 0.892 → 0.933, so the mate-in-2 anchors reinforced mate-in-1 rather than competing with it.

**The operative re-gate FAILS, and the pre-registered disagreement is the finding.**

```
mate_in_2.epd: 180/200 at 128 sims (90.0%) | gate >= 95%
over 5 Gumbel seeds: mean 0.890, range 0.875-0.900, spread 0.025
mate_in_1.epd: 100/100 at 16 sims (sanity ✓)
```

The round `d` entry above pre-registered: *"`mate2_top16_coverage` is its held-out proxy and the two are expected to agree. If they disagree, the disagreement is the finding."* They disagree by 8.7 points. Decomposed on the suite with `phase1_d`:

| Stage | Round `c` | Round `d` |
|---|---|---|
| prior top-16 coverage | 0.83 | **0.990** |
| survives Gumbel noise into the considered set | — | 0.925 |
| solved given considered | 0.954 | 0.941 |
| **gate** (product) | 0.77 | **0.870–0.900** |

**Round `d` did exactly what it was designed to do and the gate still fails.** Prior coverage went 0.83 → 0.990 — there is no headroom left on that axis. The residual is now split between two *search-side* losses that no amount of anchor training touches:

1. **Gumbel noise costs 6.5 points of consideration** (0.990 → 0.925). The considered set is top-m by `g + logits`, so a move the prior ranks *first* can still be knocked out by an unlucky draw. At round `c`'s 0.83 coverage this was invisible under the prior's own deficit; at 0.990 it is the dominant term.
2. **Selection given consideration is 0.941**, essentially unchanged from round `c`'s 0.954 — the search's ability to convert a considered forcing move into a chosen one was never the thing round `d` addressed.

**Decision taken:** the permitted escalation (`phase1_epochs: 3` *or* mixes 0.10/0.10) targets prior coverage, which is at 0.990 and saturated after one epoch. Spending it would buy at most ~0.5 points against a 6-point gap. Recorded as evidence in `BLOCKED-2026-08-08-mate2-gate.md` rather than spent. **Phase 1 is NOT re-closed; `runs/phase1_anchor.pt` is unchanged.**

**Follow-up, 2026-08-09 (no new training run — this is not a round).** Tom's resolution path was pre-registered in `BLOCKED-2026-08-08` before any measurement and then executed in full: an evaluation profile (deterministic root acting, `root_noise=false`) took the gate 0.890 → **0.940** with the seed spread collapsing from ±2.5 points to exactly zero — inside the pre-registered 0.93 ± 0.02 projection; then one bounded sweep of the eval-scoped σ constants and `m` on a disjoint 500-position tuning suite selected `m=32, c_visit=100, c_scale=0.5` (93.4% there, vs 92.2% for the paper defaults) and the single-shot gate came back at **0.940 — the same score as untuned, on a different set of twelve failures.**

Two things this adds to the campaign record, both of which outlive this gate:

1. **The firewall paid for itself the first time it was used.** Binomial SE at n=500, p≈0.93 is ±1.14 points and the 18-config grid spans ~2.5 SE, so its maximum is mostly noise — and the out-of-sample gate said so. Run on the gate suite, 93.4% would have been written up as a tuning win. Rule 4's "eval sets are frozen" now has a sibling worth stating: *a set you select a configuration against is spent, whether or not you trained on it.*
2. **The gate is not a data problem, so there is no round `e` to plan.** Decomposed on the suite, `0.990 coverage × 0.949 conversion = 0.940` exactly. Prior coverage is saturated, the noise term is 1.000 by construction, and raising `m` trades coverage against conversion to a wash. The residual is ten positions where the winning move is simulated and still not chosen — a chunk 6/7 search property. Phase 1 anchor training has no lever left that points at it.

**Closed out, 2026-08-09 (same day, after the review fixes).** Point 2 was right that the residual was search-side and right that no round `e` was needed — but it stopped one step short of a diagnosis. Triaging the twelve failures *individually* rather than reasoning about the aggregate split them into **9 BUG / 2 COVERAGE / 1 CAPABILITY**, and the nine were one defect: the search **proved** a mate-in-2 and played a **proven mate-in-5**, because all proven wins tie at q̂ = +1 and the acting argmax then fell through to the policy prior. Inducting mate distance through the solver and ordering wins by it took the gate to **0.985**. The three survivors are exactly the three non-BUG rows.

**Round `d` final verdict: the round did its job.** Its own gate (`mate2_top16_coverage` 0.977) passed, `phase1_d` is the strictly better checkpoint on every held-out metric, and it is now **anchor v2**. The operative gate it was launched to clear needed two further search fixes that round `d` could not have made — but coverage at 0.990 is what allowed the mis-ordering to surface at all, because at 0.83 the search rarely proved two wins in the same position.

### The M3 no-regress baseline, pinned by measurement (2026-08-10)

Re-run before chunk 10 so the number the campaign is judged against is one this
build produced, not one inherited from a document.

```
$ python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_2.epd \
      --sims 128 --ckpt runs/phase1_anchor.pt --seed 0
mate_in_2.epd: 197/200 solved at 128 sims (98.5%) | eval=runs/phase1_anchor.pt root=gumbel profile=eval(deterministic) m=16 c_visit=50 c_scale=1 | gate >= 95%
  failed: mate2_129, mate2_165, mate2_166
$ ... --seed 7                                      # identical, as the profile requires
mate_in_2.epd: 197/200 solved at 128 sims (98.5%) | ... | gate >= 95%
  failed: mate2_129, mate2_165, mate2_166
```

| | |
|---|---|
| **M3 no-regress baseline** | **0.985** (197/200), anchor v2 `27dcdd05…`, 128 sims, eval profile |
| M3 clause (unchanged) | mate-in-2 **≥ 0.95 absolute** on the frozen gate suite |
| Seed identity | 0 and 7 byte-identical, including the failure list — the eval profile draws no noise |

**The brief expected ≈0.965 and the correction it anticipated had not in fact
been applied to this path.** The promotion cell-mass change — dividing a
promotion cell's prior by 0.25 across its four edges, as chunk 5's docstring
described but its code did not — was deliberately scoped to
`promo_prior_from_head=True` and **not** made unconditional, because making it
unconditional costs 0.985 → 0.965: on four gate positions the key promotion falls
out of Gumbel's top-16. `search.py:370-380` carries that reasoning in the code.
So there is nothing to retire: 0.985 is not a stale number produced by superseded
code, it is what the default path produces today, twice, deterministically.

The three failures are exactly the three watchlist FENs below — the same
2 COVERAGE + 1 CAPABILITY rows, unchanged. That is the useful part of this
re-measurement: it confirms the residual is still the *same three positions*,
so the watchlist remains a valid instrument rather than a coincidence of counts.

**Both promotion questions are M3-end events, and they are one package.**
Flipping `promo_prior_from_head` (review S4) *is* the genuine cell-share: the flag
changes the head's involvement and the share arithmetic together, because the
share is `cell × P(promo | from, to)` and there is no head-free version of it that
is not just the ×0.25 uniform. Neither is a mid-campaign change.

**Pre-registered follow-up, decided before the measurement (2026-08-10).** If the
M3-end re-measure of the gate *regresses* when the flag goes on, the sanctioned
next step is exactly **one attribution run — share-only, i.e. uniform ×0.25 with
the head still out of it — before any conclusion is drawn about the head.** Two
changes moved at once, so a regression is unattributed until they are separated,
and "the trained promo head is bad" is the conclusion that would otherwise get
written down on no evidence. One run, then the conclusion.

Why the current default survives on its merits, stated on the right grounds:
a prior is a **heuristic, not a correctness property** — the defect was the
docstring's false claim about what the code did, and that has been fixed. Changing
the default path's prior shape mid-campaign would perturb the semantics of a
frozen gate to answer a question with no ground truth, and the principled version
already exists behind a flag, waiting for a head with real promotion signal to
divide by.

### M3 watchlist — the three residual gate positions

The three positions the 0.985 gate still misses, carried forward so chunk 10's ladder can track
them. Machine-readable copy: `benchmarks/m3_watchlist.json`. These are **not** a gate; they are
the concrete thing "Phase 2 value learning should sharpen" means, and the value head's estimate
on each key-move child is the number to watch move.

| id | verdict | FEN | value head on key child | why it fails |
|---|---|---|---|---|
| `mate2_129` | CAPABILITY | `2r3n1/4N3/p4p1r/k5p1/b1P1PPPp/7P/1RNPB1R1/R2K4 w - - 0 1` | **+0.036** | key move considered but got 2 sims; needs 17 on that edge to expand all 16 replies and reach a proof |
| `mate2_165` | COVERAGE | `8/k7/B7/3R1p1p/R2P1P2/3K4/8/1N2Bn2 w - - 0 1` | **+0.030** | key move outside the prior's top-16, so never simulated |
| `mate2_166` | COVERAGE | `8/k7/B7/4Rp1p/R2P1P2/3K2n1/8/1N2B3 w - - 0 1` | **+0.032** | same |

All three are **won positions the value head scores at ≈ +0.03** — not merely imprecise but
nearly indifferent, which is exactly the deficiency self-play value training addresses. The two
COVERAGE rows also need the *policy* prior to rank the key move top-16, so they are the ones
that would move if Phase 2 sharpens the prior rather than only the value.

Read these from here in chunk 10; do not re-derive them, and do not add them to any anchor or
tuning set — they are gate-suite positions (`tests/suites/mate_in_2.epd`).

**Campaign-rule lesson worth carrying (a sibling to rule 5).** *An aggregate residual is not a diagnosis.* Three rounds treated "conversion" as a single quantity and reasoned about it in aggregate. One triage script that classified each failing position separately produced a nine-position bug with a one-line remedy. When a gate is short, enumerate the failures before theorising about the shortfall.

---

## Round template (copy for `phase1_d`, …)

```markdown
### Round `phase1_X` — planned: <one-line intent>
**Deltas vs `W`:** <exhaustive list>
**Pre-flight:** <data gen if any; mode audit; timing slice if config/stack changed>
**Config:** configs/phase1_X.yaml → <yaml delta>
**The run:** <exact command, new --out, seed noted>
**Post-run:** <eval command against frozen eval sets>
**Pre-registered expectations:** <per gate>
**Decision rules:** <what result leads to what next round>
---
**Result (filled after):** <gate table row> | git SHA | wall-clock | skips | anomalies
**Decision taken:** <next round / phase closed / BLOCKED>
```

---

## Phase 2 (placeholder — activate at Chunk 9)

Phase 2 rounds are iteration campaigns, not epoch runs, and get their own section here when the loop exists. The cadence will be: launch `scripts/train_phase2.py --out runs/p2_<letter>` from the Phase 1 anchor checkpoint, ladder-eval every 10 iterations against the frozen anchors (Chunk 10), and judge rounds by Elo-vs-anchor CI separation — never by loss. The same rules apply: fresh letter per round, one lever, pre-registered expectations, frozen anchor ladder, gates don't move. First planned rounds: `p2_a` (defaults, 24h, M3 gate) and `p2_b` (PUCT ablation at identical budget).
