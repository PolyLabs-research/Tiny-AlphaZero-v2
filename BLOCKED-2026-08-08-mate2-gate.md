# BLOCKED — Chunk 6 gate: mate-in-2 ≥ 95% at 128 sims

**Date:** 2026-08-08 · **Branch:** `chunk-06-gumbel` · **Status:** **STILL BLOCKED as of
2026-08-09 — gate 0.940 vs 0.95 after both pre-registered stages. Back with Tom.**

History: opened 2026-08-08 at 0.770 → decision "fix the model, the gate does not move" → round
`d` 0.890 → evaluation profile 0.940 → bounded tuning 0.940. The threshold has never moved.
Prior coverage, exploration noise and the σ/m constants are now all excluded as explanations;
what remains is search conversion at 128 sims. Read "Outcome, 2026-08-09" below first.

## Decision (Tom, 2026-08-08)

Reopen the Phase 1 closure per return condition 1 and run a targeted round `d`. **The chunk 6
threshold of ≥95% is unchanged**, and neither option 1 (lower the threshold) nor option 3
(re-scope to conditional-on-considered) was taken.

Rationale, for the record: `BLOCKED-2026-08-06` retired gates that measured properties the
pipeline never consumes — an unmasked argmax is never taken in Phase 2. The mate-in-2 gate
measures a property the pipeline consumes **directly**: Gumbel's top-m by `g + logits` *is*
the Phase 2 root mechanism, so the prior's rank coverage of forcing moves is load-bearing.
Opposite situation, opposite remedy. A retired proxy and a load-bearing measurement look
similar from the inside — the test is whether anything downstream reads the number.

**Return-to-BLOCKED condition:** round `d` fails to clear ≥0.95 on `mate2_top16_coverage`
*or* on the chunk 6 mate-in-2 re-gate, after the single permitted escalation (see
`TRAINING_STRATEGY.md` round `d`, decision rules).

## Round `d` outcome, 2026-08-09 — the condition has fired

Round `d` **succeeded at what it was designed to do and the gate still fails.**

| Measurement | Round `c` | Round `d` | Gate |
|---|---|---|---|
| `mate2_top16_coverage` (held-out) | 0.8100 | **0.9770** | ≥0.95 ✓ |
| `mate1_top8_masked` (no-regress) | 0.9800 | **0.9940** | ≥0.97 ✓ |
| chunk 6 mate-in-2 @128 sims | 0.770 | **0.890** (5 seeds: 0.875–0.900) | ≥0.95 ✗ |

**The two measurements disagree by 8.7 points, and the pre-registration named that
disagreement as itself the finding.** Decomposed on the suite with `phase1_d`:

```
prior top-16 coverage                 0.990   (round c: 0.83)  <- round d fixed this
   survives Gumbel noise into top-m   0.925   <- loses 6.5 points HERE
      solved given considered         0.941   (round c: 0.954)
                              gate =  0.870
```

`mate2_top16_coverage` was treated as an estimator of the gate. It is not — it is an
**upper bound**, the first of two multiplicative factors. Round `d` drove that factor
from 0.83 to 0.990, leaving essentially no headroom, and the gate still sits at 0.89
because the other two factors were never what round `d` addressed:

1. **Gumbel noise now dominates.** The considered set is top-m by `g + logits`, so a
   move the prior ranks *first* can still be knocked out by an unlucky draw. At 0.83
   coverage this was hidden inside the prior's own deficit; at 0.990 it is the largest
   single loss in the chain.
2. **Selection given consideration is 0.941** and barely moved (0.954 → 0.941), because
   converting a considered forcing move into a chosen one is a search property.

**The permitted escalation is aimed at a saturated axis.** Both offered levers
(`phase1_epochs: 3`, mixes 0.10/0.10) act on prior coverage, which is at 0.990 and was
already saturated after epoch 1 (epoch 1 → 2 moved it +0.001). It would buy at most
~0.5 points against a 6-point gap. Not spent; recorded instead.

**Gate measurement quality, for the threshold conversation:** over 5 Gumbel seeds the
gate reads 0.875–0.900, spread 0.025. The shortfall is far outside that noise, so this
is a real 6-point gap and not a seeding artefact — but a ±2.5-point gate is worth
knowing about when the threshold is 0.95 and a pass would be declared on one run.

**Phase 1 is NOT re-closed.** `runs/phase1_anchor.pt` still points at `phase1_c/final.pt`;
`runs/phase1_d/final.pt` is a strictly better checkpoint on every held-out gate and is
available to promote if the gate conversation resolves that way.

**What is now known that was not on 2026-08-08:** the gate is not prior-limited. It was,
and that was worth fixing — the prior is now nearly perfect on this task. What remains is
search-side, and the levers are chunk 6 configuration (`gumbel_m`, the Gumbel noise scale,
the σ formulation), not Phase 1 data. That is a different conversation from the one this
document opened, and it belongs to Tom.

Supervised anchoring is now demonstrated *sufficient* for what it was aimed at — prior
coverage 0.83 → 0.990 — and *insufficient* to clear this gate alone, which is a sharper
result than "it didn't work" and the reason the escalation was recorded rather than spent.

**This fires return condition 1 of `BLOCKED-2026-08-06-phase1-gates.md`**, verbatim:
*"Chunk 5/6 search gates fail: mate-in-1 < 100% @16 sims (stub), or **mate-in-2 < 95% @128
sims with this checkpoint**."* The mate-in-1 half did **not** fire (100/100). The mate-in-2
half has. Per that document, the Phase 1 gate-retirement decision is now formally reopened.

## Resolution path — pre-registered 2026-08-09, before any measurement

**Tom's decision on the round-`d` split result: the gate number does not move.** Not to 0.93,
not to "conditional on considered". Two staged interventions, both aimed at the search-side
losses the round-`d` decomposition exposed, with a measurement firewall between them.

### Stage 1 — an explicit evaluation profile

The gate has been testing **self-play exploration mode on a match-play question**. Gumbel's
root noise exists to diversify self-play data; it has no business in a suite, a ladder match,
or any other measurement of strength. Deterministic acting at evaluation is orthodox
Gumbel-AlphaZero practice (Danihelka et al. 2022 §5), and chunk 10's anchor ladder needs it
regardless — a ladder whose rungs move by ±2.5 points under reseeding cannot resolve Elo.

So this is not a gate dodge: it is a contract that should have existed from chunk 6, and it
happens to remove the single largest term in the decomposition (0.990 → 0.925, the 6.5 points
lost to noise knocking a top-ranked forcing move out of the considered set).

Contract — `MCTSConfig.root_noise: bool = True`, and when false:

- root candidate selection is deterministic **top-m by raw logits** (no Gumbel draws);
- final action selection is argmax over survivors of **`logits + σ(q̂)`** (no `g` term);
- Sequential Halving, σ, the solver, and the budget convention are **unchanged**.

Self-play keeps `root_noise=true`. Suites, ladder, and match play run the eval profile.

**Pre-registered projection for stage 1 alone: ≈0.93 ± 0.02.** This is expected to *still
fail*, which is why stage 2 exists. Arithmetic: removing the noise term recovers most of the
0.990 → 0.925 loss, leaving roughly `0.99 × 0.941 ≈ 0.93`.

**Stop-and-report rule:** if stage 1 lands far outside `[0.88, 0.95)`, stop before tuning and
report. A number outside that window means the three-factor decomposition is wrong somewhere,
and tuning on top of a wrong model of the problem is noise, not evidence.

### Stage 2 — one bounded tuning pass, behind a firewall

Only if stage 1 < 0.95. Sweep `eval_c_visit ∈ {25, 50, 100}` × `eval_c_scale ∈ {0.5, 1.0, 2.0}`
× `eval_m ∈ {16, 32}` at 128 sims, **on a disjoint 500-position tuning suite**
(`runs/data/mate2_tune_500`, seed 102), verified disjoint from both gate suites, all held-out
eval sets, and all anchor sets.

These overrides are **scoped to the eval profile** — new `eval_*` fields defaulting to the
self-play values. The self-play constants are not touched by anything here; tuning those is
Phase 2 evidence territory, not a gate conversation.

**The firewall:** the tuning suite is burned as tuning data the moment it is used and must
never gate anything. Selection is by tuning-suite solve rate alone, ties broken toward the
paper defaults (m=16, c_scale=1.0, c_visit=50) — minimal delta from the published
configuration wins. The gate suite is then touched **exactly once**, with the chosen config.
No re-pick after seeing the gate number: a second attempt with a different config is exactly
the overfit this firewall exists to prevent.

### Decision rules (pre-registered)

- **≥ 0.95** → re-close Phase 1: promote `runs/phase1_d/final.pt` to `runs/phase1_anchor.pt`,
  append the v2 digest to `runs/ANCHORS.sha256` keeping the dated v1 line, merge `round-d` →
  `main`, annotated tag `phase1-closed-r2` carrying the gate number and the chosen eval
  config, both BLOCKED docs closed, strategy-doc row + result block. The chosen constants are
  recorded as **configuration in `configs/default.yaml`, not lore**, and the plan's chunk 10
  section states that the ladder runs the eval profile.
- **< 0.95 after stage 2** → stop. Record the complete evidence chain here — coverage
  saturated at 0.990, noise removed, constants tuned, full table — and hand the gate-number
  conversation back to Tom. Do not iterate further; do not touch the threshold.

### Implementation note — where this code lands

The eval profile is a **chunk 6** feature (it is the Gumbel root-selection contract), and the
`src/tinyalphazero/mcts/` package exists only on the `chunk-05/06/07` chain — `round-d` has no
search code at all, so the work cannot physically go there. Stage 1/2 code and tests land on
`chunk-06-gumbel`; this document and `TRAINING_STRATEGY.md` are authored on `round-d`. The
measurement is unaffected either way: `phase1_d/final.pt` loads on both branches, because
`_resolve_field_types` drops config keys the branch does not know (`phase1_mate2_mix` here).

## Outcome, 2026-08-09 — both stages run, gate 0.940, **STILL BLOCKED**

Both pre-registered interventions were executed in order. The gate is **0.940** against a
threshold of 0.95. Per the pre-registered `< 0.95` rule: **stop, record, hand the gate-number
conversation back to Tom.** Nothing was promoted, merged or tagged. `runs/phase1_anchor.pt`
still points at `phase1_c/final.pt`, and the threshold was not touched.

| Stage | mate-in-2 @128 sims | Seed spread |
|---|---|---|
| round `c` anchor, self-play profile | 0.770 | — |
| round `d`, self-play profile | 0.890 | 0.875–0.900 (5 seeds) |
| **stage 1** — eval profile, paper constants | **0.940** | **0.000** (exact across seeds) |
| **stage 2** — eval profile, tuned constants | **0.940** | **0.000** |

### Stage 1 landed inside its pre-registered window

Projection was **0.93 ± 0.02**; the measurement was **0.940**, at the top edge. The
stop-and-report rule did not fire, so the three-factor decomposition was sound and stage 2
proceeded as authorised rather than as a reaction to a surprise.

The seed spread collapsing to exactly zero is worth as much as the point gain. Both seeds
agree on the score *and* on which twelve positions failed. The ±2.5-point gate noted on
2026-08-08 as "worth knowing about when the threshold is 0.95" is gone, and chunk 10's ladder
now has the deterministic acting it needs to resolve Elo.

### Stage 2 bought 1.2 points of tuning-suite gain and transferred none of it

Sweep on `runs/data/mate2_tune_500` (disjoint, verified), 18 configs, 128 sims, full table in
`PROGRESS.md` on `chunk-06-gumbel`. Range 90.6–93.4%; paper defaults 92.2%; selected
`m=32, c_visit=100, c_scale=0.5` at 93.4% — a unique maximum, so the tie-break toward defaults
never came into play.

Single-shot gate with the chosen config: **188/200 = 0.940. Identical to the untuned profile,
on a different set of twelve failures.**

That is the firewall earning its keep, and it is the most transferable thing this session
produced. The binomial standard error at n=500, p≈0.93 is ±1.14 points; the whole grid spans
~2.5 SE. Selecting the maximum of 18 such configs is mostly selecting noise, and the honest
out-of-sample number says so. **Had the sweep been run on the gate suite, 93.4% would have
gone into this document as a tuning win.**

### The decomposition now closes exactly, and the residual has one name

`scripts/diagnose_mate2.py`, gate suite, eval profile:

| Factor | m=16 (defaults) | m=32 (selected) |
|---|---|---|
| prior coverage @ top-m | 0.990 | 0.995 |
| solved given considered | 0.949 | 0.945 |
| **product** | **0.940** | **0.940** |
| measured gate | 0.940 | 0.940 |

Under noise the chain read `0.990 × 0.925 × 0.941`. The middle factor is now **1.000 by
construction**. Prior coverage is saturated (top-1 0.660, top-8 0.975, top-16 0.990, top-32
0.995; median forcing-move rank 0). **The entire residual is conversion**, and `m` trades
against it almost exactly — +0.005 coverage for −0.004 conversion, product unchanged to three
decimal places. Twelve failures decompose as **2 never considered + 10 considered and not
converted**. Clearing 0.95 at coverage 0.990 requires conversion ≥ 0.960, up from 0.949.

### What is now excluded, and what the conversation is actually about

Four candidate explanations have been tested and three are closed:

1. ~~the prior can't rank forcing moves~~ — round `d`: coverage 0.83 → 0.990;
2. ~~exploration noise is eating the considered set~~ — stage 1: that term is now 1.000;
3. ~~the σ constants and `m` are mistuned~~ — stage 2: 18 configs, best transfers zero;
4. **what the search does with 128 simulations once the right move is already in front of it** —
   untested, and now the only thing left.

Every remaining lever is inside chunk 6/7 search behaviour, not Phase 1 data and not the eval
configuration. Ten positions where the winning move *is* simulated and still isn't chosen is a
small, concrete, individually-inspectable set — that is a real diagnosis with a next step, not
a plateau.

**Three attempts were spent on 2026-08-08 (per `AGENTS.md`) and the gate was not weakened
then or now.** The threshold conversation is Tom's; the options remain the three listed at the
bottom of this document, now with the prior, the noise and the constants all eliminated as
explanations.

## The failure

```
$ python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_2.epd \
      --sims 128 --ckpt runs/phase1_anchor.pt
mate_in_2.epd: 154/200 solved at 128 sims (77.0%) | eval=runs/phase1_anchor.pt root=gumbel | gate >= 95%
```

Every other chunk 6 gate passes: mate-in-1 100/100 at 16 and at 48 sims (stub), Sequential
Halving budget exact for all tested (n, m), `improved_policy` well-formed and >0.9 on a
proven mate. 185 tests pass, ruff clean.

## Diagnosis — the search is at its ceiling; the *policy prior* is the binding constraint

Diagnosis order was as instructed: search first (stub), net second.

| Measurement | Result |
|---|---|
| Stub evaluator (search alone) | 30.5% |
| `phase1_anchor.pt` | 77.0% |
| **Solved *given* the forcing move was in the considered set** | **95.4%** (144/151) |
| Forcing move in considered set at m=16 | 75.5% (151/200) |

The conditional number is the finding: **when search gets to look at the winning move, it
solves the position 95.4% of the time — at the gate.** What fails is *consideration*.
Gumbel takes the top-m by `g + logits`, so an action the network's prior ranks below ~16th
is never simulated at all, and no amount of search budget recovers it.

Network prior rank of the forcing move (first 60 positions):

```
median rank=1   top1=42%   top8=73%   top16=83%   top32=93%
```

`top16 = 83%` is a hard ceiling on a 95% gate at m=16, independent of search quality.

Raising `m` does not rescue it, because consideration and per-action budget trade off:

| m | 16 | 24 | 32 | 48 | 128 (all) |
|---|---|---|---|---|---|
| solved | 76.0% | 73.5% | 80.0% | 81.0% | 81.0% |

Nor does budget (first 60 positions, m=32): 128 → 81.7%, 256 → 81.7%, 512 → 91.7%,
1024 → 86.7%. Non-monotone and plateauing below the gate at 8× the specified sims.

## Why this is a real tension, not a tuning miss

`BLOCKED-2026-08-06` retired `mate1_solve_rate` on the argument that *"chunk 5's own gate
proves mate-in-1 correctness is search's job (100% at 16 sims with a uniform stub
evaluator)"*. That argument held and still holds — mate-in-1 is 100% with **no network at
all**.

It does not extend to mate-in-2. A mate-in-2 cannot be found by terminal detection from the
root; the first move must be *considered*, and consideration is decided by the policy
prior. So the chunk 6 gate asks the Phase 1 policy head for exactly the tactical ranking
quality that Phase 1 deliberately stopped gating. The anchor was trained on random play and
has never seen a tactical position; `top16 = 83%` is a fair description of that.

Both decisions are individually defensible. Together they are inconsistent, and this gate is
where the inconsistency surfaces.

## Attempts made (3, per AGENTS.md; the gate was not weakened)

1. **Exact value when expansion reveals a mate.** Returning the evaluator's estimate for a
   node that just proved a mate available dilutes the parent's Q with a known-wrong number.
   → 75.0% → 76.0%. *Also exposed a regression:* deeper proven wins then tie with immediate
   mates at q=+1 and `g+logits` broke the tie by noise, costing 2/100 on mate-in-1.
2. **`IMMEDIATE_WIN_BONUS`.** A shorter forced mate is strictly better play; an exactly-known
   immediate win must not lose a coin-flip to a longer one. → mate-in-1 back to 100/100 at
   both 16 and 48 sims. mate-in-2 unchanged at 76.0%.
3. **MCTS-Solver proven-value propagation** (Winands et al. 2008), kept in `proven` /
   `proven_value` separate from `terminal` so rule 2 still holds. A refuted reply set makes
   the parent edge *proven*, immune to Sequential Halving elimination, where an average
   would still read well under 1. → 76.0% → 77.0%.

All three are principled improvements and are **kept** — they are correct regardless of this
gate. None addresses the binding constraint, because none of them can make the search
consider a move the prior ranked 20th.

## What I did not do

- Did not lower 95%, widen the solved-predicate, or make the suite easier.
- Did not change `gumbel_m` from the spec default of 16 to flatter the number — and it would
  not have worked anyway (81% at best).
- Did not proceed to chunk 7.

## The decision, which is not mine

Roughly, the options:

1. **Accept a lower chunk 6 threshold**, on the reasoning that 95% was calibrated against an
   assumed prior quality that Phase 1 explicitly declined to gate. 95.4%-given-considered is
   the number that measures *search*, and it passes.
2. **Reopen Phase 1** per the return condition — e.g. a tactical-anchor mixture so the prior
   ranks forcing moves top-16, which is the one lever that moves this gate.
3. **Re-scope the gate to measure search**, e.g. conditional-on-considered, or with an
   oracle-augmented considered set — measuring the thing chunk 6 actually builds.

Option 1 is a threshold change and needs the same deliberateness as the last one. Option 2 is
expensive but is what the return condition contemplated. Option 3 changes what is being
measured and should be argued for explicitly, not slid in.

## Reproduce

```bash
python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_2.epd --sims 128 --stub
python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_2.epd --sims 128 \
    --ckpt runs/phase1_anchor.pt
pytest tests/test_mcts.py -q
```

Note: the plan's chunk 6 DONE-WHEN cites `runs/phase1_a/final.pt`. That is the **invalid**
first run (`BLOCKED`-documented, all four §5 gates missed). The correct anchor is
`runs/phase1_anchor.pt`; the plan predates the campaign and its path is stale.
