# BLOCKED — Chunk 6 gate: mate-in-2 ≥ 95% at 128 sims

**Date:** 2026-08-08 · **Branch:** `chunk-06-gumbel` · **Status:** **RESOLVED 2026-08-08 —
reopen Phase 1 and fix the model; the gate does not move.**

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
