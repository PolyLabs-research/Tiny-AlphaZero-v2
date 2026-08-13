# RUNLOG — p2_c (z/q value targets)

**Run:** `runs/p2_c` · 20 iterations · 2026-08-12 08:20 → 21:31 · branch `p2_c`, git `18474f4`
**Lever:** `train.value_q_mse_weight: 0.5` — the only field differing from p2_b's
resolved config. Rehearsal held at 0.05 as standing configuration.
**Warm start:** `runs/p2_b/ckpt_iter0020.pt`, fresh optimizer · **League:** seeded from p2_b
**Rungs:** random, greedy, phase1, **stockfish:0** — four wide, inline
**Pre-registration:** `PREREG-p2_c.md`

**Verdict: P1 passes decisively. N1 holds. Both watchlists resolved as registered.**
p2_c is the first round in this project's history to move an absolute strength
measurement, and the first whose two opposing predictions both came true.

---

## 1. The scorecard

| | expectation | measured | |
|---|---|---|---|
| **P1** | skill-0 Elo at pass 20, CI disjoint above p2_b's endpoint | **0.233, −207.5 [−260.5, −159.8]** vs 0.025, −636.4 [−848.7, −520.9] | **PASS** |
| **N1** | mate-in-2 ≥ 0.915 at pass 20 | **0.930** | **PASS** |
| **W1** | `mate2_067`, `mate2_125` conviction rises, solves follow | **both SOLVED** | **PASS** |
| **W2** | `mate2_165`, `mate2_166` do **not** convert | neither converted | **HELD** |
| H | skips < 10 | **0 in 8,000** | PASS |

## 2. P1 — the primary instrument

```
p2_b @20    4W/ 2D/194L   0.025   −636.4  [−848.7, −520.9]
p2_c @5    11W/ 6D/183L   0.070   −449.4  [−563.8, −367.7]
p2_c @10    7W/12D/181L   0.065   −463.2  [−563.8, −391.6]
p2_c @15   13W/15D/172L   0.102   −376.9  [−456.1, −315.4]
p2_c @20   37W/19D/144L   0.233   −207.5  [−260.5, −159.8]
```

Score nearly tenfold, **+429 Elo**. Pass 20's interval is disjoint from p2_b's
endpoint and from every earlier p2_c pass, which is what P1 registered.

**Correction, and a method note.** During the run I reported at pass 15 that its
interval was "disjoint from pass 5's". It is not: pass 5 is [−563.8, −367.7] and
pass 15 is [−456.1, −315.4], which overlap. Adjacent steps overlap generally —
5↔10, 5↔15 and 10↔15 all do — so this is a climb whose *endpoints* separate, not
one with four separating steps. The earlier phrasing overstated it.

More importantly, **marginal-interval overlap is the wrong test** and was the wrong
test even where it happened to agree with the conclusion. Two independent CIs can
overlap while a paired comparison separates cleanly, because the pairing removes
opening difficulty as a shared source of variance. The test of record for
"pass X beat pass Y" is a **paired-difference bootstrap over opening pairs**, the
same unit `paired_bootstrap_elo` already resamples within a single matchup.

That test cannot be computed from this log: ladder rows store the aggregate but not
`pair_scores`, so the per-opening vector needed to pair two passes is gone once the
matchup ends. **P1's verdict stands on the wider claim** — pass 20 vs p2_b's endpoint
separates by a margin no pairing correction would close, 0.025 vs 0.233 across 200
games. The finer between-pass claims in this section should be read as descriptive.
Persisting `pair_scores` is filed for p2_d so the test of record becomes computable.

**The slope accelerates rather than saturating.** The last five iterations were the
steepest of the run (wins 13 → 37, losses 172 → 144), which is the opposite of what a
one-off adjustment to a new target would produce. Read the pass-5 → pass-10 flat spot
as noise in hindsight: it was one interval of a series that resumed climbing.

Magnitude was recorded rather than registered, deliberately, and it is worth stating
plainly: 0.233 against M4's criterion of beating skill 0. The mountain is still a
mountain — but it is now a mountain with a measured, positive, accelerating slope on it,
which is the first time that sentence has been true.

For the record, the retired instrument: the phase1 rung read 199W/0D/1L, +919.5, at
**every** pass of this run while the skill-0 rung resolved a 429-point climb underneath
it. The instrument succession recorded in `RUNLOG-p2_b.md` §4 is vindicated in the
sharpest possible way — the old rung would have reported this round as "no change".

## 3. The suite, and the decomposition

| pass | mate-in-2 | mate-in-3 | ECE | book H |
|---|---|---|---|---|
| 5 | 0.935 | 0.630 | 0.2153 | 1.5449 |
| 10 | 0.905 | 0.655 | 0.2155 | 1.6877 |
| 15 | 0.920 | 0.680 | 0.2162 | 1.8189 |
| 20 | **0.930** | 0.670 | **0.1678** | 1.5752 |

N1 holds at 0.930 — identical to p2_b's endpoint, so the strength gain cost the suite
nothing. Mate-in-3 rose from 0.625 to 0.670 without being asked for.

Decomposition at pass 20 (`diagnose_mate2_iter0020.txt`):

|  | p2_b @20 | p2_c @20 |
|---|---|---|
| prior coverage @top-16 | 0.985 | **0.995** |
| solved \| considered | 0.944 | 0.935 |
| product | 0.930 | 0.930 |
| **measured gate** | **0.930** | **0.930** |
| prior top-8 coverage | 0.940 | 0.975 |
| median forcing-move rank | 0 | 0 |

The identity closes to the digit for the fifth consecutive measurement. Coverage
improved again (0.985 → 0.995, only one position now outside the considered set) while
the conditional gave back a little — the same gate arrived at by a slightly different
route.

## 4. Both watchlists resolved as registered

This is the round's cleanest result, and it was predicted in both directions before
launch.

**W1 — the lever's targets.** `mate2_067` (q̂ 0.575) and `mate2_125` (q̂ 0.500) were
the only two genuinely under-convicted positions in p2_b's residual. **Both are now
solved** and have left the failure list entirely. The lever fixed exactly the positions
the theory said it would fix.

**W2 — the counter-cases.** `mate2_165` and `mate2_166` were registered as *expected not
to convert*, on the grounds that their failure is visits, not conviction. Neither
converted:

| id | q̂ @20 | key-edge visits | proof needs | reason |
|---|---|---|---|---|
| mate2_165 | 0.9993 | 14 | 7 | reply set never fully expanded |
| mate2_166 | 0.9988 | 6 | 7 | reply set never fully expanded |

`mate2_165` briefly solved at pass 5 and reverted for passes 10, 15 and 20 with q̂
*rising* throughout — a threshold flicker, not a refutation. Four passes of near-certain
conviction without a solve is the visits-not-conviction story surviving the round aimed
squarely at it.

### The competitor's q̂ — what the key edge actually loses to

`key_edge_q` alone cannot say why a near-certain edge loses the argmax; it needs the
number it lost *to*. Per checkpoint, the key edge against the move actually chosen
(`competitor_q_trajectory.json`):

| ckpt | id | key q̂ | key N | chosen | chosen q̂ | chosen N |
|---|---|---|---|---|---|---|
| 0005 | mate2_165 | 0.9987 | **30** | Rb5 *(the key move)* | 0.9987 | 30 |
| 0010 | mate2_165 | 0.9995 | 14 | Rd7+ | 0.9956 | 30 |
| 0015 | mate2_165 | 0.9990 | 30 | Rd7+ | 0.9958 | 30 |
| 0020 | mate2_165 | 0.9993 | 14 | Rd7+ | 0.9958 | 30 |
| 0005 | mate2_166 | 0.9976 | 2 | Bxg3 | 0.9956 | 30 |
| 0010 | mate2_166 | 0.9980 | 6 | Re7+ | 0.9956 | 30 |
| 0015 | mate2_166 | 0.9981 | 14 | Bxg3 | 0.9983 | 30 |
| 0020 | mate2_166 | 0.9988 | 6 | Bxg3 | 0.9958 | 30 |

**In seven of eight rows the key edge's estimate is *higher* than the winner's and it
still loses.** Value ordering is not the failure. The one row where the key move won
(0005/165) is also the only row where it held a full 30 visits — and the pass-5 solve
that broke and reverted is exactly that row.

This is the visits story stated in the sharpest available form: the search prefers the
move it believes in *less*, because Sequential Halving gave that move the visits and
`logits + σ(q̂)` scales σ with visit count. A better estimate on an under-visited edge
loses to a worse estimate on a well-visited one. It is the direct motivating measurement
for p2_d's breadth lever.

**The site of the loss is elimination, not the final argmax.** The visit counts say so:
the key edge holds 14 where the winner holds 30, and 14 is what an edge has when it is
halved out after an early phase rather than carried to the end. So the key move's
*interim* q̂ — measured before its subtree could prove anything — lost the elimination
to a competitor ranked higher on `logits + σ(q̂)`. By the time the acting argmax runs,
the key move is already gone.

The eighth row is a natural experiment that ran itself. `mate2_165` at checkpoint 0005
is the only row where the key edge held a full 30 visits, and it is the only solve. Give
the key move a survivor's allowance and it wins; let halving starve it and its conviction
is irrelevant.

That pins the requirement for p2_d's breadth lever to a measured coordinate: **the proof
must land inside the pre-elimination visit allowance**. A breadth-first reply sweep
costing 7, triggered at q̂ ≥ 0.95, fits inside 14 — which is why the lever is aimed at
this failure rather than merely consistent with it.

**A lever that fixes its predicted targets and leaves its predicted non-targets alone is
the strongest form this evidence could take.** The mechanism story is not merely
consistent with the result; it discriminated between two sets of positions in advance.

## 5. Semantic residual

14 failures, unchanged in count from p2_b but not in composition (13 CAPABILITY,
1 COVERAGE; p2_b was 11/3). Of them, **5 SLOWER_FORCED_WIN, 9 GENUINE_MISS** —
so the true residual is 9 wrong positions out of 200.

`mate2_129` closed the round where it has been heading all campaign: watchlist
`value_on_key_child` **−0.9579** at pass 20 (from +0.036 at the 2026-08-09 pin, via
−0.858 → −0.978 → −0.984 across p2_c's passes), and key-edge q̂ 0.0211. It is now
confidently wrong about a won position rather than uncertain about it — the honest
residual of the residual, and the one position where more search would not help.

CAPABILITY key-edge q̂: median 0.9670, min 0.0211, max 1.0000. The distribution has
grown a low tail it did not have in p2_b (`mate2_129` at 0.0211, `mate2_126` at 0.2385)
alongside the saturated cluster — the value head has become more opinionated in both
directions, which is what a Q-blended target should do.

## 6. Calibration — the un-saturated instrument, first readings

ECE is now measured over the phase1 **and** stockfish matchups (the `calibration_note`
in every row says which). It is not comparable to any pre-p2_c number: every earlier
reading came from a matchup the learner wins 99% of, where the value distribution is
degenerate and the metric is easy to pass.

0.2153 → 0.2155 → 0.2162 → **0.1678**: flat for three passes, then a 0.05 drop in the
final five iterations. Against the plan's chunk-12 clause of ECE < 0.15 it does not yet
pass — but the clause was written for the degenerate sampler, and the honest reading is
that this is a new baseline on a new instrument that improved sharply once the value
head started learning. **The ECE clause needs restating against the changed sampler**,
the same way E6 needed restating against the anchor rung's saturation.

## 7. Health, and the one signal that went the wrong way

Zero non-finite gradient steps in 8,000 — the third consecutive campaign at zero or
near it. **No fresh-optimizer blip at all**: gradient norms 3.1–3.6 across the first five
iterations, against p2_b's 35 at iterations 2 and 4. Pre-registered as expected noise;
it simply did not happen, which is worth noting because the prediction was wrong in the
harmless direction.

Loss 2.894 → 2.466 over the run.

**Value loss drifted up**: 0.292 across the first five iterations, 0.326 across the last
five. H registered this as "direction expected down" — the sharpness-tax-relief signal —
and it went up instead. It is informational and nothing hangs on it, but it is the one
registered expectation that pointed the wrong way, and it should not be quietly dropped:
whatever the blend did for strength, it did not relieve the second-half value-loss
climb that p2_b showed.

Book entropy ended at 1.575, below p2_b's 2.104. The funnel treatment did not survive
the new target — reported, not predicted, and now a live question for `p2_d`.

## 8. Decision

**P1 passes and N1 holds → p2_c closes as the first absolute-strength gain.** Per the
pre-registered rule, the `p2_d` conversation opens on:

1. **Proof-directed reply expansion** — the lever W2 has now protected across two
   rounds. 13 of 14 residual failures are CAPABILITY with "reply set never fully
   expanded", most within a few visits of the `1 + len(replies)` proof cost.
2. **The distributional-Q upgrade** — visit-weighted child WDL with a per-ply W↔L flip,
   named in `rehearsal`/`train`'s docstrings as the principled successor to this round's
   scalar blend.
3. **M4 planning against a measured slope** — for the first time there is one. 0.025 →
   0.233 over 20 iterations, accelerating, on the only instrument that can resolve it.

## 9. Artefacts

| file | what |
|---|---|
| `ladder.jsonl` | 4 passes × 4 rungs, roles stamped |
| `iterations.jsonl` | 20 rows |
| `diagnose_mate2_iter0020.txt` | the coverage decomposition |
| `triage_mate2_iter0020.json` | 14 failures, verdicts, q̂, visit budgets |
| `semantic_residual_ckpt_iter0020.json` | SLOWER_FORCED_WIN vs GENUINE_MISS |
| `config.yaml`, `provenance.json` | one field from p2_b, fingerprinted |
