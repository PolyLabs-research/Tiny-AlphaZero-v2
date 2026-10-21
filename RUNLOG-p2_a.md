# RUNLOG — p2_a (M3 campaign)

**Run:** `runs/p2_a` · 50 iterations · 2026-08-10 07:55 → 2026-08-11 09:26 · 24.35 h
of iteration wall clock · git `fce3777` · AMD Radeon 8060S, torch 2.12.1+rocm7.2
**Init:** `runs/phase1_anchor.pt` (anchor v2) · seed 42 · config fingerprint `f8588828d1b6`

M3's 24-hour criterion is the spec's real go/no-go (§11, plan chunk 12). This is
the ledger.

---

## 1. Verdict

| Clause | Source | Required | Measured | |
|---|---|---|---|---|
| Elo vs anchor, endpoint CIs | spec §11 | non-overlapping, final > first | iter 10 `[+301.3, +463.2]` vs iter 50 `[+726.9, +1040.4]` — disjoint | **PASS** |
| Elo vs anchor, monotone | spec §11 (wording) | monotone | +372.2 → +919.5 → +798.2 → +576.2 → +919.5 | **FAIL** — see erratum #7 |
| mate-in-2 ≥ 0.95 | spec §11, plan chunk 12 | ≥ 0.95, no-regress from 0.985 | **0.830** (166/200) | **FAIL** |
| value ECE < 0.15 | plan chunk 12 | < 0.15 | 0.0479 (max across the five passes: 0.0479) | **PASS** |

**M3 does not pass.** It fails on mate-in-2, by a margin (0.830 against 0.95)
that no reading of the clause rescues, and against an entry baseline of 0.985
that makes it a regression rather than a shortfall.

The monotone clause is recorded FAIL as written and is the subject of
**erratum #7**: monotonicity of a noisy sampled series through a saturating rung
is not a meaningful demand, and the endpoint-CI requirement is what the criterion
always meant. The erratum does not change this verdict — mate-in-2 fails either
way — which is why it was safe to write it after seeing the data.

ECE is a plan-chunk-12 clause, not a spec §11 one. It is recorded as a clause and
it passes; see §6 for why the number is worth less than it looks.

---

## 2. The five protocol passes

Ladder passes at iterations 10 and 20 ran in-flight. Passes at 30, 40 and 50 were
**backfilled offline on 2026-08-11** after the campaign stopped, at the same seed
(42), config, frozen book and frozen suites. Determinism was measured, not
assumed: re-running iteration 20's pass offline reproduced the in-flight row
exactly — 200/200, 200/200, 199W/0D/1L, mate2 175/200 with an identical failed-ID
list, mate3 115/200, ECE 0.0034 to four decimals. `runs/p2_a/ladder.jsonl` holds
all five.

| iter | phase1 anchor | Elo [95% CI] | random | greedy | mate-in-2 | mate-in-3 | ECE | book H |
|---|---|---|---|---|---|---|---|---|
| 10 | 179W/0D/21L | +372.2 [+301.3, +463.2] | 200/0/0 | 164W/31D/5L | 0.880 | 0.605 | 0.0404 | 2.796 |
| 20 | 199W/0D/1L | +919.5 [+726.9, +1040.4] | 200/0/0 | 200/0/0 | 0.875 | 0.575 | 0.0034 | 2.635 |
| 30 | 198W/0D/2L | +798.2 [+636.4, +1040.4] | 200/0/0 | 200/0/0 | 0.890 | 0.555 | 0.0346 | 2.507 |
| 40 | 193W/0D/7L | +576.2 [+478.0, +728.7] | 200/0/0 | 200/0/0 | 0.895 | 0.595 | 0.0272 | 2.228 |
| 50 | 199W/0D/1L | +919.5 [+726.9, +1040.4] | 200/0/0 | 199W/1D/0L | **0.830** | 0.570 | 0.0479 | 1.905 |

Both lower rungs are saturated from iteration 20 onward and carry no further
information about this run.

### Diagnostic pass (not part of the verdict)

`runs/p2_a/ladder_diagnostic.jsonl`, on `ckpt_iter0048.pt` — the checkpoint
nearest the first-move-entropy peak — run to separate "0050 reads low because the
run oscillates" from "0050 reads low because it was sampled mid-excursion". Same
seed, config, book and suites as the protocol passes; excluded from the verdict by
provenance, not by incomparability.

| iter | phase1 anchor | Elo | mate-in-2 | mate-in-3 | ECE | book H |
|---|---|---|---|---|---|---|
| *48* | *198W/0D/2L* | *+798.2 [+636.4, +1040.4]* | *0.840* | *0.595* | *0.0568* | *2.004* |

**48 and 50 agree on the suites** (168/200 vs 166/200). The entropy excursion is a
training-metrics phenomenon that suite performance rides through; the mate-in-2
decline from 0.895 at iteration 40 to ~0.835 across 48/50 is real, not an artefact
of which checkpoint the protocol happened to land on.

---

## 3. The decomposition (`ckpt_iter0050.pt`)

Both halves measured at the frozen gate profile, 128 sims:

```
prior coverage @top-16      0.890   (178/200)
solved | considered         0.933   (166/178)
product                     0.830
measured gate               0.830   (166/200)   <- identical
prior top-1  coverage       0.250
prior top-8  coverage       0.755
prior top-32 coverage       1.000
median forcing-move rank    2
```

Triage of all 34 failures (`runs/p2_a/triage_mate2_iter0050.json`):

| verdict | n | share | meaning |
|---|---|---|---|
| COVERAGE | 22 | 65% | key move outside the prior's considered set — never simulated |
| CAPABILITY | 12 | 35% | considered, but eliminated on an interim value estimate |
| BUG | **0** | 0% | — |

Key-edge q̂ across the CAPABILITY dozen: min 0.524, **median 0.795**, max 1.000;
4 of 12 above 0.99. "Value ties everything near 1.0" is real but is the minority
mode, not the main one.

**Neither factor clears the gate alone.** Perfect coverage at the current
conditional gives 0.933, still short of 0.95. `prior top-32 coverage = 1.000`
says the remaining coverage gap is entirely in ranks 17–32 — a shallow tail, and
one that the frozen eval profile's m=16 defines the edge of. Widening m is a
measurement change, not a model change, and is deliberately not on the table
while the profile is frozen.

Watchlist at iteration 50 — value conviction has arrived, consideration has not:

| id | verdict | solved | q on key child | baseline |
|---|---|---|---|---|
| mate2_129 | CAPABILITY | no | 0.4387 | 0.036 |
| mate2_165 | COVERAGE | no | 0.9999 | 0.0297 |
| mate2_166 | COVERAGE | no | 1.0000 | 0.0321 |

165 and 166 now value the key child at ~1.0 and still fail, because the move is
not in the considered set. That is the coverage story in two positions.

---

## 4. The funnel

Book move-1 entropy, measured on the frozen 100-opening book at every pass:

```
iter 10   2.7958
iter 20   2.6354
iter 30   2.5071
iter 40   2.2276
iter 50   1.9052      (-32% from pass 1, still falling at stop)
```

**Monotone across all five passes** — the only series in this run that never
reverses. Elo oscillates around it; mate-in-2 tracks it late. The model is
progressively narrowing the openings it will play, and the coverage tail
(ranks 17–32) is where the moves it has stopped considering have gone.

Per-iteration first-move policy entropy shows the same dynamic at higher
frequency, and oscillates: 0.0003 at iteration 37 and 0.0006 at 38 (self-play
collapsed to essentially one line), 0.648 at 40, 0.647 at 48, 0.120 at 50.

**Leading hypothesis, held as hypothesis:** self-play funnelling narrows the
prior; the narrowed prior drops forcing moves out of the top-16; the gate falls.
The 30/40 backfill is consistent with it (book H falling monotonically while
coverage-limited mate-in-2 declines late). It is not proven — nothing here
isolates funnelling from any other consequence of 50 iterations of self-play.

---

## 5. Training health

| | |
|---|---|
| iterations | 50 (all committed; 0 lost to the 19:16 crash — `--resume` cost one iteration's selfplay) |
| loss | 4.224 at iter 1 → **min 3.241 at iter 37** → 3.308 at iter 50 |
| non-finite gradient steps | **4** in 20,000 (iters 27, 30, 36, 48) — 0.02%, against an abort rule of >20 *and* >1% |
| grad_norm, top 5 | 35.96 (it 25), 34.59 (it 28), 30.56 (it 6), 30.28 (it 17), 28.43 (it 8) |
| grad_norm at stop | 27.41 (it 50) — 6th highest, not the maximum |
| throughput | ~1,530–1,595 sims/s, ~29 min/iteration |

Two instabilities, at roughly iterations 36–41 and again at 48–50, each pairing an
entropy collapse-and-rebound with elevated gradient norms. The second was still in
progress at stop. Both are *within* the run's historical gradient range — the
excursions are not unprecedented in magnitude, only in their coincidence with the
entropy swings, and iteration 25's 35.96 predates the whole story.

---

## 6. Saturation — what these numbers cannot say

**The anchor rung is quantized at the top.** Elo is a deterministic function of
match score, so near-perfect scores land on a coarse lattice: 193/200 → +576.2,
198/200 → +798.2, 199/200 → +919.5, 200/200 → clamped at +1040.4. One game is
worth ~120 Elo up there. The iteration-20 and iteration-50 rows are *identical*
(+919.5, same CI) because both are 199W/0D/1L, and the "recovery" from iteration
40 is a six-game difference. **The rung cannot resolve anything finer, and three
of the five passes have a clamped upper CI bound.** Any statement of the form
"the model improved between passes X and Y" is, at this end of the ladder, a
statement about a handful of games.

**ECE saturates with it.** Calibration is measured on 5,000 positions drawn from
the phase1 anchor matchup. As the learner comes to dominate that matchup, the
value distribution collapses toward the extremes — at iteration 20, bins 0 and 9
held 2,350 and 2,302 of the 5,000 positions between them. ECE over a degenerate
distribution is easy to pass and says little; iteration 20's exceptional 0.0034
is best read as an artefact of that collapse rather than as a calibration peak.

**Consequence for M4.** No measurement above roughly +900 Elo exists on this
ladder. The Stockfish rungs are not an enhancement; they are the only way any
further progress becomes visible at all. Stockfish is **not installed** on this
box (`apt` candidate 16-1build1) and is a prerequisite for p2_b's final pass.

---

## 7. S4 flip (`mcts.promo_prior_from_head`)

Measured at the 50-mark per protocol, on `ckpt_iter0050.pt`, mate-in-2 at 128 sims,
eval profile — both arms, so the comparison is controlled rather than against a
remembered number. `runs/p2_a/s4_flip_iter0050.json`.

| arm | mate-in-2 | |
|---|---|---|
| `promo_prior_from_head=false` | 166/200 = 0.8300 | control — reproduces the ladder row exactly |
| `promo_prior_from_head=true` | 165/200 = 0.8250 | |
| **delta** | **−0.0050** (−1 position) | fixed: none · broken: `mate2_054` |

Null to slightly negative: the flag fixes nothing and breaks one position. **The
flag stays off in p2_b** regardless — enabling it would be a second lever — and
the enable-decision moves to p2_c/M4.

---

## 8. Causes, ranked by evidence

1. **Prior coverage collapse driven by self-play funnelling.** Directly measured:
   coverage 0.890 is the larger of the two factors, 22 of 34 failures are COVERAGE,
   book entropy falls monotonically across every pass, and the two watchlist
   COVERAGE positions now value the key child at ~1.0 while still failing. This is
   the strongest-evidenced cause and it is the one p2_b acts on.
2. **Under-convicted value on key children.** Measured: conditional 0.933, median
   key-edge q̂ 0.795. Real, second in magnitude, and *sufficient on its own* to
   keep the gate below 0.95 even at perfect coverage — which is why it is not
   deferred.
3. **Training instability.** Two entropy/gradient excursions, four non-finite
   steps, loss reversing after iteration 37. Weakest of the three: the excursions
   are within the run's historical gradient range, the 48-vs-50 diagnostic shows
   suite performance riding through them, and no mechanism linking them to the
   gate has been demonstrated. Recorded as weather, watched in p2_b.

Not causes: search defects (0 BUG verdicts in 34 failures), promotion-prior
handling (§7), or the crash-and-resume (iteration 24 was regenerated cleanly).

---

## 9. Artefacts

| file | what |
|---|---|
| `ladder.jsonl` | the five protocol passes |
| `ladder_diagnostic.jsonl` | the 0048 diagnostic — **not** part of the verdict |
| `iterations.jsonl` | 50 rows |
| `diagnose_mate2_iter0050.txt` | coverage factor |
| `triage_mate2_iter0050.{json,txt}` | per-position verdicts and key-edge q̂ |
| `s4_flip_iter0050.json` | both S4 arms |
| `ckpt_iter0000..0050.pt` | keep-all snapshots — what made the backfill possible |

The backfill is the reason this log has five passes instead of two. The campaign
was relaunched after a crash without `--ladder-every`, which would have left
passes 30/40/50 permanently unmeasured had the snapshots been rotated. Keep-all
plus a deterministic ladder turned a protocol deviation into a footnote.

---

## Erratum (2026-10-21) — why the resume lost the ladder

From `REVIEW-2026-10-21-full-codebase.md`, finding C3/D7. The paragraph above
records the deviation as an operator omission. It was a design defect: the ladder
cadence lived only in the `--ladder-every` CLI flag, was written to none of
`config.yaml`, `provenance.json` or `state.json`, and so could not be rebuilt by
`--resume` — while the saved `config.yaml` carried an `eval.eval_every_n_iters: 10`
that nothing read. The committed config therefore states a cadence the code never
consulted, for this run and for p2_b/c/d (whose configs say 10 while the runs
used 5). The field is now the source of truth, `--ladder-every` overrides it into
the saved config, and a resumed run reads it back. The three backfilled rows carry
`produced: null` because the field postdates them; they are the protocol passes,
as §2 states.
