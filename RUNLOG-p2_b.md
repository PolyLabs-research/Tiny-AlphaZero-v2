# RUNLOG — p2_b (anchor rehearsal)

**Run:** `runs/p2_b` · 20 iterations · 2026-08-11 15:15 → 23:42 · git `8a19cbf`
**Lever:** `train.rehearsal_frac 0.05` (mate1 0.02 / mate2 0.03), everything else
inherited from p2_a verbatim
**Warm start:** `runs/p2_a/ckpt_iter0050.pt`, fresh optimizer · **League:** seeded
from p2_a's last 20 snapshots · **Pre-registration:** `PREREG-p2_b.md`

**Verdict: partial recovery.** The lever did substantial, measurable, mechanistically
confirmed work and did not finish the job. p2_b does not close as the recovery proof.

---

## 1. The scorecard

| | expectation | measured | |
|---|---|---|---|
| E1 | mate-in-2 ≥ 0.95 at pass 15 **and** 20 | 0.945, then 0.930 | **FAIL** |
| E2 | prior top-16 coverage ≥ 0.98 at pass 20 | **0.985** | **PASS** |
| E3 | solved-given-considered ≥ 0.95 at pass 20 | 0.944 | **FAIL** |
| E4 | `mate2_165` and `mate2_166` both solved at 20 | neither, at any pass | **FAIL** |
| E5 | mate-in-3 ≥ 0.65 | 0.625 | FAIL (informational) |
| E6 | phase1-anchor lower CI ≥ +727 at pass 20 | +636.4 | **FAIL as written** — see §5 |
| E7 | no new non-finite-gradient cluster | **0 in 8,000 steps** | **PASS** |

Both failures that carry weight are narrow. E1 missed by one position at pass 15
(189/200 where 190 was needed) and by four at pass 20. E3 missed by one position
(186/197 where 187 was needed). Neither is a rounding artefact and neither is a
collapse.

**Decision rule fired:** *"Coverage fixed, conditional binding (E2 met, E3 < 0.95)
→ the conditional conversation: §12.3's z → z/q value-target mix becomes p2_c's
lever, with the CAPABILITY dozen as its watchlist."* Read mechanically. §7 records
evidence that complicates the lever named in that rule.

---

## 2. The ladder

| pass | mate-in-2 | mate-in-3 | phase1 anchor | Elo | ECE | book H |
|---|---|---|---|---|---|---|
| 5 | 0.875 | 0.625 | 199W/0D/1L | +919.5 | 0.0323 | 1.7093 |
| 10 | 0.915 | 0.615 | 198W/0D/2L | +798.2 | 0.0288 | 1.9816 |
| 15 | **0.945** | 0.625 | 198W/0D/2L | +798.2 | 0.0357 | 2.0671 |
| 20 | 0.930 | 0.625 | 198W/0D/2L | +798.2 | 0.0423 | 2.1043 |

Against p2_a's endpoint of 0.830, and p2_a's best-ever 0.895: every p2_b pass from
10 onward is outside the band p2_a reached unassisted.

The 0.945 → 0.930 step is three positions on a 200-position suite. These are
deterministic checkpoints, so it is a real difference between two models, but it
is well inside binomial scale (~0.8 SE) — "peaked and slipped" is the right size
of claim.

---

## 3. The decomposition, and the migration

|  | p2_a @50 | p2_b @10 | p2_b @20 |
|---|---|---|---|
| prior coverage @top-16 | 0.890 | 0.985 | **0.985** |
| solved \| considered | 0.933 | 0.929 | **0.944** |
| product | 0.830 | 0.915 | 0.930 |
| **measured gate** | **0.830** | **0.915** | **0.930** |
| prior top-1 coverage | 0.250 | 0.625 | 0.555 |
| median forcing-move rank | 2 | 0 | 0 |

The identity closes to the digit for the fourth consecutive measurement:
0.985 × 0.944 = 0.930 = the gate.

**Coverage was fixed by iteration 10 and held flat for the next ten.** The whole
coverage gain — 0.890 → 0.985, top-1 more than doubling, median forcing-move rank
falling to 0 — happened in the first half at 5% rehearsal, and the second half
added none of it. The conditional moved late instead: 0.929 at pass 10, 0.944 at
pass 20.

Triage of the failures (`triage_mate2_iter0020.json`): **14 failures, down from
p2_a's 34.** 11 CAPABILITY, 3 COVERAGE, **0 BUG**.

The watchlist crossed over, which is the finding stated in three positions:

| id | p2_a verdict | p2_b @20 verdict | prior rank @20 | in top-16 |
|---|---|---|---|---|
| mate2_165 | COVERAGE | **CAPABILITY** | 8 | YES |
| mate2_166 | COVERAGE | **CAPABILITY** | 13 | YES |
| mate2_129 | CAPABILITY | **COVERAGE** | 26 | no |

165 and 166 were outside the considered set at the 2026-08-09 pin and are inside
it now — their failure migrated from coverage to conversion. 129 went the other
way. Neither is solved. The head that forgot has remembered; the head that
hesitates still hesitates.

---

## 4. The absolute reading — Stockfish skill 0

The first measurement in this project's history against an opponent that is not
derived from its own weights. `role: diagnostic`, `stockfish:0` at 10,000 nodes,
200 games, seed 42.

| checkpoint | W/D/L | score | Elo | CI |
|---|---|---|---|---|
| iter 0 (= p2_a's endpoint) | 3/6/191 | 0.030 | −603.9 | [−759.0, −502.6] |
| iter 10 | 2/2/196 | 0.015 | −726.9 | [−1040.4, −589.5] |
| iter 20 | 4/2/194 | 0.025 | −636.4 | [−848.7, −520.9] |

**Twenty iterations produced no measurable strength gain.** 0.030 → 0.015 → 0.025,
all three CIs overlapping heavily, the endpoint marginally below the start. The
slope that was to be "the planning number for everything after" is flat.

This reframes without invalidating. The +919 over the Phase 1 anchor was real
*relative* learning against a random-play bootstrap, and every mechanism finding
in §3 stands on its own measurements. But the anchor ladder's saturation had been
concealing the scale entirely: M4's criterion — beat skill 0 with CI clear of 50% —
is not the next milestone, it is a mountain that has just become visible from a
starting point of 0.025.

The rung-climb registered in `PREREG-p2_b.md` Addendum 1 was armed for saturation
*upward* and met an unclamped reading *downward*. It correctly did nothing.
Instrument-range selection worked by not being needed.

### Instrument succession

Recorded here as a standing change, not a one-off note:

- **The Stockfish skill-0 rung is the primary strength instrument from now on.**
  E6-class criteria — "did strength collapse", "did strength improve" — bind to it.
- **The phase1-anchor rung retires to no-regress duty.** It quantizes at ~120 Elo
  per game at the top and has been pinned at 198W/0D/2L for three consecutive
  passes; it can register a collapse and nothing finer.
- **E6's literal FAIL is recorded with that context.** Lower CI +636.4 against a
  +727 floor is two games on an instrument that cannot resolve two games, and the
  point estimate never moved across passes 10, 15 and 20. It is recorded as FAIL
  because that is what the pre-registration says, and annotated as uninformative
  for the same reason erratum 7 retired "monotone".

---

## 5. The tension: one lever, two effects, opposite signs

Book move-1 entropy across every ladder pass of both campaigns:

```
p2_a   2.796 → 2.635 → 2.507 → 2.228 → 1.905     (monotone down, 5 passes)
p2_b   1.709 → 1.982 → 2.067 → 2.104             (monotone up, 4 passes)
```

The one series that never bent across nine passes and two campaigns reversed
completely under rehearsal, and per-iteration `H_prior` agrees — 0.388 at
iteration 10 rising to 1.052 at iteration 20. **Rehearsal is treating the funnel**,
apparently as a side effect: diverse anchor positions acting as a representational
diversity anchor. This was explicitly *not* pre-registered (`PREREG-p2_b.md`:
"Book entropy: reported, not predicted"), which is what makes it worth something.

The same second half that produced the entropy gain paid for it:

| | iteration 10 | iteration 20 |
|---|---|---|
| loss | 3.112 | 3.435 |
| value loss | 0.402 | 0.561 |
| mate-in-2 | 0.915 | 0.930 (via 0.945) |
| coverage | 0.985 | 0.985 |

Rehearsal is simultaneously the diversity treatment and a sharpness tax. Both
effects are real, they run in opposite directions, and one lever produces both.

---

## 6. Training health

Zero non-finite gradient steps in 8,000, against p2_a's four in 20,000. The
fresh-optimizer blip discharged on the predicted schedule: gradient norms of 35.4
and 35.7 at iterations 2 and 4, settled by iteration 6. E7 passes outright.

One annotated perturbation: `annotations.jsonl` records a 123-second GPU-contention
window inside iteration 13's selfplay, its cause, and why the exception was granted.

---

## 7. What the evidence says about p2_c's lever

The decision rule names §12.3's z → z/q value-target mix. The pass-20 triage
complicates that, and the complication is recorded here rather than resolved:

**The residual CAPABILITY failures are not short of value conviction.** Median
key-edge q̂ across the 11 is **0.9721** — up from 0.795 in p2_a — with 4 above
0.99. What they are short of is *visits*:

| id | key-edge visits | replies expanded | sims a proof needs | q̂ |
|---|---|---|---|---|
| mate2_009 | 2 | 1/2 | 3 | 0.9505 |
| mate2_091 | 14 | 4/8 | 9 | 0.9999 |
| mate2_124 | 14 | 7/16 | 17 | 0.9526 |
| mate2_127 | 6 | 5/13 | 14 | 0.9669 |
| mate2_165 | 6 | 3/6 | 7 | 0.9999 |
| mate2_166 | 2 | 1/6 | 7 | 0.9998 |
| mate2_195 | 6 | 4/6 | 7 | 1.0000 |

Every row reads "reply set never fully expanded", and in most the key edge is
within a few visits of the arithmetic threshold triage documents: proving a
mate-in-2 costs `1 + len(replies)` simulations on the key edge. `mate2_165` needs
7 and gets 6.

A value head that scores the key child at 0.9999 and still loses the acting argmax
is not obviously a value-target problem. It may be a search-allocation problem at
128 sims with m=16 — which is an eval-profile question, and the eval profile is
frozen. Two exceptions to note: `mate2_067` (q̂ 0.575) and `mate2_125` (q̂ 0.500)
are genuinely under-convicted, so a value lever has *some* real target, just a
much smaller one than the decision rule assumed when it was written.

This is not a recommendation to change the pre-registered branch. It is the
evidence a p2_c design has to answer before spending a run on z/q.

---

## 7b. Semantic residual — what the 14 failures actually are

The gate scores one thing: did the chosen move force mate *in 2*. A move that
forces mate in 3 scores identically to one that throws the win away, and the two
say opposite things about the value head. Since p2_c's lever is aimed at value
quality, the residual it aims at has to be described before it is spent on.

Per failing position, the eval-profile search was replayed, the move it actually
played taken, and that move tested for a forced mate at depth 3
(`semantic_residual_ckpt_iter0020.json`):

| id | verdict | played | q̂ | classification |
|---|---|---|---|---|
| mate2_054 | CAPABILITY | Bxc2 | 0.9850 | **SLOWER_FORCED_WIN** |
| mate2_091 | CAPABILITY | Qe6+ | 0.9999 | **SLOWER_FORCED_WIN** |
| mate2_195 | CAPABILITY | f4+ | 1.0000 | **SLOWER_FORCED_WIN** |
| mate2_126 | COVERAGE | Nxe8 | — | **SLOWER_FORCED_WIN** |
| mate2_009 | CAPABILITY | Qd5 | 0.9505 | GENUINE_MISS |
| mate2_067 | CAPABILITY | fxe2 | 0.5754 | GENUINE_MISS |
| mate2_124 | CAPABILITY | f5 | 0.9526 | GENUINE_MISS |
| mate2_125 | CAPABILITY | Ke1 | 0.4997 | GENUINE_MISS |
| mate2_127 | CAPABILITY | f5 | 0.9669 | GENUINE_MISS |
| mate2_128 | CAPABILITY | f5 | 0.9721 | GENUINE_MISS |
| mate2_165 | CAPABILITY | Rd7+ | 0.9999 | GENUINE_MISS |
| mate2_166 | CAPABILITY | Bxg3 | 0.9998 | GENUINE_MISS |
| mate2_129 | COVERAGE | Nd5 | — | GENUINE_MISS |
| mate2_084 | COVERAGE | Be7 | — | GENUINE_MISS |

**4 of 14 are forced wins one move slower; 10 are genuine misses.** The gate does
not move — 0.930 stands — but its residual is 10 positions the model gets wrong,
not 14.

**Mechanism.** Estimated wins carry no mate-distance ordering; only *proven* ones
do. A search that never expands the key move's full reply set cannot know that
one winning line is shorter than another, so among high-q̂ failures where it lands
is close to arbitrary — three landed on slower forced wins, eight on genuine
misses. That is the signature of missing proof, not of missing conviction.

It also sharpens §7's caution rather than softening it. `mate2_165` and
`mate2_166` hold q̂ ≈ 0.9999 on the key child and play moves that are not forced
wins *at any depth ≤ 3*. Near-certain conviction on the right child is not
converting into even a slow win, which is hard to attribute to the value target.

**`p2_d` lever parked, with this table as its motivating evidence:**
proof-directed reply expansion — spending the key edge's visits on completing its
reply set rather than on breadth — is what turns an estimated win into a proven,
distance-ordered one.

---

## 8. Two defects, found because a Stockfish rung ran for the first time

Both in the path M4's criterion depends on. Both had been latent since chunk 10.

**`close()` shipped with no caller.** `StockfishOpponent.close()` existed and
nothing anywhere called it. `chess.engine`'s transport thread is non-daemon, so
the first Stockfish rung played its 200 games, wrote a correct row at 23:47, and
then hung at interpreter exit — still hung seven hours later, with checkpoints
0010 and 0020 never started. Fixed in `1231b65` with a `finally`, tested through
`run_ladder_pass` rather than on `close()`, because the bug was in nobody calling
it. *Structural cure for the class, not yet applied:* make opponents context
managers so closure is an idiom rather than a thing to remember.

**Resumability was keyed on the spec, not the anchor's name.** Rows record
`result.anchor` — the opponent's *name* — while the skip set was keyed on the CLI
*spec*. Identical strings for random, greedy and phase1, so resumability worked
everywhere it had ever been exercised, and `"stockfish:0"` never matched the logged
`"stockfish_s0"`. A re-run recomputed a five-minute matchup and would have appended
a duplicate row; caught by watching the restarted chain sit on a checkpoint whose
row was already in the log, and killed before the duplicate landed. Fixed in
`5bb9216`. *Structural cure for the class, not yet applied:* writer and skip-set
should derive identity from one shared normalizer, never two spellings of the same
intent.

A third, recorded and unfixed: **the ladder's watchlist `verdict` field is pinned,
not live.** `watchlist_report` copies `entry["verdict"]` from the 2026-08-09 file,
so every ladder row since chunk 10 prints a static COVERAGE label beside live
numbers — and it did steer a reading in this session before being caught. The cure
belongs with the same family as `role` and `calibration_note`: a pinned field
carries its provenance, `verdict_asof: 2026-08-09`.

---

## 9. Artefacts

| file | what |
|---|---|
| `ladder.jsonl` | 4 protocol passes + 3 diagnostic Stockfish rows (`role` distinguishes them) |
| `iterations.jsonl` | 20 rows |
| `annotations.jsonl` | the contention window in iteration 13, with its cause |
| `diagnose_mate2_iter{0010,0020}.txt` | coverage decompositions |
| `triage_mate2_iter0020.json` | 14 failures, per-position verdicts and q̂ |
| `watchlist_coverage_ckpt_iter{0010,0020}.json` | the three by name, live prior ranks |

---

## Erratum (2026-10-21) — row provenance

From `REVIEW-2026-10-21-full-codebase.md`, finding C2/D2. Nothing above is edited.

The header's `git 8a19cbf` is correct: it is the launch SHA in `provenance.json`
and the code this run executed. The rows disagree with it because `git_sha` was
read from `git rev-parse HEAD` *at write time* and commits landed in the working
tree while the campaign ran (15:15 → 23:42):

| rows | stamped | committed | note |
|---|---|---|---|
| `iterations.jsonl` 1–5, ladder pass 5 `random` | `8a19cbf` | 13:03 | the launch code |
| `iterations.jsonl` 6–10, pass 5 `greedy`/`phase1`/eval, pass 10 | `cb6f634` | 15:33 | a PREREG addendum; no code change |
| `iterations.jsonl` 11–12 | `9837809` | 18:40 | **introduces ladder schema 2** |
| `iterations.jsonl` 13–20, passes 15 and 20 | `62d661b` | 19:29 | |
| `stockfish_s0` rows (backfilled offline) | `62d661b`, `5bb9216` | later | genuinely produced by that code |

The iteration rows stamped `9837809` and `62d661b` carry `schema_version: 1`
although those commits write version 2 — the stamped code could not have produced
them, which is the proof that the stamp is a clock reading and not provenance.
**Read `provenance.json` as this run's code.** The writers now capture the SHA
once at launch and stamp every row and checkpoint from it.
