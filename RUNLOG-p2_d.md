# RUNLOG — p2_d (the null lever and the breadth lever)

**Track 2:** `runs/p2_d`, 20 iterations, 2026-08-13 06:15 → 18:08, config
byte-identical to p2_c. Zero new levers.
**Track 1:** proof-directed reply expansion, built behind a flag, measured offline
on four frozen checkpoints, flag-on vs flag-off.
**Pre-registration:** `PREREG-p2_d.md` + Addendum 1 (convertibility) + Addendum 2
(the repetition design). Every verdict below was registered before it was measured.

**Verdicts: P2 PASSES. Track 1's prediction scores 13/13. The primary strength
instrument was found to randomize its own play, and every prior skill-0 CI on
record understates uncertainty.**

---

## 1. P2 — the flywheel continues with nothing added

Judged on the registered statistic: three offline replays per endpoint, per-opening
rep-means, paired bootstrap over openings on the mean-differences.

```
p2_c/0020 reps:  0.165, 0.230, 0.205      mean 0.2000   sigma 0.0328
p2_d/0020 reps:  0.330, 0.285, 0.295      mean 0.3033   sigma 0.0236

paired difference          +0.1033
95% CI                     [+0.0600, +0.1467]
SE                         0.0220        effect/SE  4.70
```

**The CI excludes zero. P2 stands.** Twenty iterations at an identical
configuration moved skill-0 score by +0.103, and the flywheel needs nothing added
to keep turning.

Two things this measurement corrects about the naive reading:

- **The single-row comparison understated the effect.** 0.233 → 0.315 is +0.082;
  the rep-means give +0.103. Both single rows happened to sit high in their own
  distributions, and the paired construction sees past that.
- **It also could not have supported the claim.** Marginal intervals on the two
  single rows overlap ([−260.5, −159.8] vs [−184.0, −86.9]). P2 as literally
  written — "CI disjoint above" on single rows — **fails**. It passes on the
  registered statistic, which is the test of record precisely because the naive
  one was known to be wrong before either was computed.

### The shape, which is now a repeated pattern

```
p2_d @5    0.180      @10  0.215      @15  0.193      @20  0.315
```

Three passes at or below the starting point, then the whole gain in the final five
iterations — the same shape p2_c showed (flat through pass 10, all the gain late).
Twice is not a law, but it is enough to stop reading a flat mid-run pass as a stall.

## 2. The instrument finding, which outranks the verdict it nearly broke

Re-running `p2_c/0020`'s skill-0 matchup at identical seed, config and book
returned **0.180** where the campaign recorded **0.233**. Cause, measured over five
fresh engine processes per position, both polarities:

```
quiet position, nodes=10000
  skill  0:  e3, e3, e3, cxd5, cxd5     -> 2 distinct   NONDETERMINISTIC
  skill 20:  e3, e3, e3, e3, e3         -> 1 distinct   deterministic
```

Stockfish's skill limiter randomizes its tie-break. **Skill 0 — this project's
primary strength instrument since `RUNLOG-p2_b.md` §4 — does not reproduce.**

Measured spread on one checkpoint across four measurements (0.233, 0.180, 0.165,
0.230): **σ ≈ 0.033**, comparable to the entire effect being measured.

**Consequences, recorded as scope corrections:**

- **Every skill-0 CI on record understates uncertainty.** The paired bootstrap
  resamples openings *within* a run; it cannot see an engine that re-randomizes
  *between* runs. This applies to every skill-0 row in `RUNLOG-p2_b.md` and
  `RUNLOG-p2_c.md`.
- **p2_c's P1 verdict survives comfortably** — 0.025 → 0.233 is ~6σ of this noise —
  and is re-affirmed with that margin stated.
- **The reproducibility claim that licensed p2_a's backfill, the 0048 diagnostic
  and the offline Stockfish rungs holds for net-vs-net rungs and not for this
  one.** The claim was proven for the rungs that existed when it was written and
  silently extended to a rung family that arrived later carrying a parameter.

**The law, now carved into a construction gate:** *a property proven for a set must
be re-proven when the set grows.* Third instance — promotion-free fixtures, the
`stockfish_s0` dedup key, and this. Every future opponent family gets the
five-process determinism probe before it is trusted.

Landed: `nondeterministic` on every ladder row so a row announces its own
epistemic status; `Threads=1`/`Hash=16` pinned; the probe as a tested gate. **A
deterministic skill-20 rung is parked, not adopted** — M4 is written against skill
0 and changing opponents mid-campaign is how continuity dies. **M4's clause is
restated as mean-over-reps with honest CIs**, the fourth restatement after
monotone, E6 and ECE, for the same reason each time.

## 3. Track 1 — the prediction scored 13/13, including the forbidding row

Scored **strictly at `p2_c/0020`**, where Addendum 1 registered it:

```
predicted CONVERT (4):  054, 091, 165, 196
actually converted:     054, 091, 165, 196
predicted NO (9):       009, 087, 126, 127, 128, 129, 139, 166, 186
converted anyway:       none

hits 4/4 · misses 0 · false positives 0 · correct negatives 9/9
```

The rule had **zero free parameters**: convert iff q̂ ≥ 0.95 **and** flag-off
allowance ≥ proof cost, both read from `triage_mate2`.

**`mate2_166` — allowance 6 against cost 7, q̂ 0.9988, one visit structurally
short — did not convert.** A prediction that forbids is worth more than ten that
permit, and this is the one that could most easily have died. The arithmetic is
exact at the boundary, not merely directional.

### The protocol

| checkpoint | gate | coverage | conditional | converted / broken |
|---|---|---|---|---|
| p2_c/0000 | 0.930 → 0.945 | 0.985 → 0.985 | 0.944 → 0.959 | +3 / 0 |
| p2_c/0010 | 0.905 → 0.940 | 0.990 → 0.990 | 0.914 → 0.950 | +7 / 0 |
| p2_c/0020 | 0.930 → 0.945 | 0.995 → 0.995 | 0.935 → 0.950 | +4 / 1 |
| p2_d/0020 | 0.915 → 0.940 | 0.985 → 0.985 | 0.929 → 0.954 | +6 / 1 |

**Coverage is identical in all four pairs.** The lever's theory says it operates
below the root, on expansion order within a candidate's subtree, and therefore
cannot touch consideration — the instrument confirms it changes exactly what it
claims and nothing else. The identity closes again in passing: 0.990 × 0.950 =
0.9405 against a measured 0.940.

**The conditional clears 0.95 at all four checkpoints**, meeting the registered
gate expectation. The gate itself lands at 0.940–0.945, short of clearing 0.95
outright: the compounding effect predicted for `SLOWER_FORCED_WIN` cases shows up
in the conditional but does not quite carry the gate.

**Unregistered corroboration**, reported as such and *not* scored as hits:
`008`, `146`, `197` at 0010; `035`, `127`, `128` at p2_d/0020. Different
checkpoints, different residuals, outside the registration scope. Counting them
would convert a forbidding prediction into a curve fit after the fact.

### The unpredicted cost

**`mate2_124` broke flag-on at p2_c/0020, and `mate2_138` at p2_d/0020** — solved
flag-off, failed flag-on. The prediction covered conversions and said nothing about
breakage, and the clean "+3/0, +7/0" at the earlier checkpoints hid it.

This is the open question Track 1 closes with. A search change that trades one
position for four is a net gain on this suite, but "net gain" is not the standard a
protocol instrument has to meet, and the mechanism by which breadth *loses* a
position is not yet understood. **Succession is therefore not taken:** flag-on
remains a parallel series, and which search the protocol ladder runs stays open,
exactly as registered.

### The residual with search out of the way

Nine of the thirteen registered positions do not convert, and the arithmetic says
why: six fail the q̂ threshold outright, three clear it but are short on allowance.
That is the first clean estimate of **genuine capability misses** this project has
— failures that survive with search doing everything the mechanism allows — and it
is `p2_e`'s intake, with `mate2_166` as the named boundary case one visit from the
line.

## 4. Track 2's dynamics, which the noise cannot touch

These are patterns across five passes, not single margins, and they stand
regardless of where any endpoint verdict falls.

**The funnel collapsed with nothing touched.** Book entropy 1.313 → 1.482 → 1.346 →
1.381 across p2_d's passes, against p2_b's 2.104 and p2_c's 1.575. Per-iteration
`H_prior` ended at **0.011**, from 0.152 at iteration 1 — two orders below p2_b's
endpoint. p2_a's collapse regime, reached under a configuration where *nothing was
changed*. The collapse is not caused by a lever; it is what this loop does when
left alone.

**Instability did not return.** Zero non-finite steps in 8,000; max gradient 20.18.
Your `p2_e` joint condition — collapse **and** instability — is **not met**. Noted;
nothing changed mid-run.

**The sharpness account stays open.** Loss 1.992 → 2.357, value loss 0.320 → 0.353
first half to second. Registered without a direction this round, after being
registered and wrong twice.

**N and ECE.** mate-in-2 ended at **0.915** — exactly the floor, having fallen
0.945 → 0.935 → 0.915 across the second half. **N holds, barely.** ECE **0.1762**
against the restated `< 0.15` clause: **ECE FAILS**, on the honest sampler, and the
restatement was made before the measurement.

`decisive_rate` fell 0.854 → 0.709: more games reaching the ply cap.

## 5. Decision

**P2 passes.** Per the registered rule, the flywheel needs nothing, and M4 planning
proceeds on the measured slope — with the slope now measured on an instrument whose
noise is quantified rather than assumed.

Open for `p2_e`, in priority order:

1. **The intrinsic funnel collapse**, which is now the strongest unexplained
   dynamic in the campaign and happens with no lever at all.
2. **The two positions the breadth lever breaks**, which gates its succession.
3. **The nine-position genuine-capability residual**, with search out of the way.
4. **ECE**, failing its restated clause on the honest sampler.

## 6. Artefacts

| file | what |
|---|---|
| `runs/p2_d/ladder.jsonl` | 4 passes × 4 rungs, `pair_scores` persisted |
| `runs/p2_{c,d}/reps_iter0020_r{1,2,3}.jsonl` | the repetition measurement |
| `runs/p2_d/p2_verdict_reps.json` | the paired bootstrap and its verdict |
| `runs/p2_{c,d}/track1_ckpt_iter00*.json` | flag-on vs flag-off, per-position solve sets |
| `runs/p2_c/ladder_pairscores.jsonl` | the replay that found the nondeterminism |

---

## Errata (2026-10-21)

From `REVIEW-2026-10-21-full-codebase.md`. Nothing above is edited.

1. **Row provenance (finding C2/D2).** Every row in `runs/p2_d/iterations.jsonl`
   and `runs/p2_d/ladder.jsonl` is stamped `git_sha: 4710690`. The process was
   launched at 06:15 from `72e7f1a` (`provenance.json`); `4710690` was committed at
   06:39, after launch, and differs from the running code in `config.py` and
   `mcts/search.py` (it adds the Track 1 flag). The stamp was `git rev-parse HEAD`
   *at write time*, not the code that produced the row. **Read `provenance.json`'s
   SHA as the run's code**, for this and every campaign. The writers now capture
   the SHA once at launch.
2. **`nondeterministic` was not "on every ladder row" (finding P1).** §2 says the
   field landed; it landed in `91293f7` at 19:30, after this run's last row was
   written at 18:08. The 16 match rows of `runs/p2_d/ladder.jsonl` and the row in
   `runs/p2_c/ladder_pairscores.jsonl` were therefore written as schema 3 without
   it and failed the schema-3 validator. On 2026-10-21 the field was **backfilled**
   onto those 17 rows — `true` for `stockfish_s0`, `false` for `random`, `greedy`
   and `phase1`, the same rule the writer applies — with every other byte of each
   row unchanged. The rep rows (`reps_iter0020_r*.jsonl`) were written after
   `91293f7` and carry it natively.
3. **The harness was not in the repository (finding P5).** The repetition
   measurement in §1, the verdict in `p2_verdict_reps.json`, and the flag-on /
   flag-off solve sets in §3 were produced by code that was never committed. They
   are now `scripts/eval_reps.py` (replays a checkpoint's rung over N seeds and
   computes the paired bootstrap over per-opening rep-means) and
   `scripts/eval_track1.py` (flag-off vs flag-on solve sets per checkpoint). The
   `verdict` subcommand reproduces `p2_verdict_reps.json` from the committed rep
   rows to the digit; the generation side cannot be re-run bit-identically because
   the rung is nondeterministic, which is §2's point.
4. **The lever fires for either mover (finding P6).** Track 1's description —
   "at an opponent node whose parent edge is already believed" — understates the
   mechanism: the breadth sweep triggers beneath *any* believed edge, including
   beneath an opponent reply believed from the opponent's side, where it sweeps the
   root mover's own replies in a line the root mover is losing. The 13/13 result in
   §3 was measured with that symmetric behaviour and stands as measured; the code
   is unchanged and its documentation now says what it does. Whether to restrict
   the trigger to opponent nodes, and re-measure, is a `p2_e` decision.
