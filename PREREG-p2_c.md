# Pre-registration — p2_c: z/q value targets

**Written before launch. Sections below not edited after.**

**Amendment policy.** A rule may be added after launch *only* for a measurement
that has not yet produced any data, and only as a dated addendum recording the
run state at the time of writing. A rule may never be added, weakened, or
reworded for a measurement already taken. (Carried from `PREREG-p2_b.md`, where
this policy was written mid-run; here it is in place from day one.)

- **Lever:** `train.value_q_mse_weight: 0.5` — the only field differing from
  `runs/p2_b/config.yaml`
- **Rehearsal:** stays at 0.05. Standing configuration, not a second lever — it is
  the control condition p2_b established.
- **Warm start:** `runs/p2_b/ckpt_iter0020.pt`, fresh optimizer
- **League:** seeded from p2_b · **Length:** 20 iterations, `--ladder-every 5`
- **Rungs:** `random,greedy,phase1,stockfish:0` — four wide, inline, chosen while
  the argument can still be chosen (p2_b's rungs were fixed at launch and
  Stockfish had to be backfilled)
- **Judged at:** pass 20

## Why this lever, restated

The decision rule that named it did so for a reason the p2_b triage killed: the
residual was supposed to be a value head short of conviction, and it is not —
median key-edge q̂ is 0.9721. The surviving case is different and stronger.

Pure outcome-z credits a move by **what the game happened to produce** rather than
by what search saw. Between players this weak that is largely noise, and noise in
the value target is the standing suspect for both of p2_b's unexplained results:
the flat skill-0 slope, and the second-half sharpness tax (value loss 0.40 → 0.56
while entropy rose).

## The strongest argument against this lever, recorded going in

`mate2_165` and `mate2_166` hold q̂ ≈ 0.9999 on the correct child and play moves
that are **not forced wins at any depth ≤ 3**. That is a demonstrated regime where
value quality is provably *not* the binding constraint — conviction saturates
without converting — and it weakens confidence in this lever's magnitude, not only
in the counter-cases.

What it does not reach is the lever's actual domain. Those two positions are a
128-sim / m=16 *suite* allocation regime; the flat skill-0 slope lives in full
games — thousands of quiet decisions where target noise on weak-play outcomes
plausibly binds in ways suite conversion never tests.

**This is why P1 registers direction and separation only, magnitude is recorded
rather than registered, and "flat again" is a first-class BLOCKED branch rather
than a surprise.** If the lever moves strength while p2_d's proof-search finishes
the suite, that is a cleaner division of labour than the decision rule imagined.

## Registered expectations

**P1 (primary).** Skill-0 Elo at pass 20 with 95% CI **disjoint above** the
p2_b-endpoint row: score 0.025, −636.4 [−848.7, −520.9]. Direction and separation
are the bar. Magnitude is recorded, not registered. This is the first round in the
project's history judged on an absolute instrument.

**N1 (no-regress).** mate-in-2 ≥ 0.915 at pass 20; per-pass values informational.
Mate-in-3, prior top-16 coverage, and solved-given-considered reported at pass 20
for the record. The suite is on no-regress duty this round.

**W1 (lever watchlist).** `mate2_067` (q̂ 0.575) and `mate2_125` (q̂ 0.500) — the
two genuinely under-convicted positions in p2_b's residual. Conviction expected to
rise materially; solves expected to follow.

**W2 (counter-cases).** `mate2_165` and `mate2_166` — expected **not** to convert.
Their failure is visits, not conviction, and their cure is parked in `p2_d`. **If
they convert anyway, the mechanism story is wrong and the RUNLOG says so.**

**H (health).** Skipped steps < 10. Reported, not registered: entropy trajectory
(does the funnel treatment survive the new target?); second-half vs first-half
value loss (the sharpness-tax-relief signal, direction expected down); ECE on the
Stockfish sample each pass — now the un-saturated half of that instrument, since
calibration samples both matchups from this round on.

Expected noise, pre-registered so it is not read as signal: a fresh-optimizer blip
through roughly iteration 8, as in p2_b.

## Decision rules

| Observation at pass 20 | Decision |
|---|---|
| **P1 passes and N1 holds** | p2_c closes as the first absolute-strength gain. The `p2_d` conversation opens on proof-directed reply expansion and the distributional-Q upgrade, with M4 planned against the measured slope. |
| **P1 flat again** | **STOP. BLOCKED.** Two levers deep with zero absolute movement is a finding about the loop itself — sims budget, opponent quality, capacity (§12 exhaustion territory). It goes to Tom as one finding, gates untouched. |
| **N1 breaks** | Standard stop rules. |

**No escalation is pre-authorised this round. λ stays 0.5.**

## What would make this run uninterpretable

- `root_q` disagreeing in sign with `outcome` — the blend would train the value
  head against its own target on every position whose mover lost. Guarded by
  `tests/test_replay.py::test_root_q_shares_the_perspective_of_the_outcome_it_blends_with`.
- Rows without a search being blended against a fabricated 0.0. Guarded by
  `tests/test_train.py::test_value_blend_masks_rows_without_a_search`.
- Any config field other than `value_q_mse_weight` differing from p2_b's resolved
  config.
