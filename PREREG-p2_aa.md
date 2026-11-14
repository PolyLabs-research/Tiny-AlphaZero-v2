# Pre-registration — p2_aa: does the diagnostic's D4 gain replicate?

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: DRAFT 2026-11-14, NOT signed.** Nothing under this file has run. (p2_z is the benchmark track's
> clock-scaled gauntlet, registered next in RUNLOG-p2_y and not scheduled; this is the training track's next round.)

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

DIAG-2026-11-13 (capacity or data) trained the standing endpoint on teacher rows only, at four nested set sizes.
Its registered reading is *not resolved (compute)*. Its strength aside, which decided nothing, gave the first gain
over a start since p2_t: the **D4** checkpoint (4,000,000 teacher rows, 16M samples, so about four passes per
row) scored **0.416 against the start's 0.379 at skill 5, +0.037 [+0.005, +0.069]** over six reps each. The D1
checkpoint (the 475,696-row ring, about thirty passes) fell to 0.113. Read together with p2_v (twice the passes
over the same ring cost skill), the hypothesis is that teacher positions help when there are many of them, each
seen a few times, and hurt when a small set is drilled.

One run, one checkpoint, six reps, from an aside that decided nothing, is not a result. Michael's rule for this
phase is one change at a time, each measured. **So this round changes nothing: it runs D4's protocol again with
a different batch order and measures both D4 and the new run at twice the reps.**

**Does a second run of the D4 protocol beat the start at skill 5, and does D4's own gain survive six more reps?**

## Contract

| | |
|---|---|
| Start | `runs/p2_w_quarter_down/ckpt_iter0010.pt` (sha256 `43494548…3857`), as the DIAG |
| Line R (the replication) | `scripts/distill_curve.py --start <start> --d1 runs/p2_u_fresh/teacher --new runs/diag_cod/teacher --holdout runs/p2_u_fresh/teacher_holdout --sizes 4000000 --seed 1 --out runs/p2_aa/r --device cuda`; everything else the DIAG's defaults (lr 2e-5 constant, batch 1,024, AdamW with the start's weight decay, teacher rows only with the Phase 2 teacher-row loss, held-out measured every 250,000 samples, patience 4, cap 16,000,000), and its result, as D4's was, the checkpoint best by held-out cross-entropy (`runs/p2_aa/r/size_4000000/best.pt`). **The one difference from D4 is `--seed 1` (D4: 0)**, which changes the batch order and the 5,000-row training probe; the rows, the start and the code are D4's |
| D4 | `runs/diag_cod/curve/size_4000000/best.pt` (sha256 `7f827f08…`), as the DIAG left it |
| Instrument | `scripts/eval_reps.py run --anchor stockfish:5 --config runs/p2_w_quarter_down/config.yaml` (the start's evaluation profile and seed for all three, so only the weights differ), the 100-opening book, two games an opening. **Twelve reps for each of start, D4 and R**: the start's six from `runs/p2_w_quarter_down/sf5/reps_iter0010_r{1..6}.jsonl` plus six new ones, D4's six from the DIAG plus six new ones, R's twelve new. The Stockfish skill limiter randomises across processes, so new reps are fresh draws. Verdicts: `eval_reps.py verdict` (per-opening rep-means, paired bootstrap over the 100 openings, 10,000 resamples, seed 0), read with the exact-zero rule |
| Where | this box, under `gpu hold` with the ladder's environment (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`), as the transient user unit `taz-p2aa`. No pod, no spend |
| Duration | R's training about 2.5 h (D4 took 2 h 32 min); 24 new reps at about 8 min each, about 3.2 h; the suite rows a few minutes. **About 6 h** |

## Registered expectations

### T — the training replicated
R's best held-out cross-entropy is reported beside D4's 2.2048. **T holds if they are within 0.01 nats.** If T
fails, R did not train as D4 did, and K1 is read as a comparison of two different trainings.

### K1 (primary) — R against the start, skill 5, twelve reps each
`p2aa_verdict_r_vs_start.json`. The direct replication.

### K0 — D4 against the start, skill 5, twelve reps each
`p2aa_verdict_d4_vs_start.json`. The DIAG's aside with six more reps on each side: does it hold up?

### K2 — mate-in-2
At R's checkpoint, from `scripts/eval_ladder.py`'s suite row (128 sims, the start's config). **K2 passes if R
solves at least 183 of 200** (p2_w's floor), and is reported beside the start's 188.

### Reported, no decision
R − D4 at skill 5 (twelve reps each; how far two runs of the protocol land apart, as p2_w's K0); R's value MSE
against the teacher and its top-1, beside D4's (0.034, 0.383); R's held-out curve beside D4's.

## Predictions (written before launch)

- T holds (R's held-out CE 2.20–2.21).
- **K1: point +0.01 to +0.05; the interval excludes zero, more likely than not** (D4's +0.037 had a standard
  error of 0.0165 at six reps; at twelve the error falls to roughly 0.012, but a second draw regresses toward the
  truth, which is likely below the first reading).
- K0: point +0.015 to +0.04 (the first six reps were a draw, and the six added regress it toward the truth); the
  interval excludes zero, about even odds.
- K2 passes (R 186–193).
- R − D4 within ±0.04.

## Decision rules

| K1 (R − start) | K0 (D4 − start) | T | Decision |
|---|---|---|---|
| **entirely above zero** | any | holds | **Replicated: training on many teacher positions, each seen a few times, raises skill over the standing endpoint.** The protocol becomes the next lever; the round after (p2_ab) changes **one** thing about it, Michael's choice: more labelled positions (8M, about $6 of labelling), a 2e-6 step-down after it, or self-play rows mixed back in. If K2 fails, the mate-in-2 cost is diagnosed first |
| includes zero | entirely above zero | holds | **D4 holds, a second run does not show it at this resolution.** Two runs of one protocol may differ by as much as the effect: R − D4 is read; a third run (seed 2) is Michael's call before the protocol is adopted |
| includes zero | includes zero | holds | **Not replicated.** D4's +0.037 was within the instrument's noise; the protocol is not adopted. The DIAG's curve still says the cap, not the model, set the top of its range; Michael's choice from the open list |
| entirely below zero | any | any | **Contradicts the aside.** Reported, diagnosed (value MSE, the probe gap) before anything is built on it |
| any | any | **fails** | **R did not train as D4 did**: the training is diagnosed first; K1 is reported, not read as a replication |

## Measurement protocol

```bash
# runs/p2_aa_run.sh as the transient user unit taz-p2aa, from main at the registration commit, under gpu hold:
.venv/bin/python scripts/distill_curve.py --start runs/p2_w_quarter_down/ckpt_iter0010.pt \
    --d1 runs/p2_u_fresh/teacher --new runs/diag_cod/teacher --holdout runs/p2_u_fresh/teacher_holdout \
    --sizes 4000000 --seed 1 --out runs/p2_aa/r --device cuda
CFG=runs/p2_w_quarter_down/config.yaml
.venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_aa/r/size_4000000/best.pt --config $CFG \
    --anchor stockfish:5 --reps 12 --out-dir runs/p2_aa/r_sf5
.venv/bin/python scripts/eval_reps.py run --ckpt runs/diag_cod/curve/size_4000000/best.pt --config $CFG \
    --anchor stockfish:5 --reps 6 --out-dir runs/p2_aa/d4_sf5_more
.venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_w_quarter_down/ckpt_iter0010.pt --config $CFG \
    --anchor stockfish:5 --reps 6 --out-dir runs/p2_aa/start_sf5_more
.venv/bin/python scripts/eval_ladder.py --ckpt runs/p2_aa/r/size_4000000/best.pt --config $CFG \
    --anchors random --games 20 --role diagnostic --out runs/p2_aa/r_ladder.jsonl      # K2: the suite row
# Verdicts (each side's twelve reps: the six on file and the six new):
S="runs/p2_w_quarter_down/sf5/reps_iter0010_r*.jsonl runs/p2_aa/start_sf5_more/reps_iter0010_r*.jsonl"
D="runs/diag_cod/curve/size_4000000/sf5/reps_best_r*.jsonl runs/p2_aa/d4_sf5_more/reps_best_r*.jsonl"
R="runs/p2_aa/r_sf5/reps_best_r*.jsonl"
.venv/bin/python scripts/eval_reps.py verdict --a start $S --b r $R --out runs/p2_aa/p2aa_verdict_r_vs_start.json   # K1
.venv/bin/python scripts/eval_reps.py verdict --a start $S --b d4 $D --out runs/p2_aa/p2aa_verdict_d4_vs_start.json # K0
.venv/bin/python scripts/eval_reps.py verdict --a d4 $D --b r $R --out runs/p2_aa/p2aa_verdict_r_vs_d4.json         # reported
```

## What would make p2_aa uninterpretable

**It replicates the training's randomness, not its data.** R sees D4's very rows, in another order. A second
set of four million positions would be a stronger test, and a different round.

**Selected by held-out cross-entropy.** Both checkpoints are the best of their runs by the teacher measure, not
by skill; the selection is on the measure the DIAG fixed, the same rule for both, and skill was never consulted.

**The start's reps.** Its first six were measured on 2026-11-11/12 by p2_w's finisher; the six added are measured now on the
same instrument. A drift in the instrument between the two dates would show as a difference between the two sets
of six, which is reported.

## Sign-off

Checked for the draft, 2026-11-14: the DIAG's records are committed (see its §7); `runs/diag_cod/teacher` holds
3,524,304 rows and `runs/p2_u_fresh/teacher` 475,696, so the 4,000,000-row set is the same rows D4 trained on.
D4's checkpoint sha256 `7f827f086b28e2b8…`; the start's `43494548987cedcc…`. Pending: the `runs/p2_aa/` negations
and `MUST_REACH`, the run script, Michael's sign-off.
