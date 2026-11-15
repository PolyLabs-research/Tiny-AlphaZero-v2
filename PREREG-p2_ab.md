# Pre-registration — p2_ab: twice the teacher data, at the replicated protocol's passes per position

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-15** (see Sign-off), at Michael's instruction ("Ok given, go"). Nothing under
> this file has run.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_aa replicated the diagnostic's D4: training the standing endpoint on 4,000,000 teacher-labelled positions,
each seen about four times, raises skill-5 strength over the start (R +0.035 [+0.011, +0.060], D4 +0.045
[+0.022, +0.068], twelve reps each; RUNLOG-p2_aa). The protocol is the lever. Michael chose its first single
change: **more data** ("I'm always about the data first", 2026-11-15).

**Does training on 8,000,000 teacher positions, at the same ~4 passes per position, play stronger at skill 5
than the 4,000,000-position protocol?**

## Contract

| | |
|---|---|
| Start | `runs/p2_w_quarter_down/ckpt_iter0010.pt` (sha256 `43494548…3857`), as D4 and R |
| The new positions (pool 2) | `scripts/pool_positions.py --out runs/p2_ab/pool --limit 4000000 --seed 0`, rings in this order: the 14 local rings of the DIAG's pool (p2_w_wide_down … p2_t_teach), then `runs/p2_s_cont/replay`, then the archived `p2_s_sims`, `p2_s_down`, `p2_r_sims` (read through `runs/_ringview/<run>/`: the archive's `.dat` files on the expansion drive, linked, beside each run's tracked `meta.json`); `--exclude-ring runs/p2_u_fresh/teacher`, `runs/p2_u_fresh/teacher_holdout` and `runs/diag_cod/pool`. Counted at draft: these rings hold **4,214,593** distinct positions that D1, the held-out ring and pool 1 do not (2,389,158 in the 14 local rings, 469,277 in p2_s_cont, 454,320 + 453,306 + 448,532 in the three archived), so the seeded 4,000,000 are 95% of them. **About 57% come from the same p2_t–p2_w networks as D4's data, about 43% from the p2_r/p2_s networks just before** |
| Labels | as the DIAG's: Stockfish 16 at 100,000 nodes, multipv 5, temperature 50 cp, `label_teacher.py --no-holdout --labels-only` on one secure RTX 4090 pod (64 vCPU asked; 54 workers last time, about 152 positions a second), the shards pulled here, the ring built here with `--outcome-model-from runs/p2_u_fresh/teacher` into `runs/p2_ab/teacher`. Pod deadline **10 h** |
| E8 (the line) | `scripts/distill_curve.py --start <start> --d1 runs/p2_u_fresh/teacher --new runs/diag_cod/teacher --new runs/p2_ab/teacher --holdout runs/p2_u_fresh/teacher_holdout --sizes 8000000 --max-samples 32000000 --seed 2 --out runs/p2_ab/e8 --device cuda` (D1's 475,696 + pool 1's 3,524,304 + pool 2's 4,000,000 rows; `distill.prefix_plan`); everything else D4's: lr 2e-5 constant, batch 1,024, AdamW with the start's weight decay, teacher rows only, held-out measured every 250,000 samples, patience 4. **32,000,000 samples is D4's four passes per position over twice the rows**; the result is the checkpoint best by held-out cross-entropy, as D4's and R's |
| The 4M protocol (the reference) | D4 (`runs/diag_cod/curve/size_4000000/best.pt`) and R (`runs/p2_aa/r/size_4000000/best.pt`), each with its twelve skill-5 reps on file (`runs/diag_cod/curve/size_4000000/sf5/` + `runs/p2_aa/d4_sf5_more/`; `runs/p2_aa/r_sf5/`): **24 reps over two runs**, so the reference already holds the run-to-run spread p2_aa measured (R − D4 −0.010 [−0.031, +0.012]). Not re-measured |
| Instrument | `eval_reps.py run --anchor stockfish:5 --config runs/p2_w_quarter_down/config.yaml`, **twelve reps of E8**; verdicts by `eval_reps.py verdict` (per-opening rep-means, paired bootstrap over the 100 openings, 10,000 resamples, seed 0) |
| Where and cost | labelling on the pod: about 7.3 h, **about $7 expected, $8.90 at the 10 h deadline** (balance about $39.6). Everything else on this box under `gpu hold`: the ring build (minutes), E8's training (32M samples at about 1,680 a second, about 5.3 h), twelve reps (about 1.7 h), the suite row. **About 15 h end to end**, run by one driver as the transient user unit `taz-p2ab`, each phase skipped once its output exists |

## Registered expectations

### L — the data are what this file says
L1: pool 2 holds 4,000,000 rows (`pool.json`), none of them a position of D1, the held-out ring or pool 1. L2: every
row is labelled (the pod's shards, pulled), the ring built here has 4,000,000 rows, its outcome model is D1's
(`fitted_here` false), and its settings are 100,000 nodes, multipv 5, temperature 50, Stockfish 16. If L fails,
nothing below is read until it is explained.

### C — the teacher measure (reported)
E8's best held-out cross-entropy beside D4's 2.2048 and R's 2.2051, its top-1 and value MSE beside theirs, and
whether it ended at the 32M cap still improving (best at 31,750,000 samples or later, the DIAG's Addendum A
test).

### K1 (primary) — E8 against the 4M protocol, skill 5
`p2ab_verdict_e8_vs_4m.json`: E8's twelve reps against D4's and R's twenty-four.

### K2 — mate-in-2
At E8's checkpoint, from `scripts/eval_ladder.py`'s suite row (128 sims, the start's config). **K2 passes if E8
solves at least 183 of 200.**

### Reported, no decision
E8 − D4, E8 − R (twelve reps each side) and E8 − start (the start's twelve reps from p2_aa); the share of E8's
pool-2 rows from each source ring.

## Predictions (written before launch)

- L holds. Labelling 7–8 h.
- C: E8's held-out CE 2.19–2.21; more likely than not it ends at the cap still improving.
- **K1: point +0.000 to +0.030; the interval includes zero, more likely than not.** At twelve reps against
  twenty-four the interval's half-width is about ±0.019, and a doubling of data at fixed passes is unlikely to
  be worth more than that.
- K2 passes (E8 185–193).
- E8 − start: +0.03 to +0.07, excluding zero.

## Decision rules

| L | K1 (E8 − 4M protocol) | K2 | Decision |
|---|---|---|---|
| holds | **entirely above zero** | passes | **More data helps at this scale.** The 8M protocol is standing; p2_ac changes one thing, data first: twice the data again (16M, so 8M more positions: the local and archived rings hold 12.8M distinct positions not yet labelled, about 8.8M of them left after pool 2, so 16M is within reach only by reaching into the older archived rings, p2_e–p2_q, or fresh self-play), unless Michael picks otherwise |
| holds | includes zero | passes | **No measurable gain from doubling the data at this resolution.** Not a proof that more data does nothing: the effect may be under ±0.019. The 4M protocol stays the reference (cheaper); p2_ac is Michael's choice: more reps to resolve a smaller effect, a learning-rate step-down after the protocol, or self-play rows mixed back in |
| holds | **entirely below zero** | any | **Twice the data cost skill.** Read first against the 43% of pool 2 from the p2_r/p2_s networks (an older distribution) before anything about data size; diagnosed before the next round |
| holds | any | **fails** | The mate-in-2 cost is diagnosed first; K1 is reported |
| **fails** | | | Nothing is read until the data are explained |

## Measurement protocol

```bash
# runs/p2_ab_driver.sh as the transient user unit taz-p2ab, from main at the registration commit:
# 1. pool 2 (this box):
R="--ring runs/p2_w_wide_down/replay … --ring runs/p2_t_teach/replay --ring runs/p2_s_cont/replay
   --ring runs/_ringview/p2_s_sims --ring runs/_ringview/p2_s_down --ring runs/_ringview/p2_r_sims"
.venv/bin/python scripts/pool_positions.py --out runs/p2_ab/pool --limit 4000000 --seed 0 $R \
    --exclude-ring runs/p2_u_fresh/teacher --exclude-ring runs/p2_u_fresh/teacher_holdout --exclude-ring runs/diag_cod/pool
# 2. the pod (as the DIAG's driver): stage the code and pool 2's boards only; on the pod
python3 scripts/label_teacher.py --ring runs/p2_ab/pool --out runs/p2_ab/teacher --no-holdout --labels-only --stockfish /usr/games/stockfish
# 3. pull the shards every 10 min; when DONE and all 4,000,000 rows are here, PULLED (the pod deletes itself)
# 4. the ring, here: the same command, rerun, labels nothing and builds it
.venv/bin/python scripts/label_teacher.py --ring runs/p2_ab/pool --out runs/p2_ab/teacher --no-holdout \
    --outcome-model-from runs/p2_u_fresh/teacher --stockfish /usr/games/stockfish --workers 2
# 5. E8, under gpu hold:
.venv/bin/python scripts/distill_curve.py --start runs/p2_w_quarter_down/ckpt_iter0010.pt --d1 runs/p2_u_fresh/teacher \
    --new runs/diag_cod/teacher --new runs/p2_ab/teacher --holdout runs/p2_u_fresh/teacher_holdout \
    --sizes 8000000 --max-samples 32000000 --seed 2 --out runs/p2_ab/e8 --device cuda
# 6. twelve reps, the suite row, the verdicts (eval_reps verdict exits 1 when its interval includes zero: not a failure):
.venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_ab/e8/size_8000000/best.pt --config runs/p2_w_quarter_down/config.yaml \
    --anchor stockfish:5 --reps 12 --out-dir runs/p2_ab/e8_sf5
.venv/bin/python scripts/eval_ladder.py --ckpt runs/p2_ab/e8/size_8000000/best.pt --config runs/p2_w_quarter_down/config.yaml \
    --anchors random --games 20 --role diagnostic --out runs/p2_ab/e8_ladder.jsonl
FOUR_M="runs/diag_cod/curve/size_4000000/sf5/reps_best_r*.jsonl runs/p2_aa/d4_sf5_more/reps_best_r*.jsonl runs/p2_aa/r_sf5/reps_best_r*.jsonl"
.venv/bin/python scripts/eval_reps.py verdict --a protocol_4m $FOUR_M --b e8 runs/p2_ab/e8_sf5/reps_best_r*.jsonl \
    --out runs/p2_ab/p2ab_verdict_e8_vs_4m.json                                                        # K1
# reported: e8 vs d4, e8 vs r, e8 vs start (p2_w_quarter_down's six + p2_aa's six)
```

## What would make p2_ab uninterpretable

**Two quantities change together.** Holding passes per position at about four (D4's) means E8 trains for twice the
samples. That is the registered meaning of "twice the data at this protocol"; the other meaning, 8M rows in D4's
16M samples (two passes each), changes passes per position instead, and is a different round.

**Pool 2 is not pool 1's distribution.** About 43% of it comes from the p2_r/p2_s networks, a week or two older
than D4's. Taking only p2_t–p2_w positions would have capped pool 2 at 2.39M. A result against E8 is read against
this first (the third decision row).

**Resolution.** Twelve reps against twenty-four resolve about ±0.019 at skill 5. A real gain smaller than that is
read as "includes zero", which the second row says plainly.

**The selection rule.** As D4 and R: the checkpoint best by held-out cross-entropy, never by skill.

## Sign-off

Checked for the draft, 2026-11-15, on this box: the counts in the Contract (a read-only pass of `select_rows`
over the 15 local and 22 archived v4 rings with the exclusions above, 31 s; all 22 archived v4 rings together hold
12,845,607 such positions); `distill_curve.py` takes several `--new` rings, read whole and in order
(`distill.prefix_plan`, tested; the end-to-end test runs two). Disk: 44 GB free; pool 2, its labels and its ring
take about 9 GB. **Registered 2026-11-15** at Michael's instruction ("Ok given, go"). The run is
`runs/p2_ab_driver.sh` as the transient user unit `taz-p2ab` (the DIAG's pod and pull phases with this round's
pool, rows and a 10 h deadline; each phase skipped once its output exists; verdict steps read `eval_reps verdict`'s
exit 1 as an interval that includes zero, not a failure). The round's records (pool 2's `pool.json`, the ring's
`teacher.json` and `meta.json`, E8's `result.json` and `log.jsonl`, the reps, the suite row, the verdicts) are
reachable and red-checked through `MUST_REACH`; the ring-view directories and pod staging copies are pinned
ignored (their `meta.json` is a copy). Balance before launch is recorded in the driver's log.
