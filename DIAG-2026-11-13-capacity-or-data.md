# DIAG 2026-11-13 — capacity or data: can the 6.5M network take in more of the teacher?

**Status: DRAFT plan, 2026-11-13. Nothing has run.** Awaiting Michael's OK (the labelling uses a RunPod pod,
about $5). Results go in §7, below the line; nothing above it changes after the report is read, except by a dated
addendum.

## 1. Question

Since p2_t the network has not moved at skill 5 (RUNLOG-p2_w §6 reading 5), and the teacher levers that worked
once are spent: a fresh ring gained nothing (p2_u), a larger share of each batch cost skill (p2_v, p2_w). At the
standing endpoint the network agrees with the teacher's first choice on 36% of held-out positions, at a
cross-entropy of 2.367 nats (T1, RUNLOG-p2_w). Every teacher round so far trained on one ring of about 475,000
labelled positions.

**Is the network's distance from the teacher set by its size or by how few labelled positions it has seen?** If
more labelled positions keep moving it toward the teacher, the next round is large-scale distillation with no
change to the model (the project's aim: push the small model). If it flattens, the model is near what it can hold,
and the next lever has to be something else.

This is a diagnostic: it measures agreement with the teacher, not strength. p2_v already showed that ending closer
to the teacher is not the same as playing better (the half line was 0.097 nats closer and weaker). §5 reports a
strength reading beside it, which decides nothing here.

## 2. Instrument

**Positions.** The distinct positions of the 14 self-play rings on this box (the hot and step-down rings of
`p2_t_teach`, `p2_u_fresh`, `p2_u_stale`, `p2_v_half`, `p2_v_quarter`, `p2_w_quarter`, `p2_w_wide`; 500,000 rows
each), selected as `teacher.select_rows` selects them (book, start position and mate suites out; rows that
repeat a position of their own game out), pooled and de-duplicated across rings by `position_key`. Excluded as
well: every position of the held-out ring below, and every position already in `runs/p2_u_fresh/teacher` (those
labels are reused, not repeated).

**Labels.** Stockfish 16 at the standing teacher's settings as `runs/p2_u_fresh/teacher/teacher.json` records
them (**100,000 nodes**, multipv 5, temperature 50 cp, hash 64 MB; the first draft said 200,000, the
`TeacherSettings` default, which is not what D1 was labelled at), by `scripts/label_teacher.py --no-holdout
--outcome-model-from runs/p2_u_fresh/teacher`, on a RunPod pod as p2_u's labelling ran (about 214 positions a
second on 54 workers there). The value targets of every new row come from D1's fitted outcome model (a 303, s
367), so all four sets carry one fit.

**Four nested training sets**, each containing the one before:

| Set | Rows | Made of |
|---|---|---|
| D1 | 475,696 | `runs/p2_u_fresh/teacher` as it stands (already labelled) |
| D2 | 1,000,000 | D1 + 524,304 new rows |
| D3 | 2,000,000 | D2 + 1,000,000 new rows |
| D4 | 4,000,000 | D3 + 2,000,000 new rows |

New rows enter in a seeded random order (seed 0) fixed before labelling, so D2 ⊂ D3 ⊂ D4 by construction.
3,524,304 positions are labelled. If the pool holds fewer distinct positions than that, D4 is what the pool holds
and that is reported.

**Held-out set.** `runs/p2_u_fresh/teacher_holdout` (5,000 rows): the T1 ring every teacher round has read, so §7
lines up with RUNLOG-p2_t to p2_w. None of its positions is in any training set.

**Training, per set.** From `runs/p2_w_quarter_down/ckpt_iter0010.pt` (the standing endpoint), teacher rows only,
with the loss Phase 2 applies to teacher rows (policy cross-entropy to the teacher's distribution, value to the
fitted outcome, promotion as for teacher rows), batch 1,024, AdamW as in its `config.yaml`, learning rate 2e-5
(the hot rate) held constant. Held-out cross-entropy is measured every 250,000 samples; training stops after 4
measurements without a new best, or at 16,000,000 samples (4 passes over D4), and the best checkpoint is the
set's result. Seed 0 for batch order. Run on this box's GPU after p2_y.

**Measures, at each set's best checkpoint** (`scripts/teacher_agreement.py`, plus two additions):
- `C(D)`: held-out policy cross-entropy to the teacher (nats). Baseline: the start, 2.367.
- Held-out top-1 agreement and value MSE against the teacher (reported).
- The teacher's own entropy on the held-out rows, `H_T`, the floor no network can pass; `C(D) − H_T` is the
  part of the gap that is learnable.
- `C_train(D)`: the same cross-entropy on a fixed 5,000-row sample of D1 (seed 0; in every training set), so a
  held-out-to-training gap shows whether the network is fitting what it sees.

## 3. Reading rule (registered here, before any label exists)

`Δ_top = C(D4) − C(D3)`, with a 95% interval from a paired bootstrap over the 5,000 held-out rows (10,000
resamples, seed 0).

| Δ_top | Reading |
|---|---|
| interval entirely below −0.01 | **Data-limited.** The network was still moving toward the teacher when its data doubled from 2M to 4M. Next: a training round with large-scale distillation (a PREREG; the teacher at a dose p2_v allows, from many more labelled positions) |
| interval entirely above −0.01, and `C(D4) − C_train(D4)` < 0.05 | **Capacity-limited.** Doubling the data moved it less than 0.01 nats, and it does no better on rows it has seen. Next: not more teacher data; Michael's choice from the open list |
| interval entirely above −0.01, and the gap ≥ 0.05 | **In between.** Flat on held-out, but fitting its training rows better than new ones: more data alone is not paying, the fit is. Reported with the curve; next step decided with Michael |
| interval contains −0.01 | **Not resolved at this size.** Reported; whether to add a D5 (8M) is Michael's call |

The whole curve `C(D1..D4)` is reported whatever the row, with `H_T` beside it.

## 4. Limits, stated before

- **Agreement is not strength** (§1, p2_v). A data-limited reading licenses a distillation round, not a claim
  that it will play better.
- **One teacher setting.** A deeper teacher (more nodes) is a different question.
- **One starting point.** The curve is measured from the standing endpoint, not from scratch; it says how much
  further this network can move, which is the question a next round would ask.
- **The positions are the network's own.** They come from p2_t–p2_w self-play, so the curve is about the
  positions it reaches; it says nothing about positions it never plays into.
- **Learning rate held constant**, with no step-down: every set gets the same schedule, so the comparison is fair,
  and none reaches what a step-down might add.

## 5. Strength aside (reported, decides nothing)

The best D4 checkpoint and the best D1 checkpoint, six skill-5 reps each, against the start's six: whether moving
toward the teacher on these positions shows at the board at all.

## 6. What it costs and what has to be built

- **Labelling:** 3,524,304 positions at about 214 a second is about 4.6 hours on one secure RTX 4090 pod (64
  vCPU, $0.89/h, as p2_u): **about $4.50 expected, $7.12 at an 8-hour deadline.** Balance checked and stated at
  launch. Upload: the 14 rings' position files; download: the labelled rows, about 3.8 GB (the box has 86 GB free).
- **Training:** this box's GPU after p2_y, about 1,200 samples a second (`benchmarks/results.jsonl`): at most
  4 × 16M samples, more likely 15–30M in all with early stopping, so roughly 4–7 hours, no spend.
- **Code, with tests, before anything runs:** `scripts/pool_positions.py` (new; `teacher.pool_rows`,
  `teacher.write_pool`) pools the 14 rings into one ring in the seeded order, de-duplicated across rings and
  without D1's or the held-out ring's positions; `scripts/label_teacher.py` gains `--no-holdout` and
  `--outcome-model-from`; `scripts/distill_curve.py` (new; `tinyalphazero.distill`) builds the nested sets,
  trains each with `train_step_phase2` and early stopping, writes one JSONL row per measurement, and applies §3
  (`distill.read_curve`); `teacher.agreement` (moved out of `scripts/teacher_agreement.py`, its summary
  arithmetic unchanged and tested to be) adds `H_T` and per-row CE for the paired bootstrap.

---

## 7. Results

**Read 2026-11-14 ~19:45 UTC**, after the driver ended (18:39:44) and after Addendum A was committed alone
(`1b124d6`). Everything below is from the run's files: `runs/diag_cod/pool/pool.json`,
`runs/diag_cod/teacher/teacher.json`, `runs/diag_cod/curve/summary.json`, `size_*/result.json` and `log.jsonl`,
`size_*/sf5/diag_verdict_vs_start.json` and the reps beside them, and `runs/diag_cod_driver.console.log`.

### 7.1 What ran

- **Pool** (`598d892`): 5,913,462 distinct positions across the 14 rings after selection, de-duplication and the
  two excluded rings; 3,524,304 pooled (seed 0). `p2_t_teach_down` contributed none: all its selected positions
  are D1's or the held-out ring's (495,898 excluded). Pool 3.6 GB.
- **Labelling**: transient unit `taz-diag-cod` from `58c6c11`, armed 00:22:39 UTC. Pod `1zzvowvgdyhx19` (secure
  RTX 4090, 64 vCPU, cgroup quota 54.4 CPUs, 54 workers; Stockfish `16-1build1`; `boards.dat` sha256 matched on
  the pod) launched 00:22:42; job started 00:26:05; `job DONE`, exit 0, at 06:51:42 with all 3,524,304 rows
  pulled here; `PULLED` written 06:51:44 and the pod deleted itself. About 152 positions a second (23,137 s), below
  p2_u's 214. **Cost $5.81** (balance $45.38 before launch, $39.57 after; no other pod of ours at either reading;
  about 6.5 pod-hours at $0.89/h, $5.78, consistent); worst case was $7.12.
- **Ring**: the same labeller command rerun here, 06:51:44: 0 of 3,524,304 rows to label (a 32 s scan), ring
  written by 06:53:49 — 3,524,304 rows at 100,000 nodes, multipv 5, temperature 50 cp, outcome model D1's
  (a 303.35, s 367.36, `fitted_here: false`), draw rate 0.305.
- **Curve** (`distill_curve.py` under `gpu hold`, lr 2e-5, batch 1,024, measured every ~250,000 samples,
  patience 4, cap 16,000,000): D1 06:53:52–09:19:25 (15.0M samples), D2 –11:54:30 (16.0M), D3 –14:28:55
  (16.0M), D4 –17:00:50 (16.0M); about 1,720 samples a second with measurements.
- **The start's measurement reproduced the registered T1 reading exactly**: top-1 0.3658, cross-entropy
  2.367300341796875, value MSE 0.037847105026245116, the values in
  `runs/p2_w_quarter_down/teacher_agreement_iter0010.json` (RUNLOG-p2_w). `teacher.agreement` measures what the
  script measured. Its training-probe CE at the start: 2.3577.
- **Strength aside**: six skill-5 reps of each best checkpoint, with the start's `config.yaml`: D1 17:00:52–17:47:01
  (7.4–8.0 min a rep), D4 17:47:01–18:39:44 (8.3–9.2 min a rep).

### 7.2 The curve

The teacher's own entropy on the held-out rows, `H_T` = **1.1886** nats (the floor).

| Set | Rows | `C(D)` held-out CE | Top-1 | Value MSE | `C_train` (probe) | Held-out − probe | Best at (samples) | Run (samples) | Passes at best |
|---|---|---|---|---|---|---|---|---|---|
| start | — | 2.3673 | 0.3658 | 0.0378 | 2.3577 | 0.010 | — | — | — |
| D1 | 475,696 | **2.2012** | 0.3804 | 0.0943 | 2.1120 | 0.089 | 14,000,128 | 15,000,576 (patience) | 29.4 |
| D2 | 1,000,000 | **2.1936** | 0.3858 | 0.0562 | 2.1605 | 0.033 | 15,750,144 | 16,000,000 (cap) | 15.8 |
| D3 | 2,000,000 | **2.1990** | 0.3862 | 0.0401 | 2.1812 | 0.018 | 16,000,000 | 16,000,000 (cap) | 8.0 |
| D4 | 4,000,000 | **2.2048** | 0.3826 | 0.0342 | 2.1935 | 0.011 | 15,750,144 | 16,000,000 (cap) | 3.9 |

**Δ_top** = `C(D4) − C(D3)` = **+0.0058 [+0.0038, +0.0079]** (paired bootstrap over the 5,000 held-out rows,
10,000 resamples, seed 0). D4 minus D1: +0.0036 [−0.0016, +0.0086]. **Fit gap at D4** (`C(D4) − C_train(D4)`):
0.0114.

### 7.3 The reading: **not resolved (compute)**

Under §3 alone, Δ_top's interval lies entirely above −0.01 and the fit gap is under 0.05: the capacity-limited
row. That is what `summary.json`'s `reading` field says, because `distill.read_curve` applies §3 alone. **Addendum A
overrides it.** D3's best came at its last measurement (16,000,000 samples) and D4's at 15,750,144: both ran to the
cap with their best at 15.75M or later, so both *ended at the cap still improving*. Δ_top is not data-limited
(its interval is above −0.01, not below), so the capacity-limited row does not apply and the reading is **not
resolved (compute)**: at a 16M-sample cap the larger sets were stopped before converging, and the curve cannot say
whether the network is at its capacity. Whether to rerun D3 and D4 with a higher cap is Michael's call.

### 7.4 Strength aside (§5; decides nothing)

Six skill-5 reps each against the start's six (`runs/p2_w_quarter_down/sf5/reps_iter0010_r*.jsonl`; start mean
0.3788), paired bootstrap over the 100 openings:

| Checkpoint | Reps | Mean | Paired difference against the start |
|---|---|---|---|
| D1 best (`size_475696/best.pt`) | 0.1125, 0.1125, 0.1025, 0.115, 0.1275, 0.105 | **0.1125** | **−0.266 [−0.297, −0.235]** |
| D4 best (`size_4000000/best.pt`) | 0.4175, 0.4025, 0.3825, 0.425, 0.425, 0.4425 | **0.4158** | **+0.037 [+0.005, +0.069]** |

D4's interval excludes zero, and 0.4158 is the highest six-rep skill-5 mean the project has measured (RUNLOG-p2_w
§6 reading 5 gave 0.379 as the best). It is one checkpoint, from an instrument registered here as deciding
nothing.

### 7.5 Readings the rule does not cover (not registered; hypotheses)

1. **At equal samples, the larger set trails, slightly.** Held-out CE at matched samples (D1 / D2 / D3 / D4): 3.75M
   2.2226 / 2.2267 / 2.2303 / 2.2314; 7.75M 2.2089 / 2.2122 / 2.2186 / 2.2205; 11.75M 2.2021 / 2.2020 / 2.2090 /
   2.2122. A row seen less often is learnt more slowly, so under one cap the larger sets were the least converged.
   This is the effect Addendum A was written for.
2. **D1 over-fitted, and it played far worse.** Its probe-to-held-out gap grew to 0.089 by its best (0.010 at the
   start), against 0.033, 0.018 and 0.011 for D2–D4. It ended 0.166 nats closer to the teacher than the start on
   held-out positions and lost 0.266 at skill 5.
3. **The value head improved with set size.** Value MSE against the teacher: 0.094, 0.056, 0.040, 0.034 for D1–D4,
   against the start's 0.038. Only D4 ended below the start. D1's collapse at the board coincides with its worst
   value head.
4. **A hypothesis that fits p2_v.** p2_v's half line drew each row of a 475,696-row teacher ring about 12.9 times
   against the quarter line's 6.5 (PREREG-p2_w) and lost skill while moving closer to the teacher. Here, 29 passes
   over that ring's positions cost 0.266, and four passes over four million positions gained 0.037. **Hypothesis,
   not result:** teacher rows hurt when the same rows are repeated many times (over-fitting the policy and the
   value head) and help when many distinct rows are each seen a few times. That would make the lever "many more
   labelled positions, each seen rarely", not "more teacher per batch".

### 7.6 Next

Michael agreed (2026-11-14) to a registered training round that tests whether the D4 result replicates, drafted
separately as PREREG-p2_aa. Rerunning D3 and D4 with a higher cap to resolve §3 is not scheduled.

## Addendum A — 2026-11-14 13:30 UTC, during the run, before D3 or D4 has a result

Written at Michael's OK, while `taz-diag-cod` runs (from `58c6c11`): D1 and D2 had finished, D3 was at 9.75M of
its 16M samples (held-out CE 2.2116) and D4 had not started. No reading of §3 had been made, no `summary.json`
existed, and neither D3's nor D4's best was known. Committed after the driver ends, so that nothing tracked changes
while it runs.

**Why.** §2's cap (16,000,000 samples) gives the larger sets fewer passes over their rows: 16 over D2, 8 over D3,
4 over D4. Observed so far: at equal samples the larger set trails slightly (each row is seen less often), D1
stopped on patience at 15.0M (best 2.2012 at 14.0M) and D2 ran to the cap still improving (best 2.1936 at
15.75M). A set stopped by the cap rather than by patience has not converged, so its `C(D)` overstates where it
would end, and a flat Δ_top could then be the cap, not the network's size. §3 did not separate the two.

**Rule added.** A set *ended at the cap still improving* if it ran to 16,000,000 samples and its best held-out CE
came at one of its last two measurements (at 15,750,000 samples or later).

- If D3 or D4 ended at the cap still improving, §3's **capacity-limited** and **in between** rows do not apply:
  the reading is **not resolved (compute)**. The curve and Δ_top with its interval are reported; whether to rerun
  D3 and D4 with a higher cap is Michael's call.
- §3's **data-limited** row stands whatever the cap did: the cap works against the larger sets, so a Δ_top
  interval entirely below −0.01 under it understates, not overstates, what more data does.
- §3's **not resolved** row is unchanged.

Everything else in §2–§6 is unchanged. Observed during the run and noted here before any reading, not as a rule:
the value head's MSE against the teacher rose under teacher-only training (start 0.038; D1's best 0.094, D2's
0.056), which bears on §5's strength aside, not on the policy curve.
