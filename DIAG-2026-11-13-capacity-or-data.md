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

*(Empty until the run.)*
