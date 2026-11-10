# Pre-registration — p2_v: the standing teacher at twice its dose, half of every batch, against a quarter

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-10** (see Sign-off), at Michael's instruction. Nothing under this file
> has run.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_t made the teacher standing: a quarter of every batch from self-play positions relabelled by Stockfish beat the
same schedule without it (K1 +0.050 [+0.014, +0.085]). p2_u then ran the standing teacher for another round, on a
fresh ring and on p2_t's, and neither line ended above its start (0.356 and 0.361 against 0.360; K1 −0.005
[−0.038, +0.028]). The cross-entropy to the teacher moved 0.02–0.03 nats in that round, where p2_t's moved 0.52
(RUNLOG-p2_u §6, reading 2). At a quarter of the batch the teacher's effect looks used up. Michael chose
(2026-11-10) the first of RUNLOG-p2_u's levers: **a larger dose, 0.5.**

**H_dose: the teacher at half of every batch ends stronger at skill 5 than the same teacher at a quarter, without
costing mate-in-2.**

The network stays at 6.5M parameters. The instrument stays where it was.

## The ring and the dose

| | Line A, "half" (the lever) | Line B, "quarter" (the teacher as it stands) |
|---|---|---|
| Teacher ring | **`runs/p2_u_fresh/teacher`**, as p2_u built it: 475,696 rows from `runs/p2_t_teach_down`'s final ring, Stockfish 16 at 100,000 nodes, `multipv` 5, T = 50 cp; outcome model a = 303.3, s = 367.4 (29.5% of level positions drawn). Held out: `runs/p2_u_fresh/teacher_holdout`, 5,000 rows | the same ring |
| `train.teacher_frac` | **0.5**: 512 of each 1,024-row batch; the self-play ring takes 461, the mate rehearsal keeps its 51 | 0.25: 256; the ring takes 717, the rehearsal 51 |
| Teacher draws over 30 iterations (12,000 steps) | 6.14 million, about **12.9 passes** over the ring | 3.07 million, about 6.5 passes |
| Self-play ring draws over 30 iterations | 5.53 million | 8.60 million |

No ring is labelled this round. p2_u found relabelling gains nothing measurable, so reuse is the default, and this
ring is the one the start network has never drawn a teacher row from: the start is p2_u's line B, which trained on
p2_t's ring. Its positions are the start's lineage's own: `runs/p2_t_teach_down`'s final ring, which p2_u's line B
drew as self-play rows, with the search's targets, until its ring turned over.

## Contract

| | |
|---|---|
| Start | **`p2_u_stale_down/0010`**, p2_u's line B endpoint: the standing protocol's lineage (p2_t's teacher reused, 0.25, 2e-5 then 2e-6), and the higher of p2_u's two endpoints by six-rep skill-5 mean (0.3608 against 0.3563). Both lines: `--seed 43 --init runs/p2_u_stale_down/ckpt_iter0010.pt --seed-pool runs/p2_u_stale_down --seed-replay runs/p2_u_stale_down` (ring head 1,037,734, offset −10) |
| Line A, half | **`p2_v_half`, hot:** `--config configs/p2_v_half_hot.yaml` (48 sims, **2e-5**, teacher **0.5** from `runs/p2_u_fresh/teacher`; seed-43 fingerprint `d010ccc7…3285`; differs from `configs/p2_u_fresh_hot.yaml` in `teacher_frac` only, `test_p2v_half_configs_are_the_brief`), 20 iterations. Then **`p2_v_half_down`:** `--config configs/p2_v_half_down.yaml` (48, **2e-6**, 0.5; `3ebcb15e…6181`), `--init runs/p2_v_half/ckpt_iter0020.pt`, `--seed-pool` and `--seed-replay runs/p2_v_half`, 10 iterations |
| Line B, quarter | **`p2_v_quarter`, hot:** `--config configs/p2_u_fresh_hot.yaml` itself (2e-5, teacher 0.25 from the same ring; `066043ca…d67d`, as p2_u's line A), the same start, pool and ring, 20 iterations. Then **`p2_v_quarter_down`:** `--config configs/p2_u_fresh_down.yaml` itself (2e-6; `036cd665…01da`), `--init runs/p2_v_quarter/ckpt_iter0020.pt`, from `runs/p2_v_quarter`, 10 iterations |
| Why this schedule | the standing one (p2_t, p2_u): 2e-5 for 20 iterations, then 2e-6 for 10. In both rounds the 2e-5 stage cost 0.03–0.09 and the step down returned it; the teacher's lead in p2_t formed at 2e-5 and the step down kept it |
| What differs between the lines | the dose, and with it two things the design does not separate (see the end): each teacher row drawn about twice as often, and 256 fewer self-play rows in every batch. Start, seed, games, ring, schedule, network and instrument are the same |
| Start reference | `p2_u_stale_down/0010`: six skill-5 reps (`runs/p2_u_stale_down/sf5/`, mean 0.3608), three skill-3 reps (0.8333), three skill-0 book reps (0.9750), three initial-position reps (0.9875); mate-in-2 191/200, mate-in-3 156/200, ECE 0.1049 (its ladder at 10). T1 on the held-out ring: top-1 0.3590, cross-entropy 2.391; on the whole training ring: 0.3600, 2.393 (read for this draft) |
| Training | RunPod, secure RTX 4090, ≥ 24 vCPU, the image of p2_g–p2_u; variants recorded. If none is offered the registered launch is retried (p2_u's step downs waited through 44 and 66 refusals; Michael chose waiting over another class); a pod that never comes up is terminated and replaced (`cloud/launch_run.sh`); nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, `gpu hold`, nothing until each run's final pull, under the user units, `MEASURE_AFTER_PULL=1`. Drivers judged: `p2_v_half` and `p2_v_quarter` `20`; both step downs `5 10` (ladder with ECE and N, three skill-0 reps; initial-position reps at each run's last). Then a finisher (`runs/p2_v_finish.sh`) against `stockfish:5`: three reps at both lines' 0020 and down/0005; **six at both down/0010**; three against `stockfish:3` at the two endpoints; T1 on the held-out ring at both 0020s and both endpoints; the training-ring reading (H12) at both endpoints |
| Lifecycle | four guards. Each hot stage deadline 8 h, worst $7.12; each step down 6 h, $5.34. **Worst case $24.92**; expected about **$11** (about 3.8 h per hot stage and 2.0 h per step down at $0.89/h, as p2_u's line B). Both hot stages launch together; each step down launches after its own hot stage's final pull |

## Registered expectations

### L — the continuity applied
- **L1.** Each run's `provenance.json`: the `init_from`, `seed_replay` head and offset above; `seed` 43; `git_dirty`
  false; `config_fingerprint` as above, equal to its `config.yaml`'s sha256; `lr` on every row 2e-5 (hot) or 2e-6
  (down); simulations per position 48.0; `teacher_frac` and `teacher_dir` as registered in each `config.yaml`.
- **L2.** Book positions in each final ring: reported, no bound.
- **L3, the ring as registered.** The staged `runs/p2_u_fresh/teacher/` on each pod has the sha256 of this box's
  copy (the launcher checks every input), and its `teacher.json` reads 475,696 rows.

### T1 — the lever applied: agreement with the teacher on the held-out 5,000
`scripts/teacher_agreement.py` on `runs/p2_u_fresh/teacher_holdout`: the share of positions where the network's prior
(no search) puts its top move on Stockfish's best, and its cross-entropy against the teacher's policy. At the start,
both 0020s and both endpoints. **T1 holds if line A's endpoint has the lower cross-entropy.** Top-1 agreement is
reported, not judged. T1 is not a strength claim: it says whether twice the dose moved the network further toward
the teacher on positions it never trained on. Without it, K1 cannot be read as a test of the dose.

### K1 (primary) — the two lines at the end, skill 5
`p2_v_half_down/0010` − `p2_v_quarter_down/0010`, against `stockfish:5`, **six reps per side**, per-opening
rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed 0;
`runs/p2_v_half_down/sf5/p2v_verdict_half_vs_quarter.json`), read with the exact-zero rule. **Expected: the 95% CI
entirely above zero.** Six reps resolve about ±0.033 (p2_u's K1 half-width); a smaller difference will read as
"includes zero".

### K2 — mate-in-2
At the endpoints: **`p2_v_half_down/0010` solves at least as many of the 200 as `p2_v_quarter_down/0010` minus 4**,
and **both ≥ 183**. p2_t's teacher cost 1–4 positions at every measure (RUNLOG-p2_t §6, reading 3); twice the dose
is the place that cost would grow.

### K3 — each line against the start (informational)
At skill 5: both 0020s and both down/0005s (three reps), both endpoints (six), each against `p2_u_stale_down/0010`'s
six. At skill 3: the endpoints' three reps against the start's three. Skill 0, M4 and the initial position as before.

### ECE, H
- **ECE:** < 0.15 at the endpoints, both lines. p2_t's teacher line ran about 0.01 higher at each measure. A break is
  handled in PREREG-p2_l's order.
- **H1–H10** as PREREG-p2_r, per run, direction-free. **H7:** the policy and value losses of line A against line B;
  the batches mix teacher and ring rows in different shares, so the totals are not comparable between the lines or
  with earlier rounds. **H2:** `skipped_steps` < 20 per line.
- **H12, memorisation (new, reported, no bound).** The cross-entropy on the whole training ring (475,696 rows) minus
  that on the held-out 5,000, at both endpoints. At the start it is +0.002 (2.393 against 2.391), since the start
  network never drew these rows as teacher rows. A gap that opens further in line A than in line B is the 12.9
  passes showing.

## Predictions (written before launch)
- T1 holds: line A's endpoint 0.02–0.10 nats below line B's on the held-out ring; top-1 agreement within 2 points.
- H12: line A's gap −0.01 to −0.05, line B's 0.00 to −0.02.
- K1: point −0.01 to +0.04; more likely than not the CI includes zero. K2 passes, line A at or 1–4 below line B.
- K3: both 0020s below the start (the cost of 2e-5); at the endpoints line A at or above the start, line B within
  noise of it, as in p2_u.
- ECE 0.08–0.13, line A at or above line B; N 185–195; L2 0–5 per run.

## Decision rules

| K1 | K2 | T1 | Decision |
|---|---|---|---|
| **CI entirely above zero** | passes | holds | **Twice the dose works.** The teacher becomes standing at 0.5 with this schedule. p2_w's lever is Michael's choice: a larger dose still (0.75), a deeper engine, the 2e-7 step-down, the public benchmark |
| CI entirely above zero | fails | | The dose stays at 0.25 until the mate-in-2 cost is diagnosed |
| **CI includes zero** | any | holds | **Twice the dose moved the network further toward the teacher and gained nothing measurable.** More of this teacher (100,000 nodes) does not reach skill 5 at this schedule; the dose stays at 0.25. p2_w's lever is Michael's choice: a deeper engine, the 2e-7 step-down, the public benchmark, the third line (RUNLOG-p2_t §6, reading 4) |
| CI includes zero | any | fails | **The dose was not applied in effect**: the network did not end closer to the teacher on held-out positions. H12 is read first (memorisation of the training ring in place of generalisation); not read as "more teacher does nothing" |
| **CI entirely below zero** | any | any | **Twice the dose costs skill.** The dose stays at 0.25. Diagnosed before p2_w: H12 (memorisation) and the self-play ring's share (461 rows in place of 717) are the first suspects |
| N < 183 or ECE ≥ 0.15 at an endpoint | | | The broken guard first |

## Measurement protocol

```bash
# Before launch, on this box: restore the start's league pool (the final pull excludes seed_*; the nine files are
# byte copies of runs/p2_u_stale/ckpt_iter0012..0020.pt, hard-linked as the names its state.json lists). Done for
# the draft; see Sign-off.

# All four launches: runs/p2_v_launch.sh (cloud/launch_run.sh each, as transient user units). Both hot stages together:
#   p2_v_half     (p2vh, 8 h): train_phase2.py --config configs/p2_v_half_hot.yaml <start> --out runs/p2_v_half --iterations 20
#   p2_v_quarter  (p2vq, 8 h): train_phase2.py --config configs/p2_u_fresh_hot.yaml <start> --out runs/p2_v_quarter --iterations 20
#   <start> = --seed 43 --init runs/p2_u_stale_down/ckpt_iter0010.pt --seed-pool runs/p2_u_stale_down
#             --seed-replay runs/p2_u_stale_down
#   inputs: the start's 20 pool checkpoints, state.json, config.yaml, replay/, runs/p2_u_fresh/teacher/, the mate
#   data, phase1_anchor.pt
# Each step down after its hot stage's final pull (WAIT_FOR=runs/<hot>/.remote_done), 6 h each:
#   p2_v_half_down:    configs/p2_v_half_down.yaml from runs/p2_v_half, inputs including runs/p2_u_fresh/teacher/
#   p2_v_quarter_down: configs/p2_u_fresh_down.yaml from runs/p2_v_quarter, the same

# After all four: runs/p2_v_finish.sh under taz-finisher@p2_v — reps, each verdict against the start, T1, H12, K1:
.venv/bin/python scripts/eval_reps.py verdict --a quarter runs/p2_v_quarter_down/sf5/reps_iter0010_r*.jsonl \
    --b half runs/p2_v_half_down/sf5/reps_iter0010_r*.jsonl \
    --out runs/p2_v_half_down/sf5/p2v_verdict_half_vs_quarter.json
```

## What would make p2_v uninterpretable
- A difference between the lines beyond the dose (the config test and L1).
- A teacher ring other than the registered one on either line (L3).
- A launch from anything but clean, pushed `main`; a measurement-instrument change.
- Losing a run's final pull: re-run by addendum, not read from what is left.

**What the design does not separate.** Doubling the teacher's share does two things at once: every teacher row is drawn
about twice as often (12.9 passes against 6.5), and every batch carries 256 fewer self-play rows (461 against 717),
so the search's own targets get about two thirds of the weight they had. The round answers the working question
(0.5 or 0.25 from here), not which of the two moves skill. H12 bears on the first. A fixed self-play share with a
larger batch would separate the second; it changes the step and is not run here.

**What the statistic does not cover.** As PREREG-p2_k: each line is one run; K1 resamples openings.

---

## Sign-off

Checked for the draft, 2026-11-10, on this box: `runs/p2_u_stale_down`: iteration 10, step 4,000, seed 43, 0
skipped steps, ring head 1,037,734 equal to `state.json` and `replay/meta.json`. Its league pool lists 20 snapshots:
`ckpt_iter0000`–`0010` (present) and `seed_p2_u_stale_ckpt_iter0012`–`0020`, which the campaign pull excludes; those
nine were restored as hard links to `runs/p2_u_stale/ckpt_iter0012`–`0020.pt`, the files they were copied from
(`_copy_seed_pool` uses `shutil.copy2`), and the pool is now complete. Its six skill-5, three skill-3, three skill-0
and three initial-position reps present, with the means above. `configs/p2_v_half_{hot,down}.yaml` differ from
`configs/p2_u_fresh_{hot,down}.yaml` in `teacher_frac` only, and the batch plans are 461/51/512 and 717/51/256 (test
passes). Seed-43 fingerprints:
`d010ccc7d6bb6aff8d6b2c190d65bedea5104a1ecd00d3ca2230d8291c9d3285` (half hot),
`3ebcb15e1f2f9009f3d1ae298fbd3e2f08cf787f94d9867d5f14130bd2e76181` (half down),
`066043cad1786a4c86f8780f8915a5e958cf905ea9c65ac73a5f90fc3aa4d67d` (quarter hot) and
`036cd665672a35348bff7ce876011f3350b2c5426780bdc95d1dafd6666e01da` (quarter down), the last two equal to
`sha256sum runs/p2_u_fresh{,_down}/config.yaml`. A zero-iteration run on CPU with line A's hot config and the
registered start (`--iterations 0`, into the session scratchpad): pool seeded with all 20 snapshots, ring seeded with
500,000 rows from donor head 1,037,734 at offset −10, `teacher_frac` 0.5 in its `config.yaml`, fingerprint
`d010ccc7…3285`. H12's reading at the start (`runs/p2_u_stale_down/teacher_agreement_train_iter0010.json`, 4.5 min
under `gpu hold`). RunPod balance $44.54 (auto-reload at $10), no pods of ours; secure RTX 4090 at $0.89/h, stock
none at the reading; each pod's worst case passes the $15 ceiling, and the two hot stages' $14.24 leaves $30.30.
Disk 106 GB free; the round adds about 6.5 GB.
Read at registration, 2026-11-10 18:27 UTC: balance $44.54, no pods of ours, secure RTX 4090 at $0.89/h (stock
none: the launches retry until one is offered, as registered); each pod's worst case passes the ceiling and the margin
at its own launch. `make test` passed on the draft's tree, `5f711a3` (1,127 passed, 4 skipped); this commit changes
only this file. Nothing is committed to `main` between this commit and the step downs' launches, so all four runs
carry one `git_sha`.

**Signed: the Claude Code session, at Michael's instruction ("go", 2026-11-10, after the draft and its cost —
expected about $11, worst case $24.92 — were put to him).** The commit carrying this line is the registration.
