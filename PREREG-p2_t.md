# Pre-registration — p2_t: distillation from Stockfish, a quarter of every batch, against the same schedule without it

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: DRAFT 2026-11-07, not signed.** Nothing under this file has run.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

Skill 5 has held at 0.32–0.34 for three rounds (RUNLOG-p2_s §6). Better self-play targets have now had a fair test:
128 simulations at a rate that can use them, then a step down. They gave nothing (p2_s K1 −0.005 [−0.033, +0.023]).
The losses are broad: no kind-by-phase cell holds half of them, and at two blunders in three the model's own value
was well above the engine's (DIAG-2026-11-07-where-points-go §5). Michael chose (2026-11-07) **distillation from
Stockfish evaluations**: targets from a much stronger player than our search, on our own positions.

**H_teacher: training with a quarter of every batch drawn from self-play positions relabelled by Stockfish ends
stronger at skill 5 than the same schedule without them, without costing mate-in-2.**

The network stays at 6.5M parameters; that is the project's aim. The instrument stays where it was.

## The teacher

| | |
|---|---|
| Positions | `runs/p2_s_cont`'s final ring (head 2,069,348): each distinct position once, minus every position of the five suites the instrument reads (book, start, mate-in-1/2/3; 551 rows), minus rows that repeat a position of their game (3,883), minus duplicates (18,337): **477,229 positions**. A seeded 5,000 (seed 0) go to a held-out ring that training never reads; **472,229** train (`scripts/label_teacher.py`, `teacher.select_rows`) |
| Engine | Stockfish 16 (the image's apt `16-1build1`, as this box), **100,000 nodes**, `multipv` 5, one thread, 64 MB hash cleared before each position; a position's labels depend on it alone |
| Policy target | softmax over the five lines' scores, **T = 50 cp**: a move 50 cp worse than the best gets e⁻¹ of its weight; moves outside the five get none |
| Value target | `root_q` = P(win) − P(loss) under a two-parameter WDL model of the engine's score, **fitted to the ring's own game outcomes** (maximum likelihood on the 472,229). Stockfish's own WDL model is fitted to engine games and calls a 45 cp deficit 92% drawn; our self-play games are about 80% decisive. The game outcome `z` is the original row's. The p2_c blend (`value_q_mse_weight` 0.5) trains toward `root_q` as it does toward search-Q |
| Everything else | the position, its legal mask and `z` are the source row's; the row is drawn as a ring row is (`ReplayBuffer.sample`), so the losses and the trainer are untouched |
| Dose | **`train.teacher_frac` 0.25**: 256 of each 1,024-row batch; the ring takes 717 and the mate rehearsal keeps its 51. Over 30 iterations, 3.07 million draws, about 6.5 passes over the 472,229 |
| Where it is built | on line A's first pod, before training (`label_teacher.py`, one engine per allowed CPU), into `runs/p2_t_teach/teacher/`, pulled with the run; staged to line A's second pod as an input |

## Contract

| | |
|---|---|
| Line A, the teacher | **`p2_t_teach`, hot:** `--config configs/p2_t_teach_hot.yaml --seed 43` (48 sims, **2e-5**, teacher 0.25; seed-43 fingerprint `62fd3945…1573`; differs from `configs/p2_k_lr.yaml` in `teacher_frac` and `teacher_dir` only, `test_p2t_teach_configs_are_the_brief`), `--init runs/p2_s_cont/ckpt_iter0030.pt`, `--seed-pool` and `--seed-replay runs/p2_s_cont` (head 2,069,348, offset −30), 20 iterations. Then **`p2_t_teach_down`:** `--config configs/p2_t_teach_down.yaml --seed 43` (48, **2e-6**, teacher 0.25; `80de3169…5120`), `--init runs/p2_t_teach/ckpt_iter0020.pt`, `--seed-pool` and `--seed-replay runs/p2_t_teach`, 10 iterations |
| Line B, the same schedule without it | **`p2_t_ctl`, hot:** `--config configs/p2_k_lr.yaml --seed 43` (48, 2e-5; `816e88dc…2546`), the same start, pool and ring, 20 iterations. Then **`p2_t_ctl_down`:** `--config configs/p2_o_lr.yaml --seed 43` (48, 2e-6; `e2e479fb…90f2`), `--init runs/p2_t_ctl/ckpt_iter0020.pt`, from `runs/p2_t_ctl`, 10 iterations |
| Why this schedule | new targets at 2e-6 barely move the weights (RUNLOG-p2_r §6); every gain so far followed a step down after time at a higher rate. Both lines take the same 2e-5 → 2e-6 path, so K1 isolates the teacher. p2_s showed the schedule alone costs about 0.03 at 2e-5 and returns it after the step (RUNLOG-p2_s §6) |
| What differs between the lines | the teacher: its 256 rows per batch in place of 256 ring rows. Two things come with them: Stockfish's targets, and the start ring's positions seen again after the ring has turned over. The design does not separate the two (see the end). Start, seed, games, schedule, network and instrument are the same |
| Start reference | `p2_s_cont/0030`: six skill-5 reps (`runs/p2_s_cont/sf5/`, mean 0.3300), three skill-3 reps (0.7783), three skill-0 book and three initial-position reps; and its agreement with the teacher (T1), read once the held-out ring is pulled |
| Training | RunPod, secure RTX 4090, ≥ 24 vCPU, the image of p2_g–p2_s; variants recorded. If none is offered the registered launch is retried; a pod that never comes up is terminated and replaced (`cloud/launch_run.sh`); nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, `gpu hold`, nothing until each run's final pull, under the user units, `MEASURE_AFTER_PULL=1`. Drivers judged: `p2_t_teach` and `p2_t_ctl` `20`; `p2_t_teach_down` and `p2_t_ctl_down` `5 10` (ladder with ECE and N, three skill-0 reps; initial-position reps at each run's last). Then a finisher against `stockfish:5`: three reps at both lines' 0020 and down/0005; **six at both down/0010**; three against `stockfish:3` at the two endpoints; T1 at both lines' 0020 and endpoints |
| Lifecycle | four guards. `p2_t_teach` deadline 10 h, worst $8.90 (labelling then 20 iterations); `p2_t_ctl` 8 h, $7.12; each step down 6 h, $5.34. **Worst case $26.70**; expected about **$11** (about 4.6 h, 3.4 h, 2.0 h and 2.0 h at $0.89/h). Both hot stages launch together; each step down launches after its own hot stage's final pull |

## Registered expectations

### L — the continuity applied
- **L1.** Each run's `provenance.json`: the `init_from`, `seed_replay` head and offset above; `seed` 43; `git_dirty`
  false; `config_fingerprint` as above, equal to its `config.yaml`'s sha256; `lr` on every row 2e-5 (hot) or 2e-6
  (down); simulations per position 48.0.
- **L2.** Book positions in each final ring: reported, no bound.
- **L3, the teacher as registered.** `runs/p2_t_teach/teacher/teacher.json`: 472,229 rows, Stockfish 16, 100,000
  nodes, `multipv` 5, T 50; selection counts as above; the held-out ring 5,000 rows on the training fit. The
  outcome model's `a` and `s` are reported. No bound is set on them; a fit that calls fewer than 5% or more than
  60% of level positions drawn is reported as a defect before training is read.

### T1 — the lever applied: agreement with the teacher on the held-out 5,000
`scripts/teacher_agreement.py`: the share of positions where the network's prior (no search) puts its top move on
Stockfish's best, and its cross-entropy against the teacher's policy. At `p2_s_cont/0030`, both 0020s and both
endpoints. **T1 holds if line A's endpoint agrees more often than line B's and has the lower cross-entropy.** T1 is
not a strength claim. It says whether the network moved toward the teacher; without it, K1 cannot be read as a
test of distillation.

### K1 (primary) — the two lines at the end, skill 5
`p2_t_teach_down/0010` − `p2_t_ctl_down/0010`, against `stockfish:5`, **six reps per side**, per-opening rep-means,
paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed 0;
`runs/p2_t_teach_down/sf5/p2t_verdict_teach_vs_ctl.json`), read with the exact-zero rule. **Expected: the 95% CI
entirely above zero.** Forecast point +0.03 to +0.10.

### K2 — mate-in-2
At the endpoints: **`p2_t_teach_down/0010` solves at least as many of the 200 as `p2_t_ctl_down/0010` minus 4**, and
**both ≥ 183**.

### K3 — each line against the start (informational)
At skill 5: both 0020s and both down/0005s (three reps), both endpoints (six), each against `p2_s_cont/0030`'s six.
At skill 3: the endpoints' three reps against the start's three. Skill 0, M4 and the initial position as before.

### ECE, H
- **ECE:** < 0.15 at the endpoints, both lines. The teacher's value target is the part of this round most likely to move
  it; a break is handled in PREREG-p2_l's order.
- **H1–H10** as PREREG-p2_r, per run, direction-free. **H7:** the policy and value losses of line A against line B;
  the batch mixes ring and teacher rows, so line A's losses are not comparable to earlier rounds'. **H2:**
  `skipped_steps` < 20 per line. **H11:** labelling time and workers on line A's first pod.

## Predictions (written before launch)
- L3: the fit calls 15–35% of level positions drawn; `s` 100–250 cp.
- T1 holds, with line A's top-1 agreement at least 5 points above line B's.
- K1 passes, point +0.03 to +0.10; K2 passes, line A at or above line B.
- K3: both 0020s near or below the start (the cost of 2e-5); line A above the start at its endpoint, line B within noise.
- ECE 0.08–0.14; N 185–195; L2 0–5 per run.

## Decision rules

| K1 | K2 | T1 | Decision |
|---|---|---|---|
| **CI entirely above zero** | passes | holds | **Distillation works.** The teacher becomes standing (0.25, this schedule). p2_u's lever is Michael's choice of more of it: a fresh teacher ring from the new endpoint's own positions, a larger dose, or a deeper engine |
| CI entirely above zero | fails | | The teacher stays off until the mate-in-2 cost is diagnosed |
| **CI includes zero** | any | holds | **The network moved toward the teacher and gained nothing measurable at this dose.** p2_u's lever is Michael's choice: a larger dose (0.5, more positions), the 2e-7 step-down, the public benchmark |
| CI includes zero | any | fails | **The lever was not applied**: the network did not move toward the teacher. Diagnosed (dose, rate) before p2_u; not read as "distillation does nothing" |
| **CI entirely below zero** | any | any | **The teacher costs skill.** Diagnosed before p2_u; the value scale is the first suspect |
| N < 183 or ECE ≥ 0.15 at an endpoint | | | The broken guard first |

## Measurement protocol

```bash
# Before launch, on this box: T1 at the start (the held-out ring does not exist yet, so this runs after the
# teacher's final pull, on the same p2_s_cont/0030 checkpoint; it is a reading of the start, not of either line).

# Both hot stages together (cloud/launch_run.sh: launch, wait for a runtime, stage, sha256, start, enable the driver).
# inputs: runs/p2_s_cont/ckpt_iter00{11..30}.pt, state.json, config.yaml, replay/, the mate data, phase1_anchor.pt
cloud/launch_run.sh p2tt 10 p2_t_teach "20" 20 START_REPS START_STARTPOS \
  "python3 scripts/label_teacher.py --ring runs/p2_s_cont/replay --out runs/p2_t_teach/teacher \
     --holdout-out runs/p2_t_teach/teacher_holdout --holdout 5000 \
   && python3 scripts/train_phase2.py --config configs/p2_t_teach_hot.yaml --seed 43 \
     --init runs/p2_s_cont/ckpt_iter0030.pt --seed-pool runs/p2_s_cont --seed-replay runs/p2_s_cont \
     --out runs/p2_t_teach --iterations 20" INPUTS...
cloud/launch_run.sh p2tc 8 p2_t_ctl "20" 20 START_REPS START_STARTPOS \
  "python3 scripts/train_phase2.py --config configs/p2_k_lr.yaml --seed 43 --init runs/p2_s_cont/ckpt_iter0030.pt \
     --seed-pool runs/p2_s_cont --seed-replay runs/p2_s_cont --out runs/p2_t_ctl --iterations 20" INPUTS...

# Each step down after its hot stage's final pull (WAIT_FOR=runs/<hot>/.remote_done), 6 h each:
#   p2_t_teach_down: configs/p2_t_teach_down.yaml --init runs/p2_t_teach/ckpt_iter0020.pt, from runs/p2_t_teach,
#     inputs including runs/p2_t_teach/teacher/;  p2_t_ctl_down: configs/p2_o_lr.yaml from runs/p2_t_ctl.

# After all four: the finisher's reps, each verdict against the start, T1, and K1:
.venv/bin/python scripts/eval_reps.py verdict --a ctl runs/p2_t_ctl_down/sf5/reps_iter0010_r*.jsonl \
    --b teach runs/p2_t_teach_down/sf5/reps_iter0010_r*.jsonl \
    --out runs/p2_t_teach_down/sf5/p2t_verdict_teach_vs_ctl.json
```

## What would make p2_t uninterpretable
- A difference between the lines beyond the teacher (the config tests and L1).
- The teacher not as registered (L3), or the lever not applied (T1; its own row above).
- A teacher position that is a gate position (the exclusion is enforced at selection, and
  `test_the_labeller_excludes_every_measured_suite` guards that every suite is excluded).
- A launch from anything but clean, pushed `main`; a measurement-instrument change.
- Losing a run's final pull: re-run by addendum, not read from what is left.

**What the design does not separate.** Line A's teacher rows are positions from the start's ring, seen again after the
ring has turned over. Line B sees them only until it does, about ten iterations. A gain could in principle come from the
old positions rather than from Stockfish's targets on them. That alternative is weak: the old positions in line A carry
only Stockfish's targets, not the search's. Separating the two would take a third line, with the same positions
and their original targets; it is not run here.

**What the statistic does not cover.** As PREREG-p2_k: each line is one run; K1 resamples openings.

---

## Sign-off

Checked for the draft, 2026-11-07, on this box: `runs/p2_s_cont`: iteration 30, step 12,000, seed 43, 0 skipped
steps, pool `ckpt_iter0011`–`0030` (all present), ring head 2,069,348 equal to `state.json`; its six skill-5, three
skill-3, three skill-0 and three initial-position reps present. `configs/p2_t_teach_{hot,down}.yaml` differ from
`configs/p2_k_lr.yaml` and `configs/p2_o_lr.yaml` in `teacher_frac` and `teacher_dir` only (test passes). Seed-43
fingerprints at `8b56ccb`: `62fd3945e77a5ab81dd655db37fa94486f6bcb6e8dfac614c157acd0009f1573` (teach hot),
`80de31690b6d41e872894ce17b8d90fb7f8b5670cd55fc1b39727a83348e5120` (teach down),
`816e88dcdeb092010f86cac844c4a9ab644a0e29e19e6b23a36f40c6c3972546` (ctl hot),
`e2e479fbce7d0198067e48ab159de21a10461d0aaeca2ae42f220485511390f2` (ctl down). `label_teacher.py` run on 400 of
the ring's positions on this box (2 workers, 22 s): the selection counts above, the rings written and read back,
`teacher_agreement.py` read the held-out ring. `make test` 1,060 passed, 2 skipped. Read for the draft: RunPod
balance $18.69 (auto-reload at $10), no pods of ours; secure RTX 4090 at $0.89/h (stock "Medium"); each pod's worst
case passes the $15 ceiling and the $2 margin. Disk 19 GB free; the round adds about 7.5 GB.
