# Pre-registration — p2_u: the standing teacher on a ring relabelled from this round's start, against p2_t's ring as it stands

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: DRAFT 2026-11-09.** Not signed. Nothing under this file has run.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_t made the teacher standing: a quarter of every batch from self-play positions relabelled by Stockfish beat the
same schedule without it (K1 +0.050 [+0.014, +0.085], RUNLOG-p2_t §6). Its ring is `runs/p2_s_cont`'s positions,
two rounds old now, and the network that starts this round has already drawn each of its rows about 6.5 times
(3.07 million draws over 472,229 rows). Another round on it takes that to about 13. Michael chose (2026-11-09) the
first of RUNLOG-p2_t's three levers: **a fresh teacher ring from the new endpoint's own positions.**

**H_fresh: the standing teacher on a ring relabelled from this round's start ends stronger at skill 5 than the same
teacher on p2_t's ring, without costing mate-in-2.**

The answer sets how the standing teacher runs from here: relabelled every round (about 1.3–3.5 h of a pod's CPUs
each time), or reused until a lever needs a new one. The network stays at 6.5M parameters. The instrument stays where
it was.

## The two rings

| | Line A, "fresh" (the lever) | Line B, "stale" (the teacher as it stands) |
|---|---|---|
| Positions | **`runs/p2_t_teach_down`'s final ring** (head 1,039,731), the start's own: each distinct position once, minus the five measured suites' positions (513), rows that repeat a position of their game (4,102) and duplicates (14,689): **480,696 positions** (counted on this box, `teacher.select_rows`). A seeded 5,000 (seed 0) go to a held-out ring; **475,696** train | `runs/p2_s_cont`'s ring, as p2_t built it: **472,229** train, 5,000 held out (`runs/p2_t_teach/teacher/`, unchanged) |
| Labels | built on line A's first pod before training by `scripts/label_teacher.py` with p2_t's settings: Stockfish 16 (the image's apt `16-1build1`), 100,000 nodes, `multipv` 5, one thread, 64 MB hash cleared per position, policy softmax at T = 50 cp | as built in p2_t, same settings |
| Value target | `root_q` under the two-parameter outcome model **fitted to this ring's own game outcomes**, as p2_t's procedure does | p2_t's fit: a = 227.6, s = 316.7 (34.5% of level positions drawn) |
| Rows already drawn by the start network | none | about 6.5 each |
| Where it lives | `runs/p2_u_fresh/teacher/`, pulled with the run, staged to line A's second pod as an input | `runs/p2_t_teach/teacher/`, staged to both of line B's pods as an input |

Everything else is p2_t's standing teacher: **`train.teacher_frac` 0.25** (256 of each 1,024-row batch; the ring
takes 717 and the mate rehearsal 51), rows drawn as a ring row is, losses and trainer untouched. Over 30 iterations
each line draws 3.07 million teacher rows: about 6.5 passes over its ring.

## Contract

| | |
|---|---|
| Start | **`p2_t_teach_down/0010`**, p2_t's endpoint and the best six-rep skill-5 mean so far (0.360). Both lines: `--seed 43 --init runs/p2_t_teach_down/ckpt_iter0010.pt --seed-pool runs/p2_t_teach_down --seed-replay runs/p2_t_teach_down` (ring head 1,039,731, offset −10) |
| Line A, fresh | **`p2_u_fresh`, hot:** the labeller on `runs/p2_t_teach_down/replay` (`--stockfish /usr/games/stockfish`, PREREG-p2_t Addendum A), then `--config configs/p2_u_fresh_hot.yaml` (48 sims, **2e-5**, teacher 0.25 from `runs/p2_u_fresh/teacher`; seed-43 fingerprint `066043ca…d67d`; differs from `configs/p2_t_teach_hot.yaml` in `teacher_dir` only, `test_p2u_fresh_configs_are_the_brief`), 20 iterations. Then **`p2_u_fresh_down`:** `--config configs/p2_u_fresh_down.yaml` (48, **2e-6**; `036cd665…01da`), `--init runs/p2_u_fresh/ckpt_iter0020.pt`, `--seed-pool` and `--seed-replay runs/p2_u_fresh`, 10 iterations |
| Line B, stale | **`p2_u_stale`, hot:** `--config configs/p2_t_teach_hot.yaml` itself (2e-5, teacher 0.25 from `runs/p2_t_teach/teacher`; `62fd3945…1573`, as p2_t), the same start, pool and ring, 20 iterations. Then **`p2_u_stale_down`:** `--config configs/p2_t_teach_down.yaml` itself (2e-6; `80de3169…5120`), `--init runs/p2_u_stale/ckpt_iter0020.pt`, from `runs/p2_u_stale`, 10 iterations |
| Why this schedule | p2_t's, which is now the standing one: the teacher's lead formed at 2e-5 and the step down kept it (RUNLOG-p2_t §6, reading 1) |
| What differs between the lines | the teacher ring, and with it three things the design does not separate (see the end): positions from the start's own play rather than two rounds back; rows the start network has never drawn; and an outcome model refitted to those games. Start, seed, games, dose, schedule, network and instrument are the same |
| Start reference | `p2_t_teach_down/0010`: six skill-5 reps (`runs/p2_t_teach_down/sf5/`, mean 0.3600), three skill-3 reps (0.7925), three skill-0 book reps (0.9692) and three initial-position reps (0.9767); T1 on p2_t's held-out ring (top-1 0.3588, cross-entropy 2.370); T1 on line A's held-out ring, read once it is pulled |
| Training | RunPod, secure RTX 4090, ≥ 24 vCPU, the image of p2_g–p2_t; variants recorded. If none is offered the registered launch is retried; a pod that never comes up is terminated and replaced (`cloud/launch_run.sh`); nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, `gpu hold`, nothing until each run's final pull, under the user units, `MEASURE_AFTER_PULL=1`. Drivers judged: `p2_u_fresh` and `p2_u_stale` `20`; both step downs `5 10` (ladder with ECE and N, three skill-0 reps; initial-position reps at each run's last). Then a finisher (`runs/p2_u_finish.sh`) against `stockfish:5`: three reps at both lines' 0020 and down/0005; **six at both down/0010**; three against `stockfish:3` at the two endpoints; T1 on both held-out rings at both lines' 0020 and endpoints |
| Lifecycle | four guards. `p2_u_fresh` deadline 10 h, worst $8.90 (labelling then 20 iterations; labelling takes about 1.3 h on a 64-vCPU pod and about 3.5 h on a 24-vCPU one, at p2_t's 0.53 s per position per worker); `p2_u_stale` 8 h, $7.12; each step down 6 h, $5.34. **Worst case $26.70**; expected about **$13** (about 5.5 h, 4.5 h, 2.2 h and 2.2 h at $0.89/h). Both hot stages launch together; each step down launches after its own hot stage's final pull |

## Registered expectations

### L — the continuity applied
- **L1.** Each run's `provenance.json`: the `init_from`, `seed_replay` head and offset above; `seed` 43; `git_dirty`
  false; `config_fingerprint` as above, equal to its `config.yaml`'s sha256; `lr` on every row 2e-5 (hot) or 2e-6
  (down); simulations per position 48.0; `teacher_dir` as registered in each `config.yaml`.
- **L2.** Book positions in each final ring: reported, no bound.
- **L3, the fresh ring as registered.** `runs/p2_u_fresh/teacher/teacher.json`: 475,696 rows, Stockfish 16, 100,000
  nodes, `multipv` 5, T 50; selection counts as above; the held-out ring 5,000 rows on the training fit. The outcome
  model's `a` and `s` are reported. No bound is set on them; a fit that calls fewer than 5% or more than 60% of level
  positions drawn is reported as a defect before training is read. Reported at close, no bound: how many of line A's
  held-out positions are in line B's training ring, and how many of p2_t's held-out positions are in line A's.

### T1 — the lever applied: agreement with the teacher on line A's held-out 5,000
`scripts/teacher_agreement.py`: the share of positions where the network's prior (no search) puts its top move on
Stockfish's best, and its cross-entropy against the teacher's policy. At `p2_t_teach_down/0010`, both 0020s and both
endpoints, on line A's held-out ring; the same on p2_t's held-out ring, informational. **T1 holds if line A's
endpoint has the lower cross-entropy on line A's held-out ring.** Top-1 agreement is reported, not judged: in p2_t
the lines' endpoints differed by 0.6 points on it and by 0.43 nats on the cross-entropy (RUNLOG-p2_t §6, reading 2). T1 is not a strength
claim; it says whether the fresh ring moved the network on its own positions further than the old ring did.

### K1 (primary) — the two lines at the end, skill 5
`p2_u_fresh_down/0010` − `p2_u_stale_down/0010`, against `stockfish:5`, **six reps per side**, per-opening
rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed 0;
`runs/p2_u_fresh_down/sf5/p2u_verdict_fresh_vs_stale.json`), read with the exact-zero rule. **Expected: the point
above zero.** Six reps resolve about ±0.035 (p2_t's K1 half-width); a difference smaller than that will read as
"includes zero".

### K2 — mate-in-2
At the endpoints: **`p2_u_fresh_down/0010` solves at least as many of the 200 as `p2_u_stale_down/0010` minus 4**,
and **both ≥ 183**.

### K3 — each line against the start (informational)
At skill 5: both 0020s and both down/0005s (three reps), both endpoints (six), each against `p2_t_teach_down/0010`'s
six. At skill 3: the endpoints' three reps against the start's three. Skill 0, M4 and the initial position as before.

### ECE, H
- **ECE:** < 0.15 at the endpoints, both lines. A break is handled in PREREG-p2_l's order.
- **H1–H10** as PREREG-p2_r, per run, direction-free. **H7:** the policy and value losses of line A against line B;
  both mix ring and teacher rows, so here the lines are comparable with each other and with p2_t's line A. **H2:**
  `skipped_steps` < 20 per line. **H11:** labelling time and workers on line A's first pod.

## Predictions (written before launch)
- L3: the fit calls 20–40% of level positions drawn; `s` 250–400 cp (p2_t's came out at 317 against a forecast of
  100–250).
- T1 holds; top-1 agreement within 1.5 points between the lines.
- K1: point 0.00 to +0.04; more likely than not the CI includes zero. K2 passes.
- K3: both 0020s near or below the start (the cost of 2e-5); at the endpoints line A at or above the start, line B
  within noise of it.
- ECE 0.08–0.14; N 185–195; L2 0–5 per run.

## Decision rules

| K1 | K2 | Decision |
|---|---|---|
| **CI entirely above zero** | passes | **Relabel every round.** The standing teacher is rebuilt from each round's start ring before training. p2_v's lever is Michael's choice: a larger dose on a fresh ring (0.5), a deeper engine, the 2e-7 step-down, the public benchmark |
| CI entirely above zero | fails | The fresh ring stays off until the mate-in-2 cost is diagnosed |
| **CI includes zero** | any | **Relabelling gains nothing measurable at this dose.** It is not standing; a later round may reuse a ring or relabel as its lever needs. T1 says which reading: holds, the fresh ring moved the network on its own positions and it did not show at skill 5; fails, the two rings taught the network the same thing. p2_v's lever is Michael's choice from the same list |
| **CI entirely below zero** | any | **The fresh ring costs skill.** Diagnosed before p2_v; the refitted outcome model and the fresh ring's overlap with the carried self-play ring are the first suspects |
| N < 183 or ECE ≥ 0.15 at an endpoint | | The broken guard first |

## Measurement protocol

```bash
# Before launch, on this box: restore the start's league pool (the final pull excludes seed_*; the nine files are
# byte copies of runs/p2_t_teach/ckpt_iter0012..0020.pt, hard-linked as the names its state.json lists). Done for
# the draft; see Sign-off.

# All four launches: runs/p2_u_launch.sh (cloud/launch_run.sh each, as transient user units). Both hot stages together:
#   p2_u_fresh  (p2uf, 10 h): label_teacher.py --ring runs/p2_t_teach_down/replay --out runs/p2_u_fresh/teacher
#                --holdout-out runs/p2_u_fresh/teacher_holdout --holdout 5000 --stockfish /usr/games/stockfish
#                && train_phase2.py --config configs/p2_u_fresh_hot.yaml <start> --out runs/p2_u_fresh --iterations 20
#   p2_u_stale  (p2us, 8 h):  train_phase2.py --config configs/p2_t_teach_hot.yaml <start> --out runs/p2_u_stale
#                --iterations 20; inputs also runs/p2_t_teach/teacher/
#   <start> = --seed 43 --init runs/p2_t_teach_down/ckpt_iter0010.pt --seed-pool runs/p2_t_teach_down
#             --seed-replay runs/p2_t_teach_down
#   inputs: the start's 20 pool checkpoints, state.json, config.yaml, replay/, the mate data, phase1_anchor.pt
# Each step down after its hot stage's final pull (WAIT_FOR=runs/<hot>/.remote_done), 6 h each:
#   p2_u_fresh_down: configs/p2_u_fresh_down.yaml from runs/p2_u_fresh, inputs including runs/p2_u_fresh/teacher/
#   p2_u_stale_down: configs/p2_t_teach_down.yaml from runs/p2_u_stale, inputs including runs/p2_t_teach/teacher/

# After all four: runs/p2_u_finish.sh under taz-finisher@p2_u — reps, each verdict against the start, T1, and K1:
.venv/bin/python scripts/eval_reps.py verdict --a stale runs/p2_u_stale_down/sf5/reps_iter0010_r*.jsonl \
    --b fresh runs/p2_u_fresh_down/sf5/reps_iter0010_r*.jsonl \
    --out runs/p2_u_fresh_down/sf5/p2u_verdict_fresh_vs_stale.json
```

## What would make p2_u uninterpretable
- A difference between the lines beyond the teacher ring (the config test and L1).
- The fresh ring not as registered (L3).
- A teacher position that is a gate position (enforced at selection;
  `test_the_labeller_excludes_every_measured_suite`).
- A launch from anything but clean, pushed `main`; a measurement-instrument change.
- Losing a run's final pull: re-run by addendum, not read from what is left.

**What the design does not separate.** The fresh ring differs from the stale one in three ways at once: its
positions come from the start's own games, not from two rounds back; the start network has never drawn its rows,
while it has drawn each stale row about 6.5 times; and its outcome model is fitted to the newer games. The round
answers the working question (relabel each round or not), not which of the three matters. As in p2_t's line A, the
fresh ring's positions are also the carried self-play ring's positions, with Stockfish's targets in place of the
search's.

**What the statistic does not cover.** As PREREG-p2_k: each line is one run; K1 resamples openings.

---

## Sign-off

Checked for the draft, 2026-11-09, on this box: `runs/p2_t_teach_down`: iteration 10, step 4,000, seed 43, 0
skipped steps, ring head 1,039,731 equal to `state.json` and `replay/meta.json`. Its league pool lists 20 snapshots:
`ckpt_iter0000`–`0010` (present) and `seed_p2_t_teach_ckpt_iter0012`–`0020`, which the campaign pull excludes;
those nine were restored as hard links to `runs/p2_t_teach/ckpt_iter0012`–`0020.pt`, the files they were copied from
(`_copy_seed_pool` uses `shutil.copy2`), and the pool is now complete. Its six skill-5, three skill-3, three skill-0
and three initial-position reps present, with the means above. `configs/p2_u_fresh_{hot,down}.yaml` differ from
`configs/p2_t_teach_{hot,down}.yaml` in `teacher_dir` only (test passes). Seed-43 fingerprints:
`066043cad1786a4c86f8780f8915a5e958cf905ea9c65ac73a5f90fc3aa4d67d` (fresh hot),
`036cd665672a35348bff7ce876011f3350b2c5426780bdc95d1dafd6666e01da` (fresh down),
`62fd3945e77a5ab81dd655db37fa94486f6bcb6e8dfac614c157acd0009f1573` (stale hot) and
`80de31690b6d41e872894ce17b8d90fb7f8b5670cd55fc1b39727a83348e5120` (stale down), the last two equal to
`sha256sum runs/p2_t_teach{,_down}/config.yaml`. `label_teacher.py` on 400 of the start ring's positions (2 workers,
20 s): the selection counts above; `teacher_agreement.py` read its 40-row held-out ring at the start checkpoint. A
zero-iteration run on CPU with line B's hot config and the registered start (`--iterations 0`, into the session
scratchpad): pool seeded with all 20 snapshots, ring seeded with 500,000 rows from donor head 1,039,731 at offset
−10, fingerprint `62fd3945…1573`. RunPod balance $55.91 (auto-reload at $10), no pods of ours; secure RTX 4090 at
$0.89/h (stock "Low"); the launch dry run plans $8.90 for 10 h; each pod's worst case passes the $15 ceiling and the
$2 margin, and the two hot stages' $16.02 leaves $39.89. Disk 116 GB free after the closed rounds' checkpoints moved
to the external drive (2026-11-09); the round adds about 7.5 GB.
