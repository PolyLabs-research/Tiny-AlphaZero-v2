# Pre-registration — p2_s: 128 simulations with a rate that can use them, then a step down, against the standing protocol

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-06** (see Sign-off), at Michael's instruction. Nothing under this file
> has run.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_r read "no effect seen at this length" (K1 −0.020 [−0.055, +0.015]), and RUNLOG-p2_r §6 explains why that
reading is narrow: the sims arm's policy loss climbed for about ten iterations as the ring filled with
128-simulation targets, leaving about 4,000 steps on them at 2e-6, where the weights barely move. Michael chose
(2026-11-06) the fair test: **the same lever at a rate that can use it, then a step down**, the shape every gain
so far has had (2e-4 → 2e-5 in p2_k; 2e-5 → 2e-6 in p2_o and p2_p).

**H_targets: self-play at 128 simulations, trained first at 2e-5 and then stepped down to 2e-6, ends stronger at
skill 5 than the standing protocol (48 simulations, 2e-6) over the same number of iterations, without costing
mate-in-2.**

The network stays at 6.5M parameters; that is the project's aim. The instrument stays where it was.

## Contract

| | |
|---|---|
| Line A, 128 simulations | **`p2_s_sims`, the hot stage:** `--config configs/p2_s_hot.yaml --seed 43` (128 sims, **2e-5**; resolved fingerprint `75bbaeb5…13eb`; differs from `configs/p2_r_sims.yaml` in `lr` only, `test_p2s_hot_config_is_the_brief`), `--init runs/p2_r_sims/ckpt_iter0020.pt`, `--seed-pool` and `--seed-replay runs/p2_r_sims` (head 1,582,883, offset −20), 20 iterations. Then **`p2_s_down`, the step down:** `--config configs/p2_r_sims.yaml --seed 43` (128 sims, **2e-6**; fingerprint `1c284171…6a08`), `--init runs/p2_s_sims/ckpt_iter0020.pt`, `--seed-pool` and `--seed-replay runs/p2_s_sims`, 10 iterations |
| Line B, the standing protocol | **`p2_s_cont`:** `--config configs/p2_o_lr.yaml --seed 43` (48 sims, 2e-6; fingerprint `86fcc672…7d65`), `--init runs/p2_r_cont/ckpt_iter0020.pt`, `--seed-pool` and `--seed-replay runs/p2_r_cont` (head 1,549,359, offset −20), 30 iterations |
| Why these starts | Each line continues its own p2_r arm: line A's ring already holds 128-simulation targets, so about ten iterations of ring-filling are not paid for twice. Both lines descend from p2_q/0040 and reach the same count: 50 iterations and 20,000 training steps from there |
| What differs between the lines | the search budget (128 against 48) and, inside line A, twenty iterations at 2e-5 in place of 2e-6; p2_r's arm difference comes with the starts (−0.020 [−0.055, +0.015], within noise). Games per iteration, the seed, the network and the instrument are the same |
| Why two pods for line A | a pod's job, its driver and its self-deletion follow one run directory; the step down needs its own `--init`. Stage two launches after stage one's final pull |
| Start reference | **skill 5:** each line's own six reps at 20 (`runs/p2_r_sims/sf5/`, mean 0.3212; `runs/p2_r_cont/sf5/`, mean 0.3412). **skill 3:** their three reps. **skill 0:** their three book reps and three initial-position reps |
| Training | RunPod, secure RTX 4090, ≥ 24 vCPU, the image of p2_g–p2_r; variants recorded. If none is offered the registered launch is retried; a pod with no runtime after 20 minutes is terminated and the launch rerun (RUNLOG-p2_r §1); nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, `gpu hold`, nothing until each run's final pull, under the user units (`taz-campaign@`, `taz-finisher@`), with `MEASURE_AFTER_PULL=1`. Drivers judged: `p2_s_sims` `20`, `p2_s_down` `5 10`, `p2_s_cont` `20 30` (ladder with ECE and N, three skill-0 reps; initial-position reps at each run's last). Then a finisher against `stockfish:5`: three reps at `p2_s_sims/0020`, `p2_s_down/0005`, `p2_s_cont/0020`; **six at `p2_s_down/0010` and `p2_s_cont/0030`**; and three against `stockfish:3` at those two endpoints |
| Lifecycle | four guards. `p2_s_sims` deadline 14 h, worst $12.46; `p2_s_down` deadline 8 h, worst $7.12; `p2_s_cont` deadline 8 h, worst $7.12. **Worst case $26.70**; expected about **$18.5** (about 9.9 h, 5.2 h and 5.7 h at $0.89/h). Every run takes one pod at a time on its own launch |

## Registered expectations

### L — the continuity applied
- **L1.** Each run's `provenance.json`: the `init_from`, `seed_replay` head and offset above; `seed` 43; `git_dirty`
  false; `config_fingerprint` as above, equal to its `config.yaml`'s sha256; `lr` on every row 2e-5 (`p2_s_sims`) or
  2e-6 (`p2_s_down`, `p2_s_cont`); simulations per position 128.0 (`p2_s_sims`, `p2_s_down`), 48.0 (`p2_s_cont`).
- **L2.** Book positions in each final ring: reported, no bound.

### K1 (primary) — the two lines at the end, skill 5
`p2_s_down/0010` − `p2_s_cont/0030`, against `stockfish:5`, **six reps per arm**, per-opening rep-means, paired
bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed 0;
`runs/p2_s_down/sf5/p2s_verdict_down_vs_cont.json`), read with the exact-zero rule. **Expected: the 95% CI
entirely above zero.** Forecast point +0.02 to +0.08.

### K2 — mate-in-2
At the endpoints: **`p2_s_down/0010` solves at least as many of the 200 as `p2_s_cont/0030` minus 4**, and **both ≥
183**.

### K3 — each line against its own start (informational)
At skill 5: `p2_s_sims/0020`, `p2_s_down/0005` and `p2_s_cont/0020` (three reps), and the two endpoints (six). The
step's own size: `p2_s_down/0010` against `p2_s_sims/0020` (six against three). At skill 3: the two endpoints' three
reps. Skill 0, M4 and the initial position as before.

### ECE, H
- **ECE:** < 0.15 at the endpoints, both lines; a break is handled in PREREG-p2_l's order.
- **H1–H10** as PREREG-p2_r, per run, direction-free. **H7** is the reading behind this round: whether the policy
  loss falls at 2e-5 from p2_r_sims's 1.527, and what the step down does to it.
  **H2:** `skipped_steps` < 20 per line.

## Predictions (written before launch)
- K1 passes, point +0.02 to +0.08; K2 passes.
- K3: `p2_s_sims/0020` near or below its start (the noise of 2e-5 before the step), the step down gaining by
  `p2_s_down/0005`; `p2_s_cont` within noise of its start.
- H7: line A's policy loss falls during the hot stage.
- ECE 0.08–0.13; N 185–195; L2 0–5 per run.

## Decision rules

| K1 | K2 | Decision |
|---|---|---|
| **CI entirely above zero** | passes | **Better targets win.** 128 simulations becomes standing, with 2e-5 before each step down as its schedule; p2_t continues `p2_s_down/0010` at 128 and 2e-6 as the default round, judged at skill 5 (or the next rung, if it reads above 0.75 there) |
| CI entirely above zero | fails | 48 stays until the mate-in-2 cost is diagnosed |
| **CI includes zero** | any | **No gain from more search at this budget.** 48 stays, and 128 simulations is closed as a lever. p2_t's lever is Michael's choice (distillation, 2e-7, the public benchmark) |
| **CI entirely below zero** | any | **The schedule costs skill.** 48 and 2e-6 stay; diagnosed before p2_t |
| N < 183 or ECE ≥ 0.15 at an endpoint | | The broken guard first |

## Measurement protocol

```bash
# Stage line A's hot stage and line B together (a tested launch script committed to cloud/ at registration):
# launch, wait for a runtime, stage, sha256 on the pod, start the job, enable taz-campaign@RUN; taz-finisher@p2_s waits for all three
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 14 --tag p2ss --vcpu 24 --ram 32 --disk 60 --yes
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2sc --vcpu 24 --ram 32 --disk 60 --yes
# inputs: runs/p2_r_{sims,cont}/ckpt_iter00{01..20}.pt, state.json, config.yaml, replay/, the mate data, phase1_anchor.pt

python3 scripts/train_phase2.py --config configs/p2_s_hot.yaml --seed 43 --init runs/p2_r_sims/ckpt_iter0020.pt \
    --seed-pool runs/p2_r_sims --seed-replay runs/p2_r_sims --out runs/p2_s_sims --iterations 20
python3 scripts/train_phase2.py --config configs/p2_o_lr.yaml --seed 43 --init runs/p2_r_cont/ckpt_iter0020.pt \
    --seed-pool runs/p2_r_cont --seed-replay runs/p2_r_cont --out runs/p2_s_cont --iterations 30

# After p2_s_sims's final pull: the step down, on its own pod
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2sd --vcpu 24 --ram 32 --disk 60 --yes
# inputs: runs/p2_s_sims/ckpt_iter00{01..20}.pt, state.json, config.yaml, replay/, the mate data, phase1_anchor.pt
python3 scripts/train_phase2.py --config configs/p2_r_sims.yaml --seed 43 --init runs/p2_s_sims/ckpt_iter0020.pt \
    --seed-pool runs/p2_s_sims --seed-replay runs/p2_s_sims --out runs/p2_s_down --iterations 10

# After all three campaigns: the finisher's skill-5 and skill-3 reps, each verdict against its line's start, and K1:
.venv/bin/python scripts/eval_reps.py verdict --a cont runs/p2_s_cont/sf5/reps_iter0030_r*.jsonl \
    --b down runs/p2_s_down/sf5/reps_iter0010_r*.jsonl --out runs/p2_s_down/sf5/p2s_verdict_down_vs_cont.json
```

## What would make p2_s uninterpretable
- A difference between the lines beyond the search budget and the hot stage's rate (the config tests and L1).
- The lever not applied (simulations per position, L1).
- A launch from anything but clean, pushed `main`; a measurement-instrument change.
- Losing a run's final pull: re-run by addendum, not read from what is left.

**What the statistic does not cover.** As PREREG-p2_k: each line is one run; K1 resamples openings.

---

## Sign-off

Checked for the draft, 2026-11-06, on this box: `runs/p2_r_sims` and `runs/p2_r_cont`: iteration 20, step 8,000,
seed 43, 0 skipped steps, pool `ckpt_iter0001`–`0020` (all present), ring heads 1,582,883 and 1,549,359 equal to
`state.json`; each arm's six skill-5, three skill-3, three skill-0 and three initial-position reps present;
`configs/p2_s_hot.yaml` differs from `configs/p2_r_sims.yaml` in `lr` only (test passes), seed-43 fingerprint
`75bbaeb58b18812d7576c62c4925fe08b0cdda4949d5b7668f2981a626a313eb`. Read for the draft: RunPod balance $37.22
(auto-reload at $10), no pods of ours; secure RTX 4090 at $0.89/h (stock "Low"); each pod's worst case passes the $15
ceiling and the $2 margin. Disk 25 GB free; the round adds about 6 GB. Read at registration, 2026-11-06 14:54 UTC:
balance $37.22, no pods of ours, secure RTX 4090 at $0.89/h (stock "Low"); each pod's worst case passes the ceiling
and the margin, and the combined $26.70 leaves $10.52 before any reload. `cloud/launch_run.sh` (`45915fd`, tested against
a stub) carries the launches. `make test` passed on this commit's tree (1,018 passed, 2 skipped).

**Signed: the Claude Code session, at Michael's instruction ("Go", 2026-11-06, after the draft and its cost — expected
about $18.5, worst case $26.70 — were put to him).** The commit carrying this line is the registration.
