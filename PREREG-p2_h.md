# Pre-registration — p2_h: p2_g continued, sixty more iterations, trained on RunPod

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-10-25** (see Sign-off), at Michael's instruction. Nothing
> under this file has run.

**Amendment policy.** A rule may be added after launch *only* for a measurement
that has not yet produced any data, and only as a dated addendum recording the run
state at the time of writing. A rule may never be added, weakened, or reworded for
a measurement already taken.

---

## The question

p2_g ended still climbing. Against its start, its six judged checkpoints read +0.130,
+0.183, +0.181, +0.181, +0.223, +0.243: a plateau from 20 to 40, then a second rise at
50 and 60 (RUNLOG-p2_g §3). Whether that rise continues or 0060 is near what this
protocol reaches is the cheapest open question with the best chance of a gain, and it
decides how every later lever is tested: against a continuation, or against a plateau.

**H_cont: sixty more iterations of p2_g's protocol, with nothing changed, improve
skill-0 from the frozen book.**

This is not on RUNLOG-p2_g §6's open list. Michael chose it over that list
(2026-10-25); the list's levers stay open.

## Contract

| | |
|---|---|
| Config | **`configs/p2_g.yaml`, unchanged**: zero deltas. The run's `config_fingerprint` must equal p2_g's, `7563c9f74f4d6231f0ef6688644ee1994d4a50426db1e1fdd5e714d8ab4abccc` (L1) |
| Warm start | `runs/p2_g/ckpt_iter0060.pt` |
| Continuity | `--seed-pool runs/p2_g` (its 20-checkpoint league, 0041–0060), `--seed-replay runs/p2_g`: p2_g's committed head is 3,611,615 = its durable head, so the window is its newest 500,000 rows, offset −60 |
| What differs from one uninterrupted 120-iteration run | Exactly what differed between p2_f and p2_g: `--init` starts a fresh AdamW (lr is constant at 2e-4, so no schedule restarts), and the per-iteration seed streams restart at iteration 1, so the opening draws repeat p2_g's. Both are the standing protocol, named so neither is mistaken for a lever |
| Start reference | p2_g/0060's three book reps **measured on torch 2.14.1** (0.7375, 0.7725, 0.715; mean 0.7417; `runs/p2_g_bridge214/`), per the bridge's decision (BRIDGE-2026-10-25 Results) |
| Training | RunPod, **secure RTX 4090** with ≥ 24 vCPU, `runpod/pytorch:1.4.1-cu1281-torch2121-ubuntu2404`: the image, GPU class and cloud p2_g trained on. If none is offered, nothing launches without Michael's OK on the alternative, recorded by addendum. Code from a git bundle of clean, pushed `main` (`cloud/stage_pod.sh`); the job runs under `cloud/pod_job.sh` |
| Measurement | this box, torch 2.14.1+rocm7.2, under `gpu hold`, by `cloud/remote_campaign.sh`: at iterations 10, 20, 30, 40, 50, 60 a ladder pass (`random,greedy,phase1,stockfish:0`, suites, ECE, book entropy; `role=protocol`) and three skill-0 reps from the book; at 60 also three from the initial position |
| Lifecycle | four guards (`cloud/README.md`); pod deadline 16 h from launch; the job waits up to 4 h for the final pull before deleting the pod; worst case $14.24 at $0.89/h |
| Resume | `--iterations` counts the invocation (`HOW_TO_TRAIN.md`) |

---

## Before launch

### Step 0 — the initial-position reference on this stack (this box)

p2_g/0060, three skill-0 reps from the initial position, on 2.14.1, into
`runs/p2_g_bridge214/startpos/`. No expectation is registered; it is K4's reference.
(2.12.1 read 0.7625, 0.74, 0.81; the bridge rule re-measures cross-stack references.)

No training-step bridge: the pod is the stack p2_g trained on. L1 is checked as soon
as `provenance.json` exists, before iteration 1 commits; if it fails, the run is
stopped and the pod terminated.

## Registered expectations

### L — the continuity applied

- **L1.** `runs/p2_h/provenance.json`: `config_fingerprint` = p2_g's (above);
  `init_from` `runs/p2_g/ckpt_iter0060.pt`; `seed_replay.rows` 500,000,
  `donor_committed` = `donor_head` = 3,611,615, `iteration_offset` −60; `git_dirty`
  false.
- **L2. Contamination check.** Rows of p2_h's final ring equal (squares, side,
  castling, en passant) to any of the book's 100 positions: **expected 0** (p2_g: 0).

### K1 (primary) — the endpoint

p2_h/0060 against the start reference, on the registered statistic (three reps,
per-opening rep-means, paired bootstrap over openings, 10,000 resamples). **Expected:
CI entirely above zero.** The forecast point is +0.05 to +0.10 (p2_g's points rose
+0.062 from 0040 to 0060); it is a forecast, not a bar.

### K2 — the trajectory

The six judged checkpoints against the start. **Expected: no dip** (no checkpoint's
CI entirely below zero; p2_g had none after its own restart). The six points and
their least-squares slope per ten iterations are reported, and so is the 120-iteration
trajectory p2_g + p2_h. p2_g's six points were measured on 2.12.1, so that combined
curve carries the bridge's ~±0.04 resolution at the seam.

### K3 — M4 holds

Mean-over-reps book score with its CI entirely above 0.675 at 0060, by the method
RUNLOG-p2_g §4 defined (per-opening rep-means bootstrapped over the 100 openings,
10,000 resamples, seed 0, percentile). **Expected: holds** (start 0.742).

### K4 — informational: the initial position

p2_h/0060 against p2_g/0060 (Step 0), three reps each from the initial position,
both on 2.14.1. Reported with its CI. No decision row turns on it.

### N, ECE

- **N:** mate-in-2 ≥ 0.915 (≥ 183/200) at 0060; reported at every judged checkpoint.
  **Expected: holds, narrowly**: the start reads 184/200 on both stacks, and p2_g's six
  passes read 184–190.
- **ECE:** < 0.15 at 0060. **Expected: holds**, no direction (p2_g: 0.040–0.059).

### H (reported)

- **H1.** `move1_prior_entropy` (initial-position games only) < 0.05 at iteration 60
  (p2_g rose from 0.004 to 0.026). Direction registered: the collapse persists.
- **H2.** `skipped_steps` summed < 30 over 60 iterations (p2_g: 0 in 24,000).
- **H3–H5.** `value_loss` by halves, `decisive_rate`, `promo_loss`: direction-free.
- **H6.** `book_move1_entropy`: direction-free (p2_g 1.435 → 0.903).

## Decision rules

| K1 | Decision |
|---|---|
| **CI entirely above zero** | **H_cont supported: still climbing.** Continuation stays the default round. Any lever from the open list is tested as an A/B from p2_h/0060 against a continuation arm, not against a fixed reference. |
| **CI includes zero** | **Plateau** at about p2_g/0060's strength under this protocol. p2_i takes a lever from the open list (`value_q_blend: per_row`, the residual, ECE, a mate-in-2 lever) as an A/B from p2_h/0060 against a continuation arm. |
| **CI entirely below zero** | **Regressed.** Stop; nothing launches until it is diagnosed. The suspects, named now: the restart (fresh AdamW moments, seed streams from iteration 1) and the training pod. |
| **N or ECE breaks at 0060** | Whatever K1 reads, p2_i addresses the broken guard first and the round's gain is not taken as standing. |

K2's dip, if any, and K4 add their readings to whichever row applies.

## Measurement protocol

```bash
# Step 0 (this box)
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_g/ckpt_iter0060.pt \
    --anchor stockfish:0 --reps 3 --openings tests/suites/openings_startpos.epd \
    --out-dir runs/p2_g_bridge214/startpos

# Launch and stage the pod
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 16 \
    --tag p2h --vcpu 24 --ram 32 --disk 60 --yes
bash cloud/stage_pod.sh POD runs/p2_g/ckpt_iter00{41..60}.pt runs/p2_g/state.json \
    runs/p2_g/config.yaml runs/p2_g/replay/ runs/data/mate1_100k_clean/ \
    runs/data/mate2_50k_clean/ runs/phase1_anchor.pt

# The run (on the pod, under the job guard; the ssh that starts it redirects all
# three streams, so it returns instead of blocking until the pod is gone)
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --config configs/p2_g.yaml \
   --init runs/p2_g/ckpt_iter0060.pt --seed-pool runs/p2_g --seed-replay runs/p2_g \
   --out runs/p2_h --iterations 60" > /dev/null 2>&1 < /dev/null &

# This box, detached, started on its own (never chained after the ssh above)
START_REPS='runs/p2_g_bridge214/reps_iter0060_r*.jsonl' \
START_STARTPOS='runs/p2_g_bridge214/startpos/reps_iter0060_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD p2_h "10 20 30 40 50 60" 60 \
    > /dev/null 2>&1 < /dev/null &
```

Verdicts land in `runs/p2_h/p2_verdict_reps_iter00{10..60}.json` (and
`p2_verdict_reps.json` for 0060) and K4 in `runs/p2_h/startpos/p2_verdict_reps.json`.
Every record is reachable by git (`.gitignore`'s `runs/**` negations).

## What would make p2_h uninterpretable

- **A config delta.** L1's fingerprint check catches any.
- **A launch from anything but a clean, pushed `main`.**
- **A measurement-instrument change**: every measurement runs on this box at
  torch 2.14.1 and the instrument code the bridge measured (`3ebd31e`).
- **Losing the final pull** (this box down past the 4 h wait): checkpoints are pulled
  every five minutes, so the judged ones survive; the ring would not. Reported if it
  happens.

---

## Sign-off

Checked before signing, 2026-10-25:

- p2_g's state and ring meta: iteration 60, committed = head = 3,611,615, pool
  `ckpt_iter0041`–`0060`, 0 skipped steps (`runs/p2_g/state.json`, `replay/meta.json`);
- the start reference: `runs/p2_g_bridge214/p2_verdict_reps.json` (reps 0.7375,
  0.7725, 0.715) and its ladder row (mate-in-2 184, ECE 0.0402);
- the config fingerprint: `runs/p2_g/provenance.json`;
- the pod: secure RTX 4090 at $0.89/h, stock low; balance $37.42; per-pod ceiling
  $15 (`~/.config/taz-cloud/limits.json`), so 16 h × $0.89 = $14.24 passes the budget
  guard with $23.18 left, above the $5 margin.

**Signed: the Claude Code session, at Michael's instruction ("Continue p2_g" and
"RunPod 4090", 2026-10-25).** The commit carrying this line is the registration.
