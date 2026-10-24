# Pre-registration — p2_g: diverse self-play openings, one long round, trained on RunPod

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-10-24** (see Sign-off), at Michael's instruction. Nothing
> under this file has run.

**Amendment policy.** A rule may be added after launch *only* for a measurement
that has not yet produced any data, and only as a dated addendum recording the run
state at the time of writing. A rule may never be added, weakened, or reworded for
a measurement already taken.

---

## The question

The skill-0 instrument plays every game from a frozen book of 100 random 4-ply
openings (`tests/suites/openings_4ply.epd`). **None of those 100 positions appears
anywhere in p2_f's 500,000-row ring**, and self-play's openings are a funnel: across
the ring's 4,641 games, 2 distinct positions after one ply, 6 after two, 39 after
four, 285 after eight (RUNLOG-p2_f §6). It is the move-1 prior collapse (H1, every
round since p2_c) seen in the data. The network is measured exclusively on positions
from a distribution it does not train on.

**H_open: training from diverse openings improves skill-0 from the frozen book.** The
lever starts each self-play game k uniformly random legal plies in, k ~ U{0..8},
never at one of the book's positions, and searches and stores it from there.

Two things change with it, both chosen by Michael (2026-10-24) and both protocol:
**one long round** (60 iterations, judged on its trajectory every ten), and **training
on a RunPod GPU** with every measurement made on this box. Measurement therefore
stays on the instrument every earlier round used; training moves to a CUDA stack,
bridged in Step 0b.

## Contract

| | |
|---|---|
| Config | `configs/p2_g.yaml` = `runs/p2_f/config.yaml` plus exactly four deltas (`test_p2g_config_is_the_brief`): `selfplay.opening_random_plies: 8` (**the lever**), `selfplay.actor_processes: 20` (throughput only; the ring is the same games, `tests/test_actors.py`), `eval.eval_every_n_iters: 0` (no inline ladder; measured locally), `phase2.iterations: 60` |
| Lever detail | `opening_exclude` = the skill-0 book: a start position in it, or a finished game, is redrawn. The draw has its own RNG stream (seeds and resignation flags unchanged); the move-1 statistics read games from the initial position only (k = 0, ~1/9) |
| Warm start | `runs/p2_f/ckpt_iter0020.pt` |
| Continuity | `--seed-pool runs/p2_f`, `--seed-replay runs/p2_f` (standing protocol since p2_f): p2_f's committed head is 1,556,008 = its durable head, so the window is its newest 500,000 rows, offset −20 |
| Start reference | p2_f/0020's three book reps (0.5275, 0.5075, 0.455; mean 0.4967), measured at `fe51064` |
| Training | RunPod, one pod for Step 0b and the run: secure RTX A4000 (or A5000) with ≥ 24 vCPU, `runpod/pytorch:1.4.1-cu1281-torch2121-ubuntu2404` (Python 3.12.3, torch 2.12.1+cu129, Stockfish 16-1build1). Code from a git bundle of clean `main` (`cloud/stage_pod.sh`); the job runs under `cloud/pod_job.sh` |
| Measurement | this box, ROCm, under `gpu hold`, by `cloud/remote_campaign.sh`: at iterations 10, 20, 30, 40, 50, 60 a ladder pass (`random,greedy,phase1,stockfish:0`, suites, ECE, book entropy; `role=protocol`) and three skill-0 reps from the book; at 60 also three skill-0 reps from the initial position (`tests/suites/openings_startpos.epd`) |
| Lifecycle | four guards (`cloud/README.md`); pod deadline 16 h from launch; the job waits up to 4 h for the final pull before deleting the pod; worst case $4.00 at $0.25/h |
| Resume | `--iterations` counts the invocation (`HOW_TO_TRAIN.md`) |

---

## Before launch

### Step 0a — the distribution check's baseline (this box)

p2_f/0020, three skill-0 reps from the initial position, into `runs/p2_f/startpos/`.
No expectation is registered; it is the reference for G4.

### Step 0b — stack bridge and shakedown (the pod), gates the launch

1. **Training-step bridge.** `scripts/stack_bridge.py` on p2_f/0020 and p2_f's ring,
   8 fixed batches, on the pod and on this box. **Pass:** every batch's total loss
   within 2% (relative) of this box's, and `grad_norm` within 10%.
2. **Search bridge.** `eval_ladder.py --anchors random` on p2_f/0020 on the pod.
   **Pass:** mate-in-2 within ±4 of this box's 189/200 (p2_f pass 20) and mate-in-3
   within ±6 of 140/200.
3. **Shakedown.** Two iterations of `configs/p2_g.yaml` into `runs/p2_g_shake` (not
   `runs/p2_g`). **Pass:** both commit; `provenance.json` records the lever and the
   500,000-row seed replay; no NaN abort; the iteration time is recorded. No strength
   is measured on it.

**Gate.** All three pass → p2_g launches on the same pod without asking again. Any
fails → nothing launches, the pod is terminated, and the failure is reported. The
shakedown's loss rows are seen before launch; no expectation below is about them.

## Registered expectations

### L — the lever and the continuity applied

- **L1.** `runs/p2_g/provenance.json`: `seed_replay.rows` 500,000,
  `donor_committed` = `donor_head` = 1,556,008, `iteration_offset` −20; config records
  `opening_random_plies: 8`.
- **L2. Contamination check.** Rows of p2_g's final ring equal (squares, side,
  castling, en passant) to any of the book's 100 positions: **expected 0** (p2_f: 0).
  The firewall is on start positions; a transposition into a book position is
  possible and would be reported by name.

### G1 (primary) — the endpoint

p2_g/0060 against the start reference, on the registered statistic (three reps,
per-opening rep-means, paired bootstrap over openings). **Expected: CI entirely above
zero; point ≥ +0.15.**

### G2 — the lever, against the ring-carried loop's own pace

p2_g/0020 against the start, read against p2_f's twenty-iteration gain on the same
statistic, **+0.089 [+0.038, +0.140]**:

- **adds:** p2_g/0020's point estimate above +0.140;
- **within:** in [+0.038, +0.140];
- **costs:** below +0.038.

**Expected: adds.** This is a directional comparison with a historical control, not
an A/B: the two rounds start from different checkpoints (0.4075 and 0.4967) and train
on different stacks. It is registered because it is the reading available; it cannot
attribute a difference to the lever alone.

### G3 — the trajectory

The six judged checkpoints against the start. **Expected: no dip** — no checkpoint's
CI entirely below zero (p2_f had none). The six point estimates and their
least-squares slope per ten iterations are reported.

### G4 — informational: is the gain only distribution-matching?

p2_g/0060 against p2_f/0020, three reps each, both from the initial position (Step 0a).
Reported with its CI. **Registered reading:** if G1 passes and G4's CI is entirely
above zero, the gain is not only the book's distribution; if G1 passes and G4's CI
includes zero or lies below, the RUNLOG says the measured gain is at least partly
distribution-matching. No decision row turns on G4 alone.

### M4 (armed)

At any judged checkpoint, the mean-over-reps book score with its CI entirely above
0.675 (PREREG-p2_e Addendum 2). Not expected: 0.50 + 0.15 falls short.

### N, ECE

- **N:** mate-in-2 ≥ 0.915 at 0060; reported at every judged checkpoint.
  **Expected: holds** (p2_f 0.945).
- **ECE:** < 0.15 at 0060. **Expected: holds**, no direction (p2_f's passes swung
  0.053–0.098).

### H (reported)

- **H1. Collapse at the initial position persists (direction registered).**
  `move1_prior_entropy` (initial-position games only) < 0.05 at iteration 60: the
  lever does not train the initial position more, it trains around it.
- **H2.** `skipped_steps` summed < 30 over 60 iterations (p2_f: 1 in 20).
- **H3–H5.** `value_loss` by halves, `decisive_rate`, `promo_loss`: direction-free.
- **H6. `book_move1_entropy`** (the prior's entropy at the book positions):
  direction-free — the book becomes in-distribution, which could sharpen or widen it.

## Decision rules

| G1 | G2 | Decision |
|---|---|---|
| **passes** | adds | **H_open supported.** Random-opening starts become standing for self-play. p2_h takes one lever from the open list (`value_q_blend: per_row`, the residual, ECE). |
| **passes** | within | The loop keeps turning on the long protocol; the lever is not shown to add. p2_h settles it with an A/B from p2_g/0060, ten iterations each, lever on and off, same stack. |
| **passes** | costs | The lever is suspected of costing pace. Same A/B as above, before anything else. |
| **flat or regressed** | any | Stop. The two suspects are the stack and the lever: an A/B from p2_f/0020, ten iterations each, lever on and off, on the pod, then the same on this box. |
| **N or ECE breaks** | — | Standard stop rules. |

G4 adds its registered reading to whichever row applies. Gates are untouched.

## Measurement protocol

```bash
# Step 0a (this box)
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_f/ckpt_iter0020.pt \
    --anchor stockfish:0 --reps 3 --openings tests/suites/openings_startpos.epd \
    --out-dir runs/p2_f/startpos

# Step 0b.1 bridge, this box (the pod runs the same with --device cuda there)
gpu hold --by tinyalphazero -- .venv/bin/python scripts/stack_bridge.py \
    --ckpt runs/p2_f/ckpt_iter0020.pt --replay runs/p2_f/replay \
    --config runs/p2_f/config.yaml --out runs/p2_g_bridge/bridge_local.json

# Launch and stage the pod, then on it: 0b.1, 0b.2, 0b.3
python scripts/runpod_pod.py launch --gpu "RTX A4000" --cloud SECURE --hours 16 \
    --tag p2g --vcpu 24 --ram 32 --disk 60 --yes
bash cloud/stage_pod.sh POD runs/p2_f/ckpt_iter00{01..20}.pt runs/p2_f/state.json \
    runs/p2_f/config.yaml runs/p2_f/replay/ runs/data/mate1_100k_clean/ \
    runs/data/mate2_50k_clean/ runs/phase1_anchor.pt

# The run (on the pod, under the job guard)
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --config configs/p2_g.yaml \
   --init runs/p2_f/ckpt_iter0020.pt --seed-pool runs/p2_f --seed-replay runs/p2_f \
   --out runs/p2_g --iterations 60" &

# This box, detached: pull, measure 10..60, final pull, release the pod
setsid nohup bash cloud/remote_campaign.sh POD p2_g "10 20 30 40 50 60" 60 &
```

Verdicts land in `runs/p2_g/p2_verdict_reps_iter00{10..60}.json` (and
`p2_verdict_reps.json` for 0060) against `runs/p2_f/reps_iter0020_r*.jsonl`, and G4 in
`runs/p2_g/startpos/p2_verdict_reps.json`. Every record is reachable by git
(`MUST_REACH`).

## What would make p2_g uninterpretable

- **A config delta beyond the four.** Guarded by `test_p2g_config_is_the_brief`.
- **Step 0b failing and the run going ahead anyway.** The gate forbids it.
- **A launch from anything but a clean, pushed `main`.**
- **A measurement-instrument change**: every measurement runs on this box at the
  code the ladder ran for p2_f, apart from the additive `--openings` flag.
- **Losing the final pull** (this box down past the 4 h wait): the run's checkpoints
  are pulled every five minutes, so the judged checkpoints survive; the ring would
  not, and p2_h would start on an empty one. Reported if it happens.

---

## Sign-off

Checked before signing, 2026-10-24:

- the evidence: 0/100 book positions in p2_f's 500,000 rows; 2 / 6 / 39 / 285
  distinct positions 1 / 2 / 4 / 8 plies into its 4,641 games (computed from the ring);
- the lever: off is byte-identical to the pre-actors reference fingerprint; on, 512
  draws give 408 distinct starts, k spread over 0..8, none in the book, none finished;
  in-process and parallel rings byte-identical with it on (`tests/test_actors.py`);
- the config: `test_p2g_config_is_the_brief` passes and catches a one-field mutant;
- the numbers: p2_f/0020 reps and verdicts (`runs/p2_f/`), p2_f's state and ring meta
  (committed = head = 1,556,008), the RunPod pod facts (`cloud/README.md`).

**Signed: the Claude Code session, at Michael's instruction ("b, on the lever, go with
whichever you think has the best chance of working"), 2026-10-24.** The commit carrying
this line is the registration.
