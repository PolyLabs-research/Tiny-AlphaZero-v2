# Pre-registration — p2_l: the default round at 2e-5, forty iterations from p2_k_lr/0020, judged on a stronger rung

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: DRAFT 2026-10-30 — not registered.** Nothing runs (Step 0 included) until Michael's cost OK;
> the registration commit flips this line and carries the sign-off.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_k's registered row (RUNLOG-p2_k §6): the tenfold step-down won (K1 +0.117 [+0.093, +0.140]), so 2e-5
is standing protocol and **p2_l continues `p2_k_lr/0020` at 2e-5 as the default round.**

**H_cont: forty more iterations at 2e-5 from p2_k_lr/0020, with nothing else changed, improve skill from
the frozen book.**

RUNLOG-p2_k §6 left two things this PREREG must settle before the round runs, so that neither is
discovered afterwards:

1. **The instrument is near its ceiling.** p2_k_lr/0020 scores 0.903 against Stockfish skill 0 from
   the book (12 losses in 200 on the ladder replay). The gains this round is looking for cannot be
   resolved there. **The endpoint moves to a stronger Stockfish rung, chosen before launch by a
   registered climb (Step 0); skill 0 stays as the continuity reading.**
2. **ECE is the guard nearest to breaking.** It rose 0.063 → 0.090 over p2_k_lr's second ten
   iterations (+0.0027 per iteration); continued, it passes 0.15 within this round. **The guard is
   unchanged (< 0.15 at 40, the same measurement as p2_g–p2_k); what a break decides, and
   in what order it is diagnosed, is registered below.** The named first suspect is the measurement's
   own sample: `evaluate.value_calibration` draws its positions from the ladder's phase1 and skill-0
   matchups, and the skill-0 matchup was chosen as "the measurement where the model mostly *loses*"
   (`evaluate.py`, the calibration-positions comment). It now wins 90% of it.

## Step 0 — the rung (this box, after registration, before launch)

Nothing trains while Step 0 runs: no pod is up, so the heavy local load cannot cost a pod's only copy
(RUNLOG-p2_j §5).

**0a. The climb.** `p2_k_lr/0020`, **one** book rep at each of `stockfish:1`, `stockfish:2`,
`stockfish:3`, `stockfish:5`, `stockfish:8` in that order (10,000 nodes, the standing anchor settings),
**stopping at the first skill whose rep scores ≤ 0.75.** That skill is **the rung, N**. If skill 8 still
scores above 0.75, N = 8 and that is reported. As PREREG-p2_b's climb, and binding for the same reason —
it selects the instrument's range, not the result:
- every climb rep is logged and committed (`runs/p2_k_lr/climb_s{k}/reps_iter0020_r1.jsonl`), the ones
  above 0.75 included;
- the climb is one-directional and stops at the first skill at or below 0.75; no descending, no
  continuing past it for a nicer number.

0.75 leaves a quarter of the scale above the start, and a skill at which the start still wins three in
four is not so strong that the round's checkpoints all read near zero.

**0b. The start reference at N.** **Six fresh** book reps of `p2_k_lr/0020` against `stockfish:N`
(`runs/p2_k_lr/sf{N}/reps_iter0020_r{1..6}.jsonl`). The climb's selecting rep is not one of them, so the
start reference is not the draw that chose the rung.

Step 0 is complete when the six files exist; then the pod launches. Its readings are reported in
RUNLOG-p2_l §1 whatever they are.

## Contract

| | |
|---|---|
| Config | **`configs/p2_k_lr.yaml`, unchanged**: zero deltas. The run's `config_fingerprint` must equal p2_k_lr's, `8ab654e2721decee87b6af33b3ea2c444bb19f8ecd1463f539234233895c298c` (L1) |
| Warm start | `runs/p2_k_lr/ckpt_iter0020.pt` |
| Continuity | `--seed-pool runs/p2_k_lr` (its league, 0001–0020), `--seed-replay runs/p2_k_lr`: committed head 1,555,800 = durable head (`state.json`, `replay/meta.json`), so the window is its newest 500,000 rows, offset −20 |
| What differs from one uninterrupted run | As every round since p2_g: `--init` starts a fresh AdamW (here at 2e-5) and the per-iteration seed streams restart at iteration 1. Standing protocol, named so neither is mistaken for a lever |
| Length | `--iterations 40` (the invocation, `HOW_TO_TRAIN.md`) |
| Start reference | **at the rung N:** p2_k_lr/0020's six reps from Step 0b. **at skill 0 (continuity):** its six book reps (0.88, 0.91, 0.92, 0.90, 0.875, 0.935; mean 0.9033; `runs/p2_k_lr/reps_iter0020_r{1..6}.jsonl`) and three initial-position reps (0.92, 0.94, 0.9425; mean 0.934; `runs/p2_k_lr/startpos/`). Its ladder row at 20: mate-in-2 189, ECE 0.0904 |
| Training | RunPod, **secure RTX 4090**, ≥ 24 vCPU, `runpod/pytorch:1.4.1-cu1281-torch2121-ubuntu2404` (the image, GPU class and cloud of p2_g–p2_k); the variant is recorded. If none is offered, the registered launch is retried; nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, under `gpu hold`, **nothing until the final pull**. (1) `cloud/remote_campaign.sh` with `MEASURE_AFTER_PULL=1`: at 10, 20, 30, 40 the ladder pass (`random,greedy,phase1,stockfish:0`, suites, **ECE**, book entropy; unchanged) and three skill-0 book reps; at 40 three skill-0 reps from the initial position. (2) then a finisher: at 10, 20, 30, 40 **three book reps against `stockfish:N`** (`runs/p2_l/sf{N}/`), and at 40 three more (r4–r6), so the endpoint has six reps against a six-rep start |
| Lifecycle | four guards (`cloud/README.md`); pod deadline **12 h** from launch; worst case **$10.68** at $0.89/h; expected about **$7.5** (40 iterations at p2_k's 11.4 min = 7.6 h, plus boot, staging and the final pull) |

## Registered expectations

### L — the continuity applied
- **L1.** `runs/p2_l/provenance.json`: `config_fingerprint` = p2_k_lr's; `init_from`
  `runs/p2_k_lr/ckpt_iter0020.pt`; `seed_replay.rows` 500,000, `donor_committed` = `donor_head` =
  1,555,800, `iteration_offset` −20; `git_dirty` false. Every iteration row's `lr` is 2e-5.
- **L2.** Book positions in the final ring, by `scripts/ring_book_rows.py`: **reported, no bound**
  (p2_k 1 and 2; p2_j 2).

### K1 (primary) — the endpoint, at the rung N
p2_l/0040 against p2_k_lr/0020, **both against `stockfish:N`, six reps each side**, per-opening
rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples,
seed 0; `runs/p2_l/sf{N}/p2_verdict_reps.json`). **Expected: the 95% CI entirely above zero.** Forecast
point +0.02 to +0.06: p2_k_lr took +0.110 of its +0.132 in its first ten iterations, so the step-down's
gain is front-loaded and the round expects the slower part of it. A forecast, not a bar.
`paired_rep_verdict` refuses rows from two anchors (this commit), so a skill-0 file cannot enter an N
verdict by a glob.

### K2 — the trajectory, at the rung N
The four judged checkpoints against the six-rep start at N, three reps each, the standard statistic.
**Expected: no dip** (no checkpoint's CI entirely below zero). The four points and their least-squares
slope per ten iterations are reported.

### K3 — skill 0, the continuity reading
At each judged checkpoint, the three skill-0 reps against p2_k_lr/0020's six (the statistic every round
since p2_d has reported) and M4 (bar 0.675). **Expected: M4 holds at every checkpoint.** No decision
row turns on skill 0's difference: near 0.90 it is compressed, and that is why the endpoint moved.

### K4 — informational: the initial position
p2_l/0040's three skill-0 reps from the initial position against p2_k_lr/0020's three. Reported.

### N, ECE, H
- **N:** mate-in-2 ≥ 183/200 at 40; reported at every judged checkpoint.
- **ECE:** < 0.15 at 40, the unchanged measurement; **reported at 10, 20, 30, 40 with its bins.** A
  reading ≥ 0.15 at 10, 20 or 30 is reported and stops nothing (measurement follows the final pull, so
  there is nothing left to stop); the guard is judged at 40.
- **H1–H6** as PREREG-p2_h, direction-free. **H7 (new, direction-free):** the training losses by
  quarters — p2_k_lr's value and policy losses rose while it played better (RUNLOG-p2_k §4).
  **H2:** `skipped_steps` summed < 20 over 40 iterations.

## Predictions (written before launch)
- Step 0: the climb stops at skill 2, 3 or 5.
- K1 passes, point +0.02 to +0.06.
- No dip (K2); M4 at skill 0 met at every checkpoint (K3).
- ECE keeps rising: 0.10–0.18 at 40. Whether it crosses 0.15 is this round's open question, which is
  why its consequence is registered.
- N 183–192; L2 0–5.

## Decision rules

| K1 (at N) | Decision |
|---|---|
| **CI entirely above zero** | **Still climbing at 2e-5.** Continuation at 2e-5 remains the default round; p2_m continues p2_l/0040, judged at N |
| **CI includes zero** | **Plateau at 2e-5.** p2_m is the next step-down (2e-6) as an A/B from p2_l/0040 against a continuation arm, judged at N |
| **CI entirely below zero** | **Regressed.** Stop; nothing launches until diagnosed. Suspects: the restart, ECE's rise, the pod |
| **N < 183 or ECE ≥ 0.15 at 0040** | Whatever K1 reads, the broken guard is addressed first and the round's gain is not taken as standing. **For ECE, in this order:** (1) the instrument — ECE of p2_l/0040 re-measured, offline and without training, on positions from its `stockfish:N` games, the matchup it does not mostly win; if that reads < 0.15, the calibration sample is named as the cause and the guard's measurement is revisited by its own PREREG; (2) only if it also reads ≥ 0.15, a training-side lever, A/B'd |

## Measurement protocol

```bash
# Step 0a, this box, nothing running on RunPod: the climb (stop at the first rep scoring <= 0.75)
for k in 1 2 3 5 8; do
  gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_k_lr/ckpt_iter0020.pt \
    --anchor stockfish:$k --reps 1 --out-dir runs/p2_k_lr/climb_s$k
  # read its score; stop here if <= 0.75 (N = k)
done
# Step 0b: the start reference at N
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_k_lr/ckpt_iter0020.pt \
  --anchor stockfish:N --reps 6 --out-dir runs/p2_k_lr/sfN

# Stage the pod (launch, wait-ssh, stage; inputs' sha256 checked on the pod)
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 12 --tag p2l --vcpu 24 --ram 32 --disk 60 --yes
bash cloud/stage_pod.sh POD runs/p2_k_lr/ckpt_iter00{01..20}.pt runs/p2_k_lr/state.json runs/p2_k_lr/config.yaml \
    runs/p2_k_lr/replay/ runs/data/mate1_100k_clean/ runs/data/mate2_50k_clean/ runs/phase1_anchor.pt

# On the pod (ssh redirects all three streams, so it returns)
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --config configs/p2_k_lr.yaml \
   --init runs/p2_k_lr/ckpt_iter0020.pt --seed-pool runs/p2_k_lr --seed-replay runs/p2_k_lr \
   --out runs/p2_l --iterations 40" > /dev/null 2>&1 < /dev/null &

# This box: the driver (skill 0, ladder, ECE), measuring only after the final pull
MEASURE_AFTER_PULL=1 START_REPS='runs/p2_k_lr/reps_iter0020_r*.jsonl' START_STARTPOS='runs/p2_k_lr/startpos/reps_iter0020_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD p2_l "10 20 30 40" 40 > /dev/null 2>&1 < /dev/null &

# After "campaign complete" (a finisher like runs/p2_k_finish.sh): the rung N
for n in 10 20 30 40; do
  nn=$(printf %04d $n); reps=3; [ $n = 40 ] && reps=6
  gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_l/ckpt_iter$nn.pt --anchor stockfish:N --reps $reps --out-dir runs/p2_l/sfN
  .venv/bin/python scripts/eval_reps.py verdict --a start runs/p2_k_lr/sfN/reps_iter0020_r*.jsonl \
    --b p2_l_$nn runs/p2_l/sfN/reps_iter${nn}_r{1,2,3}.jsonl --out runs/p2_l/sfN/p2_verdict_reps_iter$nn.json
done
.venv/bin/python scripts/eval_reps.py verdict --a start runs/p2_k_lr/sfN/reps_iter0020_r*.jsonl \
    --b p2_l_0040 runs/p2_l/sfN/reps_iter0040_r*.jsonl --out runs/p2_l/sfN/p2_verdict_reps.json
.venv/bin/python scripts/eval_reps.py m4 --rows p2_l_0040 runs/p2_l/reps_iter0040_r*.jsonl --out runs/p2_l/p2_m4_iter0040.json
.venv/bin/python scripts/ring_book_rows.py runs/p2_l --json runs/p2_l/book_rows.json
```

## What would make p2_l uninterpretable
- A config delta (L1 catches it). A launch from anything but clean, pushed `main`.
- A measurement-instrument change: every measurement on this box at torch 2.14.1; `scripts/eval_reps.py`,
  `scripts/eval_ladder.py` and `evaluate.paired_rep_verdict` as committed at registration (the anchor
  guard added with this draft refuses mixed rows and changes no number: p2_j's and p2_k's verdict files
  reproduce byte for byte under it).
- A climb that broke its rule (a rep dropped, a skill skipped or revisited).
- Losing the final pull: the round is re-run by addendum, as p2_j's was.

**What the statistic does not cover.** As PREREG-p2_k: the bootstrap resamples openings, not training
runs; the round is one run.

---

## Sign-off

Checked for the draft, 2026-10-30, on this box: p2_k_lr's `state.json` iteration 20, 0 skipped steps,
pool `ckpt_iter0001`–`0020` (all twenty present), `replay_total_written` 1,555,800 = the ring's
`total_written`; its six skill-0 book reps and three initial-position reps present; `configs/p2_k_lr.yaml`
fingerprint `8ab654e2…298c`. Read for the draft: RunPod balance $30.42 (the account also carries
Michael's own pods, about $1.1/h, not this round's), no pods of ours; secure RTX 4090 listed at $0.89/h
(stock "Low"); 12 h × $0.89 = $10.68 passes the $15 per-pod ceiling and leaves $19.74, above the $5
margin. Price and balance are read again at registration.

**Not signed.** Registration waits for Michael's OK on the round, the rung rule, the ECE rule and the
cost — expected about $7.5, worst case $10.68.
