# Pre-registration — p2_k: a tenfold learning-rate step-down against a continuation arm, from p2_j/0040

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: DRAFT 2026-10-29 — not registered.** Nothing launches until Michael's cost OK; the
> registration commit flips this line and carries the sign-off.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_j read **plateau** against a six-rep start (RUNLOG-p2_j §6: K1 −0.027 [−0.062, +0.006]; the four
judged checkpoints flat, slope +0.001 per ten iterations). Its registered row names this round: the
learning-rate step-down, as an A/B from p2_j/0040 against a continuation arm — the candidate PREREG-p2_i
named as the one with the best chance against a plateau reached at a constant learning rate.

**Why tenfold, to 2e-5.** Phase 2 has trained at a constant 2e-4 throughout (`train.lr`; `--init`
starts a fresh AdamW at it each round). A step-down's usual effect at a plateau is quick — less
gradient noise around the point the network has settled at — and tenfold is the conventional step
(the AlphaGo Zero and AlphaZero schedules step by factors of ten). A smaller step risks an effect below
what this round can resolve: p2_i's arm-against-arm SE was 0.0155 with six reps per arm, so the
interval's half-width is about 0.03. 2e-5 is inside the range this network has trained at: phase 1's
schedule ended at 3e-5. AdamW's decoupled weight decay is applied as lr × `weight_decay`, so it steps
down with the rate; that is part of the lever, not a second one. The named risk: at a tenth of the
rate the network follows its own fresh self-play data more slowly, which could read as a cost.

**H_lr: a tenfold step-down from p2_j/0040 improves skill-0 from the frozen book over a continuation arm
within twenty iterations, without costing mate-in-2.**

## Contract

| | |
|---|---|
| Arms | **`p2_k_cont`**: `configs/p2_g.yaml` unchanged (fingerprint = p2_g's = p2_j's, `7563c9f7…`). **`p2_k_lr`**: `configs/p2_k_lr.yaml`, p2_g's with `train.lr: 2.0e-05` and nothing else (fingerprint `8ab654e2…298c`; `tests/test_rehearsal.py::test_p2k_lr_config_is_the_brief`) |
| Warm start | `runs/p2_j/ckpt_iter0040.pt`, both arms |
| Continuity | `--seed-pool runs/p2_j` (its league, 0021–0040), `--seed-replay runs/p2_j`: committed head 2,566,550 = durable head (`state.json`, `replay/meta.json`), so the window is its newest 500,000 rows, offset −40; both arms identical. The fresh AdamW and the seed streams restarting at iteration 1 are the standing protocol and the same in both arms, so unlike p2_j's reading the restart cancels between the arms |
| Length | `--iterations 20` per arm (the invocation, `HOW_TO_TRAIN.md`) |
| Start reference | p2_j/0040's **six** book reps on torch 2.14.1 (0.7625, 0.795, 0.7575, 0.725, 0.7925, 0.795; mean 0.7713; `runs/p2_j/reps_iter0040_r{1..6}.jsonl`) and its three initial-position reps (0.78, 0.8175, 0.8375; mean 0.8117; `runs/p2_j/startpos/`). Its ladder row at 40: mate-in-2 188, ECE 0.0604 |
| Training | RunPod, **two pods in parallel**, each a secure RTX 4090 with ≥ 24 vCPU, `runpod/pytorch:1.4.1-cu1281-torch2121-ubuntu2404` (the image, GPU class and cloud of p2_g–p2_j). The API lists one "RTX 4090" type; the 24 GB or 48 GB variant cannot be chosen and is recorded (`provenance.json` `vram_gb`). If no secure RTX 4090 is offered, nothing launches without Michael's OK on the alternative, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, under `gpu hold`, by `cloud/remote_campaign.sh` with **`MEASURE_AFTER_PULL=1`** (PREREG-p2_j Addendum A: nothing is measured until that arm's final pull, ring included, is complete), one driver per arm: at iterations 10 and 20 the ladder pass and three skill-0 reps from the book; at 20 also three from the initial position; then **three more book reps at 20 per arm** (r4–r6), so the endpoint statistic has six reps per arm |
| Lifecycle | four guards (`cloud/README.md`); pod deadline **8 h** from launch; worst case **$7.12 per pod, $14.24 together** at $0.89/h; expected about **$7.5** together (20 iterations at p2_j's 10.5 min = 3.5 h per pod, plus boot, staging and the final pull) |

## Registered expectations

### L — the continuity applied
- **L1.** Each arm's `provenance.json`: `init_from` `runs/p2_j/ckpt_iter0040.pt`; `seed_replay.rows`
  500,000, `donor_committed` = `donor_head` = 2,566,550, `iteration_offset` −40; `git_dirty` false;
  `p2_k_cont`'s fingerprint = p2_g's; `p2_k_lr`'s config differs from p2_g's in `lr` only.
- **L2.** Book positions in each arm's final ring, by `scripts/ring_book_rows.py`: **reported, no bound**
  (p2_j 2, p2_i_cont 2, p2_i_row 3, p2_h 4 in 500,000). Not fixed in this round, so as not to add a
  second change.

### K1 (primary) — the step-down against the continuation
`p2_k_lr` − `p2_k_cont` at iteration 20, from the frozen book, **six reps per arm**, per-opening
rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples,
seed 0; `runs/p2_k_lr/p2k_verdict_lr_vs_cont.json`). **Expected: the 95% CI entirely above zero.**
Forecast point +0.02 to +0.06; a forecast, not a bar. With a half-width near 0.03, a true gain much
below +0.03 will probably read as "includes zero": that reading is *no effect large enough to see in
twenty iterations*, not *no effect*.

### K2 — mate-in-2
At iteration 20: **`p2_k_lr` solves at least as many of the 200 as `p2_k_cont` minus 4**, and **both ≥
183** (the standing N floor). Reported at 10 too.

### K3 — each arm against the start (informational)
Each arm at 10 and 20 against the six-rep start, the standard three-rep statistic, and M4
(mean-over-reps book score with its CI above 0.675, RUNLOG-p2_g §4's method; six reps at 20). No
decision row turns on it; it reads whether the continuation stays flat a second time.

### K4 — informational: the initial position
Each arm at 20 against p2_j/0040's three initial-position reps, three reps each. Reported with its CI.

### ECE, H
- **ECE** < 0.15 at 20, both arms.
- **H1–H6** as PREREG-p2_h, both arms, direction-free. `value_loss` keeps its meaning across the arms
  (the loss is unchanged; only the step size moves), so the arms are compared by halves. **H1** is read
  with p2_j's divergence in mind: `move1_prior_entropy` rose 0.09 → 0.63 over p2_j while attempt 1 of the
  same protocol stayed 0.05–0.14 through 21 iterations (RUNLOG-p2_j §4).
  **H2:** `skipped_steps` summed < 10 per arm over 20 iterations.

## Predictions (written before launch)
- K1 passes, point +0.02 to +0.06.
- K2 passes; the arms within 4 positions of each other.
- K3: the continuation arm stays within noise of the start (the plateau holds a second time); the
  step-down arm reads above it at 20.
- The step-down arm's training losses (policy and value, by halves) read below the continuation arm's,
  and its `move1_prior_entropy` moves less over the twenty iterations.
- N 183–192, ECE 0.04–0.07 in both arms; L2 between 0 and 5 rows per arm.

## Decision rules

| K1 | K2 | Decision |
|---|---|---|
| **CI entirely above zero** | passes | **Step-down supported.** 2e-5 becomes standing protocol: p2_l continues `p2_k_lr/0020` at 2e-5 as the default round. A further step is a later question |
| CI entirely above zero | fails | 2e-4 stays until the mate-in-2 cost is diagnosed |
| **CI includes zero** | any | **No effect seen at this length.** 2e-4 stays; the plateau stands; p2_l's lever is Michael's choice from the open list. The record reports the point and whether a longer step-down arm is worth its cost |
| **CI entirely below zero** | any | **The step-down costs skill-0** at this length: 2e-4 stays, and the cost is diagnosed before p2_l (the named suspect: slower tracking of fresh self-play) |
| N or ECE breaks in either arm | | Standard stop rules |

## Measurement protocol

```bash
# Stage two pods (each: launch, wait-ssh, stage), the same inputs on both; check the inputs' sha256 on each pod
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2kc --vcpu 24 --ram 32 --disk 60 --yes
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2kl --vcpu 24 --ram 32 --disk 60 --yes
bash cloud/stage_pod.sh POD runs/p2_j/ckpt_iter00{21..40}.pt runs/p2_j/state.json runs/p2_j/config.yaml \
    runs/p2_j/replay/ runs/data/mate1_100k_clean/ runs/data/mate2_50k_clean/ runs/phase1_anchor.pt

# On each pod (ssh redirects all three streams, so it returns)
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --config configs/p2_g.yaml \
   --init runs/p2_j/ckpt_iter0040.pt --seed-pool runs/p2_j --seed-replay runs/p2_j \
   --out runs/p2_k_cont --iterations 20" > /dev/null 2>&1 < /dev/null &
#   ... and on the other pod --config configs/p2_k_lr.yaml --out runs/p2_k_lr

# This box, one detached driver per arm, each started on its own, measuring only after its arm's final pull
MEASURE_AFTER_PULL=1 START_REPS='runs/p2_j/reps_iter0040_r*.jsonl' START_STARTPOS='runs/p2_j/startpos/reps_iter0040_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD_C p2_k_cont "10 20" 20 > /dev/null 2>&1 < /dev/null &
MEASURE_AFTER_PULL=1 START_REPS='runs/p2_j/reps_iter0040_r*.jsonl' START_STARTPOS='runs/p2_j/startpos/reps_iter0040_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD_L p2_k_lr "10 20" 20 > /dev/null 2>&1 < /dev/null &

# After both campaigns complete (a finisher like runs/p2_i_finish.sh): three more book reps at 20 per arm
# (r1-r3 are skipped as already logged), then K1, M4 per arm, L2 per arm
for arm in p2_k_cont p2_k_lr; do
  gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/eval_reps.py run --ckpt runs/$arm/ckpt_iter0020.pt --anchor stockfish:0 --reps 6
  .venv/bin/python scripts/eval_reps.py m4 --rows $arm runs/$arm/reps_iter0020_r*.jsonl --out runs/$arm/p2_m4_iter0020.json
  .venv/bin/python scripts/ring_book_rows.py runs/$arm --json runs/$arm/book_rows.json
done
.venv/bin/python scripts/eval_reps.py verdict --a cont runs/p2_k_cont/reps_iter0020_r*.jsonl \
    --b lr runs/p2_k_lr/reps_iter0020_r*.jsonl --out runs/p2_k_lr/p2k_verdict_lr_vs_cont.json
```

## What would make p2_k uninterpretable
- A second difference between the arms (the config test and L1 catch it).
- A launch from anything but clean, pushed `main`; a measurement-instrument change (every measurement
  on this box at torch 2.14.1; `scripts/eval_reps.py` and `scripts/eval_ladder.py` as committed at
  registration).
- Losing an arm's final pull. With measurement deferred, this box only pulls while the pods train; if it
  is down past an arm's `PULL_WAIT`, that arm's checkpoints go with its pod (RUNLOG-p2_j §5). The round
  is then re-run by addendum, not read from one arm.
- The arms on different GPU variants: then the arm difference carries the hardware too. Recorded and
  reported if it happens, not a stop.

**What the statistic does not cover.** K1's bootstrap resamples openings over the measured reps; it does
not resample training runs. Each arm is one run. The one replicate on record — p2_j's two attempts of
the same protocol — agreed at 0010 within 0.007 (−0.0325 and −0.026) but diverged on H1 by 20, so an
arm difference near the interval's edge is read with that in mind.

---

## Sign-off

Checked for the draft, 2026-10-29, on this box: p2_j's `state.json` iteration 40, 0 skipped steps, pool
`ckpt_iter0021`–`0040` (all twenty present), `replay_total_written` 2,566,550 = the ring's
`total_written`; the six start-reference book reps and three initial-position reps present;
`configs/p2_k_lr.yaml` differs from `configs/p2_g.yaml` in `lr` only (test passes), fingerprint
`8ab654e2721decee87b6af33b3ea2c444bb19f8ecd1463f539234233895c298c`. Read for the draft: RunPod balance
$45.78 (the account also carries Michael's own pods, about $1/h, not this round's), no pods of ours;
secure RTX 4090 listed at $0.89/h (stock "Low"); per-pod ceiling $15 (`~/.config/taz-cloud/limits.json`),
so 8 h × $0.89 = $7.12 per pod passes, and $14.24 for both leaves $31.54, above the $5 margin. Price and
balance are read again at registration.

**Not signed.** Registration waits for Michael's OK on the round and its cost — expected about $7.5,
worst case $14.24.
