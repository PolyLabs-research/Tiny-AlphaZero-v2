# Pre-registration — p2_i: the corrected value blend against a continuation arm, from p2_h/0060

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: DRAFT, 2026-10-27. Not registered.** Registration needs Michael's go and his OK on the
> cost below (worst case $14.24, two pods); the commit carrying a filled-in Sign-off line is the
> registration. Nothing under this file has run.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_h read **plateau** (RUNLOG-p2_h §6: K1 −0.045 [−0.093, +0.001] after sixty more iterations), so by
its registered row p2_i takes a lever from the open list as an A/B from p2_h/0060 against a
continuation arm. The open list: `value_q_blend: per_row`, the mate-in-2 capability residual, ECE, a
mate-in-2 lever.

**Why `per_row`.** ECE holds with room (0.0445 against 0.15). The residual's lever (proof-directed reply
expansion) is a search change, and search is also the instrument (ladder, suites, skill-0), so it cannot
be A/B'd without moving the ruler (REVIEW-2026-10-21 P2, P6). `per_row` is one config field, touches
training only, and is a correctness fix: since p2_c the code has halved the value cross-entropy of every
row without a search (the 5% mate-in-1/2 rehearsal rows), while every config comment and PREREG said
those rows keep it at full weight (REVIEW C1). It is also the open list's mate-in-2 lever: those rows
are the mate-in-1/2 positions.

**H_row: the corrected blend is not worse than the standing one** on skill-0 from the frozen book, and
does not cost mate-in-2. A non-inferiority question, honestly: a 5% change of value weight is not
expected to lift the plateau. If it holds, the code trains what the record has always said it trains.

## Contract

| | |
|---|---|
| Arms | **`p2_i_cont`**: `configs/p2_g.yaml` unchanged (fingerprint = p2_g's = p2_h's, `7563c9f7…`). **`p2_i_row`**: `configs/p2_i_row.yaml`, p2_g's with `train.value_q_blend: per_row` and nothing else (`tests/test_rehearsal.py::test_p2i_row_config_is_the_brief`) |
| Warm start | `runs/p2_h/ckpt_iter0060.pt`, both arms |
| Continuity | `--seed-pool runs/p2_h` (its league, 0041–0060), `--seed-replay runs/p2_h`: committed head 3,660,595 = durable head, so the window is its newest 500,000 rows, offset −60; both arms identical. The fresh AdamW and the seed streams restarting at iteration 1 are the standing protocol, the same in both arms |
| Length | `--iterations 20` per arm (the invocation, `HOW_TO_TRAIN.md`) |
| Start reference | p2_h/0060's three book reps on torch 2.14.1 (0.71, 0.685, 0.695; `runs/p2_h/reps_iter0060_r*.jsonl`) and its three initial-position reps (`runs/p2_h/startpos/`) |
| Training | RunPod, **two pods in parallel**, each a secure RTX 4090 with ≥ 24 vCPU, `runpod/pytorch:1.4.1-cu1281-torch2121-ubuntu2404` (p2_g's and p2_h's image, GPU class and cloud). If none is offered, nothing launches without Michael's OK on the alternative, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, under `gpu hold`, by `cloud/remote_campaign.sh`, one driver per arm: at iterations 10 and 20 the ladder pass and three skill-0 reps from the book; at 20 also three from the initial position; then **three more book reps at 20 per arm** (r4–r6), so the endpoint statistic has six reps per arm |
| Lifecycle | four guards (`cloud/README.md`); pod deadline 8 h from launch; worst case **$7.12 per pod, $14.24 together** at $0.89/h (the last price seen, 2026-10-25); expected about **$6** (20 iterations at p2_h's 8.6 min, plus boot and pulls) |

## Registered expectations

### L — the continuity applied
- **L1.** Each arm's `provenance.json`: `init_from` `runs/p2_h/ckpt_iter0060.pt`; `seed_replay` rows 500,000, `donor_committed` = `donor_head` = 3,660,595, offset −60; `git_dirty` false; `p2_i_cont`'s fingerprint = p2_g's; `p2_i_row`'s config differs from p2_g's in `value_q_blend` only.
- **L2.** Book positions in each arm's final ring: **reported, no bound.** p2_h stored 4 in 500,000 (the exclusion acts on starts, not on positions reached; RUNLOG-p2_h §2). Not fixed in this round, so as not to add a second change.

### K1 (primary) — non-inferiority on skill-0
`p2_i_row` − `p2_i_cont` at iteration 20, from the frozen book, **six reps per arm**, per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed 0). **Expected: the 95% CI's lower end above −0.05.** The bound is half of p2_g's twenty-iteration gain (+0.183 at 20 against its start); six reps put the interval's half-width near 0.03 (p2_h's three-rep paired SE was 0.022–0.024).

### K2 — mate-in-2
At iteration 20: **`p2_i_row` solves at least as many of the 200 as `p2_i_cont` minus 4**, and **both ≥ 183** (the standing N floor). Reported at 10 too. Expected: row ≥ cont.

### K3 — the second plateau reading (informational)
Each arm at 10 and 20 against the start reference, the standard three-rep statistic, and M4 (mean-over-reps book score with its CI above 0.675, RUNLOG-p2_g §4's method). No decision row turns on it; it reads whether the continuation stays flat.

### ECE, H
- **ECE** < 0.15 at 20, both arms.
- **H1–H6** as in PREREG-p2_h, both arms. `value_loss` is **not comparable between the arms**: under `per_row` the rehearsal rows' CE enters at full weight, so the logged number changes meaning; reported, not compared.

## Predictions (written before launch)
- K1 passes, point between −0.02 and +0.02.
- K2 passes; `p2_i_row` at or above `p2_i_cont` by 0–5 positions.
- K3: the continuation arm stays within noise of the start (the plateau holds); M4 met by neither arm at 20 or by both, not by one only.

## Decision rules

| K1 | K2 | Decision |
|---|---|---|
| passes | passes | **`per_row` becomes standing protocol**: the code then trains what every record since p2_c says it trains. The plateau stands either way; p2_j's lever is the owner's choice (off the list, the candidate with the best chance against a plateau at a constant learning rate is a learning-rate step-down) |
| passes | fails | `batch_mean` stays; the mate-in-2 cost is diagnosed before anything else |
| fails | any | `batch_mean` stays and the record says the corrected blend costs skill-0 (unexpected: diagnose before p2_j) |
| N or ECE breaks in either arm | | Standard stop rules |

## Measurement protocol

```bash
# Stage two pods (each: launch, wait-ssh, stage), the same inputs on both
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2ic --vcpu 24 --ram 32 --disk 60 --yes
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2ir --vcpu 24 --ram 32 --disk 60 --yes
bash cloud/stage_pod.sh POD runs/p2_h/ckpt_iter00{41..60}.pt runs/p2_h/state.json runs/p2_h/config.yaml \
    runs/p2_h/replay/ runs/data/mate1_100k_clean/ runs/data/mate2_50k_clean/ runs/phase1_anchor.pt

# On each pod (ssh redirects all three streams, so it returns)
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --config configs/p2_g.yaml \
   --init runs/p2_h/ckpt_iter0060.pt --seed-pool runs/p2_h --seed-replay runs/p2_h \
   --out runs/p2_i_cont --iterations 20" > /dev/null 2>&1 < /dev/null &
#   ... and on the other pod --config configs/p2_i_row.yaml --out runs/p2_i_row

# This box, one detached driver per arm, each started on its own
START_REPS='runs/p2_h/reps_iter0060_r*.jsonl' START_STARTPOS='runs/p2_h/startpos/reps_iter0060_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD_C p2_i_cont "10 20" 20 > /dev/null 2>&1 < /dev/null &
START_REPS='runs/p2_h/reps_iter0060_r*.jsonl' START_STARTPOS='runs/p2_h/startpos/reps_iter0060_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD_R p2_i_row "10 20" 20 > /dev/null 2>&1 < /dev/null &

# After both campaigns complete: three more book reps at 20 per arm (r1-r3 are skipped as already logged)
for arm in p2_i_cont p2_i_row; do
  gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/eval_reps.py run --ckpt runs/$arm/ckpt_iter0020.pt --anchor stockfish:0 --reps 6
done
.venv/bin/python scripts/eval_reps.py verdict --a cont runs/p2_i_cont/reps_iter0020_r*.jsonl \
    --b row runs/p2_i_row/reps_iter0020_r*.jsonl --out runs/p2_i_row/p2i_verdict_row_vs_cont.json
```

## What would make p2_i uninterpretable
- A second difference between the arms (the config test and L1 catch it).
- A launch from anything but clean, pushed `main`; a measurement-instrument change (every measurement on this box at torch 2.14.1).
- One arm on a different GPU class or image from the other: then the arm difference carries the hardware; reported if it happens.

---

## Sign-off

Checked for the draft, 2026-10-27, on this box: p2_h's `state.json` iteration 60, 0 skipped steps,
pool `ckpt_iter0041`–`0060` (all twenty present), `replay_total_written` 3,660,595 = the ring's
`total_written`; the six start-reference rep files present; `configs/p2_i_row.yaml` differs from
`configs/p2_g.yaml` in `value_q_blend` only (test passes). **Not checked:** the RunPod balance and the
current 4090 price (last seen $29.47 and $0.89/h, 2026-10-25); to be read at registration.

**Signed:** _(to be filled at registration, with Michael's go and cost OK)_
