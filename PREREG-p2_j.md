# Pre-registration — p2_j: p2_i_cont continued, forty more iterations, trained on RunPod

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: DRAFT 2026-10-27.** Awaiting Michael's cost OK; nothing under this file has run.
> Registration is the commit that changes this line.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that
has not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

RUNLOG-p2_h read p2_h/0060 as a plateau and p2_i was built on that reading. p2_i's continuation arm
overturned it: twenty more iterations of the standing protocol from p2_h/0060, with nothing changed,
read **+0.113 [+0.069, +0.155]** against their start at 20 and **0.798** over six book reps, the
highest book score measured on this stack (RUNLOG-p2_i §3, §6 item 1). PREREG-p2_h's first decision
row covers exactly this case: when the interval is entirely above zero, *continuation stays the
default round*. So p2_j is that round — not a lever — and it decides what p2_k is measured against.

**H_cont: forty more iterations of the standing protocol from p2_i_cont/0020, with nothing changed,
improve skill-0 from the frozen book.**

The lever named for a plateau (a learning-rate step-down) waits for a plateau that has actually been
read; the `per_row` lever is closed (RUNLOG-p2_i §6, FINDING-2026-10-27).

## Contract

| | |
|---|---|
| Config | **`configs/p2_g.yaml`, unchanged**: zero deltas. The run's `config_fingerprint` must equal p2_g's, `7563c9f74f4d6231f0ef6688644ee1994d4a50426db1e1fdd5e714d8ab4abccc` (L1) |
| Warm start | `runs/p2_i_cont/ckpt_iter0020.pt` |
| Continuity | `--seed-pool runs/p2_i_cont` (its league, 0001–0020), `--seed-replay runs/p2_i_cont`: committed head 1,551,304 = durable head (`state.json`, `replay/meta.json`), so the window is its newest 500,000 rows (iterations 11–20), offset −20 |
| What differs from one uninterrupted run | As every round since p2_g: `--init` starts a fresh AdamW (constant lr 2e-4, no schedule), and the per-iteration seed streams restart at iteration 1. Standing protocol, named so neither is mistaken for a lever. (RUNLOG-p2_i §6 item 1 names the restart as a candidate for the continuation arm's climb; this round does not separate it — a resume-vs-restart A/B is a later question, if the climb holds) |
| Length | `--iterations 40` (the invocation, `HOW_TO_TRAIN.md`) |
| Start reference | p2_i_cont/0020's **six** book reps on torch 2.14.1 (0.825, 0.7975, 0.805, 0.7625, 0.8225, 0.7775; mean 0.7983; `runs/p2_i_cont/reps_iter0020_r{1..6}.jsonl`) and its three initial-position reps (0.8475, 0.7975, 0.82; mean 0.8217; `runs/p2_i_cont/startpos/`). Its ladder row at 20: mate-in-2 187, ECE 0.0537 |
| Training | RunPod, **secure RTX 4090**, ≥ 24 vCPU, `runpod/pytorch:1.4.1-cu1281-torch2121-ubuntu2404` (the image, GPU class and cloud of p2_g–p2_i). The API lists one type, "RTX 4090", 24 GB; p2_i's two pods came as the 48 GB variant under that type and ran 10.7–10.9 min per iteration against p2_h's 8.6 on the 24 GB card. Whichever is served is recorded (`provenance.json` `vram_gb`); it is between rounds, not within one. If no secure RTX 4090 is offered, nothing launches without Michael's OK on the alternative, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, under `gpu hold`, by `cloud/remote_campaign.sh`: at iterations 10, 20, 30, 40 the ladder pass and three skill-0 reps from the book; at 40 also three from the initial position; then **three more book reps at 40** (r4–r6), so the endpoint has six reps against a six-rep start |
| Lifecycle | four guards (`cloud/README.md`); pod deadline **12 h** from launch; worst case **$10.68** at $0.89/h; expected about **$7** (40 iterations at p2_i's 10.8 min = 7.2 h, plus boot and pulls; $5.5 at p2_h's pace) |

## Registered expectations

### L — the continuity applied
- **L1.** `runs/p2_j/provenance.json`: `config_fingerprint` = p2_g's; `init_from`
  `runs/p2_i_cont/ckpt_iter0020.pt`; `seed_replay.rows` 500,000, `donor_committed` = `donor_head` =
  1,551,304, `iteration_offset` −20; `git_dirty` false.
- **L2.** Book positions in the final ring, by `scripts/ring_book_rows.py` (the committed counter;
  RUNLOG-p2_i §2): **reported, no bound** (p2_h 4, p2_i_cont 2, p2_i_row 3 in 500,000). The exclusion
  acts on starts, not on positions reached (RUNLOG-p2_h §2); fixing that is a later round's change.

### K1 (primary) — the endpoint
p2_j/0040 against the start reference, **six reps each side**, per-opening rep-means, paired bootstrap
over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed 0). **Expected: the 95%
CI entirely above zero.** Forecast point +0.03 to +0.08 (p2_i_cont rose +0.057 by 10 and +0.113 by 20
from its start; a climb that is slowing would land here). It is a forecast, not a bar.

### K2 — the trajectory
The four judged checkpoints against the start (three reps each, the standard statistic). **Expected:
no dip** (no checkpoint's CI entirely below zero). The four points and their least-squares slope per
ten iterations are reported, and the trajectory p2_h → p2_i_cont → p2_j against p2_h/0060.

### K3 — M4 holds
Mean-over-reps book score with its CI entirely above 0.675 at 40 (`scripts/eval_reps.py m4`,
RUNLOG-p2_g §4's method). **Expected: holds** (start 0.798 [0.777, 0.820]). Reported at every judged
checkpoint.

### K4 — informational: the initial position
p2_j/0040 against p2_i_cont/0020, three reps each from the initial position. Reported with its CI.

### N, ECE, H
- **N:** mate-in-2 ≥ 183/200 at 40; reported at every judged checkpoint. **Expected: holds** (start
  187; p2_i read 185–187).
- **ECE:** < 0.15 at 40. **Expected: holds** (start 0.0537).
- **H1–H6** as PREREG-p2_h. H1 direction-free this time: p2_i read `move1_prior_entropy` 0.071 at 20
  where p2_h had read 0.013, so "the collapse persists" is no longer the registered direction.
  **H2:** `skipped_steps` summed < 20 over 40 iterations.

## Predictions (written before launch)
- K1 passes, point +0.03 to +0.08; the slope per ten iterations is smaller than p2_i_cont's +0.056.
- No dip (K2). M4 met at every checkpoint. N 183–190, ECE 0.04–0.07.
- L2 between 0 and 5 rows.

## Decision rules

| K1 | Decision |
|---|---|
| **CI entirely above zero** | **H_cont supported: still climbing.** Continuation remains the default round; the lever list stays open and any lever is tested as an A/B against a continuation arm |
| **CI includes zero** | **Plateau**, read for the first time against a six-rep start. p2_k is the learning-rate step-down as an A/B from p2_j/0040 against a continuation arm (the candidate PREREG-p2_i named) |
| **CI entirely below zero** | **Regressed.** Stop; nothing launches until diagnosed. Suspects: the restart (named above), the pod |
| **N or ECE breaks at 0040** | Whatever K1 reads, the broken guard is addressed first and the round's gain is not taken as standing |

## Measurement protocol

```bash
# Stage the pod
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 12 --tag p2j --vcpu 24 --ram 32 --disk 60 --yes
bash cloud/stage_pod.sh POD runs/p2_i_cont/ckpt_iter00{01..20}.pt runs/p2_i_cont/state.json runs/p2_i_cont/config.yaml \
    runs/p2_i_cont/replay/ runs/data/mate1_100k_clean/ runs/data/mate2_50k_clean/ runs/phase1_anchor.pt

# On the pod (ssh redirects all three streams, so it returns)
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --config configs/p2_g.yaml \
   --init runs/p2_i_cont/ckpt_iter0020.pt --seed-pool runs/p2_i_cont --seed-replay runs/p2_i_cont \
   --out runs/p2_j --iterations 40" > /dev/null 2>&1 < /dev/null &

# This box, one detached driver, started on its own; the start reference has six reps
START_REPS='runs/p2_i_cont/reps_iter0020_r*.jsonl' START_STARTPOS='runs/p2_i_cont/startpos/reps_iter0020_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD p2_j "10 20 30 40" 40 > /dev/null 2>&1 < /dev/null &

# After "campaign complete": three more book reps at 40 (r1-r3 are skipped as already logged), then K1 and M4
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_j/ckpt_iter0040.pt --anchor stockfish:0 --reps 6
.venv/bin/python scripts/eval_reps.py verdict --a start runs/p2_i_cont/reps_iter0020_r*.jsonl \
    --b p2_j_0040 runs/p2_j/reps_iter0040_r*.jsonl --out runs/p2_j/p2_verdict_reps.json
.venv/bin/python scripts/eval_reps.py m4 --rows p2_j_0040 runs/p2_j/reps_iter0040_r*.jsonl
.venv/bin/python scripts/ring_book_rows.py runs/p2_j
```

The driver's own verdicts (three reps against the six-rep start) land in
`runs/p2_j/p2_verdict_reps_iter00{10..40}.json`; the registered K1 is the six-against-six file above.

## What would make p2_j uninterpretable
- A config delta (L1 catches it). A launch from anything but clean, pushed `main`.
- A measurement-instrument change: every measurement on this box at torch 2.14.1; `scripts/eval_reps.py`
  and `scripts/eval_ladder.py` as committed at registration.
- Losing the final pull: checkpoints are pulled every five minutes, so the judged ones survive; the ring
  would not. Reported if it happens.

---

## Sign-off

Checked for the draft, 2026-10-27, on this box: p2_i_cont's `state.json` iteration 20, 0 skipped steps,
pool `ckpt_iter0001`–`0020` (all twenty present), `replay_total_written` 1,551,304 = the ring's
`total_written`; the six start-reference book reps and three initial-position reps present; L2 counter
reproduces p2_h's four rows. Read at drafting, 2026-10-27 ~17:30 UTC: RunPod balance $22.67, no pods of
ours running; secure RTX 4090 at $0.89/h, stock low; per-pod ceiling $15 (`~/.config/taz-cloud/limits.json`),
so 12 h × $0.89 = $10.68 passes, and leaves $11.99, above the $5 margin.

**Not signed.** Registration follows Michael's cost OK.
