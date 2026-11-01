# Pre-registration — p2_n: the default round at 2e-5, forty iterations from p2_m_init/0020, judged on skill 3

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-01** (see Sign-off), at Michael's instruction. Nothing under this file
> has run.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_m's registered row (RUNLOG-p2_m §6): **the restart is not the lever at this length**; the standing
protocol is unchanged and **p2_n continues `p2_m_init/0020` at 2e-5 as the default round**, judged at
skill 3, with the 2e-6 step-down as its plateau row.

**H_cont: forty more iterations at 2e-5 from p2_m_init/0020, with nothing else changed, improve skill
from the frozen book at Stockfish skill 3.**

Written honestly, the record leans the other way. The gain over each round's start has fallen from
+0.132 (p2_k_lr, twenty iterations, skill 0) to +0.104 (p2_l, forty, skill 3) to +0.030–0.043 (p2_m,
twenty, skill 3), and twice now most of a round's gain was in place by its tenth iteration — with or
without a restart (RUNLOG-p2_m §6 item 1). This round is the default round the row names; its plateau
row is the step-down the record is pointing at. It also measures the early-gain shape properly: **six
reps at the first judged checkpoint as well as the last**, so "in place by 10" and "climbing after 10"
are each read with the endpoint's precision rather than with three reps.

## Contract

| | |
|---|---|
| Config | **`configs/p2_k_lr.yaml`, unchanged** (fingerprint `8ab654e2721decee87b6af33b3ea2c444bb19f8ecd1463f539234233895c298c`, lr 2e-5) |
| Warm start | `runs/p2_m_init/ckpt_iter0020.pt` |
| Continuity | `--seed-pool runs/p2_m_init` (its league, 0001–0020), `--seed-replay runs/p2_m_init`: committed head 1,576,622 = durable head (`state.json`, `replay/meta.json`), so the window is its newest 500,000 rows, offset −20 |
| What differs from one uninterrupted run | `--init` (fresh AdamW at 2e-5, seed streams from iteration 1), the standing protocol; p2_m found it makes no measurable difference at this length |
| Length | `--iterations 40` (the default round's length since p2_j) |
| Start reference | **skill 3:** p2_m_init/0020's six book reps (0.6025, 0.6575, 0.615, 0.625, 0.6575, 0.6275; mean 0.6308; `runs/p2_m_init/sf3/reps_iter0020_r{1..6}.jsonl`). **skill 0 (continuity):** its three book reps (0.9525, 0.9375, 0.935; `runs/p2_m_init/reps_iter0020_r{1,2,3}.jsonl`) and three initial-position reps (`runs/p2_m_init/startpos/`). Its ladder row at 20: mate-in-2 187, ECE 0.0997 |
| Training | RunPod, **secure RTX 4090**, ≥ 24 vCPU, the image of p2_g–p2_m; the variant recorded. If none is offered, the registered launch is retried; nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, under `gpu hold`, nothing until the final pull. (1) `cloud/remote_campaign.sh` with `MEASURE_AFTER_PULL=1`: at 10, 20, 30, 40 the ladder pass (ECE, N) and three skill-0 book reps; at 40 three skill-0 initial-position reps. (2) a finisher: book reps against `stockfish:3` — **six at 10**, three at 20 and 30, **six at 40** (`runs/p2_n/sf3/`) |
| Lifecycle | four guards; pod deadline **12 h**; worst case **$10.68** at $0.89/h; expected about **$7.5** (40 iterations at p2_m's 10.9 min = 7.3 h, plus boot, staging and the final pull) |

## Registered expectations

### L — the continuity applied
- **L1.** `runs/p2_n/provenance.json`: `config_fingerprint` `8ab654e2…`; `init_from`
  `runs/p2_m_init/ckpt_iter0020.pt`; `seed_replay.rows` 500,000, `donor_committed` = `donor_head` =
  1,576,622, `iteration_offset` −20; `git_dirty` false; `lr` 2e-5 on every row.
- **L2.** Book positions in the final ring: reported, no bound.

### K1 (primary) — the endpoint, at skill 3
p2_n/0040 against p2_m_init/0020, both against `stockfish:3`, **six reps each side**, per-opening
rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples,
seed 0; `runs/p2_n/sf3/p2_verdict_reps.json`). **H_cont expects the 95% CI entirely above zero.** The
prediction (below) is that it includes zero.

### K2 — the shape of the round, at skill 3
- **K2a, the early step:** p2_n/0010 against the start, **six reps each side**
  (`runs/p2_n/sf3/p2_verdict_reps_iter0010_six.json`).
- **K2b, the climb after it:** p2_n/0040 against p2_n/0010, six reps each
  (`runs/p2_n/sf3/p2_verdict_reps_0040_vs_0010.json`).
- **K2c, the trajectory:** the four judged checkpoints against the start, three reps each (r1–r3), and
  their least-squares slope per ten iterations. **Expected: no dip.**

How the shape is read, written now: **K2a above zero and K2b including zero** is "the round's gain is
its first ten iterations"; **K2b above zero** is "the round climbs after its first ten"; both including
zero, with K1 including zero, is a plateau. No decision row turns on K2; it is the record p2_o's design
reads.

### K3 — skill 0, the continuity reading
At each judged checkpoint the three skill-0 reps against p2_m_init/0020's three, and M4 (bar 0.675).
**Expected: M4 met at every checkpoint.** Skill 0 is at its ceiling (0.94–0.95).

### K4 — informational: the initial position (skill 0)
p2_n/0040's three reps against p2_m_init/0020's three. Reported.

### N, ECE, H
- **N:** mate-in-2 ≥ 183/200 at 40; reported at every judged checkpoint.
- **ECE:** < 0.15 at 40, the unchanged measurement, reported at every judged checkpoint with its bins; a
  break is handled in PREREG-p2_l's order (the calibration sample first).
- **H1–H8** as PREREG-p2_m, direction-free. **H2:** `skipped_steps` summed < 20 over 40 iterations.

## Predictions (written before launch)
- **K1 includes zero**, point +0.00 to +0.04 (the climb at 2e-5 is slowing).
- K2a above zero or near it; K2b includes zero; no dip.
- M4 at skill 0 met at every checkpoint; N 183–192; ECE 0.08–0.12; L2 0–5.

## Decision rules

| K1 (at skill 3) | Decision |
|---|---|
| **CI entirely above zero** | **Still climbing at 2e-5.** p2_o continues p2_n/0040 at 2e-5, judged at skill 3 |
| **CI includes zero** | **Plateau at 2e-5.** p2_o is the 2e-6 step-down as an A/B from p2_n/0040 against a continuation arm at 2e-5, twenty iterations each, judged at skill 3 — p2_k's design one decade down |
| **CI entirely below zero** | **Regressed.** Stop; nothing launches until diagnosed |
| **N < 183 or ECE ≥ 0.15 at 0040** | Whatever K1 reads, the broken guard is addressed first (ECE in PREREG-p2_l's order) and the round's gain is not taken as standing |

## Measurement protocol

```bash
# Stage the pod (launch, wait-ssh, stage; inputs' sha256 checked on the pod)
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 12 --tag p2n --vcpu 24 --ram 32 --disk 60 --yes
bash cloud/stage_pod.sh POD runs/p2_m_init/ckpt_iter00{01..20}.pt runs/p2_m_init/state.json runs/p2_m_init/config.yaml \
    runs/p2_m_init/replay/ runs/data/mate1_100k_clean/ runs/data/mate2_50k_clean/ runs/phase1_anchor.pt

# On the pod (ssh redirects all three streams, so it returns)
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --config configs/p2_k_lr.yaml \
   --init runs/p2_m_init/ckpt_iter0020.pt --seed-pool runs/p2_m_init --seed-replay runs/p2_m_init \
   --out runs/p2_n --iterations 40" > /dev/null 2>&1 < /dev/null &

# This box: the driver (skill 0, ladder, ECE), measuring only after the final pull
MEASURE_AFTER_PULL=1 START_REPS='runs/p2_m_init/reps_iter0020_r*.jsonl' START_STARTPOS='runs/p2_m_init/startpos/reps_iter0020_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD p2_n "10 20 30 40" 40 > /dev/null 2>&1 < /dev/null &

# After "campaign complete" (a finisher like runs/p2_l_finish.sh): skill 3
S=runs/p2_m_init/sf3/reps_iter0020_r*.jsonl
for n in 10 20 30 40; do
  nn=$(printf %04d $n); reps=3; { [ $n = 10 ] || [ $n = 40 ]; } && reps=6
  gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_n/ckpt_iter$nn.pt --anchor stockfish:3 --reps $reps --out-dir runs/p2_n/sf3
  .venv/bin/python scripts/eval_reps.py verdict --a start $S --b p2_n_$nn runs/p2_n/sf3/reps_iter${nn}_r{1,2,3}.jsonl \
    --out runs/p2_n/sf3/p2_verdict_reps_iter$nn.json
done
.venv/bin/python scripts/eval_reps.py verdict --a start $S --b p2_n_0040 runs/p2_n/sf3/reps_iter0040_r*.jsonl --out runs/p2_n/sf3/p2_verdict_reps.json
.venv/bin/python scripts/eval_reps.py verdict --a start $S --b p2_n_0010 runs/p2_n/sf3/reps_iter0010_r*.jsonl --out runs/p2_n/sf3/p2_verdict_reps_iter0010_six.json
.venv/bin/python scripts/eval_reps.py verdict --a p2_n_0010 runs/p2_n/sf3/reps_iter0010_r*.jsonl \
    --b p2_n_0040 runs/p2_n/sf3/reps_iter0040_r*.jsonl --out runs/p2_n/sf3/p2_verdict_reps_0040_vs_0010.json
.venv/bin/python scripts/eval_reps.py m4 --rows p2_n_0040 runs/p2_n/reps_iter0040_r*.jsonl --out runs/p2_n/p2_m4_iter0040.json
.venv/bin/python scripts/ring_book_rows.py runs/p2_n --json runs/p2_n/book_rows.json
```

## What would make p2_n uninterpretable
- A config delta (L1). A launch from anything but clean, pushed `main`; a measurement-instrument change.
- Losing the final pull: the round is re-run by addendum.

**What the statistic does not cover.** As PREREG-p2_k: the bootstrap resamples openings, not training
runs; the round is one run.

---

## Sign-off

Checked for the draft, 2026-11-01, on this box: p2_m_init's `state.json` iteration 20, step 8,000, 0
skipped steps, pool `ckpt_iter0001`–`0020` (all present), `replay_total_written` 1,576,622 = the ring's
`total_written`; its six skill-3 reps, three skill-0 reps and three initial-position reps present. Read
for the draft: RunPod balance **$14.37** (auto-top-up on; the account also carries Michael's own pods,
about $1.42/h), no pods of ours; secure RTX 4090 at $0.89/h (stock "Low"). **$10.68 would leave $3.69,
below the $5 margin**: the launch's budget check refuses until the balance is above $15.68. Price and
balance are read again at registration.

**The margin, changed at registration:** the account auto-reloads when its balance reaches $10
(Michael), so the $5 margin no longer guards against running dry; at Michael's instruction ("lower the
margin to $2") `~/.config/taz-cloud/limits.json` `balance_margin` is now 2.0 (per-pod ceiling $15
unchanged). Read at registration, 2026-11-01 11:54 UTC: balance $13.54, no pods of ours, secure RTX 4090
at $0.89/h (stock "Low"); $10.68 leaves $2.86, above the $2 margin. `make test` passed on this commit's
tree (916 passed, 2 skipped).

**Signed: the Claude Code session, at Michael's instruction ("go", 2026-11-01, after the draft and its
cost — expected about $7.5, worst case $10.68 — were put to him).** The commit carrying this line is the
registration.
