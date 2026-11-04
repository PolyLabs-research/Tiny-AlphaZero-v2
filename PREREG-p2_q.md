# Pre-registration — p2_q: the default round at 2e-6, forty iterations from p2_p_lr/0020, judged on the next rung

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-04** (see Sign-off), at Michael's instruction. Nothing under this file
> has run, Step 0 included.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_p's registered row (RUNLOG-p2_p §6): **the second step-down replicates** (K1 +0.075 [+0.039, +0.111]),
so 2e-6 is standing protocol and **p2_q continues `p2_p_lr/0020` at 2e-6 as the default round.** The row's
rung clause applies: `p2_p_lr/0020` reads 0.754 at skill 3, above 0.75, so p2_q is judged at the next rung
of PREREG-p2_l's climb list (1, 2, 3, **5**, 8), chosen by the same one-directional rule before launch.

**H_cont: forty more iterations at 2e-6 from p2_p_lr/0020, with nothing else changed, improve skill from
the frozen book at the new rung.**

The precedent is p2_l: the default round at 2e-5 after p2_k's step-down gained +0.104 in forty
iterations, most of it by ten. After two rounds at the new rate (p2_o, p2_p), this round reads whether 2e-6
keeps climbing on its own. The round's shape is read with six reps at 10 as well as at 40, as in p2_n.

## Step 0 — the rung (this box, after registration, before launch)

Nothing trains while Step 0 runs.

**0a. The climb, continued.** `p2_p_lr/0020`, **one** book rep at `stockfish:5`; if it scores ≤ 0.75,
**N = 5**; otherwise one rep at `stockfish:8`, and N = 8 (reported as such if 8 still scores above 0.75).
PREREG-p2_l's rule, binding as there: every climb rep is logged and committed
(`runs/p2_p_lr/climb_s{k}/reps_iter0020_r1.jsonl`); no descending, no skipping, no continuing past the
first skill at or below 0.75.

**0b. The start reference at N.** **Six fresh** book reps of `p2_p_lr/0020` against `stockfish:N`
(`runs/p2_p_lr/sf{N}/reps_iter0020_r{1..6}.jsonl`); the climb's selecting rep is not one of them.

## Contract

| | |
|---|---|
| Config | `--config configs/p2_o_lr.yaml --seed 43` — p2_p_lr's own resolved config (fingerprint `86fcc672…7d65`, lr 2e-6, seed 43): zero deltas from the run it continues |
| Warm start | `runs/p2_p_lr/ckpt_iter0020.pt` |
| Continuity | `--seed-pool runs/p2_p_lr` (its league, 0001–0020), `--seed-replay runs/p2_p_lr`: committed head 1,573,665 = durable head, newest 500,000 rows, offset −20. `--init` (fresh AdamW at 2e-6, seed streams from 1), the standing protocol |
| Length | `--iterations 40` |
| Start reference | **at the rung N:** Step 0b's six reps. **skill 3 (the last rung):** p2_p_lr/0020's six skill-3 reps (0.7825, 0.7375, 0.7425, 0.7575, 0.745, 0.76; mean 0.7542; `runs/p2_p_lr/sf3/reps_iter0020_r{1..6}.jsonl`). **skill 0:** its three book reps and three initial-position reps. Its ladder row at 20: mate-in-2 191, ECE 0.1032 |
| Training | RunPod, **secure RTX 4090**, ≥ 24 vCPU, the image of p2_g–p2_p; the variant recorded. If none is offered, the registered launch is retried; nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, `gpu hold`, nothing until the final pull. (1) `remote_campaign.sh` with `MEASURE_AFTER_PULL=1` at 10, 20, 30, 40: the ladder pass (ECE, N) and three skill-0 reps; at 40 three skill-0 initial-position reps. (2) a finisher: against `stockfish:N` six reps at 10, three at 20 and 30, six at 40 (`runs/p2_q/sf{N}/`); against `stockfish:3` three reps at 40 (`runs/p2_q/sf3/`) |
| Lifecycle | four guards; pod deadline **12 h**; worst case **$10.68** at $0.89/h; expected about **$7.6** (40 iterations at p2_p's 11.0–11.6 min ≈ 7.5 h, plus boot, staging and the final pull) |

## Registered expectations

### L — the continuity applied
- **L1.** `runs/p2_q/provenance.json`: `config_fingerprint` `86fcc672…` (= p2_p_lr's); `seed` 43;
  `init_from` `runs/p2_p_lr/ckpt_iter0020.pt`; `seed_replay.rows` 500,000, `donor_committed` =
  `donor_head` = 1,573,665, `iteration_offset` −20; `git_dirty` false; `lr` 2e-6 on every row.
- **L2.** Book positions in the final ring: reported, no bound.

### K1 (primary) — the endpoint, at the rung N
p2_q/0040 against p2_p_lr/0020, both against `stockfish:N`, **six reps each side**, per-opening rep-means,
paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed 0;
`runs/p2_q/sf{N}/p2_verdict_reps.json`), read with the exact-zero rule (`f7779a8`). **Expected: the 95% CI
entirely above zero.** Forecast point +0.02 to +0.08.

### K2 — the shape of the round, at N
As PREREG-p2_n: **K2a** 0010 against the start (six each); **K2b** 0040 against 0010 (six each); **K2c**
the four checkpoints against the start (three reps each) and their slope. **Expected: no dip.** The shape
is read as PREREG-p2_n wrote it; no row turns on it.

### K3 — the last rung and skill 0 (continuity)
p2_q/0040's three skill-3 reps against p2_p_lr/0020's six; the skill-0 reps and M4 at each checkpoint;
the initial-position reps at 40. Reported.

### N, ECE, H
- **N:** mate-in-2 ≥ 183/200 at 40, reported at every checkpoint.
- **ECE:** < 0.15 at 40, reported at every checkpoint with its bins; a break is handled in PREREG-p2_l's
  order.
- **H1–H8** as PREREG-p2_p, direction-free; **H9 (new):** the promotion loss, which rose by a quarter in
  the step-down arm of p2_p (0.36 → 0.46) — whether it settles. **H2:** `skipped_steps` < 20 over 40.

## Predictions (written before launch)
- Step 0: N = 5, the start between 0.45 and 0.70 there.
- K1 passes, point +0.02 to +0.08; K2a above zero; no dip.
- ECE 0.08–0.12; N 185–195; L2 0–5.

## Decision rules

| K1 (at N) | Decision |
|---|---|
| **CI entirely above zero** | **Still climbing at 2e-6.** p2_r continues p2_q/0040 at 2e-6, judged at N (or the next rung, if 0040 reads above 0.75 there) |
| **CI includes zero** | **Plateau at 2e-6.** p2_r's lever is Michael's choice: the next tenfold step (2e-7) as an A/B against a continuation arm, or a change of lever from the open list |
| **CI entirely below zero** | **Regressed.** Stop; diagnosed before p2_r |
| **N < 183 or ECE ≥ 0.15 at 0040** | Whatever K1 reads, the broken guard first (ECE in PREREG-p2_l's order); the round's gain is not taken as standing |

## Measurement protocol

```bash
# Step 0, this box, nothing running on RunPod (a script like runs/p2_l_step0.sh, the list continued at 5, 8)
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_p_lr/ckpt_iter0020.pt --anchor stockfish:5 --reps 1 --out-dir runs/p2_p_lr/climb_s5
#   ... at stockfish:8 only if skill 5 scored above 0.75; then six fresh reps at N:
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_p_lr/ckpt_iter0020.pt --anchor stockfish:N --reps 6 --out-dir runs/p2_p_lr/sfN

# Stage the pod (launch, wait-ssh, stage; inputs' sha256 checked on the pod)
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 12 --tag p2q --vcpu 24 --ram 32 --disk 60 --yes
bash cloud/stage_pod.sh POD runs/p2_p_lr/ckpt_iter00{01..20}.pt runs/p2_p_lr/state.json runs/p2_p_lr/config.yaml \
    runs/p2_p_lr/replay/ runs/data/mate1_100k_clean/ runs/data/mate2_50k_clean/ runs/phase1_anchor.pt

# On the pod
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --config configs/p2_o_lr.yaml --seed 43 \
   --init runs/p2_p_lr/ckpt_iter0020.pt --seed-pool runs/p2_p_lr --seed-replay runs/p2_p_lr \
   --out runs/p2_q --iterations 40" > /dev/null 2>&1 < /dev/null &

# This box: the driver (skill 0, ladder, ECE), measuring only after the final pull
MEASURE_AFTER_PULL=1 START_REPS='runs/p2_p_lr/reps_iter0020_r*.jsonl' START_STARTPOS='runs/p2_p_lr/startpos/reps_iter0020_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD p2_q "10 20 30 40" 40 > /dev/null 2>&1 < /dev/null &

# After "campaign complete": a finisher like runs/p2_n_finish.sh at stockfish:N (six at 10 and 40, three at 20 and
# 30; the K1, K2a, K2b, K2c verdicts against runs/p2_p_lr/sfN/), plus three stockfish:3 reps at 40 against
# runs/p2_p_lr/sf3/, M4 at skill 0, and L2.
```

## What would make p2_q uninterpretable
- A config delta from p2_p_lr (L1). A launch from anything but clean, pushed `main`; a
  measurement-instrument change.
- A climb that broke its rule.
- Losing the final pull: the round is re-run by addendum.

**What the statistic does not cover.** As PREREG-p2_k: the bootstrap resamples openings, not runs.

---

## Sign-off

Checked for the draft, 2026-11-04, on this box: p2_p_lr's `state.json` iteration 20, step 8,000, seed 43,
0 skipped steps, pool `ckpt_iter0001`–`0020` (all present), `replay_total_written` 1,573,665 = the ring's
`total_written`; its `config.yaml` sha256 `86fcc672…7d65`; its six skill-3 reps, three skill-0 reps and
three initial-position reps present; the Step 0 and p2_q record paths reachable. Read for the draft: RunPod
balance $24.81 (auto-reload at $10), no pods of ours; secure RTX 4090 at $0.89/h (stock "None"); $10.68
leaves $14.13, above the $2 margin. Read at registration, 2026-11-04 10:35 UTC: balance $22.20, no pods
of ours, secure RTX 4090 at $0.89/h (stock "Low"); $10.68 leaves $11.52, above the $2 margin. `make test`
passed on this commit's tree (975 passed, 2 skipped).

**Signed: the Claude Code session, at Michael's instruction ("go", 2026-11-04, after the draft and its cost
— expected about $7.6, worst case $10.68 — were put to him).** The commit carrying this line is the
registration.
