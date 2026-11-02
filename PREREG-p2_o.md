# Pre-registration — p2_o: a second tenfold step-down (2e-6) against a continuation arm, from p2_n/0040, judged on skill 3

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-02** (see Sign-off), at Michael's instruction. Nothing under this file
> has run.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_n read **plateau at 2e-5** (RUNLOG-p2_n §6: K1 +0.013 [−0.016, +0.043]; no early step, no climb after
it). Its registered row names this round: **the 2e-6 step-down as an A/B from p2_n/0040 against a
continuation arm at 2e-5, twenty iterations each, judged at skill 3 — p2_k's design one decade down.**

The precedent is direct. p2_h's plateau at 2e-4 came with falling training losses, flat skill and a
collapsed move-1 prior (0.013); p2_k's tenfold step-down from there won by +0.117 [+0.093, +0.140] at
skill 0, most of it in ten iterations. p2_n's plateau at 2e-5 has the same three signs (losses falling
round-long, skill flat, the move-1 prior from 0.51 to 0.07).

**H_lr2: a second tenfold step-down, to 2e-6, from p2_n/0040 improves skill-3 play from the frozen book
over a continuation arm at 2e-5 within twenty iterations, without costing mate-in-2.**

**The named risk** is the other side of the same step: 8,000 steps at 2e-6 move the weights a tenth as far
as at 2e-5, and a network that barely moves reads as "no effect", not as a cost. That reading has its own
row.

## Contract

| | |
|---|---|
| Arms | **`p2_o_cont`**: `configs/p2_k_lr.yaml` unchanged (2e-5; fingerprint `8ab654e2…298c`). **`p2_o_lr`**: `configs/p2_o_lr.yaml`, p2_k_lr's with `train.lr: 2.0e-06` and nothing else (fingerprint `cdf5eb97…1a7a`; `tests/test_rehearsal.py::test_p2o_lr_config_is_the_brief`) |
| Warm start | `runs/p2_n/ckpt_iter0040.pt`, both arms |
| Continuity | `--seed-pool runs/p2_n` (its league, 0021–0040), `--seed-replay runs/p2_n`: committed head 2,649,005 = durable head, so the window is its newest 500,000 rows, offset −40; both arms identical. `--init` (fresh AdamW, seed streams from 1) in both arms, so it cancels; p2_m found it no lever at 2e-5 |
| Length | `--iterations 20` per arm |
| Start reference | **skill 3:** p2_n/0040's six book reps (0.6375, 0.6125, 0.6525, 0.6725, 0.6225, 0.6675; mean 0.6442; `runs/p2_n/sf3/reps_iter0040_r{1..6}.jsonl`). **skill 0 (continuity):** its three book reps (0.96, 0.9675, 0.965; `runs/p2_n/reps_iter0040_r{1,2,3}.jsonl`) and three initial-position reps (`runs/p2_n/startpos/`). Its ladder row at 40: mate-in-2 191, ECE 0.0998 |
| Training | RunPod, **two pods in parallel**, each a secure RTX 4090 with ≥ 24 vCPU, the image of p2_g–p2_n; variants recorded. If none is offered, the registered launch is retried; nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, under `gpu hold`, nothing until each arm's final pull. One `cloud/remote_campaign.sh` per arm with `MEASURE_AFTER_PULL=1` (judged `10 20`): the ladder pass (ECE, N) and three skill-0 book reps; at 20 three skill-0 initial-position reps. Then a finisher: per arm, **three book reps against `stockfish:3` at 10 and six at 20** (`runs/p2_o_{cont,lr}/sf3/`) |
| Lifecycle | four guards; pod deadline **8 h** each; worst case **$7.12 per pod, $14.24 together** at $0.89/h; expected about **$7.7** together (20 iterations at p2_n's 10.7 min = 3.6 h per pod, plus boot, staging and the final pull) |

## Registered expectations

### L — the continuity applied
- **L1.** Each arm's `provenance.json`: `init_from` `runs/p2_n/ckpt_iter0040.pt`; `seed_replay.rows`
  500,000, `donor_committed` = `donor_head` = 2,649,005, `iteration_offset` −40; `git_dirty` false;
  `p2_o_cont`'s fingerprint = p2_k_lr's; `p2_o_lr`'s config differs from it in `lr` only; `lr` on every
  row 2e-5 (cont) and 2e-6 (lr).
- **L2.** Book positions in each arm's final ring: reported, no bound.

### K1 (primary) — the step-down against the continuation, skill 3
`p2_o_lr` − `p2_o_cont` at iteration 20, from the frozen book against `stockfish:3`, **six reps per arm**,
per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000
resamples, seed 0; `runs/p2_o_lr/sf3/p2o_verdict_lr_vs_cont.json`). **Expected: the 95% CI entirely above
zero.** Forecast point +0.02 to +0.08: a second step usually gives less than the first, but skill 3 is
less compressed than the skill-0 rung p2_k was judged on. Arm-against-arm SE at six reps has been
0.012–0.018 (p2_i, p2_k, p2_m).

### K2 — mate-in-2
At 20: **`p2_o_lr` solves at least as many of the 200 as `p2_o_cont` minus 4**, and **both ≥ 183**.
Reported at 10.

### K3 — each arm against the start (informational)
Each arm at 10 and 20 against p2_n/0040's six skill-3 reps (three reps; six at 20). Expected: the
continuation arm within noise of the start (the plateau holds a third time); the step-down arm above it.

### K4 — skill 0 and the initial position (continuity)
Three skill-0 reps per judged checkpoint against p2_n/0040's three, M4 at 20 per arm, and the
initial-position reps; reported. Skill 0 is at its ceiling (0.96).

### ECE, H
- **ECE:** < 0.15 at 20, both arms, the unchanged measurement; a break is handled in PREREG-p2_l's order.
- **H1–H8** as PREREG-p2_m, both arms, direction-free. **H1** is read against p2_n's collapse (0.51 →
  0.07): whether either arm's move-1 prior entropy recovers. **H7**: whether the step-down arm's training
  losses, falling all through p2_n, stop falling — the sign that 2e-6 has stopped moving the network.
  **H2:** `skipped_steps` < 10 per arm.

## Predictions (written before launch)
- K1 passes, point +0.02 to +0.08; K2 passes, the arms within 4 positions.
- K3: cont within noise of the start; lr above it at 20.
- ECE 0.08–0.12 in both arms; N 185–195; L2 0–5 per arm.
- H7: the lr arm's losses flatter than the cont arm's over the twenty iterations.

## Decision rules

| K1 | K2 | Decision |
|---|---|---|
| **CI entirely above zero** | passes | **Second step-down supported.** 2e-6 becomes standing protocol; p2_p continues `p2_o_lr/0020` at 2e-6 as the default round, judged at skill 3 (or the next rung, if 0020 reads above 0.75 there) |
| CI entirely above zero | fails | 2e-5 stays until the mate-in-2 cost is diagnosed |
| **CI includes zero** | any | **No effect seen at this length.** 2e-5 stays; the plateau stands. p2_p's lever is Michael's choice from the open list (the move-1 prior collapse, the calibration sample, a longer step-down arm); the record reports whether H7 says the network stopped moving |
| **CI entirely below zero** | any | **The second step-down costs skill.** 2e-5 stays; diagnosed before p2_p |
| N < 183 or ECE ≥ 0.15 in either arm | | The broken guard first (ECE in PREREG-p2_l's order) |

## Measurement protocol

```bash
# Two pods (launch, wait-ssh, stage; inputs' sha256 checked on each pod), the same inputs on both
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2oc --vcpu 24 --ram 32 --disk 60 --yes
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2ol --vcpu 24 --ram 32 --disk 60 --yes
bash cloud/stage_pod.sh POD runs/p2_n/ckpt_iter00{21..40}.pt runs/p2_n/state.json runs/p2_n/config.yaml \
    runs/p2_n/replay/ runs/data/mate1_100k_clean/ runs/data/mate2_50k_clean/ runs/phase1_anchor.pt

# On each pod (ssh redirects all three streams, so it returns)
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --config configs/p2_k_lr.yaml \
   --init runs/p2_n/ckpt_iter0040.pt --seed-pool runs/p2_n --seed-replay runs/p2_n \
   --out runs/p2_o_cont --iterations 20" > /dev/null 2>&1 < /dev/null &
#   ... and on the other pod --config configs/p2_o_lr.yaml --out runs/p2_o_lr

# This box, one driver per arm, measuring only after its own final pull
MEASURE_AFTER_PULL=1 START_REPS='runs/p2_n/reps_iter0040_r*.jsonl' START_STARTPOS='runs/p2_n/startpos/reps_iter0040_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD_C p2_o_cont "10 20" 20 > /dev/null 2>&1 < /dev/null &
MEASURE_AFTER_PULL=1 START_REPS='runs/p2_n/reps_iter0040_r*.jsonl' START_STARTPOS='runs/p2_n/startpos/reps_iter0040_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD_L p2_o_lr "10 20" 20 > /dev/null 2>&1 < /dev/null &

# After both campaigns complete (a finisher like runs/p2_m_finish.sh): skill 3
S=runs/p2_n/sf3/reps_iter0040_r*.jsonl
for arm in p2_o_cont p2_o_lr; do
  for nn in 0010 0020; do
    reps=3; [ $nn = 0020 ] && reps=6
    gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
      .venv/bin/python scripts/eval_reps.py run --ckpt runs/$arm/ckpt_iter$nn.pt --anchor stockfish:3 --reps $reps --out-dir runs/$arm/sf3
    .venv/bin/python scripts/eval_reps.py verdict --a start $S --b ${arm}_$nn runs/$arm/sf3/reps_iter${nn}_r{1,2,3}.jsonl \
      --out runs/$arm/sf3/p2_verdict_reps_iter$nn.json
  done
  .venv/bin/python scripts/eval_reps.py verdict --a start $S --b ${arm}_0020 runs/$arm/sf3/reps_iter0020_r*.jsonl --out runs/$arm/sf3/p2_verdict_reps.json
  .venv/bin/python scripts/eval_reps.py m4 --rows ${arm}_0020 runs/$arm/reps_iter0020_r*.jsonl --out runs/$arm/p2_m4_iter0020.json
  .venv/bin/python scripts/ring_book_rows.py runs/$arm --json runs/$arm/book_rows.json
done
.venv/bin/python scripts/eval_reps.py verdict --a cont runs/p2_o_cont/sf3/reps_iter0020_r*.jsonl \
    --b lr runs/p2_o_lr/sf3/reps_iter0020_r*.jsonl --out runs/p2_o_lr/sf3/p2o_verdict_lr_vs_cont.json
```

## What would make p2_o uninterpretable
- A second difference between the arms (the config test and L1 catch it).
- A launch from anything but clean, pushed `main`; a measurement-instrument change.
- The arms on different GPU variants: recorded and reported, not a stop.
- Losing either arm's final pull: the round is re-run by addendum, not read from one arm.

**What the statistic does not cover.** As PREREG-p2_k: each arm is one run.

---

## Sign-off

Checked for the draft, 2026-11-02, on this box: p2_n's `state.json` iteration 40, step 16,000, 0 skipped
steps, pool `ckpt_iter0021`–`0040` (all present), `replay_total_written` 2,649,005 = the ring's
`total_written`; its six skill-3 reps, three skill-0 reps and three initial-position reps present;
`configs/p2_o_lr.yaml` differs from `configs/p2_k_lr.yaml` in `lr` only (test passes), fingerprint
`cdf5eb97da1f6ce450e5552a1e4a5c1e85b4e071de02559d77f13d0b262d1a7a`. Read for the draft: RunPod balance
$22.36 (auto-reload at $10), no pods of ours; secure RTX 4090 at $0.89/h (stock listed "None" at that
read); 8 h × $0.89 = $7.12 per pod passes the $15 ceiling, and $14.24 for both leaves $8.12, above the
$2 margin. Read at registration, 2026-11-02 10:13 UTC: balance $21.99, no pods of ours, secure RTX 4090
at $0.89/h (stock "None"; the registered launch is retried until offered); $14.24 for both leaves $7.75,
above the $2 margin. `make test` passed on this commit's tree (934 passed, 2 skipped).

**Signed: the Claude Code session, at Michael's instruction ("ok given, go", 2026-11-02, after the draft
and its cost — expected about $7.7, worst case $14.24 — were put to him).** The commit carrying this line
is the registration.
