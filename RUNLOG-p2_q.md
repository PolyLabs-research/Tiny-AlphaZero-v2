# RUNLOG — p2_q (the default round at 2e-6, forty iterations from p2_p_lr/0020, judged on skill 5)

**Run:** `runs/p2_q` (`--config configs/p2_o_lr.yaml --seed 43`: p2_p_lr's own config, lr 2e-6), 40
iterations, trained on RunPod pod `ckr94pol17skp5`, a secure RTX 4090 of the 24 GB variant (`vram_gb`
23.52), 64 vCPU on a 54-CPU quota, from 2026-11-04 16:41:34 to 2026-11-05 00:24:19 UTC, code `5420f99`,
warm-started from `p2_p_lr/0020` with the league and the replay ring continued from `runs/p2_p_lr`. Every
measurement ran on this box (torch 2.14.1+rocm7.2): Step 0 before launch (`runs/p2_q_step0.sh`); after
the final pull the ladder, skill-0 reps and ECE by `cloud/remote_campaign.sh` with `MEASURE_AFTER_PULL=1`,
the skill-5 and skill-3 reps and the verdicts by `runs/p2_q_finish.sh`.
**Pre-registration:** `PREREG-p2_q.md` (`5420f99`), no addenda. Every verdict below was registered before
it was measured, except where marked.

**Verdict: H_cont NOT SUPPORTED — the registered "plateau at 2e-6" row; p2_r's lever is Michael's choice.
K1 (0040 against p2_p_lr/0020, Stockfish skill 5, six reps each side): +0.009 [−0.018, +0.035]. The round's
shape: no early step (K2a −0.004) and no climb after it (K2b +0.013). Flat at skill 3 too (+0.007). N and
ECE hold; L1 holds; L2 reported (1 row). Michael chose the lever: more self-play search (PREREG-p2_r).**

---

## 1. Step 0 and launch

Registered 2026-11-04 10:35 UTC (balance $22.20, $0.89/h; `make test` 975 passed, 2 skipped).

**0a, the climb continued** (10:37–10:48): `p2_p_lr/0020`'s single book rep at `stockfish:5` scored
**0.325** (42W/46D/112L), at or below 0.75, so **N = 5**; skill 8 was not run.

**0b, the start reference at skill 5** (10:48–11:43): six fresh reps 0.3025, 0.27, 0.35, 0.2725, 0.315,
0.325 — **mean 0.306 [0.287, 0.326]**. The prediction (a start between 0.45 and 0.70) **failed**: the step
from skill 3 (0.754) to skill 5 (0.306) is far larger than the step from 2 to 3 had been (0.80 to 0.555 on
p2_l's climb). Games at skill 5 run 146–163 plies. The Step 0 records were committed ahead of this RUNLOG
(`7617ebc`), at Michael's request, while the round ran.

**The pod:** secure RTX 4090s were out (listed "None" on both clouds at one read); the registered launch
was refused 57 times from 11:44 and retried every five minutes, by the pipeline and then by the wrapper
that reruns it only after a "no pod" result. The pod came at 16:32:08 (64 vCPU, 62 GB RAM); staging ran
without a retry; all 50 inputs' sha256 matched on the pod; the job started 16:41:34.

## 2. L — the continuity applied

- **L1 holds:** `config_fingerprint` `86fcc672…7d65` (= p2_p_lr's, and its `config.yaml`'s sha256); `seed`
  43; `init_from` `runs/p2_p_lr/ckpt_iter0020.pt`; `seed_replay.rows` 500,000, `donor_committed` =
  `donor_head` = 1,573,665, `iteration_offset` −20; `git_dirty` false; `git_sha` `5420f99` and `lr` 2e-6
  on all 40 rows.
- **L2 (reported, no bound):** **1 row** in the final ring (`open_027` at iteration 34); initial-position
  rows 525; distinct positions 478,156.

## 3. K — the endpoint and the shape, at skill 5

### K1 (primary) — p2_q/0040 against the start, six reps each side

Per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000
resamples, seed 0; `runs/p2_q/sf5/p2_verdict_reps.json`, reproduced byte for byte; all eighteen skill-5
rows `stockfish_s5` on their own checkpoints):

| | reps r1–r6 | mean |
|---|---|---|
| start (p2_p_lr/0020) | 0.3025, 0.27, 0.35, 0.2725, 0.315, 0.325 | 0.306 |
| p2_q/0040 | 0.3075, 0.3125, 0.325, 0.295, 0.2675, 0.38 | 0.315 |
| **0040 − start** | | **+0.009 [−0.018, +0.035]**, SE 0.0137 |

- **The interval includes zero: the plateau row.** Predicted above zero (+0.02 to +0.08); **failed**. Per
  opening, 0040 is above the start on 47 of the 100, below on 41, equal on 12. W/D/L at 0040 over six reps:
  231 / 293 / 676 of 1,200.

### K2 — the shape of the round

| reading | result |
|---|---|
| **K2a**, 0010 against the start (six each; 0.3175, 0.28, 0.2875, 0.28, 0.2875, 0.36, mean 0.302) | **−0.004 [−0.033, +0.025]** |
| **K2b**, 0040 against 0010 (six each) | **+0.013 [−0.020, +0.046]** |
| K2c, the four checkpoints against the start (r1–r3) | −0.011, +0.011, −0.006, +0.009; slope +0.004 per ten iterations |

The registered reading: K2a and K2b both include zero, with K1 including zero — **a plateau**. No dip. No
early step either: as in p2_n, the default round after a step-down's own rounds took nothing.

### K3 — the last rung and skill 0 (continuity)

- **Skill 3:** 0040's three reps 0.745, 0.7525, 0.785 (0.761) against p2_p_lr/0020's six (0.754): **+0.007
  [−0.030, +0.043]**.
- **Skill 0:** against p2_p_lr/0020's three (0.98, 0.955, 0.9875), −0.020 [−0.038, −0.003] at 10 (three
  reps each; the only interval entirely below zero in the round), then −0.018, −0.006, −0.006 [−0.022,
  +0.009] at 40. M4 at 40 **0.968**, met. Skill 0 at its ceiling; no row turns on it.
- **Initial position:** +0.002 [−0.014, +0.018].

## 4. N, ECE, H

- **N — holds:** mate-in-2 189, 190, 188, 189 (≥ 183; predicted 185–195). mate-in-3 151, 153, 153, 154.
  Watchlist: `mate2_165` and `166` solved throughout, `129` unsolved.
- **ECE — holds:** 0.1037, 0.0932, 0.0963, 0.0966 (< 0.15; predicted 0.08–0.12).
- **H7 / the losses (by quarters):** value 0.388 / 0.388 / 0.392 / 0.393 (creeping up), policy 1.430 /
  1.418 / 1.410 / 1.404 (falling). The network fits its policy targets better each quarter while its
  play does not move.
- **H9 (promotion loss):** 0.449 / 0.452 / 0.462 / 0.453 — stays at p2_p_lr's raised level (0.46); it did
  not settle back to the 2e-5 arms' 0.36.
- **H1:** `move1_prior_entropy` 0.16 at 1, 0.23 at 10, 0.29 at 20, 0.24 at 30, 0.16 at 40.
- **H6:** `book_move1_entropy` 0.623, 0.618, 0.611, 0.601.
- **H2:** 0 skipped steps in 16,000.

Rungs: random, greedy and phase1 won 200/200 at every checkpoint.

## 5. Mechanics

- **Pace:** 11.57 min per iteration (11.13–12.32), 3,693–4,142 sims/s (mean 3,986); 7.7 h of training.
- **Lifecycle:** iteration 40 committed 00:24:19; final pull with the ring 00:25:28, `PULLED` written, the
  pod deleted itself. Measurement under `gpu hold` (games at fixed nodes and simulations, so load on the
  box changes their duration, not their outcome): ladder and skill-0 reps 0010 00:25–00:53, 0020 –01:22, 0030 –01:50, 0040 with the initial-position reps
  –02:34; skill-5 reps (finisher) 0010 (six) 02:35–03:29, 0020 –03:56, 0030 –04:23, 0040 (six) –05:18;
  skill-3 reps at 40 –05:40; verdicts at 05:39:41 UTC. No retries. Two `rsync` failures in the driver's
  log date from the pod's first minutes ("no public SSH endpoint yet"); every later pull succeeded.
- **Cost:** ≈ **$7.03** at the pod's $0.89/h (16:32 to 00:26, 7.9 h); expected about $7.6, worst case
  $10.68. All RunPod use by this project to date about $94.

## 6. Decision

Per the table's second row (K1 includes zero): **plateau at 2e-6; p2_r's lever is Michael's choice.**
The options put to him (2026-11-05), each keeping the 6.5M-parameter network fixed (the project's aim):
more search in self-play; the next tenfold step (2e-7) as an A/B; distillation from Stockfish
evaluations. **Michael chose more self-play search** — `selfplay.sims_per_move` raised from 48, as an A/B
from p2_q/0040 against a continuation arm, the measuring instrument (`eval.eval_match_sims` 48)
unchanged. PREREG-p2_r.

Readings the table does not cover:

1. **The plateau is in what the network learns from, not in how fast.** The policy loss keeps falling and
   the play is flat at two rungs (skill 5 and skill 3): the network fits its self-play targets ever
   better, and those targets — 48 simulations of search per move — have stopped carrying new skill.
   Two step-downs each bought one gain and then stopped; a third (2e-7) would move the weights a tenth as
   far again. Raising the targets' quality is the lever that leaves the network's size alone.
2. **Skill 5 is a large step above skill 3.** 0.754 at skill 3 against 0.306 at skill 5 from the same
   checkpoint; at 0.31 there is room above for any lever that works.
3. **The promotion loss stays raised at 2e-6** (0.45 against 0.36 at 2e-5) without a visible cost in play;
   it is carried as a reading.

The open list, carried forward: p2_r (more self-play search); the calibration sample; the promotion loss;
L2's exclusion at storage time; the host freezes; a public benchmark (a UCI wrapper first).

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_p_lr/climb_s5/`, `runs/p2_p_lr/sf5/` | Step 0 (committed `7617ebc`) |
| `runs/p2_q/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the run |
| `runs/p2_q/ladder.jsonl` | ladder passes at 10, 20, 30, 40 (ECE, N) |
| `runs/p2_q/sf5/` | skill-5 reps; K1 (`p2_verdict_reps.json`), K2a, K2b, K2c |
| `runs/p2_q/sf3/` | skill-3 reps at 40 and their verdict (K3) |
| `runs/p2_q/reps_iter00{10..40}_r{1,2,3}.jsonl`, `p2_verdict_reps_iter*.json`, `p2_verdict_reps.json`, `p2_m4_iter0040.json`, `startpos/` | skill 0 |
| `runs/p2_q/book_rows.json` | L2 |
| `runs/p2_q_campaign.console.log`, `runs/p2_q.console.log`, `runs/p2_q_step0.console.log`, `runs/p2_q_finish.console.log`, `runs/p2_q_step0.sh`, `runs/p2_q_finish.sh`, `runs/p2_q_rung.txt` | consoles and scripts (untracked) |
