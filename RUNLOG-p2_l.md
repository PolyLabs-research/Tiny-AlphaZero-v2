# RUNLOG — p2_l (the default round at 2e-5, forty iterations from p2_k_lr/0020, judged on Stockfish skill 3)

**Run:** `runs/p2_l` (`configs/p2_k_lr.yaml` unchanged: p2_g's config at lr 2e-5), 40 iterations, trained
on RunPod pod `iofo55lyorj1bq`, a secure RTX 4090 of the 48 GB variant (`vram_gb` 47.37), 32 vCPU on a
31-CPU quota, from 2026-10-30 21:57:43 to 2026-10-31 05:04:29 UTC, code `2c9be5d`, warm-started from
`p2_k_lr/0020` with the league and the replay ring continued from `runs/p2_k_lr`. Every measurement ran on
this box (torch 2.14.1+rocm7.2): Step 0 before launch (`runs/p2_l_step0.sh`); after the final pull, the
ladder, skill-0 reps and ECE by `cloud/remote_campaign.sh` with `MEASURE_AFTER_PULL=1`, and the skill-3
reps and K1 by `runs/p2_l_finish.sh`.
**Pre-registration:** `PREREG-p2_l.md` (`2c9be5d`), no addenda. Every verdict below was registered
before it was measured, except where marked.

**Verdict: H_cont SUPPORTED — the registered "still climbing at 2e-5" row. K1 at the rung (Stockfish
skill 3, chosen by Step 0's climb): p2_l/0040 against p2_k_lr/0020, six reps each side, +0.104 [+0.071,
+0.136], above the forecast (+0.02 to +0.06). K2: no dip, but the gain was in place by iteration 10
(+0.070) and the slope over the four checkpoints is +0.005 per ten iterations. Skill 0 (continuity):
M4 met throughout, the rung saturated (0.93). ECE held flat at 0.08–0.095 — the predicted rise did not
happen and the registered ECE rule was not triggered. N holds; L1 holds; L2 reported (2 rows).**

---

## 1. Step 0 and launch

**0a, the climb** (2026-10-30, 19:20–19:42 UTC, nothing on RunPod): `p2_k_lr/0020`, one book rep each,
stopping at the first skill scoring ≤ 0.75:

| skill | W / D / L | score | mean plies | |
|---|---|---|---|---|
| 1 | 163 / 18 / 19 | 0.86 | 93.0 | above 0.75 |
| 2 | 149 / 21 / 30 | 0.7975 | 109.6 | above 0.75 |
| 3 | 96 / 30 / 74 | 0.555 | 133.5 | **≤ 0.75: N = 3** |

Skills 5 and 8 were not run, by the rule. All three rows are committed
(`runs/p2_k_lr/climb_s{1,2,3}/`). Predicted: the climb stops at 2, 3 or 5.

**0b, the start reference at skill 3** (19:42–20:34): six fresh book reps of `p2_k_lr/0020`: 0.465,
0.4875, 0.4525, 0.5125, 0.495, 0.495 — **mean 0.4846 [0.460, 0.510]** (one-sample, the M4 method;
`runs/p2_k_lr/sf3/`). The climb's selecting rep (0.555) is above all six: had it been one of them, the
start would have read higher. Games against skill 3 run longer than against skill 0 (the climb's 133.5
plies and 0040's 132–154, against ~95 at skill 0) and draw more (25–44 per 200 at 0040).

**Launch:** read at registration 19:19 UTC, balance $30.01, secure RTX 4090 at $0.89/h. Launches were
refused for want of instances at 20:35 and every five minutes after; the pod came at 21:18:19 (32 vCPU,
125 GB RAM). Staging ran without a retry; all 50 inputs' sha256 matched on the pod before the job started
at 21:57:43.

## 2. L — the continuity applied

- **L1 holds:** `config_fingerprint` `8ab654e2…298c` = p2_k_lr's; `init_from`
  `runs/p2_k_lr/ckpt_iter0020.pt`; `seed_replay.rows` 500,000, `donor_committed` = `donor_head` =
  1,555,800, `iteration_offset` −20; `git_dirty` false; `git_sha` `2c9be5d` and `lr` 2e-5 on all 40
  iteration rows; `ckpt_iter0040.pt` carries 2e-5 in its config and its optimizer state.
- **L2 (reported, no bound):** **2 rows** in the final ring (500,000; `total_written` 2,644,005):
  `open_027` (iteration 38), `open_009` (iteration 35). Initial-position rows 520; distinct positions
  480,222. Predicted 0–5.

## 3. K — the endpoint and the trajectory, at skill 3

### K1 (primary) — p2_l/0040 against the start, both against Stockfish skill 3, six reps each side

Per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000
resamples, seed 0; `runs/p2_l/sf3/p2_verdict_reps.json`, reproduced byte for byte; all twelve rows
`stockfish_s3`, each side on its own checkpoint):

| | reps r1–r6 | mean |
|---|---|---|
| start (p2_k_lr/0020) | 0.465, 0.4875, 0.4525, 0.5125, 0.495, 0.495 | 0.485 |
| p2_l/0040 | 0.575, 0.585, 0.57, 0.5675, 0.6125, 0.62 | 0.588 |
| **0040 − start** | | **+0.104 [+0.071, +0.136]**, SE 0.0168 |

- **K1 passes: the interval lies entirely above zero.** The point is above the forecast (+0.02 to +0.06).
  Every 0040 rep is above every start rep (lowest 0.5675 against highest 0.5125). Per opening, 0040's
  rep-mean is above the start's on 66 of the 100, below on 21, equal on 13 (median +0.083). W/D/L at
  0040 over the six reps: 603 / 206 / 391 of 1,200.

### K2 — the trajectory (three reps per checkpoint against the six-rep start)

| checkpoint | reps | mean | vs start [95% CI] |
|---|---|---|---|
| 0010 | 0.54, 0.5725, 0.55 | 0.554 | +0.070 [+0.029, +0.110] |
| 0020 | 0.5175, 0.57, 0.5975 | 0.562 | +0.077 [+0.033, +0.120] |
| 0030 | 0.49, 0.565, 0.59 | 0.548 | +0.064 [+0.020, +0.108] |
| 0040 | 0.575, 0.585, 0.57 | 0.577 | +0.092 [+0.046, +0.136] |

- **No dip** (predicted). Least-squares slope **+0.005 per ten iterations.** Every checkpoint's interval
  is above zero, and the four points are within noise of one another: the round's gain was in place by
  iteration 10 (*a reading the decision table does not cover, §6*).

### K3 — skill 0, the continuity reading

Three skill-0 reps per checkpoint against p2_k_lr/0020's six (0.9033), and M4 (bar 0.675):

| checkpoint | reps | vs start [95% CI] | M4: mean [95% CI] |
|---|---|---|---|
| 0010 | 0.94, 0.92, 0.9225 | +0.024 [−0.001, +0.049] | 0.928 [0.908, 0.947] met |
| 0020 | 0.9625, 0.9125, 0.93 | +0.032 [+0.010, +0.053] | 0.935 [0.918, 0.953] met |
| 0030 | 0.90, 0.9325, 0.94 | +0.021 [−0.003, +0.044] | 0.924 [0.906, 0.942] met |
| 0040 | 0.9575, 0.9275, 0.9275 | +0.034 [+0.013, +0.055] | 0.9375 [0.921, 0.953] met |

M4 met at every checkpoint (predicted). The skill-0 differences are a third of skill 3's: at 0.93 the
rung has 7% left, which is why the endpoint moved (PREREG-p2_l, the question). The ladder's single
skill-0 replay (*not registered*): 0.9475, 0.93, 0.9225, 0.93 (183W/6D/11L at 40).

### K4 — informational: the initial position (skill 0)

0040's three reps 0.955, 0.95, 0.905 (mean 0.937) against p2_k_lr/0020's 0.934: **+0.003 [−0.022,
+0.027]**. At the ceiling of the skill-0 rung; uninformative this round.

## 4. N, ECE, H

- **N — holds:** mate-in-2 **188, 188, 187, 189** at 10/20/30/40 (≥ 183; predicted 183–192). mate-in-3
  149, 152, 150, 146. Failures at 40 (11): `mate2_084`, `091`, `124`, `126`, `127`, `129`, `138`, `139`,
  `140`, `166`, `196`. Watchlist: `mate2_165` solved at every checkpoint, `129` and `166` unsolved.
- **ECE — holds, and flat:** **0.0903, 0.0803, 0.0950, 0.0914** (< 0.15), 5,000 positions each. The
  prediction (a rise to 0.10–0.18) **fails**: the 0.063 → 0.090 rise over p2_k_lr's second half did not
  continue. The registered ECE rule (diagnose the calibration sample first) was not triggered. The bins
  at 40 (*not registered, for the record*) show where the 0.09 comes from — the value head is
  **under**confident in won positions:

  | bin | n | predicted | realized |
  |---|---|---|---|
  | 0 | 950 | 0.009 | 0.013 |
  | 1–3 | 375 | 0.15–0.35 | 0.07–0.32 |
  | 4 | 380 | 0.468 | 0.565 |
  | 5 | 640 | 0.542 | 0.823 |
  | 6 | 409 | 0.647 | 0.924 |
  | 7 | 316 | 0.747 | 0.919 |
  | 8 | 265 | 0.848 | 0.959 |
  | 9 | 1,665 | 0.987 | 0.996 |

  Positions it rates 0.54–0.75 are won 82–92% of the time. The calibration games are against phase1 and
  skill 0, against which it now scores 1.00 and 0.93, so the realized column is lifted by the opponents' weakness — the
  suspect PREREG-p2_l named, read here in the bins rather than tested.
- **H1 (direction-free):** `move1_prior_entropy` 0.70 at 1, 0.67 at 10, 0.63 at 20, 0.70 at 30, **0.46 at
  40** (0.56–0.70 through iteration 39, a late drop).
- **H2:** 0 skipped steps in 16,000 (< 20).
- **H3–H5, H7** (direction-free, by quarters 1–10 / 11–20 / 21–30 / 31–40): value loss 0.352 / 0.356 /
  0.351 / 0.350 (flat); policy loss 1.595 / 1.567 / 1.548 / 1.545 (falling); decisive rate 0.796 / 0.788
  / 0.791 / 0.788; promotion loss 0.342 / 0.363 / 0.356 / 0.363. p2_k_lr's loss rise (RUNLOG-p2_k §4) did
  not continue: the value loss settled at its level, the policy loss came down.
- **H6** (`book_move1_entropy`): 0.603, 0.574, 0.548, 0.533 — falling.

Rungs: random, greedy and phase1 won 200/200 at every checkpoint. Self-play games 104–105 plies,
53,600 positions per iteration.

## 5. Mechanics

- **Pace:** 10.67 min per iteration (10.23–11.10), 4,208–4,565 sims/s (mean 4,395), training 52.9 s; 7.1 h
  of training. Faster than p2_k's two pods on one host (11.4–11.5).
- **Lifecycle:** iteration 40 committed 05:04:29; final pull with the ring 05:08:43, `PULLED` written,
  the pod deleted itself. Measurement after the pull, under `gpu hold`: ladder and skill-0 reps 0010
  05:09–05:38, 0020 –06:08, 0030 –06:38, 0040 with the initial-position reps –07:24; skill-3 reps
  (finisher) 0010 07:29–07:55, 0020 –08:21, 0030 –08:46, 0040 (six) –09:37; K1, M4 and L2 at 09:37:29 UTC.
  No retries. This session's container restarted while the measurement ran; the driver and the finisher
  are detached from it and completed unaffected.
- **Cost:** ≈ **$6.98** at the pod's $0.89/h (21:18 to 05:09, 7.8 h); expected about $7.5, worst case
  $10.68. Michael turned on auto-top-up on 2026-10-31. All RunPod use to date about $59.

## 6. Decision

Per the table's first row (K1 entirely above zero): **still climbing at 2e-5. Continuation at 2e-5
remains the default round; p2_m continues p2_l/0040, judged at skill 3.** N and ECE hold at 0040, so the
guard row does not apply. Any launch is a new PREREG and Michael's cost OK.

Readings the table does not cover:

1. **The gain arrives early, twice now.** p2_k_lr took +0.110 of its +0.132 (skill 0) by iteration 10;
   p2_l took +0.070 of its +0.092 (skill 3, three reps) by iteration 10, and its four checkpoints are
   flat within noise (slope +0.005 per ten). "Still climbing" is true of the endpoint against the start,
   not of the trajectory inside the round. Each round restarts AdamW and the seed streams at iteration
   1 (the standing protocol); whether the early step is the restart, the rung, or forty iterations being
   longer than the climb lasts is not separated by anything run so far. p2_j's continuation at 2e-4 read
   −0.026 at 10 after the same restart, so the restart alone does not produce it. p2_m's PREREG should
   say in advance how a flat 10 → 40 trajectory is read, and whether a shorter round, or a
   resume-without-restart arm (RUNLOG-p2_i §6 item 1), is the better use of the next pod.
2. **ECE is flat, and its bins point at the sample.** The rise RUNLOG-p2_k §6 projected did not happen;
   the 0.09 comes from underconfidence in won positions measured against opponents the model beats. The
   guard holds; a calibration sample from a matchup the model does not mostly win (its skill-3 games) is
   the measurement change the PREREG named, for its own PREREG if wanted.
3. **Skill 3 has room.** 0.588 at 40 leaves 0.41 above; skill 0 (0.93) is retired as a resolving rung
   and kept as the continuity reading.

The open list, carried forward: p2_m (continuation at 2e-5, judged at skill 3) and how it reads the
early-gain shape; the calibration sample; mate-in-2 near its floor (187–189, floor 183); L2's exclusion at
storage time; the host freezes; pace.

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_k_lr/climb_s{1,2,3}/reps_iter0020_r1.jsonl` | Step 0a, the climb |
| `runs/p2_k_lr/sf3/reps_iter0020_r{1..6}.jsonl` | Step 0b, the start reference at skill 3 |
| `runs/p2_l/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the run |
| `runs/p2_l/ladder.jsonl` | ladder passes at 10, 20, 30, 40 (ECE and its bins, N), measured on this box |
| `runs/p2_l/sf3/reps_iter00{10,20,30}_r{1,2,3}.jsonl`, `reps_iter0040_r{1..6}.jsonl`, `p2_verdict_reps_iter00{10..40}.json` | K2 at skill 3 |
| `runs/p2_l/sf3/p2_verdict_reps.json` | K1 |
| `runs/p2_l/reps_iter00{10..40}_r{1,2,3}.jsonl`, `p2_verdict_reps_iter00{10..40}.json`, `p2_verdict_reps.json`, `p2_m4_iter0040.json` | K3, skill 0 |
| `runs/p2_l/startpos/` | K4 |
| `runs/p2_l/book_rows.json` | L2 |
| `runs/p2_l_campaign.console.log`, `runs/p2_l.console.log`, `runs/p2_l_step0.console.log`, `runs/p2_l_finish.console.log`, `runs/p2_l_step0.sh`, `runs/p2_l_finish.sh`, `runs/p2_l_rung.txt` | consoles and scripts (untracked) |
