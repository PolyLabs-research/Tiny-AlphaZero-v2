# RUNLOG — p2_n (the default round at 2e-5, forty iterations from p2_m_init/0020, judged on skill 3)

**Run:** `runs/p2_n` (`configs/p2_k_lr.yaml` unchanged, lr 2e-5), 40 iterations, trained on RunPod pod
`puwezkdkqskp7l`, a secure RTX 4090 of the 24 GB variant (`vram_gb` 23.52), 32 vCPU on a 31-CPU quota,
from 2026-11-01 12:05:22 to 19:13:28 UTC, code `e744125`, warm-started from `p2_m_init/0020` with the
league and the replay ring continued from `runs/p2_m_init`. Every measurement ran on this box (torch
2.14.1+rocm7.2): the ladder, skill-0 reps and ECE by `cloud/remote_campaign.sh` with
`MEASURE_AFTER_PULL=1`, the skill-3 reps and the verdicts by `runs/p2_n_finish.sh`.
**Pre-registration:** `PREREG-p2_n.md` (`e744125`), no addenda. Every verdict below was registered
before it was measured, except where marked.

**Verdict: H_cont NOT SUPPORTED — the registered "plateau at 2e-5" row, as predicted. K1 (0040 against
p2_m_init/0020, skill 3, six reps each side): +0.013 [−0.016, +0.043]. The round's shape, measured with
six reps at 10 as well as at 40: no early step (K2a −0.012 [−0.045, +0.021]) and no climb after it (K2b
+0.025 [−0.010, +0.060]) — the registered plateau shape. N (191–193) and ECE (0.094–0.103) hold; L1
holds; L2 reported (4 rows). The training losses kept falling and the move-1 prior entropy collapsed
(0.51 → 0.06) while play stayed flat.**

---

## 1. Before launch

The start reference at skill 3 is p2_m_init/0020's six book reps (0.6025, 0.6575, 0.615, 0.625, 0.6575,
0.6275; mean 0.6308); at skill 0 its three book reps (0.9525, 0.9375, 0.935) and three initial-position
reps. Nothing was re-measured.

**The launch guard's margin.** The draft read the balance at $14.37, which the $5 margin would have
refused against the $10.68 worst case. The account auto-reloads at $10 (Michael), and at his instruction
the margin in `~/.config/taz-cloud/limits.json` was lowered to $2 before registration (the per-pod
ceiling stays $15). Read at registration, 11:54 UTC: balance $13.54, secure RTX 4090 at $0.89/h;
`make test` 916 passed, 2 skipped. The auto-reload fired during the run (balance $58.64 at 14:02).

**The pod:** launched at 11:58:29 on the first try (32 vCPU, 125 GB RAM); staging ran without a retry; all
50 inputs' sha256 matched on the pod before the job started at 12:05:22.

## 2. L — the continuity applied

- **L1 holds:** `config_fingerprint` `8ab654e2…298c`; `init_from` `runs/p2_m_init/ckpt_iter0020.pt`;
  `seed_replay.rows` 500,000, `donor_committed` = `donor_head` = 1,576,622, `iteration_offset` −20;
  `git_dirty` false; `git_sha` `e744125` and `lr` 2e-5 on all 40 rows.
- **L2 (reported, no bound):** **4 rows** in the final ring (500,000; `total_written` 2,649,005):
  `open_027` (iteration 38), `open_097` (32), `open_021` twice (37). Initial-position rows 516; distinct
  positions 477,232. Predicted 0–5.

## 3. K — the endpoint and the shape, at skill 3

### K1 (primary) — p2_n/0040 against the start, six reps each side

Per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000
resamples, seed 0; `runs/p2_n/sf3/p2_verdict_reps.json`, reproduced byte for byte; all eighteen skill-3
rows `stockfish_s3`, on their own checkpoints):

| | reps r1–r6 | mean |
|---|---|---|
| start (p2_m_init/0020) | 0.6025, 0.6575, 0.615, 0.625, 0.6575, 0.6275 | 0.631 |
| p2_n/0040 | 0.6375, 0.6125, 0.6525, 0.6725, 0.6225, 0.6675 | 0.644 |
| **0040 − start** | | **+0.013 [−0.016, +0.043]**, SE 0.0149 |

- **The interval includes zero: the plateau row.** Predicted (includes zero, point +0.00 to +0.04); H_cont
  expected it above zero. Per opening, 0040's rep-mean is above the start's on 46 of the 100, below on 46,
  equal on 8 (median 0.000). W/D/L at 0040 over six reps: 677 / 192 / 331 of 1,200.

### K2 — the shape of the round

| reading | reps | result |
|---|---|---|
| **K2a**, 0010 against the start (six each) | 0.6125, 0.5925, 0.6225, 0.625, 0.5775, 0.685 (0.619) | **−0.012 [−0.045, +0.021]** |
| **K2b**, 0040 against 0010 (six each) | | **+0.025 [−0.010, +0.060]** |
| K2c, 0010 against the start (r1–r3) | 0.6125, 0.5925, 0.6225 | −0.022 [−0.061, +0.016] |
| K2c, 0020 | 0.575, 0.625, 0.6725 | −0.007 [−0.046, +0.031] |
| K2c, 0030 | 0.64, 0.695, 0.615 | +0.019 [−0.018, +0.057] |
| K2c, 0040 | 0.6375, 0.6125, 0.6525 | +0.003 [−0.035, +0.040] |

- **The registered reading: both K2a and K2b include zero, with K1 including zero — a plateau.** No dip
  (predicted). Slope +0.010 per ten iterations.
- **No early step this round.** p2_k_lr, p2_l and both p2_m arms each had most of their gain by
  iteration 10; p2_n, measured at 10 with six reps, has none. The early step was present while the
  rounds still had a gain to take and is absent now that there is none; three-rep noise at 10 cannot be
  ruled out as part of the earlier readings.

### K3 — skill 0, the continuity reading

Three skill-0 reps per checkpoint against p2_m_init/0020's three: +0.014 [−0.008, +0.037], +0.001,
+0.003, +0.023 [+0.002, +0.043]. M4 (bar 0.675) met at every checkpoint: 0.956, 0.943, 0.945, **0.964
[0.951, 0.976]** at 40 (predicted). The ladder's single skill-0 replay (*not registered*): 0.9625,
0.9575, 0.965, 0.955.

### K4 — informational: the initial position (skill 0)

0040's three reps 0.9725, 0.97, 0.9325 against p2_m_init/0020's: **+0.011 [−0.010, +0.031]**, at the
ceiling.

## 4. N, ECE, H

- **N — holds:** mate-in-2 **192, 191, 193, 191** at 10/20/30/40 (≥ 183; predicted 183–192, 193 at 30 is
  one above), the highest of any round from p2_i to p2_n (p2_j's lost attempt 1 read 194 at its 0020).
  mate-in-3 151, 154, 153, 153. Failures at 40
  (9): `mate2_054`, `078`, `091`, `124`, `125`, `126`, `129`, `141`, `196`. Watchlist: `mate2_165` and
  `166` solved throughout, `129` unsolved.
- **ECE — holds:** 0.1034, 0.0935, 0.1029, **0.0998** (< 0.15; predicted 0.08–0.12). The bins at 40
  (*not registered*) show the underconfidence RUNLOG-p2_l read, wider: positions rated 0.54–0.65 are won
  86–97% of the time (bin 5: predicted 0.544, realized 0.863; bin 6: 0.647, 0.968), against opponents
  (phase1, skill 0 at 0.96) the model beats.
- **H1 (direction-free) — the move-1 prior collapsed:** `move1_prior_entropy` 0.51 at iteration 1, 0.21 at
  10, 0.15 at 20, 0.07 at 30, 0.09 at 40 (0.06–0.09 from iteration 23 on). p2_l read 0.46–0.70, p2_m
  0.26–0.50; p2_h's plateau came with 0.013. It is the largest fall within one round from p2_h to p2_n.
- **H2:** 0 skipped steps in 16,000 (< 20).
- **H3–H5, H7** (by quarters): value loss 0.340 / 0.333 / 0.330 / 0.327, policy loss 1.525 / 1.519 /
  1.517 / 1.511 — **both falling all round**, while skill at skill 3 did not move; decisive rate 0.779 /
  0.779 / 0.785 / 0.772; promotion loss 0.365 / 0.369 / 0.380 / 0.369.
- **H8:** first five iterations against the last five: value 0.341 → 0.325, policy 1.527 → 1.510.
- **H6** (`book_move1_entropy`): 0.540, 0.568, 0.538, 0.533.

Rungs: random, greedy and phase1 won 200/200 at every checkpoint.

## 5. Mechanics

- **Pace:** 10.70 min per iteration (10.20–11.20), 4,252–4,639 sims/s (mean 4,405); 7.1 h of training.
- **Lifecycle:** iteration 40 committed 19:13:28; final pull with the ring 19:17:07, `PULLED` written, the
  pod deleted itself. Measurement under `gpu hold`: ladder and skill-0 reps 0010 19:17–19:46, 0020
  –20:15, 0030 –20:44, 0040 with the initial-position reps –21:28; skill-3 reps (finisher) 0010 (six)
  21:29–22:18, 0020 –22:42, 0030 –23:06, 0040 (six) –23:55; verdicts at 23:54:37 UTC. No retries.
- **Cost:** ≈ **$6.50** at the pod's $0.89/h (11:58 to 19:17, 7.3 h); expected about $7.5, worst case
  $10.68. All RunPod use to date about $72.

## 6. Decision

Per the table's second row (K1 includes zero): **plateau at 2e-5. p2_o is the 2e-6 step-down as an A/B
from p2_n/0040 against a continuation arm at 2e-5, twenty iterations each, judged at skill 3 — p2_k's
design one decade down.** N and ECE hold at 0040. Any launch is a new PREREG and Michael's cost OK.

Readings the table does not cover:

1. **The plateau has the same signature as p2_h's.** Training losses falling round-long (value −4%,
   policy −1%), skill flat, and the move-1 prior collapsing from 0.51 to 0.07 — p2_h read 0.013 at its
   plateau, and p2_k's step-down followed. The step-down is the registered lever; the collapse is a
   reading to carry into p2_o (both arms), not a cause this record establishes.
2. **The early step is gone.** With six reps at 10 there is no early gain (K2a −0.012); the shape the
   last three RUNLOGs discussed belonged to rounds that still had a gain to take.
3. **Mate-in-2 is at its best (191–193) and ECE holds at ~0.10**, with the calibration bins pointing, as
   before, at the measurement's sample rather than the model.

The open list, carried forward: p2_o (the 2e-6 step-down A/B); the move-1 prior collapse; the
calibration sample; L2's exclusion at storage time; the host freezes.

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_n/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the run |
| `runs/p2_n/ladder.jsonl` | ladder passes at 10, 20, 30, 40 (ECE and its bins, N) |
| `runs/p2_n/sf3/reps_iter0010_r{1..6}.jsonl`, `reps_iter00{20,30}_r{1,2,3}.jsonl`, `reps_iter0040_r{1..6}.jsonl` | skill-3 reps |
| `runs/p2_n/sf3/p2_verdict_reps.json` | K1 |
| `runs/p2_n/sf3/p2_verdict_reps_iter0010_six.json`, `p2_verdict_reps_0040_vs_0010.json`, `p2_verdict_reps_iter00{10..40}.json` | K2a, K2b, K2c |
| `runs/p2_n/reps_iter00{10..40}_r{1,2,3}.jsonl`, `p2_verdict_reps_iter00{10..40}.json`, `p2_verdict_reps.json`, `p2_m4_iter0040.json` | K3, skill 0 |
| `runs/p2_n/startpos/` | K4 |
| `runs/p2_n/book_rows.json` | L2 |
| `runs/p2_n_campaign.console.log`, `runs/p2_n.console.log`, `runs/p2_n_finish.console.log`, `runs/p2_n_finish.sh` | consoles and the finisher (untracked) |
