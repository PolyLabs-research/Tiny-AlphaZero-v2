# RUNLOG — p2_w (the quarter line's batch with 256 more teacher rows on top, beside a second run of the quarter line; judged on skill 5)

**Runs:** line A, "wide": `runs/p2_w_wide` (`--config configs/p2_w_wide_hot.yaml --seed 43`: 48 simulations, lr 2e-5,
batch 1,280 = 717 self-play + 20 + 31 rehearsal + 512 teacher rows from `runs/p2_u_fresh/teacher`, 20 iterations from
`p2_u_stale_down/0010`), then `runs/p2_w_wide_down` (`configs/p2_w_wide_down.yaml`: 48, 2e-6, the same batch, 10
iterations from `p2_w_wide/0020`). Line B, "quarter": `runs/p2_w_quarter` (`configs/p2_u_fresh_hot.yaml`: 48, 2e-5,
batch 1,024 with 256 teacher rows, 20 iterations from the same start), then `runs/p2_w_quarter_down`
(`configs/p2_u_fresh_down.yaml`: 48, 2e-6, 10 iterations from `p2_w_quarter/0020`): p2_v's line B again, the same
configs, seed and start. Four RunPod pods, each a secure RTX 4090: wide on 36 vCPU (`tjcanxtgofkgi1`, a 49,140 MiB
variant), quarter on 32 vCPU (`m6qnkmgamgkprf`, 24,564 MiB), quarter down on 36 vCPU (`4y1zrxn78039yn`, 49,140 MiB),
wide down on 64 vCPU (`r9tvpoxmxie6d8`, 24,564 MiB). Training 2026-11-11 14:59 to 18:50 (wide); 14:59 to 18:40
(quarter); 19:19 to 21:07 (quarter down); 20:39 to 22:54 (wide down) UTC, code `67641b0`. No ring was labelled.
Every measurement ran on this box (torch 2.14.1+rocm7.2) under the user units
`taz-campaign@p2_w_{wide,wide_down,quarter,quarter_down}` (`MEASURE_AFTER_PULL=1`) and `taz-finisher@p2_w`
(`runs/p2_w_finish.sh`). **Pre-registration:** `PREREG-p2_w.md` (`67641b0`), no addenda. Every verdict below was
registered before it was measured, except where marked.

**Verdict: the registered fourth row, "not separated". Wide sits between the lines and the round resolves neither
difference. The dose stays at 0.25 and the batch at 1,024; the "more of this teacher" lever is closed. K0 (the
quarter line against itself, quarter down/0010 − p2_v's quarter down/0010, skill 5, six reps per side): +0.001
[−0.036, +0.038], includes zero, so K1 and K1b are read as the earlier rounds' comparisons were. K1 (wide down/0010 −
quarter down/0010): −0.027 [−0.061, +0.007]. K1b (wide down/0010 − p2_v's half down/0010): +0.018 [−0.018, +0.055].
T1 holds by 0.065 nats (held-out cross-entropy 2.302 against 2.367), so the extra rows were applied. K2 passes (192
against 188). ECE and N hold; L1, L3, L4 and L5 hold. K0b, reported: the second quarter line against p2_v's half, +0.045
[+0.012, +0.078], p2_v's K1 again on a second draw of its quarter line. Every written prediction held. Michael chose
the public benchmark for p2_x, a UCI wrapper first (§6).**

---

## 1. Before launch

The start reference is `p2_u_stale_down/0010`, as p2_v's: six skill-5 book reps (`runs/p2_u_stale_down/sf5/`, mean
0.3608), three skill-3 reps (0.8333), three skill-0 book reps (0.9750) and three initial-position reps (0.9875);
nothing was re-measured. The p2_v lines are read as p2_v measured them: six skill-5 reps each at
`runs/p2_v_{quarter,half}_down/sf5/reps_iter0010_r{1..6}.jsonl`, unchanged from `8c9f61d` (checked at close with
`git diff 8c9f61d`). Read at registration, 2026-11-11 14:07 UTC: balance $33.68, no pods of ours, secure RTX 4090
at $0.89/h; `make test` 1,162 passed, 4 skipped on the draft's tree (`e27174b`).

**The pods.** All four came from `cloud/launch_run.sh` (`runs/p2_w_launch.sh`), each a transient unit. Both hot
stages were refused eight times (14:07:59 to 14:43:41, no instances) and launched at 14:48:16 (wide) and 14:48:44
(quarter). Each step down waited for its hot stage's `.remote_done`: quarter down was refused six times (18:40:29 to
19:05:37) and launched at 19:10:39; wide down 20 times (18:50:28 to 20:26:04) and launched at 20:31:06. The longest
wait was 101 minutes (p2_v's 86). No pod was unusable, none replaced. Every input's sha256 matched on each pod (65
files each, the teacher ring among them). Nothing was committed to `main` between registration and the second step
down's launch, so all four runs carry one `git_sha`.

## 2. L — the continuity applied

- **L1 holds, all four runs:** `seed` 43; `git_sha` `67641b0` (provenance and every iteration row), `git_dirty`
  false; each `config_fingerprint` equal to its `config.yaml`'s sha256: `99a0b4ae…ee13` (wide), `b812a7b4…c12c`
  (wide down), `066043ca…d67d` (quarter), `036cd665…01da` (quarter down), the registered four.
  - wide and quarter: `init_from` `runs/p2_u_stale_down/ckpt_iter0010.pt`; pool and ring from
    `runs/p2_u_stale_down`; `seed_replay` 500,000 rows, `donor_committed` = `donor_head` = 1,037,734, offset −10;
    `lr` 2e-5 on every row.
  - wide down: `init_from` `runs/p2_w_wide/ckpt_iter0020.pt`; `seed_replay` 500,000, committed = head = 1,589,041
    (wide's final ring), offset −20; `lr` 2e-6 on every row.
  - quarter down: `init_from` `runs/p2_w_quarter/ckpt_iter0020.pt`; `seed_replay` 500,000, committed = head =
    1,585,587, offset −20; `lr` 2e-6 on every row.
  - Simulations per stored position **48.0** at every iteration of all four. In both wide configs: `batch_size`
    1280, `teacher_frac` 0.4, `rehearsal_frac` 0.03984375 (`rehearsal_mate1_frac` 0.015625, `rehearsal_mate2_frac`
    0.02421875); in both quarter configs 1024, 0.25, 0.05 (0.02, 0.03); `teacher_dir` `runs/p2_u_fresh/teacher` in
    all four. **Line B's two `config.yaml` files are byte-identical** to `runs/p2_v_quarter/config.yaml` and
    `runs/p2_v_quarter_down/config.yaml`.
- **L2 (reported, no bound):** wide 3 (`open_021` at 20, `open_001` at 11, `open_009` at 13), wide down 1
  (`open_012` at 5), quarter 3 (the same three), quarter down 1 (`open_012` at 5) in 500,000. Predicted 0–5 per
  run: held. The same openings at the same iterations as p2_v's half line; the random-opening schedule is seeded,
  and the lines share the seed.
- **L3 holds:** the launcher verified the staged `runs/p2_u_fresh/teacher/` against this box's sha256 on all four
  pods; its `teacher.json` reads 475,696 rows.
- **L4 holds:** `git diff 56328bb 67641b0 -- src scripts cloud` is empty; Stockfish `16-1build1` and torch
  2.14.1+rocm7.2 on this box; the twelve p2_v rep files are as committed at `8c9f61d`. The pods ran the image of
  p2_g–p2_v (torch 2.12.1+cu129).
- **L5 holds: line B is a second realisation, not a copy.** Its first iteration logged 52,998 positions against
  p2_v's quarter's 53,146 (wide 52,847; p2_v's half 52,779: four counts from one seed, one start network and self-play
  before any training step), and `runs/p2_w_quarter_down/ckpt_iter0010.pt` has sha256 `43494548…3857` against
  p2_v's `c7a97b23…d755`. K0 is not void. Predicted: held.

## 3. T1, H12 and H13 — the lever applied

`scripts/teacher_agreement.py` (prior only, no search), `runs/p2_w_*/teacher_agreement_*.json`; the held-out ring is
`runs/p2_u_fresh/teacher_holdout` (5,000 rows), the training ring `runs/p2_u_fresh/teacher` (475,696):

| checkpoint | held-out: top-1 | held-out: cross-entropy | held-out: value MSE | training ring: top-1 | training ring: cross-entropy | H12 (training − held-out) |
|---|---|---|---|---|---|---|
| start, `p2_u_stale_down/0010` | 0.3590 | 2.391 | 0.0395 | 0.3600 | 2.393 | +0.002 |
| wide 0020 | 0.3642 | 2.302 | 0.0482 | | | |
| quarter 0020 | 0.3644 | 2.367 | 0.0470 | | | |
| **wide down 0010 (A's endpoint)** | **0.3628** | **2.302** | 0.0396 | 0.3753 | 2.282 | **−0.020** |
| **quarter down 0010 (B's endpoint)** | **0.3658** | **2.367** | 0.0378 | 0.3680 | 2.355 | **−0.013** |
| p2_v half down 0010 (as run) | 0.3664 | 2.272 | 0.0426 | 0.3791 | 2.250 | −0.021 |
| p2_v quarter down 0010 (as run) | 0.3614 | 2.369 | 0.0366 | 0.3682 | 2.354 | −0.015 |

- **T1 holds:** wide's endpoint has the lower cross-entropy on the held-out ring, by **0.065 nats**. Predicted wide
  2.27–2.33 and line B 2.35–2.39: both held (2.302, 2.367). Top-1 agreement is line B's by 0.3 points (reported,
  not judged).
- From the start, wide's held-out cross-entropy fell 0.089 nats, line B's 0.024 (p2_v: half 0.119, quarter 0.022).
  The three lines order by teacher rows per self-play row (half 1.111, wide 0.714, quarter 0.357) on this reading.
- As in p2_v, both lines reached their final cross-entropy by the end of the 2e-5 stage (2.302 and 2.367 at 0020);
  the step down did not move it.
- **H12 (reported, no bound):** wide −0.020, line B −0.013. Predicted −0.01 to −0.04: held, both. On top-1 the
  training ring reads 1.3 points above held-out in wide and 0.2 in line B.
- **H13 (reported, no bound), `grad_norm` before clipping:** wide 4.15–7.82 (mean 4.83), wide down 5.19–7.81
  (6.51); quarter 4.68–8.46 (5.31), quarter down 5.53–8.31 (7.29). p2_v: half 4.34–9.29 (5.24), half down
  5.04–7.17 (5.88), quarter 4.77–8.55 (5.28), quarter down 5.69–8.23 (6.97). Every step in every line was clipped
  (`grad_clip` 1.0). Wide's norms ran lower than line B's in both stages, as a larger batch averages more rows.

## 4. K — the lines at the end, skill 5

Per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples,
seed 0). All four registered files were reproduced byte for byte at close. All 24 endpoint rows are
`stockfish_s5`, each set on its own checkpoint:

| line | reps r1–r6 | mean | W / D / L (1,200 games) |
|---|---|---|---|
| p2_v quarter, `p2_v_quarter_down/0010` (as run) | 0.355, 0.42, 0.36, 0.41, 0.365, 0.3575 | 0.3779 | 313 / 281 / 606 |
| p2_v half, `p2_v_half_down/0010` (as run) | 0.335, 0.3525, 0.3275, 0.325, 0.325, 0.3375 | 0.3338 | 277 / 247 / 676 |
| **B, `p2_w_quarter_down/0010`** | 0.4275, 0.3575, 0.39, 0.38, 0.3275, 0.39 | **0.3788** | 299 / 311 / 590 |
| **A, `p2_w_wide_down/0010`** | 0.395, 0.44, 0.305, 0.305, 0.355, 0.3125 | **0.3521** | 286 / 273 / 641 |

### K0 — the quarter line against itself (read first)

`p2_w_quarter_down/0010` − `p2_v_quarter_down/0010`
(`runs/p2_w_quarter_down/sf5/p2w_verdict_quarter_replicate.json`): **+0.001 [−0.036, +0.038]**, SE 0.0190.
**Includes zero.** Predicted line B's endpoint 0.35–0.38 (held, 0.3788) and the point −0.03 to +0.005 (held);
about two in three that the CI includes zero (held). Per opening, line B is above p2_v's quarter on 43, below on
49, equal on 8 (median 0.000). As registered, this licenses reading K1 and K1b as the earlier rounds' comparisons
were read, and no more: it is one pair, resolved to about ±0.037.

**K0b (reported, no decision):** `p2_w_quarter_down/0010` − `p2_v_half_down/0010`
(`p2w_verdict_quarter_vs_p2v_half.json`): **+0.045 [+0.012, +0.078]**, SE 0.0169, entirely above zero. p2_v's K1
was −0.044 [−0.078, −0.010] (half − quarter): the second draw of the quarter line reproduces it to 0.001.

### K1 (primary) — wide against quarter, concurrent

`p2_w_wide_down/0010` − `p2_w_quarter_down/0010` (`runs/p2_w_wide_down/sf5/p2w_verdict_wide_vs_quarter.json`):
**−0.027 [−0.061, +0.007]**, SE 0.0174. **Includes zero.** Predicted the point −0.03 to +0.02 (held) and more likely
than not the CI includes zero (held). Per opening, wide is above line B on 39, below on 51, equal on 10 (median
−0.042). Wide drew less (0.228 against 0.259) and lost more (641 against 590). Wide's six reps span 0.305–0.44, the
widest spread of the four endpoints here.

### K1b — wide against p2_v's half

`p2_w_wide_down/0010` − `p2_v_half_down/0010` (`p2w_verdict_wide_vs_half.json`): **+0.018 [−0.018, +0.055]**, SE
0.0184. **Includes zero.** Predicted the point 0.00 to +0.05 (held) and more likely than not the CI includes zero
(held). Per opening, 46 above, 46 below, 8 equal (median 0.000).

### K2 — mate-in-2

At the endpoints: **wide down/0010 192/200, quarter down/0010 188/200**; 192 ≥ 188 − 4 and ≥ 183. **K2 passes.**
Predicted wide 190–194: held. At the other measures: wide 0020 190, wide down/0005 192; quarter 0020 189, quarter
down/0005 188; the start 191; p2_v's quarter 191, 193, 193. Six positions failed at all six measures of both lines:
`mate2_091`, `mate2_124`, `mate2_126`, `mate2_127`, `mate2_128`, `mate2_129` (the start fails all six).
`mate2_138` failed at every line-B measure and at no wide measure; `mate2_054`, `mate2_140` and `mate2_141` at
every line-B measure and at one wide measure each (054 at the endpoint, 140 and 141 at 0020). Line B's three
readings (189, 188, 188) print the ladder's "M3 CLAUSE FAILED", its comparison with the old 0.95 milestone; the
registered floor is 183.

### K3 — each line against the start, and skill 3 (informational)

| checkpoint | reps | vs start [95% CI] |
|---|---|---|
| wide 0020, skill 5 (end of 2e-5) | 0.285, 0.2775, 0.33 | **−0.063 [−0.100, −0.026]** |
| quarter 0020, skill 5 (end of 2e-5) | 0.355, 0.32, 0.36 | −0.016 [−0.057, +0.025] |
| wide down 0005, skill 5 | 0.3275, 0.31, 0.38 | −0.022 [−0.063, +0.020] |
| quarter down 0005, skill 5 | 0.3625, 0.38, 0.365 | +0.008 [−0.026, +0.043] |
| **wide down 0010, skill 5 (six)** | mean 0.3521 | **−0.009 [−0.043, +0.026]** |
| **quarter down 0010, skill 5 (six)** | mean 0.3788 | **+0.018 [−0.015, +0.051]** |
| wide down 0010, skill 3 | 0.8075, 0.8075, 0.83 | −0.018 [−0.062, +0.022] |
| quarter down 0010, skill 3 | 0.8, 0.8625, 0.7875 | −0.017 [−0.053, +0.020] |

Predicted: both 0020s below the start (held as points; only wide's interval is entirely below zero); both endpoints
within noise of it (held). At skill 3 the endpoints are equal (0.815 and 0.817; p2_v's quarter 0.820) and neither
differs from the start.

**Computed at close, not registered** (same method; `p2w_verdict_unreg_*.json`): wide − quarter at 0020 **−0.048
[−0.092, −0.003]**; at down/0005 −0.030 [−0.073, +0.013]. Each line's step, down/0010 (six) − 0020 (three): wide
**+0.055 [+0.013, +0.096]**, quarter +0.034 [−0.009, +0.077] (p2_v: quarter +0.070, half −0.005). Line B against
p2_v's quarter at 0020 +0.037 [−0.009, +0.082], at down/0005 +0.003 [−0.042, +0.047].

### K4 — skill 0 and the initial position (continuity)

Skill-0 book reps against the start (0.9750): wide 0020 −0.003 [−0.018, +0.013], wide down/0005 +0.008, wide
down/0010 +0.008 [−0.006, +0.023]; quarter 0020 +0.006, quarter down/0005 −0.003, quarter down/0010 +0.008 [−0.006,
+0.023]. Initial position against the start (0.9875): wide 0020 −0.012, wide down/0010 −0.013 [−0.028, +0.003];
quarter 0020 −0.005, quarter down/0010 −0.011 [−0.023, +0.001]. No interval excludes zero. M4 at skill 0: wide down
0.983 [0.973, 0.993], quarter down 0.983 [0.974, 0.992], met (bar 0.675).

## 5. N, ECE, H

- **N — holds:** mate-in-2 wide 190; wide down 192, 192; quarter 189; quarter down 188, 188. Mate-in-3 wide 165;
  wide down 164, 165; quarter 161; quarter down 161, 164 (start 156). Predicted 185–195: held. The watchlist reads
  the same at all six measures: `mate2_129` CAPABILITY, `mate2_165` and `mate2_166` SOLVED.
- **ECE — holds:** wide 0.0828; wide down 0.1039, 0.1008; quarter 0.1013; quarter down 0.1088, 0.1031 (< 0.15;
  start 0.1049; p2_v's quarter endpoint 0.0958; predicted 0.08–0.13: held).
- **H7 — the losses.** Wide's totals are not comparable with line B's, as registered. Line B beside p2_v's quarter
  (H14): policy 1.754 → 1.751 (p2_v 1.754 → 1.746), down 1.748 → 1.743 (1.744 → 1.740); value 0.392 → 0.335
  (0.391 → 0.332), down 0.339 → 0.382 (0.336 → 0.383). Wide: policy 1.876 → 1.876, down 1.873 → 1.875; value
  0.396 → 0.324, down 0.325 → 0.369. The value loss fell at 2e-5 and rose back through the step down in both lines,
  as in p2_s–p2_v.
- **H1:** `move1_prior_entropy` fell through both hot stages (wide 0.38 → 0.18, quarter 0.38 → 0.14) and moved
  little through the step downs (wide down 0.14 → 0.14, quarter down 0.09 → 0.17). The search target at move 1
  stayed below 0.12 in all four runs. **Book move-1 entropy (H6): wide 1.480, 1.477, 1.471; line B 1.242, 1.268,
  1.281** (start 1.227; p2_v half 1.61, quarter 1.23). Predicted wide between 1.23 and 1.61: held.
- Promotion loss by halves: wide 0.375 / 0.340, wide down 0.381 / 0.414; quarter 0.377 / 0.346, quarter down
  0.378 / 0.413 (rose through both step downs again). Decisive rate wide 0.778 / 0.775, wide down 0.780 / 0.759;
  quarter 0.785 / 0.780, quarter down 0.770 / 0.782. Self-play games 106.4, 106.8 (A) and 106.0, 106.6 (B) plies.
- **H2:** 0 skipped steps in 8,000 + 4,000 per line.
- **H10, pace:** wide 11.31 min per iteration (self-play 10.14), wide down 12.97 (11.76); quarter 10.93 (9.69),
  quarter down 10.64 (9.72). Sims/s 4,159–4,405 (wide, 36 vCPU), 3,382–4,057 (wide down, 64), 4,200–4,796
  (quarter, 32), 4,351–4,643 (quarter down, 36). Training 70, 73, 75 and 55 s per iteration: the larger batch
  did not show in the training time, across different hosts. The 64-vCPU pod was the slowest at self-play for the
  third round running; the pods are different hosts, so nothing is attributed.
- **H14, the replicate elsewhere (reported, no bound):** line B beside p2_v's quarter: T1 2.367 (2.369); mate-in-2
  188 (193); ECE 0.1031 (0.0958); skill 3 0.817 (0.820); skill 5 at 0020 0.345 (0.308), at down/0005 0.369
  (0.367); the step +0.034 (+0.070); H6 1.281 (1.234); H12 −0.013 (−0.015); mate-in-3 164 (163).

Rungs: random and greedy won 200/200 at every checkpoint in all four runs; phase1 200/200 except at quarter down
(198 won, 2 drawn at 0005; 199 won, 1 drawn at 0010).

## 6. Decision

Per the table's fourth row (K0 includes zero; K1 and K1b include zero; T1 holds): **not separated. Wide sits
between the lines (0.352, against p2_v's half 0.334 and the quarter line's 0.378 and 0.379) and the round resolves
neither difference. Read as consistent with a cost that grows with the teacher's share of the batch; not read as
support for H_rows. The dose stays at 0.25 and the batch at 1,024; the "more of this teacher" lever is closed.**
N and ECE hold at wide's endpoint, so no guard row applies.

**p2_x: Michael chose the public benchmark (2026-11-12), a UCI wrapper first.** It needs its own registration.

Readings the table does not cover:

1. **Two runs of the quarter protocol landed together.** Endpoint 0.3779 and 0.3788; at down/0005 0.367 and 0.369;
   T1 2.369 and 2.367. They differed most at 0020 (+0.037, inside its interval) and on mate-in-2 (193 against 188),
   so K2's margin of 4 is inside the run-to-run spread seen here. Per-opening rep-means did not correlate between
   the two runs (r = −0.10), and neither do the two halves of one checkpoint's own six reps (r from −0.16 to +0.12
   at four six-rep checkpoints; unregistered): at this resolution the per-opening differences are game-level noise, and K0
   shows no run-level spread above it. One pair; it does not bound the spread.
2. **p2_v's result now stands against two draws of the quarter line** (K0b +0.045 [+0.012, +0.078]). The half line
   is still one run.
3. **Wide's dip formed at 2e-5, and the step down returned it** (unregistered). Wide's 0020 was the lowest 2e-5
   reading of the four lines of p2_v and p2_w (−0.063 against the start), and wide − quarter there was −0.048
   [−0.092, −0.003]; the step down returned +0.055 [+0.013, +0.096], where p2_v's half line got nothing back (−0.005). At 0.4 of a larger
   batch the 2e-6 stage behaved as it did at 0.25, not as at 0.5 of the standard batch. Whether that follows the
   self-play rows or the ratio, the round does not say.
4. **Closer to the teacher, not stronger, again.** Across p2_v and p2_w the three batches order the same way on
   held-out cross-entropy (half 2.272 < wide 2.302 < quarter 2.367/2.369) and in reverse at skill 5 (half 0.334 <
   wide 0.352 < quarter 0.378/0.379), and in teacher rows per self-play row (1.111, 0.714, 0.357). Only the half–
   quarter difference separates at 95%. The registration said a wide endpoint between the lines is the result that
   would make per-source loss weights the next round (holding the self-play rows' weight at 0.700 as well as their
   number; a code change). Michael chose the benchmark; the weights go on the open list.
5. **The plateau.** No six-rep endpoint of p2_t–p2_w separates from its own start at skill 5. The best six-rep
   skill-5 mean is now 0.379 (line B), 0.018 above its start. With the teacher lever closed, a public benchmark
   gives the first absolute reading of where the 6.5M network stands.

The open list, carried forward: a deeper engine; the 2e-7 step-down; per-source loss weights (reading 4); the
calibration sample; the promotion loss at 2e-6 (it rose through both step downs again: 0.378 → 0.413 in line B);
L2's exclusion at storage time; the host crashes (none this round; this box stayed up throughout, up since
2026-11-05); the third line that separates old positions from the teacher's targets (p2_t §6, reading 4). **Disk:
86 GB free at close** (this round's four run directories hold 6.6 GB).

## 7. Mechanics

- **Unattended throughout.** `launch_run.sh` launched, staged, verified and started all four pods and enabled their
  drivers; each step down launched itself after its hot stage's final pull and the refusals above. No hand
  intervention, no measurement retried, no `FAILED` line in the finisher's log.
- **Lifecycle:** final pulls with the ring at 18:40:08 (quarter), 18:50:13 (wide), 21:07:14 (quarter down) and
  22:53:53 (wide down); each pod deleted itself after `PULLED` (0 pods of this round at close, read 2026-11-12).
  Measurements under `gpu hold`: quarter 0020 18:40–19:21; wide 0020 19:22–20:04; quarter down 0005 21:07–21:34,
  0010 –22:14; wide down 0005 22:53–23:20, 0010 –00:02. The finisher from 00:03:25: T1 at the four checkpoints on
  the held-out ring, H12 at both endpoints on the training ring (4.5 min each), skill-5 reps wide/0020,
  quarter/0020, wide down/0005, quarter down/0005, wide down/0010 (six), quarter down/0010 (six), skill-3 reps at
  both endpoints, M4, L2, L5, and K0, K0b, K1, K1b at 04:30:03 UTC on 2026-11-12.
- **Cost:** pod time at $0.89/h ≈ **$10.87** (wide 4.03 h $3.59; quarter 3.86 h $3.43; quarter down 1.94 h $1.73;
  wide down 2.38 h $2.12). Expected about $11.50, worst case $24.92. **The balance readings do not isolate the
  round:** they fell $8.74 between the hot launches ($33.68) and quarter down's launch ($24.94), against $7.02 of
  this round's pod time, and $2.56 in the next window (to $22.38 at wide down's launch) against $1.19. The account
  was paying for other pods too (another project's pod was running on it at close), and no reading was taken after
  the last pull. The pod-time figure is the round's cost. All RunPod use by this project to date about $170.

## 8. Artefacts

| file | what |
|---|---|
| `runs/p2_w_{wide,wide_down,quarter,quarter_down}/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the runs |
| `runs/p2_w_*/teacher_agreement_iter*.json` | T1 (the held-out ring) |
| `runs/p2_w_{wide,quarter}_down/teacher_agreement_train_iter0010.json` | H12 (the training ring) |
| `runs/p2_w_*/ladder.jsonl` | ladder passes at wide and quarter 20, both step downs at 5 and 10 (ECE, N, H6) |
| `runs/p2_w_*/sf5/` | skill-5 reps and each checkpoint's verdict against the start (K3) |
| `runs/p2_w_quarter_down/sf5/p2w_verdict_quarter_replicate.json`, `p2w_verdict_quarter_vs_p2v_half.json` | K0, K0b |
| `runs/p2_w_wide_down/sf5/p2w_verdict_wide_vs_quarter.json`, `p2w_verdict_wide_vs_half.json` | K1, K1b |
| `runs/p2_w_wide/sf5/p2w_verdict_unreg_wide_vs_quarter_iter0020.json`, `runs/p2_w_wide_down/sf5/p2w_verdict_unreg_wide_vs_quarter_iter0005.json`, `runs/p2_w_{wide,quarter}_down/sf5/p2w_verdict_unreg_step.json`, `runs/p2_w_quarter/sf5/p2w_verdict_unreg_quarter_replicate_iter0020.json`, `runs/p2_w_quarter_down/sf5/p2w_verdict_unreg_quarter_replicate_iter0005.json` | computed at close, not registered |
| `runs/p2_w_{wide,quarter}_down/sf3/` | skill-3 reps at the endpoints and their verdicts |
| `runs/p2_w_*/reps_iter*_r*.jsonl`, `p2_verdict_reps*.json`, `p2_m4_iter*.json`, `startpos/` | K4, skill 0 |
| `runs/p2_w_*/book_rows.json` | L2 |
| `runs/p2_w_*_launch.console.log`, `runs/p2_w_*_campaign.console.log`, `runs/p2_w_*.console.log`, `runs/p2_w_finish.console.log`, `runs/p2_w_finish.sh`, `runs/p2_w_launch.sh` | consoles, launcher and finisher (untracked) |
