# RUNLOG — p2_v (the standing teacher at twice its dose, half of every batch, against a quarter; judged on skill 5)

**Runs:** line A, "half": `runs/p2_v_half` (`--config configs/p2_v_half_hot.yaml --seed 43`: 48 simulations, lr 2e-5,
`teacher_frac` 0.5 from `runs/p2_u_fresh/teacher`, 20 iterations from `p2_u_stale_down/0010`), then
`runs/p2_v_half_down` (`configs/p2_v_half_down.yaml`: 48, 2e-6, 0.5, 10 iterations from `p2_v_half/0020`). Line B,
"quarter": `runs/p2_v_quarter` (`configs/p2_u_fresh_hot.yaml`: 48, 2e-5, teacher 0.25 from the same ring, 20
iterations from the same start), then `runs/p2_v_quarter_down` (`configs/p2_u_fresh_down.yaml`: 48, 2e-6, 10
iterations from `p2_v_quarter/0020`). Four RunPod pods, each a secure RTX 4090: half on 36 vCPU (`cywfjpybzuhvij`, a
49,140 MiB variant), quarter on 32 vCPU (`8gjzfq0sb518r2`), quarter down on 64 vCPU (`0ej6mpjhe3h28e`), half down on
32 vCPU (`ozhljmzaa9yjbs`); the last three 24,564 MiB. Training 2026-11-10 18:37 to 22:12 (half); 18:56 to 22:36
(quarter); 23:17 to 01:27 (quarter down); 23:48 to 01:41 (half down) UTC, code `56328bb`. No ring was labelled.
Every measurement ran on this box (torch 2.14.1+rocm7.2) under the user units
`taz-campaign@p2_v_{half,half_down,quarter,quarter_down}` (`MEASURE_AFTER_PULL=1`) and `taz-finisher@p2_v`
(`runs/p2_v_finish.sh`). **Pre-registration:** `PREREG-p2_v.md` (`56328bb`), no addenda. Every verdict below was
registered before it was measured, except where marked.

**Verdict: the registered fifth row: twice the dose costs skill. The dose stays at 0.25. K1 (half down/0010 −
quarter down/0010, skill 5, six reps per side): −0.044 [−0.078, −0.010], entirely below zero. T1 holds by 0.097 nats
(cross-entropy on the held-out ring 2.272 against 2.369), so the dose was applied: line A ended closer to the teacher
on positions it never trained on, and played worse. K2 passes (192 and 193). ECE and N hold; L1 holds in all four
runs; L3 holds. H12 does not separate the lines (−0.021 against −0.015), which leaves the self-play ring's share as
the first suspect (§6). Against the start neither endpoint separates (half −0.027 [−0.057, +0.003], quarter +0.017
[−0.017, +0.051]). The row asks for a diagnosis before p2_w.**

---

## 1. Before launch

The start reference is `p2_u_stale_down/0010`: six skill-5 book reps (`runs/p2_u_stale_down/sf5/`, mean 0.3608),
three skill-3 reps (0.8333), three skill-0 book reps (0.9750) and three initial-position reps (0.9875); nothing was
re-measured. Read at registration, 2026-11-10 18:27 UTC: balance $44.54, no pods of ours, secure RTX 4090 at
$0.89/h; `make test` 1,127 passed, 4 skipped on the draft's tree (`5f711a3`).

**The pods.** All four came from `cloud/launch_run.sh` (`runs/p2_v_launch.sh`), each a transient unit. Half
launched at the first request (18:27:53); quarter was refused four times (18:28 to 18:43, no instances) and launched
at 18:48:30. Each step down waited for its hot stage's `.remote_done`: half down was refused 17 times (22:13:28 to
23:33:59) and launched at 23:39:02; quarter down 5 times (from 22:42:23) and launched at 23:07:34. The longest wait
was 86 minutes (p2_u's were 3.7 and 5.5 hours). No pod was unusable, none replaced. Every input's sha256 matched on
each pod (65 files each, the teacher ring among them). Nothing was committed to `main` between registration and the
second step down's launch, so all four runs carry one `git_sha`.

## 2. L — the continuity applied

- **L1 holds, all four runs:** `seed` 43; `git_sha` `56328bb` (provenance and every iteration row), `git_dirty`
  false; each `config_fingerprint` equal to its `config.yaml`'s sha256: `d010ccc7…3285` (half), `3ebcb15e…6181`
  (half down), `066043ca…d67d` (quarter), `036cd665…01da` (quarter down), the registered four.
  - half and quarter: `init_from` `runs/p2_u_stale_down/ckpt_iter0010.pt`; pool and ring from
    `runs/p2_u_stale_down`; `seed_replay` 500,000 rows, `donor_committed` = `donor_head` = 1,037,734, offset −10;
    `lr` 2e-5 on every row.
  - half down: `init_from` `runs/p2_v_half/ckpt_iter0020.pt`; `seed_replay` 500,000, committed = head = 1,585,576
    (half's final ring), offset −20; `lr` 2e-6 on every row.
  - quarter down: `init_from` `runs/p2_v_quarter/ckpt_iter0020.pt`; `seed_replay` 500,000, committed = head =
    1,573,989, offset −20; `lr` 2e-6 on every row.
  - Simulations per stored position **48.0** at every iteration of all four. `teacher_dir`
    `runs/p2_u_fresh/teacher` in all four configs; `teacher_frac` 0.5 in both line-A configs and 0.25 in both
    line-B configs.
- **L2 (reported, no bound):** half 3 (`open_021` at 20, `open_001` at 11, `open_009` at 13), half down 1
  (`open_012` at 5), quarter 1 (`open_001` at 11), quarter down 1 (`open_012` at 5) in 500,000. Predicted 0–5 per
  run: held.
- **L3 holds:** the launcher verified the staged `runs/p2_u_fresh/teacher/` against this box's sha256 on all four
  pods; its `teacher.json` reads 475,696 rows (RUNLOG-p2_u §2 describes the ring).

## 3. T1 and H12 — the lever applied

`scripts/teacher_agreement.py` (prior only, no search), `runs/p2_v_*/teacher_agreement_*.json`; the held-out ring is
`runs/p2_u_fresh/teacher_holdout` (5,000 rows), the training ring `runs/p2_u_fresh/teacher` (475,696):

| checkpoint | held-out: top-1 | held-out: cross-entropy | held-out: value MSE | training ring: top-1 | training ring: cross-entropy | H12 (training − held-out) |
|---|---|---|---|---|---|---|
| start, `p2_u_stale_down/0010` | 0.3590 | 2.391 | 0.0395 | 0.3600 | 2.393 | +0.002 |
| half 0020 | 0.3660 | 2.272 | 0.0487 | | | |
| quarter 0020 | 0.3616 | 2.366 | 0.0471 | | | |
| **half down 0010 (A's endpoint)** | **0.3664** | **2.272** | 0.0426 | 0.3791 | 2.250 | **−0.021** |
| **quarter down 0010 (B's endpoint)** | **0.3614** | **2.369** | 0.0366 | 0.3682 | 2.354 | **−0.015** |

- **T1 holds:** line A's endpoint has the lower cross-entropy on the held-out ring, by **0.097 nats**. Predicted
  0.02–0.10: held, at the top of the range. Top-1 agreement is line A's by 0.5 points (predicted within 2: held;
  reported, not judged).
- From the start, line A's held-out cross-entropy fell 0.119 nats and line B's 0.022. p2_u's two lines moved
  0.02–0.03 at this dose; twice the dose moved the network about five times as far toward the teacher.
- Both lines reached their final cross-entropy by the end of the 2e-5 stage (2.272 and 2.366 at 0020); the step
  down did not move it (2.272 and 2.369).
- **H12 (reported, no bound):** line A −0.021, line B −0.015. Predicted −0.01 to −0.05 and 0.00 to −0.02: both
  held. The gap opened 0.006 nats further in line A than in line B, against a 0.097-nat difference on held-out
  positions: 12.9 passes over the ring taught line A the teacher's policy on positions it had not seen, not the
  ring's rows. On top-1 the training ring reads 1.3 points above held-out in line A and 0.7 in line B.
- The value MSE against the teacher's `root_q` ended above the start's in line A (0.0426 against 0.0395) and below
  it in line B (0.0366), after rising at 2e-5 in both, as in p2_t and p2_u.

## 4. K — the two lines at the end, skill 5

### K1 (primary) — half down/0010 against quarter down/0010, six reps per side

Per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples,
seed 0; `runs/p2_v_half_down/sf5/p2v_verdict_half_vs_quarter.json`, reproduced byte for byte at close; all twelve
rows `stockfish_s5`, each side on its own checkpoint):

| line | reps r1–r6 | mean | W / D / L (1,200 games) |
|---|---|---|---|
| B, `p2_v_quarter_down/0010` | 0.355, 0.42, 0.36, 0.41, 0.365, 0.3575 | 0.3779 | 313 / 281 / 606 |
| A, `p2_v_half_down/0010` | 0.335, 0.3525, 0.3275, 0.325, 0.325, 0.3375 | 0.3338 | 277 / 247 / 676 |
| **A − B** | | **−0.044** | **[−0.078, −0.010]**, SE 0.0172 |

- **The interval is entirely below zero.** Forecast: the point −0.01 to +0.04, missed (−0.044); more likely than
  not the CI includes zero, missed. Per opening, A's rep-mean is above B's on 40 of the 100, below on 50, equal on
  10 (median −0.021). Line A drew less (0.206 against 0.234) and lost more (676 against 606). The half-width, 0.034,
  is the registered resolution.
- Line A's six reps span 0.325–0.3525, all below the start's mean; line B's span 0.355–0.42, with two reps above
  0.40.

### K2 — mate-in-2

At the endpoints: **half down/0010 192/200, quarter down/0010 193/200**; 192 ≥ 193 − 4 and both ≥ 183. **K2
passes.** Predicted line A at or 1–4 below line B: held (1 below). At the other measures: half 0020 191, half
down/0005 193; quarter 0020 191, quarter down/0005 193; the start 191. Five positions failed at all six measures of
both lines: `mate2_091`, `mate2_124`, `mate2_126`, `mate2_127`, `mate2_129` (the start fails all five). `mate2_125`
failed at every line-A measure and at no line-B measure; `mate2_196` failed at every line-B measure and at no line-A
measure; the start fails both. Twice the dose did not grow p2_t's mate-in-2 cost.

### K3 — each line against the start, and skill 3 (informational)

| checkpoint | reps | vs start [95% CI] |
|---|---|---|
| half 0020, skill 5 (end of 2e-5) | 0.3225, 0.3375, 0.355 | −0.023 [−0.065, +0.020] |
| quarter 0020, skill 5 (end of 2e-5) | 0.305, 0.2925, 0.3275 | **−0.053 [−0.090, −0.015]** |
| half down 0005, skill 5 | 0.3575, 0.335, 0.29 | −0.033 [−0.076, +0.010] |
| quarter down 0005, skill 5 | 0.355, 0.395, 0.35 | +0.006 [−0.035, +0.047] |
| **half down 0010, skill 5 (six)** | mean 0.3338 | **−0.027 [−0.057, +0.003]** |
| **quarter down 0010, skill 5 (six)** | mean 0.3779 | **+0.017 [−0.017, +0.051]** |
| half down 0010, skill 3 | 0.78, 0.8325, 0.845 | −0.014 [−0.050, +0.022] |
| quarter down 0010, skill 3 | 0.8325, 0.815, 0.8125 | −0.013 [−0.047, +0.022] |

Predicted: both 0020s below the start (held as points; only line B's interval is entirely below zero); line A at or
above the start at its endpoint (missed: −0.027, the interval reaching +0.003); line B within noise of it (held).
At skill 3 the two endpoints are equal (0.819 and 0.820) and neither differs from the start.

**Computed at close, not registered** (same method; `p2v_verdict_unreg_*.json`): A − B at 0020 +0.030 [−0.016,
+0.074]; at down/0005 −0.039 [−0.085, +0.007]. Each line's step, down/0010 (six) − 0020 (three): A −0.005 [−0.049,
+0.040], B **+0.070 [+0.031, +0.108]**.

### K4 — skill 0 and the initial position (continuity)

Skill-0 book reps against the start (0.9750): half 0020 −0.006, half down/0005 +0.013 [−0.001, +0.027], half
down/0010 +0.003 [−0.010, +0.017]; quarter 0020 +0.011, quarter down/0005 +0.002, quarter down/0010 +0.008 [−0.008,
+0.024]. Initial position against the start (0.9875): half 0020 −0.009, half down/0010 −0.013 [−0.028, +0.002];
quarter 0020 −0.006, quarter down/0010 0.000 [−0.013, +0.013]. No interval excludes zero. M4 at skill 0: half down
0.978 [0.968, 0.988], quarter down 0.983 [0.973, 0.993], met (bar 0.675).

## 5. N, ECE, H

- **N — holds:** mate-in-2 half 191; half down 193, 192; quarter 191; quarter down 193, 193. Mate-in-3 half 160;
  half down 161, 163; quarter 160; quarter down 162, 163 (start 156). Predicted 185–195: held. The watchlist reads
  the same at all six measures: `mate2_129` CAPABILITY, `mate2_165` and `mate2_166` SOLVED.
- **ECE — holds:** half 0.0977; half down 0.1048, 0.0976; quarter 0.0963; quarter down 0.0961, 0.0958 (< 0.15;
  start 0.1049; predicted 0.08–0.13: held). Line A at or above line B at every measure, as predicted, by
  0.001–0.009.
- **H7 — the losses.** The totals are not comparable between the lines, as registered: the policy loss sits at
  1.92–1.93 in line A and 1.74–1.75 in line B from the first iteration (half 1.931 → 1.930, half down 1.927 →
  1.927; quarter 1.754 → 1.746, quarter down 1.744 → 1.740), the difference being the teacher rows' share. Neither
  line's policy loss moved by more than 0.01 over 30 iterations.
- **The value loss** fell at 2e-5 and rose back through the step down in both lines, as in p2_s, p2_t and p2_u:
  half 0.391 → 0.315, half down 0.314 → 0.344; quarter 0.391 → 0.332, quarter down 0.336 → 0.383.
- **H1:** `move1_prior_entropy` fell through both hot stages (half 0.38 → 0.23, quarter 0.38 → 0.14; in p2_u it
  rose in both) and held through the step downs (half down 0.13 → 0.18, quarter down 0.08 → 0.10). The search
  target at move 1 stayed below 0.09 in all four runs. **Book move-1 entropy (H6): line A 1.597, 1.612, 1.607;
  line B 1.237, 1.239, 1.234** (start 1.227). Line A's is 0.37 above line B's and the start's at every measure:
  twice the dose spread the network's first book move, and the spread was there by the end of the 2e-5 stage.
  Direction-free; no row turns on it.
- Promotion loss by halves: half 0.366 / 0.331, half down 0.347 / 0.375; quarter 0.377 / 0.343, quarter down
  0.378 / 0.412 (rose through both step downs again). Decisive rate half 0.792 / 0.784, half down 0.791 / 0.775;
  quarter 0.794 / 0.791, quarter down 0.783 / 0.775. Self-play games 106.0, 105.7 (A) and 104.9, 106.0 (B) plies.
- **H2:** 0 skipped steps in 8,000 + 4,000 per line.
- **H10, pace:** half 10.76 min per iteration (self-play 9.83), half down 11.26 (9.77); quarter 11.02 (10.01),
  quarter down 13.01 (11.97). Sims/s 4,205–4,643 (half, 36 vCPU), 4,225–4,613 (half down, 32), 4,116–4,492
  (quarter, 32), 3,434–3,998 (quarter down, 64). Training 56, 89, 60 and 62 s per iteration. The 64-vCPU pod was
  again the slowest at self-play (as p2_u's was); the pods are different hosts, so nothing is attributed.

Rungs: random, greedy and phase1 won 200/200 at every checkpoint in all four runs.

## 6. Decision

Per the table's fifth row (K1 entirely below zero): **twice the dose costs skill. The dose stays at 0.25. Diagnosed
before p2_w: H12 (memorisation) and the self-play ring's share (461 rows in place of 717) are the first suspects.**
K2, N and ECE hold in all four runs, so no guard row applies.

Readings the table does not cover:

1. **H12 does not look like the cause.** The training-ring gap is −0.021 in line A and −0.015 in line B, and line
   A's advantage on held-out positions (0.097 nats) is sixteen times the 0.006 by which its gap is wider. Line A
   learned the teacher's policy in a form that transfers to unseen positions. Of the two registered suspects that
   leaves the self-play ring's share: 461 rows of the search's own targets per batch in place of 717. The round does
   not separate the two (PREREG, "What the design does not separate"), and H12 has no bound, so this is a reading.
   One run per line.
2. **Closer to the teacher, and weaker.** In p2_t the teacher line ended 0.435 nats closer to the teacher and
   +0.050 stronger than its control. Here line A ended 0.097 nats closer than line B and 0.044 weaker. Agreement
   with this teacher's policy is not a proxy for skill 5 from this start; T1 said so in its registration, and this
   is the first round where a T1 margin of this size and K1 point in opposite directions.
3. **The difference formed in the step down** (unregistered). At the end of 2e-5, line A was level with or ahead of
   line B (+0.030 [−0.016, +0.074]) and had dipped less against the start (−0.023 against −0.053). The step down
   then returned line B's dip (+0.070 [+0.031, +0.108]; p2_u's two lines: +0.056 and +0.090) and returned
   nothing to line A (−0.005 [−0.049, +0.040]). At 0.5 the 2e-6 stage did not do what it did in all three lines
   at 0.25. Whether line A never dipped or never recovered cannot be told from three reps at 0020.
4. **How much of K1 is line B.** Line B's endpoint, 0.378, is the highest six-rep skill-5 mean of p2_t–p2_v (0.360, 0.361), but it does
   not separate from the start (+0.017 [−0.017, +0.051]), and the same two configs from the previous start ended
   at 0.356 in p2_u. Line A's endpoint does not separate from the start either (−0.027 [−0.057, +0.003]). K1 is
   registered between the lines and is read as registered; against the start, the round shows a line that fell
   by about 0.03 and a line that rose by about 0.02, each inside its own interval. Best six-rep skill-5 mean: 0.378.
5. **Line A's first book move spread out** (H6: 1.60 against 1.23 in line B and at the start), while its prior at
   the initial position in self-play did not (`move1_prior_entropy` 0.18 against 0.10). The cost did not show at
   skill 0 or skill 3 (the endpoints are equal at both), only at skill 5.
6. **Twice the dose did not cost mate-in-2 or calibration:** 192 against 193, ECE within 0.002 at the endpoints.
   The registration named mate-in-2 as the place the cost would grow; it grew at skill 5 instead.

**Before p2_w (the row's diagnosis; nothing is registered here).** The cheapest separation of the two suspects is
the one the registration named: hold the self-play rows at 717 per batch and raise the teacher's rows on top (a
larger batch), against 0.25 as it stands. It changes the step size, so it is a registered round with its own
contract, not a reading of this one. Which lever p2_w takes is Michael's choice.

The open list, carried forward: the 2e-7 step-down; a deeper engine; the calibration sample; the promotion loss at
2e-6 (it rose through both step downs again: 0.378 → 0.412 in line B); L2's exclusion at storage time; the host
crashes (none this round; this box stayed up throughout); a public benchmark (a UCI wrapper first); the third line
that separates old positions from the teacher's targets (p2_t §6, reading 4). **Disk: 98 GB free at close** (this
round's four run directories hold 6.5 GB).

## 7. Mechanics

- **Unattended throughout.** `launch_run.sh` launched, staged, verified and started all four pods and enabled their
  drivers; each step down launched itself after its hot stage's final pull and the refusals above. No hand
  intervention, no measurement retried, no `FAILED` line in the finisher's log.
- **Lifecycle:** final pulls with the ring at 22:12:47 (half), 22:41:54 (quarter), 01:31:38 (quarter down) and
  01:41:41 (half down); each pod deleted itself after `PULLED` (0 pods of ours at close, read 2026-11-11 10:30
  UTC). Measurements under `gpu hold`: half 0020 22:12–22:53; quarter 0020 22:53–23:35; quarter down 0005
  01:31–01:57, 0010 –02:37; half down 0005 02:37–03:04, 0010 –03:45. The finisher from 03:48:19: T1 at the four
  checkpoints on the held-out ring, H12 at both endpoints on the training ring (4.5 min each), skill-5 reps
  half/0020, quarter/0020, half down/0005, quarter down/0005, half down/0010 (six), quarter down/0010 (six),
  skill-3 reps at both endpoints, M4, L2, and K1 at 08:08:18 UTC on 2026-11-11.
- **Cost:** pod time at $0.89/h ≈ **$10.76** (half 3.75 h $3.34; quarter 3.89 h $3.46; quarter down 2.40 h $2.14;
  half down 2.04 h $1.82). The balance went from $44.54 to $33.68 with no reload: **$10.86**. Expected about $11,
  worst case $24.92. All RunPod use by this project to date about $159.

## 8. Artefacts

| file | what |
|---|---|
| `runs/p2_v_{half,half_down,quarter,quarter_down}/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the runs |
| `runs/p2_v_*/teacher_agreement_iter*.json` | T1 (the held-out ring) |
| `runs/p2_v_{half,quarter}_down/teacher_agreement_train_iter0010.json`, `runs/p2_u_stale_down/teacher_agreement_train_iter0010.json` | H12 (the training ring; the start's was read for the draft) |
| `runs/p2_v_*/ladder.jsonl` | ladder passes at half and quarter 20, both step downs at 5 and 10 (ECE, N) |
| `runs/p2_v_*/sf5/` | skill-5 reps and each checkpoint's verdict against the start (K3) |
| `runs/p2_v_half_down/sf5/p2v_verdict_half_vs_quarter.json` | K1 |
| `runs/p2_v_half/sf5/p2v_verdict_unreg_half_vs_quarter_iter0020.json`, `runs/p2_v_half_down/sf5/p2v_verdict_unreg_half_vs_quarter_iter0005.json`, `runs/p2_v_{half,quarter}_down/sf5/p2v_verdict_unreg_step.json` | computed at close, not registered |
| `runs/p2_v_{half,quarter}_down/sf3/` | skill-3 reps at the endpoints and their verdicts |
| `runs/p2_v_*/reps_iter*_r*.jsonl`, `p2_verdict_reps*.json`, `p2_m4_iter*.json`, `startpos/` | K4, skill 0 |
| `runs/p2_v_*/book_rows.json` | L2 |
| `runs/p2_v_*_launch.console.log`, `runs/p2_v_*_campaign.console.log`, `runs/p2_v_finish.console.log`, `runs/p2_v_finish.sh`, `runs/p2_v_launch.sh` | consoles, launcher and finisher (untracked) |
