# RUNLOG — p2_u (the standing teacher on a ring relabelled from this round's start, against p2_t's ring as it stands; judged on skill 5)

**Runs:** line A, "fresh": `runs/p2_u_fresh` (the labeller on `runs/p2_t_teach_down/replay`, then
`--config configs/p2_u_fresh_hot.yaml --seed 43`: 48 simulations, lr 2e-5, `teacher_frac` 0.25 from
`runs/p2_u_fresh/teacher`, 20 iterations from `p2_t_teach_down/0010`), then `runs/p2_u_fresh_down`
(`configs/p2_u_fresh_down.yaml`: 48, 2e-6, 10 iterations from `p2_u_fresh/0020`). Line B, "stale": `runs/p2_u_stale`
(`configs/p2_t_teach_hot.yaml`: 48, 2e-5, teacher 0.25 from `runs/p2_t_teach/teacher`, 20 iterations from the same
start), then `runs/p2_u_stale_down` (`configs/p2_t_teach_down.yaml`: 48, 2e-6, 10 iterations from `p2_u_stale/0020`).
Four RunPod pods, each a secure RTX 4090 (24,564 MiB): fresh on 64 vCPU (`xb2qrb9al7uztd`), stale on 24 vCPU
(`wzmloq1t7plclc`), both step downs on 32 vCPU (`5i07jh94f7dr6a` stale down, `5mnd6r2zjgbu87` fresh down).
Labelling 2026-11-09 10:39 to 11:16, then training 11:16 to 15:20 (fresh); 10:37 to 14:12 (stale); 18:05 to 19:54
(stale down); 21:04 to 22:52 (fresh down) UTC, code `8ae801b`. Every measurement ran on this box (torch
2.14.1+rocm7.2) under the user units `taz-campaign@p2_u_{fresh,fresh_down,stale,stale_down}`
(`MEASURE_AFTER_PULL=1`) and `taz-finisher@p2_u` (`runs/p2_u_finish.sh`). **Pre-registration:** `PREREG-p2_u.md`
(`8ae801b`), no addenda. Every verdict below was registered before it was measured, except where marked.

**Verdict: the registered third row: relabelling gains nothing measurable at this dose. The fresh ring is not
standing; a later round may reuse a ring or relabel as its lever needs, and p2_v's lever is Michael's choice. K1
(fresh down/0010 − stale down/0010, skill 5, six reps per side): −0.005 [−0.038, +0.028]. K2 passes (190 and 191).
T1 holds by 0.011 nats (cross-entropy on line A's held-out ring 2.380 against 2.391), about one fortieth of p2_t's
margin. ECE and N hold; L1 holds in all four runs; L3 holds. Neither endpoint differs from the start at skill 5
(fresh −0.004 [−0.034, +0.027], stale +0.001 [−0.027, +0.030]): 30 more iterations of the standing teacher, on either
ring, returned to where p2_t ended (§6).**

---

## 1. Before launch

The start reference is `p2_t_teach_down/0010`: six skill-5 book reps (`runs/p2_t_teach_down/sf5/`, mean 0.3600),
three skill-3 reps (0.7925), three skill-0 book reps (0.9692) and three initial-position reps (0.9767); nothing was
re-measured. Read at registration, 2026-11-09 10:12 UTC: balance $55.91, no pods of ours, secure RTX 4090 at
$0.89/h; `make test` 1,094 passed, 4 skipped on the draft's tree (`3948b03`).

**The pods.** All four came from `cloud/launch_run.sh` (`runs/p2_u_launch.sh`), each a transient unit armed at
10:13. Both hot stages were refused three times (10:13 to 10:23, no instances) and launched at 10:28:35 (fresh) and
10:29:04 (stale). Each step down waited for its hot stage's `.remote_done` and was then refused for hours: stale
down 44 times (14:15:59 to 17:52:34), launched at 17:57:37; fresh down 66 times (15:22:59 to 20:50:14), launched at
20:55:17. At 19:15 Michael chose to keep waiting for a 4090 (no other GPU class, no addendum). No pod was unusable,
none replaced; fresh down's first SSH probe found no public endpoint yet and the next one did. Every input's sha256
matched on each pod (50 files on fresh; 65 on the other three, which also took a teacher ring). Nothing was committed
to `main` between registration and the second step down's launch, so all four runs carry one `git_sha`.

## 2. L — the continuity applied

- **L1 holds, all four runs:** `seed` 43; `git_sha` `8ae801b` (provenance and every iteration row), `git_dirty`
  false; each `config_fingerprint` equal to its `config.yaml`'s sha256: `066043ca…d67d` (fresh), `036cd665…01da`
  (fresh down), `62fd3945…1573` (stale), `80de3169…5120` (stale down), the registered four.
  - fresh and stale: `init_from` `runs/p2_t_teach_down/ckpt_iter0010.pt`; pool and ring from
    `runs/p2_t_teach_down`; `seed_replay` 500,000 rows, `donor_committed` = `donor_head` = 1,039,731, offset −10;
    `lr` 2e-5 on every row.
  - fresh down: `init_from` `runs/p2_u_fresh/ckpt_iter0020.pt`; `seed_replay` 500,000, committed = head =
    1,595,615 (fresh's final ring), offset −20; `lr` 2e-6 on every row.
  - stale down: `init_from` `runs/p2_u_stale/ckpt_iter0020.pt`; `seed_replay` 500,000, committed = head =
    1,571,722, offset −20; `lr` 2e-6 on every row.
  - Simulations per stored position **48.0** at every iteration of all four. `teacher_dir` `runs/p2_u_fresh/teacher`
    in both line-A configs and `runs/p2_t_teach/teacher` in both line-B configs, `teacher_frac` 0.25 in all four.
- **L2 (reported, no bound):** fresh 3 (`open_021` at 20, `open_001` at 11, `open_028` at 17), fresh down 1
  (`open_012` at 5), stale 4 (`open_021` at 20, `open_001` at 11, `open_009` at 13, `open_028` at 17), stale down
  2 (`open_010` at 4, `open_012` at 5) in 500,000. Predicted 0–5 per run: held.
- **L3 holds** (`runs/p2_u_fresh/teacher/teacher.json`): 475,696 rows; Stockfish 16 (`/usr/games/stockfish`,
  `16-1build1`); 100,000 nodes, `multipv` 5, T 50 cp, 64 MB hash; source ring `runs/p2_t_teach_down/replay` at
  1,039,731 written. Selection: 500,000 ring rows read, 513 excluded (the five suites), 4,102 repeated, 14,689
  duplicates, 480,696 selected, as registered. The held-out ring (`teacher_holdout/`) has 5,000 rows (seed 0) on the
  training fit (`fitted_here` false, the same `a` and `s`).
  - **The outcome model: `a` 303.3 cp, `s` 367.4 cp**, fitted on the 475,696. At a level score it calls **39.1%
    drawn** (1 − 2σ(−a/s); `teacher.json`'s `draw_rate`, 0.295, is the share of drawn game outcomes among the rows).
    Inside 5–60%: no defect. Under it a 45 cp edge is W 0.331 / D 0.390 / L 0.279 (p2_t's fit: 0.360 / 0.343 /
    0.297). Mean |`root_q`| 0.357 (p2_t's ring 0.401).
  - Predicted 20–40% drawn: held (39.1%). Predicted `s` 250–400 cp: held (367.4).
  - **Overlaps (reported at close, no bound; `position_key`, the selection's own key):** 31 of line A's 5,000
    held-out positions are in line B's training ring; 34 of p2_t's 5,000 held-out positions are in line A's. The two
    training rings share **2,937 positions** (0.6% of either; computed at close, not registered).

## 3. T1 — the lever applied: agreement with the teacher on line A's held-out 5,000

`scripts/teacher_agreement.py` (prior only, no search), `runs/p2_u_*/teacher_agreement_*.json`:

| checkpoint | A's held-out: top-1 | A's held-out: cross-entropy | A's held-out: value MSE | p2_t's held-out: top-1 | p2_t's held-out: cross-entropy |
|---|---|---|---|---|---|
| start, `p2_t_teach_down/0010` | 0.3550 | 2.411 | 0.0386 | 0.3588 | 2.370 |
| fresh 0020 | 0.3572 | 2.382 | 0.0463 | 0.3690 | 2.353 |
| stale 0020 | 0.3592 | 2.391 | 0.0504 | 0.3672 | 2.346 |
| **fresh down 0010 (A's endpoint)** | **0.3560** | **2.380** | 0.0371 | 0.3676 | 2.353 |
| **stale down 0010 (B's endpoint)** | **0.3590** | **2.391** | 0.0395 | 0.3678 | 2.346 |

(The start's row on p2_t's held-out ring is RUNLOG-p2_t §3's.)

- **T1 holds:** line A's endpoint has the lower cross-entropy on line A's held-out ring, by **0.011 nats** (p2_t's
  T1 margin was 0.435). Top-1 agreement is line B's by 0.3 points (reported, not judged).
- On p2_t's held-out ring (informational) the order reverses by about the same amount: line B 2.346, line A 2.353.
  Each line ends marginally closer to the teacher on positions like its own ring's, by 0.007–0.011 nats.
- From the start, line A's cross-entropy on its own held-out ring fell 0.031 nats and line B's 0.020; in p2_t one
  round took 0.52 off. The value MSE against the teacher's `root_q` ended at or below the start's in both lines
  (0.0371 and 0.0395 against 0.0386), after rising at 2e-5 in both, as in p2_t.
- Predicted T1 holds: held. Predicted top-1 agreement within 1.5 points between the lines: held (0.3 points).

## 4. K — the two lines at the end, skill 5

### K1 (primary) — fresh down/0010 against stale down/0010, six reps per side

Per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples,
seed 0; `runs/p2_u_fresh_down/sf5/p2u_verdict_fresh_vs_stale.json`, reproduced byte for byte; all twelve rows
`stockfish_s5`, each side on its own checkpoint):

| line | reps r1–r6 | mean | W / D / L (1,200 games) |
|---|---|---|---|
| B, `p2_u_stale_down/0010` | 0.365, 0.395, 0.3075, 0.3275, 0.395, 0.375 | 0.3608 | 294 / 278 / 628 |
| A, `p2_u_fresh_down/0010` | 0.3025, 0.3775, 0.39, 0.3625, 0.3625, 0.3425 | 0.3563 | 289 / 277 / 634 |
| **A − B** | | **−0.005** | **[−0.038, +0.028]**, SE 0.0166 |

- **The interval includes zero.** Forecast: the point 0.00 to +0.04, missed (−0.005); more likely than not the CI
  includes zero, held. Per opening, A's rep-mean is above B's on 45 of the 100, below on 46, equal on 9 (median 0).
  The draw rates are the same (0.231 and 0.232). The half-width, 0.033, is p2_t's (0.035) as registered.

### K2 — mate-in-2

At the endpoints: **fresh down/0010 190/200, stale down/0010 191/200**; 190 ≥ 191 − 4 and both ≥ 183. **K2 passes.**
At the other measures: fresh 0020 189, fresh down/0005 188; stale 0020 190, stale down/0005 190; the start 189.
Eight positions failed at all six measures of both lines: `mate2_091`, `mate2_124`–`129` and `mate2_196` (the
start fails all eight; `mate2_127` and `mate2_128` were not among p2_t's all-measure seven). One failed at every
line-A measure and at no line-B measure: `mate2_141`, which the start solves. `mate2_054`, line A's in p2_t, failed
at every line-B measure and at both line-A step-down measures. `mate2_138`, the other line-A-only failure in p2_t,
was solved at all six.

### K3 — each line against the start, and skill 3 (informational)

| checkpoint | reps | vs start [95% CI] |
|---|---|---|
| fresh 0020, skill 5 (end of 2e-5) | 0.28, 0.33, 0.29 | **−0.060 [−0.099, −0.022]** |
| stale 0020, skill 5 (end of 2e-5) | 0.32, 0.2375, 0.255 | **−0.089 [−0.125, −0.053]** |
| fresh down 0005, skill 5 | 0.3325, 0.3525, 0.3325 | −0.021 [−0.058, +0.017] |
| stale down 0005, skill 5 | 0.3775, 0.33, 0.305 | −0.023 [−0.058, +0.015] |
| **fresh down 0010, skill 5 (six)** | mean 0.3563 | **−0.004 [−0.034, +0.027]** |
| **stale down 0010, skill 5 (six)** | mean 0.3608 | **+0.001 [−0.027, +0.030]** |
| fresh down 0010, skill 3 | 0.7925, 0.855, 0.8175 | +0.029 [−0.011, +0.068] |
| stale down 0010, skill 3 | 0.84, 0.8175, 0.8425 | **+0.041 [+0.002, +0.080]** |

Predicted: both 0020s near or below the start (held: both below); line A at or above the start at its endpoint (at
it: −0.004, within noise; not above); line B within noise of it (held). Line B's endpoint is above the start at
skill 3 (informational; no row turns on it).

**Computed at close, not registered** (same method; `p2u_verdict_unreg_*.json`): A − B at 0020 +0.029 [−0.017,
+0.074]; at down/0005 +0.002 [−0.045, +0.047]. Each line's step, down/0010 (six) − 0020 (three): A **+0.056
[+0.018, +0.095]**, B **+0.090 [+0.050, +0.128]**.

### K4 — skill 0 and the initial position (continuity)

Skill-0 book reps against the start (0.9692): fresh 0020 +0.008, fresh down/0005 +0.001, **fresh down/0010 +0.014
[+0.001, +0.028]** (above zero; informational, near the ceiling); stale 0020 −0.004, stale down/0005 −0.002, stale
down/0010 +0.006 [−0.011, +0.022]. Initial position against the start (0.9767): fresh 0020 −0.004, fresh down/0010
−0.002; stale 0020 −0.009, stale down/0010 +0.011 [−0.001, +0.023]. M4 at skill 0: fresh down 0.983, stale down
0.975, met (bar 0.675).

## 5. N, ECE, H

- **N — holds:** mate-in-2 fresh 189; fresh down 188, 190; stale 190; stale down 190, 191. Mate-in-3 fresh 159;
  fresh down 159, 160; stale 156; stale down 156, 156 (start 158). Predicted 185–195: held.
- **ECE — holds:** fresh 0.0963; fresh down 0.0999, 0.0969; stale 0.0980; stale down 0.0969, 0.1049 (< 0.15;
  start 0.1010; predicted 0.08–0.14: held). p2_t's line-A excess over line B (about 0.01) did not recur here: both
  lines carry the teacher, and they sit within 0.008 of each other.
- **H7 — the losses of line A against line B.** Both lines mix ring and teacher rows, so here they are comparable.
  Policy loss: fresh 1.758 → 1.761 (first five 1.749, last five 1.757), fresh down 1.755 → 1.754; stale 1.747 →
  1.757 (1.739, 1.755), stale down 1.755 → 1.745. p2_t's line A: 1.700 → 1.755, 1.758 → 1.753. The two lines'
  policy losses are within 0.01 throughout.
- **The value loss** fell at 2e-5 and rose back through the step down in both lines, as in p2_s and p2_t: fresh
  0.395 → 0.335, fresh down 0.338 → 0.384; stale 0.384 → 0.328, stale down 0.332 → 0.382.
- **H1:** `move1_prior_entropy` rose through both hot stages (fresh 0.27 → 0.94, stale 0.27 → 0.91; p2_t's
  line A held near 0.2 through its hot stage and ended at 0.26). Through the step downs, line A's held (0.85 → 0.83) and
  line B's fell back (0.91 → 0.38). The search target at move 1 (`move1_policy_entropy`) did the same: fresh down
  0.88 → 0.47, stale down 0.19 → 0.06. **Book move-1 entropy (H6): line A 1.250, 1.262, 1.274; line B 1.262,
  1.248, 1.227** (start 1.253). Direction-free; no row turns on it.
- Promotion loss by halves: fresh 0.382 / 0.351, fresh down 0.381 / 0.419; stale 0.352 / 0.339, stale down
  0.395 / 0.432 (rose through both step downs again). Decisive rate fresh 0.762 / 0.776, fresh down 0.784 / 0.784;
  stale 0.783 / 0.785, stale down 0.783 / 0.783. Self-play games 107.0, 106.8 (A) and 104.7, 105.0 (B) plies.
- **H2:** 0 skipped steps in 8,000 + 4,000 per line.
- **H10, pace:** fresh 12.18 min per iteration (self-play 11.27), fresh down 10.80 (9.72); stale 10.75 (9.88),
  stale down 10.91 (9.95). Sims/s 3,728–4,018 (fresh, 64 vCPU), 4,406–4,693 (fresh down, 32), 4,025–4,471 (stale,
  24), 4,147–4,453 (stale down, 32). Training 55, 65, 52 and 58 s per iteration. The 64-vCPU pod was the slowest at
  self-play; the pods are different hosts, so nothing is attributed.
- **H11, labelling:** 2,242 s (37.4 min) for 480,696 positions with 54 workers on the 64-vCPU pod (CPU quota
  54.4): about **0.25 s per position per worker**, half p2_t's 0.53 at the same settings and worker count
  (4,668 s). The registration's estimate (about 1.3 h on 64 vCPU) was built on p2_t's rate. The engine searches a
  fixed 100,000 nodes, so the difference is the host's speed; not attributed further.

Rungs: random, greedy and phase1 won 200/200 at every checkpoint in all four runs.

## 6. Decision

Per the table's third row (K1 includes zero): **relabelling gains nothing measurable at this dose. The fresh ring is
not standing; a later round may reuse a ring or relabel as its lever needs. T1 holds, so the table's reading is that
the fresh ring moved the network on its own positions and it did not show at skill 5. p2_v's lever is Michael's
choice: a larger dose on a fresh ring (0.5), a deeper engine, the 2e-7 step-down, the public benchmark.** K2, N and
ECE hold in all four runs.

Readings the table does not cover:

1. **T1 holds by a margin that does not separate the rings.** The registered rule is met, 2.380 against 2.391, but
   on p2_t's held-out ring the order reverses by 0.007, and both margins are about one fortieth of p2_t's 0.435.
   The two training rings share 0.6% of their positions (2,937), yet on either held-out ring neither line ends more
   than 0.011 nats closer to the teacher than the other. Read with K1, the plainer description is that the two rings taught the
   network the same thing; the 0.011 nats rest on one run per line and no interval was registered for them.
2. **Another round of the standing teacher returned to where p2_t ended.** Both endpoints are within 0.005 of the
   start at skill 5 (0.356 and 0.361 against 0.360), and both 0020s were below it (−0.060 and −0.089, each interval
   entirely below zero). The 2e-5 stage cost each line 0.06–0.09 and the step down returned it (+0.056 and +0.090;
   unregistered). p2_t's teacher line gained +0.030 over its start in one round; this round, from that endpoint,
   added nothing on either ring. The cross-entropy to the teacher moved 0.02–0.03 nats where p2_t's round moved
   0.52. At 0.25 and this schedule, the teacher's measurable effect looks used up: a reading, not a registered claim.
3. **Line B dipped further and came back further** (−0.089 against −0.060 at 0020; A − B at 0020 +0.029
   [−0.017, +0.074], unregistered). The order did not survive the step down (A − B +0.002 at 0005, −0.005 at 0010).
4. **The rows the start had drawn 6.5 times were drawn about 13 times by line B's end** with no cost K1 can see.
   Reuse is the cheaper default: relabelling took 37 min of a 64-vCPU pod here (0.25 s per position per worker),
   about $0.55.
5. **p2_t's small mate-in-2 and calibration cost did not recur as a gap**, as expected with the teacher in both
   lines: line A 188–190, line B 190–191; ECE within 0.008. Against the start, mate-in-2 is 188–191 against 189.

The open list, carried forward: the 2e-7 step-down; the calibration sample; the promotion loss at 2e-6 (it rose
through both step downs again: 0.395 → 0.432 in line B); L2's exclusion at storage time; the host crashes (none this
round; this box stayed up throughout); a public benchmark (a UCI wrapper first); the third line that separates old
positions from the teacher's targets (p2_t §6, reading 4). **Disk: 106 GB free at close** (this round's four run
directories hold 7.0 GB, the fresh teacher ring included).

## 7. Mechanics

- **Unattended throughout.** `launch_run.sh` launched, staged, verified and started all four pods and enabled their
  drivers; each step down launched itself after its hot stage's final pull and the refusals above. No hand
  intervention, no measurement retried, no `FAILED` line in the finisher's log.
- **Lifecycle:** final pulls with the ring at 14:15:12 (stale), 15:22:31 (fresh), 19:59:15 (stale down) and 22:52:29
  (fresh down); each pod deleted itself after `PULLED` (0 pods of ours at close). Measurements under `gpu hold`:
  stale 0020 14:15–14:58; fresh 0020 15:22–16:05; stale down 0005 19:59–20:26, 0010 –21:09; fresh down 0005
  22:52–23:19, 0010 –00:00. The finisher from 00:03:12: T1 at the five checkpoints on both held-out rings, skill-5
  reps fresh/0020, stale/0020, fresh down/0005, stale down/0005, fresh down/0010 (six), stale down/0010 (six),
  skill-3 reps at both endpoints, M4, L2, and K1 at 04:15:17 UTC on 2026-11-10.
- **Cost:** pod time at $0.89/h ≈ **$11.26** (fresh 4.90 h $4.36, of which labelling about 0.6 h; stale 3.77 h
  $3.35; stale down 2.03 h $1.80; fresh down 1.95 h $1.74). The balance went from $55.91 to $44.54 with no
  reload: **$11.37**. Expected about $13, worst case $26.70. All RunPod use by this project to date about $148.

## 8. Artefacts

| file | what |
|---|---|
| `runs/p2_u_{fresh,fresh_down,stale,stale_down}/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the runs |
| `runs/p2_u_fresh/teacher/teacher.json`, `teacher/meta.json`, `teacher_holdout/teacher.json`, `teacher_holdout/meta.json` | L3; the rings themselves stay untracked |
| `runs/p2_u_*/teacher_agreement_*.json` | T1 (both held-out rings) |
| `runs/p2_u_*/ladder.jsonl` | ladder passes at fresh and stale 20, both step downs at 5 and 10 (ECE, N) |
| `runs/p2_u_*/sf5/` | skill-5 reps and each checkpoint's verdict against the start (K3) |
| `runs/p2_u_fresh_down/sf5/p2u_verdict_fresh_vs_stale.json` | K1 |
| `runs/p2_u_fresh/sf5/p2u_verdict_unreg_fresh_vs_stale_iter0020.json`, `runs/p2_u_fresh_down/sf5/p2u_verdict_unreg_fresh_vs_stale_iter0005.json`, `runs/p2_u_{fresh,stale}_down/sf5/p2u_verdict_unreg_step.json` | computed at close, not registered |
| `runs/p2_u_{fresh,stale}_down/sf3/` | skill-3 reps at the endpoints and their verdicts |
| `runs/p2_u_*/reps_iter*_r*.jsonl`, `p2_verdict_reps*.json`, `p2_m4_iter*.json`, `startpos/` | K4, skill 0 |
| `runs/p2_u_*/book_rows.json` | L2 |
| `runs/p2_u_*_launch.console.log`, `runs/p2_u_*_campaign.console.log`, `runs/p2_u_finish.console.log`, `runs/p2_u_finish.sh`, `runs/p2_u_launch.sh` | consoles, launcher and finisher (untracked) |
