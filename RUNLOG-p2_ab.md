# RUNLOG — p2_ab (twice the teacher data, at the replicated protocol's passes per position)

**Run:** `runs/p2_ab`. A second pool of **4,000,000** positions not in D1, the held-out ring or the DIAG's pool
(`scripts/pool_positions.py`, seed 0: the 14 local rings' remainder, `p2_s_cont`, and the archived `p2_s_sims`,
`p2_s_down`, `p2_r_sims`), labelled by Stockfish 16 at 100,000 nodes on one secure RTX 4090 pod (`--labels-only`, 54
workers), the ring built here with D1's outcome model. Line **E8**: `scripts/distill_curve.py` on D1 + the DIAG's
ring + pool 2 (8,000,000 rows), teacher rows only, from `p2_w_quarter_down/0010` (sha256 `43494548…3857`), lr 2e-5
constant, batch 1,024, held-out measured every 250,000 samples, patience 4, cap 32,000,000, seed 2; the checkpoint
best by held-out cross-entropy. Then twelve skill-5 reps of E8, its mate-in-2 suite row, four verdicts. 2026-11-15
21:57 to 2026-11-16 11:11 UTC, the pod on RunPod and everything else on this box under `gpu hold`, driven by
`runs/p2_ab_driver.sh` as the transient user units `taz-p2ab` and, after a false stop, `taz-p2ab-resume` (§7), code
`5b06886`. **Pre-registration:** `PREREG-p2_ab.md` (`5b06886`). Every verdict below was registered before it was
measured, except where marked.

**Verdict: the registered first row, "more data helps at this scale". K1: E8 beats the 4M protocol at skill 5,
+0.0232 [+0.0011, +0.0451] (E8 0.4446 against D4's and R's 24 reps, 0.4214). The margin is thin: the interval's
lower end is +0.0011. K2 passes: E8 solves 192 of 200 mate-in-2 positions (floor 183; the start 188). L holds: pool 2
is 4,000,000 rows, every one labelled, the ring built here with D1's outcome model. C, reported: E8's best held-out
cross-entropy is 2.2028 (D4 2.2048, R 2.2051), and E8 stopped on patience at 19,750,912 samples, about 2.5 passes,
not at the 32M cap. Reported: E8 − start +0.0635 [+0.0392, +0.0877], the largest gain over the start yet. The 8M
protocol is standing; p2_ac is data scale again, its size being settled with Michael (§6). A false "pod gone" at
00:58 UTC stopped the first driver for five hours; the pod had kept labelling, and every label was pulled before its
deadline (§7).**

---

## 1. Before launch

The draft is `218bdf2`, the code it needs `e6c603c` (`distill_curve.py` takes several `--new` rings, read whole and
in order, `distill.prefix_plan`; the full suite passed 1,280), and the registration `5b06886` (2026-11-15, Michael:
"Ok given, go"). The round is p2_aa's first row's single change, chosen by Michael: more data ("I'm always about the
data first"). Pool 2's sources and counts were fixed at draft by a read-only pass of `select_rows` (4,214,593
distinct positions available in the registered rings). At registration the round's records were negated in
`.gitignore` and red-checked through `MUST_REACH`, and the ring-view directories (`runs/_ringview/`) and pod staging
copies were pinned ignored: the global `meta.json` negation had reached their copies. `git diff e6c603c 5b06886 --
src scripts cloud` is empty, so the run's code is the code the suite passed.

## 2. L — the data

**L1 holds.** `runs/p2_ab/pool/pool.json`: 4,000,000 rows drawn (seed 0) from 4,214,593 available, with D1, the
held-out ring and the DIAG's pool excluded. `p2_t_teach_down` contributed none: all 495,898 of its rows that do not repeat a
position of their game are positions of D1, the held-out ring or the excluded suites (D1 was labelled from it), as
in the DIAG. Re-derived for this log by rerunning the selection, which
reproduces the pool's boards at 2,000 sampled rows, pool 2's rows come **56.7% (2,267,578) from the 14 p2_t–p2_w
rings**, 11.1% (445,301) from `p2_s_cont`, 10.8% (431,244) from `p2_s_sims`, 10.8% (430,306) from `p2_s_down` and
10.6% (425,571) from `p2_r_sims`: 43.3% from the p2_r/p2_s networks, against the "about 43%" registered.

**L2 holds.** The pod's job labelled all 4,000,000 rows (exit 0); the ring built here
(`runs/p2_ab/teacher/teacher.json`) holds 4,000,000 rows, its outcome model D1's (a 303.35, s 367.36, `fitted_here`
false; draw rate 0.296 on its own rows), 100,000 nodes, multipv 5, temperature 50 cp, Stockfish 16. The rebuild
labelled nothing (`labelling_seconds` 33.3, the scan).

## 3. C — the teacher measure (reported)

E8 ran 80 measurements, from 0 to 19,750,912 samples. **Its best: held-out cross-entropy 2.2028 at 18,750,464
samples** (top-1 0.3844, value MSE against the teacher 0.0319, training-probe cross-entropy 2.1831); the four
measurements after it did not beat it, and **patience stopped the run at 19,750,912 samples**, about 2.5 passes over
its 8,000,000 rows (2.3 at the best), well short of the 32M cap. So it did **not** end at the cap still improving
(the DIAG's Addendum A test, best at 31,750,000 or later). The start measured 2.3673, top-1 0.3658, value MSE 0.0378:
the registered T1 reading, to the digit, as in every run of this script.

E8 beside D4 and R at matched samples (unregistered):

| samples | E8 held-out | D4 held-out | R held-out | E8 top-1 | E8 value MSE |
|---|---|---|---|---|---|
| 0 | 2.3673 | 2.3673 | 2.3673 | 0.3658 | 0.0378 |
| 1M | 2.2455 | 2.2447 | 2.2459 | 0.3686 | 0.0322 |
| 4M | 2.2308 | 2.2300 | 2.2300 | 0.3728 | 0.0308 |
| 8M | 2.2201 | 2.2198 | 2.2206 | 0.3758 | 0.0317 |
| 12M | 2.2145 | 2.2116 | 2.2123 | 0.3756 | 0.0315 |
| 16M | 2.2078 | 2.2051 | 2.2051 | 0.3826 | 0.0317 |
| 18.75M (E8's best) | 2.2028 | — | — | 0.3844 | 0.0319 |

At best: D4 2.2048 (top-1 0.3826, value MSE 0.0342), R 2.2051 (0.3796, 0.0333), E8 2.2028 (0.3844, **0.0319, the
lowest value MSE of any run**). E8's last eight measurements lie within 0.0019 of each other (2.2047 to 2.2028).

## 4. K — skill 5

The instrument as registered: `eval_reps.py run --anchor stockfish:5 --config runs/p2_w_quarter_down/config.yaml`,
the 100-opening book, 200 games a rep, 48 simulations a move; verdicts by per-opening rep-means with a paired
bootstrap over the 100 openings (10,000 resamples, seed 0). The 4M protocol's 24 reps (D4's twelve, R's twelve) are
those on file from the DIAG and p2_aa, not re-measured.

### K1 (primary) — E8 against the 4M protocol
**+0.0232 [+0.0011, +0.0451]**, se 0.0111 (`p2ab_verdict_e8_vs_4m.json`): E8 0.4446 over twelve reps against the 4M
protocol's 0.4214 over twenty-four. **Entirely above zero, by 0.0011 at the lower end.**

### K2 — mate-in-2
**192 of 200** at E8's checkpoint (`e8_ladder.jsonl`, the suite row at 128 sims under the start's config). **K2
passes** (floor 183). The start solves 188, R 189. E8 solves `mate2_054`, `mate2_138`, `mate2_140` and `mate2_141`,
which the start fails, and fails none the start solves; its eight failures are all the start's. Mate-in-3: 167
(the start 164, R 163). The ladder's console prints "REGRESSED" against the phase 1 baseline of 0.985; the registered
floor is 183.

### Reported, no decision
- **E8 − D4: +0.0183 [−0.0056, +0.0423]** (`p2ab_verdict_e8_vs_d4.json`, se 0.0123): includes zero.
- **E8 − R: +0.0281 [+0.0031, +0.0525]** (`p2ab_verdict_e8_vs_r.json`, se 0.0126): excludes zero.
- **E8 − start: +0.0635 [+0.0392, +0.0877]** (`p2ab_verdict_e8_vs_start.json`, se 0.0122; the start's twelve reps
  from p2_w and p2_aa): the largest gain over the start this project has measured.
- E8's twelve reps run 0.3925–0.4875; the 4M protocol's twenty-four 0.3500–0.4750.

## 5. The predictions

| prediction (written before launch) | result | |
|---|---|---|
| L holds | holds | held |
| Labelling 7–8 h | 7.12 h (25,617 s) | held |
| C: E8's held-out CE 2.19–2.21 | 2.2028 | held |
| C: more likely than not ends at the cap still improving | stopped on patience at 19.75M | **missed** |
| K1 point +0.000 to +0.030 | +0.0232 | held |
| K1's interval includes zero, more likely than not | [+0.0011, +0.0451] | **missed**, by 0.0011 |
| K2 passes (E8 185–193) | 192 | held |
| E8 − start +0.03 to +0.07, excluding zero | +0.0635 [+0.0392, +0.0877] | held |

## 6. Decision

Per the table's first row (L holds; K1 entirely above zero; K2 passes): **more data helps at this scale. The 8M
protocol is standing. p2_ac changes one thing, data first: the data's size again. The row named 16M; after the
verdict Michael asked whether to go further than 16M, and the size is being settled with him before the draft.**

Readings the table does not cover (all unregistered):

1. **A thin margin.** K1's lower end is +0.0011. E8 − D4 alone includes zero; E8 − R excludes it. The result
   satisfies the registered row, and the next round should firm it up rather than lean on it.
2. **Skill rose while the teacher measure barely moved.** E8's held-out cross-entropy (2.2028) is 0.0020 under D4's
   and its last eight measurements are flat within 0.0019, while E8 played +0.0232 better than the 4M protocol and its
   value MSE (0.0319) is the lowest yet. At matched samples E8 tracks D4 and R within about 0.003, slightly behind at
   12–16M (larger sets trail at equal samples, as the DIAG found). One hypothesis: the extra positions sharpen the
   value head and the play more than the policy's agreement with the teacher on this held-out ring, which is drawn
   from p2_t's positions and may not see what 43% older-network positions add. A hypothesis, not a result.
3. **Twice the data at about the same compute.** The registration held passes per position at about four by
   allowing 32M samples; patience ended E8 at 19.75M, 1.23 times D4's 16M. So the comparison as run is closer to
   "twice the data, about a quarter more steps" than to "the same passes": the gain came with little extra compute.
4. **On the rung's own scale** (unregistered): the start's 0.3810 and E8's 0.4446 against `stockfish:5` are about
   −84 and −39 Elo by the logistic relation, about +46. This is not a CCRL measurement; the gauntlets of p2_x and
   p2_y measured the start, not E8.

The open list, carried forward: data scale (p2_ac, being sized); the 2e-6 step-down after the protocol; self-play
rows mixed back in; **p2_z, the benchmark track's clock-scaled gauntlet**, and a gauntlet of the standing network
(the start's CCRL readings no longer describe it); the runner logging an engine's exception (RUNLOG-p2_y §3); and
from p2_w: a deeper engine, the 2e-7 step-down, per-source loss weights, the calibration sample, the promotion loss at
2e-6, L2's exclusion at storage time.

## 7. Mechanics

- **Launch.** `taz-p2ab` armed at 2026-11-15 21:57:29 UTC at `5b06886`. Pool 2 was written at 22:01:34. The launch
  plan read worst case $8.90 at the 10 h cap; the pod `yhniczfdam99hk` (secure RTX 4090, 64 vCPU, 62 GB; cgroup
  quota 54.4 CPUs) launched at 22:01:38 and checked out: code `5b06886`, Stockfish 16 (`16-1build1`), 4,000,000 pool
  rows on the pod, `boards.dat`'s sha256 equal to this box's. The job started at 22:06:41.
- **The false stop.** At 00:58:59 the driver's `alive()` check made one `runpod_pod.py status` call, got no line for
  the pod, concluded it was gone, logged "a human must decide (rerun the job for the rest)" and exited, with 1,738,160
  rows pulled. **The pod had not gone.** It kept labelling; the job finished at about 05:13:38 (all 4,000,000 rows in
  25,617 s, about 156 a second, exit 0) and then waited for a pull. Michael asked how the round was going at 06:10;
  the pod was found running with 111 minutes to its deadline. `alive()` in `runs/p2_ab_driver.sh` was changed to try
  three times, 20 s apart, before reading a pod as gone (between the two units; no unit was running), and the driver
  was resumed as `taz-p2ab-resume` at 06:11:26: phases 0 and A were skipped (pool and pod recorded), it pulled all
  4,000,000 rows at 06:11:43, wrote `PULLED` at 06:11:45, and the pod deleted itself. **No label was lost and none was
  relabelled.** The stall cost 5 h 12 min of wall time and about 58 minutes of idle pod time.
- **Then, on this box.** The ring built 06:11:45–06:14:06; E8 trained 06:14:06–09:24:38 (3 h 10 min); twelve reps to
  11:10:09 (1 h 46 min); the suite row and the four verdicts by 11:11:52 ("p2_ab complete"). Each step took `gpu hold
  --by tinyalphazero` (ComfyUI not running, nothing stopped, 101.8 GB free) with
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTHONUNBUFFERED=1`. The verdict steps read `eval_reps.py verdict`'s exit 1
  as an interval that includes zero (RUNLOG-p2_aa §7); E8 − D4's did, and the run carried on. 13 h 14 min from arming
  to completion, the false stop included. No commit went to `main` while either unit ran.
- **Scoring.** At the 13:45 check-in (rescheduled from 13:15 after the stall). Michael was notified, saw the verdict
  and agreed to closing (2026-11-16).
- **Cost:** the pod lived about 8.2 h (22:01:38 to about 06:12) at $0.89/h, **about $7.30**, including the 58 idle
  minutes the false stop caused; under the $8.90 worst case. The balance readings cannot isolate it: the PREREG wrote
  "about $39.6", but the balance at launch was **$21.08** (other projects had spent in between); Michael topped the
  account up overnight; it read $56.21 at 06:10 and $42.89 at 13:47 UTC on 2026-11-16, with another project's pod
  billing.
- **The box** stayed up throughout.

## 8. Artefacts

| file | what |
|---|---|
| `runs/p2_ab/pool/pool.json`, `pool/meta.json` | pool 2's sources, counts and selection (L1) |
| `runs/p2_ab/teacher/teacher.json`, `teacher/meta.json` | the ring's settings, outcome model and provenance (L2) |
| `runs/p2_ab/e8/size_8000000/result.json`, `log.jsonl` | E8's best measurement and all 80 (C; §3's table) |
| `runs/p2_ab/e8_sf5/reps_best_r{1..12}.jsonl` | E8's twelve skill-5 reps (K1) |
| `runs/p2_ab/e8_ladder.jsonl` | E8's suite row (K2) and the 20-game random match that carried it |
| `runs/p2_ab/p2ab_verdict_{e8_vs_4m,e8_vs_d4,e8_vs_r,e8_vs_start}.json` | K1 and the reported comparisons |
| `runs/p2_ab/pool/*.dat`, `teacher/*.dat`, `teacher_labels/`, `e8/size_8000000/best.pt`, `ce_rows.npy` | the pool, the ring, the label shards, E8's checkpoint and per-row CE (ignored; regenerable or large) |
| `runs/p2_ab_driver.sh`, `runs/p2_ab_driver.console.log`, `runs/p2_ab_pod_job.log` | the units' script, console and the pod job's log (untracked) |
