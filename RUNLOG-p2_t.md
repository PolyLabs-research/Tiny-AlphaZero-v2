# RUNLOG — p2_t (distillation from Stockfish, a quarter of every batch, against the same schedule without it; judged on skill 5)

**Runs:** line A, the teacher: `runs/p2_t_teach` (`--config configs/p2_t_teach_hot.yaml --seed 43`: 48
simulations, lr 2e-5, `teacher_frac` 0.25, 20 iterations from `p2_s_cont/0030`), then `runs/p2_t_teach_down`
(`configs/p2_t_teach_down.yaml`: 48, 2e-6, teacher 0.25, 10 iterations from `p2_t_teach/0020`). Line B, the same
schedule without it: `runs/p2_t_ctl` (`configs/p2_k_lr.yaml`: 48, 2e-5, 20 iterations from the same start), then
`runs/p2_t_ctl_down` (`configs/p2_o_lr.yaml`: 48, 2e-6, 10 iterations from `p2_t_ctl/0020`). Four RunPod pods, each
a secure RTX 4090: line A on the 24 GB variant with 64 vCPU (`evt16z4uconhhq`, `e8cqhsu3r1p8xb`), line B on the
48 GB variant (49,140 MiB) with 32 vCPU (`f1ctavn36spo5v`, `1bxmote9dk6wdz`). Training from 2026-11-07 22:06 (after
labelling) to 2026-11-08 02:07:06 (teach), 02:19:02 to 04:23:06 (teach down), 2026-11-07 18:36:30 to 22:02:40 (ctl)
and 22:09:47 to 23:57:57 (ctl down) UTC, code `7163608`. Every measurement ran on this box (torch 2.14.1+rocm7.2)
under the user units `taz-campaign@p2_t_{teach,teach_down,ctl,ctl_down}` (`MEASURE_AFTER_PULL=1`) and
`taz-finisher@p2_t` (`runs/p2_t_finish.sh`). **Pre-registration:** `PREREG-p2_t.md` (`7163608`), one addendum (A:
the labeller's Stockfish path). Every verdict below was registered before it was measured, except where marked.

**Verdict: the registered first row — distillation works. The teacher becomes standing (0.25, this schedule), and
p2_u's lever is Michael's choice. K1 (teach down/0010 − ctl down/0010, skill 5, six reps per side): +0.050
[+0.014, +0.085]. K2 passes (189 and 190). T1 holds: line A's endpoint agrees with Stockfish's best move more often
than line B's (0.3588 against 0.3530) and has the lower cross-entropy (2.370 against 2.805). ECE and N hold; L1 holds
in all four runs; L3 holds. Against the start, line A's endpoint is +0.030 [−0.001, +0.060]: the gain is
established against the control schedule, not yet against the start (§6).**

---

## 1. Before launch

The start reference is `p2_s_cont/0030`: six skill-5 book reps (`runs/p2_s_cont/sf5/`, mean 0.3300), three skill-3
reps (0.7783), three skill-0 book reps (0.9683) and three initial-position reps (0.9717); nothing was re-measured.
Read at registration, 2026-11-07 18:11 UTC: balance $18.69, no pods of ours, secure RTX 4090 at $0.89/h; `make
test` 1,067 passed, 4 skipped.

**The pods.** All four came from `cloud/launch_run.sh`, each a transient unit armed at 18:13. Ctl was refused twice
(no instances) and launched at 18:23:49. Teach was refused 18 times, 18:13 to 19:38, and launched at 19:43:52. Each
step down waited for its hot stage's `.remote_done`: ctl down launched at 22:02:47, 7 s after ctl's final pull, and
teach down at 02:07:47, 41 s after teach's. No pod was unusable, none replaced. Every input's sha256 matched on each
pod (50 files; 65 on teach down, which also took the teacher ring).

**Addendum A.** Line A's first job (19:54:46) failed at once: the job's `PATH` lacked `/usr/games`, so
`label_teacher.py` found no Stockfish. The registered job was restarted at 20:48:07 on the same pod with
`--stockfish /usr/games/stockfish`, the same binary (`16-1build1`); nothing else changed. The addendum's text was
written at 20:49 UTC (file mtime), before any data existed. It was meant to be committed once both step downs had
launched (02:07:47) and was not; it is committed at this close, unchanged, in the commit before this RUNLOG.

## 2. L — the continuity applied

- **L1 holds, all four runs:** `seed` 43; `git_sha` `7163608`, `git_dirty` false; each `config_fingerprint`
  equal to its `config.yaml`'s sha256: `62fd3945…1573` (teach), `80de3169…5120` (teach down), `816e88dc…2546`
  (ctl), `e2e479fb…90f2` (ctl down), the registered four.
  - teach and ctl: `init_from` `runs/p2_s_cont/ckpt_iter0030.pt`; pool and ring from `runs/p2_s_cont`;
    `seed_replay` 500,000 rows, `donor_committed` = `donor_head` = 2,069,348, offset −30; `lr` 2e-5 on every row.
  - teach down: `init_from` `runs/p2_t_teach/ckpt_iter0020.pt`; `seed_replay` 500,000, committed = head =
    1,571,927 (teach's final ring), offset −20; `lr` 2e-6 on every row.
  - ctl down: `init_from` `runs/p2_t_ctl/ckpt_iter0020.pt`; `seed_replay` 500,000, committed = head = 1,551,070,
    offset −20; `lr` 2e-6 on every row.
  - Simulations per stored position **48.0** at every iteration of all four. The two lines' resolved configs differ
    in `teacher_dir` and `teacher_frac` only (`runs/p2_t_teach*/config.yaml` against `runs/p2_t_ctl*/config.yaml`).
- **L2 (reported, no bound):** teach 3 (`open_021` at 20, `open_001` at 11, `open_028` at 17), teach down 1
  (`open_012` at 5), ctl 2 (`open_021` at 20, `open_001` at 11), ctl down 1 (`open_012` at 5) in 500,000.
  Predicted 0–5 per run: held.
- **L3 holds** (`runs/p2_t_teach/teacher/teacher.json`): 472,229 rows; Stockfish 16 (`/usr/games/stockfish`);
  100,000 nodes, `multipv` 5, T 50 cp, 64 MB hash. Selection: 500,000 ring rows read, 551 excluded (the five
  suites), 3,883 repeated, 18,337 duplicates, 477,229 selected, as registered. The held-out ring
  (`teacher_holdout/`) has 5,000 rows (seed 0) on the training fit (`fitted_here` false, the same `a` and `s`).
  - **The outcome model: `a` 227.6 cp, `s` 316.7 cp**, fitted on the 472,229. At a level score it calls **34.5%
    drawn** (1 − 2σ(−a/s), computed at close from `a` and `s`; `teacher.json`'s `draw_rate` field, 0.247, is the
    share of drawn game outcomes among the rows, not the model's). Inside 5–60%: no defect. Under it a 45 cp edge
    is W 0.360 / D 0.343 / L 0.297, where Stockfish's own model calls it 92% drawn. Mean |`root_q`| 0.401.
  - Predicted 15–35% drawn: held (34.5%). Predicted `s` 100–250 cp: missed (316.7).

## 3. T1 — the lever applied: agreement with the teacher on the held-out 5,000

`scripts/teacher_agreement.py` (prior only, no search), `runs/p2_t_*/teacher_agreement_*.json`:

| checkpoint | top-1 agreement | cross-entropy vs teacher policy | value MSE vs teacher `root_q` |
|---|---|---|---|
| start, `p2_s_cont/0030` | 0.3522 | 2.895 | 0.0442 |
| teach 0020 | 0.3602 | 2.375 | 0.0503 |
| ctl 0020 | 0.3536 | 2.814 | 0.0594 |
| **teach down 0010 (A's endpoint)** | **0.3588** | **2.370** | 0.0427 |
| **ctl down 0010 (B's endpoint)** | **0.3530** | **2.805** | 0.0451 |

- **T1 holds:** line A's endpoint agrees more often (+0.58 points) and has the lower cross-entropy (−0.435 nats).
- Predicted line A's top-1 agreement at least 5 points above line B's: **missed** (0.58 points). The network moved
  toward the teacher's distribution over moves (cross-entropy −0.52 from the start, line B −0.09) and hardly at all
  toward its first choice. The value MSE against the teacher's `root_q` ended near the start's in both lines
  (0.0427 and 0.0451 against 0.0442).

## 4. K — the two lines at the end, skill 5

### K1 (primary) — teach down/0010 against ctl down/0010, six reps per side

Per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples,
seed 0; `runs/p2_t_teach_down/sf5/p2t_verdict_teach_vs_ctl.json`, reproduced byte for byte; all twelve rows
`stockfish_s5`, each side on its own checkpoint):

| line | reps r1–r6 | mean | W / D / L (1,200 games) |
|---|---|---|---|
| B, `p2_t_ctl_down/0010` | 0.2975, 0.3175, 0.32, 0.295, 0.3375, 0.2925 | 0.310 | 235 / 274 / 691 |
| A, `p2_t_teach_down/0010` | 0.3475, 0.415, 0.3475, 0.34, 0.3325, 0.3775 | 0.360 | 294 / 276 / 630 |
| **A − B** | | **+0.050** | **[+0.014, +0.085]**, SE 0.0179 |

- **The interval is entirely above zero.** Forecast +0.03 to +0.10: held. Per opening, A's rep-mean is above B's on
  56 of the 100, below on 34, equal on 10 (median +0.0625). The draw rates are the same (0.230 and 0.228); A wins
  59 more of the 1,200 games and loses 61 fewer.

### K2 — mate-in-2

At the endpoints: **teach down/0010 189/200, ctl down/0010 190/200**; 189 ≥ 190 − 4 and both ≥ 183. **K2 passes.**
At the other measures: teach 0020 188, teach down/0005 189; ctl 0020 192, ctl down/0005 192; the start 189. Seven
positions failed at all six measures of both lines: `mate2_091`, `mate2_124`–`126`, `mate2_129`, `mate2_139` and
`mate2_196`. Two failed at every line-A measure and at no line-B measure: `mate2_054` (which the start also fails)
and `mate2_138`. Predicted line A at or above line B: missed by one position at the endpoints, by four at 0020.

### K3 — each line against the start, and skill 3 (informational)

| checkpoint | reps | vs start [95% CI] |
|---|---|---|
| teach 0020, skill 5 (end of 2e-5) | 0.2825, 0.325, 0.2875 | −0.032 [−0.071, +0.007] |
| ctl 0020, skill 5 (end of 2e-5) | 0.24, 0.25, 0.275 | **−0.075 [−0.111, −0.038]** |
| teach down 0005, skill 5 | 0.2575, 0.2975, 0.35 | −0.028 [−0.065, +0.009] |
| ctl down 0005, skill 5 | 0.2875, 0.2875, 0.2625 | **−0.051 [−0.088, −0.012]** |
| **teach down 0010, skill 5 (six)** | mean 0.360 | **+0.030 [−0.001, +0.060]** |
| **ctl down 0010, skill 5 (six)** | mean 0.310 | **−0.020 [−0.051, +0.011]** |
| teach down 0010, skill 3 | 0.7375, 0.8225, 0.8175 | +0.014 [−0.029, +0.056] |
| ctl down 0010, skill 3 | 0.7725, 0.785, 0.7575 | −0.007 [−0.051, +0.037] |

Predicted: both 0020s near or below the start (held: A within noise, B below); line A above the start at its
endpoint (the point +0.030; the interval touches zero by 0.001, so not shown at 95%); line B within noise (held).

**Computed at close, not registered** (same method; `p2t_verdict_unreg_*.json`): A − B at 0020 **+0.043 [+0.003,
+0.083]**; at down/0005 +0.023 [−0.022, +0.067]. Each line's step, down/0010 (six) − 0020 (three): A **+0.062
[+0.026, +0.098]**, B **+0.055 [+0.020, +0.089]**.

### K4 — skill 0 and the initial position (continuity)

Skill-0 book reps against the start (0.9683): teach 0020 −0.008, teach down/0005 +0.004, teach down/0010 +0.001
[−0.017, +0.018]; ctl 0020 −0.010, ctl down/0005 −0.001, ctl down/0010 +0.012 [−0.005, +0.029]. Initial position
against the start (0.9717): teach 0020 +0.003, teach down/0010 +0.005; **ctl 0020 −0.018 [−0.034, −0.001]**, below
zero (informational; no row turns on it), ctl down/0010 0.000. M4 at skill 0: teach down 0.969, ctl down 0.980,
met (bar 0.675).

## 5. N, ECE, H

- **N — holds:** mate-in-2 teach 188; teach down 189, 189; ctl 192; ctl down 192, 190. Mate-in-3 teach 159; teach
  down 156, 158; ctl 161; ctl down 159, 160 (start 156). Predicted 185–195: held.
- **ECE — holds:** teach 0.0925; teach down 0.0987, 0.1010; ctl 0.0842; ctl down 0.0908, 0.0896 (< 0.15; start
  0.0972; predicted 0.08–0.14: held). Line A sits about 0.01 above line B at each measure.
- **H7 — the losses of line A against line B.** Line A's batches mix ring rows and teacher rows, whose targets are
  spread over up to five moves, so its policy loss is higher by construction: teach 1.700 → 1.755 (first five
  1.695, last five 1.750), teach down 1.758 → 1.753; ctl 1.384 → 1.455 (1.375, 1.446), ctl down 1.458 → 1.451. Both
  lines' policy loss rose about 0.06 over the 2e-5 stage and held through the step down.
- **The value loss** fell at 2e-5 and rose back through the step down in both lines, as in p2_s: teach 0.394 →
  0.342, teach down 0.347 → 0.387; ctl 0.391 → 0.331, ctl down 0.339 → 0.397.
- **H1:** `move1_prior_entropy` teach 0.046 → 0.204, teach down 0.181 → 0.256; ctl 0.046 → 0.038, ctl down 0.041 →
  0.116. **Book move-1 entropy (H6): line A 1.234, 1.262, 1.253; line B 0.574, 0.594, 0.591** (start 0.585).
- Promotion loss by halves: teach 0.394 / 0.349, teach down 0.387 / 0.429; ctl 0.373 / 0.366, ctl down 0.412 /
  0.474. Decisive rate teach 0.796 / 0.792, teach down 0.779 / 0.792; ctl 0.806 / 0.788, ctl down 0.789 / 0.806.
  Self-play games 104.7, 105.4 (A) and 102.6, 103.1 (B) plies.
- **H2:** 0 skipped steps in 8,000 + 4,000 per line.
- **H10, pace:** teach 11.76 min per iteration (self-play 10.82), teach down 11.89 (10.97); ctl 10.12 (9.23), ctl
  down 10.56 (9.67). Sims/s 3,695–4,125 (teach), 3,881–4,032 (teach down), 4,270–4,824 (ctl), 4,201–4,495 (ctl
  down). Training 56 s per iteration in line A against 53 s in line B. The lines ran on different pod types (64
  vCPU and the 24 GB card against 32 vCPU and the 48 GB card), so the self-play gap is not attributed.
- **H11, labelling:** 4,668 s (77.8 min) for 477,229 positions with 54 workers on the 64-vCPU pod (CPU quota
  54.4), about 0.53 s per position per worker.

Rungs: random, greedy and phase1 won 200/200 at every checkpoint in all four runs.

## 6. Decision

Per the table's first row (K1 entirely above zero, K2 passes, T1 holds): **distillation works. The teacher becomes
standing at 0.25 with this schedule. p2_u's lever is Michael's choice of more of it: a fresh teacher ring from the
new endpoint's own positions, a larger dose, or a deeper engine.** N and ECE hold in all four runs.

Readings the table does not cover:

1. **The lead formed in the 2e-5 stage, and the step down kept it.** A − B was +0.043 [+0.003, +0.083] at 0020
   and +0.050 at the end; each line's step gained about the same (+0.062 and +0.055; all unregistered). In terms of
   the start: the 2e-5 stage cost line B 0.075 and line A 0.032, and both steps returned about 0.06. Line A's
   endpoint, 0.360, is the highest six-rep skill-5 mean so far (the earlier six-rep means run 0.302–0.341, p2_q to
   p2_s; one three-rep reading, `p2_s_cont/0020`, was 0.3675), but against the start it is +0.030 [−0.001, +0.060].
   The registered claim is against the control schedule. A gain over the start is likely and not yet shown at 95%.
2. **What moved was the shape of the policy, not its first choice.** Cross-entropy to the teacher fell 0.52 nats;
   top-1 agreement rose 0.7 points; the value MSE against the teacher's `root_q` ended where it began. The book move-1
   entropy doubled (1.25 against 0.59). The teacher's targets spread weight over up to five moves within about 50 cp
   of each other, and the network learned that spread. One candidate mechanism: wider priors let the 48-simulation
   search consider moves it otherwise drops. This is a hypothesis, not tested here.
3. **A small, consistent cost at mate-in-2 and in calibration.** Line A was 1–4 positions below line B at every
   measure (188/189/189 against 192/192/190), with `mate2_054` and `mate2_138` failing at every line-A measure
   only; its ECE was about 0.01 higher at each measure. Both are inside the guards and both point the same way. A
   larger dose should watch them.
4. **The design's open alternative stands.** Line A's teacher rows are the start ring's positions, seen again after
   the ring turned over; the gain could in part come from the old positions, not from Stockfish's targets on them.
   Reading 2 makes the targets the likelier source (the policy moved toward the teacher's distribution, which only
   its targets carry), but the third line that would separate them was not run.
5. **The schedule alone returned to the start again** (ctl down/0010 −0.020 [−0.051, +0.011]), as in p2_s.

The open list, carried forward: the 2e-7 step-down; the calibration sample; the promotion loss at 2e-6 (it rose
through both step downs again: 0.412 → 0.474 in line B); L2's exclusion at storage time; the host crashes; a public
benchmark (a UCI wrapper first); the third line that separates old positions from the teacher's targets. **Disk:
11 GB free (98%) at close; this round added about 7.5 GB, so the next round will not fit with any margin unless
old run directories are pruned or moved** (Michael's call).

## 7. Mechanics

- **Unattended after Addendum A.** `launch_run.sh` launched, staged, verified and started all four pods and enabled
  their drivers; each step down launched itself after its hot stage's final pull. The one hand intervention was
  Addendum A's restart. While the failed job's `DONE` stood, line A's driver logged 10 "final pull INCOMPLETE"
  rounds (19:55–20:45, no run directory yet); harmless. No measurement was retried, and this box stayed up
  throughout.
- **Lifecycle:** final pulls with the ring at 22:02:40 (ctl), 23:57:57 (ctl down), 02:07:06 (teach) and 04:23:06
  (teach down); each pod deleted itself after `PULLED` (0 pods of ours at close). Measurements under `gpu hold`:
  ctl 0020 22:02–22:48; ctl down 0005 23:57–00:26, 0010 –01:09; teach 0020 02:07–02:51; teach down 0005
  04:23–04:51, 0010 –05:34. The finisher from 05:38:13: T1 at the five checkpoints, skill-5 reps teach/0020,
  ctl/0020, teach down/0005, ctl down/0005, teach down/0010 (six), ctl down/0010 (six), skill-3 reps at both
  endpoints, M4, L2, and K1 at 09:58:52 UTC on 2026-11-08.
- **Cost:** pod time at $0.89/h ≈ **$12.65** (teach 6.39 h $5.69, of which Addendum A's idle about $0.80; ctl
  3.65 h $3.25; ctl down 1.92 h $1.71; teach down 2.26 h $2.01). The balance went from $18.69 to $55.91 across one
  auto-reload at the $10 floor, between 22:02 ($13.39) and 02:07 ($57.97); from those launch-log readings the reload
  was $50, so the spend was **$12.78**. Expected about $11, worst case $26.70. All RunPod use by this project to
  date about $137.

## 8. Artefacts

| file | what |
|---|---|
| `runs/p2_t_{teach,teach_down,ctl,ctl_down}/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the runs |
| `runs/p2_t_teach/teacher/teacher.json`, `teacher/meta.json`, `teacher_holdout/teacher.json`, `teacher_holdout/meta.json` | L3; the rings themselves stay untracked |
| `runs/p2_t_*/teacher_agreement_*.json` | T1 |
| `runs/p2_t_*/ladder.jsonl` | ladder passes at teach and ctl 20, both step downs at 5 and 10 (ECE, N) |
| `runs/p2_t_*/sf5/` | skill-5 reps and each checkpoint's verdict against the start (K3) |
| `runs/p2_t_teach_down/sf5/p2t_verdict_teach_vs_ctl.json` | K1 |
| `runs/p2_t_teach/sf5/p2t_verdict_unreg_teach_vs_ctl_iter0020.json`, `runs/p2_t_teach_down/sf5/p2t_verdict_unreg_teach_vs_ctl_iter0005.json`, `runs/p2_t_{teach,ctl}_down/sf5/p2t_verdict_unreg_step.json` | computed at close, not registered |
| `runs/p2_t_{teach,ctl}_down/sf3/` | skill-3 reps at the endpoints and their verdicts |
| `runs/p2_t_*/reps_iter*_r*.jsonl`, `p2_verdict_reps*.json`, `p2_m4_iter*.json`, `startpos/` | K4, skill 0 |
| `runs/p2_t_*/book_rows.json` | L2 |
| `runs/p2_t_*_launch.console.log`, `runs/p2_t_*_campaign.console.log`, `runs/p2_t_finish.console.log`, `runs/p2_t_finish.sh` | consoles and the finisher (untracked) |
