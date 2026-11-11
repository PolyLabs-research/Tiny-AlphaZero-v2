# Pre-registration — p2_w: the quarter line's batch with 256 more teacher rows on top, beside a second run of the quarter line

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-11** (see Sign-off), at Michael's instruction. Nothing under this file
> has run.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_v ran the standing teacher at half of every batch against a quarter, and the half line ended weaker at skill 5
(K1 −0.044 [−0.078, −0.010]) although it ended 0.097 nats closer to the teacher on held-out positions. The
registered row asks for a diagnosis before the next lever, with two suspects the design did not separate: each
teacher row drawn twice as often (12.9 passes against 6.5), and 256 fewer self-play rows in every batch (461
against 717). H12 did not point at the first (training-ring gap −0.021 against −0.015). Michael chose (2026-11-11)
the separation RUNLOG-p2_v §6 named: **keep the quarter line's batch whole and add the extra teacher rows on top.**

**H_rows: the half line's cost came from the self-play rows it gave up. With all 717 kept, 512 teacher rows a step
cost nothing against 256.**

A second question rides beside it. p2_v's verdict, and every A/B since p2_k, compares one run per arm, and the
quarter line's 0.378 is the highest six-rep endpoint of p2_t–p2_v without separating from its start. **The quarter
line is run a second time here, unchanged**, which gives wide a concurrent comparison line and gives the project
its first endpoint-to-endpoint measure of how far two runs of one protocol land apart (K0).

The network stays at 6.5M parameters. The instrument stays where it was.

## The batches

Two lines run. Two more are p2_v's, already run from the same start, seed, ring and schedule.

| | Line A, "wide" (runs here) | Line B, "quarter" (runs here; p2_v's line B again) | p2_v "quarter" (as run) | p2_v "half" (as run) |
|---|---|---|---|---|
| Batch | **1,280** | 1,024 | 1,024 | 1,024 |
| Self-play ring rows | **717** | 717 | 717 | 461 |
| Mate rehearsal rows (mate-in-1 + mate-in-2) | 20 + 31 | 20 + 31 | 20 + 31 | 20 + 31 |
| Teacher rows, from `runs/p2_u_fresh/teacher` | **512** | 256 | 256 | 512 |
| Teacher draws over 30 iterations (12,000 steps) | 6.14 million, **12.9 passes** | 3.07 million, 6.5 passes | as line B | 6.14 million, 12.9 passes |
| Self-play ring draws over 30 iterations | **8.60 million** | 8.60 million | as line B | 5.53 million |
| Share of the batch-mean loss: ring / teacher / rehearsal | 0.560 / 0.400 / 0.040 | 0.700 / 0.250 / 0.050 | as line B | 0.450 / 0.500 / 0.050 |
| Teacher rows per self-play row | 0.714 | 0.357 | 0.357 | 1.111 |
| Endpoint, skill 5, six reps | to be measured | to be measured | 0.3779 | 0.3338 |

Wide equals quarter in every self-play and rehearsal row count and equals half in every teacher row count. Against
half, wide differs only in having 256 more self-play rows a step; against quarter, only in having 256 more teacher
rows a step.

## Contract

| | |
|---|---|
| Start | **`p2_u_stale_down/0010`**, p2_v's start, for both lines: `--seed 43 --init runs/p2_u_stale_down/ckpt_iter0010.pt --seed-pool runs/p2_u_stale_down --seed-replay runs/p2_u_stale_down` (ring head 1,037,734, offset −10) |
| Line A, wide | **`p2_w_wide`, hot:** `--config configs/p2_w_wide_hot.yaml` (48 sims, **2e-5**, batch 1,280; seed-43 fingerprint `99a0b4ae…ee13`), 20 iterations. Then **`p2_w_wide_down`:** `--config configs/p2_w_wide_down.yaml` (48, **2e-6**; `b812a7b4…c12c`), `--init runs/p2_w_wide/ckpt_iter0020.pt`, `--seed-pool` and `--seed-replay runs/p2_w_wide`, 10 iterations |
| Line B, quarter | **`p2_w_quarter`, hot:** `--config configs/p2_u_fresh_hot.yaml` (48, **2e-5**, batch 1,024; `066043ca…d67d`), 20 iterations. Then **`p2_w_quarter_down`:** `--config configs/p2_u_fresh_down.yaml` (48, **2e-6**; `036cd665…01da`), from `runs/p2_w_quarter` in the same way, 10 iterations. The commands are p2_v's line B's with the output directories renamed; the configs, seed and start are the same files and values |
| What differs between the lines' configs | five fields of `train`, and nothing else (`test_p2w_wide_configs_are_the_brief`): `batch_size` 1280; `teacher_frac` 0.4 (512 of 1,280); `rehearsal_mate1_frac` 0.015625 (20), `rehearsal_mate2_frac` 0.02421875 (31), `rehearsal_frac` 0.03984375 (their sum, 51). The fractions are exact in binary. The test pins the three batch plans: (717, 20, 31, 512), (717, 20, 31, 256), (461, 20, 31, 512) |
| The p2_v lines | `p2_v_quarter_down/0010` and `p2_v_half_down/0010`, as p2_v measured them (RUNLOG-p2_v, `8c9f61d`): six skill-5 reps each (`runs/p2_v_{quarter,half}_down/sf5/reps_iter0010_r{1..6}.jsonl`), not re-measured |
| Why the same seed for line B | a re-run at seed 43 is a second realisation, not a copy: the pods do not reproduce each other bit for bit. p2_v's two lines shared the seed, the start network and the first iteration's self-play (played before any training step) and logged 53,146 and 52,779 positions in it; p2_j's two attempts at one protocol read 194 and 189 on mate-in-2 at 0020. Keeping the seed keeps line B the protocol p2_v ran, so K0 measures the spread of that protocol and K1 stays a same-seed comparison. L5 checks the premise on the runs themselves |
| Start reference | `p2_u_stale_down/0010`: six skill-5 reps (mean 0.3608), three skill-3 reps (0.8333), three skill-0 book reps (0.9750), three initial-position reps (0.9875); mate-in-2 191/200, mate-in-3 156/200, ECE 0.1049. T1 on the held-out ring: top-1 0.3590, cross-entropy 2.391; on the whole training ring 2.393 |
| Training | RunPod, secure RTX 4090, ≥ 24 vCPU, the image of p2_g–p2_v; variants recorded. If none is offered the registered launch is retried; a pod that never comes up is terminated and replaced (`cloud/launch_run.sh`); nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, `gpu hold`, nothing until each run's final pull, under the user units, `MEASURE_AFTER_PULL=1`. Drivers judged: each hot run `20`; each step down `5 10` (ladder with ECE and N, three skill-0 reps; initial-position reps at each run's last). Then a finisher (`runs/p2_w_finish.sh`) against `stockfish:5`, for both lines: three reps at 0020 and down/0005; **six at down/0010**; three against `stockfish:3` at the endpoint; T1 on the held-out ring at 0020 and the endpoint; H12 at the endpoint |
| Lifecycle | four guards. Each hot stage deadline 8 h, worst $7.12; each step down 6 h, $5.34. **Worst case $24.92**; expected about **$11.50** (p2_v's two lines cost $10.86; wide adds a few seconds a step for the larger batch). Both hot stages launch together; each step down launches after its own hot stage's final pull |

## Registered expectations

### L — the continuity applied
- **L1.** Each run's `provenance.json`: the `init_from`, `seed_replay` head and offset above; `seed` 43; `git_dirty`
  false; `config_fingerprint` as above, equal to its `config.yaml`'s sha256; `lr` on every row 2e-5 (hot) or 2e-6
  (down); simulations per position 48.0; `batch_size`, `teacher_frac`, the three rehearsal fractions and
  `teacher_dir` as registered in each `config.yaml`. Line B's two `config.yaml` files are byte-identical to
  `runs/p2_v_quarter/config.yaml` and `runs/p2_v_quarter_down/config.yaml`.
- **L2.** Book positions in each final ring: reported, no bound.
- **L3, the ring as registered.** The staged `runs/p2_u_fresh/teacher/` on each pod has the sha256 of this box's
  copy, and its `teacher.json` reads 475,696 rows.
- **L4, the instrument and the trainer as p2_v's.** `git diff 56328bb <registration commit> -- src scripts cloud`
  is empty: the code that trains and measures both lines is the code that trained and measured p2_v's. Stockfish
  `16-1build1`; torch 2.14.1+rocm7.2 on this box. The twelve p2_v rep files are byte-identical to the ones
  committed at `8c9f61d`.
- **L5, line B is a second realisation.** `runs/p2_w_quarter/iterations.jsonl` differs from
  `runs/p2_v_quarter/iterations.jsonl` in its first row's self-play counts (p2_v's: 53,146 positions), and
  `runs/p2_w_quarter_down/ckpt_iter0010.pt` does not have p2_v's endpoint's sha256 (`c7a97b23…d755`). If the two
  runs turn out identical, K0 is void and is reported as such, K1 is a comparison against p2_v's realisation, and
  the table's K0 column is read as "includes zero" with that limit stated.

### T1 — the lever applied
`scripts/teacher_agreement.py` on `runs/p2_u_fresh/teacher_holdout`, at both 0020s and both endpoints. **T1 holds if
wide's endpoint cross-entropy is below line B's.** Reported beside p2_v's 2.369 (quarter) and 2.272 (half); top-1
reported, not judged. Without T1, K1 cannot be read as a test of the extra teacher rows.

### K0 — the quarter line against itself, skill 5
`p2_w_quarter_down/0010` − `p2_v_quarter_down/0010`, against `stockfish:5`, six reps per side, per-opening
rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed 0;
`runs/p2_w_quarter_down/sf5/p2w_verdict_quarter_replicate.json`), read with the exact-zero rule. Six reps resolve
about ±0.034 (p2_v's K1 half-width). The two runs share config, seed, start, code and instrument, so K0 has no
treatment in it: it is one draw of the difference between two runs of one protocol, which K1 and K1b cannot see
because they resample openings only. **K0 is read first, and it decides how far K1 and K1b may be read** (the
decision table).

**K0b (reported, no decision):** `p2_w_quarter_down/0010` − `p2_v_half_down/0010`
(`runs/p2_w_quarter_down/sf5/p2w_verdict_quarter_vs_p2v_half.json`): p2_v's K1 with a second draw of its quarter
line.

### K1 (primary) — wide against quarter, concurrent, skill 5
`p2_w_wide_down/0010` − `p2_w_quarter_down/0010`, the same method
(`runs/p2_w_wide_down/sf5/p2w_verdict_wide_vs_quarter.json`). Both lines are run and measured in this round, and
neither endpoint is known as this is written.

### K1b — wide against p2_v's half, skill 5
`p2_w_wide_down/0010` − `p2_v_half_down/0010`, the same method
(`runs/p2_w_wide_down/sf5/p2w_verdict_wide_vs_half.json`). K1b is the direct test of H_rows: the two lines differ
only in the 256 self-play rows. The half line is not re-run, so K1b is against one realisation, measured on
2026-11-11.

### K2 — mate-in-2
At the endpoint: **`p2_w_wide_down/0010` solves at least line B's endpoint count minus 4, and at least 183.**

### K3 — against the start (informational)
For both lines at skill 5: 0020 and down/0005 (three reps), the endpoint (six), each against
`p2_u_stale_down/0010`'s six. At skill 3: each endpoint's three reps against the start's three. Skill 0, M4 and the
initial position as before. Each line's step, down/0010 − 0020, is computed and reported beside p2_v's two (+0.070
quarter, −0.005 half).

### ECE, H
- **ECE:** < 0.15 at wide's endpoint. A break is handled in PREREG-p2_l's order. Line B's is reported.
- **H1–H10** as PREREG-p2_r, direction-free. **H7:** wide's batch mixes its sources in a third proportion, so its
  loss totals are not comparable with line B's or with either p2_v line; line B's are comparable with p2_v's
  quarter's and are reported beside them. **H2:** `skipped_steps` < 20. **H6:** book move-1 entropy, reported
  beside p2_v's 1.61 (half) and 1.23 (quarter).
- **H12, memorisation (reported, no bound):** cross-entropy on the whole training ring minus that on the held-out
  5,000, at both endpoints; beside p2_v's −0.021 (half) and −0.015 (quarter).
- **H13, the step (new, reported, no bound):** `grad_norm` before clipping, per run, beside p2_v's. `grad_clip` is
  1.0 and p2_v's norms ran 5–9, so every step is clipped: the update's size is set by the clip and the learning
  rate in every line, and the batch decides its direction.
- **H14, the replicate elsewhere (new, reported, no bound):** line B beside p2_v's quarter on every reading the
  two share: T1 (2.369), mate-in-2 (193), ECE (0.0958), skill 3 (0.820), the 0020 and down/0005 skill-5 readings,
  the step (+0.070), H6 (1.23).

## Predictions (written before launch)
- L5 holds: line B is not a copy.
- K0: line B's endpoint 0.35–0.38, point difference −0.03 to +0.005, on the view that 0.378 was a somewhat high
  draw. About two in three that the CI includes zero.
- T1 holds: wide's endpoint at 2.27–2.33 on the held-out ring, line B's at 2.35–2.39. H12 −0.01 to −0.04.
- K1: point −0.03 to +0.02; more likely than not the CI includes zero.
- K1b: point 0.00 to +0.05; more likely than not the CI includes zero.
- So the likeliest single outcome is "not separated". The round separates the suspects only if wide lands clear
  of line B on one side or clear of p2_v's half on the other.
- K2 passes (wide 190–194). K3: both 0020s below the start; both endpoints within noise of it.
- ECE 0.08–0.13; N 185–195; L2 0–5 per run; wide's H6 between 1.23 and 1.61.

## Decision rules

K0 is read first.

| K0 (line B − p2_v quarter) | K1 (wide − line B) | K1b (wide − p2_v half) | T1 | Decision |
|---|---|---|---|---|
| **CI excludes zero**, either side | any | any | any | **Two runs of one protocol differ by as much as the differences being judged.** K1 and K1b are reported and decide nothing. The dose stays at 0.25 and the batch at 1,024. RUNLOG-p2_v takes a dated addendum giving K0 and K0b beside its K1. p2_x replicates before any lever is read from a single pair again; how (more seeds per arm, or more runs of these lines) is Michael's choice |
| includes zero | **CI entirely above zero** | any | holds | **More teacher helps once the self-play rows are kept.** The standing batch becomes 1,280 (717 / 51 / 512) with this schedule, if K2 passes; if K2 fails it stays as it is until the mate-in-2 cost is diagnosed |
| includes zero | **CI includes zero** | **CI entirely above zero** | holds | **H_rows supported: the half line's cost was the self-play rows it gave up.** The extra teacher rows neither help nor cost; the standing batch stays 1,024 at 0.25, the cheaper of two equals. The "more of this teacher" lever is closed; p2_x's lever is Michael's choice: a deeper engine, the 2e-7 step-down, the public benchmark, the third line (RUNLOG-p2_t §6, reading 4) |
| includes zero | CI includes zero | CI includes zero | holds | **Not separated.** Wide sits between the lines and the round resolves neither difference. Read as consistent with a cost that grows with the teacher's share of the batch; not read as support for H_rows. The dose stays at 0.25 and the lever is closed as above |
| includes zero | **CI entirely below zero** | any | holds | **The extra teacher rows cost skill by themselves**, with every self-play row kept: H_rows is refuted as the whole account. The dose stays at 0.25; the lever is closed as above |
| includes zero | any | any | fails | **The lever was not applied in effect**: wide did not end closer to the teacher than line B on held-out positions. H12 and H13 are read first; not read as a result about the rows |
| any other combination | | | | Reported as measured, no decision; the dose stays at 0.25 |
| N < 183 or ECE ≥ 0.15 at wide's endpoint | | | | The broken guard first |

K0 including zero does not show the spread is small: it is one pair, resolved to about ±0.034. It licenses reading
K1 and K1b as the earlier rounds' comparisons were read, and no more.

## Measurement protocol

```bash
# The launch: runs/p2_w_launch.sh (cloud/launch_run.sh four times, as transient user units).
#   p2_w_wide         (p2ww, 8 h): train_phase2.py --config configs/p2_w_wide_hot.yaml <start> --out runs/p2_w_wide --iterations 20
#   p2_w_quarter      (p2wq, 8 h): train_phase2.py --config configs/p2_u_fresh_hot.yaml <start> --out runs/p2_w_quarter --iterations 20
#   <start> = --seed 43 --init runs/p2_u_stale_down/ckpt_iter0010.pt --seed-pool runs/p2_u_stale_down
#             --seed-replay runs/p2_u_stale_down
#   inputs: the start's 20 pool checkpoints, state.json, config.yaml, replay/, runs/p2_u_fresh/teacher/, the mate
#   data, phase1_anchor.pt
#   p2_w_wide_down    (p2wwd, 6 h), after its hot stage's final pull (WAIT_FOR=runs/p2_w_wide/.remote_done):
#   configs/p2_w_wide_down.yaml from runs/p2_w_wide, inputs including runs/p2_u_fresh/teacher/
#   p2_w_quarter_down (p2wqd, 6 h), after runs/p2_w_quarter/.remote_done: configs/p2_u_fresh_down.yaml from
#   runs/p2_w_quarter, likewise

# After all four: runs/p2_w_finish.sh under taz-finisher@p2_w — reps, each verdict against the start, T1, H12, then:
V=runs/p2_v_quarter_down/sf5 H=runs/p2_v_half_down/sf5 Q=runs/p2_w_quarter_down/sf5 W=runs/p2_w_wide_down/sf5
.venv/bin/python scripts/eval_reps.py verdict --a p2v_quarter $V/reps_iter0010_r*.jsonl \
    --b quarter $Q/reps_iter0010_r*.jsonl --out $Q/p2w_verdict_quarter_replicate.json        # K0
.venv/bin/python scripts/eval_reps.py verdict --a p2v_half $H/reps_iter0010_r*.jsonl \
    --b quarter $Q/reps_iter0010_r*.jsonl --out $Q/p2w_verdict_quarter_vs_p2v_half.json      # K0b
.venv/bin/python scripts/eval_reps.py verdict --a quarter $Q/reps_iter0010_r*.jsonl \
    --b wide $W/reps_iter0010_r*.jsonl --out $W/p2w_verdict_wide_vs_quarter.json             # K1
.venv/bin/python scripts/eval_reps.py verdict --a p2v_half $H/reps_iter0010_r*.jsonl \
    --b wide $W/reps_iter0010_r*.jsonl --out $W/p2w_verdict_wide_vs_half.json                # K1b
```

## What would make p2_w uninterpretable
- A difference between the two lines' configs beyond the five fields (the config test and L1), or a line B config
  that is not p2_v's quarter's.
- A change to the trainer or the instrument since `56328bb` (L4), or a p2_v rep file that is not p2_v's.
- A teacher ring other than the registered one (L3).
- A launch from anything but clean, pushed `main`.
- Losing a run's final pull: re-run by addendum, not read from what is left.

**What the design does not separate.** Wide holds the number of self-play rows, not their weight. The loss is a
mean over the batch, so the 717 self-play rows carry 0.560 of it in wide against 0.700 in quarter and 0.450 in
half; in teacher rows per self-play row wide (0.714) sits halfway between quarter (0.357) and half (1.111). Every
step is clipped to the same norm, so that ratio is what the batch contributes to the update. If the cost in p2_v
follows the ratio and not the row count, wide lands between the lines, which is the "not separated" row. Holding
the weight as well would need per-source loss weights, a code change, and is not run here; a wide endpoint between
the lines is the result that would make it the next round. The larger batch also averages 25% more rows a step; no
line isolates that.

**What the statistic does not cover.** As PREREG-p2_k: each line is one run; K1 and K1b resample openings. K0 puts
one number on what that leaves out, for the quarter protocol only, and one pair cannot bound it. Wide and the half
line are still single runs. K1b's reference was measured on 2026-11-11 and its endpoint (0.3338) was known when
this was written; the instrument is the same code and binaries (L4), and no drift between days has been seen on
it, but none is measured here beyond what K0 would show.

---

## Sign-off

Checked for the draft, 2026-11-11, on this box: `configs/p2_w_wide_{hot,down}.yaml` differ from
`configs/p2_u_fresh_{hot,down}.yaml` in the five registered fields only, and the three batch plans are
(717, 20, 31, 512), (717, 20, 31, 256) and (461, 20, 31, 512) (test passes). Seed-43 fingerprints:
`99a0b4ae74b99baa162ef054e716439a2841ef47e165b1a1949f91f259e3ee13` (wide hot),
`b812a7b4aba54a112f9447001e10b51d3a83926867b2aaf1d2b22eb94a57c12c` (wide down),
`066043cad1786a4c86f8780f8915a5e958cf905ea9c65ac73a5f90fc3aa4d67d` (quarter hot) and
`036cd665672a35348bff7ce876011f3350b2c5426780bdc95d1dafd6666e01da` (quarter down), the last two read from p2_v's
quarter runs' `provenance.json` and equal to their `config.yaml` sha256. A zero-iteration run on CPU with the wide
hot config and the registered start (`--iterations 0`, into the session scratchpad): batch 1,280, rehearsal
0.03984375 (0.015625 + 0.02421875), pool seeded with all 20 snapshots, ring seeded with 500,000 rows from donor
head 1,037,734 at offset −10, `teacher_frac` 0.4 in its `config.yaml`, whose sha256 is the fingerprint above. The
start's league pool is complete (`ckpt_iter0000`–`0010` and the nine `seed_p2_u_stale_ckpt_iter0012`–`0020`
hard links p2_v's draft restored). `git diff 56328bb HEAD -- src scripts cloud` is empty. The twelve p2_v rep
files are tracked at `8c9f61d`. p2_v's quarter endpoint checkpoint has sha256
`c7a97b239620b71d5948b0368691fdce74d72b42e1f79011aa87bd3f3025d755`. The `p2w_verdict_*` ignore negation is added
and red-checked through `MUST_REACH` (the verdict files fail without it). RunPod balance $33.68 (auto-reload at
$10), no pods of ours; secure RTX 4090 at $0.89/h; each pod's worst case passes the $15 ceiling. The two hot
stages' worst case, $14.24, leaves $19.44; the round's worst case, $24.92, leaves $8.76, **below the $10
auto-reload line**, so a round that runs every guard to its deadline triggers a reload; the expected $11.50 leaves
about $22. Disk 98 GB free; the round adds about 6.5 GB.

Read at registration, 2026-11-11 14:07 UTC: balance $33.68, no pods of ours, secure RTX 4090 at $0.89/h; each pod's
worst case passes the ceiling and the margin at its own launch. `make lint` and `make test` passed on the draft's
tree, `e27174b` (1,162 passed, 4 skipped); this commit changes only this file. `git diff 56328bb e27174b -- src
scripts cloud` is empty. Nothing is committed to `main` between this commit and the step downs' launches, so all
four runs carry one `git_sha`.

**Signed: the Claude Code session, at Michael's instruction ("go", 2026-11-11, after the second draft and its cost —
expected about $11.50, worst case $24.92, the worst case passing under the $10 auto-reload line — were put to
him).** The commit carrying this line is the registration.
