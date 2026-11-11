# Pre-registration — p2_w: the quarter line's batch with 256 more teacher rows on top, against p2_v's two lines

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: DRAFT 2026-11-11.** Not registered. Nothing under this file has run, and nothing launches until
> Michael has seen the draft and its cost and said go.

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

The network stays at 6.5M parameters. The instrument stays where it was.

## The three batches

One line runs. The other two are p2_v's, already run from the same start, seed, ring and schedule.

| | Line A, "wide" (runs here) | "quarter" (p2_v line B, as run) | "half" (p2_v line A, as run) |
|---|---|---|---|
| Batch | **1,280** | 1,024 | 1,024 |
| Self-play ring rows | **717** | 717 | 461 |
| Mate rehearsal rows (mate-in-1 + mate-in-2) | 20 + 31 | 20 + 31 | 20 + 31 |
| Teacher rows, from `runs/p2_u_fresh/teacher` | **512** | 256 | 512 |
| Teacher draws over 30 iterations (12,000 steps) | 6.14 million, **12.9 passes** | 3.07 million, 6.5 passes | 6.14 million, 12.9 passes |
| Self-play ring draws over 30 iterations | **8.60 million** | 8.60 million | 5.53 million |
| Share of the batch-mean loss: ring / teacher / rehearsal | 0.560 / 0.400 / 0.040 | 0.700 / 0.250 / 0.050 | 0.450 / 0.500 / 0.050 |
| Teacher rows per self-play row | 0.714 | 0.357 | 1.111 |
| Endpoint, skill 5, six reps | to be measured | 0.3779 | 0.3338 |

Wide equals quarter in every self-play and rehearsal row count and equals half in every teacher row count. Against
half, wide differs only in having 256 more self-play rows a step; against quarter, only in having 256 more teacher
rows a step.

## Contract

| | |
|---|---|
| Start | **`p2_u_stale_down/0010`**, p2_v's start: `--seed 43 --init runs/p2_u_stale_down/ckpt_iter0010.pt --seed-pool runs/p2_u_stale_down --seed-replay runs/p2_u_stale_down` (ring head 1,037,734, offset −10) |
| Line A, wide | **`p2_w_wide`, hot:** `--config configs/p2_w_wide_hot.yaml` (48 sims, **2e-5**, batch 1,280; seed-43 fingerprint `99a0b4ae…ee13`), 20 iterations. Then **`p2_w_wide_down`:** `--config configs/p2_w_wide_down.yaml` (48, **2e-6**; `b812a7b4…c12c`), `--init runs/p2_w_wide/ckpt_iter0020.pt`, `--seed-pool` and `--seed-replay runs/p2_w_wide`, 10 iterations |
| What differs from the quarter line's configs | five fields of `train`, and nothing else (`test_p2w_wide_configs_are_the_brief`): `batch_size` 1280; `teacher_frac` 0.4 (512 of 1,280); `rehearsal_mate1_frac` 0.015625 (20), `rehearsal_mate2_frac` 0.02421875 (31), `rehearsal_frac` 0.03984375 (their sum, 51). The fractions are exact in binary. The test pins the three batch plans: (717, 20, 31, 512), (717, 20, 31, 256), (461, 20, 31, 512) |
| The comparison lines | `p2_v_quarter_down/0010` and `p2_v_half_down/0010`, as p2_v measured them (RUNLOG-p2_v, `8c9f61d`): six skill-5 reps each (`runs/p2_v_{quarter,half}_down/sf5/reps_iter0010_r{1..6}.jsonl`), not re-measured. Neither line is re-run |
| Why no line is re-run | the two lines exist from this start, seed and code, measured on this instrument on 2026-11-11; a second run of either would buy a replicate, not the comparison. The cost of that choice is under "What the statistic does not cover" |
| Start reference | `p2_u_stale_down/0010`: six skill-5 reps (mean 0.3608), three skill-3 reps (0.8333), three skill-0 book reps (0.9750), three initial-position reps (0.9875); mate-in-2 191/200, mate-in-3 156/200, ECE 0.1049. T1 on the held-out ring: top-1 0.3590, cross-entropy 2.391; on the whole training ring 2.393 |
| Training | RunPod, secure RTX 4090, ≥ 24 vCPU, the image of p2_g–p2_v; variants recorded. If none is offered the registered launch is retried; a pod that never comes up is terminated and replaced (`cloud/launch_run.sh`); nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, `gpu hold`, nothing until each run's final pull, under the user units, `MEASURE_AFTER_PULL=1`. Drivers judged: `p2_w_wide` `20`; `p2_w_wide_down` `5 10` (ladder with ECE and N, three skill-0 reps; initial-position reps at each run's last). Then a finisher (`runs/p2_w_finish.sh`) against `stockfish:5`: three reps at 0020 and down/0005; **six at down/0010**; three against `stockfish:3` at the endpoint; T1 on the held-out ring at 0020 and the endpoint; H12 at the endpoint |
| Lifecycle | two guards. Hot stage deadline 8 h, worst $7.12; step down 6 h, $5.34. **Worst case $12.46**; expected about **$5.50** (about 3.8 h and 2.1 h at $0.89/h, as p2_v's lines, plus a few seconds a step for the larger batch). The step down launches after the hot stage's final pull |

## Registered expectations

### L — the continuity applied
- **L1.** Each run's `provenance.json`: the `init_from`, `seed_replay` head and offset above; `seed` 43; `git_dirty`
  false; `config_fingerprint` as above, equal to its `config.yaml`'s sha256; `lr` on every row 2e-5 (hot) or 2e-6
  (down); simulations per position 48.0; `batch_size`, `teacher_frac`, the three rehearsal fractions and
  `teacher_dir` as registered in each `config.yaml`.
- **L2.** Book positions in each final ring: reported, no bound.
- **L3, the ring as registered.** The staged `runs/p2_u_fresh/teacher/` on each pod has the sha256 of this box's
  copy, and its `teacher.json` reads 475,696 rows.
- **L4, the instrument and the trainer as p2_v's (new).** `git diff 56328bb <registration commit> -- src scripts
  cloud` is empty: the code that trains the wide line and measures it is the code that trained and measured the
  two comparison lines. Stockfish `16-1build1`; torch 2.14.1+rocm7.2 on this box. The twelve reference rep files
  are byte-identical to the ones committed at `8c9f61d`.

### T1 — the lever applied
`scripts/teacher_agreement.py` on `runs/p2_u_fresh/teacher_holdout`, at wide 0020 and wide down/0010. **T1 holds if
wide's endpoint cross-entropy is below the quarter line's 2.369.** Reported beside the half line's 2.272; top-1
reported, not judged. Without T1, K1 cannot be read as a test of the extra teacher rows.

### K1 (primary) — wide against quarter, skill 5
`p2_w_wide_down/0010` − `p2_v_quarter_down/0010`, against `stockfish:5`, **six reps per side**, per-opening
rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed 0;
`runs/p2_w_wide_down/sf5/p2w_verdict_wide_vs_quarter.json`), read with the exact-zero rule. Six reps resolve about
±0.034 (p2_v's K1 half-width).

### K1b — wide against half, skill 5
`p2_w_wide_down/0010` − `p2_v_half_down/0010`, the same method
(`runs/p2_w_wide_down/sf5/p2w_verdict_wide_vs_half.json`). K1b is the direct test of H_rows: the two lines differ
only in the 256 self-play rows.

### K2 — mate-in-2
At the endpoint: **`p2_w_wide_down/0010` solves at least 189 of the 200** (the quarter line's 193 minus 4) **and at
least 183**.

### K3 — against the start (informational)
At skill 5: 0020 and down/0005 (three reps), the endpoint (six), each against `p2_u_stale_down/0010`'s six. At skill
3: the endpoint's three reps against the start's three. Skill 0, M4 and the initial position as before. The step,
down/0010 − 0020, is computed and reported beside p2_v's two (+0.070 quarter, −0.005 half).

### ECE, H
- **ECE:** < 0.15 at the endpoint. A break is handled in PREREG-p2_l's order.
- **H1–H10** as PREREG-p2_r, direction-free. **H7:** the batch mixes its sources in a third proportion, so the
  loss totals are not comparable with either p2_v line. **H2:** `skipped_steps` < 20. **H6:** book move-1 entropy,
  reported beside the half line's 1.61 and the quarter line's 1.23.
- **H12, memorisation (reported, no bound):** cross-entropy on the whole training ring minus that on the held-out
  5,000, at the endpoint; beside −0.021 (half) and −0.015 (quarter).
- **H13, the step (new, reported, no bound):** `grad_norm` before clipping, per run, beside p2_v's. `grad_clip` is
  1.0 and p2_v's norms ran 5–9, so every step is clipped: the update's size is set by the clip and the learning
  rate in all three lines, and the batch decides its direction.

## Predictions (written before launch)
- T1 holds: wide's endpoint at 2.27–2.33 on the held-out ring. H12 −0.01 to −0.04.
- K1: point −0.04 to +0.01; more likely than not the CI includes zero.
- K1b: point 0.00 to +0.05; more likely than not the CI includes zero.
- So the likeliest single outcome is the third row below, "not separated": the two reference endpoints are 0.044
  apart and each comparison resolves ±0.034. The round separates the suspects only if wide lands near one end.
- K2 passes (190–194). K3: 0020 below the start; the endpoint within noise of it.
- ECE 0.08–0.13; N 185–195; L2 0–5 per run; H6 between 1.23 and 1.61.

## Decision rules

| K1 (wide − quarter) | K1b (wide − half) | T1 | Decision |
|---|---|---|---|
| **CI entirely above zero** | any | holds | **More teacher helps once the self-play rows are kept.** The standing batch becomes 1,280 (717 / 51 / 512) with this schedule, if K2 passes; if K2 fails it stays as it is until the mate-in-2 cost is diagnosed |
| **CI includes zero** | **CI entirely above zero** | holds | **H_rows supported: the half line's cost was the self-play rows it gave up.** The extra teacher rows neither help nor cost; the standing batch stays 1,024 at 0.25, the cheaper of two equals. The "more of this teacher" lever is closed; p2_x's lever is Michael's choice: a deeper engine, the 2e-7 step-down, the public benchmark, the third line (RUNLOG-p2_t §6, reading 4) |
| CI includes zero | CI includes zero | holds | **Not separated.** Wide sits between the two lines and the round resolves neither difference. Read as consistent with a cost that grows with the teacher's share of the batch; not read as support for H_rows. The dose stays at 0.25 and the lever is closed as above |
| **CI entirely below zero** | any | holds | **The extra teacher rows cost skill by themselves**, with every self-play row kept: H_rows is refuted as the whole account. The dose stays at 0.25; the lever is closed as above |
| any | any | fails | **The lever was not applied in effect**: wide did not end closer to the teacher than quarter on held-out positions. H12 and H13 are read first; not read as a result about the rows |
| any other pair | | | Reported as measured, no decision; the dose stays at 0.25 |
| N < 183 or ECE ≥ 0.15 at the endpoint | | | The broken guard first |

## Measurement protocol

```bash
# The launch: runs/p2_w_launch.sh (cloud/launch_run.sh twice, as transient user units).
#   p2_w_wide      (p2ww, 8 h): train_phase2.py --config configs/p2_w_wide_hot.yaml <start> --out runs/p2_w_wide --iterations 20
#   <start> = --seed 43 --init runs/p2_u_stale_down/ckpt_iter0010.pt --seed-pool runs/p2_u_stale_down
#             --seed-replay runs/p2_u_stale_down
#   inputs: the start's 20 pool checkpoints, state.json, config.yaml, replay/, runs/p2_u_fresh/teacher/, the mate
#   data, phase1_anchor.pt
#   p2_w_wide_down (p2wwd, 6 h), after the hot stage's final pull (WAIT_FOR=runs/p2_w_wide/.remote_done):
#   configs/p2_w_wide_down.yaml from runs/p2_w_wide, inputs including runs/p2_u_fresh/teacher/

# After both: runs/p2_w_finish.sh under taz-finisher@p2_w — reps, each verdict against the start, T1, H12, then:
.venv/bin/python scripts/eval_reps.py verdict --a quarter runs/p2_v_quarter_down/sf5/reps_iter0010_r*.jsonl \
    --b wide runs/p2_w_wide_down/sf5/reps_iter0010_r*.jsonl \
    --out runs/p2_w_wide_down/sf5/p2w_verdict_wide_vs_quarter.json
.venv/bin/python scripts/eval_reps.py verdict --a half runs/p2_v_half_down/sf5/reps_iter0010_r*.jsonl \
    --b wide runs/p2_w_wide_down/sf5/reps_iter0010_r*.jsonl \
    --out runs/p2_w_wide_down/sf5/p2w_verdict_wide_vs_half.json
```

## What would make p2_w uninterpretable
- A difference between wide and the quarter line's configs beyond the five fields (the config test and L1).
- A change to the trainer or the instrument since `56328bb` (L4), or a reference rep file that is not p2_v's.
- A teacher ring other than the registered one (L3).
- A launch from anything but clean, pushed `main`.
- Losing a run's final pull: re-run by addendum, not read from what is left.

**What the design does not separate.** Wide holds the number of self-play rows, not their weight. The loss is a
mean over the batch, so the 717 self-play rows carry 0.560 of it in wide against 0.700 in quarter and 0.450 in
half; in teacher rows per self-play row wide (0.714) sits halfway between quarter (0.357) and half (1.111). Every
step is clipped to the same norm, so that ratio is what the batch contributes to the update. If the cost in p2_v
follows the ratio and not the row count, wide lands between the two lines, which is the third row. Holding the
weight as well would need per-source loss weights, a code change, and is not run here. The larger batch also
averages 25% more rows a step; no line isolates that.

**What the statistic does not cover.** As PREREG-p2_k: each line is one run; K1 and K1b resample openings. Two
further limits are particular to this round. The comparison lines were not run beside wide: their endpoints were
known when this was written (0.3779 and 0.3338), the quarter line's is the highest six-rep mean of p2_t–p2_v and
did not separate from the start (+0.017 [−0.017, +0.051]), and a second run of that line could land lower. K1 is
therefore a comparison against one realisation of the quarter line, and a wide endpoint between about 0.344 and
0.368 would include zero against both references. The reference reps were also played on 2026-11-11 and wide's
will be played after its runs; the instrument is the same code and binaries (L4), and no drift between days has
been seen on it, but none is measured here.

---

## Sign-off

Checked for the draft, 2026-11-11, on this box: `configs/p2_w_wide_{hot,down}.yaml` differ from
`configs/p2_u_fresh_{hot,down}.yaml` in the five registered fields only, and the three batch plans are
(717, 20, 31, 512), (717, 20, 31, 256) and (461, 20, 31, 512) (test passes). Seed-43 fingerprints:
`99a0b4ae74b99baa162ef054e716439a2841ef47e165b1a1949f91f259e3ee13` (wide hot) and
`b812a7b4aba54a112f9447001e10b51d3a83926867b2aaf1d2b22eb94a57c12c` (wide down). A zero-iteration run on CPU with the
hot config and the registered start (`--iterations 0`, into the session scratchpad): batch 1,280, rehearsal
0.03984375 (0.015625 + 0.02421875), pool seeded with all 20 snapshots, ring seeded with 500,000 rows from donor
head 1,037,734 at offset −10, `teacher_frac` 0.4 in its `config.yaml`, whose sha256 is the fingerprint above. The
start's league pool is complete (`ckpt_iter0000`–`0010` and the nine `seed_p2_u_stale_ckpt_iter0012`–`0020`
hard links p2_v's draft restored). `git diff 56328bb HEAD -- src scripts cloud` is empty. The twelve reference rep
files are tracked at `8c9f61d`. The `p2w_verdict_*` ignore negation is added and red-checked through `MUST_REACH`
(both verdict files fail without it). RunPod balance $33.68 (auto-reload at $10), no pods of ours; secure RTX 4090
at $0.89/h; each pod's worst case passes the $15 ceiling, and the round's $12.46 leaves $21.22. Disk 98 GB free;
the round adds about 3.3 GB.

**Not signed.** Registration is a later commit that changes only this file, at Michael's instruction.
