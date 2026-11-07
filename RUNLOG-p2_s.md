# RUNLOG — p2_s (128 self-play simulations at 2e-5, then a step down to 2e-6, against the standing protocol; judged on skill 5)

**Runs:** line A: `runs/p2_s_sims` (`--config configs/p2_s_hot.yaml --seed 43`: 128 simulations, lr 2e-5, 20
iterations from `p2_r_sims/0020`), then `runs/p2_s_down` (`--config configs/p2_r_sims.yaml --seed 43`: 128, 2e-6,
10 iterations from `p2_s_sims/0020`); line B: `runs/p2_s_cont` (`--config configs/p2_o_lr.yaml --seed 43`: 48,
2e-6, 30 iterations from `p2_r_cont/0020`). Three RunPod pods, each a secure RTX 4090 of the 24 GB variant
(`vram_gb` 23.52): `430qqkz1qs7jib` (sims, 64 vCPU), `7g0u7n4qzb9q75` (down, 64 vCPU), `tupwe663ilibg0` (cont,
28 vCPU). Training from 2026-11-06 15:15:10 to 2026-11-07 01:13:44 (sims), 01:21:40 to 07:05:55 (down) and
2026-11-06 15:02:49 to 19:27:04 (cont) UTC, code `405f2dd`, each line continuing its own p2_r arm with the
league and the ring. Every measurement ran on this box (torch 2.14.1+rocm7.2) under the user units
`taz-campaign@p2_s_{sims,down,cont}` (`MEASURE_AFTER_PULL=1`) and `taz-finisher@p2_s` (`runs/p2_s_finish.sh`).
**Pre-registration:** `PREREG-p2_s.md` (`405f2dd`), no addenda. Every verdict below was registered before it
was measured, except where marked.

**Verdict: the registered "CI includes zero" row — no gain from more search at this budget. 48 stays, 128
simulations is closed as a lever, p2_t's lever is Michael's choice. K1 (down/0010 − cont/0030, skill 5, six
reps per side): −0.005 [−0.033, +0.023]. K2 passes (193 and 189). The lever was applied exactly (128.0
simulations per position in both line-A runs, 48.0 in line B; lr 2e-5 then 2e-6). The 2e-5 stage lost a little
at skill 5 and at skill 0, and the step down won it back and no more (§3, §6). ECE and N hold; L1 holds in all
three runs; L2 reported (3, 1 and 3 rows).**

---

## 1. Before launch

The start references are each line's p2_r arm at 20: six skill-5 book reps (`p2_r_sims` mean 0.3212,
`p2_r_cont` 0.3412), three skill-3 reps (0.8000, 0.7733), three skill-0 book reps and three initial-position
reps; nothing was re-measured. Read at registration, 2026-11-06 14:54 UTC: balance $37.22, no pods of ours,
secure RTX 4090 at $0.89/h; `make test` 1,018 passed, 2 skipped.

**The pods.** All three came from `cloud/launch_run.sh` (`45915fd`), its first use on real pods, each a
transient unit: cont launched at 14:55:05; sims was refused once (14:55:04) and launched at 15:00:09; down
waited for `runs/p2_s_sims/.remote_done` and launched at 01:14:06, 22 s after sims's final pull. No pod was
unusable, none replaced. Every input's sha256 matched on each pod (50 files each).

## 2. L — the continuity applied

- **L1 holds, all three runs:** `seed` 43; `git_sha` `405f2dd`, `git_dirty` false; each `config_fingerprint`
  equal to its `config.yaml`'s sha256: `75bbaeb5…13eb` (sims), `1c284171…6a08` (down), `86fcc672…7d65` (cont).
  - sims: `init_from` `runs/p2_r_sims/ckpt_iter0020.pt`; `seed_replay` 500,000 rows, `donor_committed` =
    `donor_head` = 1,582,883, offset −20; `lr` 2e-5 on every row.
  - down: `init_from` `runs/p2_s_sims/ckpt_iter0020.pt`; `seed_replay` 500,000, committed = head = 1,561,484 (sims's
    final ring), offset −20; `lr` 2e-6 on every row.
  - cont: `init_from` `runs/p2_r_cont/ckpt_iter0020.pt`; `seed_replay` 500,000, committed = head = 1,549,359,
    offset −20; `lr` 2e-6 on every row.
- **The lever applied:** simulations per stored position **128.0** in sims and down, **48.0** in cont, every
  iteration. 20 + 10 iterations (12,000 steps) in line A, 30 (12,000 steps) in line B.
- **L2 (reported, no bound):** sims 3 (`open_021` at 20, `open_001` at 11, `open_028` at 17), down 1 (`open_012`
  at 5), cont 3 (`open_027` at 25, 28 and 29) in 500,000. Predicted 0–5 per run: held.

## 3. K — the two lines at the end, skill 5

### K1 (primary) — down/0010 against cont/0030, six reps per side

Per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000
resamples, seed 0; `runs/p2_s_down/sf5/p2s_verdict_down_vs_cont.json`, reproduced byte for byte; all twelve rows
`stockfish_s5`, each side on its own checkpoint):

| line | reps r1–r6 | mean | W / D / L (1,200 games) |
|---|---|---|---|
| B, `p2_s_cont/0030` (48) | 0.32, 0.375, 0.335, 0.3475, 0.31, 0.2925 | 0.330 | 253 / 286 / 661 |
| A, `p2_s_down/0010` (128) | 0.3075, 0.3425, 0.295, 0.375, 0.3175, 0.31 | 0.325 | 247 / 285 / 668 |
| **A − B** | | **−0.005** | **[−0.033, +0.023]**, SE 0.0143 |

- **The interval includes zero: "no gain from more search at this budget".** Forecast +0.02 to +0.08: failed.
  Per opening, A's rep-mean is above B's on 44 of the 100, below on 45, equal on 11 (median 0.000). The draw
  rates are the same (0.237 and 0.238).

### K2 — mate-in-2

At the endpoints: **down/0010 193/200, cont/0030 189/200**; 193 ≥ 189 − 4 and both ≥ 183. **K2 passes.** At the
other measures: sims/0020 189, down/0005 191, cont/0020 188. Seven positions failed at every one of the five
measures, in both lines: `mate2_091`, `mate2_124`–`127`, `mate2_129` and `mate2_196`, which are down/0010's seven
failures exactly; `mate2_128` failed at three of the five.

### K3 — each line against its own start, the step, and skill 3 (informational)

| checkpoint | reps | vs start [95% CI] |
|---|---|---|
| sims 0020, skill 5 (end of 2e-5) | 0.2825, 0.2925, 0.295 | −0.031 [−0.072, +0.010] |
| down 0005, skill 5 | 0.3425, 0.3575, 0.3275 | +0.021 [−0.020, +0.064] |
| **down 0010, skill 5 (six)** | | **+0.003 [−0.028, +0.035]** |
| cont 0020, skill 5 | 0.3875, 0.345, 0.37 | +0.026 [−0.013, +0.065] |
| **cont 0030, skill 5 (six)** | | **−0.011 [−0.045, +0.023]** |
| **the step: down 0010 (six) − sims 0020 (three)** | | **+0.035 [−0.005, +0.074]** |
| down 0010, skill 3 | 0.775, 0.7825, 0.7825 | −0.020 [−0.060, +0.021] |
| cont 0030, skill 3 | 0.77, 0.7975, 0.7675 | +0.005 [−0.033, +0.043] |

Predicted: sims/0020 near or below its start (held, −0.031, within noise); the step down gaining by down/0005
(the point +0.021 against the start, within noise); cont within noise of its start (held). The step's own size
is the largest positive point of the round and still touches zero. At skill 3 both endpoints sit at 0.78.

### K4 — skill 0 and the initial position (continuity)

Skill-0 book reps against each line's start: **sims/0020 −0.022 [−0.042, −0.003]** and initial position **−0.023
[−0.042, −0.006]**, both below zero (informational; no row turns on them); down/0005 −0.000, down/0010 −0.003
[−0.018, +0.012], initial position −0.001; cont/0020 +0.004, cont/0030 +0.001, initial position −0.004. M4 at
skill 0: down 0.972, cont 0.968, met (bar 0.675).

## 4. N, ECE, H

- **N — holds:** mate-in-2 sims 189; down 191, 193; cont 188, 189. Mate-in-3 sims 153; down 156, 158; cont 157,
  156. Predicted 185–195: held.
- **ECE — holds:** sims 0.0970; down 0.0859, 0.0939; cont 0.0938, 0.0972 (< 0.15; predicted 0.08–0.13).
- **H7 — the reading behind this round.** At 2e-5 the policy loss **fell from 1.525 to 1.505 by iteration 7,
  then rose to 1.556 by 20** (first five 1.517, last five 1.550). Predicted to fall: it fell for seven
  iterations, then gave the fall back and more. In the step down it held at 1.555–1.573. Line B: 1.395 → 1.385.
  The 128-simulation targets are a different distribution from the 48-simulation ones, so the two lines' losses
  are not comparable to each other.
- **The value loss** fell at 2e-5 from 0.384 to 0.325 (flat from iteration 11), then **rose through the step down
  every iteration from 0.335 to 0.399** (0.397 at 10). Line B held at 0.384–0.394.
- **H10, pace:** sims 29.71 min per iteration (self-play 28.79), down 33.87 (32.80), cont 8.75 (7.96).
  Sims/s 2,958–4,208 (sims), 2,698–4,013 (down), 5,038–5,522 (cont).
- **H1:** `move1_prior_entropy` sims 0.09 → 0.12, **down 0.23 → 0.56**, cont 0.08 → 0.05. Book entropy (H6) sims
  0.656; down 0.656, 0.640; cont 0.579, 0.585.
- Promotion loss by halves: sims 0.346 / 0.353, down 0.404 / 0.476, cont 0.425 / 0.430. Decisive rate sims
  0.794 / 0.785, down 0.771 / 0.780, cont 0.807 / 0.822; self-play games 103.7, 106.3 and 102.2 plies.
- **H2:** 0 skipped steps in 8,000 (sims), 4,000 (down) and 12,000 (cont).

Rungs: random, greedy and phase1 won 200/200 at every checkpoint in all three runs.

## 5. Mechanics

- **Unattended end to end.** `launch_run.sh` launched, staged, verified and started all three pods and enabled
  their drivers; the step down launched itself after line A's first stage. Nothing was restarted by hand, no
  measurement was retried, and this box stayed up throughout.
- **Lifecycle:** final pulls with the ring at 19:27:04 (cont, 11-06), 01:13:44 (sims) and 07:05:55 (down); each
  pod deleted itself after `PULLED`. Measurements under `gpu hold`: cont 0020 19:27–19:55, 0030 –20:39; sims 0020
  01:13–01:59; down 0005 07:05–07:34, 0010 –08:17; the finisher from 08:20: skill-5 reps sims/0020, down/0005,
  cont/0020, down/0010 (six), cont/0030 (six), skill-3 reps down/0010 then cont/0030; K1 at 12:12:46 UTC on
  2026-11-07.
- **Cost:** pod time at $0.89/h ≈ **$18.35** (cont 4.53 h $4.03, sims 10.23 h $9.10, down 5.86 h $5.22); the
  balance went from $37.22 to $18.69 ($18.53). Expected about $18.5, worst case $26.70. All RunPod use by this
  project to date about $124.

## 6. Decision

Per the table's third row (K1 includes zero): **no gain from more search at this budget. 48 stays, and 128
simulations is closed as a lever. p2_t's lever is Michael's choice.** N and ECE hold in all three runs.

**This was the fair test that p2_r was not.** p2_r gave 128 simulations about ten iterations on its own targets at a
rate that barely moves the weights. p2_s gave them twenty iterations at 2e-5 and then the step down that every gain
so far has followed. The result is the same.

**Michael's choice (2026-11-07): distillation from Stockfish evaluations as p2_t's lever, and an analysis of
where the points are lost, run on this box alongside it.** The analysis is a diagnostic with its own record. It
came from a report Michael raised of LLM-written weight edits ("weight-space programming"): in its write-ups the
edits fixed single localized faults and failed where there was none. The analysis asks whether the losses at
skill 5 have one recurring cause, which an edit could address, or many. PREREG-p2_t follows.

Readings the table does not cover:

1. **The 2e-5 stage cost a little and the step down returned it.** At skill 5 −0.031 then the step +0.035; at
   skill 0 −0.022 [−0.042, −0.003] and at the initial position −0.023, both below zero, then back to the start
   after the step (−0.003, −0.001). The pattern of p2_k and p2_o (gains after a step down) held in shape. The
   step this time recovered a loss; it did not add a gain.
2. **The value loss the 2e-5 stage gained did not carry.** 0.325 at the end of 2e-5, back to 0.397 by the end of
   the step down, the 48-simulation line's level. The rise was steady, iteration by iteration, as the ring turned
   over. That fits a network that fitted the ring at 2e-5 rather than learning something that held. It is a
   reading, not a test.
3. **Skill 5 has held at 0.32–0.34 for three rounds:** p2_q/0040 0.315, the p2_r arms 0.321 and 0.341, the p2_s
   endpoints 0.325 and 0.330. Six-rep comparisons put a 95% interval of about ±0.03 around a difference. A lever
   worth less than that cannot be seen at this rung with this many games.
4. **The mate-in-2 failures are a fixed set.** `mate2_091`, `mate2_124`–`127`, `mate2_129` and `mate2_196` fail at
   every measure in both lines, as p2_r's eleven did in both arms. Whether they share one pattern is a question for
   the error analysis.

The open list, carried forward: the 2e-7 step-down; the calibration sample; the promotion loss at 2e-6; L2's
exclusion at storage time; the host crashes (the sensor trace is running, no crash since 11-05); a public
benchmark (a UCI wrapper first).

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_s_{sims,down,cont}/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the runs |
| `runs/p2_s_{sims,down,cont}/ladder.jsonl` | ladder passes at sims 20, down 5 and 10, cont 20 and 30 (ECE, N) |
| `runs/p2_s_{sims,down,cont}/sf5/` | skill-5 reps and each checkpoint's verdict against its start (K3) |
| `runs/p2_s_down/sf5/p2_verdict_reps_step.json` | the step's own size |
| `runs/p2_s_down/sf5/p2s_verdict_down_vs_cont.json` | K1 |
| `runs/p2_s_{down,cont}/sf3/` | skill-3 reps at the endpoints and their verdicts |
| `runs/p2_s_*/reps_iter*_r*.jsonl`, `p2_verdict_reps*.json`, `p2_m4_iter*.json`, `startpos/` | K4, skill 0 |
| `runs/p2_s_{sims,down,cont}/book_rows.json` | L2 |
| `runs/p2_s_*_launch.console.log`, `runs/p2_s_*_campaign.console.log`, `runs/p2_s_finish.console.log`, `runs/p2_s_finish.sh` | consoles and the finisher (untracked) |
