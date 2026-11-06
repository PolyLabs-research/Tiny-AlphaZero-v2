# RUNLOG — p2_r (self-play at 128 simulations against 48, twenty iterations each from p2_q/0040, judged on skill 5)

**Runs:** `runs/p2_r_cont` (`--config configs/p2_o_lr.yaml --seed 43`: 48 simulations per self-play move) and
`runs/p2_r_sims` (`--config configs/p2_r_sims.yaml --seed 43`: the same with `selfplay.sims_per_move: 128`),
both at lr 2e-6, 20 iterations each, on two RunPod pods — `5qpjp55lbsjxtg` (cont) and `n79z3ylig26ek0`
(sims), each a secure RTX 4090 of the 24 GB variant (`vram_gb` 23.52) — from 2026-11-05 10:43:39 to 14:11:50
(cont) and 10:32:40 to 19:54:45 (sims) UTC, code `6876f04`, both warm-started from `p2_q/0040` with the league
and the ring continued from `runs/p2_q`. Every measurement ran on this box (torch 2.14.1+rocm7.2), by
`cloud/remote_campaign.sh` with `MEASURE_AFTER_PULL=1` (one driver per arm) and `runs/p2_r_finish.sh`.
**Pre-registration:** `PREREG-p2_r.md` (`6876f04`), no addenda. Every verdict below was registered before
it was measured, except where marked.

**Verdict: H_sims NOT SUPPORTED — the registered "no effect seen at this length" row: 48 stays, the plateau
stands, p2_s's lever is Michael's choice. K1 (sims − cont at 20, skill 5, six reps per arm): −0.020 [−0.055,
+0.015]. K2 passes (189 and 189). The lever was applied exactly (128.0 simulations per position against
48.0). A weak test, by this round's design: the arm had about ten iterations on a ring full of the new
targets, at a rate that barely moves the weights (§6). This box crashed during the round; nothing was
lost (§5). ECE and N hold; L1 holds; L2 reported (2 and 2 rows).**

---

## 1. Before launch

The start reference is p2_q/0040's: six skill-5 book reps (mean 0.3146), three skill-3 reps (0.7608), three
skill-0 reps and three initial-position reps; nothing was re-measured. Read at registration, 2026-11-05
10:04 UTC: balance $17.18, no pods of ours, secure RTX 4090 at $0.89/h (stock "Medium"); `make test` 993
passed, 2 skipped.

**The pods:** the sims arm got its pod on the second try (10:11:40; 64 vCPU, 54-CPU quota, 14 h deadline).
The cont arm's first pod (`cx5xfhwj1o1g8x`, 10:07:14) **never came up**: 27 minutes after launch RunPod showed
no runtime, no public IP and no port mappings, and `wait-ssh` had timed out twice. Its pipeline was stopped
by PID, the pod terminated (about $0.40, no data), and the same registered launch rerun: `5qpjp55lbsjxtg` at
10:35:50 (24 vCPU, 20-CPU quota, 8 h deadline). Every input's sha256 matched on each pod.

## 2. L — the continuity applied

- **L1 holds, both arms:** `init_from` `runs/p2_q/ckpt_iter0040.pt`; `seed` 43; `seed_replay.rows` 500,000,
  `donor_committed` = `donor_head` = 2,620,249, `iteration_offset` −40; `git_dirty` false; `git_sha`
  `6876f04`. `config_fingerprint` `86fcc672…7d65` (cont) and `1c284171…6a08` (sims), each equal to its
  `config.yaml`'s sha256; the configs differ in `sims_per_move` only, with `eval_match_sims` 48 in both; `lr`
  2e-6 on every row.
- **The lever applied:** simulations per stored position **128.0** in the sims arm, **48.0** in cont, over all
  twenty iterations.
- **L2 (reported, no bound):** cont 2 (`open_001` at iteration 11, `open_028` at 17), sims 2 (`open_021` at 20,
  `open_001` at 11) in 500,000.

## 3. K — 128 against 48, at skill 5

### K1 (primary) — sims against cont at 20, six reps per arm

Per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000
resamples, seed 0; `runs/p2_r_sims/sf5/p2r_verdict_sims_vs_cont.json`, reproduced byte for byte; all rows
`stockfish_s5`, each arm on its own checkpoint):

| arm | reps r1–r6 | mean | W / D / L (1,200 games) |
|---|---|---|---|
| `p2_r_cont` (48) | 0.3225, 0.39, 0.30, 0.3375, 0.3425, 0.355 | 0.341 | 262 / 295 / 643 |
| `p2_r_sims` (128) | 0.345, 0.33, 0.32, 0.2625, 0.3575, 0.3125 | 0.321 | 241 / 289 / 670 |
| **sims − cont** | | **−0.020** | **[−0.055, +0.015]**, SE 0.0178 |

- **The interval includes zero: "no effect seen at this length".** Forecast +0.02 to +0.08: failed, the point
  on the other side. Per opening, sims' rep-mean is above cont's on 34 of the 100, below on 54, equal on 12
  (median −0.042).

### K2 — mate-in-2

At 20: **cont 189/200, sims 189/200**; sims ≥ cont − 4 and both ≥ 183. **K2 passes.** The eleven failures at 20
are the same eleven positions in both arms. At 10: 190 and 188.

### K3 — each arm against the start, and the last rung (informational)

| arm / checkpoint | reps | vs start [95% CI] |
|---|---|---|
| cont 0010, skill 5 | 0.305, 0.34, 0.335 | +0.012 [−0.028, +0.053] |
| **cont 0020, skill 5 (six)** | | **+0.027 [−0.006, +0.060]** |
| sims 0010, skill 5 | 0.365, 0.325, 0.3125 | +0.020 [−0.018, +0.058] |
| **sims 0020, skill 5 (six)** | | **+0.007 [−0.028, +0.040]** |
| cont 0020, skill 3 | 0.795, 0.74, 0.785 | +0.013 [−0.027, +0.052] |
| sims 0020, skill 3 | 0.8125, 0.78, 0.8075 | +0.039 [+0.003, +0.076] |

The continuation stayed within noise of the start (predicted). The sims arm did too, not above it as
predicted. At skill 3 (three reps a side, informational) the lean runs the other way; no row turns on it.

### K4 — skill 0 and the initial position (continuity)

Skill-0 reps against p2_q/0040's three: cont +0.008, −0.001; sims +0.002, +0.007. M4 at skill 0: cont 0.968,
sims 0.975, met. Initial position: cont +0.003 [−0.013, +0.018], sims +0.006 [−0.012, +0.023].

## 4. N, ECE, H

- **N — holds:** mate-in-2 cont 190, 189; sims 188, 189. mate-in-3 cont 151, 154; sims 149, 151.
- **ECE — holds:** cont 0.0942, 0.1002; sims 0.0951, 0.1015 (< 0.15; predicted 0.08–0.12).
- **H7 / H10 — what the lever did to training.** Policy loss: cont flat at 1.396; **sims 1.413 at iteration 1,
  rising to 1.527 by about iteration 10, then flat** (first five 1.442, last five 1.527). Predicted higher
  than cont: held. Value loss cont 0.391 → 0.389, sims 0.390 → 0.385. Promotion loss by halves cont 0.447 /
  0.440, sims 0.442 / 0.418. Decisive rate cont 0.805 / 0.811, sims 0.788 / 0.790; self-play games 102.5 and
  105.8 plies; book entropy (H6) cont 0.601, 0.586, sims 0.689, 0.697.
- **Pace (H10):** cont 10.40 min per iteration (self-play 9.5); sims **28.10** (self-play 27.2; predicted
  24–30), 2.7 times the search for 2.7 times the wall-clock. Sims/s 4,261–4,548 and 4,080–4,476.
- **H1:** `move1_prior_entropy` cont 0.14 → 0.08, sims 0.14 → 0.09.
- **H2:** 0 skipped steps in 8,000, both arms.

Rungs: random, greedy and phase1 won 200/200 at every checkpoint in both arms.

## 5. Mechanics

- **The crash, 2026-11-05 about 11:59 UTC.** This box went down silently: the journal ends at 11:59:06 with no
  shutdown sequence, `wtmp` says crash, there is no panic dump, and it came back at 12:10 on kernel 7.0.0-42
  (from -38). Chess was on RunPod, the box only pulling; the load (17–19) was protocell's stage 4d, 24 all-core
  workers (that project's record). **Both pods trained on untouched.** The drivers and the finisher died
  with the box, the failure that lost p2_j's first attempt. They were found at 12:20 and restarted by hand
  with their original settings; pulls resumed and nothing was lost. The GPU, torch 2.14.1+rocm7.2, Stockfish and
  the lease were verified on the new kernel. **Every p2_r measurement ran on kernel -42**, so K1 and K2 compare
  like with like; K3's start reference was measured on -38. Measurement fixes nodes and simulations, not
  time.
- **Drivers made reboot-proof, during the round** (`7d76e15`): at Michael's instruction both drivers and the
  finisher moved under systemd user units (`taz-campaign@`, `taz-finisher@`; `Restart=on-failure`, Linger),
  stopped and restarted by PID at 12:28 while they were only pulling. `remote_campaign.sh` itself did not
  change. A `kill -9` of a live driver came back in 60 s, re-armed. A real reboot has not been tested.
  A sensor trace (`sensorlog.service`, one synced line a minute) has run on this box since 12:26.
- **Lifecycle:** final pulls with the ring at 14:14:18 (cont) and 19:55:02 (sims), `PULLED` written, both pods
  deleted themselves. Measurements under `gpu hold`: cont 0010 14:14–14:42, 0020 –15:25; sims 0010 19:55–20:24,
  0020 –21:08; the finisher's skill-5 reps cont 21:09–22:33, sims 22:33–23:55; skill-3 reps cont –00:17, sims
  –00:39; K1 at 00:38:50 UTC on 2026-11-06. No retries.
- **Cost:** ≈ **$12.34** at $0.89/h (the dead pod $0.42, cont 3.65 h $3.25, sims 9.74 h $8.67); expected about
  $12.5, worst case $19.58. All RunPod use by this project to date about $106.

## 6. Decision

Per the table's third row (K1 includes zero): **no effect seen at this length. 48 stays; the plateau stands;
p2_s's lever is Michael's choice.** N and ECE hold in both arms.

**Why this reading is narrow.** The sims arm's policy loss climbed for about ten iterations as the ring (about
ten iterations long) filled with 128-simulation targets, then held at 1.527: the targets did change, and the
network did not catch up. That left about ten iterations, 4,000 steps, on a ring full of the new targets, at
2e-6, where the weights barely move. PREREG-p2_r did not name that; it should have. The round rules out a quick
gain from more search at 2e-6; it does not show that better targets carry nothing.

The options put to Michael (2026-11-06), the network fixed at 6.5M parameters: the same lever at a rate that can
use it, then a step down; the 2e-7 step-down; distillation from Stockfish evaluations; pausing to build the
UCI wrapper for a public number. **He chose the first:** 128 simulations at 2e-5, then a step down to 2e-6,
against the standing protocol. PREREG-p2_s.

Readings the table does not cover:

1. **Every gain so far came from a step-down after time at a higher rate** (2e-4 → 2e-5 in p2_k, 2e-5 → 2e-6 in
   p2_o and p2_p). The fair test of better targets has that shape too.
2. **Both arms stayed within noise of the start again.** The plateau at 2e-6 holds a second round.

The open list, carried forward: p2_s; the calibration sample; the promotion loss at 2e-6; L2's exclusion at
storage time; the host crashes (the sensor trace is running); the launch scripts, lost with `/tmp` in the crash,
to be committed to `cloud/`; a public benchmark (a UCI wrapper first).

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_r_{cont,sims}/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the runs |
| `runs/p2_r_{cont,sims}/ladder.jsonl` | ladder passes at 10 and 20 (ECE, N) |
| `runs/p2_r_{cont,sims}/sf5/` | skill-5 reps and each checkpoint's verdict against the start (K3) |
| `runs/p2_r_{cont,sims}/sf3/` | skill-3 reps at 20 and their verdict |
| `runs/p2_r_sims/sf5/p2r_verdict_sims_vs_cont.json` | K1 |
| `runs/p2_r_{cont,sims}/reps_iter00{10,20}_r*.jsonl`, `p2_verdict_reps*.json`, `p2_m4_iter0020.json`, `startpos/` | K4, skill 0 |
| `runs/p2_r_{cont,sims}/book_rows.json` | L2 |
| `runs/p2_r_{cont,sims}_campaign.console.log`, `runs/p2_r_{cont,sims}.console.log`, `runs/p2_r_finish.console.log`, `runs/p2_r_finish.sh` | consoles and the finisher (untracked) |
