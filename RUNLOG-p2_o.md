# RUNLOG — p2_o (a second tenfold step-down, 2e-6, against a continuation arm, from p2_n/0040, judged on skill 3)

**Runs:** `runs/p2_o_cont` (`configs/p2_k_lr.yaml`, 2e-5) and `runs/p2_o_lr` (`configs/p2_o_lr.yaml`: the
same with `train.lr: 2.0e-06`), 20 iterations each, trained in parallel on two RunPod pods —
`khq0zqptj2ocli` (cont) and `toy9q551txojrs` (lr), each a secure RTX 4090 of the 24 GB variant (`vram_gb`
23.52) — from 2026-11-02 11:27:27 to 15:00:27 (cont) and 11:00:36 to 15:02:50 (lr) UTC, code `fe2f784`,
both warm-started from `p2_n/0040` with the league and the replay ring continued from `runs/p2_n`. Every
measurement ran on this box (torch 2.14.1+rocm7.2), by `cloud/remote_campaign.sh` with
`MEASURE_AFTER_PULL=1` (one driver per arm) and `runs/p2_o_finish.sh`.
**Pre-registration:** `PREREG-p2_o.md` (`fe2f784`), no addenda. Every verdict below was registered before
it was measured, except where marked.

**Verdict: H_lr2 NOT SUPPORTED, on the boundary — the registered "no effect seen at this length" row, by
Michael's ruling (2026-11-03). K1 (lr − cont at 20, skill 3, six reps per arm): +0.031, interval [0,
+0.0625] — its lower end is exactly zero. The verdict file reads `excludes_zero: true` only because
floating point rendered that zero as 5.55e-18 (§3). Every other reading favours the step-down (each arm
against the start: lr +0.0825, cont +0.051; the initial position lr +0.021), but none is the registered
test. K2 passes (190 against 192). ECE holds. 2e-5 stays.**

---

## 1. Before launch

The start reference at skill 3 is p2_n/0040's six book reps (0.6375, 0.6125, 0.6525, 0.6725, 0.6225,
0.6675; mean 0.6442); at skill 0 its three book reps and three initial-position reps. Nothing was
re-measured. Read at registration, 2026-11-02 10:13 UTC: balance $21.99 (auto-reload at $10; launch margin
$2), no pods of ours, secure RTX 4090 at $0.89/h with stock listed "None"; `make test` 934 passed, 2
skipped.

**The pods:** both launches were refused for want of instances from 10:14 and retried every five minutes
by the registered command. The lr arm got its pod at 10:50:26 (64 vCPU, 54-CPU quota, 62 GB RAM), the
cont arm at 11:20:40 (36 vCPU, 35-CPU quota, 143 GB RAM). Both staged without a retry; all 50 inputs'
sha256 matched on each pod; jobs started 11:00:36 (lr) and 11:27:27 (cont). A third `taz-` pod on the
account (`taz-protocell-s3b-pilot`, launched 12:36 by another session) is not this round's and was left
alone.

## 2. L — the continuity applied

- **L1 holds, both arms:** `init_from` `runs/p2_n/ckpt_iter0040.pt`; `seed_replay.rows` 500,000,
  `donor_committed` = `donor_head` = 2,649,005, `iteration_offset` −40; `git_dirty` false; `git_sha`
  `fe2f784`. `p2_o_cont`'s fingerprint `8ab654e2…298c` = p2_k_lr's, `lr` 2e-5 on all rows; `p2_o_lr`'s
  `cdf5eb97…1a7a`, `lr` 2e-6 on all rows; the two pulled `config.yaml` files differ in `train.lr` only.
- **L2 (reported, no bound):** cont 3 (`open_021` at 11, `open_027` at 18, `open_062` at 19), lr 4
  (`open_021` at 11, `open_076` at 15, `open_027` at 18, `open_062` at 19) in 500,000.

## 3. K — the step-down against the continuation, at skill 3

### K1 (primary) — lr against cont at 20, six reps per arm

Per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000
resamples, seed 0; `runs/p2_o_lr/sf3/p2o_verdict_lr_vs_cont.json`, reproduced byte for byte; all rows
`stockfish_s3`, each arm on its own checkpoint):

| arm | reps r1–r6 | mean | W / D / L (1,200 games) |
|---|---|---|---|
| `p2_o_cont` (2e-5) | 0.725, 0.665, 0.7125, 0.70, 0.7025, 0.6675 | 0.695 | 748 / 173 / 279 |
| `p2_o_lr` (2e-6) | 0.785, 0.7325, 0.665, 0.7525, 0.69, 0.735 | 0.727 | 778 / 188 / 234 |
| **lr − cont** | | **+0.031** | **[0, +0.0625]**, SE 0.0159 |

- **The lower end is exactly zero.** The file records `ci95` [5.551115123125783e-18, 0.0625] and
  `excludes_zero: true`. Recomputed in exact rational arithmetic over the same bootstrap draws
  (`default_rng(0)`, the same 10,000 × 100 indices): pair scores lie on {0, ¼, ½, ¾, 1}, so every
  bootstrap mean is a rational number; 241 of the 10,000 are below zero and 13 are exactly zero, and the
  2.5th percentile interpolates between ranks 249 and 250, **both exactly 0**. The interval is [0, 1/16];
  5.55e-18 is the floating-point rendering of that zero, and `excludes_zero` (`lo > 0.0`) read the
  rounding. The point is exactly 1/32.
- **Michael's ruling (2026-11-03):** the registered row requires the interval *entirely above zero*; its
  lower end is zero, not above it, so the interval includes zero and the row is "no effect seen at this
  length". The alternative reading — the verdict file's flag, "second step-down supported" — was put to
  him with this record and not taken.
- Forecast point +0.02 to +0.08: met (+0.031). Per opening, lr's rep-mean is above cont's on 47 of the
  100, below on 39, equal on 14 (median 0.000).

### K2 — mate-in-2

At 20: **cont 192/200, lr 190/200**; lr ≥ cont − 4 and both ≥ 183. **K2 passes.** At 10: 191 and 192.
Failures at 20: cont 8 (`mate2_054`, `091`, `124`, `125`, `126`, `129`, `139`, `196`), lr 10 (`054`, `091`,
`124`, `125`, `126`, `127`, `129`, `140`, `141`, `196`).

### K3 — each arm against the start (p2_n/0040, 0.644), informational

| arm / checkpoint | reps | vs start [95% CI] |
|---|---|---|
| cont 0010 | 0.70, 0.66, 0.705 | +0.044 [+0.004, +0.083] |
| cont 0020 (r1–r3) | 0.725, 0.665, 0.7125 | +0.057 [+0.017, +0.097] |
| **cont 0020 (six)** | | **+0.051 [+0.018, +0.084]** |
| lr 0010 | 0.71, 0.7275, 0.77 | +0.092 [+0.050, +0.133] |
| lr 0020 (r1–r3) | 0.785, 0.7325, 0.665 | +0.083 [+0.045, +0.121] |
| **lr 0020 (six)** | | **+0.0825 [+0.051, +0.113]** |

- **Both arms rose over the start.** The prediction (cont within noise of it, the plateau holding a third
  time) **fails**: twenty more iterations at 2e-5 from p2_n/0040 read +0.051, where p2_n's own forty read
  +0.013 against *its* start. The step-down arm reads above the continuation at every checkpoint (+0.048
  at 10 on three reps, +0.031 at 20 on six) and took its gain by iteration 10, as p2_k_lr did.

### K4 — skill 0 and the initial position (continuity)

Skill-0 reps against p2_n/0040's three: cont −0.004, −0.008; lr +0.003, +0.006. M4 at skill 0: cont
0.957, lr 0.970, both met. **Initial position:** cont −0.008 [−0.028, +0.014], **lr +0.021 [+0.003,
+0.040]**. The ladder's single skill-0 replay (*not registered*): cont 0.955, 0.97; lr 0.9575, 0.9725.

## 4. N, ECE, H

- **N — holds, both arms:** mate-in-2 cont 191, 192; lr 192, 190 (≥ 183; predicted 185–195). mate-in-3
  cont 151, 152; **lr 162** (the highest on record from p2_i on), 154. Watchlist: `mate2_165` and `166`
  solved throughout in both arms, `129` unsolved.
- **ECE — holds, both arms:** cont 0.0942, 0.1099; lr 0.0930, 0.0983 (< 0.15; predicted 0.08–0.12).
- **H7 — the step-down arm's losses moved, they did not flatten.** By halves, cont value 0.3228 / 0.3243,
  policy 1.5105 / 1.5014; **lr value 0.3770 / 0.3965 (rising), policy 1.5115 / 1.4600 (falling faster)**;
  first five iterations against the last five, lr value 0.358 → 0.395, policy 1.516 → 1.448. The
  prediction (lr's losses flatter) **fails**. The value loss rising after a step-down is what p2_k_lr
  showed (0.319 → 0.357) when it won.
- **H1 (direction-free):** `move1_prior_entropy` recovered a little in both arms from p2_n's 0.06: cont
  0.06 → 0.15, lr 0.06 → 0.12.
- **H2:** 0 skipped steps in 8,000, both arms.
- **H6** (`book_move1_entropy`): cont 0.598, 0.609; lr 0.580, 0.605.

Rungs: random, greedy and phase1 won 200/200 at every checkpoint in both arms.

## 5. Mechanics

- **Pace:** cont 10.65 min per iteration (10.24–11.16), 4,222–4,543 sims/s (mean 4,385); lr 12.11
  (11.40–14.18), 3,243–4,113 sims/s (mean 3,900). As in p2_m, the 54-CPU pod was the slower one.
- **Lifecycle:** final pulls with the ring at 15:02:33 (cont) and 15:07:08 (lr), `PULLED` written, both
  pods deleted themselves. Measurements under `gpu hold`, one arm at a time: cont 0010 15:03–15:31, 0020
  –16:16; lr 0010 16:16–16:44, 0020 –17:27; the finisher's skill-3 reps cont 0010 17:30–17:53, 0020
  –18:40, lr 0010 18:40–19:02, 0020 –19:53; K1 at 19:53:14 UTC. No retries.
- **Cost:** ≈ **$7.11** at the pods' $0.89/h (lr 10:50–15:07, cont 11:21–15:03); expected about $7.7,
  worst case $14.24. All RunPod use by this project to date about $79.

## 6. Decision

Per Michael's ruling, the table's third row (K1 includes zero, at its boundary): **no effect seen at this
length. 2e-5 stays; the plateau stands as the record's standing reading. p2_p's lever is Michael's
choice.** N and ECE hold in both arms.

Readings the table does not cover:

1. **The registered test and the rest of the record disagree in strength, not in direction.** K1 sits on
   zero; every unregistered comparison leans to the step-down (each arm against the start, the
   initial position, mate-in-3, and the value-loss signature p2_k_lr had when it won). A registered
   replication of this A/B would decide it; that is the recommended p2_p.
2. **The instrument's flag is wrong at an exact zero.** `excludes_zero` compares a floating-point
   percentile with 0.0, so an interval whose end is exactly zero can read either way by rounding: here
   +5.55e-18 ("excludes"), at p2_j attempt 1's 0010 −8.9e-18 (the same). The fix — a tolerance far below
   the statistic's lattice (1/2,400 for six reps over 100 openings) and far above rounding — is committed
   separately, before the next registration; this round's file is kept as the instrument produced it.
3. **p2_n's plateau did not hold a restart.** The continuation arm at 2e-5 gained +0.051 in twenty
   iterations from p2_n/0040, after p2_n's flat forty. Whether p2_n/0040's six-rep start (0.644) was a low
   draw — p2_n's own rep readings at 10–30 ran 0.58–0.70 — or the round's fresh start helped, this
   record does not separate.

The open list, carried forward: the replication (recommended p2_p); the calibration sample; the move-1
prior; L2's exclusion at storage time; the host freezes.

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_o_{cont,lr}/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the runs |
| `runs/p2_o_{cont,lr}/ladder.jsonl` | ladder passes at 10 and 20 (ECE, N) |
| `runs/p2_o_{cont,lr}/sf3/` | skill-3 reps and each checkpoint's verdict against the start (K3) |
| `runs/p2_o_lr/sf3/p2o_verdict_lr_vs_cont.json` | K1, as the instrument wrote it (`excludes_zero` true; §3) |
| `runs/p2_o_{cont,lr}/reps_iter00{10,20}_r*.jsonl`, `p2_verdict_reps*.json`, `p2_m4_iter0020.json`, `startpos/` | K4, skill 0 |
| `runs/p2_o_{cont,lr}/book_rows.json` | L2 |
| `runs/p2_o_{cont,lr}_campaign.console.log`, `runs/p2_o_{cont,lr}.console.log`, `runs/p2_o_finish.console.log`, `runs/p2_o_finish.sh` | consoles and the finisher (untracked) |
