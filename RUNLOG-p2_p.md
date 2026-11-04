# RUNLOG — p2_p (p2_o's A/B replicated on seed 43: 2e-6 against 2e-5, from p2_n/0040, judged on skill 3)

**Runs:** `runs/p2_p_cont` (`--config configs/p2_k_lr.yaml --seed 43`, 2e-5) and `runs/p2_p_lr`
(`--config configs/p2_o_lr.yaml --seed 43`, 2e-6), 20 iterations each, on two RunPod pods —
`tvmtadtr73udhs` (cont) and `wd4vmariqtkya4` (lr), each a secure RTX 4090 of the 24 GB variant (`vram_gb`
23.52) — from 2026-11-03 14:25:53 to 18:18:29 (cont) and 11:54:08 to 15:34:50 (lr) UTC, code `0781889`,
both warm-started from `p2_n/0040` with the league and the replay ring continued from `runs/p2_n`. Every
measurement ran on this box (torch 2.14.1+rocm7.2), by `cloud/remote_campaign.sh` with
`MEASURE_AFTER_PULL=1` (one driver per arm) and `runs/p2_p_finish.sh`, with the instrument's exact-zero
fix (`f7779a8`).
**Pre-registration:** `PREREG-p2_p.md` (`0781889`); Addendum A (`890e3c2`) corrects the registration's
date (2026-11-03, written as 11-04) and changes nothing else. Every verdict below was registered before
it was measured, except where marked.

**Verdict: H_lr2 SUPPORTED — the registered "second step-down replicates" row. K1 (lr − cont at 20, skill
3, six reps per arm): +0.075 [+0.039, +0.111], entirely above zero; K2 passes (191 and 191). 2e-6 becomes
standing protocol, and p2_q continues `p2_p_lr/0020` at 2e-6 — judged at the next rung, because 0020 reads
0.754 at skill 3, above the row's 0.75. The two seeds together (K5, informational) read +0.053 [+0.028,
+0.078]. ECE and N hold; L1 holds in both arms; L2 reported (2 and 2 rows).**

---

## 1. Before launch

The start reference is p2_o's: p2_n/0040's six skill-3 book reps (mean 0.6442), its three skill-0 book
reps and three initial-position reps; nothing was re-measured. Read at registration, 2026-11-03 08:52 UTC:
balance $30.24, secure RTX 4090 at $0.89/h with stock "None"; `make test` 954 passed, 2 skipped; the 50
staged inputs' sha256 unchanged since p2_o.

**The pods:** both registered launches were refused for want of instances from 08:55, retried every five
minutes (and, past the pipelines' two-hour window, by a wrapper that reran the same launch only after a
"no pod" result). The lr arm got its pod at 11:47:03 after 34 refusals (24 vCPU, 20-CPU quota, 109 GB
RAM); the cont arm at 14:18:29 after 64 (64 vCPU, 54-CPU quota, 62 GB RAM). Michael was asked at 13:53
whether to wait or allow another configuration by addendum, and chose to wait. Both staged without a
retry; every input's sha256 matched on each pod.

## 2. L — the continuity applied

- **L1 holds, both arms:** `init_from` `runs/p2_n/ckpt_iter0040.pt`; `seed` 43; `seed_replay.rows`
  500,000, `donor_committed` = `donor_head` = 2,649,005, `iteration_offset` −40; `git_dirty` false;
  `git_sha` `0781889`. `config_fingerprint` `d42a6d33…346e` (cont) and `86fcc672…7d65` (lr), each equal
  to its `config.yaml`'s sha256 and to the value registered; each resolved config differs from its file
  in `seed` only; `lr` 2e-5 and 2e-6 on every row; each `ckpt_iter0020.pt` carries seed 43 and its arm's
  rate in its config and its optimizer state.
- **L2 (reported, no bound):** cont 2, lr 2 in 500,000, the same two openings in both (`open_001` at
  iteration 11, `open_028` at 17).

## 3. K — the replication, at skill 3

### K1 (primary) — lr against cont at 20, six reps per arm

Per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000
resamples, seed 0; `runs/p2_p_lr/sf3/p2p_verdict_lr_vs_cont.json`, reproduced byte for byte; all rows
`stockfish_s3`, each arm on its own checkpoint):

| arm | reps r1–r6 | mean | W / D / L (1,200 games) |
|---|---|---|---|
| `p2_p_cont` (2e-5) | 0.6425, 0.6675, 0.74, 0.7025, 0.6525, 0.6675 | 0.679 | 725 / 179 / 296 |
| `p2_p_lr` (2e-6) | 0.7825, 0.7375, 0.7425, 0.7575, 0.745, 0.76 | 0.754 | 811 / 188 / 201 |
| **lr − cont** | | **+0.075** | **[+0.039, +0.111]**, SE 0.0185 |

- **K1 passes**, with no boundary question: the lower end is 0.039. Forecast +0.02 to +0.06; the point is
  above it. Per opening, lr's rep-mean is above cont's on 61 of the 100, below on 32, equal on 7 (median
  +0.083). Five of lr's six reps are above all six of cont's (cont's best, 0.74, is above lr's 0.7375).

### K2 — mate-in-2

At 20: **cont 191/200, lr 191/200**; lr ≥ cont − 4 and both ≥ 183. **K2 passes.** At 10: 192 and 191.

### K3 — each arm against the start (p2_n/0040, 0.644), informational

| arm / checkpoint | reps | vs start [95% CI] |
|---|---|---|
| cont 0010 | 0.65, 0.6375, 0.6475 | +0.001 [−0.042, +0.042] |
| **cont 0020 (six)** | | **+0.035 [+0.002, +0.067]** |
| lr 0010 | 0.755, 0.7325, 0.7475 | +0.101 [+0.063, +0.140] |
| **lr 0020 (six)** | | **+0.110 [+0.078, +0.142]** |

Predicted: both arms above the start at 20 and lr above cont at both checkpoints — all held. The
step-down arm's gain was in place by iteration 10, as in p2_o and p2_k_lr.

### K4 — skill 0 and the initial position (continuity)

Skill-0 reps against p2_n/0040's three: cont −0.003, −0.012; lr −0.003, +0.010. M4 at skill 0: cont
0.953, lr 0.974, met. Initial position: cont +0.010 [−0.011, +0.030], lr +0.013 [−0.008, +0.033]. Skill 0
is at its ceiling.

### K5 — p2_o and p2_p together (informational, registered)

Twelve skill-3 reps per arm at 20 (each seed's six): cont 0.687, lr 0.740, **lr − cont +0.053 [+0.028,
+0.078]**, SE 0.0126 (`runs/p2_p_lr/sf3/p2p_verdict_pooled_o_and_p.json`). As registered, no row turns on
it: the bootstrap resamples openings, not runs, and p2_o's half had been seen.

## 4. N, ECE, H

- **N — holds:** mate-in-2 cont 192, 191; lr 191, 191 (predicted 185–195). mate-in-3 cont 156, 154; lr 158,
  155.
- **ECE — holds:** cont 0.0897, 0.1033; lr 0.0996, 0.1032 (< 0.15; predicted 0.08–0.12).
- **H7 — the step-down arm's value loss rose again** (predicted): by halves lr 0.3749 → 0.3949, first five
  0.357 against last five 0.395 (p2_o's lr 0.358 → 0.395; p2_k_lr's 0.319 → 0.357); cont 0.3247 → 0.3270.
  Policy loss lr 1.5136 → 1.4640, cont 1.5102 → 1.5074. Three times now, the rate step that wins raises
  the value loss and lowers the policy loss.
- **H1 (direction-free):** `move1_prior_entropy` rose from p2_n's collapsed 0.07 in both arms: cont 0.07 →
  0.19, lr 0.07 → 0.15.
- **H3–H5:** decisive rate cont 0.781 / 0.779, lr 0.795 / 0.797; promotion loss cont 0.365 / 0.357, **lr
  0.449 / 0.470** — higher in the step-down arm by a quarter, direction-free.
- **H2:** 0 skipped steps in 8,000, both arms.

Rungs: random, greedy and phase1 won 200/200 at every checkpoint in both arms.

## 5. Mechanics

- **Pace:** lr 11.03 min per iteration (10.60–11.41), 4,064–4,377 sims/s; cont 11.63 (11.28–12.02),
  3,929–4,148 sims/s. As in p2_m and p2_o, the 54-CPU pod was the slower one.
- **Lifecycle:** final pulls with the ring at 15:36:23 (lr) and 18:23:43 (cont), `PULLED` written, both
  pods deleted themselves. Because the arms started 2.5 h apart and `MEASURE_AFTER_PULL` acts per arm, the
  lr arm was measured (15:36–16:49) while the cont pod held its only copy; nothing happened in that
  window. Measurements under `gpu hold`: lr 0010 15:36–16:05, 0020 –16:49; cont 0010 18:24–18:54, 0020
  –19:41; the finisher's skill-3 reps cont 0010 19:45–20:09, 0020 –20:58, lr 0010 20:58–21:22, 0020
  –22:07; K1 and K5 at 22:07:29 UTC. No retries.
- **Cost:** ≈ **$7.06** at the pods' $0.89/h (lr 11:47–15:37, cont 14:18–18:24); expected about $7.7,
  worst case $14.24. All RunPod use by this project to date about $87.

## 6. Decision

Per the table's first row (K1 entirely above zero, K2 passes): **the second step-down replicates. 2e-6
becomes standing protocol; p2_q continues `p2_p_lr/0020` at 2e-6 as the default round.** The row's rung
clause applies: `p2_p_lr/0020` reads **0.754** at skill 3 (six reps), above 0.75, so p2_q is judged at the
next rung of PREREG-p2_l's climb list (skill 5), its start reference measured before launch. N and ECE
hold. Any launch is a new PREREG and Michael's cost OK.

Readings the table does not cover:

1. **p2_o was a real effect read with too little precision.** p2_o +0.031 [0, +0.0625], p2_p +0.075
   [+0.039, +0.111], the two seeds together +0.053 [+0.028, +0.078]. Michael's ruling on p2_o (the
   boundary read as "includes zero") stands as that round's record; the replication is what decides
   2e-6.
2. **The step-down signature, three times.** p2_k_lr, p2_o_lr and p2_p_lr each gained most of their margin
   in ten iterations, each with the value loss rising and the policy loss falling, while ECE held.
3. **The rate ladder.** 2e-4 → 2e-5 (p2_k) and 2e-5 → 2e-6 (p2_o/p2_p) each won after a plateau. A third
   tenfold step lands at 2e-7, where twenty iterations move the weights very little; when p2_q plateaus,
   whether to take it or change lever is a choice to make then.

The open list, carried forward: p2_q (the default round at 2e-6, judged at skill 5); the calibration
sample; the promotion-loss rise under the step-down; L2's exclusion at storage time; the host freezes.

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_p_{cont,lr}/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the runs |
| `runs/p2_p_{cont,lr}/ladder.jsonl` | ladder passes at 10 and 20 (ECE, N) |
| `runs/p2_p_{cont,lr}/sf3/` | skill-3 reps and each checkpoint's verdict against the start (K3) |
| `runs/p2_p_lr/sf3/p2p_verdict_lr_vs_cont.json` | K1 |
| `runs/p2_p_lr/sf3/p2p_verdict_pooled_o_and_p.json` | K5 |
| `runs/p2_p_{cont,lr}/reps_iter00{10,20}_r*.jsonl`, `p2_verdict_reps*.json`, `p2_m4_iter0020.json`, `startpos/` | K4, skill 0 |
| `runs/p2_p_{cont,lr}/book_rows.json` | L2 |
| `runs/p2_p_{cont,lr}_campaign.console.log`, `runs/p2_p_{cont,lr}.console.log`, `runs/p2_p_finish.console.log`, `runs/p2_p_finish.sh` | consoles and the finisher (untracked) |
