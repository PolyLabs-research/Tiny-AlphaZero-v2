# RUNLOG — p2_m (the restart against a resume, twenty iterations each from p2_l/0040, judged on skill 3)

**Runs:** `runs/p2_m_init` (`--init runs/p2_l/ckpt_iter0040.pt`, the standing protocol: iterations 1–20)
and `runs/p2_m_resume` (a copy of p2_l's training state continued by `--resume`: iterations 41–60), both
on `configs/p2_k_lr.yaml` (lr 2e-5), trained in parallel on two RunPod pods — `h0oeicdcl98lfu` (init) and
`tdbsch5mqbpb1f` (resume), each a secure RTX 4090 of the 24 GB variant (init: `vram_gb` 23.52 in its
provenance; resume: `nvidia-smi` 24,564 MiB at staging) — from 2026-10-31 14:04:14 to 17:42:37 (init) and
14:08:19 to 17:59:24 (resume) UTC, code `110c2df`. Every measurement ran on this box (torch
2.14.1+rocm7.2), by `cloud/remote_campaign.sh` with `MEASURE_AFTER_PULL=1` (one driver per arm) and
`runs/p2_m_finish.sh`.
**Pre-registration:** `PREREG-p2_m.md` (`110c2df`), no addenda. Every verdict below was registered before
it was measured, except where marked.

**Verdict: H_restart NOT SUPPORTED — the registered row "the restart is not the lever at this length". K1
(init 0020 − resume 0060, skill 3, six reps per arm): +0.013 [−0.022, +0.048], includes zero. K2 at the
first checkpoint: −0.016 [−0.061, +0.031]. The resumed arm took its early gain too (+0.041 by iteration
50), so the early step of p2_k_lr and p2_l is not the restart. K3: init above the start at 20 (+0.043
[+0.010, +0.075]), resume within noise (+0.030 [−0.003, +0.061]). N and ECE hold; L1 holds in both arms;
L2 reported (2 and 1 rows).**

---

## 1. Before launch

The start reference at skill 3 is p2_l/0040's six book reps (0.575, 0.585, 0.57, 0.5675, 0.6125, 0.62;
mean 0.5883); at skill 0 its three book reps and three initial-position reps. Nothing was re-measured.
Registration read the balance at $51.77 (auto-top-up on), no pods of ours, secure RTX 4090 at $0.89/h;
`make test` 897 passed, 2 skipped. Disk at registration 9.1 GB free; Michael cleared space during the run
(47 GB free by 18:01).

**The resume arm's directory:** `runs/p2_l`'s `state.json`, `config.yaml`, `provenance.json`,
`README.md`, `iterations.jsonl` and `replay/` copied to `runs/p2_m_resume`, and its pool checkpoints
0021–0040 hard-linked (inode checked: the same files, two links); `state.json` and the ring's
`meta.json` identical to p2_l's. `runs/p2_l` was not modified.

**The pods:** init got its pod at 13:57:49 (24 vCPU, 20-CPU quota, 109 GB RAM); resume was refused at
13:57:49 ("no longer any instances") and got its pod on the next try at 14:02:53 (64 vCPU, 54-CPU quota,
62 GB RAM). Both staged without a retry, every input's sha256 matched (50 and 53 files), jobs started
14:04:14 and 14:08:19. Both are the 24 GB card; the CPU quotas differ (20 against 54), which moves pace,
not computation.

## 2. L — the continuity applied

- **L1 holds, init:** `config_fingerprint` `8ab654e2…298c`; `init_from` `runs/p2_l/ckpt_iter0040.pt`;
  `seed_replay.rows` 500,000, `donor_committed` = `donor_head` = 2,644,005, `iteration_offset` −40;
  `git_dirty` false; `git_sha` `110c2df`; `lr` 2e-5 on all 20 rows; `ckpt_iter0020.pt` at step 8,000
  (AdamW's per-parameter step 8,000: fresh).
- **L1 holds, resume:** after iteration 41, `state.json` iteration 41, step 16,400, `resumed_with`
  ['110c2df']; `ckpt_iter0041.pt`'s AdamW per-parameter step **16,400** against p2_l/0040's 16,000 — the
  optimizer was carried; `iterations.jsonl` rows 1–40 equal p2_l's as parsed; `lr` 2e-5 on rows 41–60;
  `ckpt_iter0060.pt` at step 24,000. The resumed rows and checkpoints carry p2_l's launch SHA
  (`2c9be5d`): `--resume` records its own code only in `resumed_with`, and `provenance.json` stays p2_l's
  (hence its 47.37 GB). No file under `src/`, `scripts/`, `cloud/` or `configs/` differs between
  `2c9be5d` and `110c2df`, so both arms ran the same training code.
- **L2 (reported, no bound):** **init 2** (`open_076` at iteration 15, `open_062` at 19), **resume 1**
  (`open_027` at 60) in 500,000.

## 3. K — the restart against the resume, at skill 3

### K1 (primary) — init 0020 against resume 0060, six reps per arm

Per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000
resamples, seed 0; `runs/p2_m_init/sf3/p2m_verdict_init_vs_resume.json`, reproduced byte for byte; all
rows `stockfish_s3`, each arm on its own checkpoint):

| arm | reps r1–r6 | mean | W / D / L (1,200 games) |
|---|---|---|---|
| `p2_m_init` 0020 | 0.6025, 0.6575, 0.615, 0.625, 0.6575, 0.6275 | 0.631 | 660 / 194 / 346 |
| `p2_m_resume` 0060 | 0.655, 0.595, 0.6325, 0.585, 0.6025, 0.6375 | 0.618 | 650 / 183 / 367 |
| **init − resume** | | **+0.013** | **[−0.022, +0.048]**, SE 0.0180 |

- **The interval includes zero.** The forecast (CI entirely above zero, +0.02 to +0.06) failed. Per
  opening, init's rep-mean is above resume's on 48 of the 100, below on 41, equal on 11 (median 0.000).

### K2 — the same at the first judged checkpoint (where a burst would show)

init 0010 (0.62, 0.575, 0.645; mean 0.613) − resume 0050 (0.67, 0.61, 0.6075; mean 0.629): **−0.016
[−0.061, +0.031]**, SE 0.023 (`…_iter10.json`). Predicted above zero; the point is on the resumed side.

### K3 — each arm against the start (p2_l/0040, 0.588)

| arm / checkpoint | reps | vs start [95% CI] |
|---|---|---|
| init 0010 | 0.62, 0.575, 0.645 | +0.025 [−0.016, +0.065] |
| init 0020 (r1–r3) | 0.6025, 0.6575, 0.615 | +0.037 [−0.006, +0.080] |
| **init 0020 (six)** | | **+0.043 [+0.010, +0.075]** |
| resume 0050 | 0.67, 0.61, 0.6075 | +0.041 [−0.004, +0.087] |
| resume 0060 (r1–r3) | 0.655, 0.595, 0.6325 | +0.039 [−0.004, +0.082] |
| **resume 0060 (six)** | | **+0.030 [−0.003, +0.061]** |

- Predicted: init above the start by 10, resume within noise. Init is not above it at 10 (its interval
  includes zero); at 20 it is. Resume's intervals all include zero, by margins of 0.003–0.004 — its
  points are as large as init's, and **its gain was in place by 50, ten iterations in, without a
  restart.** The row's condition "either arm above the start at its last checkpoint" is met by init.
- The round's gain over its start, +0.03 to +0.04 at skill 3 in each arm, is less than half of p2_l's
  (+0.104 over forty iterations from p2_k_lr/0020).

### K4 — skill 0 and the initial position (continuity)

Three skill-0 reps per checkpoint against p2_l/0040's three: init +0.002, +0.004; resume +0.006, +0.017,
all intervals including zero. M4 at skill 0: init 0.942 [0.925, 0.957], resume 0.954 [0.938, 0.969],
met. Initial position: init +0.011 [−0.011, +0.032], resume +0.002 [−0.023, +0.026]. Skill 0 at its
ceiling, as registered. The ladder's single skill-0 replay (*not registered*): init 0.94, 0.9475; resume
0.935, 0.9625.

## 4. N, ECE, H

- **N — holds, both arms:** mate-in-2 init 189 at 10, **187** at 20; resume 188 at 50, **189** at 60
  (≥ 183). mate-in-3 149, 144; 148, 144. Watchlist: `mate2_165` and `mate2_166` solved at every checkpoint
  in both arms (`166` had been unsolved through p2_l), `mate2_129` unsolved.
- **ECE — holds, both arms:** init 0.0892, **0.0997**; resume 0.0968, **0.0972** (< 0.15; predicted
  0.07–0.11).
- **H8 (new, direction-free) — no burst:** value loss over the first five iterations against the last five,
  init 0.3500 → 0.3473, resume 0.3504 → 0.3451; policy loss init 1.5392 → 1.5286, resume 1.5408 →
  1.5246. Init's first five rows (value 0.3507, 0.3509, 0.3518, 0.3480, 0.3485) look like resume's
  (0.3531, 0.3519, 0.3509, 0.3488, 0.3475); gradient norms 4.79 and 4.51 over the first five. The
  prediction (init's losses move more early) **fails**: a fresh AdamW at 2e-5 left no mark the logs show.
- **H1 (direction-free):** `move1_prior_entropy` init 0.39–0.50, flat (0.50 at 1, 0.45 at 20); resume
  falling, 0.50 at 41 to 0.30 at 60.
- **H2:** 0 skipped steps in 8,000, both arms.
- **H3–H5** (by halves): init value 0.3506 / 0.3500, policy 1.5371 / 1.5305, decisive rate 0.786 / 0.778,
  promotion 0.362 / 0.366; resume value 0.3509 / 0.3475, policy 1.5406 / 1.5297, decisive 0.780 / 0.781,
  promotion 0.372 / 0.363.
- **H6** (`book_move1_entropy`): init 0.544, 0.555; resume 0.526, 0.544.

Rungs: random, greedy and phase1 won 200/200 at every checkpoint in both arms.

## 5. Mechanics

- **Pace:** init 10.92 min per iteration (10.54–11.50), 4,080–4,395 sims/s (mean 4,294), training 52.1 s;
  resume 11.55 (11.20–11.79), 3,894–4,178 sims/s (mean 4,047), training 54.7 s. The 20-CPU pod was the
  faster one.
- **Lifecycle:** final pulls with the ring at 17:46:35 (init) and 18:01:22 (resume), `PULLED` written,
  both pods deleted themselves. Measurements under `gpu hold`, one arm at a time: init 0010 17:47–18:16,
  0020 18:16–19:08; resume 0050 19:08–19:49, 0060 19:49–20:37; the finisher's skill-3 reps init 0010
  20:39–21:03, 0020 21:03–22:03, resume 0050 22:03–22:28, 0060 22:28–23:21, verdicts at 23:20:39 UTC. No
  retries. The session's watcher raised one false alarm at 18:01 (it matched the ladder's printed
  "M3 CLAUSE FAILED", its comparison with the old 0.95 milestone; p2_m's registered floor is 183); the
  drivers were unaffected.
- **Cost:** ≈ **$6.95** at the pods' $0.89/h (init 13:58–17:47, resume 14:03–18:02). Expected about
  $7.7, worst case $14.24. All RunPod use to date about $66.

## 6. Decision

Per the table's second row (K1 includes zero; init above the start at its last checkpoint): **the
restart is not the lever at this length. The standing protocol is unchanged (`--init` between rounds,
the simpler record); p2_n continues `p2_m_init/0020` at 2e-5 as the default round, and the early-gain
shape is read again there.** N and ECE hold in both arms. Any launch is a new PREREG and Michael's cost
OK.

Readings the table does not cover:

1. **The early step is not the restart.** Without a restart the resumed arm gained +0.041 by its tenth
   iteration, no less than the restarted arm's +0.025, and the restarted arm's losses show no burst
   (H8). Whatever puts each round's gain in its first ten iterations happens with or without a fresh
   optimizer and fresh seed streams. What remains is the ring and the league each round starts from, or
   the measurement's own noise at three reps; nothing run so far separates them.
2. **The climb at 2e-5 is slowing.** p2_k_lr: +0.132 over its start in twenty iterations (skill 0); p2_l:
   +0.104 in forty (skill 3); p2_m: +0.030 to +0.043 in twenty (skill 3). If p2_n reads flat against
   `p2_m_init/0020`, the registered next step is the 2e-6 step-down A/B (PREREG-p2_l's plateau row,
   carried).
3. **Restart and resume agree.** With the restart shown not to matter at 2e-5, `--resume` is available
   as an equivalent continuation when a run's own state is the better start (it keeps one log and one
   ring), at no measured cost.

The open list, carried forward: p2_n (the default round at 2e-5 from `p2_m_init/0020`, judged at skill 3,
with the 2e-6 step-down as its plateau row); what the early gain is; the calibration sample (ECE ~0.09–0.10,
holding); mate-in-2 near its floor (187–189, floor 183); L2's exclusion at storage time; the host freezes.

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_m_{init,resume}/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the runs (resume: rows 1–40 are p2_l's; its provenance is p2_l's copy) |
| `runs/p2_m_{init,resume}/ladder.jsonl` | ladder passes (ECE, N) at each arm's two checkpoints |
| `runs/p2_m_{init,resume}/sf3/` | skill-3 reps and each checkpoint's verdict against the start (K3) |
| `runs/p2_m_init/sf3/p2m_verdict_init_vs_resume.json`, `…_iter10.json` | K1, K2 |
| `runs/p2_m_{init,resume}/reps_iter*.jsonl`, `p2_verdict_reps_iter*.json`, `p2_verdict_reps.json`, `p2_m4_iter*.json`, `startpos/` | K4, skill 0 |
| `runs/p2_m_{init,resume}/book_rows.json` | L2 |
| `runs/p2_m_{init,resume}_campaign.console.log`, `runs/p2_m_{init,resume}.console.log`, `runs/p2_m_finish.console.log`, `runs/p2_m_finish.sh` | consoles and the finisher (untracked) |
