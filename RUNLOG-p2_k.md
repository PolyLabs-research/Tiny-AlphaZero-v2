# RUNLOG — p2_k (a tenfold learning-rate step-down against a continuation arm, from p2_j/0040)

**Runs:** `runs/p2_k_cont` (`configs/p2_g.yaml` unchanged) and `runs/p2_k_lr` (`configs/p2_k_lr.yaml`:
the same with `train.lr: 2.0e-05`), 20 iterations each, trained in parallel on two RunPod pods —
`awhc72iize8gtc` (cont) and `zs6h4wfktq3j9o` (lr), each a secure RTX 4090 of the 24 GB variant
(`vram_gb` 23.52), 64 vCPU on a 54-CPU quota, on the same host address — from 2026-10-29 16:24:34 to
20:14:27 (cont) and 16:57:39 to 20:46:07 (lr) UTC, code `6cba16e`, both warm-started from `p2_j/0040`
with the league and the replay ring continued from `runs/p2_j`. Every measurement ran on this box (torch
2.14.1+rocm7.2), pulled and queued by `cloud/remote_campaign.sh` with `MEASURE_AFTER_PULL=1` (one driver
per arm) and finished by `runs/p2_k_finish.sh`.
**Pre-registration:** `PREREG-p2_k.md` (`6cba16e`), no addenda. Every verdict below was registered
before it was measured, except where marked.

**Verdict: H_lr SUPPORTED — the registered "step-down supported" row. K1: `p2_k_lr` − `p2_k_cont` at
20, six reps per arm, +0.117 [+0.093, +0.140], entirely above zero and about twice the top of the
forecast (+0.02 to +0.06). The step-down arm reads 0.903 over six book reps, the highest book score
measured on this stack. K2 passes (189 against 190). K3: the continuation arm stayed flat a second time;
the step-down arm rose +0.110 by 10. ECE holds but rose to 0.090 in the step-down arm; L1 holds; L2
reported (1 and 2 rows).**

---

## 1. Before launch

The start reference is p2_j/0040's six book reps on torch 2.14.1 (0.7625, 0.795, 0.7575, 0.725, 0.7925,
0.795; mean 0.7713) and its three initial-position reps (0.78, 0.8175, 0.8375; mean 0.8117); nothing was
re-measured. `make test` passed on the registered tree (851 passed, 2 skipped).

**The registration's reading is five hours older than its commit.** The balance and price in the
sign-off were read at 10:00 UTC ($44.70, $0.89/h) and `make test` finished at 10:12, but the session sat
idle until 15:22 and the registration commit is timed 15:22:36. The launch's own budget check at 15:23
read $42.87 and passed for each pod ($7.12 worst case).

**The pods:** at 15:23 secure RTX 4090 stock read "Low" and both launches were refused ("no instances
currently available"); the same registered launch was retried every five minutes. Cont got a pod at
15:35:09, lr at 16:16:34. Staging met transient failures of this box's network (a connection reset, a
DNS failure, one "no public SSH endpoint yet" from the API) and was retried; on each pod all 50 inputs'
sha256 matched before its job started. Cont's pod waited 49 minutes from launch to its job (p2_j's: 27).
Both pods came as the 24 GB variant on the same host address with the same CPU quota, torch
2.12.1+cu129, Stockfish 16-1build1: the arms differ in hardware by nothing recorded.

## 2. L — the continuity applied

- **L1 holds, both arms:** `init_from` `runs/p2_j/ckpt_iter0040.pt`; `seed_replay.rows` 500,000,
  `donor_committed` = `donor_head` = 2,566,550, `iteration_offset` −40; `git_dirty` false; `git_sha`
  `6cba16e` on every iteration row. `p2_k_cont`'s fingerprint `7563c9f7…abccc` = p2_g's; `p2_k_lr`'s
  `8ab654e2…298c`, and the two pulled `config.yaml` files differ in `train.lr` only (2e-4 against 2e-5).
  Each `ckpt_iter0020.pt` carries its arm's rate in both its config and its optimizer state.
- **L2 (reported, no bound):** **cont 1** (`open_080` at iteration 12), **lr 2** (`open_080` at 12,
  `open_027` at 18) in 500,000. Rows at the initial position 568 and 537; distinct positions 481,868 and
  482,045. Predicted 0–5 per arm.

## 3. K — the endpoint

### K1 (primary) — lr against cont at 20, six reps per arm

Skill-0 from the frozen book at iteration 20, per-opening rep-means over six reps per arm, paired
bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed 0;
`runs/p2_k_lr/p2k_verdict_lr_vs_cont.json`, reproduced byte for byte from the rep rows):

| arm | reps r1–r6 | mean | M4 (six reps) |
|---|---|---|---|
| `p2_k_cont` | 0.81, 0.7775, 0.765, 0.77, 0.79, 0.805 | 0.786 | 0.786 [0.768, 0.804] met |
| `p2_k_lr` | 0.88, 0.91, 0.92, 0.90, 0.875, 0.935 | 0.903 | 0.903 [0.888, 0.918] met |
| **lr − cont** | | **+0.117** | **[+0.093, +0.140]**, SE 0.0120 |

- **K1 passes.** The interval lies entirely above zero; its lower end, +0.093, is above the forecast's
  upper end (+0.06). Every one of lr's six reps is above every one of cont's (lowest lr 0.875, highest
  cont 0.81). Per opening, lr's rep-mean is above cont's on 81 of the 100, below on 13, equal on 6
  (median +0.125).

### K2 — mate-in-2

At 20: **cont 190/200, lr 189/200**; lr ≥ cont − 4 and both ≥ 183. **K2 passes.** At 10: 191 and 191.
Of the 10 (cont) and 11 (lr) failures at 20, 9 are shared (`mate2_008`, `084`, `091`, `126`, `128`, `129`,
`138`, `166`, `196`); cont alone fails `186`, lr alone `124` and `139`. The watchlist: `mate2_165` solved
in both arms at both checkpoints, `129` and `166` unsolved in both.

### K3 — each arm against the start (informational)

Three reps per checkpoint (r1–r3) against the six-rep start (0.7713), per-opening rep-means, paired
bootstrap over openings (10,000 resamples, seed 0):

| arm / checkpoint | reps | mean | vs start [95% CI] |
|---|---|---|---|
| cont 0010 | 0.7875, 0.795, 0.79 | 0.791 | +0.020 [−0.017, +0.057] |
| cont 0020 | 0.81, 0.7775, 0.765 | 0.784 | +0.013 [−0.024, +0.050] |
| lr 0010 | 0.885, 0.8575, 0.90 | 0.881 | +0.110 [+0.075, +0.144] |
| lr 0020 | 0.88, 0.91, 0.92 | 0.903 | +0.132 [+0.103, +0.162] |

- **The continuation stayed flat a second time** (predicted), now against a six-rep start and with the
  restart shared by both arms: p2_j's plateau reading holds. **The step-down arm took most of its gain
  in the first ten iterations** (+0.110 by 10, +0.132 by 20; predicted: above the start at 20).
- M4 met by both arms at both checkpoints.

### K4 — informational: the initial position

Three reps at 20 against p2_j/0040's 0.8117: cont 0.83, 0.7975, 0.815 (mean 0.814), **+0.003 [−0.033,
+0.038]**; lr 0.92, 0.94, 0.9425 (mean 0.934), **+0.123 [+0.093, +0.153]**. The step-down's gain is not
specific to the book.

*Not registered, for the record:* the ladder pass's single skill-0 replay (`role=protocol`): cont 0.785
at 10, 0.795 at 20; lr 0.84 at 10, **0.9075 at 20** (175W/13D/12L, mean 85.6 plies against cont's 95.1).

## 4. N, ECE, H

- **N / K2 detail:** mate-in-2 at 10: 191, 191; at 20: 190 (cont), 189 (lr) (predicted 183–192).
  mate-in-3 at 10: 140, 144; at 20: 141, 146.
- **ECE — holds, but rose in the step-down arm:** cont 0.0517 at 10, 0.0542 at 20; **lr 0.0629 at 10,
  0.0904 at 20** (< 0.15). The prediction (0.04–0.07 in both arms) fails for lr at 20.
- **H1 (direction-free):** `move1_prior_entropy` (initial-position games) cont 0.44–0.72 over the twenty
  (0.62 at 1, 0.59 at 20), lr 0.62–0.72 (0.62 at 1, 0.70 at 20). Predicted: lr moves less. It does,
  narrowly; neither arm continued p2_j's climb (0.09 → 0.63).
- **H2:** 0 skipped steps in 8,000, both arms.
- **H3–H5** (direction-free, by halves 1–10 → 11–20): cont value loss 0.2501 → 0.2524, policy loss
  1.4802 → 1.4726, decisive rate 0.815 → 0.817, promotion loss 0.383 → 0.378; **lr value loss 0.3191 →
  0.3571, policy loss 1.5975 → 1.6528**, decisive rate 0.796 → 0.807, promotion loss 0.385 → 0.345. The
  prediction (lr's training losses below cont's) **fails, in the other direction:** the step-down arm's
  losses are higher in both halves and rise from the first half to the second. A lower rate fits each
  batch less, and the ring the arm trains on fills with its own, stronger games (52,790 positions per
  iteration against cont's 50,430; self-play games 102–104 plies against 98); which carries the rise is
  not in this record.
- **H6** (`book_move1_entropy`, direction-free): cont 0.696 at 10, 0.707 at 20; lr 0.708, 0.635.

Rungs: random, greedy and phase1 won 200/200 at 10 and 20 in both arms.

## 5. Mechanics

- **Pace:** cont 11.49 min per iteration (11.19–11.79), 3,715–3,913 sims/s (mean 3,811), training
  53.1 s; lr 11.42 (11.05–12.00), 3,868–4,193 sims/s (mean 4,021), training 53.8 s; 3.8 h of training
  per arm. About 9% slower than p2_j's 10.50 on one pod; both pods shared a host.
- **Lifecycle:** cont's final pull with the ring completed 20:16:17, lr's 20:48:09; `PULLED` written,
  both pods deleted themselves (RunPod status after: 0 pods of ours, $0/h). `MEASURE_AFTER_PULL` acts per
  arm, so cont's measurement (from 20:16) overlapped lr's last half-hour of training — the window in
  which a freeze of this box would still have cost lr's last iterations. Nothing happened in it.
  Measurements one arm at a time under `gpu hold`: cont 0010 20:16–20:52, 0020 20:52–21:47; lr 0010
  21:47–22:20, 0020 22:20–23:08. The finisher saw both campaigns complete at 23:10 and ran cont's reps
  r4–r6 23:10–23:29, lr's 23:29–23:46, M4 and L2 per arm, and the verdict at 23:46:04 UTC. No retries.
- **Cost:** ≈ **$8.20** at the pods' $0.89/h (cont 4.7 h from launch to deletion, lr 4.5 h; the balance
  read $42.87 → $34.59 over the same span, which also carries anything else on the account). Expected
  about $7.5, worst case $14.24. All RunPod use to date about $52.

## 6. Decision

Per the table's first row (K1 entirely above zero, K2 passes): **step-down supported. 2e-5 becomes
standing protocol: p2_l continues `p2_k_lr/0020` at 2e-5 as the default round.** A further step is a
later question. N and ECE hold in both arms, so the stop row does not apply. Any launch is a new PREREG
and Michael's cost OK.

Readings the table does not cover:

1. **The size.** The step-down arm's +0.132 against its start at 20 is the largest rise of any round
   since p2_g's first twenty iterations (+0.183; p2_i_cont's was +0.113), most of it taken in the first
   ten, with every lr rep above every cont rep and the initial position moving by the same amount. One run per arm, but the one replicate on record (p2_j's attempts) differed
   by 0.007 at 10, against an arm difference of +0.090 at 10 (lr 0.881 against cont 0.791).
2. **Calibration and fit moved the other way.** ECE 0.054 → 0.090 and the training losses up, while the
   play improved. ECE is the guard nearest to moving: at its 10 → 20 rate (+0.0027 per iteration) it
   would pass 0.15 in about twenty more iterations, inside one default round. p2_l's PREREG has to read
   it at every checkpoint and say in advance what a rise past 0.15 decides.
3. **The instrument is nearing its ceiling.** Skill-0 from the book at 0.903 (the ladder replay 0.9075,
   12 losses in 200) leaves about a tenth of headroom; the next gains of this size cannot be resolved
   against Stockfish skill 0. A stronger rung for the book replays is a question for p2_l's PREREG, before
   the default round runs, so the change of instrument is registered rather than discovered.

The open list, carried forward: p2_l (the default round at 2e-5) and its instrument (the rung above
skill 0); ECE's rise; the loss rise; mate-in-2 near its floor (189–191, floor 183); L2's exclusion at
storage time; the host freezes; pace.

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_k_{cont,lr}/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the runs |
| `runs/p2_k_{cont,lr}/ladder.jsonl` | ladder passes at 10 and 20, measured on this box |
| `runs/p2_k_{cont,lr}/reps_iter0010_r{1,2,3}.jsonl`, `reps_iter0020_r{1..6}.jsonl`, `p2_verdict_reps_iter00{10,20}.json`, `p2_verdict_reps.json` (the driver's copy of the 0020 one) | K3 |
| `runs/p2_k_lr/p2k_verdict_lr_vs_cont.json` | K1 |
| `runs/p2_k_{cont,lr}/p2_m4_iter0020.json` | M4 at 20, six reps |
| `runs/p2_k_{cont,lr}/startpos/` | K4 |
| `runs/p2_k_{cont,lr}/book_rows.json` | L2 |
| `runs/p2_k_{cont,lr}_campaign.console.log`, `runs/p2_k_{cont,lr}.console.log`, `runs/p2_k_finish.console.log`, `runs/p2_k_finish.sh` | consoles and the finisher (untracked) |
