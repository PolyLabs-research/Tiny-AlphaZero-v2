# RUNLOG — p2_i (the corrected value blend against a continuation arm, from p2_h/0060)

**Runs:** `runs/p2_i_cont` (`configs/p2_g.yaml` unchanged) and `runs/p2_i_row` (`configs/p2_i_row.yaml`:
the same with `train.value_q_blend: per_row`), 20 iterations each, trained in parallel on two RunPod
pods — `klqzf7z8a5an1z` (cont) and `dqsyu5ijkh6ui5` (row), each a secure RTX 4090 of the 48 GB variant
(`vram_gb` 47.37) on a 35-CPU quota — from 2026-10-27 10:30 to 14:04 (cont) / 14:07 (row) UTC, code
`64ef834`, both warm-started from `p2_h/0060` with the league and the replay ring continued from
`runs/p2_h`. Every measurement ran on this box (torch 2.14.1+rocm7.2), pulled and queued by
`cloud/remote_campaign.sh` (one driver per arm) and finished by `runs/p2_i_finish.sh`.
**Pre-registration:** `PREREG-p2_i.md` (`64ef834`), no addenda. Every verdict below was registered
before it was measured, except where marked.

**Verdict: H_row NOT SUPPORTED. K1 FAILS: `p2_i_row` − `p2_i_cont` at 20, six reps per arm, −0.040
[−0.070, −0.010]; the lower end misses the −0.05 bound by 0.020 and the interval lies entirely below
zero, so under this protocol the corrected blend costs skill-0. K2 passes (187 / 186). K3: both arms rose
over the start (cont +0.113, row +0.075; M4 met by both), so p2_h's plateau reading did not survive the
restart. ECE and N hold; L1 holds; L2 reported (2 and 3 rows).**

---

## 1. Before launch

The start reference is p2_h/0060's three book reps on torch 2.14.1 (0.71, 0.685, 0.695; mean 0.6967)
and its three initial-position reps (`runs/p2_h/startpos/`, mean 0.7617); nothing was re-measured.
Registration read the RunPod balance at $29.47, no pods of ours running, secure RTX 4090 at $0.89/h.

**The pods:** both secure RTX 4090s came as the 48 GB variant (`vram_gb` 47.37; p2_h's was the 24 GB
card) on a 35-CPU quota, the same on both arms, torch 2.12.1+cu129 (the image p2_g and p2_h trained
on). The hardware difference from p2_h is between rounds, not between arms. The pods ran the bundle of
clean, pushed `main` at `64ef834`.

## 2. L — the continuity applied

- **L1 holds, both arms:** `init_from` `runs/p2_h/ckpt_iter0060.pt`; `seed_replay.rows` 500,000,
  `donor_committed` = `donor_head` = 3,660,595, `iteration_offset` −60; `git_dirty` false; `git_sha`
  `64ef834`. `p2_i_cont`'s `config_fingerprint` `7563c9f7…abccc` = p2_g's = p2_h's; `p2_i_row`'s is
  `787095ae…e4e7`, and its `config.yaml` differs from `p2_i_cont`'s in `value_q_blend` only
  (`tests/test_rehearsal.py::test_p2i_row_config_is_the_brief`).
- **L2 (reported, no bound):** rows of each arm's final ring (500,000 rows holding iterations 11–20)
  equal to a book position, key = squares, side, castling, en passant (`encode_board()[:70]`; the key
  without en passant gives the same counts): **cont 2** (`open_080` at iteration 12, `open_076` at 15),
  **row 3** (the same two, and `open_084` at 16). The same counter reproduces p2_h's four. Rows at the
  initial position: 551 (cont), 548 (row); distinct positions 482,178 and 482,865. The mechanism is
  RUNLOG-p2_h §2's: the exclusion acts on starts, not on positions reached.

## 3. K — the endpoint

### K1 (primary) — row against cont at 20, six reps per arm

Skill-0 from the frozen book at iteration 20, per-opening rep-means over six reps per arm, paired
bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed 0;
`runs/p2_i_row/p2i_verdict_row_vs_cont.json`, reproduced from the rep rows to the digit):

| arm | reps r1–r6 | mean | M4 (six reps) |
|---|---|---|---|
| `p2_i_cont` | 0.825, 0.7975, 0.805, 0.7625, 0.8225, 0.7775 | 0.798 | 0.798 [0.777, 0.820] met |
| `p2_i_row` | 0.7825, 0.7825, 0.75, 0.73, 0.765, 0.7425 | 0.759 | 0.759 [0.739, 0.779] met |
| **row − cont** | | **−0.040** | **[−0.070, −0.010]**, SE 0.0155 |

- **K1 FAILS.** The registered bar was a lower end above −0.05; it reads −0.070. The point, −0.040, is
  outside the predicted −0.02 to +0.02, and the interval excludes zero on the negative side: this is
  not a non-inferiority miss on noise but a measured cost. Row's best rep (0.7825) is below four of cont's six; the
  arms' six-rep M4 intervals overlap by 0.002 (0.777–0.779). Per opening, row's rep-mean is below cont's on
  57 of the 100 openings, above on 35, equal on 8 (median −0.042).

### K2 — mate-in-2

At 20: **cont 187/200, row 186/200**; row ≥ cont − 4 (186 ≥ 183) and both ≥ 183. **K2 passes.** At 10:
186 and 185. Predicted: row at or above cont by 0–5; it reads one below at both checkpoints, within the
registered margin. Of the 13 (cont) and 14 (row) failures at 20, 9 are shared (`mate2_009`, `091`,
`126`, `129`, `140`, `141`, `166`, `186`, `196`); the watchlist's three capability cases (`mate2_129`,
`165`, `166`) are unsolved in both arms at 10; at 20 `129` and `166` are unsolved in both, `165` in row
only.

### K3 — each arm against the start (informational)

Skill-0 from the frozen book, three reps per checkpoint (r1–r3), per-opening rep-means, paired
bootstrap over openings (10,000 resamples, seed 0), against the start reference (0.6967). The last
column is M4 (RUNLOG-p2_g §4's one-sample interval, bar 0.675):

| arm / checkpoint | reps | mean | vs start [95% CI] | M4: mean [95% CI] |
|---|---|---|---|---|
| cont 0010 | 0.73, 0.775, 0.755 | 0.753 | +0.057 [+0.015, +0.099] | 0.753 [0.723, 0.783] met |
| row 0010 | 0.7625, 0.78, 0.805 | 0.783 | +0.086 [+0.038, +0.134] | 0.783 [0.752, 0.813] met |
| cont 0020 | 0.825, 0.7975, 0.805 | 0.809 | +0.113 [+0.069, +0.155] | 0.809 [0.782, 0.837] met |
| row 0020 | 0.7825, 0.7825, 0.75 | 0.772 | +0.075 [+0.030, +0.121] | 0.772 [0.743, 0.801] met |

- **K3 — the plateau did not hold: both arms climbed.** The registered prediction was that the continuation
  arm stays within noise of the start; instead it reads +0.057 at 10 and +0.113 at 20, both intervals
  entirely above zero, and M4 is met by both arms at both checkpoints (predicted: by neither or by
  both). p2_h/0060 had read −0.045 [−0.093, +0.001] against *its* start after sixty iterations; twenty
  more from the same checkpoint, with nothing changed but the restart (fresh AdamW, seed streams from
  iteration 1) and the pod, read +0.113. The continuation arm is the reading RUNLOG-p2_h §6 item 2
  asked for on the restart suspects: the restart did not regress the model; it moved it up.

- **The initial position (informational, three reps at 20 against p2_h/0060's 0.7617):** cont 0.8475,
  0.7975, 0.82 (mean 0.822), **+0.060 [+0.013, +0.106]**, excludes zero; row 0.8025, 0.80, 0.7975 (mean
  0.800), **+0.038 [−0.009, +0.086]**, includes zero. p2_h had read +0.0075 there.
- *Not registered, for the record:* the ladder pass's single skill-0 replay (`role=protocol`): cont 0.76 at
  10, 0.7675 at 20; row 0.74 at 10, 0.815 at 20.

## 4. N, ECE, H

- **N / K2 detail:** mate-in-2 at 10: cont 186, row 185 (floor 183); **at 20: cont 187, row 186** (row ≥
  cont − 4 and both ≥ 183: K2's bar). mate-in-3 at 10: 143, 145; at 20: 150, 136.
- **ECE — holds, both arms:** at 10 cont 0.0558, row 0.0504; at 20 cont 0.0537, row 0.0644 (< 0.15).
- **H1 — the collapse at the initial position did NOT persist, in either arm:** `move1_prior_entropy`
  (initial-position games) read 0.0282 at iteration 1 in both arms and **0.0714 (cont) / 0.0613 (row) at
  20**, both above PREREG-p2_h's 0.05 line, where p2_h's had fallen to 0.0127 by its iteration 60. Both
  arms moved the same way, so this is the restart (fresh AdamW, seed streams from iteration 1) or the
  hardware, not the lever.
- **H2:** 0 skipped steps in 8,000, both arms.
- **H3–H5** (direction-free, by halves 1–10 → 11–20): cont value loss 0.2599 → 0.2553, decisive rate
  0.787 → 0.797, promotion loss 0.389 → 0.385, policy loss 1.536 → 1.523; row value loss 0.2611 → 0.2566
  (**not comparable** with cont's: under `per_row` the rehearsal rows' CE enters at full weight), decisive
  rate 0.777 → 0.801, promotion loss 0.388 → 0.402, policy loss 1.550 → 1.538.
- **H6** (`book_move1_entropy`, direction-free): cont 0.8014 at 10, 0.9213 at 20; row 0.91 at 10, 0.8797
  at 20 (p2_h: 0.836–0.913).

Rungs: random, greedy and phase1 won 200/200 at 10 and 20 in both arms, except one draw each against
phase1 (cont, 10) and greedy (row, 10).

## 5. Mechanics

- **Pace:** cont 10.70 min per iteration (10.30–11.03), 4,093–4,478 sims/s (mean 4,294), training
  52.8 s; row 10.86 (10.51–11.40), 4,074–4,464 sims/s (mean 4,252), training 52.4 s; 3.6 h of training
  per arm. Slower than p2_h's 8.63 min and 4,979–5,869 sims/s on the 24 GB card and 28 vCPU; the same on
  both arms, so the arm difference does not carry it.
- **Incident, 11:06–11:07 UTC:** one failed RunPod status call (a network or API error, not a 404) made
  `pod_alive` in `cloud/remote_campaign.sh` answer "gone": both drivers (pids 112591 cont, 112590 row)
  wrote `.pod_gone` and exited ("1 judged checkpoint(s) cannot be measured") while both pods kept
  training. Fixed in `a0a70f9` (12:16 UTC): a failed check is logged and retried, and a pod counts as
  gone only after two consecutive not-found answers. Markers removed, drivers restarted 12:16:11 (pids
  127815 / 127824); iteration 10 was pulled at 12:18 and measured 12:32–13:06 (cont) and 13:07–13:41
  (row). Nothing was lost; the pods ran their own bundle (`64ef834`) and the measurement instrument is
  unchanged (`a0a70f9` touches the local driver only).
- **Lifecycle:** final pulls with the ring completed 14:05:26 (cont) and 14:07:58 (row), `PULLED`
  written, both pods deleted themselves (RunPod `status` at 14:15: 0 pods, $0/h). Measurements of 0020
  one arm at a time under `gpu hold` (cont 14:05–15:01, row 15:02–15:59); the finisher saw both
  campaigns complete at 16:01 and ran cont's reps r4–r6 16:01–16:20, row's 16:20–16:42, and the
  verdict at 16:42:41 UTC. No retries.
- **Cost:** $6.80 ($29.47 → $22.67; expected about $6, worst case $14.24). All RunPod use to date $27.33.

## 6. Decision

Per the table's third row (K1 fails, any K2): **`batch_mean` stays**, and the record says the corrected
blend costs skill-0 under this protocol — the unexpected reading, so it is **diagnosed before p2_j**.
The config comments and PREREGs that describe the rehearsal rows as full-weight remain wrong
about the code (REVIEW C1): with `batch_mean` standing, the correction due is to the documentation,
not to training, and PREREG-p2_i's "the code then trains what the record has always said it trains"
does not happen.

Two readings the table does not cover:

1. **The plateau did not hold.** RUNLOG-p2_h §6 read p2_h/0060 as a plateau (−0.045 [−0.093, +0.001]
   against its start). The continuation arm here — the same config, the same checkpoint, twenty more
   iterations, nothing changed but the restart (fresh AdamW moments, seed streams from iteration 1) and
   the pod — reads +0.113 [+0.069, +0.155] against p2_h/0060 and 0.798 over six reps, the highest book
   score measured on this stack (p2_g/0060 read 0.7417 on it; p2_h's checkpoints 0.697–0.750). The
   restart suspects named in RUNLOG-p2_h §6 item 2 therefore read as a gain, not a regression; whether
   that is the restart itself (a fresh-optimizer effect), the pod's slower pace changing nothing but
   wall-clock, or p2_h/0060 having been a low draw (its ladder replay read 0.755 against its reps'
   0.697) is for the diagnosis. The plateau row's premise — that continuation was exhausted — is
   withdrawn by this reading.
2. **The cost is on the book, not on the mates.** K2 holds (186 against 187, 9 of the 13–14 failures
   shared), mate-in-3 moved 150 against 136, ECE 0.064 against 0.054, and the initial-position gain is
   +0.038 (row) against +0.060 (cont). The lever changed the weight of 5% of rows and moved skill-0 by
   −0.040 in twenty iterations; the mechanism is not in this record.

The open list, carried forward: the diagnosis above (first); mate-in-2 near its floor (186–187, floor
183); pace; the residual; ECE; L2's exclusion at storage time. p2_j's lever is Michael's choice after
the diagnosis; the LR step-down candidate named in PREREG-p2_i now has a climbing continuation arm to
be measured against, not a plateau.

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_i_{cont,row}/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the runs |
| `runs/p2_i_{cont,row}/ladder.jsonl` | ladder passes at 10 and 20, measured on this box |
| `runs/p2_i_{cont,row}/reps_iter0010_r{1,2,3}.jsonl`, `reps_iter0020_r{1..6}.jsonl`, `p2_verdict_reps_iter00{10,20}.json`, `p2_verdict_reps.json` (the driver's copy of the 0020 one) | K3 |
| `runs/p2_i_row/p2i_verdict_row_vs_cont.json` | K1 |
| `runs/p2_i_{cont,row}/startpos/` | the initial position at 20 |
| `runs/p2_i_{cont,row}_campaign.console.log`, `runs/p2_i_finish.console.log`, `runs/p2_i_finish.sh` | consoles and the finisher (untracked) |
