# RUNLOG — p2_h (p2_g continued, sixty more iterations, trained on RunPod)

**Run:** `runs/p2_h`, 60 iterations, trained on RunPod pod `05jerlv1g5lw5j` (secure RTX
4090, 28 vCPU / quota 23.8) from 2026-10-25 10:47 to 19:26 UTC, code `f1372a4`, warm-started
from `p2_g/0060` with the league and the replay ring continued from `runs/p2_g`. Every
measurement ran on this box (torch 2.14.1+rocm7.2), pulled and queued by
`cloud/remote_campaign.sh`.
**Pre-registration:** `PREREG-p2_h.md` (`f1372a4`), no addenda. Step 0 (the initial-position
reference on 2.14.1) is `b0a9b2b`. Every verdict below was registered before it was
measured, except where marked.

**Verdict: H_cont NOT SUPPORTED. K1's interval includes zero (−0.045 [−0.093, +0.001]), so
the registered row is PLATEAU. K2 no dip by the registered test; K3 (M4) NOT MET at 0060;
N and ECE hold; L2 FAILS (four ring rows equal a book position).**

---

## 1. Before launch

**Step 0** (this box): p2_g/0060 from the initial position, three skill-0 reps on torch
2.14.1 — 0.745, 0.75, 0.7675 (mean 0.754; 2.12.1 had read 0.7625, 0.74, 0.81). Committed as
`b0a9b2b`; it is K4's reference.

**The pod:** a secure RTX 4090 with 28 vCPU (quota 23.8, against p2_g's 20.4) at $0.89/h,
torch 2.12.1+cu129 (`provenance.json`), the stack p2_g trained on. The starting checkpoint
and the 500,000 carried-over positions arrived byte-identical (SHA-256). L1 was checked
from `provenance.json` before iteration 1 committed. No training-step bridge was needed and
none was run.

## 2. L — the continuity applied

- **L1 holds:** `config_fingerprint`
  `7563c9f74f4d6231f0ef6688644ee1994d4a50426db1e1fdd5e714d8ab4abccc` = p2_g's; `init_from`
  `runs/p2_g/ckpt_iter0060.pt`; `seed_replay.rows` 500,000, `donor_committed` =
  `donor_head` = 3,611,615, `iteration_offset` −60; `git_dirty` false; `config.yaml` has
  zero deltas from `configs/p2_g.yaml`.
- **L2 fails:** rows of p2_h's final ring equal to a book position: **4**, expected 0
  (p2_g: 0). `open_023` once (iteration 58), `open_027` twice (iteration 60), `open_065`
  once (iteration 56), in 500,000 rows that hold iterations 51–60 (482,847 distinct
  positions; 536 rows at the initial position). The key compares squares, side to move,
  castling and en passant; p2_g's check compared squares, side and castling, and that key
  gives the same four. *The mechanism:* `opening_exclude` redraws a game's *start* when it
  is a book position, but a game that starts fewer than four random plies in and then
  plays into a book position stores it. p2_g's zero was the same draw landing on zero.
  Four rows in 500,000 (0.0008%) cannot move the measurement; the expectation did not
  hold, and an exclusion at storage time (or a registered count) is a change for a later
  round, not a rewording of this one.

## 3. K — the endpoint and the trajectory

Skill-0 from the frozen book, three reps per checkpoint, per-opening rep-means, paired
bootstrap over openings (10,000 resamples, seed 0), against the start reference p2_g/0060
on 2.14.1 (0.7375, 0.7725, 0.715; mean 0.7417). The last column is K3's one-sample interval
(RUNLOG-p2_g §4's method), bar 0.675:

| checkpoint | reps | mean | vs start [95% CI] | M4: mean [95% CI] |
|---|---|---|---|---|
| 0010 | 0.745, 0.7175, 0.7375 | 0.733 | −0.008 [−0.053, +0.036] | 0.733 [0.702, 0.765] met |
| 0020 | 0.755, 0.7675, 0.68 | 0.734 | −0.008 [−0.050, +0.035] | 0.734 [0.703, 0.765] met |
| 0030 | 0.79, 0.755, 0.705 | 0.750 | +0.008 [−0.036, +0.053] | 0.750 [0.721, 0.778] met |
| 0040 | 0.705, 0.77, 0.7025 | 0.726 | −0.016 [−0.058, +0.027] | 0.726 [0.693, 0.757] met |
| 0050 | 0.73, 0.7375, 0.75 | 0.739 | −0.003 [−0.046, +0.041] | 0.739 [0.708, 0.770] met |
| **0060** | 0.71, 0.685, 0.695 | **0.697** | **−0.045 [−0.093, +0.001]** | **0.697 [0.663, 0.729] NOT met** |

- **K1 — CI includes zero: the PLATEAU row.** The point is −0.045 and the upper bound is
  +0.0008: the interval clears zero by less than a thousandth. The registered forecast
  was +0.05 to +0.10; it was a forecast, not a bar, and it was wrong.
- **K2 — no dip, by the registered test.** No checkpoint's interval is entirely below
  zero (0060's upper bound is +0.0008). The six points: −0.008, −0.008, +0.008, −0.016,
  −0.003, −0.045; least-squares slope −0.0055 per ten iterations. The 120-iteration
  trajectory against p2_g's start (p2_g's six points on 2.12.1, then p2_h's added to
  p2_g/0060's +0.243, the seam carrying the bridge's ±0.04): +0.130, +0.183, +0.181,
  +0.181, +0.223, +0.243 | +0.235, +0.235, +0.251, +0.227, +0.240, +0.198.
- **K3 — M4 NOT MET at 0060.** 0.697 [0.663, 0.729] against the bar of 0.675; the lower
  bound misses by 0.012. Met at every earlier checkpoint (lower bounds 0.693–0.721). The
  start reference measures 0.742 [0.709, 0.774] by the same method. The expectation was
  "holds".
- **K4 — the initial position (informational):** +0.0075 [−0.037, +0.052]; p2_h/0060 reps
  0.7525, 0.7675, 0.765 (mean 0.762) against Step 0's 0.754. From the position self-play
  trains on, nothing moved either way.
- *Not registered, for the record:* the ladder pass's single skill-0 replay at each
  checkpoint (`role=protocol`, its own seed, the same book) read 0.7725, 0.74, 0.73, 0.73,
  0.7325, 0.755 — at 0060 above all three registered reps. It is not part of the
  statistic and changes nothing above; it says 0060's three reps sit at the low end of
  what this checkpoint produces.

## 4. N, ECE, H

- **N — holds:** mate-in-2 188/200 = 0.940 at 0060 (floor 0.915). Across the judged
  checkpoints: 191, 191, 187, 191, 191, 188 (p2_g: 184–190). mate-in-3: 146, 154, 151, 145,
  145, 145.
- **ECE — holds:** 0.0445 at 0060 (< 0.15); 0.0376–0.051 at every judged checkpoint.
- **H1 — the collapse at the initial position persists, as registered:**
  `move1_prior_entropy` (initial-position games only) 0.0127 at iteration 60 (< 0.05). It
  read 0.0462 at iteration 1 and fell; p2_g's had risen from 0.004 to 0.026.
- **H2:** 0 skipped steps in 24,000.
- **H3–H5** (direction-free): value loss by halves 0.269 → 0.264; decisive rate 0.777 →
  0.794 (0.783 at iteration 1, 0.773 at 60); promotion loss by halves 0.357 → 0.368.
  Policy loss 1.568 → 1.526.
- **H6** (book entropy, direction-free): 0.862, 0.888, 0.877, 0.836, 0.913, 0.857 — flat
  where p2_g's fell from 1.435 to 0.903.

Rungs: random, greedy and phase1 won 200/200 at every judged checkpoint.

## 5. Mechanics

- **Pace:** 8.05–9.11 min per iteration (mean 8.63), 4,979–5,869 sims/s, training 47.7 s per
  iteration; 8.6 h of training. Faster than p2_g's 10.53 on the larger CPU quota; the
  games do not depend on the actor count.
- **Lifecycle:** the local driver armed at 10:49 UTC, one instance throughout; six
  measurements of 41–59 min each under `gpu hold` (llama-server paused and restored each
  time); the final pull with the ring completed at 19:31:56 UTC, `PULLED` was written and
  the pod deleted itself; `campaign complete` at 20:30:49 UTC. No incidents.
- **Cost:** $7.95 ($37.42 → $29.47). All RunPod use to date $20.53.
- **Reporting:** the session that was to score the verdict stalled on a permission prompt
  at 21:17 UTC and died at 22:58; the verdict was scored on 2026-10-26 from the committed
  records, with nothing re-measured.

## 6. Decision

Per the table's second row: **plateau** at about p2_g/0060's strength under this protocol.
Continuation is no longer the default round; p2_i takes a lever from the open list as an
A/B from p2_h/0060 against a continuation arm, not against a fixed reference.

Two readings the table does not cover, named so the choice of lever sees them:

1. **M4 is not met at 0060 while N and ECE hold.** The "broken guard first" row names N
   and ECE only; K3 is a registered expectation that failed, by 0.012 on the lower bound,
   after holding at five checkpoints. Whether p2_i treats it as a guard to repair or as
   the plateau's noise is part of the lever choice.
2. **K1 sits on the edge of the regressed row.** A negative point with the upper bound
   0.0008 above zero. The registered rule applies as written. The suspects that row had
   named (fresh AdamW moments, seed streams restarting at iteration 1, the training pod)
   are not diagnosed by this round; a continuation arm in p2_i's A/B will read on them.

The open list, carried forward: mate-in-2 near its floor (0.940 here, floor 0.915); pace
(per-forward serving overhead); `value_q_blend: per_row`, the residual, ECE; and, from this
round, L2's exclusion at storage time. RUNLOG-p2_g §6's item 1 (this box's training stack)
was closed by the bridge (BRIDGE-2026-10-25). The lever is Michael's choice.

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_h/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the run |
| `runs/p2_h/ladder.jsonl` | ladder passes at 10..60, measured on this box |
| `runs/p2_h/reps_iter00{10..60}_r{1,2,3}.jsonl`, `p2_verdict_reps{_iter00NN,}.json` | K1–K3 |
| `runs/p2_h/startpos/` | K4 |
| `runs/p2_g_bridge214/`, `runs/p2_g_bridge214/startpos/` | the start references (book; Step 0) |
| `runs/p2_h_campaign.console.log`, `runs/p2_h.console.log`, `runs/p2_h_step0.console.log` | consoles (untracked) |
