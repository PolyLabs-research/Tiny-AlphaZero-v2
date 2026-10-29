# RUNLOG — p2_j (p2_i_cont continued, forty more iterations, from p2_i_cont/0020)

**Run:** `runs/p2_j` (`configs/p2_g.yaml` unchanged), 40 iterations, trained on RunPod pod
`q8qk191iy7jg29`, a secure RTX 4090 of the 48 GB variant (`vram_gb` 47.37) on a 35-CPU quota, from
2026-10-28 10:23:07 to 17:23:17 UTC, code `9e55a6a`, warm-started from `p2_i_cont/0020` with the league
and the replay ring continued from `runs/p2_i_cont`. Every measurement ran on this box (torch
2.14.1+rocm7.2), pulled and queued by `cloud/remote_campaign.sh` with `MEASURE_AFTER_PULL=1`, and
finished by `runs/p2_j_finish.sh`. This is **attempt 2**: attempt 1 (pod `8lltzivl3qys40`, 2026-10-27)
was lost to a host freeze after 21 iterations (§5) and is kept as `runs/p2_j_attempt1`; none of its
readings enters a verdict below.
**Pre-registration:** `PREREG-p2_j.md` (`d94b30c`), Addendum A (`9e55a6a`: attempt 1 lost, the re-run,
measurement deferred until the final pull). Every verdict below was registered before it was measured,
except where marked.

**Verdict: H_cont NOT SUPPORTED — the registered Plateau row. K1: p2_j/0040 against p2_i_cont/0020,
six reps each side, −0.027 [−0.062, +0.006]; the interval includes zero and the point is below the
start, outside the forecast +0.03 to +0.08. K2: no dip (all four intervals include zero; slope +0.001
per ten iterations). K3: M4 met at every checkpoint. K4: −0.010 [−0.048, +0.029]. N, ECE hold; L1
holds; L2 reported (2 rows). Unregistered: the move-1 prior entropy at the initial position rose from
0.09 to 0.63.**

---

## 1. Before launch

The start reference is p2_i_cont/0020's six book reps on torch 2.14.1 (0.825, 0.7975, 0.805, 0.7625,
0.8225, 0.7775; mean 0.7983; M4 0.798 [0.777, 0.820]) and its three initial-position reps (0.8475,
0.7975, 0.82; mean 0.8217); nothing was re-measured. Its ladder row at 20: mate-in-2 187, ECE 0.0537.

**Attempt 2's launch** (Addendum A): read 2026-10-28 09:51 UTC, balance $62.76 after Michael's $50
top-up, no pods of ours, secure RTX 4090 at $0.89/h; 12 h × $0.89 = $10.68 worst case. The pod came as
the 48 GB variant (attempt 1's was the 24 GB card, `vram_gb` 23.52), 36 vCPU / 35-CPU quota, 143 GB RAM,
torch 2.12.1+cu129, Stockfish 16-1build1. Staging hit one transient "no public SSH endpoint yet" from the
API mid-copy; the inputs were copied again with retries and all 50 files' sha256 matched on the pod
before the job started.

## 2. L — the continuity applied

- **L1 holds:** `config_fingerprint` `7563c9f7…abccc` = p2_g's; `init_from`
  `runs/p2_i_cont/ckpt_iter0020.pt`; `seed_replay.rows` 500,000, `donor_committed` = `donor_head` =
  1,551,304, `iteration_offset` −20; `git_dirty` false; `git_sha` `9e55a6a` (Addendum A: the launch
  bundle is the addendum's commit, not `d94b30c`), the same on all 40 iteration rows. Constant lr 2e-4.
- **L2 (reported, no bound):** **2 rows** of the final ring (500,000 rows, `total_written` 2,566,550)
  equal a book position, both `open_027` (iterations 31 and 38), by `scripts/ring_book_rows.py`
  (`runs/p2_j/book_rows.json`). Rows at the initial position: 546; distinct positions 482,137. Predicted
  0–5. p2_h 4, p2_i_cont 2, p2_i_row 3.

## 3. K — the endpoint and the trajectory

### K1 (primary) — p2_j/0040 against the start, six reps each side

Skill-0 from the frozen book, per-opening rep-means, paired bootstrap over the 100 openings
(`scripts/eval_reps.py verdict`, 10,000 resamples, seed 0; `runs/p2_j/p2_verdict_reps.json`,
reproduced byte for byte from the rep rows):

| | reps r1–r6 | mean |
|---|---|---|
| start (p2_i_cont/0020) | 0.825, 0.7975, 0.805, 0.7625, 0.8225, 0.7775 | 0.798 |
| p2_j/0040 | 0.7625, 0.795, 0.7575, 0.725, 0.7925, 0.795 | 0.771 |
| **0040 − start** | | **−0.027 [−0.062, +0.006]**, SE 0.0173 |

- **The interval includes zero: the Plateau row.** The forecast (CI entirely above zero, point +0.03 to
  +0.08) failed on both counts; the point is on the negative side. Per opening, 0040's rep-mean is below
  the start's on 46 of the 100, above on 42, equal on 12 (median 0.000).

### K2 — the trajectory (three reps per checkpoint against the six-rep start)

| checkpoint | reps | mean | vs start [95% CI] | M4: mean [95% CI] |
|---|---|---|---|---|
| 0010 | 0.775, 0.7325, 0.81 | 0.773 | −0.026 [−0.060, +0.009] | 0.773 [0.745, 0.800] met |
| 0020 | 0.7925, 0.77, 0.775 | 0.779 | −0.019 [−0.053, +0.013] | 0.779 [0.752, 0.807] met |
| 0030 | 0.775, 0.78, 0.83 | 0.795 | −0.003 [−0.037, +0.030] | 0.795 [0.766, 0.823] met |
| 0040 | 0.7625, 0.795, 0.7575 | 0.772 | −0.027 [−0.071, +0.014] | 0.771 [0.748, 0.794] met (six reps) |

- **No dip:** no checkpoint's interval lies entirely below zero. Least-squares slope **+0.0013 per ten
  iterations**, against p2_i_cont's +0.056 (predicted: smaller; it is, by being flat).
- **Against p2_h/0060** (*computed for this record, not a registered file*: p2_h/0060's three book reps
  against p2_j/0040's six, same statistic): **+0.075 [+0.035, +0.114]**. The trajectory p2_h/0060 →
  p2_i_cont/0020 → p2_j/0040 reads 0.697 → 0.798 → 0.771: most of p2_i_cont's gain over p2_h stands; the
  forty iterations after it added none.

### K3 — M4 holds

At 40, six reps: **0.771 [0.748, 0.794]**, entirely above 0.675 (`runs/p2_j/p2_m4_iter0040.json`). Met at
10, 20 and 30 as well (table above; three reps each). Predicted: met at every checkpoint. Holds.

### K4 — informational: the initial position

p2_j/0040's three reps 0.78, 0.8175, 0.8375 (mean 0.812) against p2_i_cont/0020's 0.822: **−0.010
[−0.048, +0.029]** (`runs/p2_j/startpos/p2_verdict_reps.json`).

*Not registered, for the record:* the ladder pass's single skill-0 replay (`role=protocol`): 0.7875 at
10, 0.7825 at 20, 0.8075 at 30, 0.80 at 40.

## 4. N, ECE, H

- **N — holds:** mate-in-2 **190, 189, 187, 188** at 10/20/30/40 (≥ 183 at 40; predicted 183–190).
  mate-in-3 148, 154, 142, 138. Of the watchlist's capability cases, `mate2_129` and `mate2_166` are
  unsolved at every checkpoint, `mate2_165` solved from 20 on. Failures at 40 (12): `mate2_084`, `091`,
  `124`, `125`, `126`, `127`, `128`, `129`, `138`, `139`, `166`, `196`.
- **ECE — holds:** 0.0543, 0.0566, 0.0560, **0.0604** (< 0.15; predicted 0.04–0.07).
- **H1 (direction-free) — the move-1 prior entropy rose steadily:** `move1_prior_entropy`
  (initial-position games) 0.087 at iteration 1, 0.141 at 10, 0.245 at 20, 0.493 at 30, **0.630 at
  40**, close to monotone over the run. p2_i_cont read 0.03–0.08 over its twenty, p2_h 0.013 at 60.
  K4 did not move with it (−0.010). *Unregistered comparison:* attempt 1, the same config, init, seed
  and protocol on the 24 GB card, read 0.05–0.14 through its 21 iterations (0.07 at 20, where attempt 2
  read 0.24), so the rise is not forced by the protocol: two runs of it diverged on this statistic within
  twenty iterations.
- **H2:** 0 skipped steps in 16,000 (< 20).
- **H3–H5** (direction-free, by halves 1–20 → 21–40): value loss 0.2543 → 0.2525, decisive rate 0.797 →
  0.803, promotion loss 0.390 → 0.392, policy loss 1.516 → 1.492.
- **H6** (`book_move1_entropy`, direction-free): 0.812, 0.719, 0.705, 0.727 at 10/20/30/40 (p2_i_cont
  0.921 at 20).

Rungs: random, greedy and phase1 won 200/200 at every checkpoint.

## 5. Mechanics

- **Pace:** 10.50 min per iteration (9.97–11.13), 4,019–4,606 sims/s (mean 4,294), training 51.1 s; 7.0 h
  of training. p2_i's pace on the same variant (10.70–10.86); attempt 1 on the 24 GB card ran 10.37
  (9.48–10.94), so the p2_h → p2_i slowdown tracks the host, not the variant.
- **Attempt 1, lost (Addendum A).** Pod `8lltzivl3qys40` launched 2026-10-27 17:32:55 UTC (`d94b30c`, L1
  held, 24 GB card), driver measuring as it pulled. This box froze silently at about **22:04 UTC** (the
  journal ends mid-stream with no kernel warning; the second such freeze, both under chess compute on this
  box) during the 0020 measurement — after the ladder pass and book rep 1 (0.800), in rep 2 — and stayed
  down until Michael restarted it at 08:55 on 10-28. A measurement blocks the driver's pulls, so the last
  pull was iteration 21 (21:18). With nothing pulling, the pod's job trained on, waited its `PULL_WAIT`
  (4 h) and deleted itself at about 04:40 UTC, as designed: the cost guards held, and checkpoints
  0022–0040, the ring and the log past 21 went with the pod. Attempt 1 measured 0010 only: reps 0.7675,
  0.7775, 0.7525 (mean 0.766), −0.0325 [−0.065, −0.000] against the start (the upper end is zero to
  floating point; `excludes_zero` true), mate-in-2 190, ECE 0.061; and at 0020 the ladder pass (mate-in-2
  194, mate-in-3 148, ECE 0.056, book entropy 0.720). PREREG-p2_j's "losing the final pull" clause had
  assumed this box alive to pull every five minutes.
- **The fix:** `MEASURE_AFTER_PULL=1` (`cloud/remote_campaign.sh`, `9e55a6a`; opt-in, default unchanged)
  measures nothing until the final pull, ring included, is complete, so this box only pulls while the pod
  holds the only copy, and the heavy local load comes after the data is home. The cost is the interim
  readings. On 10-28 Michael also set `kernel.softlockup_panic=1`, `hardlockup_panic=1`, `panic=10`
  (proposed 2026-10-23): a detected lockup now panics into pstore and reboots. Whether these freezes are
  detectable lockups is not known.
- **Lifecycle (attempt 2):** launched 09:56:26; job started 10:23:07; driver and finisher armed 10:27:55;
  iteration 40 committed 17:23:17; final pull with the ring complete 17:26:53, `PULLED` written, the pod
  deleted itself (the next status read: pod not found, no pods of ours). Measurements under `gpu hold`: 0010 17:27–18:02,
  0020 18:02–18:37, 0030 18:37–19:12, 0040 with the initial-position reps 19:12–20:07; the finisher ran
  reps r4–r6 at 0040 20:08–20:33 and the verdict at 20:33:19 UTC. No retries.
- **Cost:** attempt 1 **$9.91** ($22.67 → $12.76: ~11.0 h at $0.898/h, training plus the 4 h wait);
  attempt 2 **≈ $6.69** (7.5 h at the pod's $0.89/h; the account balance is not the measure from 12:42 on,
  when another pod of Michael's shared it). p2_j $16.60; all RunPod use to date about $43.93.

## 6. Decision

Per the table's second row (CI includes zero): **Plateau, read for the first time against a six-rep
start. p2_k is the learning-rate step-down as an A/B from p2_j/0040 against a continuation arm** (the
candidate PREREG-p2_i named). N and ECE hold at 0040, so the fourth row does not apply. Any launch is a
new PREREG and Michael's cost OK.

Readings the table does not cover:

1. **p2_i_cont's climb did not continue.** Twenty iterations from p2_h/0060 read +0.113; forty more from
   p2_i_cont/0020, with the same restart (fresh AdamW, seed streams from iteration 1), read −0.027 and a
   flat trajectory. So "the restart moves the model up" (RUNLOG-p2_i §6 item 1) does not repeat as a
   rule. The start reference was the highest book score measured on this stack and was chosen as the
   start *because* it climbed; part of the shortfall may be a high draw at the start (its ladder replay
   read 0.7675 against its reps' 0.798). p2_j/0040 still reads +0.075 [+0.035, +0.114] over p2_h/0060.
2. **The move-1 prior is opening up at the initial position** (H1, 0.09 → 0.63) while the
   initial-position score is flat and the book entropy fell (0.92 → 0.73). Attempt 1 did not show it
   through iteration 21, so it is a run-to-run divergence of the same protocol, not its fixed course.
   Whether it matters for skill is not in this record; a continuation arm in p2_k will read it again.
3. **Attempt 1 is a partial replicate.** Same protocol, the other card: its 0010 (−0.0325) and attempt
   2's (−0.026) agree; its 0020 mate-in-2 read 194 against attempt 2's 189.

The open list, carried forward: p2_k (the LR step-down A/B); H1's divergence; mate-in-2 near its floor
(187–190, floor 183); the host freezes; pace; the residual; ECE; L2's exclusion at storage time.

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_j/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the run |
| `runs/p2_j/ladder.jsonl` | ladder passes at 10, 20, 30 and 40, measured on this box |
| `runs/p2_j/reps_iter00{10,20,30}_r{1,2,3}.jsonl`, `reps_iter0040_r{1..6}.jsonl`, `p2_verdict_reps_iter00{10..40}.json` | K2 |
| `runs/p2_j/p2_verdict_reps.json` | K1 (six against six; the finisher's file replaces the driver's three-rep copy) |
| `runs/p2_j/p2_m4_iter0040.json` | K3 at 40 |
| `runs/p2_j/startpos/` | K4 |
| `runs/p2_j/book_rows.json` | L2 |
| `runs/p2_j_attempt1/` | attempt 1: iterations 1–21, the 0010 measurement, the 0020 ladder pass and rep 1 |
| `runs/p2_j_campaign.console.log`, `runs/p2_j.console.log`, `runs/p2_j_finish.console.log`, `runs/p2_j_finish.sh`, `runs/p2_j_attempt1/console/` | consoles and the finisher (untracked) |
