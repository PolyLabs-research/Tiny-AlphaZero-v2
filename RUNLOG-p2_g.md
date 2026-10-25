# RUNLOG — p2_g (diverse self-play openings, one long round, trained on RunPod)

**Run:** `runs/p2_g`, 60 iterations, trained on RunPod pod `7ovptr9vaaeb2z` (secure RTX
4090, 24 vCPU / quota 20.4, EPYC 7352) from 2026-10-24 18:07 to 2026-10-25 04:36 UTC,
code `0939a39`, warm-started from `p2_f/0020` with the league and the replay ring
continued from `runs/p2_f`. Every measurement ran on this box (ROCm), pulled and queued
by `cloud/remote_campaign.sh`.
**Pre-registration:** `PREREG-p2_g.md` (`191c4dd`) + Addendum 1 (the GPU became an RTX
4090) + Addendum 2 (Step 0 results; the training-step bridge failed its gate) +
Addendum 3 (launch on CUDA as a recorded deviation from that gate, Michael's decision).
Every verdict below was registered before it was measured, except where marked.

**Verdict: H_open SUPPORTED. G1 PASSES (+0.243), G2 "adds", G3 no dip, G4 says the
gain is not only the book's distribution. M4 IS MET at iterations 50 and 60. N holds,
narrowly.**

---

## 1. Before launch

**Step 0a** (this box): p2_f/0020 from the initial position, three reps — 0.66, 0.6675,
0.6375 (mean 0.655), against 0.4967 from the book. The network was already much
stronger in the opening it trains on than in the positions the instrument uses.

**Step 0b failed its gate** (pod `h10w6nqxio60zt`, RTX 4090): the search bridge matched
exactly (mate-in-2 189, mate-in-3 140) and a two-iteration shakedown committed, but the
training-step bridge missed its tolerance (gradient norm 55% off on batch 0, 12.3% on
batch 6; losses within 0.05%). Nothing launched; the pod was terminated ($0.65).

**The diagnosis, measured after the gate** (`FINDING-2026-10-24-rocm-layernorm-backward.md`):
against a CPU fp32 reference on the gate's own batches, the pod's CUDA gradients were
within 1.3% on every batch and this box's ROCm gradients were not. The cause is
pytorch/pytorch#199265 — on wave32 RDNA, torch 2.12.1's ROCm LayerNorm backward never
writes the odd-indexed elements of `weight.grad` / `bias.grad` for 128–65,536 rows. In
this model that is the value head's LayerNorm: every round trained on this box (Phase 1,
p2_a–p2_f) trained those 512 parameters on half-missing gradients, and where the error
dominated the norm, clipping shrank every other update with it. Fixed upstream in torch
≥ 2.13; torch 2.14.1+rocm7.2 verified correct on this gfx1151. Measurement is forward-only
and unaffected.

**The pace, fixed before launch.** The old actor protocol made one round trip per
network per flush; under a four-opponent league that was up to five. The hub (`f118680`)
sends one message per flush for every network: byte-identical games (reference
fingerprint, in-process and parallel), +22% on the pod's A/B (6,300 → 7,672 sims/s on
the p2_g workload capped at 80 plies), and a real iteration on the run's pod in 10.6 min
against 14.4 before. A server gather window (`0939a39`) made things slower and stayed
off. The registered config was not touched.

**Addendum 3** recorded the decision to launch on CUDA anyway, with a third confound for
G2 named before any data: p2_f trained its value-head LayerNorm on broken gradients and
p2_g does not.

## 2. L — the lever and the continuity applied

- **L1 holds:** `provenance.json` `seed_replay` rows 500,000, donor committed = head =
  1,556,008, offset −20; config records `opening_random_plies: 8`, 20 actors, no inline
  ladder, 60 iterations.
- **L2 holds:** rows of p2_g's final ring equal to any of the book's 100 positions: **0**.
  The ring holds 483,149 distinct positions in 500,000 rows (p2_f's games reached 39
  distinct positions four plies in, across 4,641 games).

## 3. G — the trajectory

Skill-0 from the frozen book, three reps per checkpoint, per-opening rep-means, paired
bootstrap over openings, against the start reference p2_f/0020 (0.5275, 0.5075, 0.455;
mean 0.4967):

| checkpoint | reps | mean | vs start [95% CI] |
|---|---|---|---|
| 0010 | 0.6075, 0.605, 0.6675 | 0.627 | +0.130 [+0.079, +0.181] |
| 0020 | 0.6725, 0.6775, 0.69 | 0.680 | +0.183 [+0.132, +0.238] |
| 0030 | 0.65, 0.7225, 0.66 | 0.678 | +0.181 [+0.135, +0.228] |
| 0040 | 0.6625, 0.6925, 0.6775 | 0.678 | +0.181 [+0.131, +0.231] |
| 0050 | 0.7525, 0.69, 0.7175 | 0.720 | +0.223 [+0.170, +0.275] |
| **0060** | 0.745, 0.75, 0.7225 | **0.739** | **+0.243 [+0.194, +0.290]** |

- **G1 — passes.** CI entirely above zero; point +0.243 against the registered ≥ +0.15.
- **G2 — adds.** p2_g/0020's +0.183 is above the top of p2_f's twenty-iteration interval
  (+0.089 [+0.038, +0.140]). Directional, as registered: the rounds start from different
  checkpoints and train on different stacks, and p2_g's value-head LayerNorm receives
  correct gradients where p2_f's did not. The lever and the corrected gradients are not
  separable in this round.
- **G3 — no dip.** Every checkpoint is significantly above the start. The shape: most of
  the gain in the first twenty iterations, a plateau through forty, a second rise at
  fifty and sixty; least-squares slope +0.020 per ten iterations over 10–60.
- **G4 — the gain is not only distribution-matching.** From the initial position
  (`tests/suites/openings_startpos.epd`), three reps: p2_g/0060 0.7625, 0.74, 0.81 (mean
  0.771) against p2_f/0020's 0.655: **+0.116 [+0.073, +0.161]**. By the registered reading,
  the network is stronger where it was already trained, not only on the book's positions.

## 4. M4, N, ECE, H

- **M4 — met at 0050 and 0060.** Mean-over-reps book score with its 95% CI entirely above
  0.675 (PREREG-p2_e Addendum 2): 0050 **0.720 [0.686, 0.753]**, 0060 **0.739 [0.709,
  0.768]**; 0020–0040 sit at 0.678–0.680 with lower bounds 0.643–0.646. *Method, defined
  at scoring because the registration named the bar but not the interval:* per-opening
  rep-means bootstrapped over the 100 openings, 10,000 resamples, seed 0, percentile
  interval — the one-sample counterpart of the registered paired statistic. M4 is the
  cleared-table instrument's equivalent of reliably beating Stockfish skill 0.
- **N — holds, narrowly:** mate-in-2 184/200 = 0.920 at 0060 (floor 0.915). Across the
  judged checkpoints: 184, 190, 187, 187, 188, 184. Closer to the floor than p2_f (0.945).
- **ECE — holds:** 0.059 at 0060 (< 0.15); 0.040–0.059 at every judged checkpoint.
- **H1 — the collapse at the initial position persists, as registered:**
  `move1_prior_entropy` (initial-position games only) 0.026 at iteration 60 (< 0.05),
  though it rose from 0.004 at iteration 1.
- **H2:** 0 skipped steps in 24,000 (< 30). Every ROCm round had some; the CUDA stack
  had none. Consistent with the LayerNorm defect contributing to those skips; not tested.
- **H3–H5** (direction-free): value loss by halves 0.283 → 0.272; decisive rate 0.814 →
  0.766 (halves 0.799 → 0.789); promotion loss by halves 0.351 → 0.363. Policy loss sat
  at 1.56–1.63, above p2_f's 1.53 — the positions are far more varied.
- **H6** (book entropy, direction-free): 1.435, 1.082, 0.869, 0.799, 0.750, 0.903 — the
  prior at the book positions sharpened as they became in-distribution.

Rungs: random, greedy and phase1 won outright at every judged checkpoint (greedy
0.995 once). mate-in-3: 147, 139, 129, 136, 140, 139. Single ladder skill-0 replays:
0.6275, 0.675, 0.75, 0.685, 0.695, 0.735.

## 5. Mechanics

- **Pace:** 10.1–11.0 min per iteration (mean 10.53), 4,032–4,525 sims/s, training 52 s
  per iteration. The serving process spends most of its time in per-forward overhead
  with the GPU ~25% busy — the next pace lever.
- **Lifecycle:** the pod trained, the local driver pulled every five minutes, the final
  pull (ring included) completed at 04:41:30 UTC, `PULLED` was written and the pod
  deleted itself. Six local measurements of 38–58 min each paused llama-server.
- **Cost:** $11.56 for this pod, pre-launch re-test included ($48.98 → $37.42); all RunPod
  use to date $12.58.
- **Incident, harmless:** a second instance of the local driver started at 04:41:41 UTC.
  It was the launch of 19:09 BST, concluded dead at the time: it had chained the driver
  after an `ssh` that started the pod's job in the background, and that `ssh` stayed
  blocked until the pod deleted itself. The second instance found 10–50 measured, waited
  for the GPU lease, re-ran 0060 skipping every row already logged, and recomputed
  identical verdicts (each rep file has one row; the iteration-60 ladder has five; the
  0060 verdict recomputes identically). Never chain a local start after an `ssh` that
  starts remote background jobs.

## 6. Decision

Per the table's first row: **random-opening starts become standing for self-play.**
p2_h takes one lever from the open list. Also open, in order:

1. **This box's training stack.** torch 2.12.1+rocm7.2 is defective (§1); 2.14.1 fixes it.
   Upgrading is a stack change: re-measure one checkpoint before trusting new local
   numbers, as p2_e's bridge did.
2. **mate-in-2 is near its floor** (0.920 against 0.915): the long round's one soft spot.
3. **Pace:** per-forward serving overhead, the GPU ~25% busy.
4. The levers carried from earlier rounds: `value_q_blend: per_row`, the residual, ECE.

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_g/iterations.jsonl`, `config.yaml`, `provenance.json`, `replay/meta.json` | the run |
| `runs/p2_g/ladder.jsonl` | ladder passes at 10..60, measured on this box |
| `runs/p2_g/reps_iter00{10..60}_r{1,2,3}.jsonl`, `p2_verdict_reps{_iter00NN,}.json` | G1–G3 |
| `runs/p2_g/startpos/` | G4 |
| `runs/p2_f/startpos/` | Step 0a |
| `runs/p2_g_bridge/`, `runs/p2_g_shake/`, `runs/p2_g_shake2/` | Step 0b and the hub's shakedown |
| `benchmarks/results.jsonl` | the pod's hub A/B rows |
| `runs/p2_g_campaign.console.log`, `runs/p2_g.console.log` | consoles (untracked) |
