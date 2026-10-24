# RUNLOG — p2_f (the replay ring carried across the round boundary)

**Run:** `runs/p2_f`, 20 iterations, 2026-10-23 17:44 → 2026-10-24 05:49 BST, from
`fe51064`, warm-started from `p2_e/0020` with the league **and the replay ring**
continued from `runs/p2_e` (`--seed-replay`, 444,741 rows). Config identical to p2_e's
run config (`test_p2f_config_is_the_brief`). One `gpu hold` covered Step 0, the run
and the F1–F3 replays (`runs/p2_f_queue.console.log`).
**Pre-registration:** `PREREG-p2_f.md`, registered at `fe51064` before Step 0 ran.

**Verdict: H_ring SUPPORTED. F1 REMOVED, F2 PASSES. The mid-round dip was the
replay ring's fill phase; carried across the boundary, the round climbed from its
first pass.**

---

## 1. Step 0 — p2_e's own trajectory at three reps

| p2_e checkpoint | reps | mean | vs p2_e's start (0.476) | registered |
|---|---|---|---|---|
| 0005 | 0.32, 0.3075, 0.37 | 0.3325 | −0.143 [−0.192, −0.095] | dip ✓ |
| 0010 | 0.30, 0.2625, 0.39 | 0.3175 | −0.158 [−0.201, −0.114] | dip ✓ |
| 0015 | 0.3825, 0.455, 0.405 | 0.414 | −0.062 [−0.105, −0.019] | "includes zero": **missed** |

Launch gate passed. The 0015 miss is informative rather than damaging: p2_e's
pass-15 ladder replay (0.465) was a high single draw, and at three reps p2_e never
recovered — 0.476 → 0.333 → 0.318 → 0.414 → 0.408. p2_f therefore did not start in a
trough relative to p2_e/0015 (F3's registered caveat). A 4-minute accidental
36.5 GB VRAM load (idle, GPU busy unchanged) fell inside p2_e/0005 rep 2; annotated in
`runs/p2_e_midrun/annotations.jsonl`, no effect (that rep ran faster than rep 1).

## 2. L — the lever applied, and its signature

- **L1 holds:** `provenance.json` `seed_replay` = 444,741 rows, committed 1,130,902,
  head 1,186,161, offset −20 — the dry run's numbers exactly.
- **L2 holds:** iteration 1 `replay_size` 500,000 = min(444,741 + 55,272, 500,000).
- **L3 holds:** mean policy loss over iterations 1–5 **1.564** (registered ≥ 1.45;
  p2_e 1.319). Iteration 1 read 1.578, p2_e's iteration 20 read 1.581: training saw no
  round boundary.

## 3. F1–F3 — the trajectory (three reps each, against the start p2_e/0020 = 0.4075)

| checkpoint | reps | mean | vs start | |
|---|---|---|---|---|
| p2_f/0005 | 0.4625, 0.4725, 0.45 | 0.4617 | **+0.054 [+0.008, +0.102]** | above the start |
| p2_f/0010 | 0.49, 0.4625, 0.5025 | 0.4850 | **+0.078 [+0.029, +0.127]** | above the start |
| p2_f/0015 | 0.4525, 0.4825, 0.445 | 0.4600 | +0.053 [+0.007, +0.099] | above the start |
| p2_f/0020 | 0.5275, 0.5075, 0.455 | 0.4967 | **+0.089 [+0.038, +0.140]** | F2: CI above zero |

**F1 — Removed** (both 0005 and 0010 not below zero and above −0.054; in fact both
significantly above zero). **F2 — passes** (+0.089, registered ≈ +0.06). Side by side,
each relative to its own start:

```
p2_d (empty ring)    0 → −0.108  −0.107  +0.036  +0.059
p2_e (empty ring)    0 → −0.143  −0.158  −0.062  −0.068
p2_f (carried ring)  0 → +0.054  +0.078  +0.053  +0.089
```

## 4. The rest of the registration

| | registered | measured | |
|---|---|---|---|
| **N** mate-in-2 ≥ 0.915 at pass 20 | holds | **0.945** (189/200); passes 190, 191, 191, 189 | holds |
| **ECE** < 0.15 at pass 20 | holds, no direction | **0.0742**; passes 0.0882, 0.0526, 0.0982, 0.0742 | holds |
| **M4** CI entirely above 0.675 | not expected | 0.4967 | not met |
| H1 prior entropy < 0.05 at it 20; book H ≤ 1.575 | collapse persists | 0.0136; 1.249, 1.279, 1.231, 1.293 | holds |
| H2 skipped steps < 10 | — | 1 (iteration 8) | holds |
| H3 value loss by halves | direction-free | 0.336 → 0.323 (p2_e rose 0.321 → 0.355) | reported |
| H4 decisive rate | direction-free | 0.770 → 0.793 (halves 0.777 → 0.780) | reported |
| H5 promotion loss | reported | halves 0.317 → 0.337 | reported |

Rungs: random 1.0 at every pass; greedy 0.9975 / 0.9975 / 0.9925 / 1.0; phase1 1.0 /
0.98 / 0.965 / 0.995. mate-in-3 136, 137, 140, 140 /200. Single ladder skill-0 replays
0.4475, 0.475, 0.43, 0.505. Pace 31–34 min per iteration, 1,502–1,553 sims/s, steady.

## 5. Decision

Per the table's first row: **ring carry becomes standing protocol** for every
warm-started round, like `--seed-pool`. p2_g takes one lever from the open list.

## 6. What p2_g is built on (found after the verdict)

Michael chose (2026-10-24) one long continuous round — trajectory judged every ten
iterations — trained on a RunPod GPU with the measurements run on this box, and
delegated the lever "to whichever has the best chance of working". Two measurements
on p2_f's ring decided it:

- **None of the 100 openings the skill-0 rung plays from appears anywhere in p2_f's
  500,000-row ring** (position identity: squares, side to move, castling, en passant).
- **Self-play's openings are a funnel:** across the ring's 4,641 games, 2 distinct
  positions after one ply, 6 after two, 39 after four, 285 after eight — the move-1
  prior collapse (H1, every round since p2_c) seen in the data rather than the prior.

The instrument measures strength exclusively from positions the network never trains
on. p2_g's lever is diverse self-play openings; `PREREG-p2_g.md` registers it.

## 7. Artefacts

| file | what |
|---|---|
| `runs/p2_f/iterations.jsonl`, `ladder.jsonl` | 20 iterations; passes 5/10/15/20 |
| `runs/p2_f/reps_iter00{05,10,15,20}_r{1,2,3}.jsonl`, `p2_verdict_reps{_iter0005,_iter0010,_iter0015,}.json` | F1–F3 |
| `runs/p2_e_midrun/` | Step 0, with its annotation |
| `runs/p2_f/config.yaml`, `provenance.json` | config, launch SHA, the `seed_replay` record |
| `runs/p2_f_queue.console.log`, `runs/p2_f.console.log` | consoles (untracked) |
