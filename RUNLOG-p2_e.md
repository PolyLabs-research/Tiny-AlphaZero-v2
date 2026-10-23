# RUNLOG — p2_e (the null re-baseline on the fixed code)

**Run:** `runs/p2_e`, 20 iterations, launched 2026-10-22 15:12 BST from `0913957`,
warm-started from `p2_d/0020` with the league seeded from `runs/p2_d`, config
resolving to p2_d's (`test_p2e_config_is_the_brief`). Interrupted by a host hard
stop at 00:25 on 2026-10-23, resumed at 10:04 from `dc51296` (PREREG-only delta),
iteration 20 committed about 14:16. No lever.
**Pre-registration:** `PREREG-p2_e.md` + Addendum 1 (the bridge and the skill-0
instrument) + Addendum 2 (continuity, M4 restated) + Addendum 3 (the regressed row
and Control C, written after passes 5 and 10 were seen and before any P2e or
Control C data existed). Every verdict below was registered before it was measured.

**Verdicts: P2e REGRESSED. Control C DIPPED, at least as deep. Per Addendum 3 the
endpoint lies within the trough the old code also reaches: protocol, not code. N
holds at 0.970. M4 not met. ECE inside its clause, worse than the start.**

---

## 1. Step 0 — the bridge, and the instrument it exposed

Recorded in full in Addenda 1 and 2; summarised here. On p2_d's own endpoint, the
fixed code reproduced every deterministic instrument to the digit (mate-in-2
183/200 with the same 17 failures, mate-in-3 129/200, the three seeded rungs
including `pair_scores` element for element). Skill-0 did not: 0.476 against the
recorded 0.303. Attribution by restoring the pre-fix engine call under today's code
(0.275, 0.285): the shift is the Stockfish transposition table, which the P9 fix
clears on every game switch and the old call shared across a whole matchup. The
cleared-table instrument was kept, p2_c/0020 re-measured on it (0.417), and M4
restated to a CI entirely above 0.675.

## 2. P2e — regressed

Judged on the registered statistic: three replays per endpoint, per-opening
rep-means, paired bootstrap over the 100 openings.

```
bridge p2_d/0020 reps:  0.435, 0.510, 0.4825     mean 0.4758
p2_e/0020 reps:         0.405, 0.385, 0.4325     mean 0.4075
                        (65W/32D/103L, 61W/32D/107L, 66W/41D/93L)

paired difference          −0.0683
95% CI                     [−0.1150, −0.0208]
SE                         0.0237        effect/SE  −2.88
```

**The CI lies entirely below zero.** Twenty iterations on the fixed code, at p2_d's
configuration, ended below where they started. The registered expectation (CI above
zero, point ≈ +0.10, restated as ≈ +0.06 on the kept instrument) missed.

## 3. Control C — the old code dips too

Added by Addendum 3 before it produced data, and run whatever P2e's outcome: p2_d's
mid-run checkpoints on the cleared-table instrument, three replays each, against
p2_d's start (p2_c/0020).

| checkpoint | reps | mean | vs start 0.417 | |
|---|---|---|---|---|
| p2_d/0005 | 0.315, 0.320, 0.290 | 0.308 | **−0.108 [−0.158, −0.059]** | below zero |
| p2_d/0010 | 0.340, 0.2875, 0.3025 | 0.310 | **−0.107 [−0.159, −0.055]** | below zero |
| p2_d/0015 | 0.470, 0.4525, 0.435 | 0.4525 | +0.036 [−0.011, +0.083] | includes zero |
| p2_d/0020 | (the bridge) | 0.476 | +0.059 [+0.010, +0.108] | above zero (Addendum 2) |

**Dip: yes. Depth: −0.108.** On the shared-table instrument these same checkpoints
read 0.180, 0.215 and 0.193 against a start of 0.200 — no dip of this size. The two
instruments disagree about the shape of p2_d's run, which is Addendum 2's point
about a non-constant offset made concrete.

### The decision, read off the table

P2e regressed (−0.068), and Control C's depth (−0.108) is at least as deep. Addendum
3's row: *the endpoint lies within the trough the old code reaches* — **protocol,
not code. No lever and no A/B.** The fix pass is not implicated by this evidence.

## 4. The shape, now three rounds

Skill-0 on the cleared-table instrument (three-rep means unless marked; `*` is a
single ladder replay):

```
p2_d   0.417 → @5 0.308  @10 0.310  @15 0.4525  @20 0.476
p2_e   0.476 → @5 0.338*  @10 0.353*  @15 0.465*  @20 0.4075   (ladder @20: 0.4325*)
```

Both rounds dip by about 0.11–0.14 through pass 10 and recover by pass 15. p2_d then
kept rising; p2_e fell back. A single 20-iteration endpoint sits on a trajectory
whose swing (~0.12) is twice the slope it is trying to measure (+0.059), which is
why Addendum 3 sends this to protocol.

### A candidate mechanism, found after the verdict (not registered)

**Every round starts with an empty replay ring.** Capacity is 500,000 positions;
an iteration generates ~56,000 and trains on 400 × 1024 = 409,600 draws. So
iteration 1 draws each stored position ~7–8 times, and the ring is full only at
iteration 9–10:

| iteration | 1 | 2 | 3 | 5 | 8 | 9–10 | 15, 20 |
|---|---|---|---|---|---|---|---|
| p2_d draws per stored row | 8.37 | 4.20 | 2.79 | 1.65 | 1.01 | 0.89–0.82 | 0.82 |
| p2_e draws per stored row | 6.97 | 3.55 | 2.41 | 1.47 | 0.91 | 0.82 | 0.82 |
| p2_e policy loss | 1.385 | 1.318 | 1.288 | 1.313 | 1.414 | 1.433–1.467 | 1.533, 1.581 |

Both dips sit exactly in the fill phase, and the policy loss is lowest where the
ring is smallest: the round opens by fitting a few tens of thousands of games
several times over. The learning rate is constant (2e-4), so no schedule restart
confounds it. The optimizer is the one other thing a warm start resets: `--init`
loads weights only and builds a fresh AdamW. The mechanism does not explain the
late divergence (p2_d rising 15 → 20, p2_e falling).

## 5. The rest of the registration

| | registered | measured | |
|---|---|---|---|
| **N** mate-in-2 ≥ 0.915 at pass 20 | holds, narrowly (≈ 0.92) | **0.970** (194/200); passes 184, 186, 188, 194 | holds, rising |
| **M4** CI entirely above 0.675 | not expected | 0.4075 | not met |
| **ECE** < 0.15 at pass 20 | "fails again" (drafted on the old instrument; read against the bridge, Addendum 1) | **0.1267**; passes 0.1255, 0.1192, 0.0626, 0.1267 | inside the clause, worse than the bridge's 0.0798 |
| H1 prior entropy < 0.05 at it 20; book H ≤ 1.575 every pass | collapse persists | 0.0055 (max 0.0088 over the run); 1.206, 1.238, 1.316, 1.228 | holds |
| H2 skipped steps < 10 | — | 4 in 8,000 (iterations 2, 8, 19, 20) | holds |
| H3 value loss by halves | direction-free | 0.321 → 0.355 | reported |
| H4 decisive rate | direction-free | 0.730 → 0.742 (halves 0.765 → 0.748) | reported |
| H5 promotion loss | reported, not compared | halves 0.338 → 0.319 | reported |

mate-in-3: 116, 120, 132, 130 /200 (bridge 129). The watchlist ended with `129` and
`165` solved and `166` a capability miss. The weak deterministic rungs, perfect at
the bridge, dropped a game here and there from pass 10: greedy 1.000 / 0.9875 /
0.995 / 0.99, phase1 0.99 / 0.975 / 0.995 / 0.9925.

ECE moved by a factor of two between consecutive passes (0.063 at 15, 0.127 at
20). Its 5,000 positions are drawn from the nondeterministic skill-0 games, and
that spread has still never been measured; B4 declined a tolerance for this reason.

## 6. Incidents

1. **Iteration 3 ran at 637 sims/s** (16:22–17:34, 72 min) against ~1,500 for
   every other iteration and for p2_d. No kernel, thermal, login or process event in
   the window; cause unidentified. Wall clock only: self-play is
   batching-independent by the A1 test, so the data are unaffected.
2. **Host hard stop at 00:25:07–00:26:01 BST** during iteration 15's self-play. The
   box was found frozen. The journal ends mid-stream with no OOM, amdgpu, thermal,
   MCE, lockup or shutdown record and no pstore dump; `sar` shows 16 GB available,
   flat swap and no blocked tasks up to 00:20. Cause unknown. ~6 minutes of
   uncommitted self-play lost. Resumed 10:04 from `dc51296`; `state.json` records
   `resumed_with: ['dc51296']`; annotated (`runs/p2_e/annotations.jsonl`). RNG is
   derived per iteration, so iteration 15 was regenerated, not continued.
3. **Iteration-21 overrun.** The resume passed `--iterations 20`, which
   `train_phase2.py` reads as iterations *this invocation* (its help says so;
   `HOW_TO_TRAIN.md`'s example read as a total and is now corrected). After the
   pass-20 ladder it began iteration 21. SIGINT was ignored (a background-launched
   process inherits SIGINT ignored); stopped with SIGTERM at 14:30:19, ~14 minutes in.
   **Nothing past 20 was committed**: `state.json`, `iterations.jsonl`, the ladder and
   the last checkpoint all stop at 20. The ring's durable head had advanced: 55,259
   rows tagged iteration 21 sit past `replay_total_written` (1,130,902), overwriting
   the oldest committed slots. Measured by the ring's own `iteration` column. Annotated.

## 7. Decision

Addendum 3's row is "protocol, not code". Of its options, **Michael chose on
2026-10-23 to carry the replay ring across rounds**, pre-registered as **p2_f**:
warm-start from `p2_e/0020` with the league *and* the ring continued from
`runs/p2_e`, so a round boundary stops resetting what training samples. It tests §4's
mechanism directly: if the dip is the fill phase, p2_f should not have one.

## 8. Artefacts

| file | what |
|---|---|
| `runs/p2_e/iterations.jsonl`, `ladder.jsonl` | 20 iterations; passes 5/10/15/20 |
| `runs/p2_e/reps_iter0020_r{1,2,3}.jsonl`, `p2_verdict_reps.json` | P2e |
| `runs/p2_d_midrun/reps_iter00{05,10,15}_r{1,2,3}.jsonl`, `p2_verdict_reps_iter00{05,10,15}.json` | Control C |
| `runs/p2_e_bridge/`, `runs/p2_c_bridge/` | Step 0 and continuity (Addenda 1–2) |
| `runs/p2_e/annotations.jsonl` | the host stop and the overrun |
| `runs/p2_e/config.yaml`, `provenance.json` | launch config and SHA |
| `runs/p2_e.console.log`, `runs/p2_e_post.console.log` | consoles (untracked) |
