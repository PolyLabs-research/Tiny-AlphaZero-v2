# Pre-registration — p2_e: the null re-baseline on the fixed code

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-10-22** (see Sign-off). The expectations were drafted
> from p2_d's measurements and checked against the records and the scripts before
> signing. No data has been produced under them: Step 0 has not run.

**Amendment policy.** A rule may be added after launch *only* for a measurement
that has not yet produced any data, and only as a dated addendum recording the run
state at the time of writing. A rule may never be added, weakened, or reworded for
a measurement already taken.

---

## The question

`REVIEW-2026-10-21-full-codebase.md` and its fix pass changed the code under an
unchanged configuration: what the replay ring stores (C4), how the search orders
proven wins and schedules roots (A1–A6, D15), and how records are written. p2_e
is the first round on that code. It adds **no lever**: twenty more iterations at
p2_d's configuration, warm-started from p2_d's endpoint.

It answers two things, in order:

1. **Did the fix pass move the instruments?** Measured offline, before any
   training, on p2_d's own endpoint (Step 0, "the bridge"). Same weights, same
   config, new code. The review predicts almost nothing should move, and that is
   testable at the digit wherever the instrument is deterministic.
2. **Does the loop keep turning on the fixed code?** p2_d moved skill-0 by +0.103
   in twenty iterations at an identical config. p2_e asks whether it still does,
   measured against the bridge, so both endpoints are read through the same
   code. A positive result makes the fixed loop the baseline that p2_f's single
   lever is attributed against.

## Contract

| | |
|---|---|
| Config | `configs/p2_e.yaml`. Resolves to exactly `runs/p2_d/config.yaml`; `test_p2e_config_is_the_brief` asserts zero deltas. The fields the fix pass added are written out at p2_d's behaviour: `value_q_blend: batch_mean`, `attn_init: torch_default`, `phase1_anchor_weighting: legacy`, `proof_directed_expansion: false` |
| Warm start | `runs/p2_d/ckpt_iter0020.pt` |
| League | seeded from `runs/p2_d` |
| Length | 20 iterations, `--ladder-every 5`, rungs `random,greedy,phase1,stockfish:0` |
| Recorded cadence | The ladder flags are now written into a new run's `config.yaml` (C3/D7), so `runs/p2_e/config.yaml` will read `eval.eval_every_n_iters: 5`, `eval.ladder_anchors: random,greedy,phase1,stockfish:0` and `eval.anchor_ckpt: runs/phase1_anchor.pt`, where `runs/p2_d/config.yaml` reads 10, the default three rungs and no anchor. p2_d ran at exactly this cadence and these rungs (its rows at 5/10/15/20 say so) and did not record them. Records only: the ladder is read-only with respect to training, and `test_p2e_config_is_the_brief` compares the launch config, not the run's |
| Judged at | pass 20 |
| Code | `main` at launch, clean tree. `provenance.json`'s `git_sha` is the run's code (the writers capture it once at launch since C2/D2) |
| GPU | the run is launched under `gpu hold` (`HOW_TO_TRAIN.md` Step 4), so llama-server is stopped for its duration; a shared-GPU stretch (ComfyUI used mid-run) is logged in the RUNLOG with its times |
| Stack | torch 2.12.1+rocm7.2, Radeon 8060S, same as p2_d (`provenance.json`) |

## What changed between p2_d and p2_e: the code

| Finding | Touches | Reaches this round? |
|---|---|---|
| **C4** | the replay ring stores the move the game played, not the improved-policy argmax (ring `VERSION 4`) | **Yes.** The promotion head's target cell moves; under self-play noise the two differed on ~95% of positions. Promo loss is therefore not comparable with p2_a–d |
| A3 | `proven_depth` keeps inducting after an edge is proven | Measured inert: 0 stale of 3,611 proven edges over both suites at 128 sims (review §A3). Expected not to move the bridge |
| A1 | a root expanded twice when `batch_leaves ≤ concurrent_games` | No: 512 > 256 |
| A2 | force-include capped at the sim budget | No: unreachable at 48 / 128 sims |
| D15 | a terminal opening adjudicated before anyone moves | No: the book has no terminal position; self-play starts from the initial position |
| A4/C7, A5, A6, P4, P8, P12, P13 | debug-tree retention, test arming, evaluation counting, thread pinning, the lever's scan | No effect on the moves played (P12/P13 sit behind the lever, which is off) |
| C5 | model init seeded | No: `--init` overwrites the weights |
| C1, B3, E1 | value/CE blend, attention init, Phase 1 anchor arithmetic | Pinned to p2_d's behaviour (C1) or inert here (B3, E1) |
| C2/D2, P1, P10, C3/D7, D1, D4–D6, B1/C9 | provenance SHA, `nondeterministic`, per-family schema versions, persisted ladder cadence, torn-tail guard, ladder resume, atomic writes | Records only |
| B6, B7, C10 | ring sidecar checks, a NaN policy row, a resignation tracker that fails to bind | Guards: each now **raises** where it was silent. If one fires, the run stops instead of quietly continuing (C10 used to switch resignation off for that game) |

If the bridge moves anywhere the review measured as inert, that is the finding of
Step 0, and it is reported before p2_e is judged.

---

## Registered expectations

### Bridge (Step 0): `p2_d/0020` re-measured on the fixed code

Deterministic instruments are predicted **identical to the digit** to the rows
recorded at p2_d pass 20 (`runs/p2_d/ladder.jsonl`):

- **B1. Suites.** mate-in-2 **183/200 (0.915)** with the **same 17 failures**
  (`009 035 048 054 084 091 124 126 127 128 129 139 140 141 166 186 196`);
  mate-in-3 **129/200 (0.645)**.
- **B2. Net-vs-net rungs.** random **1.000**, greedy **1.000**, phase1 **0.9975**
  (199 W, 1 D, 0 L).
- **B3. Skill-0 (nondeterministic).** Three replays, verdict against p2_d's
  recorded old-code reps (`runs/p2_d/reps_iter0020_r*.jsonl`, mean 0.3033): the
  paired-difference 95% CI **includes zero**.
- **B4. ECE — reported, no tolerance registered.** Reference **0.1762**. Its
  5,000 positions are drawn from the phase1 and skill-0 games, so the skill-0
  rung's own randomization moves it, and that spread has never been measured on
  ECE. A tolerance here would be a guess dressed as a rule; the bridge value is
  recorded as the first measurement of it.

Any B1/B2 difference means the fix pass changed search or evaluation more than the
review measured. It does not stop the round. p2_e is judged against the bridge
regardless, and the RUNLOG states the size and the positions.

### P2e (primary)

Skill-0 at p2_e pass 20 against the bridge, on the registered statistic: three
replays per endpoint, per-opening rep-means, paired bootstrap over openings.

**Expected: the CI excludes zero above.** Point estimate **≈ +0.10**, p2_d's
slope at the same configuration. Expected shape, seen twice now: flat or below the
start through pass 15, with the gain late. A flat mid-run pass is not read as a
stall.

### M4 trigger (armed, not required)

Mean-over-reps score vs skill 0 with its CI entirely above 0.5. **Not expected**:
0.30 + 0.10 falls short of it. The run continues to 20 regardless.

### N (no-regress)

mate-in-2 ≥ 0.915 at pass 20 under the current (fixed) search. Gates don't move.

**Expected: holds, narrowly (point ≈ 0.92).** p2_d fell 0.945 → 0.935 → 0.915
across its second half and ended exactly at the floor. This is the likeliest
failure of the round, and is registered as such.

### ECE (restated clause)

ECE < 0.15 at pass 20 on the honest sampler. **Expected: FAILS again (≈ 0.17–0.18).**
p2_c 0.1678 → p2_d 0.1762, and nothing in the code delta targets calibration.

### H (reported)

- **H1. The funnel collapse persists (direction registered).**
  `move1_prior_entropy` (`iterations.jsonl`) at iteration 20 < 0.05 (p2_d:
  0.0107), and `book_move1_entropy` (`ladder.jsonl`) at every pass ≤ 1.575
  (p2_c's; p2_d: 1.313, 1.482, 1.346, 1.381). p2_d showed the collapse with
  nothing touched, and the code delta does not touch exploration (root noise,
  considered set, acting rule).
- **H2.** `skipped_steps` summed over the run < 10 (p2_d: 0 in 8,000).
- **H3. `value_loss` by halves — direction-free.** Registered with a direction
  twice, wrong both times.
- **H4. `decisive_rate` — direction-free.** p2_d fell 0.854 → 0.709.
- **H5. `promo_loss` — reported, not compared** with p2_a–d (C4 moved its target).

## Decision rules

| Observation | Decision |
|---|---|
| **Bridge B1/B2 differ** | Recorded as a code effect, with the positions named. p2_e is still judged against the bridge, never against p2_d's recorded rows. |
| **P2e passes, N holds** | The fixed loop is the baseline. p2_f takes **one** lever; candidates are `value_q_blend: per_row` (C1) and p2_d's open items (funnel collapse, the two lever-broken positions, the nine-position residual, ECE). |
| **P2e flat** | The loop did not continue on the fixed code, and C4 is the only change measured to reach training: a BLOCKED conversation on C4 before any lever. Gates untouched. |
| **N or ECE breaks** | Standard stop rules. |

## Measurement protocol

Step 0 runs **after sign-off and before launch**, with the box to itself. The bridge
measures p2_d's checkpoint under p2_d's own run config, so the only thing that
differs from the recorded rows is the code.

```bash
# Step 0a — the bridge ladder pass (suites, three seeded rungs, skill 0, ECE)
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/eval_ladder.py --ckpt runs/p2_d/ckpt_iter0020.pt \
    --config runs/p2_d/config.yaml --anchors random,greedy,phase1,stockfish:0 \
    --anchor-ckpt runs/phase1_anchor.pt --iteration 20 --role diagnostic \
    --out runs/p2_e_bridge/ladder.jsonl

# Step 0b — three skill-0 replays. --out-dir is required: the default is beside
# --ckpt, i.e. runs/p2_d, which holds the recorded old-code reps.
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_d/ckpt_iter0020.pt \
    --anchor stockfish:0 --reps 3 --out-dir runs/p2_e_bridge

# Step 0c — B3: fixed-code reps against the recorded ones
.venv/bin/python scripts/eval_reps.py verdict \
    --a p2_d runs/p2_d/reps_iter0020_r*.jsonl \
    --b p2_d_fixed runs/p2_e_bridge/reps_iter0020_r*.jsonl \
    --out runs/p2_e_bridge/p2_verdict_reps.json

# The run
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/train_phase2.py --config configs/p2_e.yaml \
    --init runs/p2_d/ckpt_iter0020.pt --seed-pool runs/p2_d --out runs/p2_e \
    --iterations 20 --ladder-every 5 \
    --ladder-anchors random,greedy,phase1,stockfish:0 \
    --ladder-anchor-ckpt runs/phase1_anchor.pt

# P2e — three replays at the endpoint, then the verdict against the bridge
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_e/ckpt_iter0020.pt \
    --anchor stockfish:0 --reps 3
.venv/bin/python scripts/eval_reps.py verdict \
    --a p2_d_fixed runs/p2_e_bridge/reps_iter0020_r*.jsonl \
    --b p2_e runs/p2_e/reps_iter0020_r*.jsonl \
    --out runs/p2_e/p2_verdict_reps.json
```

Every record above is reachable by git (`tests/test_gitignore_musttrack.py`
`MUST_REACH`).

## What would make p2_e uninterpretable

- **A config delta.** Guarded by `test_p2e_config_is_the_brief`.
- **Launching from anything but a clean `main`.** `provenance.json` records
  `git_sha` and `git_dirty`; a dirty launch is restarted, not annotated.
- **A shared GPU.** Guarded by `gpu hold` for llama-server. ComfyUI can still be
  used mid-run; if it is, the RUNLOG names the stretch, and no throughput or
  wall-clock comparison is drawn across it.
- **A stack change** (torch, ROCm, driver) between Step 0 and pass 20 (campaign rule 6).

---

## Sign-off

Checked before signing, 2026-10-22 13:13 BST:

- every number above against `runs/p2_d/ladder.jsonl` (pass-20 rows: suites,
  the 17 failures, the three seeded rungs, ECE, `book_move1_entropy` per pass),
  `runs/p2_d/reps_iter0020_r*.jsonl` (0.330, 0.285, 0.295), `runs/p2_d/p2_verdict_reps.json`
  (+0.1033 [+0.0600, +0.1467], SE 0.0220, 100 openings) and
  `runs/p2_d/iterations.jsonl` (`move1_prior_entropy` 0.0107, `skipped_steps` 0,
  `decisive_rate` 0.709 at iteration 20);
- every command against the scripts' current flags: `eval_ladder.py`
  `--config/--iteration/--role`, `eval_reps.py run --out-dir`, `eval_reps.py
  verdict --a/--b`, `train_phase2.py --init/--seed-pool/--ladder-*`; the row
  writer creates `runs/p2_e_bridge/` itself;
- Stockfish: `eval.stockfish_path: ''` resolves to `which stockfish` =
  `/usr/games/stockfish` (Stockfish 16), the same fallback p2_d ran on, so no
  `--stockfish` is needed;
- `runs/p2_d/state.json` holds a 20-entry league pool for `--seed-pool`;
- the code table's claims against `configs/p2_e.yaml` (`batch_leaves` 512 >
  `concurrent_games` 256).

Three corrections came out of the check and are in the text above: the H
expectations name the fields they are read from; B4 is reported with no
tolerance; the contract states the recorded-cadence difference.

**Signed: the Claude Code session that drafted it, at Michael's instruction,
2026-10-22 13:13 BST.** Michael asked that the draft be checked and signed as
drafted; the commit carrying this line is the registration.

---

## Addendum 1 (2026-10-22 14:40 BST) — Step 0: the bridge, and the skill-0 instrument

**Run state at the time of writing:** nothing has trained; `runs/p2_e` does not
exist. Step 0 ran 13:17–14:01 BST at `43dbe42` (clean tree, `= main`) under
`gpu hold`; the attribution replays below ran 14:19–14:35. Records:
`runs/p2_e_bridge/{ladder.jsonl, reps_iter0020_r{1,2,3}.jsonl, p2_verdict_reps.json,
ladder_diagnostic.jsonl, ladder_diagnostic_r2.jsonl}`; console in
`runs/p2_e_bridge.console.log` (untracked, like `p2_d.console.log`). Nothing
above this line is edited.

### Measured against B1–B4

| | Registered | Measured | |
|---|---|---|---|
| B1 | mate-in-2 183/200, the same 17 failures; mate-in-3 129/200 | 183/200, the same 17; 129/200 | identical |
| B2 | random 1.000, greedy 1.000, phase1 0.9975 (199W/1D/0L) | identical, including `pair_scores` element for element and mean plies 32.03 / 32.66 / 56.61 | identical |
| B3 | skill-0, three reps: paired CI vs p2_d's recorded reps includes zero | **0.435, 0.510, 0.483** (mean 0.476) vs recorded 0.330, 0.285, 0.295 (mean 0.303); paired difference **+0.1725 [+0.1225, +0.2217]**, excludes zero | **FAILED** |
| B4 | ECE reported, reference 0.1762 | **0.0798** (n = 5000) | moved with B3 |

The skill-0 games changed character with the score: 131–136 mean plies against
p2_d's 124–127, draws 46–50 per 200 against 22–26, losses 73–90 against 121–131.

### Attribution

Ruled out by measurement: the learner (B2's `pair_scores` are identical, so the
search plays the same games at 48 sims); the Stockfish binary (`16-1build1`,
installed 2026-08-11, two days before p2_d ran; file dated 2024-04-22); the new
60 s wall-clock ceiling (20 positions at skill 0: 10,000–10,025 nodes searched
with and without it).

What remains is the P9 fix. `StockfishOpponent.choose` now passes a game key to
python-chess, which sends `ucinewgame` whenever the key changes. The ladder calls
`choose(board, rng)` with no key, so the key is `id(board)`, which changes on
every interleaved switch between the ~200 concurrent games that share one engine
process: **the 16 MB table is cleared on essentially every engine move.** Before
the fix the call was `play(board, Limit(nodes=10000))` with no key: one
`ucinewgame` per matchup, and the table accumulated over the ~13,000 engine
moves of the match, so a move in game k drew on the search of games 1..k-1 and on
the order the scheduler interleaved them.

**Test.** The pre-fix call restored, verbatim, under today's code, p2_d/0020,
two replays: **0.275** (45W/20D/135L, 124.2 plies) and **0.285** (45W/24D/131L,
125.7 plies) — inside the old instrument's spread and with its game profile.
Three readings of one checkpoint:

| engine call | code | skill-0 score |
|---|---|---|
| shared table (no key) | p2_d's (`91293f7`) | 0.330, 0.285, 0.295 — mean **0.303** |
| shared table (no key) | today's (`43dbe42`) | 0.275, 0.285 — mean **0.280** |
| table cleared per switch (key) | today's (`43dbe42`) | 0.435, 0.510, 0.483 — mean **0.476** |

The shift is the table, entirely. The rest of the fix pass contributes nothing
measurable to the skill-0 reading.

### What this changes, recorded before any p2_e data exists

- The p2_b..p2_d skill-0 series was measured on an engine whose effective depth
  grew through each matchup and whose moves depended on game order — the
  non-reproducibility P9 named. Its numbers are internally consistent (same
  instrument throughout) and discontinuous with the fixed code.
- Neither instrument is an engine game: an engine keeps its table *within* a
  game. A per-game table needs one engine process per concurrent game, or
  sequential matches — a design change, not a flag. It is not made here.
- **M4** (mean-over-reps score vs skill 0 with its CI entirely above 0.5) was
  written against the shared-table instrument, on which p2_d/0020 reads 0.303.
  On the cleared-table instrument the same weights read 0.476, one rep at 0.510.
- **P2e stands as written** (+0.10 against the bridge, CI excluding zero above).
  It was drafted on the old scale; it is not reworded.
- **The ECE expectation stands as written** ("fails again, ≈ 0.17–0.18") and is
  already known, before data, to have been drafted on the other instrument: the
  bridge reads 0.080 on these weights. A p2_e ECE below 0.15 is therefore read
  against the bridge's 0.0798 (same instrument), not scored as a calibration
  gain against p2_d's 0.1762. Registered here so it cannot be decided after.
- **N stands as written.** mate-in-2 is a deterministic suite and the bridge
  reproduced it to the position.

### Decisions pending (Michael), before launch — the launch waits on them

1. **Instrument.** Keep the cleared-table call, or restore the shared table.
2. **Continuity.** Re-measure p2_c/0020 (three reps) and p2_b's endpoint on
   whichever instrument is kept, so the series is one instrument end to end.
3. **M4.** Restate its bar against the kept instrument now, or declare it a
   different test. Not to be decided after p2_e's number is seen.

---

## Addendum 2 (2026-10-22 15:15 BST) — continuity on the kept instrument, and M4 restated

**Decisions taken** (Michael, 2026-10-22 ~14:45, on Addendum 1's three questions):
the cleared-table instrument is kept; p2_c/0020 is re-measured on it; M4 is
restated here, before launch, the restatement delegated to the session; p2_e
launches at the commit carrying this addendum. **Run state:** nothing has
trained; `runs/p2_e` does not exist. The re-measure ran 14:44–15:09 BST at
`bc1d88d`.

### p2_c/0020 on the cleared-table instrument

Three reps (`runs/p2_c_bridge/reps_iter0020_r{1,2,3}.jsonl`): **0.4325, 0.395,
0.4225** (mean **0.417**; 64W/45D/91L, 55W/48D/97L, 64W/41D/95L) against p2_c's
recorded shared-table reps 0.165, 0.230, 0.205 (mean 0.200).

| checkpoint | shared table (recorded) | cleared table | offset |
|---|---|---|---|
| p2_c/0020 | 0.200 | 0.417 | +0.217 |
| p2_d/0020 | 0.303 | 0.476 | +0.173 |

The fixed-code slope p2_c → p2_d (`runs/p2_c_bridge/p2_verdict_reps.json`,
paired bootstrap over the 100 openings on rep-means): **+0.059 [+0.010,
+0.108]**, SE 0.025, 2.4 SE. The recorded shared-table verdict was +0.103
[+0.060, +0.147], 4.7 SE. **p2_d's P2 verdict is re-affirmed on the kept
instrument, with less margin:** the CI excludes zero on both.

The two offsets differ by 0.044, about 1.5 SE of their difference. Two points
cannot show the offset constant; what they show is a smaller offset at the
stronger checkpoint. The instrument change compresses the score scale in this
region: the same twenty iterations of training read as +0.06 where they read
+0.10.

### M4 restated

M4 was "mean-over-reps score vs skill 0 with its 95% CI entirely above 0.5",
written against the shared-table instrument, on which p2_d/0020 reads 0.303.
Restated, before any p2_e data:

**M4: mean-over-reps skill-0 score on the cleared-table instrument, with its 95%
CI entirely above 0.675.**

0.675 is 0.5 carried through the offset measured at the checkpoint nearest the
bar (p2_d/0020, +0.1725, rounded up to 0.175). If the offset is constant, this is
the original milestone on the new scale. If it shrinks toward parity, as the two
points suggest, the true equivalent lies below 0.675 and this bar is stricter
than the original. It is not weaker in either case: no p2_e result can meet M4
by the instrument change. A milestone, not a stopping rule: the run continues to
20 regardless. **Not expected in p2_e:** 0.476 plus the fixed-code slope is about
0.54.

### P2e, re-read on the kept instrument — the criterion is unchanged

P2e's registered criterion is the paired CI against the bridge excluding zero
above; it stands. Its point expectation, +0.10, was the shared-table slope; the
fixed-code slope of the same step is +0.059. Recorded here so that a p2_e
difference near +0.06 with its CI above zero is read as the flywheel continuing
at its measured fixed-code rate, and a shortfall against +0.10 alone is not a
miss of the registered criterion.

### Launch

p2_e launches from clean `main` at the commit carrying this addendum, with the
command in "Measurement protocol", under `gpu hold`.

**Signed: the Claude Code session, at Michael's instruction (decision 3
delegated the restatement to it), 2026-10-22 15:15 BST.**

**Launch note (2026-10-22 15:20 BST).** Launched 15:12 BST from clean `main` =
`0913957` under `gpu hold`; `runs/p2_e/provenance.json` records that SHA,
`git_dirty: false`, `init_from: runs/p2_d/ckpt_iter0020.pt`,
`seed_pool_from: runs/p2_d`, the p2_d stack (torch 2.12.1+rocm7.2, Radeon 8060S),
and config fingerprint `17f92fa7…` (= `sha256sum runs/p2_e/config.yaml`). One
correction to the contract's "Recorded cadence" row: the run's resolved config
differs from `runs/p2_d/config.yaml` in **two** fields, `eval.eval_every_n_iters`
(10 → 5) and `eval.ladder_anchors` (three rungs → four); `eval.anchor_ckpt`
resolves to the pinned anchor under both, so it is not a delta. Records only,
as stated.

---

## Addendum 3 (2026-10-23 10:20 BST) — a regressed row, and a control before any A/B

**Run state at the time of writing.** Iterations 0–14 are committed
(`state.json`: iteration 14, step 5600, 2 skipped steps). The host stopped hard at
**00:25:07 BST** during iteration 15's self-play: the journal ends mid-stream with
no OOM, amdgpu, thermal, MCE or shutdown record, and no crash dump was written.
The box was back at 09:29. p2_e **resumed at 10:04 BST** under `gpu hold` from
`dc51296`, which differs from the launch SHA `0913957` in this file only
(`git diff --stat 0913957 dc51296`: `PREREG-p2_e.md`, 11 insertions). Resume
is reconstruction: RNG is derived from (seed, iteration, stream), so iteration 15
redraws what it would have drawn. `state.json` records the resuming SHA in
`resumed_with` at the iteration-15 snapshot. The pass-15 ladder has not run. The
RUNLOG carries the full account at close.

**Data already seen.** Two protocol ladder passes, both read before this addendum
was written, and the reason for it:

| pass | skill-0 (one replay) | vs the bridge's 0.476 | mate-in-2 | ECE |
|---|---|---|---|---|
| 5 | 0.3375 (49W/37D/114L) | −0.139 | 0.920 | 0.1255 |
| 10 | 0.3525 (48W/45D/107L) | −0.124 | 0.930 | 0.1192 |

**P2e has produced no data.** Its statistic is three replays at
`ckpt_iter0020`, and that checkpoint does not exist yet. Under the amendment
policy a rule may be added for it. This addendum leaves P2e's expectation and
criterion as written (CI against the bridge excluding zero above). It changes
only what follows when P2e does *not* pass, and adds one measurement (the control
below) before that measurement has produced any data. The control runs **whatever
P2e's outcome**, so running it does not depend on the result.

### Why the "P2e flat" row is replaced

The registered row reads: flat → "C4 is the only change measured to reach
training: a BLOCKED conversation on C4 before any lever." Two problems, both
known before P2e's data:

1. **It has no row for a regression.** P2e has three outcomes on its statistic:
   the paired CI is entirely above zero (passes), includes zero (flat), or is
   entirely below zero (regressed). Passes 5 and 10 sit about 0.13 below the
   start, roughly four times a single replay's spread on this instrument (the
   bridge's three reps span 0.435–0.510). A regressed endpoint is a live outcome,
   and the table says nothing about it.
2. **C4 is a weak suspect.** In the ring, the stored move is read only to place
   the promotion target at the acted (from, to) cell (`replay.py` sampling,
   `promo_mask`). The mask is on for **0.58%** of rows (20,000 rows sampled from
   the p2_e ring, 2026-10-22 22:36 BST). A change confined to the promotion head
   on 0.58% of rows is an unlikely cause of a skill-0 drop. Pointing the
   investigation at C4 first would test the cheapest hypothesis, not the likeliest.

Before any code is suspected, one cheaper question has to be answered: **does
the old code also dip mid-run on this instrument?** Only p2_d's endpoints (0000 via
p2_c/0020, and 0020) have been read on the cleared-table instrument. Its mid-run
passes were read on the shared-table instrument only, where they sat at 0.180,
0.215 and 0.1925 against a start of 0.200 (p2_c/0020's recorded reps): no dip of
p2_e's size. Addendum 2 showed the two instruments do not map by a constant
offset, so that reading does not settle the question. The control does.

### Control C (added): p2_d's mid-run on the cleared-table instrument

`runs/p2_d/ckpt_iter0005.pt`, `…0010.pt`, `…0015.pt`, three skill-0 replays each,
with the same invocation as the bridge (`eval_reps.py run`, config beside the
checkpoint, default seed) and output in `runs/p2_d_midrun/`. Each is read against
p2_d's start on the same instrument, p2_c/0020 (`runs/p2_c_bridge/reps_iter0020_r*.jsonl`,
mean 0.417), on the registered statistic (paired bootstrap over openings on
rep-means). `src/`, `scripts/` and `configs/` are unchanged between the
reference reps (`bc1d88d`, `43dbe42`) and `dc51296`, so all of these are one
instrument.

- **Dip:** at least one of the three has its CI entirely below zero.
- **No dip:** none does.
- **Depth:** the most negative of the three point estimates. It is used only in
  the regressed rows below.

Runs after the P2e replays, under one `gpu hold` with them, about 75 minutes.

### Decision rules (replace the "P2e flat" row and add the regressed case)

| P2e at pass 20 | Control C | Decision |
|---|---|---|
| **passes, N holds** | either | As registered: the fixed loop is the baseline, and p2_f takes one lever. C is recorded as the reference trajectory for reading later rounds' mid-run passes. |
| **flat** | dip | The old code dips mid-run on this instrument too, so p2_e's trajectory is not evidence against the fixed code. **Next is protocol, not code:** a single 20-iteration endpoint cannot resolve the fixed-code slope (+0.059, SE 0.025; at that true slope the test reads flat about one time in three). Options for Michael, to be pre-registered before the next round: more replays per endpoint, longer rounds, or a trajectory statistic. No lever and no A/B. |
| **regressed** | dip at least as deep as P2e's point estimate | Same as the row above: the endpoint lies within the trough the old code reaches. |
| **flat** | no dip | **The fixed code is implicated → Step 2.** |
| **regressed** | no dip, or a dip shallower than P2e's point estimate | **The fixed code is implicated → Step 2.** |
| **N or ECE breaks** | — | Standard stop rules (unchanged). |

Gates are untouched in every row. No lever is added while a row points at Step 2.

**Step 2 (registered in outline; it gets its own pre-registration before any
compute).** An A/B from one start. Five iterations of p2_d's training code
(`72e7f1a`, the SHA in `runs/p2_d/provenance.json`) from `runs/p2_d/ckpt_iter0020.pt`
at `runs/p2_d/config.yaml`, league seeded from `runs/p2_d`, in a separate
worktree. Its 0005 is measured against `runs/p2_e/ckpt_iter0005.pt`, three replays
each, both on today's cleared-table instrument.

- If the old code's 0005 is above p2_e's (CI above zero), bisect the fix-pass
  commits that touch the training path. C4 is one of them, not the first.
- If both drop alike, the code is not implicated, and the decision falls back to
  the protocol row.

### Commands (added to "Measurement protocol")

```bash
# Control C — after the P2e replays, under the same gpu hold
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_d/ckpt_iter0005.pt \
    --anchor stockfish:0 --reps 3 --out-dir runs/p2_d_midrun
# … and the same for ckpt_iter0010.pt and ckpt_iter0015.pt
.venv/bin/python scripts/eval_reps.py verdict \
    --a p2_c_fixed runs/p2_c_bridge/reps_iter0020_r*.jsonl \
    --b p2_d_iter0005 runs/p2_d_midrun/reps_iter0005_r*.jsonl \
    --out runs/p2_d_midrun/p2_verdict_reps_iter0005.json
# … and the same for 0010 and 0015
```

The negation `!runs/**/p2_verdict_reps_*.json` and `MUST_REACH` entries for
`runs/p2_d_midrun/` land in the commit carrying this addendum.

**Signed: the Claude Code session, at Michael's instruction (2026-10-23 ~10:00:
"resume it, and do both the addendum and control"), 2026-10-23 10:20 BST.**
