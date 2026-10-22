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
