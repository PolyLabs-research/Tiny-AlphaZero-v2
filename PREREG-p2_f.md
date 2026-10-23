# Pre-registration — p2_f: carry the replay ring across the round boundary

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-10-23** (see Sign-off), at Michael's instruction.
> Step 0 has not run and nothing has trained.

**Amendment policy.** A rule may be added after launch *only* for a measurement
that has not yet produced any data, and only as a dated addendum recording the run
state at the time of writing. A rule may never be added, weakened, or reworded for
a measurement already taken.

---

## The question

p2_d and p2_e both fall 0.11–0.14 below their own start by pass 5–10 and recover by
pass 15 (RUNLOG-p2_e §3–4; p2_d at three reps: −0.108, −0.107, +0.036, +0.059).
p2_e's endpoint fell back below its start; Addendum 3 of PREREG-p2_e read that as
the loop's shape, not the fixed code, and sent the next step to protocol.

Every round so far opened on an **empty replay ring**. Training draws 400 × 1024 =
409,600 rows per iteration against ~56,000 new positions, so iteration 1 draws each
stored row 7–8 times. The ring fills only at iteration 9–10, after which every
iteration draws each row 0.82 times. Both dips sit in that fill phase, and p2_e's
policy loss is lowest where the ring is smallest (1.29–1.39 at iterations 1–5,
1.47–1.58 at 10–20).

**H_ring: the mid-round dip is the fill phase.** p2_f continues p2_e with p2_e's
ring carried in, so the ratio is 0.82 from iteration 1. If H_ring is right, p2_f
does not dip.

This is a protocol change about how rounds are chained: with the weights, the
league and the ring all continued, a round boundary resets only the optimizer.
It is still attributed like a lever, alone.

## Contract

| | |
|---|---|
| Config | `configs/p2_f.yaml` = `runs/p2_e/config.yaml` verbatim, ladder cadence (5) and rungs (`random,greedy,phase1,stockfish:0`) included; `test_p2f_config_is_the_brief` asserts zero deltas |
| Warm start | `runs/p2_e/ckpt_iter0020.pt` |
| League | `--seed-pool runs/p2_e` (continuity, as in every warm-started round) |
| **Lever** | `--seed-replay runs/p2_e`: p2_e's committed, intact rows. Dry run 2026-10-23: **444,741 rows**, p2_e global rows 686,161–1,130,901, iterations 13 (42,758, partial) through 20, re-tagged −7..0. The 55,259 torn iteration-21 rows past p2_e's commit (1,130,902) are left behind |
| Not carried | the optimizer: `--init` loads weights only, and AdamW starts fresh, as in every round. LR is constant (2e-4), so no schedule restarts |
| Length | 20 iterations, ladder every 5 |
| Judged at | **the trajectory**: three replays at 0005, 0010, 0015 and 0020 |
| Start reference | p2_e/0020's three reps (0.405, 0.385, 0.4325; mean 0.4075), measured at `9171da5` |
| Code | clean `main` at launch. The delta from `9171da5` (where the start reference was measured) is `32804f6` + this file: `replay.py` gains `seed_from` and a read-only open (additive), `phase2.py` threads `seed_replay` through run creation, the CLI gains the flag. The instrument (`evaluate.py`) loads `replay.py` only through `selfplay`'s import and never constructs a ring; the change there is additive (a new method, and a `readonly` parameter whose default is the old behaviour). Every other module the instrument loads is byte-identical, so no bridge is needed |
| GPU | `gpu hold` (HOW_TO_TRAIN Step 4); a shared-GPU stretch is logged with its times |
| Resume | `--iterations` counts the invocation: a resume passes `20 − committed` |
| Disk | ~4 GB (21 checkpoints, 20 seeded snapshots, the ring); 36 GB free at writing |

## What changed between p2_e and p2_f

Only the ring at iteration 0. Everything else a warm start sets is set as p2_e
set it: weights from the previous endpoint, the league continued, a fresh
optimizer, the same config, seed 42.

---

## Step 0 — p2_e's own trajectory at three reps (before launch)

p2_f is compared with p2_e: same code, same config, empty ring versus carried ring.
p2_e's mid-run passes exist only as single ladder replays (0.3375, 0.3525, 0.465),
so Step 0 measures them on the registered statistic: `p2_e/0005`, `0010` and `0015`,
three skill-0 replays each, each against p2_e's start (the bridge reps,
`runs/p2_e_bridge/reps_iter0020_r*.jsonl`, mean 0.476). About 75 minutes.

**S0 (expected): p2_e/0005 and p2_e/0010 dip** (CI entirely below zero); 0015's CI
includes zero.

**Launch gate.** If *neither* 0005 nor 0010 dips at three reps, the dip p2_f is built
to remove does not exist on this code at this statistic. That is reported before
launch, and the launch waits for Michael. Otherwise p2_f launches without asking
again.

## Registered expectations

### L — the lever did what it claims (manipulation check)

- **L1.** `provenance.json`'s `seed_replay` reads `rows: 444741`,
  `donor_committed: 1130902`, `donor_head: 1186161`, `iteration_offset: -20`.
- **L2.** `iterations.jsonl` iteration 1: `replay_size` = min(444,741 + `positions`,
  500,000).
- **L3. Mechanism signature (direction registered).** Mean `policy_loss` over
  iterations 1–5 **≥ 1.45**. p2_e read 1.319 there on a filling ring; its full-ring
  iterations 10–20 read 1.47–1.58, and p2_f's iteration 1 trains mostly on rows
  p2_e/0020 was already fitting at 1.58.

L1 or L2 failing makes F1–F2 uninterpretable (decision table). L3 is reported
against its direction; it is not a gate.

### F1 (primary) — the dip

`p2_f/0005` and `p2_f/0010`, each against the start reference, on the registered
statistic (three replays, per-opening rep-means, paired bootstrap over openings).
The threshold is half of Control C's depth (−0.108):

- **Removed:** at *both* checkpoints the CI is not entirely below zero **and** the
  point estimate is above −0.054.
- **Persists:** at *either* checkpoint the CI is entirely below zero **and** the
  point estimate is at or below −0.054.
- **Partial:** anything else.

**Expected: Removed.** Power, from Control C's SEs (0.024–0.027): at a true
difference of 0 each checkpoint reads Removed with probability ~0.975. At an
unchanged dip of −0.108 it does so with probability ~0.01.

### F2 — the endpoint

`p2_f/0020` against the start reference. **Expected: CI entirely above zero, point
≈ +0.06** (p2_c → p2_d on the kept instrument, the one fixed-instrument round that
turned). Held with low confidence: H_ring explains the dip, not p2_e's fall from pass
15 to 20.

### F3 — the trajectory (reported, read against Step 0 and Control C)

p2_f at 0005/0010/0015/0020 relative to its start, side by side with p2_e (Step 0)
and p2_d (Control C), each relative to its own start. No statistic is scored on the
comparison. **Reading caveat, registered now:** p2_f starts from p2_e/0020, which may
sit in a trough (p2_e's pass-15 single replay read 0.465). The RUNLOG reports
p2_f/0005 against p2_e/0015 (Step 0) as well, so that a start from a trough is not
read as a removed dip.

### M4 (armed, not expected)

Mean-over-reps skill-0 at 0020 with its CI entirely above 0.675 (PREREG-p2_e
Addendum 2). 0.41 + 0.06 falls far short.

### N (no-regress)

mate-in-2 ≥ 0.915 at pass 20. **Expected: holds** (p2_e: 184 → 194/200 over its four
passes).

### ECE

ECE < 0.15 at pass 20. **Expected: holds**, no direction registered. p2_e's passes
read 0.1255, 0.1192, 0.0626 and 0.1267: the measurement swings by 2× between
consecutive passes, and its spread is still unmeasured.

### H (reported)

- **H1. Funnel collapse persists (direction registered).** `move1_prior_entropy` at
  iteration 20 < 0.05 and `book_move1_entropy` ≤ 1.575 at every pass. The lever does
  not touch exploration.
- **H2.** `skipped_steps` summed < 10 (p2_e: 4).
- **H3. `value_loss` by halves**: direction-free.
- **H4. `decisive_rate`**: direction-free.
- **H5. `promo_loss`**: reported, comparable with p2_e (both C4 rings).

## Decision rules

| F1 | F2 | Decision |
|---|---|---|
| **Removed** | passes | **H_ring supported.** Ring carry becomes standing protocol for every warm-started round, like `--seed-pool`. p2_g takes one lever from the open list: `value_q_blend: per_row`, the funnel collapse, the two lever-broken positions, the residual, ECE. |
| **Removed** | flat or regressed | The dip was the fill phase, and removing it did not make twenty iterations turn. Ring carry is kept, since it removed a confound. Late-round behaviour (p2_e's 15 → 20 fall) becomes the open question, with Addendum 3's protocol options: more replays, longer rounds, a trajectory statistic. Michael's call. |
| **Persists** | any | The ring is not sufficient. The named next suspect is the optimizer reset; p2_g carries the AdamW state as well, as its one added lever, and keeps the ring. |
| **Partial** | any | Inconclusive on H_ring at this power. Reported. Michael's call (e.g. more replays at 0005/0010). |
| L1 or L2 fails | — | The lever did not apply as claimed. Diagnose before reading anything else. |
| **N or ECE breaks** | — | Standard stop rules. |

Gates are untouched in every row.

## Measurement protocol

```bash
# Step 0 — p2_e's trajectory at three reps, then the verdicts
for it in 0005 0010 0015; do
  gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
      .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_e/ckpt_iter$it.pt \
      --anchor stockfish:0 --reps 3 --out-dir runs/p2_e_midrun
  .venv/bin/python scripts/eval_reps.py verdict \
      --a p2_e_start runs/p2_e_bridge/reps_iter0020_r*.jsonl \
      --b p2_e_iter$it runs/p2_e_midrun/reps_iter${it}_r*.jsonl \
      --out runs/p2_e_midrun/p2_verdict_reps_iter$it.json
done

# The run
gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    .venv/bin/python scripts/train_phase2.py --config configs/p2_f.yaml \
    --init runs/p2_e/ckpt_iter0020.pt --seed-pool runs/p2_e --seed-replay runs/p2_e \
    --out runs/p2_f --iterations 20

# F1-F3 — three replays at each judged checkpoint, then the verdicts
for it in 0005 0010 0015 0020; do
  gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
      .venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_f/ckpt_iter$it.pt \
      --anchor stockfish:0 --reps 3
done
for it in 0005 0010 0015; do
  .venv/bin/python scripts/eval_reps.py verdict \
      --a p2_f_start runs/p2_e/reps_iter0020_r*.jsonl \
      --b p2_f_iter$it runs/p2_f/reps_iter${it}_r*.jsonl \
      --out runs/p2_f/p2_verdict_reps_iter$it.json
done
.venv/bin/python scripts/eval_reps.py verdict \
    --a p2_f_start runs/p2_e/reps_iter0020_r*.jsonl \
    --b p2_f runs/p2_f/reps_iter0020_r*.jsonl \
    --out runs/p2_f/p2_verdict_reps.json
```

Every record above is reachable by git (`tests/test_gitignore_musttrack.py`
`MUST_REACH`). Wall clock: Step 0 ~75 min, the run ~12.5 h (34 min per iteration
plus four ~20-min ladder passes), F1–F3 ~100 min.

## What would make p2_f uninterpretable

- **A config delta.** Guarded by `test_p2f_config_is_the_brief`.
- **The lever not applying.** L1/L2.
- **Launching from anything but a clean `main`.** A dirty launch is restarted,
  not annotated.
- **An instrument change** between `9171da5` and the F1–F3 replays: the modules
  importing `tinyalphazero.evaluate` loads (`config`, `encoding`, `mcts`,
  `opponents`, `selfplay`, `replay`) must not change beyond `32804f6`'s additive
  `replay.py` change.
- **A shared GPU** or **a stack change** (torch, ROCm, driver), as for p2_e.

---

## Sign-off

Checked before signing, 2026-10-23 16:35 BST:

- the carry, by running it: `_copy_seed_replay(runs/p2_e)` into a scratch ring
  carried 444,741 rows in 5.6 s, tags −7..0 with per-iteration counts matching
  p2_e's ring, oldest and newest rows byte-identical to p2_e's slots, p2_e's ring
  files unchanged (sha256 before and after);
- every number above against `runs/p2_e/{iterations,ladder}.jsonl`,
  `runs/p2_e/p2_verdict_reps.json`, `runs/p2_d_midrun/p2_verdict_reps_iter*.json`,
  `runs/p2_e/state.json` and `runs/p2_e/replay/meta.json`;
- every command against the scripts' current flags: `train_phase2.py
  --init/--seed-pool/--seed-replay/--iterations`, `eval_reps.py run --out-dir`,
  `eval_reps.py verdict --a/--b/--out`; `runs/p2_e/state.json` holds a 20-entry pool;
- the instrument: `git diff --stat 9171da5 32804f6 -- src scripts configs` names
  `phase2.py`, `replay.py`, `train_phase2.py` and `configs/p2_f.yaml` only.
  Importing `tinyalphazero.evaluate` loads `replay` (through `selfplay`) but not
  `phase2`, and nothing on the instrument path constructs a `ReplayBuffer`; the
  draft's first wording ("everything it imports is byte-identical") was wrong and
  was corrected before signing;
- the test suite at `32804f6`: 682 passed, 2 skipped; ruff clean.

**Signed: the Claude Code session, at Michael's instruction ("carry the replay
buffer across rounds, pre-register it as p2_f"), 2026-10-23 16:35 BST.** The commit
carrying this line is the registration.
