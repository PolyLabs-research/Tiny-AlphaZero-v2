# Pre-registration — p2_b: anchor rehearsal

**Written before launch. Not edited after.** Anything learned during the run goes
in `RUNLOG-p2_b.md`, not here. The point of this file is that the decision rules
existed before the data did.

- **Lever:** `train.rehearsal_frac: 0.05`, split `mate1 0.02 / mate2 0.03`
- **Everything else:** inherited from `runs/p2_a/config.yaml` verbatim, enforced by
  `tests/test_rehearsal.py::test_p2b_config_is_the_brief`
- **Warm start:** `--init-weights runs/p2_a/ckpt_iter0050.pt` (weights only, fresh optimizer)
- **League continuity:** `--seed-pool runs/p2_a` (its last 20 snapshots)
- **Length:** 20 iterations, `--ladder-every 5` → passes at 5, 10, 15, 20
- **Judged at:** pass 20

## Why this lever

p2_a's decomposition (`RUNLOG-p2_a.md` §3) split the gate exactly:
`0.890 coverage × 0.933 conditional = 0.830`. Rehearsal is the one intervention
acting on both factors — the anchor sets re-promote forcing moves into the prior's
top-16 (coverage) and carry +1 labels on the key children (conditional). Neither
factor clears 0.95 alone, which is why a coverage-only or value-only lever was not
chosen.

## This is a recovery experiment

Warm-started from p2_a's endpoint, so the question is whether the lever *restores*
lost capability and *holds* it. Holding it across 20 further iterations of the same
training that eroded it is also the prevention result — demonstrated in one run
rather than argued.

## Pre-registered expectations

| # | Expectation | Threshold |
|---|---|---|
| E1 | mate-in-2 recovered **and stable** | ≥ 0.95 at pass 15 **and** ≥ 0.95 at pass 20 |
| E2 | coverage tail re-promoted | prior top-16 coverage ≥ 0.98 at pass 20 |
| E3 | conditional firmed | solved-given-considered ≥ 0.95 at pass 20 |
| E4 | watchlist converts | `mate2_165` and `mate2_166` both solved at pass 20 |
| E5 | mate-in-3 reverses direction | ≥ 0.65 (from 0.570) — informational, no decision hangs on it |
| E6 | no strength collapse | phase1-anchor lower CI bound ≥ +727 at pass 20 |
| E7 | training health | no new non-finite-gradient cluster (> 4 in 8,000 steps) |

`mate2_129` is expected to stay unsolved and is not part of E4 — it is the honest
wobbler, CAPABILITY-verdicted with key-child value swinging 0.036 → 0.94 → 0.028 →
0.69 → 0.44 across p2_a's passes.

Book entropy: **reported, not predicted.** The funnel hypothesis is not what this
run tests, and a directional expectation here would invite reading the result
either way after the fact.

Expected noise, pre-registered so it is not later read as signal:

- A loss blip in the first ~200 steps as AdamW moments rebuild from the fresh
  optimizer. Not a regression.
- Gradient excursions of the kind p2_a showed (up to ~36). Weather until shown
  otherwise.
- **E6's threshold is one game wide.** The anchor rung quantizes at the top
  (`RUNLOG-p2_a.md` §6): 199/200 → +919.5, 198/200 → +798.2. E6 tests for
  *collapse*, not for improvement, and cannot do otherwise on this ladder.

## Decision rules

| Observation at pass 20 | Decision |
|---|---|
| **All of E1–E4 met** | p2_b closes as the recovery proof. S4-enable and opening-diversification go to a `p2_c` design with this evidence. M4 planning opens on the Stockfish rungs. |
| **Coverage fixed, conditional binding** (E2 met, E3 < 0.95) | The conditional conversation: §12.3's z → z/q value-target mix becomes `p2_c`'s lever, with p2_a's CAPABILITY dozen as its watchlist. |
| **Coverage still binding at pass 10** (top-16 coverage < 0.95) | One escalation permitted — same lever, intensity only: `rehearsal_frac: 0.10`. Re-registered in this file's successor *before* applying. |
| **Coverage still binding at pass 20** after that escalation | **STOP. BLOCKED.** Rehearsal-resistant forgetting at this scale is a major finding and gets written up as one. Gates untouched. |
| **Funnel continues monotone *and* instability recurs** | Diversification's `p2_c` case is made by the log. Note it; change nothing mid-run. |

## Gating dependency

**Stockfish must be installed before the pass-20 ladder.** The anchor rung is
saturated — three of p2_a's five passes have a clamped upper CI bound and no
measurement above ~+900 Elo exists on this ladder. Every `phase1` rung in this run
will be read against that ceiling. `sudo apt install stockfish` (candidate
16-1build1), then the `StockfishOpponent` smoke. Absent the binary the run still
completes and the mate-in-2 clauses are unaffected — Stockfish rungs skip cleanly —
but E6 stays the blunt instrument described above.

## What would make this run uninterpretable

Listed so they are checked rather than discovered:

- Rehearsal pointed at a contaminated set → the gate would measure memorisation.
  Guarded by `tests/test_contamination.py::test_rehearsal_dirs_are_guarded`, which
  asserts the config's *default dirs* are inside the checked set.
- Any config field other than the three `rehearsal_*` ones differing from p2_a.
  Guarded by `test_p2b_config_is_the_brief`.
- A fresh league pool against warm-started weights. Avoided by `--seed-pool`.
