# Pre-registration — p2_d: the null lever and the breadth lever

**Written before Track 2 launches. Not edited after, except by dated addendum.**

**Amendment policy.** A rule may be added after launch *only* for a measurement
that has not yet produced any data, and only as a dated addendum recording the run
state at the time of writing. A rule may never be added, weakened, or reworded for
a measurement already taken.

`p2_d` is two tracks with **independent attribution**. Track 2 owns the GPU and
adds nothing. Track 1 is built on CPU behind a flag and measured offline on frozen
checkpoints, where its effect is isolated to the search alone. Track 1's flag is
**off everywhere in Track 2's run**, so the skill-0 and suite series stay
continuous with p2_c.

---

# Track 2 — the null lever (continuation)

- **Config:** `configs/p2_d.yaml`, byte-identical to `runs/p2_c/config.yaml`.
  Zero new levers. `test_p2d_config_is_the_brief` asserts zero deltas.
- **Warm start:** `runs/p2_c/ckpt_iter0020.pt` · **League:** seeded from p2_c
- **Length:** 20 iterations, `--ladder-every 5`, rungs
  `random,greedy,phase1,stockfish:0`
- **Judged at:** pass 20

**The question.** p2_c took skill-0 score 0.025 → 0.233 with its last five
iterations the steepest. Does that continue *with nothing added*? A self-sustaining
loop and a one-off adjustment to a new target look identical at the end of p2_c and
different at the end of p2_d. This is the cheapest round the campaign can run and
the only one that can tell them apart.

## P2 (primary)

Skill-0 Elo at pass 20 with 95% CI disjoint above p2_c's endpoint
(**0.233, −207.5 [−260.5, −159.8]**).

**Test of record.** Where two protocol passes are compared, the comparison is a
**paired-difference bootstrap over opening pairs**, not marginal-interval overlap.
Overlap is neither necessary nor sufficient: two independent intervals can overlap
while the paired difference separates cleanly, because pairing removes opening
difficulty as shared variance. This is the pass-15 lesson from p2_c, where an
overlap was reported as a separation (`RUNLOG-p2_c.md` §2).

`pair_scores` are persisted in ladder rows from this round on, which is what makes
the test computable at all — in p2_c the per-opening vector was discarded when each
matchup ended.

## M4 trigger (armed, not required)

M4's criterion — score vs skill 0 with 95% CI **entirely above 0.5** — is declared
**met** at the first protocol pass that satisfies it. **The run continues to 20
regardless.** A milestone is a threshold, not a stopping rule, and the trajectory
past it is data. Reaching it is not registered as an expectation; 0.233 is a long
way from 0.5.

## N (no-regress)

mate-in-2 ≥ 0.915 at pass 20 under the **current search** (the continuity
instrument). Mate-in-3 and the full decomposition reported at pass 20.

## ECE (restated clause)

**ECE < 0.15 at pass 20 on the honest sampler** (phase1 + stockfish positions).

This restates chunk-12's number against the instrument that replaced the one it was
written for. The original clause was measured on the phase1 matchup alone — a
matchup the learner wins ~99% of, where the value distribution is degenerate and the
metric is easy to pass. Two rounds of honest data now sit behind the restatement:
0.2153 at p2_c pass 5, 0.1678 at pass 20. Same threshold, correct instrument, and
the same reasoning that retired "monotone" in erratum 7 and the anchor rung in
`RUNLOG-p2_b.md` §4.

## H (reported, direction-free)

- **Value loss by halves — reported without a registered direction.** A direction
  was registered twice and was wrong both times (p2_b expected weather and got a
  tax; p2_c expected relief and got a further climb). The sharpness-tax account
  stays open rather than being repeatedly mispredicted.
- **Book-entropy series continues.** If it falls toward p2_a's collapse regime
  **and** instability returns, diversification's `p2_e` case is complete. Note it;
  change nothing mid-run.
- Skips < 10.

## Decision rules

| Observation at pass 20 | Decision |
|---|---|
| **P2 passes** | The flywheel needs nothing. M4 planning proceeds on the measured slope, and possibly on a met trigger. |
| **P2 flat** | The acceleration was a transient, and *that* is the finding: a BLOCKED conversation on what sustains the loop. Gates untouched. |
| **N or ECE breaks** | Standard stop rules. |

---

# Track 1 — proof-directed reply expansion (the breadth lever)

Built behind a flag, measured offline on frozen checkpoints **after** Track 2
completes. The box belongs to the run.

## The mechanism, from the evidence

`mate2_165` held **14 visits against a proof cost of 7** and never proved. Budget
was met; breadth was not. Below-root selection re-grinds the strongest reply, while
MCTS-Solver needs *every* reply refuted before it can induct a proof back. The lever
is breadth, not budget.

p2_c's competitor-q̂ measurement sharpens it: in seven of eight rows the key edge's
estimate was **higher** than the move that beat it, and it lost anyway — on visits,
through σ's visit scaling. Value ordering is not the failure.

## Contract

`mcts.proof_directed_expansion: bool = false`. When true, at an opponent node whose
**parent edge's q̂ ≥ `proof_q_threshold` (0.95)**, selection prefers **unexpanded
replies** until that node's reply set is fully expanded; normal selection resumes
thereafter. Exactly one ply of breadth at near-certain edges — bounded by
construction. Flag-off behaviour is byte-identical to today.

## Registered expectations

**W3.** `mate2_165` and `mate2_166` **convert flag-on**. Their proof-needs sit at 7
against available budget and this lever is aimed at exactly them. **If they do not
convert, the breadth theory is wrong and the RUNLOG says so.**

**Gate expectation.** Conditional ≥ 0.95 flag-on at `p2_c/0020`. Note the
compounding effect: `SLOWER_FORCED_WIN` cases should convert too, because a *landed*
proof engages the existing proven-depth ordering and the fast mate wins the argmax —
so the gate flag-on may clear the original ≥ 0.95 outright.

**Succession is not automatic.** Whatever it reads, **flag-on becomes a parallel
instrument series, not a replacement**. Which search the protocol ladder runs is
decided at `p2_d` close with both series on the table — exactly as the
anchor→Stockfish succession was done.

**The residual that survives flag-on** is, for the first time, a clean estimate of
**genuine capability misses** with search fully out of the way. Recorded as such for
`p2_e`.

## Measurement protocol

On frozen checkpoints `p2_c/0000`, `p2_c/0010`, `p2_c/0020`, `p2_d/0020`: mate-in-2
and mate-in-3 suites plus the full decomposition, **flag-on vs flag-off, same
checkpoint**. Same net, same positions, search-only delta — the purest isolation
this project can construct.

## What would make Track 1 uninterpretable

- The flag altering flag-off behaviour. Guarded by running the existing pinned and
  equivalence suites flag-off, untouched.
- The flag reaching self-play data generation. It stays off in Track 2's config and
  in self-play generally; enabling it in generation is a `p2_e` decision, not a side
  effect.

---

## Addendum 1 (2026-08-13 06:40) — Track 1's convertibility prediction

**Run state:** Track 2 at iteration 1, 23 minutes elapsed, no ladder pass yet.
Track 1's full measurement protocol has **not been run** — the only reading taken
is a preliminary two-position CPU probe on `p2_c/0020` (165 converts flag-on, 166
does not), recorded in commit `ea63564`. This addendum registers a prediction over
**thirteen** positions, eleven of which have no data at all, and it is filed before
the protocol runs.

### The rule, with zero free parameters

A residual CAPABILITY position converts flag-on **iff**

```
key-edge q̂ ≥ proof_q_threshold (0.95)      AND      flag-off allowance ≥ proof cost
```

where *allowance* is the edge's flag-off visit count and *cost* is
`1 + len(replies)`, both already recorded by `triage_mate2`. Nothing is fitted.

**Why this is the rule.** The sweep changes the *order* in which replies are
expanded, not the *number* of visits an edge receives — an edge's pre-elimination
allowance is set by Sequential Halving and is untouched by the flag. So the lever
can only land a proof that fits inside the allowance the edge already had. 165 held
14 against a cost of 7: the sweep fits, the proof lands, and it converted. 166 holds
6 against 7 — structurally one visit short, and no expansion order proves a
six-reply refutation in six visits.

### The prediction, per position, at `p2_c/0020`

| id | q̂ | allowance | cost | prediction |
|---|---|---|---|---|
| mate2_054 | 0.9894 | 14 | 8 | **CONVERT** |
| mate2_091 | 0.9989 | 14 | 9 | **CONVERT** |
| mate2_165 | 0.9993 | 14 | 7 | **CONVERT** *(already observed)* |
| mate2_196 | 1.0000 | 14 | 7 | **CONVERT** |
| mate2_166 | 0.9988 | 6 | 7 | no — one visit short |
| mate2_009 | 0.9670 | 2 | 3 | no |
| mate2_127 | 0.9679 | 6 | 14 | no |
| mate2_128 | 0.8407 | 2 | 14 | no — below threshold *and* short |
| mate2_087 | 0.6935 | 2 | 3 | no — below threshold |
| mate2_139 | 0.6818 | 2 | 3 | no — below threshold |
| mate2_186 | 0.7458 | 2 | 3 | no — below threshold |
| mate2_126 | 0.2385 | 2 | 17 | no — below threshold |
| mate2_129 | 0.0211 | 2 | 17 | no — below threshold |

**4 of 13 convert.** COVERAGE positions are excluded by construction: the lever
cannot help a move that was never in the considered set.

### What each outcome means

- **Prediction holds across all four checkpoints** → the mechanism is not merely
  fixed but *fully characterised*: Track 1 closes as "it converts precisely the set
  the arithmetic says it can", which is a stronger claim than "it helped".
- **Extra conversions** → allowance is not flag-invariant after all. Plausible: the
  sweep changes which values back up, which can change halving outcomes, so an edge
  may survive elimination it would otherwise have lost. That would be a real finding
  and the RUNLOG says so.
- **Missing conversions** → the arithmetic is not sufficient, and the residual needs
  a mechanism beyond breadth.

**The survivors define the parked eval-profile conversation.** Whatever fails to
convert with search fully out of the way is what an `m` change — and only `m` —
would be about, with **`mate2_166` as its named boundary case**: one visit from the
line, and therefore the cleanest possible test of whether allowance is the binding
constraint.

### Stated caveat

Allowance is measured flag-off and assumed unchanged by the flag. That is
approximately, not exactly, true — see the "extra conversions" branch above. The
prediction is registered with this assumption explicit rather than silently.
