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
