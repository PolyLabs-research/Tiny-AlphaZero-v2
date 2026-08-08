# BLOCKED — Chunk 6 gate: mate-in-2 ≥ 95% at 128 sims

**Date:** 2026-08-08 · **Branch:** `chunk-06-gumbel` · **Status:** OPEN — needs a decision, not a fix

**This fires return condition 1 of `BLOCKED-2026-08-06-phase1-gates.md`**, verbatim:
*"Chunk 5/6 search gates fail: mate-in-1 < 100% @16 sims (stub), or **mate-in-2 < 95% @128
sims with this checkpoint**."* The mate-in-1 half did **not** fire (100/100). The mate-in-2
half has. Per that document, the Phase 1 gate-retirement decision is now formally reopened.

## The failure

```
$ python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_2.epd \
      --sims 128 --ckpt runs/phase1_anchor.pt
mate_in_2.epd: 154/200 solved at 128 sims (77.0%) | eval=runs/phase1_anchor.pt root=gumbel | gate >= 95%
```

Every other chunk 6 gate passes: mate-in-1 100/100 at 16 and at 48 sims (stub), Sequential
Halving budget exact for all tested (n, m), `improved_policy` well-formed and >0.9 on a
proven mate. 185 tests pass, ruff clean.

## Diagnosis — the search is at its ceiling; the *policy prior* is the binding constraint

Diagnosis order was as instructed: search first (stub), net second.

| Measurement | Result |
|---|---|
| Stub evaluator (search alone) | 30.5% |
| `phase1_anchor.pt` | 77.0% |
| **Solved *given* the forcing move was in the considered set** | **95.4%** (144/151) |
| Forcing move in considered set at m=16 | 75.5% (151/200) |

The conditional number is the finding: **when search gets to look at the winning move, it
solves the position 95.4% of the time — at the gate.** What fails is *consideration*.
Gumbel takes the top-m by `g + logits`, so an action the network's prior ranks below ~16th
is never simulated at all, and no amount of search budget recovers it.

Network prior rank of the forcing move (first 60 positions):

```
median rank=1   top1=42%   top8=73%   top16=83%   top32=93%
```

`top16 = 83%` is a hard ceiling on a 95% gate at m=16, independent of search quality.

Raising `m` does not rescue it, because consideration and per-action budget trade off:

| m | 16 | 24 | 32 | 48 | 128 (all) |
|---|---|---|---|---|---|
| solved | 76.0% | 73.5% | 80.0% | 81.0% | 81.0% |

Nor does budget (first 60 positions, m=32): 128 → 81.7%, 256 → 81.7%, 512 → 91.7%,
1024 → 86.7%. Non-monotone and plateauing below the gate at 8× the specified sims.

## Why this is a real tension, not a tuning miss

`BLOCKED-2026-08-06` retired `mate1_solve_rate` on the argument that *"chunk 5's own gate
proves mate-in-1 correctness is search's job (100% at 16 sims with a uniform stub
evaluator)"*. That argument held and still holds — mate-in-1 is 100% with **no network at
all**.

It does not extend to mate-in-2. A mate-in-2 cannot be found by terminal detection from the
root; the first move must be *considered*, and consideration is decided by the policy
prior. So the chunk 6 gate asks the Phase 1 policy head for exactly the tactical ranking
quality that Phase 1 deliberately stopped gating. The anchor was trained on random play and
has never seen a tactical position; `top16 = 83%` is a fair description of that.

Both decisions are individually defensible. Together they are inconsistent, and this gate is
where the inconsistency surfaces.

## Attempts made (3, per AGENTS.md; the gate was not weakened)

1. **Exact value when expansion reveals a mate.** Returning the evaluator's estimate for a
   node that just proved a mate available dilutes the parent's Q with a known-wrong number.
   → 75.0% → 76.0%. *Also exposed a regression:* deeper proven wins then tie with immediate
   mates at q=+1 and `g+logits` broke the tie by noise, costing 2/100 on mate-in-1.
2. **`IMMEDIATE_WIN_BONUS`.** A shorter forced mate is strictly better play; an exactly-known
   immediate win must not lose a coin-flip to a longer one. → mate-in-1 back to 100/100 at
   both 16 and 48 sims. mate-in-2 unchanged at 76.0%.
3. **MCTS-Solver proven-value propagation** (Winands et al. 2008), kept in `proven` /
   `proven_value` separate from `terminal` so rule 2 still holds. A refuted reply set makes
   the parent edge *proven*, immune to Sequential Halving elimination, where an average
   would still read well under 1. → 76.0% → 77.0%.

All three are principled improvements and are **kept** — they are correct regardless of this
gate. None addresses the binding constraint, because none of them can make the search
consider a move the prior ranked 20th.

## What I did not do

- Did not lower 95%, widen the solved-predicate, or make the suite easier.
- Did not change `gumbel_m` from the spec default of 16 to flatter the number — and it would
  not have worked anyway (81% at best).
- Did not proceed to chunk 7.

## The decision, which is not mine

Roughly, the options:

1. **Accept a lower chunk 6 threshold**, on the reasoning that 95% was calibrated against an
   assumed prior quality that Phase 1 explicitly declined to gate. 95.4%-given-considered is
   the number that measures *search*, and it passes.
2. **Reopen Phase 1** per the return condition — e.g. a tactical-anchor mixture so the prior
   ranks forcing moves top-16, which is the one lever that moves this gate.
3. **Re-scope the gate to measure search**, e.g. conditional-on-considered, or with an
   oracle-augmented considered set — measuring the thing chunk 6 actually builds.

Option 1 is a threshold change and needs the same deliberateness as the last one. Option 2 is
expensive but is what the return condition contemplated. Option 3 changes what is being
measured and should be argued for explicitly, not slid in.

## Reproduce

```bash
python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_2.epd --sims 128 --stub
python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_2.epd --sims 128 \
    --ckpt runs/phase1_anchor.pt
pytest tests/test_mcts.py -q
```

Note: the plan's chunk 6 DONE-WHEN cites `runs/phase1_a/final.pt`. That is the **invalid**
first run (`BLOCKED`-documented, all four §5 gates missed). The correct anchor is
`runs/phase1_anchor.pt`; the plan predates the campaign and its path is stale.
