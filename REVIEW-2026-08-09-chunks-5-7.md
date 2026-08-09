# Review — Chunks 5–7 (MCTS core) — 2026-08-09

**Scope:** `src/tinyalphazero/mcts/` (tree, search, gumbel, batch_runner) at `chunk-07-batched` tip (`d2f1b00`), reviewed from the public repo. Method: independent reproduction in a clean environment, adversarial line-read against the standing question list (Q1–Q4), and empirical probes where reading left doubt.

**Verdict: PASS WITH CONDITIONS.** The search core is correct, unusually well-documented, and honest about its own seams. Three should-fix findings (S1–S3) are conditions for merging the 5→6→7 chain to `main`; one (S4) can ride into early Phase 2. Nothing found invalidates any reported gate number.

## Independently reproduced (clean environment, this reviewer's hardware)

- Full suite: **222 passed, 3 skipped** at the chain tip.
- Chunk 5 gate: **mate-in-1 100/100 at 16 sims, stub evaluator**.
- Stub-only mate-in-2 baseline, eval profile: **60.5%** at 128 sims — brackets the anchor's contribution (+33 pts) for the record.
- Budget identity `sims_run == num_sims` held in every probe, terminals included.
- Equivalence at default combos (m=16; sims 16, 48): sync and batched visit vectors **identical**.

## Findings

### S2 — VERIFIED BUG: sync/batched visit-allocation parity diverges at odd (m, sims)
Reproduced: `m=5, sims=6` → sync visits `[2,1,…]`, batched `[1,2,…]`. Mechanism: `Search.run_gumbel` distributes a phase's remainder via `allocate_within_phase` (extras to top-ranked); `BatchRunner._next_forced_edge` round-robins `sims_run % len(survivors)`, and the phase-entry offset is non-zero for some non-power-of-2 survivor counts. Budget totals still match; only the distribution differs — which can flip elimination ties. Invisible at the tested combos (power-of-2 m, 16/48 sims), so 222-green is consistent with the bug. Reachable in practice: `m = min(gumbel_m, n_actions, sims)` — low-mobility positions produce odd m routinely.
**Fix:** at each phase entry compute `allocate_within_phase(budget, len(survivors))` and consume a per-survivor queue in the batched driver. **Extend the equivalence parametrization over m ∈ {3, 5, 12} × odd sims** so the contract is tested where it can break. This matters beyond the corner case: *"batch_leaves=1 reproduces the synchronous search exactly" is the load-bearing safety argument for the entire batched path* — a contract with exceptions is not a contract.

### S1 — The batched Gumbel root is still a partial reimplementation
`_finish_root`/`_next_forced_edge` re-derive considered-set formation (noise draw, top-m, force-include) and the elimination step. The code comments admit it and explain why `run_gumbel` can't be called directly (blocking loop vs. resumable state machine) — legitimate — but the *shareable* pieces aren't shared, and the `root_noise` bug lived precisely in the duplicated span. **Fix:** extract `form_considered_set(...)` and `halve_survivors(...)` into `gumbel.py`; both drivers consume them. The schedule-driving machinery may legitimately differ; the decisions must not. (This also collapses S2's fix into one place.)

### S3 — Trace positions encode repetitions against a zeroed history
`_commit_move` builds training tokens via `leaf_tokens(..., lambda _b: 0)` — the stored REP token ignores game history even when a `repetitions_fn` is supplied. Separately, game-level draw detection currently rests on `board.is_game_over()` (fivefold/75-move auto-rules), not threefold/50-move claims. Both are chunk-8 territory (per-game history counter from the move stack, claim-draw adjudication) — but the zero-lambda must not be inherited silently. **Fix now:** thread the real `base_reps` into the trace encoding; **required chunk-8 test:** a game that repeats a position across moves stores `REP_1` on the recurrence. This is the same defect family as the chunk-5 repetition off-by-one — a corrupted repetition signal that no current gate reads, surfacing later as value miscalibration.

### S4 — Promotion priors ignore the promotion sub-head
`_build_node` splits a promotion cell's mass across the four promo edges by count; the trained `P(promo | from, to)` head is never consulted at search time. Marginal for strength, but it biases Phase 2's improved-policy targets on promotion decisions and quietly wastes a trained component. **Fix:** weight the four edges by the promo head's distribution. Early-Phase-2 priority, not a merge blocker.

### Notes (no action required)
- **N1:** `torch_evaluate_batch` allocates fresh tensors per flush; the plan's pinned-buffer item is unimplemented. Perf-only at 1,917 sims/s; revisit on profile evidence, not principle.
- **N2:** Per-leaf Python masked-softmax in `_flush` — same category.
- **N3:** `_grow_nodes` leaves `np.resize` tiled garbage beyond `n_nodes` (never read; one-line hygiene when touched next).
- **N4:** `IMMEDIATE_WIN_BONUS = 1e3` is safe under the tuned constants: it only separates actions already tied at q̂ = +1, where their σ terms are equal — it cannot promote a worse action at any (c_visit, c_scale, sims) in the explored ranges. The in-code justification is correct.
- **N5:** `RootResult.q` reports raw W/N (terminal-but-unvisited edges show 0); the consumer that matters (`TracePosition.root_q` for resignation) correctly reads exact-aware completed-Q instead.

## Verified sound (the ledger)

Q1 repetition invariant: named (`path_reps` includes the standing position, root included), both derived counts used at every site checked, oracle fuzz present and mutation-tested. Q2 budget convention: honoured literally, `evaluations` reported separately with the right rationale. Q3: `_root_completed_q` applies exact values before both σ and the improved-policy target; `gumbel.improved_policy`'s signature forces the completed vector. Q4: terminal short-circuits never touch vloss or the leaf buffer; counters exposed for the fuzz. Also verified: sign convention consistent end-to-end (W is parent-perspective at every consumer); solver algebra correct in both rules including loss propagation via `max(mover_view)`; vloss × exact-value semantics exactly as ratified; eval profile is a single g ≡ 0 substitution on one code path (no drift surface, no RNG consumption); Sequential Halving budgets partition exactly with the worked example in the docstring; `Board.copy` ban upheld; TT correctly recorded as a DAG-blocker in the module docstring; suite scoring verifies mates rather than matching `bm` (measures search, not suite authorship); the sharding-legitimacy note on `solve_positions` is the kind of reasoning that prevents a future bug.

## Conditions and sequencing

1. Fix **S1 + S2** (one change, shared root helpers + allocation alignment) and **S3's** trace-encoding thread-through; extend equivalence parametrization; suite green.
2. Merge `chunk-05-mcts → chunk-06-gumbel → chunk-07-batched → main`.
3. S4 and the chunk-8 repetition/adjudication tests ride into the next brief.

Signed: the reviewer who has been threatening this review since chunk 5, satisfied.

---

## Outcome addendum (2026-08-09, post-fix — commit this version; the original never reached the repo)

**Delivery note:** this document was produced but never committed; the agent independently re-derived the S2 parity probe and the `m=5, sims=6` repro from the summary alone, found 8 pre-fix disagreements, and confirmed 0 post-fix — which upgrades S2 from "reviewer-reported" to "independently reproduced twice." A second divergence (the batched loop halved on zero-budget phases where sync skips) was found and fixed in the same pass.

**Conditions discharged.** S1 (shared `form_considered_set`/`halve_survivors`), S2 (allocation alignment, parity probe clean across the extended parametrization), S3 (real `base_reps` threaded into trace encoding, REP_1 test armed). B1 re-measurement 0.940 → 0.940, correctly read as behaviour-preservation: at (128, 16) every phase budget divides evenly, and the suite runner drives `Search`, not `BatchRunner`.

**A finding this review missed, and the triage it mandated caught.** Per-position triage of the twelve failures returned 9 BUG / 2 COVERAGE / 1 CAPABILITY. The nine were one defect: with multiple *proven* wins at a root, all tie at q̂ = +1, σ ties with them, and the acting argmax fell through to `g + logits` — the search played a proven mate-in-5 over a proven mate-in-2 on the network's guess. This review analysed the adjacent mechanism (N4) and correctly cleared the depth-1 case (`IMMEDIATE_WIN_BONUS`) without generalising the ordering question to proven-vs-proven at depth — a reviewer miss, on the record. The fix (`Tree.proven_depth`, graded `EXACT_WIN_BONUS − depth × WIN_DEPTH_STEP`) is the standard mate-distance ordering and is ratified. Gate: **0.940 → 0.985** on the single permitted re-run; the three residual failures are exactly the three non-BUG triage rows. The defect was invisible until prior coverage (round `d`), determinism (eval profile), and constants (tuning) had each been eliminated — the "failed" interventions were the instrument.

**Final verdict: PASS, conditions discharged.** `main` @ `f3f5916` = chunks 0–7 + Phase 1 (r2); 377 tests; parity probe 0; gate 98.5% vs `phase1_anchor.pt` (v2, digest 27dcdd05…); tag `phase1-closed-r2`; remotes verified in sync.

**Carried forward (chunk 8 register), reviewer's leaning recorded:** `improved_policy` spreads target mass equally across proven wins (completed-Q is ±1, depth-blind). Default ruling: leave the target orthodox — every proven win is a correct move, the acting fix already guarantees self-play *plays* the shortest, so the behavioural risk is gone and only CE mass on unplayed longer mates remains. Pre-register in the chunk 8 brief; revisit only if M3 shows mate-lengthening in played games.
