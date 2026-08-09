# TinyAlphaZero v2 — Rebuild Specification

**Goal:** A ground-up rebuild of TinyAlphaZero that fixes the four structural defects of v1, keeps the total parameter count in the 4–6M range, and is engineered so that the self-play loop can actually saturate the Strix Halo (96GB unified memory, ROCm/gfx1151) rather than idling behind a Python bottleneck.

**Philosophy (unchanged):** smallest complete implementation of the AlphaZero loop. Every component must be independently testable, and no component graduates to the next milestone without passing its acceptance criteria. Clarity beats cleverness; correctness beats throughput; throughput beats elegance.

**Hardware target:** AMD Strix Halo, 96GB unified memory, ROCm + PyTorch (bf16), single box. No distributed anything.

---

## 1. Post-mortem: what v1 taught us

| # | v1 defect | Consequence | v2 fix |
|---|-----------|-------------|--------|
| 1 | 69-token state omits en passant, halfmove clock, repetition count | State is **not Markovian**: identical encodings have different legal moves and different draw status → aliased policy/value targets, hard ceiling on legal-move accuracy | 72-token encoding (§3) carrying EP file, halfmove-clock bucket, repetition count |
| 2 | Dense 4096-way policy head from a mean-pooled vector | ~2.2M params (half the model) spent on the weakest component; spatial structure destroyed before the head that needs it | Factorized from→to bilinear policy over per-square token features (§4.3); frees ~2M params for the trunk |
| 3 | MCTS expansion deep-copies the board into every child node | ~30 `board.copy()` per expansion; memory churn dominates runtime; batched variant inherits the same node design | Array-backed tree, edges not child-boards, single scratch board with push/pop, batched leaf evaluation with virtual loss (§6) |
| 4 | Arena gating: 4 games, 0.55 threshold | Gate decisions are coin flips; "best model" churns randomly; Elo signal is noise | Drop gating entirely (AlphaZero-style continuous training); measure progress against a fixed anchor ladder with ≥200-game matches (§8) |
| 5 | Queen-only promotion in a 4096 move space | Rare correctness gap; more importantly the flat space wastes softmax mass on ~4000 always-illegal indices | Factorized head makes the space cheap; add a 4-way promotion sub-head (§3.2) |
| 6 | Random-play Phase 1 data is overwhelmingly draws/noise for the value head | Value head anchored only by mate puzzles; weak value signal entering Phase 2 | Keep mate-puzzle anchoring, add resign/adjudication rules in self-play, and weight value loss by game decisiveness (§5, §7) |
| 7 | `sys.path` hacks, committed `.pyc` and checkpoints, duplicated `CheckpointPool` in two modules, config split across CLI flags | Friction, drift, and two sources of truth | Proper package with `uv`, single typed config, `.gitignore` done right (§9) |

Items 1–4 are the load-bearing fixes. Everything else is hygiene.

---

## 2. Design principles for v2

1. **Markovian state or bust.** The encoded position must uniquely determine the legal move set and the draw-rule status. This is a hard invariant with a property-based test (§9.4), not an aspiration.
2. **Spend parameters where the gradient is.** The trunk gets the budget; heads are cheap projections of per-square features.
3. **The tree is data, not objects.** MCTS state lives in preallocated numpy arrays. Python objects appear only at the API boundary.
4. **Fewer, better simulations.** Use Gumbel root selection (Sequential Halving) so 32–64 sims produce usable policy targets, instead of hoping 200 noisy PUCT sims average out. This is the single biggest compute lever for a small-scale project.
5. **Measure against fixed anchors.** Self-referential Elo (current vs. previous checkpoint) drifts. The anchor ladder (§8) never changes within a run.
6. **One process owns the GPU.** Self-play workers are CPU-only game actors; a single inference server batches their leaf evaluations. No per-worker model copies fighting over ROCm.

---

## 3. State and action representation

### 3.1 Board encoding — 72 tokens

Positions 0–63 are square contents exactly as v1 (a1…h8, 13 piece tokens). Positions 64–71 are state tokens:

| Pos | Content | Token values |
|-----|---------|--------------|
| 64 | Side to move | `TURN_W`, `TURN_B` |
| 65 | White kingside castling | `CWK_Y/N` |
| 66 | White queenside castling | `CWQ_Y/N` |
| 67 | Black kingside castling | `CBK_Y/N` |
| 68 | Black queenside castling | `CBQ_Y/N` |
| 69 | **En passant file** | `EP_NONE`, `EP_A`…`EP_H` (9 tokens) |
| 70 | **Halfmove clock bucket** | `HM_0`…`HM_7` (8 buckets: 0, 1–9, 10–19, 20–29, 30–39, 40–49, 50–74, 75+) |
| 71 | **Repetition count** | `REP_0`, `REP_1`, `REP_2` (times this position has already occurred) |

Vocabulary: 13 piece + 2 turn + 8 castle + 9 EP + 8 halfmove + 3 repetition = **43 tokens**.

Notes:
- EP file is only set when an en passant capture is actually legal (python-chess `board.ep_square` combined with `board.has_legal_en_passant()`), matching FEN-normalized semantics. This is what makes `encode(board_a) == encode(board_b) ⇒ legal_moves(a) == legal_moves(b)` hold.
- The halfmove bucket boundaries are deliberately denser near 0 and near the 50-move rule; the model mostly needs "fresh vs. approaching the draw rule."
- Repetition count replaces AlphaZero's 8-frame history stack. It is the minimal signal needed for the network to understand threefold; we do not feed move history. If Phase 2 shows repetition-blindness pathologies (engine shuffles into draws it evaluates as wins), the escalation path is a 2-frame history, but do not start there.
- `decode_board(encode_board(b))` must round-trip everything except exact halfmove value (bucket only) — tested.

### 3.2 Move encoding — factorized (from, to, promo)

A move is the triple `(from ∈ 0..63, to ∈ 0..63, promo ∈ {none, N, B, R, Q})`. There is no flat 4096/4672 index in the model; flat indices exist only as a storage convention in the replay buffer (`from*64+to` plus a promo byte).

The policy is factorized (§4.3): the network scores (from, to) pairs directly from per-square features, and a small promotion head resolves the piece choice only when the (from, to) pair is a promotion. Legality masking happens at the (from, to) level using the legal move list from python-chess; underpromotions are first-class from day one because they cost ~nothing in this design.

---

## 4. Model architecture

### 4.1 Trunk

Encoder-only transformer, pre-norm, GELU, no causal mask — same family as v1 but rebalanced now that the policy head no longer eats the budget:

| Parameter | Value | v1 | Rationale |
|-----------|-------|----|-----------|
| d_model | 256 | 256 | unchanged |
| n_layers | 8 | 6 | budget freed by the cheap policy head goes into depth |
| n_heads | 8 | 8 | head_dim 32 |
| d_ff | 1024 | 1024 | 4× d_model |
| dropout | 0.0 in Phase 2, 0.1 in Phase 1 | 0.1 | dropout hurts RL value estimates; it exists in Phase 1 only as regularization against the small random-play distribution |
| seq len | 72 | 69 | new state tokens |

Positional encoding: keep the v1 factored rank+file embedding for squares 0–63 (it was the right call) and an 8-slot flag embedding for tokens 64–71. Precompute the full (72, d_model) table once at module init instead of recomputing per forward (v1 rebuilt it every call).

Add one **[CLS] token is not used** — instead the value head pools over state tokens 64–71 plus a learned query (§4.4). Mean-pooling over all 72 tokens is gone.

Estimated trunk parameters: ~6.3M at these settings. If that overshoots the "tiny" brief, drop to n_layers=6 (~4.8M); the layer count is a config knob, not an architectural decision.

### 4.2 Optional attention bias (deferred)

The Chebyshev-distance attention bias from the v1 spec remains the documented fallback if the trunk struggles with long-range piece geometry (bishops/rooks across the board). Trigger condition: Phase 1 legal-move accuracy plateaus below 98% with the fixed Markovian encoding. Do not implement preemptively.

### 4.3 Policy head — bilinear from→to over square tokens

Let `h_i ∈ R^d` be the final-layer feature of square token `i` (i ∈ 0..63).

```
u_i = W_from · h_i          # (64, d_p)   "this piece wants to move"
v_j = W_to   · h_j          # (64, d_p)   "this square wants to receive"
score(i, j) = (u_i · v_j) / sqrt(d_p)     # (64, 64) logits
```

with `d_p = 128`. Parameter cost: 2 · 256 · 128 ≈ 65K — versus ~2.2M in v1. The (64, 64) logit matrix is flattened, masked to the legal (from, to) set, and softmaxed.

Promotion sub-head: for a (from, to) pair that is a pawn move to the last rank, a 4-way classifier over {N, B, R, Q} conditioned on `[h_from ; h_to]` (concat → linear(512→4)). Its loss is only applied on promotion moves; at inference it factorizes: `P(move) = P(from,to) · P(promo | from,to)`.

This is the same shape of idea as Leela's transformer-era attention policy maps: the policy is a relation between squares, so score it as one, using the representations that already live on those squares.

### 4.4 Value head — WDL, attention-pooled

Keep the 3-class WDL head (v1's best decision). Pooling: a single learned query vector attends over all 72 token features (one-head attention pool), then MLP(d→d→3). Value scalar for MCTS remains `P(win) − P(loss)`, always from the side-to-move perspective.

Optional auxiliary head (Phase 1 only, then frozen off): predict the legal-move mask as 64×64 multi-label logits from the same bilinear structure with separate projections. This gives the trunk a dense, unambiguous supervised signal about piece mobility and costs another ~65K params. It is dropped entirely in Phase 2.

---

## 5. Phase 1 — supervised bootstrap

Purpose unchanged: give the trunk a representation of the game and the policy head a legality prior before self-play, so early MCTS is not searching over uniform noise. Scope tightened: Phase 1 is a *warm start*, not a milestone to gold-plate — legality is enforced by masking at search time regardless.

**Data.** 50K random-legal games (~2M positions) as before, generated straight to memmap (no JSON intermediate; a small `.meta.json` sidecar carries shapes/dtypes/version). Per position store: 72 int8 board tokens, played move as (from, to, promo) int16/int16/int8, legal-move mask as a packed 512-byte bitset (4096 bits), and outcome int8. ~2M positions ≈ 1.3GB — trivially fits in RAM, memmap is for restart convenience.

Augment with **mate-in-1 anchor set**: ~20K positions (generated by random play + filtering for `board.is_checkmate()` one ply ahead, or pulled from the Lichess puzzle dump). These carry decisive value labels and the mating move as the policy target. Sampled at a fixed 5% mixture rate.

**Objectives.**
- Policy: cross-entropy on the played (from,to) under the legal mask; promotion CE where applicable.
- Legality auxiliary: BCE on the 64×64 legal mask (unmasked — this is the point).
- Value: WDL cross-entropy. Down-weight to 0.25 on random-play positions (labels are near-noise) and 1.0 on mate anchors.

**Training.** AdamW, lr 3e-4 cosine to 3e-5, batch 1024, bf16, ~3 epochs. On this hardware Phase 1 should be an under-an-hour job; if it isn't, something is wrong with the input pipeline, not the model.

**Acceptance criteria (gate to Phase 2):**
- Top-1 prediction is legal ≥ 99% on held-out positions (raised from v1's 95% — the Markovian encoding plus the legality auxiliary should make this easy; failure indicates an encoding bug).
- Mate-in-1 solve rate ≥ 90% (argmax policy, no search).
- Legality-mask auxiliary AUC ≥ 0.995.
- Value head does not collapse: predicted WDL distribution on mate anchors has ≥ 0.8 mass on the correct class.

---

## 6. MCTS v2

### 6.1 Tree as arrays

Preallocated per-tree numpy arrays sized for `max_nodes = sims + 1` nodes and a parallel edge table:

```
nodes:  first_edge  int32[max_nodes]
        num_edges   int16[max_nodes]
edges:  child       int32[max_edges]   # -1 until expanded
        move_ft     int16[max_edges]   # from*64+to
        promo       int8 [max_edges]
        prior       float32[max_edges]
        N           int32[max_edges]
        W           float32[max_edges]
        vloss       int16[max_edges]   # virtual loss counter
```

Selection walks edges with vectorized PUCT/Gumbel math over each node's edge slice. **No board is stored in the tree.** Each simulation replays its path on a single scratch `chess.Board` via `push()`, and unwinds with `pop()` — python-chess makes push/pop cheap; it was the per-child `copy()` that killed v1. Expansion creates edges (from the legal move list) and defers child node allocation until an edge is first traversed.

### 6.2 Gumbel root selection (default)

At the root, use Gumbel AlphaZero-style acting and policy targets:

- Sample Gumbel noise on root logits, take top-`m` actions (m = 16), run Sequential Halving over the simulation budget (default 48 sims/move for self-play).
- Act via the Gumbel argmax over `g + logits + σ(q̂)`; the improved-policy target is the completed-Q softmax over root actions.

Rationale: at 200 sims, classic PUCT visit counts on a 30-branching game are barely past the prior; Gumbel's guarantees are exactly designed for tiny budgets and it eliminates Dirichlet-noise and temperature-schedule hyperparameters (both of which v1 carried in two slightly different copies). Classic PUCT with Dirichlet noise stays implemented behind a config flag as the control condition — comparing the two at equal sims is itself an interesting result for the write-up.

Below the root, standard deterministic selection on `q + prior · c · sqrt(N_parent)/(1+N_edge)` with completed-Q for unvisited edges (v1 treated unvisited Q as 0, which systematically overrated them from losing positions).

### 6.3 Batched leaf evaluation

Self-play runs `G` games concurrently (config, default 256) inside one process using coroutine-style stepping, not threads:

1. Each game advances its simulation until it hits an unevaluated leaf, appends the leaf's 72-token encoding to a shared buffer, applies virtual loss along its path, and yields.
2. When the buffer reaches `B` leaves (default 512) or all games are blocked, one forward pass evaluates the batch on GPU.
3. Results are scattered back; paths are backed up; virtual losses removed; repeat.

This is v1's "virtual batching" idea done at the leaf level rather than the game level, and with array trees it composes: total Python work per sim is one board replay plus O(branching) numpy ops.

Terminal handling: terminal leaves short-circuit with exact values (win/loss/draw including 50-move, threefold via `board.can_claim_*`), never touching the GPU.

### 6.4 Correctness gates (before any Phase 2 run)

- Mate-in-1 suite: 100% at ≥ 16 sims.
- Mate-in-2 suite (200 positions): ≥ 95% at 128 sims with the Phase 1 network.
- Zero illegal moves across 10K self-play games (assert, not sample).
- Tree invariants under fuzzing: N(parent) == Σ N(edges) + 1, W bounds, no negative vloss.

---

## 7. Phase 2 — self-play loop

Single process orchestrating three components: the self-play actor pool (§6.3), the replay buffer, and the learner. On one box the simplest correct arrangement is alternating (generate K games → train S steps → snapshot), not fully asynchronous — async adds staleness bookkeeping for no benefit at this scale.

**Per iteration (defaults):**

| Knob | Value | Notes |
|------|-------|-------|
| Games / iteration | 512 | across 256 concurrent boards |
| Sims / move | 48 | Gumbel, m=16 root actions |
| Max game length | 160 plies | then adjudicate as draw |
| Resignation | if best root q < −0.90 for 8 consecutive own moves; disabled in 10% of games to calibrate false-resign rate (<5%) | attacks the draw-glut problem at the source |
| Replay capacity | 500K positions (~1000 games window) | uniform sampling |
| Train steps / iteration | 400 | batch 1024 |
| Optimizer | AdamW, lr 2e-4 constant, wd 1e-4, grad clip 1.0 | no schedule until runs are long enough to need one |
| Loss | policy CE vs Gumbel improved policy + promo CE + WDL CE (weight 1.0) | legality auxiliary OFF |
| Snapshot | every iteration, keep all; league pool = uniform sample over last 20 | league play: 80% latest-vs-latest, 20% latest-vs-pool (kept from v1) |

**Value targets:** final game outcome z from side-to-move perspective, WDL one-hot. Adjudicated-length games count as draws. Resigned games count as losses for the resigner (that is the point of the calibration holdout).

**No gating.** The learner always trains on and generates from the latest weights (AlphaZero-style). The v1 gate is deleted, not fixed: at 4–100 arena games per iteration it either does nothing or randomly rejects progress, and the anchor ladder (§8) provides the actual progress signal.

---

## 8. Evaluation — fixed anchor ladder

A separate offline script, run every N iterations against frozen opponents. All matches: ≥ 200 games, openings from a fixed 4-ply random book played in color-swapped pairs, 95% CIs via paired bootstrap over opening pairs.

| Anchor | What it measures |
|--------|------------------|
| Uniform-random mover | floor sanity (should hit ~100% almost immediately) |
| Greedy material capture (1-ply) | basic tactics |
| Phase 1 network + 48-sim search (frozen) | pure self-play gain over the bootstrap |
| Stockfish skill 0, then 3, then 5 (fixed nodes) | external calibrated scale |
| Mate-in-2 / mate-in-3 suites | tactical depth, cheap to run every iteration |

Reported per evaluation: Elo vs each anchor with CI, draw rate, average game length, resign false-positive rate, policy entropy at move 1 (collapse detector), and value calibration (predicted P(win) vs realized outcome, 10 bins). The dashboard reads these from a run directory of JSONL logs — the Gradio dashboard is rebuilt to be a pure reader with zero import coupling to training code.

---

## 9. Engineering

### 9.1 Repo layout

```
tinyalphazero/
  pyproject.toml            # uv-managed; package installs as `tinyalphazero`
  src/tinyalphazero/
    encoding.py             # board/move codecs + vocab (single source of truth)
    model.py                # trunk + heads
    mcts/
      tree.py               # array tree, selection, backup
      gumbel.py             # root selection / sequential halving
      batch_runner.py       # concurrent games + leaf batching
    selfplay.py             # actor loop, resign/adjudication
    replay.py               # buffer (memmap-backed ring)
    train.py                # phase1 + phase2 learners (shared step fn)
    evaluate.py             # anchor ladder, suites, Elo/CI
    config.py               # ONE dataclass tree, loadable from YAML, saved into every checkpoint
  scripts/                  # thin CLI wrappers only (argparse → config → library call)
  tests/
  runs/                     # gitignored: checkpoints, JSONL logs, memmaps
```

Rules carried from v1's pain: no `sys.path` mutation anywhere (the package is installed with `uv pip install -e .`); `.gitignore` covers `runs/`, `__pycache__`, `*.pt`, `data/`; `CheckpointPool` exists exactly once; every checkpoint embeds its full config and the git SHA.

### 9.2 Determinism and reproducibility

Single seed in config fans out to numpy/torch/python RNGs per component. Self-play is seeded per game index so a run is replayable game-by-game. Every run directory gets `config.yaml`, `git_sha.txt`, and environment lockfile snapshot.

### 9.3 ROCm specifics

- bf16 autocast everywhere; fp32 master weights.
- `torch.compile` behind a flag, default off — validate on gfx1151 before trusting it, and benchmark eager first so the comparison is honest.
- The inference batch path pins one persistent CUDA/HIP stream and reuses preallocated input tensors; the v1 pattern of building a fresh tensor per leaf is gone.
- Environment check script (successor to `debug_rocm.py`) asserts device visibility, VRAM, and a matmul-throughput smoke number at startup and writes it to the run log.

### 9.4 Test strategy

- **Property tests (hypothesis):** encode/decode round-trip; the Markov invariant — generate random reachable positions, assert `encode(a)==encode(b) ⇒ legal_moves(a)==legal_moves(b)` on hash-bucketed collisions; move codec bijectivity including underpromotions and en passant.
- **MCTS unit tests:** invariants of §6.4, plus a fixed-seed regression test that pins exact visit counts for one position (catches silent math changes).
- **Golden mini-run:** 2 iterations, 8 games, 16 sims on CPU in CI (<3 min), asserting loss decreases and no NaNs.
- Benchmarks (`benchmarks.py` successor): positions/sec self-play, leaves/sec inference, tracked per commit in a JSONL so throughput regressions are visible.

---

## 10. Performance budget (Strix Halo)

The bottleneck ordering to expect and verify, in order: (1) python-chess move generation and push/pop, (2) Python-side tree ops, (3) GPU inference — the GPU should be nearly idle at this model size, which is the correct outcome, not a problem to fix.

Back-of-envelope targets at 256 concurrent games, 48 sims/move:
- ≥ 3,000 simulations/sec aggregate (≈ 60 moves/sec ≈ 5K–8K self-play games/day).
- Inference batches of 256–512 leaves at < 8 ms/forward in bf16.
- If sims/sec lands under ~1,500 after profiling, the sanctioned escalation is swapping move generation to a compiled backend (`python-chess` → a thin binding over a Rust movegen) behind the existing `Board`-shaped interface — an isolated change because nothing else touches chess rules. Do not reach for this before profiling proves it necessary.

> ### Erratum 6 (2026-08-09) — both §10 targets are wrong for this box; the operative constraint is iteration wall-clock
>
> Measured at the stated configuration (256 games, 48 sims/move), anchor v2, `benchmarks/results.jsonl`:
>
> | Target | Measured | |
> |---|---|---|
> | ≥ 3,000 sims/sec aggregate | **1,917** | misses |
> | ≥ 3,000 sims/sec, **zero-cost stub evaluator** | **2,676** | — |
> | < 8 ms/forward at batch 512, bf16 | **82.0 ms** (7.2 / 38.0 at 64 / 256) | misses ~10× |
>
> **3,000 sims/sec was never reachable here, and not because of the GPU.** With the network removed entirely the runner still tops out at 2,676 sims/sec, so the target sits above the ceiling of python-chess movegen plus tree ops. The bottleneck ordering this section predicts is confirmed — the GPU is the *smallest* term, costing ≈ 0.15 ms/sim, about 28% on top of runner overhead. The 1,500 escalation floor is **not** crossed, so the compiled-movegen swap is not triggered.
>
> The forward-latency target was calibrated for different hardware. Per-position throughput *falls* with batch size (8,896 pos/s at 64 → 6,244 at 512), so the trunk is compute-bound by batch 64 and there is no batching headroom to recover — the same iGPU characteristic that makes training batch 2048 a throughput cliff. A consequence worth carrying into chunk 8: `batch_leaves=512` never fills, because the buffer flushes when every game is blocked, so mean fill tracks `concurrent_games` (255.8/512 at 256 games), not `batch_leaves`.
>
> **Replacement constraint: a Phase 2 iteration must complete in ≤ ~35 minutes at default config.** That is what the sims/sec target was a proxy for, and it is what the M3 24h run actually needs (≈ 40 iterations/day, enough for the ladder cadence of one eval per 10 iterations to produce a curve). At 512 games × ≤160 plies × 48 sims ÷ 1,917 sims/s ≈ 34 min, the current build meets it with no margin to spare — so treat a *regression* in sims/sec as the thing to escalate on, rather than the absolute number.

Memory is a non-issue by design (~1.3GB Phase 1 data, ~350MB replay, model is megabytes); the 96GB budget buys concurrency and huge inference batches, not model size.

---

## 11. Milestones

| M | Deliverable | Acceptance |
|---|-------------|------------|
| M0 | Encoding + move codec + tests | All §9.4 property tests green; Markov invariant fuzzed over ≥ 1M positions |
| M1 | Model + Phase 1 pipeline | §5 acceptance criteria; Phase 1 wall-clock ≤ 1h |
| M2 | MCTS (array tree, Gumbel, batched leaves) | §6.4 gates; ≥ 3K sims/sec benchmark |
| M3 | Phase 2 loop + anchor ladder | Golden mini-run in CI; a 24h run shows monotone Elo vs Phase 1 anchor with non-overlapping CIs between start and end; **and mate-in-2 ≥ 0.95 at 128 sims, evaluation profile, on the frozen gate suite** (no-regress — already met at entry: 0.985 with anchor v2, 2026-08-09) |
| M4 | Scale run + write-up | ≥ 5-day run; beats Stockfish skill 0 with CI clear of 50%; Gumbel-vs-PUCT ablation at equal sims documented |

M3's 24h criterion is the real go/no-go: v1 never had a defensible "is it actually improving" measurement, and everything in this spec ultimately serves making that one plot trustworthy.

---

## 12. Open questions (decide during build, not now)

1. **History vs. repetition token.** §3.1 bets a single repetition token suffices. Watch for shuffle-into-draw pathologies in M3; escalation is 2-frame history.
2. **Gumbel m and sims.** m=16/48 sims is the literature-informed default for ~30-branching games; sweep {m=8,16,32} × {32,48,64} once the loop is stable.
3. **Value target: z vs. mixed z/q.** Start with pure outcome z. If value calibration is poor at M3, blend in root completed-Q (0.5/0.5) — a one-line change.
4. **Auxiliary legality head in Phase 2.** Off by default; if early Phase 2 policy entropy collapses, re-enabling it at low weight is the cheapest regularizer available.
