# TinyAlphaZero v2 — Agent Implementation Plan

This document is a sequential build plan for an autonomous coding agent (e.g. Claude Code) implementing `tiny_alphazero_v2_spec.md`. Keep the spec in the repo root; this plan references it by section (§).

## How to use this document

Work through chunks **strictly in order**. Each chunk is sized to fit comfortably in one agent session and ends with a **DONE-WHEN** block of shell commands. A chunk is complete only when every DONE-WHEN command passes. Do not start chunk N+1 with chunk N failing.

## Global rules for the agent

1. **Never modify a completed chunk's public interface** without being explicitly asked. Interfaces defined in this plan (signatures, array layouts, constants) are contracts between chunks.
2. **Commit once per chunk**, message format: `chunk NN: <summary>`. Run the full test suite before each commit.
3. **No `sys.path` mutation, ever.** The package is installed editable; imports are absolute (`from tinyalphazero.encoding import ...`).
4. All randomness flows from `config.seed`. No bare `random.random()` / `np.random.*` global-state calls in library code — take an `rng: np.random.Generator` or `torch.Generator` parameter.
5. Library code raises exceptions; only `scripts/` prints and exits. No `print()` in `src/`.
6. Type-hint everything public. Run `ruff check` and `ruff format` before every commit.
7. If a DONE-WHEN gate fails after 3 distinct fix attempts, **stop and write `BLOCKED.md`** describing the failure, hypotheses tested, and evidence — do not weaken the gate, do not skip the test, do not proceed.
8. Performance chunks (7, 11): benchmark **before** optimizing, record numbers in `benchmarks/results.jsonl`, and never trade a correctness test for speed.
9. Hardware note: the target box is AMD Strix Halo under ROCm. All chunks up to 11 must pass on **CPU only**; never make CUDA/HIP availability a requirement for tests.

---

## Chunk 0 — Repository skeleton and configuration

**Objective:** installable package, single config system, CI harness. No chess logic yet.

**Create:**

```
pyproject.toml                      # uv-managed; deps: torch, numpy, python-chess, pyyaml, pytest, hypothesis, ruff
src/tinyalphazero/__init__.py       # __version__
src/tinyalphazero/config.py
tests/test_config.py
scripts/check_env.py
.gitignore                          # runs/, data/, __pycache__/, *.pt, *.pyc, .ruff_cache, *.dat
Makefile                            # make test, make lint, make bench
```

**`config.py` requirements (§9.1):** a dataclass tree with sub-configs `ModelConfig`, `MCTSConfig`, `SelfPlayConfig`, `TrainConfig`, `EvalConfig`, all nested under `Config`, each field carrying the spec defaults (§4.1, §6.2, §7 tables). Provide:

```python
def load_config(path: str | None = None, overrides: dict | None = None) -> Config
def save_config(cfg: Config, path: str) -> None      # YAML round-trip
def config_fingerprint(cfg: Config) -> str            # sha256 of canonical YAML
```

Round-trip must be exact: `load_config(save_config(cfg)) == cfg`.

**`scripts/check_env.py` (§9.3):** prints torch version, device availability, VRAM if GPU present, and a 4096×4096 bf16 matmul throughput number; writes the same as JSON to stdout with `--json`. Must exit 0 on CPU-only machines.

**DONE-WHEN:**
```bash
uv pip install -e ".[dev]" && python -c "import tinyalphazero"
pytest tests/ -q                     # config round-trip + fingerprint tests pass
ruff check src/ tests/ scripts/
python scripts/check_env.py
```

---

## Chunk 1 — Encoding (spec §3)

**Objective:** the Markovian board codec and factorized move codec. This chunk is the foundation of correctness for everything downstream; be pedantic.

**Create:** `src/tinyalphazero/encoding.py`, `tests/test_encoding.py`.

**Constants (exact values, exported):**

```python
# piece tokens 0-12 exactly as v1: ES=0, WP..WK=1..6, BP..BK=7..12
TURN_W, TURN_B = 13, 14
CWK_Y, CWK_N, CWQ_Y, CWQ_N, CBK_Y, CBK_N, CBQ_Y, CBQ_N = 15..22
EP_NONE = 23; EP_A..EP_H = 24..31
HM_0..HM_7 = 32..39        # buckets: {0}, 1-9, 10-19, 20-29, 30-39, 40-49, 50-74, 75+
REP_0, REP_1, REP_2 = 40, 41, 42
VOCAB_SIZE = 43
SEQ_LEN = 72
```

**Functions:**

```python
def encode_board(board: chess.Board, repetitions: int) -> np.ndarray   # (72,) int8
def decode_board(tokens: np.ndarray) -> chess.Board                    # halfmove = bucket lower bound
def halfmove_bucket(clock: int) -> int
def encode_move(move: chess.Move) -> tuple[int, int, int]              # (from, to, promo) promo: 0=none,1=N,2=B,3=R,4=Q
def decode_move(from_sq: int, to_sq: int, promo: int) -> chess.Move
def legal_mask_ft(board: chess.Board) -> np.ndarray                    # (64,64) bool
def legal_moves_packed(board: chess.Board) -> np.ndarray               # (512,) uint8 bitset over from*64+to
```

**Critical semantics:**
- EP token is set **only** when `board.has_legal_en_passant()` is true (§3.1). Not merely when `board.ep_square` is set.
- `repetitions` is how many times the current position has already occurred earlier in the game (0, 1, or 2+ clamped to 2). The caller supplies it; `encode_board` does not scan history.
- Promotion: `encode_move` on a non-promotion returns promo=0; a queen promotion returns promo=4. `decode_move` with promo=0 on a pawn-to-last-rank (from,to) is **invalid input** — raise `ValueError` (forces callers to be explicit; no silent queening as in v1).

**Tests (§9.4):**
1. Round-trip: for 10K random reachable positions (random legal playouts), `decode_board(encode_board(b, r))` reproduces piece placement, turn, castling rights, EP square (when legal), and halfmove bucket.
2. Move codec bijectivity over every legal move of 1K random positions, including generated underpromotion and en passant cases (construct at least 50 of each explicitly).
3. **Markov invariant fuzz** (hypothesis or seeded loop): play ≥ 200K random positions, bucket by `hash(tokens.tobytes())`; for any two positions colliding on the full 72-token encoding, assert identical legal move sets **and** identical `can_claim_fifty_moves` / `can_claim_threefold_repetition` status *up to bucket resolution* — i.e. also assert no collision pairs the halfmove-bucket edge cases {49 vs 50} or repetition {1 vs 2} (these differ by token, so a collision would be a bug).
4. `legal_moves_packed` agrees bit-for-bit with `legal_mask_ft` on 1K positions.

**DONE-WHEN:**
```bash
pytest tests/test_encoding.py -q                       # all pass
python -m tinyalphazero.encoding --fuzz 1000000        # add a __main__ fuzz mode; exits 0, prints collision stats
```

---

## Chunk 2 — Phase 1 data generation (spec §5)

**Objective:** random-play dataset and mate-in-1 anchors written straight to memmap.

**Create:** `src/tinyalphazero/data.py`, `scripts/generate_phase1.py`, `tests/test_data.py`.

**Memmap record layout, one directory per dataset (`runs/data/phase1/`):**

```
boards.dat   int8   (N, 72)
moves.dat    int16  (N, 2)        # from, to
promos.dat   int8   (N,)
legals.dat   uint8  (N, 512)      # packed bitset
outcomes.dat int8   (N,)          # +1/0/-1 from side-to-move perspective
meta.json                          # N, dtype map, version, git sha, generation config
```

**API:**

```python
def generate_random_games(num_games: int, rng, max_plies: int = 200, min_plies: int = 10) -> Iterator[GameRecord]
def generate_mate_in_1(num_positions: int, rng) -> Iterator[PositionRecord]   # random playouts filtered for a mating reply
def write_dataset(records, out_dir: Path) -> DatasetMeta
class Phase1Dataset(torch.utils.data.Dataset)   # opens memmaps read-only; __getitem__ returns dict of tensors
```

- Track repetition counts during generation via a position-hash counter per game so `encode_board` gets correct `repetitions`.
- Mate-in-1 records: policy target is the mating move, outcome +1 (side to move mates). Store in a separate directory `runs/data/mate1/`; the training loader mixes at 5% (§5).
- `generate_mate_in_1` must not be pathologically slow: harvest mate-in-1 positions as a by-product of random games (check each position for an immediate mate among legal moves) rather than rejection-sampling whole games.

**Tests:** dataset writes/reads round-trip on a 50-game sample; every stored legal bitset re-validates against a fresh `chess.Board` reconstruction; mate-in-1 records verified (`board.push(move); board.is_checkmate()`); outcome perspective test on a known forced-win mini game.

**DONE-WHEN:**
```bash
pytest tests/test_data.py -q
python scripts/generate_phase1.py --games 200 --out runs/data/smoke --seed 1     # < 60s, prints position count
python scripts/generate_phase1.py --mate1 500 --out runs/data/mate1_smoke --seed 1
```

---

## Chunk 3 — Model (spec §4)

**Objective:** trunk, bilinear policy head, promotion head, attention-pooled WDL value head, legality auxiliary head.

**Create:** `src/tinyalphazero/model.py`, `tests/test_model.py`.

**Interface (contract for all later chunks):**

```python
class ChessNet(nn.Module):
    def __init__(self, cfg: ModelConfig): ...
    def forward(self, tokens: torch.LongTensor            # (B, 72)
                ) -> ModelOutput

@dataclass
class ModelOutput:
    policy_ft: torch.Tensor      # (B, 64, 64) raw logits, unmasked
    promo: torch.Tensor          # (B, 64, 64, 4) logits (computed lazily is fine; see note)
    value_wdl: torch.Tensor      # (B, 3) logits, order [loss, draw, win] from side to move
    legality: torch.Tensor | None  # (B, 64, 64) logits; None when cfg.aux_legality=False

def masked_policy(policy_ft, legal_mask) -> torch.Tensor   # (B, 4096) probs; -inf where illegal
def value_scalar(value_wdl) -> torch.Tensor                # P(win) - P(loss), (B,)
```

Implementation notes:
- Positional table: rank/file factored embedding for squares + 8-slot flag embedding, precomputed once into a `(72, d_model)` buffer at init and re-derived on `.to(device)` (register as buffer built from the embeddings in `forward` is acceptable if cheap; do not rebuild embeddings per token as v1 did).
- Policy per §4.3: `u = W_from h`, `v = W_to h`, `d_p = cfg.d_policy (128)`, scores `u @ v.T / sqrt(d_p)`.
- Promo head per §4.3: `Linear(2*d_model, 4)` on `[h_from; h_to]`. For memory, expose `promo_logits(hidden, from_sq, to_sq)` computing only requested pairs; the dense `(B,64,64,4)` form is for tests only.
- Value pool: single learned query, one-head attention over all 72 token features, then MLP(d→d→3).
- Dropout from `cfg.dropout` (0.1 Phase 1, 0.0 Phase 2 — config, not code).
- `param_count()` helper; expect ~6.3M at defaults, ~4.8M at n_layers=6 (§4.1). Assert in a test that the policy head (from/to projections + promo) is < 200K params.

**Tests:** output shapes; masked softmax sums to 1 over legal set and is exactly 0 on illegal; gradient flows to every parameter from a combined loss; `value_scalar` bounds; forward is deterministic in eval mode; per-square feature sensitivity (perturbing one input square token changes that square's `u_i` — guards against accidental pooling).

**DONE-WHEN:**
```bash
pytest tests/test_model.py -q
python -c "from tinyalphazero.model import ChessNet; from tinyalphazero.config import load_config; m=ChessNet(load_config().model); print(sum(p.numel() for p in m.parameters()))"
```

---

## Chunk 4 — Phase 1 training (spec §5)

**Objective:** supervised bootstrap loop hitting the §5 acceptance criteria.

**Create:** `src/tinyalphazero/train.py` (the shared `train_step` lives here and is reused by Phase 2), `scripts/train_phase1.py`, `scripts/eval_phase1.py`, `tests/test_train.py`.

**Requirements:**
- Losses per §5: **unmasked** policy CE on the played (from,to) over all 4096 pairs (`TrainConfig.phase1_policy_masked=False`); promo CE on promotion samples only; legality BCE on the full 64×64 grid (unmasked); WDL CE with per-sample weight 0.25 (random-play) / 1.0 (mate anchors). Total = sum with weights from `TrainConfig`.
  - **This originally said "masked policy CE" and that was a defect in this plan.** Masking sets illegal logits to −inf, which gives them exactly zero gradient, so nothing can train the unmasked `top1_legal_rate` the §5 gate measures. Worse, on uniform-random-play data the masked CE is already at its optimum (E[ln n_legal]) at init, so it is flat from step 1. The `runs/phase1_a` failure was this. Phase 2 sets `masked=True`, where search-improved targets over the legal set make masking correct.
- The aux legality head must **not** be detached from the trunk. Per spec §4.4 it exists to push a dense mobility signal into the trunk; detaching makes it a linear probe and starves the trunk.
- Datasets must be validated before training: `assert_dataset_mode(dir, mode)` on both the random-play and mate-in-1 dirs. `runs/data/mate1` was once generated with `--games` instead of `--mate1` and nothing caught it.
- Metrics must be computed on **held-out** dirs (`runs/data/eval_rand`, `runs/data/eval_mate1`), never the training dirs.
- AdamW lr 3e-4 cosine→3e-5, batch 1024, bf16 autocast when GPU available, grad clip 1.0.
- Checkpoint format (contract for all later chunks): `{"model": state_dict, "optimizer": ..., "config": asdict(cfg), "git_sha": str, "step": int, "phase": "1"|"2"}` saved with `torch.save`; loader `load_checkpoint(path, device) -> (ChessNet, dict)` validates config fingerprint compatibility of model-shape fields.
- Metrics per eval (JSONL to run dir): top1_legal_rate, top1_exact_rate, mate1_solve_rate, legality_auc, wdl_accuracy_on_anchors, losses.
- `scripts/eval_phase1.py --ckpt ... --data ...` prints the §5 acceptance table and exits non-zero if any gate fails: legal ≥ 0.99, mate1 ≥ 0.90, AUC ≥ 0.995, anchor-WDL mass ≥ 0.8.

**Tests:** overfit-single-batch test asserts the **policy component** drops ≥4× within 200 steps on 64 samples, for both masked and unmasked modes (the total-loss ratio is not a usable bar: the aux legality BCE is 4096 binary targets per sample and keeps a floor on one batch); checkpoint save/load round-trip preserves outputs bit-exactly in eval mode. Plus regression guards: unmasked CE starts at ln(4096) while masked starts at E[ln n_legal]; legality BCE produces non-zero trunk gradient.

**DONE-WHEN:**
```bash
pytest tests/test_train.py -q
# Full run (agent executes on the target box; CPU smoke acceptable in CI with reduced gates):
python scripts/generate_phase1.py --games 50000 --out runs/data/phase1 --seed 1
python scripts/generate_phase1.py --mate1 20000 --out runs/data/mate1 --seed 2
python scripts/generate_phase1.py --games 500 --out runs/data/eval_rand --seed 99
python scripts/generate_phase1.py --mate1 500 --out runs/data/eval_mate1 --seed 100
python scripts/train_phase1.py --data runs/data/phase1 --mate1 runs/data/mate1 \
    --eval-data runs/data/eval_rand --eval-mate1 runs/data/eval_mate1 \
    --out runs/phase1_a --epochs 3
python scripts/eval_phase1.py --ckpt runs/phase1_a/final.pt --data runs/data/eval_rand \
    --mate1 runs/data/eval_mate1   # exits 0
```
If the full-run gates fail: first suspect the encoding (re-run chunk 1 fuzz), then the legality auxiliary wiring, then lr. Do not lower the gates.

**How to read the loss (learned from `runs/phase1_a`):** log and read the *components*, never the total. Expected initial total ≈ 10.6 with policy at ln(4096) ≈ 8.32, falling steeply. Converged policy CE returns to ≈ 3.1–3.3 = E[ln n_legal] — the same number the failed run showed, but reached by *learning* rather than granted by the mask, so judge by `top1_legal_rate`, not by the policy value. Promo CE has an irreducible ln(4) ≈ 1.39 floor on uniform-random promotions. Converged total ≈ 4.8–5.5 is expected and fine.

---

## Chunk 5 — Array-tree MCTS, synchronous single game (spec §6.1, §6.4)

**Objective:** correct search on one game, no batching yet, classic PUCT selection below root, uniform root for now. Correctness before speed; Gumbel arrives in chunk 6.

**Create:** `src/tinyalphazero/mcts/tree.py`, `src/tinyalphazero/mcts/search.py`, `tests/test_mcts.py`, `tests/suites/mate_in_1.epd`, `tests/suites/mate_in_2.epd` (generate 100 and 200 positions respectively with a helper script using random play + `python-chess` verification; commit the EPD files).

**Tree contract (exact layout, §6.1):**

```python
class Tree:
    # preallocated for capacity nodes / edges:
    first_edge: np.ndarray  # int32 (cap_nodes,)
    num_edges:  np.ndarray  # int16 (cap_nodes,)
    child:      np.ndarray  # int32 (cap_edges,), -1 = unexpanded
    move_ft:    np.ndarray  # int16 (cap_edges,)  from*64+to
    promo:      np.ndarray  # int8  (cap_edges,)
    prior:      np.ndarray  # float32 (cap_edges,)
    N:          np.ndarray  # int32 (cap_edges,)
    W:          np.ndarray  # float32 (cap_edges,)
    vloss:      np.ndarray  # int16 (cap_edges,)
    def add_node(self, edges: list[tuple[int,int,float]]) -> int   # (move_ft, promo, prior)
```

**Search (synchronous):**

```python
class Search:
    def __init__(self, evaluate_fn, cfg: MCTSConfig): ...
    # evaluate_fn(tokens (72,)) -> (policy_ft (64,64) probs already legal-masked, value float)
    def run(self, board: chess.Board, repetitions_fn, num_sims: int) -> RootResult
    # RootResult: edges (move_ft, promo), visit counts, root Q values, root priors
```

- One scratch board per simulation path via `push()`/`pop()` — **no `board.copy()` anywhere in the search loop** (grep-able rule; add a test that monkeypatches `chess.Board.copy` to raise inside `run`).
- Selection: `Q + c_puct * prior * sqrt(N_parent) / (1 + N_edge)`; unvisited edges use completed-Q = parent Q (from child perspective), not 0 (§6.2 last para).
- Terminal leaves short-circuit with exact values including 50-move and threefold via claim checks; repetition counting along the search path must include in-search repetitions (maintain a path-local hash counter on top of the game-history counter supplied by `repetitions_fn`).
- Backup flips value sign per ply; virtual-loss fields exist but are unused in this chunk.

**Tests / gates (§6.4):**
1. Mate-in-1 suite: 100/100 solved at 16 sims with a **uniform-policy, zero-value** stub evaluator (search alone must find forced mates — this isolates search bugs from model quality).
2. Invariant fuzz: after every `run`, `1 + Σ N(root edges) == num_sims + 1` (allowing terminal short-circuit accounting — define and assert the exact identity), `vloss == 0` everywhere, all visited edges have finite W.
3. Fixed-seed regression: pin exact visit counts for one midgame FEN at 64 sims with the stub evaluator.
4. `Board.copy` ban test.

**DONE-WHEN:**
```bash
pytest tests/test_mcts.py -q
python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_1.epd --sims 16 --stub    # 100%
```

---

## Chunk 6 — Gumbel root selection (spec §6.2)

**Objective:** Gumbel top-m + Sequential Halving at the root; improved-policy targets; PUCT-with-Dirichlet retained behind `cfg.mcts.root_mode = "gumbel" | "puct"`.

**Modify:** `mcts/search.py`; add `mcts/gumbel.py`, extend `tests/test_mcts.py`.

**Requirements (follow Danihelka et al. 2022 formulation):**
- Root: sample Gumbel `g_a`, take top-`m` (cfg, default 16) actions by `g_a + logits_a`; Sequential Halving over budget `n` sims (default 48): phases halve the candidate set, splitting the budget equally per phase and evenly across surviving actions per phase (implement the standard schedule; document the exact per-phase allocation in a docstring with a worked example for n=48, m=16).
- Action selection at the end: argmax over survivors of `g_a + logits_a + σ(q̂_a)` with `σ(q) = (c_visit + max_N) * c_scale * q` (defaults c_visit=50, c_scale=1.0, in `MCTSConfig`).
- **Policy target:** `softmax(logits + σ(completed_q))` over *all* legal root actions (completed-Q fills unvisited actions with the root value estimate). Return it in `RootResult.improved_policy` as a sparse (move_ft, promo, prob) list.
- Non-root selection unchanged from chunk 5 (deterministic, no noise).
- PUCT mode: Dirichlet(α=0.3, frac=0.25) at root, temperature schedule (1.0 → 0.1 at ply 30) applied to visit counts for action sampling — matching v1 semantics as the ablation control.

**Tests:**
- With the stub evaluator, Gumbel at 48 sims still solves mate-in-1 100% and mate-in-2 ≥ 95% at 128 sims (start using the Phase 1 checkpoint here if available; keep the stub gate too).
- Sequential Halving budget accounting: total leaf evaluations exactly equals the budget for several (n, m) combos.
- `improved_policy` sums to 1, is zero on illegal moves, and concentrates on the mating move (>0.9) for mate-in-1 positions at 48 sims with the Phase 1 net.

**DONE-WHEN:**
```bash
pytest tests/test_mcts.py -q
python -m tinyalphazero.mcts.search --suite tests/suites/mate_in_2.epd --sims 128 --ckpt runs/phase1_anchor.pt   # ≥ 95%
```

---

## Chunk 7 — Batched leaf evaluation across concurrent games (spec §6.3)

**Objective:** the throughput engine: G concurrent games, coroutine-stepped simulations, shared leaf buffer, one batched forward per flush, virtual loss.

**Create:** `src/tinyalphazero/mcts/batch_runner.py`, `benchmarks/bench_selfplay.py`, `tests/test_batch_runner.py`.

**Design (single process, single thread + GPU):**

```python
class BatchRunner:
    def __init__(self, net: ChessNet, cfg, device): ...
    def play_games(self, num_games: int, seeds: list[int]) -> Iterator[GameTrace]
```

- Each live game holds: board, history repetition counter, its `Tree`, and a generator representing the in-flight simulation. A simulation generator runs selection until it reaches an unevaluated leaf, applies +1 `vloss` along its path, `yield`s the leaf tokens, and on resume receives `(policy_ft, value)`, expands, backs up (removing vloss), and finishes.
- Runner loop: round-robin games, stepping each game's next simulation until the leaf buffer holds `cfg.batch_leaves` (512) entries or all games are blocked → stack tokens into a **preallocated pinned tensor**, one forward in `torch.inference_mode()` with bf16 autocast, scatter results, resume generators.
- Legal-mask application happens runner-side (masks computed on CPU when the leaf yields, applied to logits after the batch forward).
- Selection under virtual loss: `Q_eff = (W - vloss) / (N + vloss)`; test that concurrent sims within one tree diverge paths.
- Games finish per §7: terminal, ply-cap adjudication, resignation handled in chunk 8 (runner exposes root Q so the actor can decide).
- `GameTrace` per position: tokens, improved policy (sparse), side to move, ply; outcome filled by caller.

**Equivalence gate:** with `batch_leaves=1` and `num_games=1`, BatchRunner must reproduce chunk 6's synchronous search exactly (same seeds → identical visit counts). This is the test that keeps the concurrent path honest.

**Benchmark:** `bench_selfplay.py --games 256 --sims 48` reports sims/sec, leaves/sec, forwards/sec, mean batch fill, appends to `benchmarks/results.jsonl`. Targets (§10, on the Strix Halo, not CI): ≥ 3,000 sims/sec aggregate; forward < 8 ms at batch 512. If < 1,500 sims/sec: profile with `py-spy`, record the flame graph findings in `benchmarks/NOTES.md` before touching anything.

**DONE-WHEN:**
```bash
pytest tests/test_batch_runner.py -q          # incl. equivalence gate
python benchmarks/bench_selfplay.py --games 32 --sims 16 --device cpu   # runs to completion on CPU
```

---

## Chunk 8 — Self-play actor and replay buffer (spec §7)

**Objective:** full game generation with resignation, adjudication, league opponents; memmap-ring replay buffer.

**Create:** `src/tinyalphazero/selfplay.py`, `src/tinyalphazero/replay.py`, `tests/test_selfplay.py`, `tests/test_replay.py`.

**Self-play rules (§7 table):**
- Max 160 plies then draw adjudication.
- Resignation: side resigns if its best root Q < −0.90 for 8 consecutive of its own moves; disabled uniformly at random in 10% of games (flag stored on the trace). Track and report false-resign rate: among resign-disabled games where the threshold *would* have fired, fraction not ultimately lost by that side — must be measurable from logs.
- League: per game, opponent = latest weights (80%) or uniform sample from last-20 snapshot pool (20%). `CheckpointPool` lives in `selfplay.py` — exactly one implementation.
- Outcomes written into every position of the trace from side-to-move perspective (+1/0/−1); resignations score as losses.

**Replay buffer:** memmap-backed ring, capacity 500K positions, fields matching the chunk 2 layout plus `improved_policy` stored sparse (indices int16 + probs float16, padded to max 96 entries) and `iteration` int32. `sample(batch_size, rng) -> dict[str, torch.Tensor]` densifies policy targets to (B, 4096) plus promo targets. Persisted across restarts (it's memmap — reopening resumes).

**Tests:** resign bookkeeping on synthetic Q sequences; outcome perspective; ring wraparound; sparse policy densify round-trip; pool sampling ratios (χ² over 10K draws).

**DONE-WHEN:**
```bash
pytest tests/test_selfplay.py tests/test_replay.py -q
python -m tinyalphazero.selfplay --ckpt runs/phase1_anchor.pt --games 8 --sims 16 --out runs/sp_smoke --device cpu
python -c "from tinyalphazero.replay import ReplayBuffer; b=ReplayBuffer.open('runs/sp_smoke/replay'); print(len(b))"
```

---

## Chunk 9 — Phase 2 loop (spec §7)

**Objective:** the alternating orchestrator: generate → train → snapshot, with JSONL logging and clean resume.

**Create:** `src/tinyalphazero/phase2.py`, `scripts/train_phase2.py`, `tests/test_phase2.py`.

**Per iteration:** 512 games (256 concurrent) at 48 sims → push traces to replay → 400 train steps (batch 1024, AdamW lr 2e-4 const, wd 1e-4, clip 1.0; losses: policy CE vs improved policy + promo CE + WDL CE; legality aux OFF, dropout 0.0) → snapshot `runs/<name>/ckpt_iter{i:04d}.pt` → append pool → log JSONL row (iteration, games, plies mean, draw rate, resign rate + false-resign rate, losses, lr, sims/sec, wall-clock).

**No gating** (§7): training and generation always use the latest weights.

**Resume:** `--resume runs/<name>` reconstructs iteration counter, pool, replay, optimizer state from the run directory. Killing the process at any point and resuming must not corrupt the replay ring (write a crash-resume test using a subprocess killed mid-iteration on a tiny config).

**Golden mini-run (§9.4), added to CI:** 2 iterations, 8 games, 16 sims, batch 64, CPU, < 3 min; asserts finite decreasing loss, nonzero policy entropy, and a loadable final checkpoint.

**DONE-WHEN:**
```bash
pytest tests/test_phase2.py -q
python scripts/train_phase2.py --config configs/mini.yaml --out runs/golden --device cpu   # the golden mini-run
```

---

## Chunk 10 — Anchor ladder evaluation (spec §8)

**Objective:** the trustworthy progress signal. Offline, reads checkpoints, writes JSONL.

**Create:** `src/tinyalphazero/evaluate.py`, `scripts/eval_ladder.py`, `src/tinyalphazero/opponents.py`, `tests/test_evaluate.py`, `tests/suites/openings_4ply.epd` (fixed book: 100 positions from 4 random legal plies, generated once with seed 42 and committed).

**Opponents:** `RandomMover`, `GreedyMaterial` (1-ply, standard piece values, random tiebreak), `NetSearchOpponent(ckpt, sims)` for the frozen Phase 1 anchor, `StockfishOpponent(skill, nodes)` via `chess.engine` (path from config/env; every Stockfish-dependent test and eval must **skip cleanly** when the binary is absent).
**Match protocol (§8):** each of the 100 openings played twice with colors swapped (200 games); Elo difference from score with 95% CI via paired bootstrap over opening pairs (≥ 10K resamples); draws = 0.5.
**Suites:** mate-in-2 / mate-in-3 solve rates at 128 sims.
**Extra metrics per eval:** move-1 policy entropy, value calibration (10-bin predicted P(win) vs realized, ECE).
Output: one JSONL row per (checkpoint, anchor) into `runs/<name>/ladder.jsonl`.

**Tests:** Elo/CI math on synthetic results (known score → known Elo; CI shrinks with n); color-swap pairing; deterministic given seed; RandomMover-vs-GreedyMaterial sanity match (greedy wins ≥ 90%).

**DONE-WHEN:**
```bash
pytest tests/test_evaluate.py -q
python scripts/eval_ladder.py --ckpt runs/golden/ckpt_iter0002.pt --anchors random,greedy --games 20 --sims 16 --device cpu
```

---

## Chunk 11 — Dashboard, benchmarks, hardening

**Objective:** read-only Gradio dashboard, benchmark tracking, ROCm switches.

**Create:** `scripts/dashboard.py`, `benchmarks/bench_inference.py`; touch `train.py`/`batch_runner.py` for flags only.

- Dashboard: pure reader of a run directory's JSONL files (training curves, ladder Elo with CI bands, draw/resign rates, calibration plot, sims/sec). Zero imports from training modules beyond a shared `logschema.py`.
- `--compile` flag (default off) wrapping the net in `torch.compile`; benchmark eager vs compiled on the target box with `bench_inference.py` (batch {64, 256, 512}, bf16) and record both in `results.jsonl` before enabling anywhere (§9.3).
- Startup of every long-running script logs `check_env.py --json` output into the run dir.
- Add `make bench` running both benchmark scripts and appending to `results.jsonl` with git sha.

**DONE-WHEN:**
```bash
python benchmarks/bench_inference.py --device cpu --batches 64
python scripts/dashboard.py --run runs/golden --smoke     # renders once headless, exits 0
pytest -q                                                  # entire suite green
ruff check src/ tests/ scripts/ benchmarks/
```

---

## Chunk 12 — Scale runs (M3/M4 playbook, spec §11)

This chunk is operational, executed on the Strix Halo. The agent's job is to launch, monitor, and write up — not to change code unless a gate fails.

1. **Throughput check:** `make bench` on GPU; confirm ≥ 3,000 sims/sec and < 8 ms forwards. If not, stop and profile (chunk 7 escalation path; a movegen backend swap is a *new negotiated chunk*, not an improvisation).
2. **24h run (M3 gate):** default config, `scripts/train_phase2.py --out runs/m3`. Ladder eval vs {random, greedy, phase1-anchor} every 10 iterations. **Pass:** final Elo vs phase1-anchor exceeds iteration-0 Elo with non-overlapping 95% CIs, and value ECE < 0.15. **Fail:** write `RUNLOG.md` with curves and the three most plausible causes ranked by evidence; check the §12 open-question levers (z/q mix, aux legality re-enable, m/sims sweep) before touching architecture.
3. **Ablation:** identical 24h budget with `root_mode=puct`; ladder both against the same anchors; add the comparison to `RUNLOG.md`.
4. **5-day run (M4):** extend ladder with Stockfish skill {0, 3}. Pass: beat skill 0 with CI clear of 50%.
5. Deliverable: `RUNLOG.md` per run (config fingerprint, git sha, curves, ladder table, incidents) — the raw material for the write-up.

---

## Appendix A — suggested config files to commit

- `configs/default.yaml` — spec defaults (§4.1, §6.2, §7).
- `configs/mini.yaml` — golden mini-run: d_model 64, n_layers 2, games 8, sims 16, batch 64, replay 5K.
- `configs/puct_ablation.yaml` — default with `root_mode: puct`.

## Appendix B — dependency pins

torch (ROCm build on target, CPU build in CI), numpy, python-chess ≥ 1.10, pyyaml, gradio (dashboard only), pytest, hypothesis, ruff, py-spy (dev). Lock with `uv lock`; commit the lockfile.
