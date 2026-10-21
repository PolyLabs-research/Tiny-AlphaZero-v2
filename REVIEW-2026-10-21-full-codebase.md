# Review — Full codebase at `p2_d` — 2026-10-21

**Scope:** everything under `src/`, `scripts/`, `benchmarks/`, `tests/`, `configs/`, the build files, and the committed records under `runs/`, at `p2_d` tip `8133746` (local; 19 commits ahead of `origin/main` `fce3777`). Method: one pass over the six p2_d commits as a diff (findings `P*`), then five independent adversarial line-reads of the whole tree by area (`A` MCTS core, `B` encoding/data/replay/model, `C` selfplay/phase2/rehearsal/train, `D` evaluate/opponents/logschema/config + records, `E` scripts/benchmarks/build), each verifying its findings with probes against the real functions before reporting. Probe scripts are in the session scratchpad (`review/agent*/`), not the repo.

**Baseline:** `ruff check` clean; `ruff format --check` fails on one list comprehension (`search.py:485`), so `make lint` is red. Full suite: **547 passed, 1 skipped, 1 failed** in 4m58s on CPU; the failure is a test defect (D13), not a code defect. Tree clean before and after.

**Verdict: PASS WITH CONDITIONS.** No finding changes a recorded ladder, suite or verdict number: the colour attribution, Elo/CI, paired bootstrap, suite scoring, codec round trips, sign conventions and the sync/batched search equivalence all held under independent probes, and the p2_d verdict statistic was recomputed from the committed rep rows to the digit. What several findings change is what the record *means*: p2_c trained something other than what PREREG/RUNLOG say it trained (C1); every Phase 1 round trained at a different anchor mixture than its config states (E1); rows and checkpoints carry git SHAs of code that did not produce them (C2/D2); 87 committed rows fail the project's own validators (P1, P10); and the p2_d verdict harness is not in the repo (P5). Those five need a fix or an amendment to the record before another round is run. The rest is latent — scheduler, resumability, provenance and test-vacuity defects that have not yet fired.

---

## Summary (by severity)

| ID | Sev | Area | Finding | Touches a recorded result? |
|----|-----|------|---------|---------------------------|
| C1 | **High** | train | z/q blend halves the CE weight of every row, incl. the rehearsal rows the records say keep "pure CE at full weight" — p2_c was two changes, not one | Yes: p2_c/p2_d trained under it; `value_loss` column redefined mid-record |
| C2 / D2 | **High** | train, evaluate | `git_sha` on rows and checkpoints is HEAD at *write* time, not launch code; p2_b/p2_d rows stamped with later commits | Yes: provenance of p2_b, p2_d rows and checkpoints |
| P1 | **High** | logschema | `nondeterministic` added to schema v3 after v3 rows were written; 17 committed v3 rows fail `validate_ladder` | Yes: `runs/p2_d/ladder.jsonl`, `runs/p2_c/ladder_pairscores.jsonl` |
| P10 | **High** | logschema | one `SCHEMA_VERSION` for iteration and ladder rows; bumping it rejects every unchanged v1/v2 iteration row (70 rows) | Yes: p2_a/p2_b/p2_c `iterations.jsonl` |
| P5 | **High** | repo | the reps/verdict/Track 1 harness that produced `p2_verdict_reps.json`, `reps_iter0020_r*.jsonl`, `track1_ckpt_iter*.json` is not in the repo | Yes: p2_d verdict unreproducible from the checkout |
| E1 | **High** | train_phase1 | the anchor sampler double-applies the `(1−a)` normalisation: configured 5% trains at 5.25%; phase1_d's 8%/7% trained at 9.17%/8.02% | Yes: every Phase 1 round, incl. the anchor v2; config and PROGRESS misstate the mixture |
| B3 | Medium | model | all 8 attention `in_proj_weight`s are byte-identical and std 0.044, not the 0.02 init the code claims | Every trained network started from this; deterministic, so reproducible |
| E2 | Medium | train_phase1 | `--ckpt` resume takes the config from `--config`/defaults, not the checkpoint, and replays epoch 1's sample order | No recorded round was resumed this way (verify against TRAINING_STRATEGY) |
| E5 | Medium | pyproject | no ROCm torch index pin (AGENTS §2 rule 2): a fresh `uv pip install -e ".[dev]"` pulls the CUDA wheel; `gradio` dev dep unused | No |
| E4 | Medium | Makefile | `make bench` is a silent no-op (`--batches` doesn't exist, `--ckpt` required, errors hidden by `|| true`); `make lint` red; `benchmarks/` outside lint scope | No |
| E3 | Medium | scripts, phase2 | run dirs lack the AGENTS §4 artifacts (Phase 1 writes no `config.yaml`/SHA at all; no `check_env` output anywhere); `--seed` override only in gitignored `state.json` (= C5) | Phase 1 runs tied to a SHA only via a 78 MB checkpoint |
| A1 | Medium | batch_runner | a root request flushed mid-loop is expanded twice (double Gumbel draw, orphan node, extra sim) whenever `batch_leaves <= concurrent_games` | No (all production configs have B > games) |
| C4 | Medium | selfplay, replay | `moves.dat` ("move actually played") stores argmax(improved policy), which differs from the played move ~95% of the time; the promo head is trained at that cell | Yes, mildly: promo loss in all four campaigns |
| P2 | Medium | config, search | `proof_directed_expansion` is not eval-profile scoped; turning it on for the ladder also turns it on in self-play; no CLI exposes it | Track 1 "13/13" has no in-repo entry point |
| P6 | Medium | search | the lever fires beneath believed edges for *either* mover, not just at opponent nodes as documented | Track 1 measurement semantics |
| D1 | Medium | logschema | torn JSONL tail + resume: resumed row lands on the torn line and is silently dropped; a second resume makes the file unreadable | No (not yet hit) |
| B1 / C9 | Medium | replay, phase2 | `meta.json` (replay durable head) and `iterations.jsonl` rewritten non-atomically; a kill mid-write makes the run unresumable | No |
| D3 | Medium | opponents | the Phase 1 anchor is pinned by digest in prose only; nothing hashes `runs/phase1_anchor.pt`; `NetSearchOpponent.digest` is dead | Not today (file verified = v2) |
| C3 / D7 | Medium-low | config, phase2, scripts | `eval_every_n_iters`, `stockfish_path`, `openings_path` are dead; ladder cadence lives only in a CLI flag, is not persisted, and is lost on `--resume` (p2_a passes 30/40/50 were backfilled for this reason) | Yes: p2_a `produced=None` rows; every `config.yaml` says 10 while runs used 5 |
| D4 / D5 / D6 | Medium-low | evaluate, eval_ladder | resume key is `(checkpoint, anchor)` only; eval row never skipped; CLI defaults (`--config` none, `--seed 0`) differ from the inline pass | No (backfills coincided) |
| P3 | Medium-low | evaluate | `getattr(opponent, "nondeterministic", False)` + dataclass default make "deterministic" the silent fallback for every family but Stockfish | Latent |
| P9 | Medium-low | opponents | Stockfish TT carries across all games of a matchup (no `game=` key / `ucinewgame`); a "deterministic" skill-20 rung depends on game order | Only the parked deterministic rung |
| C5 | Low-med | phase2 | model init is unseeded; `--seed` override is stored in `state.json` only, not `config.yaml` or checkpoints | No (all rounds used 42 + `--init`) |
| C8 | Low-med | selfplay, train | `root_q` on pool-mover rows is a stale snapshot's search value, blended into the learner's target; not stated in PREREG-p2_c | p2_c/p2_d ~10% of rows |
| D8 | Low-med | configs | `puct_ablation.yaml` differs from default in dropout and legality weight, not only `root_mode` | M4 not yet run |
| A2 | Low | gumbel | force-include can push `len(considered) > num_sims` → `ValueError` on legal input (sims ≤ 3) | No |
| A3 | Low | search | `proven_depth` freezes at first proof; stale on mate-in-3+ in principle (0 of 3,611 proven edges stale in practice) | No |
| A4 / C7 | Low | batch_runner | `debug_trees` holds every tree of a wave (~1.9 GiB host RAM per 256-game wave) | No |
| B4 | Low | data | `data._git_sha` is the cwd-relative `"unknown"` variant `train.py` replaced; every Phase 1 dataset sidecar says `"unknown"` | Provenance of `runs/data/*` (regenerable from seeds — verified) |
| B5 / B6 | Low | data, replay | stale `keymoves.dat` read on rewrite; sidecar dtypes/capacity never cross-checked; memmap silently grows a short file | No |
| B7 | Low | model | `masked_policy` returns NaN for a row with no legal move | Latent |
| D9 / D10 / D11 | Low | config | no type validation (`root_noise: 'false'` leaves noise on); override typos dropped; `config_fingerprint` not regenerable after a field is added (sha256 of the saved YAML is, undocumented) | D11: recorded fingerprints only match via the file hash |
| D12 | Low | opponents | `close()` on a dead engine masks the original exception; nodes-only `play()` has no timeout | No |
| D14 / D15 | Low | evaluate, batch_runner | `±Infinity` written to JSON for `--games 2/3`; terminal opening crashes `RandomMover` | No |
| P7 | Low | tests | four new tracked record kinds are not in `MUST_TRACK` | Guard gap |
| P12 / P13 | Low | search | `_unexpanded_reply` ignores `vloss`; Python-level scan in the hot path | No |
| A6 | Low | search | `RootResult.evaluations` always 0 on the batched path | No |
| E7 | Low | eval_phase1, train_phase1 | without `--mate1` the verdict is gated on 200 freshly generated anchors while printing `n_mate1=1000`; training-dir fallback for eval metrics is silent | Only if the documented "Custom config" command was used |
| E8 | Low | HOW_TO_TRAIN | Step 3 evaluates on the training dirs; acceptance table lists retired gates; "uses `configs/default.yaml`" is wrong and would train Phase 1 with the legality head off | Docs |
| E6 | Low | benchmarks | `results.jsonl` has 10 row shapes in 31 rows; `date` never stamped; sweep SHA is cwd-relative; `mate_in` inferred from filename | Benchmark provenance |
| E9 | Low | scripts | `cfg.seed` dead for Phase 1 (`--seed 42` duplicated); anchor list and anchor path spelled in 4–5 places | No |
| E10 | Low | .gitignore | the mate-in-2 tuning suite (`runs/data/mate2_tune_500/mate_in_2.epd`) the sweep firewall rests on is gitignored and not in `MUST_TRACK` | Record completeness |
| E11 | Low | src, tests | `sys.path.insert` in tests (banned); three printing `main()`s inside `src/` (banned); `_build_loader` is library code living in a script | No |
| E12 | Low | generate_phase1, train_phase1 | `--max/min-plies` inert for anchor modes; an anchor dir with mix 0 is silently never sampled while `--eval-mate2` still gates | No |
| E13 | Low | triage/diagnose | diagnostics hard-code the default eval profile, not the campaign's; "mate in 2" defined three times | No (values coincide today) |
| T1–T12 | Low | tests | twelve tests that cannot detect the bug they describe, leak process-global state, or depend on this machine (see "Test-suite defects") | Guards, not results |

---

## P — the p2_d commits as a diff (`main..p2_d`)

### P1 — `nondeterministic` added to schema v3 after v3 rows had been written — **High**
`src/tinyalphazero/logschema.py:255`. 16 of the 20 rows in `runs/p2_d/ladder.jsonl` and the row in `runs/p2_c/ladder_pairscores.jsonl` are stamped `schema_version: 3` and lack the field; `validate_ladder` raises `missing=['nondeterministic']` on each (verified by running the validator). RUNLOG-p2_d.md:91 and PREREG-p2_d.md:282 say the field is "on every ladder row". A reader that defaults the absent key reads the four `stockfish_s0` protocol rows — the exact rung the field exists to mark — as reproducible. Two row shapes now share one version number. **Fix:** backfill the 17 rows (the value is derivable: `skill < 20`) in a commit that says so, or bump to v4 and make v3 mean "pre-field".

### P2 — `proof_directed_expansion` has no eval-profile scoping — Medium
`src/tinyalphazero/config.py:97`; `selfplay.py:375` uses `cfg.mcts` unchanged; the ladder and suites use `replace(cfg.mcts, root_noise=False)`; both read the same flag. The comment's "OFF by default and off in self-play" is unenforced: setting it in a YAML to measure the lever on the ladder also changes the training corpus. The established mechanism (`eval_m`/`eval_c_*` folded in `resolved()` when noise is off) is bypassed, and neither `search.py`'s main nor `scripts/eval_ladder.py` exposes the flag, so the Track 1 result has no in-repo entry point. **Fix:** make it an `eval_*` field folded by `resolved()`, and expose it on `eval_ladder.py`.

### P3 — "deterministic" is the silent fallback for every opponent family — Medium-low
`src/tinyalphazero/evaluate.py:704` (`getattr(opponent, "nondeterministic", False)`) plus `LadderMatchRow.nondeterministic = False`. Only `StockfishOpponent` declares the property; `test_opponent_families_declare_whether_they_randomize` only instantiates Stockfish. A new family without the attribute gets `nondeterministic: false` on every row with no error. **Fix:** declare it on every class, read it directly, drop the default, iterate the families `build_opponent` can return in the test.

### P4 — `test_breadth_lever_is_deterministic_under_the_eval_profile` shares one seed — Low (test)
`tests/test_mcts.py:1289` builds both searches with `default_rng(0)`, so it cannot detect the "reorders under reseeding" failure its docstring names. Use two seeds.

### P5 — the verdict harness is not in the repository — **High**
`runs/p2_d/p2_verdict_reps.json`, `reps_iter0020_r*.jsonl`, `track1_ckpt_iter*.json`: nothing in `src/` or `scripts/` references reps, verdict, track1 or the lever flag; `scripts/` is untouched by the branch; PREREG-p2_d.md:291 defers the reps harness to "run by script" without committing it. The statistic itself *is* recomputable from the committed rep rows (reviewer D did: paired difference 0.1033, CI [0.0600, 0.1467], SE 0.0220, rep σ 0.0328/0.0236 — all equal to the JSON and RUNLOG §1), but the rows' generation is not. **Fix:** commit the harness (as `scripts/eval_reps.py` / `scripts/eval_track1.py`), record the exact command in RUNLOG-p2_d.

### P6 — the lever fires for either mover — Medium
`src/tinyalphazero/mcts/search.py:436`: the condition is `parent_edge is not None`, so beneath an opponent reply the opponent believes at q̂ ≥ 0.95 it forces breadth over the *root mover's* replies in a line the root mover is losing — spending `1 + len(replies)` sims per descent proving nothing the root argmax can use. The config comment and docstring describe the opponent-node case only. **Fix:** restrict to odd depth relative to the root, or document the symmetric behaviour and re-measure.

### P7 — new record kinds absent from `MUST_TRACK` — Low (test)
`.gitignore:38` adds negations for `track1_ckpt_iter*.json`, `p2_verdict_reps.json`, `reps_iter*.jsonl`, `ladder_pairscores.jsonl`; `tests/test_gitignore_musttrack.py` names none of them, against its own docstring. The guard exists because this incident class has occurred four times.

### P8 — real-checkpoint test leaks thread count and depends on this machine — Low (test)
`tests/test_mcts.py:1251`: `torch.set_num_threads(4)` with no restore; depends on gitignored `runs/p2_c/ckpt_iter0020.pt`, so W3 skips on any other clone. Pin threads in a fixture that restores; mark as slow/benchmark.

### P9 — Stockfish transposition table carries across a matchup — Medium-low
`src/tinyalphazero/opponents.py:221`: `play(board, Limit(nodes=...))` with no `game=` key, so no `ucinewgame` between the ~200 games of a matchup; `Hash=16` pins the size but not the contents. A skill-20 "deterministic" rung's move in game k depends on games 1..k-1, and resuming after a kill starts cold. Pass a per-game `game=` object.

### P10 — one `SCHEMA_VERSION` for two row families — **High**
`src/tinyalphazero/logschema.py:41`. Bumping to 3 for a ladder-only change makes `validate()` reject every unchanged v1/v2 `IterationRow` (`schema_version 2 != 3`), and stamps byte-identical v3 iteration rows that have the v2 shape. Reviewer D confirmed p2_a/p2_b's 70 v1 iteration rows have exactly the current `FIELDS` and are rejected on version alone. **Fix:** a version constant per row family; validators accept the versions whose shape they know.

### P11 — `test_p2d_config_is_the_brief` skips top-level scalars — Low (test)
`tests/test_rehearsal.py:345` only compares dict sections, so `seed` is outside the "zero deltas" assertion. Compare the flattened `asdict` including scalars.

### P12 — `_unexpanded_reply` ignores `vloss` — Low
`search.py:488` treats `child == -1` as unexpanded without consulting `tree.vloss`. Harmless while `BatchRunner` suspends one sim per game; any runner with >1 in flight would double-build the same pending child.

### P13 — Python-level scan in the selection hot path — Low (perf)
`search.py:485`: list comprehension + `max(key=lambda)` per descent beneath a believed edge; three numpy ops give the same first-max tie-break. (This is also the line `ruff format` wants to rewrap.)

---

## A — MCTS core

### A1 — double root expansion when a root request is flushed mid-loop — Medium
`src/tinyalphazero/mcts/batch_runner.py:201-210` (`_resume`), `:364-410`. A root-expansion request can sit in `pending` across a loop iteration. If a flush fires at a lower game index, `_resume` sets `gs.gen = None` and queues `_pending_roots`, but `gs.in_root` stays True because `_finish_root` only runs at the top of the `while`; when the `for` loop reaches that game it opens a **second** `expand_root_gen` on the same tree. Variant A: `_finish_root` runs twice, `gs.rng` draws Gumbel noise twice, `gs.root` is re-pointed to node 1 and node 0 is orphaned — under self-play noise a game's considered set then depends on other games' timing, breaking per-game reproducibility from `(seed, index)`. Variant B: the second root build is counted as a simulation (`sims_run += 1`) and adds a forward row.
**Evidence (re-run by me):** 12 of 64 `(games, B, sims, seed)` configs with `B <= games` show it; `games=4 B=3 sims=8: positions=24 finish_root_calls=28 expand_root_calls=34 trees_with_2+_roots=10 stats.sims=198 (expected 192)`; `game 2 ply 0: root=1 n_nodes=10 orphan_nodes=[0]`. Both accounting identities break in that state, but `test_leaf_accounting_balances_with_terminals_present` and `test_sims_per_move_is_exactly_the_budget` use 3 games with B ∈ {1,4,16,32} and never create a mid-loop flush.
**Impact on records:** none — every production path has `batch_leaves > concurrent_games` (self-play 512 > 256; ladder 512 > 200; mini 64 > 4), consistent with "mean fill 255.79/512". **Fix:** finish or flag a completed root in `_resume` before the `for` loop can see it; add an accounting test with `batch_leaves < num_games`.

### A2 — force-include can exceed `num_sims` and crash — Low
`gumbel.py:145-147, :184-186`, raise at `:73`. `form_considered_set` caps `m ≤ num_sims`, but `force_include_proven_wins` appends wins once the evictable set is exhausted; `halving_phase_budgets` then raises `m=2 exceeds num_sims=1`. Probe: at sims=1, 28/100 mate-in-1 suite positions raise; sims=2 → 4/100; sims=3 → 1/100. Unreachable at production budgets. Truncate back to `num_sims` keeping wins first.

### A3 — `proven_depth` freezes at first proof — Low (possible)
`search.py:618`. Once proven, an edge's depth is never revisited, so a slower proven line that arrives first leaves an overestimate; the acting rule orders on upper bounds for mate-in-3+. Reproduced on a hand-built tree; recomputing the induction over the final trees of all 200 mate-in-2 and 200 mate-in-3 positions at 128 sims found **0 stale** of 3,611 proven edges, 0 root choices affected. Fix is one line (recompute depth when already proven).

### A4 — `debug_trees` retained for the whole wave — Low (= C7)
`batch_runner.py:467` stores every `(tree, root, sims)` unconditionally; 63.5 KiB per `Tree(cap_nodes=49)`, ~1.9 GiB per 256-game wave at chunk-8 ply counts, on the box whose scarce resource is host RAM. Put it behind a constructor flag that only the equivalence tests set.

### A5 — two MCTS tests cannot fail — Low (test)
`tests/test_mcts.py:326-342`: `test_unvisited_edges_use_completed_q_not_zero` asserts `chosen in (sl.start, sl.start+1)` on a two-edge node — true for any return. `:372-394`: `test_in_search_repetition_counts_on_top_of_game_history` asserts only `visits.sum() == 32` in both runs, which the budget identity guarantees regardless of repetition handling. Also `:214-217` loops over all nodes but asserts only on the root.

### A6 — `RootResult.evaluations` is always 0 on the batched path — Low
`search.py:530` / `batch_runner.py:440`: the shared `Search`'s counter is reported, and it never evaluates on that path. Nothing reads it.

---

## B — encoding / data / replay / model

### B1 — replay durable head written non-atomically — Medium (= C9)
`src/tinyalphazero/replay.py:184-203` (`_write_meta`, also `rewind_to`:265). The module docstring states the crash-safety contract ("the durable head is meta.json and moves only in flush"), but `Path.write_text` truncates then writes; a SIGKILL/OOM between the two leaves 0 bytes, and both `ReplayBuffer.open` and `open_or_create` raise `JSONDecodeError`. `phase2._commit_state` (`phase2.py:567`) already uses tmp + `os.replace`; `phase2._truncate_log` (`:305`) has the same non-atomic rewrite of `iterations.jsonl` on resume. The ring data is intact but unreachable without hand repair. **Fix:** tmp + `os.replace` in both.

### B2 — the Phase 1 value-target sign is guarded only by vacuous tests — Medium (test)
`tests/test_data.py:71-104, 257-291`. `test_random_games_outcome_perspective` asserts only that adjacent decisive positions alternate — an inverted `stm_outcome` also alternates; the two "known win"/"draw" tests build a `PositionRecord` literal and assert the literal back without calling the generator. Monkeypatching `generate_random_games` to negate every outcome passes all three. The code itself is right (54 decisive games replayed, 6,792 positions, 0 sign errors) — but the convention the spec singles out has no effective coverage. **Fix:** replay the stored moves to the terminal board and assert `outcome == +1` exactly for the winner's positions.

### B3 — all eight attention in-projections start identical, and at the wrong scale — Medium
`src/tinyalphazero/model.py:88-101, 135-145`. `nn.TransformerEncoder` deep-copies one `encoder_layer`; `_init_weights` re-draws `nn.Linear`/`Embedding`/`LayerNorm` (so `out_proj`, `linear1/2` differ per layer) but `self_attn.in_proj_weight` is a bare `Parameter` on `nn.MultiheadAttention`, not an `nn.Linear`, so it keeps PyTorch's xavier draw — and the *same* draw in all 8 layers. **Verified by me:** `in_proj identical across all layers: True; linear1/out_proj identical: False; in_proj std 0.0442 vs linear1 std 0.0200`. Every Phase 1 network, and every Phase 2 run seeded from the anchor, trained from this init. It is deterministic under the seed, so results are reproducible; it is simply not the init the code claims. **Fix:** add a `MultiheadAttention` branch (`trunc_normal_(in_proj_weight, std=0.02)`, zero bias) — and record that it changes init for future runs.

### B4 — `data._git_sha` is the forbidden duplicate — Low
`src/tinyalphazero/data.py:481-491`: cwd-relative `git rev-parse`, silent `"unknown"`; `train._git_sha` (`train.py:689`) was written to replace exactly this and `tests/test_train.py:396-445` forbid it, but `data.py` still stamps every dataset sidecar with it. `runs/data/phase1` (9.5M positions), `mate1_100k`, `mate1`, `eval_rand`, `eval_mate1*` all say `git_sha: "unknown"` (generated before the initial commit). Mitigation verified: prefixes of six datasets regenerate byte-identically from their recorded seeds with current code. Delete the duplicate; share one function.

### B5 — stale `keymoves.dat` is read on rewrite — Low
`data.py:361-367, 442-447`: the reader memmaps `keymoves.dat` whenever the file exists, ignoring `meta.has_key_moves`; `write_dataset` never unlinks it. Rewriting a directory without key moves leaves the old file, and `np.memmap(mode="r")` reads a prefix when the new `n` is ≤ the old one — `train.mate2_coverage` would score against another dataset's bitset. Probe confirmed. Gate on the meta flag; unlink when not writing.

### B6 — sidecars never cross-checked against layout constants — Low
`data.py:434-438`, `replay.py:136-144, 164-178`: `Phase1Dataset` hardcodes dtypes and ignores `meta.dtype_map`; `ReplayBuffer.open` ignores `fields` and `max_policy_entries`, so a changed `MAX_POLICY_ENTRIES` without a `VERSION` bump reinterprets an existing ring at the wrong stride. `np.memmap(mode="r+")` silently extends a file shorter than `capacity*rowsize` into zero rows that `sample` treats as positions. Assert sidecar constants and `.dat` sizes in both `open` paths.

### B7 — `masked_policy` returns NaN for a row with no legal move — Low (latent)
`model.py:32-46`. `masked_fill(-inf)` over an all-False mask → NaN row, propagated silently; `_apply_update` would skip the step with no indication of which row. Callers avoid it today. Raise, or document the precondition.

### B8 — `test_root_q_shares_the_perspective_of_the_outcome_it_blends_with` tests storage, not perspective — Low (test; = C6)
`tests/test_replay.py:202-247`, cited by PREREG-p2_c.md:92-94 as the guard for "the whole safety of the z/q blend". It writes rows whose `root_q` and `outcome` already agree and asserts they round-trip through `add`/`sample`; a sign error in `batch_runner.py:462` or `selfplay.fill_outcomes` is invisible to it. The property does hold today (reviewer C drove `BatchRunner` from White and Black mate-in-1 starts: `root_q=+1.000, outcome=+1` for the mover). Replace with that end-to-end check.

### B9 — fresh-clone dependence and global state in indirect tests — Low (test)
`tests/test_contamination.py`: 34 tests, 8 skip guards on gitignored `runs/data/*`; on CI the disjointness gate passes as skips. `tests/test_train.py:233,603,625`: `torch.manual_seed(0)` three times with no restore.

---

## C — selfplay / phase2 / rehearsal / train

### C1 — the z/q blend halves the CE weight on every row, including rehearsal rows — **High**
`src/tinyalphazero/train.py:103-112`; `config.py:196-198`; `configs/p2_c.yaml:12`. `_blend_value_toward_search_q` returns `(1-w)*ce + w*mse` where `ce` is the full-batch mean over **all** rows; masked-off rows (rehearsal anchors, v2-ring rows) are excluded only from the MSE term, so their CE is multiplied by `1-w = 0.5`. The docstring (`:91`), the config comment and the p2_c.yaml header all say those rows "keep pure CE at full weight"; the code comment at `:110-111` acknowledges the full-batch mean and says re-weighting "would make the lever a batch-composition change too" — but *not* re-weighting is the batch-composition change: from p2_c on, the ~51 anchor rows per 1024-row batch receive half the value supervision they had in p2_b. The behaviour is also discontinuous (`return ce` when no row is masked on, otherwise half weight for everyone). Two consequences for the record: (1) p2_c vs p2_b is "blend toward q on ring rows" **and** "halve the anchor rows' value weight", so the "first absolute-strength gain" has two candidate causes, and the conviction/watchlist readings cannot be attributed to the blend alone; (2) the logged `value_loss` column changed meaning (pure CE before p2_c, `0.5·CE + 0.5·MSE` after) with no schema change, so RUNLOG value-loss levels are not comparable across the p2_b/p2_c boundary.
**Evidence:** per-row gradient-norm ratio (`q_w=0.5` / `q_w=0.0`) on four masked-off rows: `[0.5, 0.5, 0.5, 0.5]`; all-masked-off batch: `1.0`. The PREREG-p2_c guard `test_value_blend_masks_rows_without_a_search` only checks that hidden `root_q` values do not move the loss, so it cannot see the weight change.
**Fix:** per-row CE weighted per row (`ce_rows = F.cross_entropy(..., reduction="none"); v = (ce_rows * where(mask, 1-w, 1)).mean() + w*mse_masked`) with a test that masked-off rows' gradient is independent of `q_w` — **and** an erratum in RUNLOG-p2_c/PREREG-p2_c stating what was actually trained. Changing the code changes the lever; the record must say which version each round ran.

### C2 — `git_sha` is HEAD at write time, not the running code — **High** (= D2)
`train.py:689-709` (`_git_sha`), `phase2.py:521` (row), `train.py:408` (checkpoint), `evaluate.py:700, 765` (ladder rows). `_git_sha()` shells out to `git rev-parse HEAD` on every write; commits landed in the working tree while campaigns ran, so rows and checkpoints are stamped with code that did not produce them, contradicting `provenance.json`. **Actual record:** p2_b `iterations.jsonl` carries four SHAs in one process (`8a19cbf`×5, `cb6f634`×5, `9837809`×2, `62d661b`×8; run window 15:15→23:42); rows 11–20 and `ckpt_iter0012/0015.pt` are stamped with commits in which `SCHEMA_VERSION = 2` yet were written as `schema_version: 1` — the stamped code could not have produced them. p2_d: every row and checkpoint says `4710690` (committed 06:39, after the 06:15 launch; it adds the Track 1 flag to `search.py`/`config.py`) while `provenance.json` says `72e7f1a`; p2_b's pass-5 ladder rows mix two SHAs across anchors. **Fix:** resolve the SHA (plus a dirty flag) once at `Phase2Run.create` / the ladder caller, persist it in `state.json`, stamp from the stored value; `--resume` records the resuming SHA separately. Amend the affected rows' provenance in the RUNLOGs rather than rewriting the rows.

### C3 — ladder cadence is a CLI flag that is not persisted and is lost on `--resume` — Medium-low (= D7)
`config.py:268` (`eval_every_n_iters`, never read — `grep` hits only the definition), `phase2.py:174`, `scripts/train_phase2.py:85-94,157`. `--ladder-every` and `--ladder-anchors` reach neither `config.yaml`, `provenance.json` nor `state.json`, so `resume` cannot reconstruct them — against "`--resume` rebuilds everything from the run directory". Every round's saved `config.yaml` says `eval_every_n_iters: 10` while p2_b/c/d ran at 5 (PREREG-p2_b/c/d all say `--ladder-every 5`). **Actual record:** p2_a crashed at iteration 24, was resumed without the flag, and passes 30/40/50 had to be backfilled offline (RUNLOG-p2_a.md:250; `produced=None` on those rows). Related duplicate spelling: `mcts.sims` and `selfplay.sims_per_move`; self-play honours only the latter. **Fix:** make the config field the source of truth, have the CLI override it before `save_config`, read it back on resume.

### C4 — `moves.dat` is not "the move actually played" — Medium
`selfplay.py:502-509` (`_store`), `replay.py:11,227,306`. `TracePosition` does not carry the acting move, so `_store` substitutes `argmax(improved_policy)`; the runner acts on `argmax over survivors of g + logits + σ(q̂) (+ EXACT_WIN_BONUS)` (`batch_runner.py:447`). The two differ on **95.0%** of positions with the uniform stub and 98.3% with a random net. `ReplayBuffer.sample` builds `promo_target`/`promo_mask` from the promotion entries *at that cell*, so the promo CE is computed at a cell the game may not have played, and rows where the played move was a promotion but the argmax was not get no promo signal. Blast radius: `replay.py:306` is the only consumer, so this is the promo loss plus the field's meaning in all four campaigns. **Fix:** record `result.best_move()` on `TracePosition` and store that; or rename the field.

### C5 — model init unseeded; `--seed` not persisted into config or checkpoint — Low-med
`phase2.py:203, 201/224`. `ChessNet(cfg.model)` draws from torch's global RNG with no `torch.manual_seed` anywhere in the library or `train_phase2.py`; two `Phase2Run.create()` with the same config give different initial weights. `seed = cfg.seed if seed is None else seed` is stored only in `state.json`; `save_config` and the checkpoint's embedded config keep the unoverridden value (`create(seed=7)` → `config.yaml seed: 42`). No recorded round is affected (all used 42 and `--init`). **Fix:** `cfg = replace(cfg, seed=seed)` before saving; seed the init from `iteration_seed(seed, 0, STREAM_INIT)`.

### C6 — see B8.

### C7 — see A4.

### C8 — pool-mover `root_q` blended into the learner's value target — Low-med (design gap)
`selfplay.py:515`, `batch_runner.py:462`, `phase2.py:462`, `replay.py:357`. In league games `root_q` is `completed_q.max()` of whichever network searched; with `train_on_pool_moves=True` (every round) the ~10% pool rows pull the learner's value head toward a 1–20-iteration-old snapshot's estimate at `w=0.5`. The staleness is documented for the policy target (`config.py:220-227`), not for the value target; PREREG-p2_c does not mention it; `mover_is_latest` is not consulted. Either mask `root_q_mask &= mover_is_latest` or record the choice.

### C9 — see B1 (and: a ladder pass killed mid-way is never resumed) — Low
`phase2.py:352-354`: `iterate()` runs the ladder after the state commit; a kill during the ladder leaves `state.iteration = i` and the next `iterate()` runs `i+1`. On resume, if `ladder_every and state.iteration % ladder_every == 0`, re-run `_run_ladder(state.iteration)` (it already skips completed matchups).

### C10 — test hygiene — Low (test)
`tests/test_train.py:233,603,625`: process-global `torch.manual_seed(0)`. `tests/test_mixed_batch.py:120,136`: `sys.path.insert(0, scripts)` twice — banned by AGENTS.md §3. `tests/test_selfplay.py:210-229`: `max_plies=6` makes every game a `ply_cap` draw, so the end-to-end outcome sign through `play_games → _store` is never exercised with a decisive game. `selfplay.py:410`: `stop_fn` silently returns `None` if the tracker binding fails, disabling resignation for that game without error.

---

## D — evaluate / opponents / logschema / config / records

### D1 — torn JSONL tail + resume loses a row, then poisons the file — Medium
`logschema.py:164-168` (`_append`), `:379-397` (`read_jsonl`). `_append` writes `json.dumps(row) + "\n"` without checking the file ends in a newline; `read_jsonl` drops a torn *final* line. After a kill mid-write the resumed pass's first row lands on the torn fragment; the concatenated line fails to parse and, being last, is silently dropped — the matchup just measured is invisible to `completed_matchups`. A further resume replays and appends again, the corrupt line is no longer final, and `read_jsonl` raises for the whole file; no pass can be appended again. Probe: `resume 1 sees {random}` (greedy written but invisible); `resume 2: completed_matchups RAISES JSONDecodeError`. **Fix:** in `_append`, if the last byte is not `\n`, write one first (or truncate the torn tail); add the kill-then-resume case to the resumability test.

### D2 — see C2.

### D3 — the anchor is pinned by digest in prose only — Medium
`opponents.py:127-133` (`digest: str = ""`, never set or read), `:274-277`; `runs/ANCHORS.sha256`. `build_opponent("phase1")` loads whatever sits at `--anchor-ckpt` (default `runs/phase1_anchor.pt`, gitignored); the row records only `anchor: "phase1"`; v1 and v2 both lived at that path. A restored v1 anchor would silently move the ladder's zero point, and `completed_matchups` would treat the rows as the same matchup. The current file does hash to v2 (`27dcdd05…`). **Fix:** hash the file in `build_opponent`, compare to the pinned digest, raise on mismatch, and record the digest in the row.

### D4 — re-running a completed pass appends a second eval row — Medium-low
`evaluate.py:602, 641-647, 710-771`. `completed_matchups` indexes only `row_type == "match"`; after all matchups are skipped the suites/watchlist/entropy run unconditionally and append another eval row. Probe: run 0 appended `[match, eval]`, run 1 appended `[eval]`. Committed logs have no duplicates (p2_b's backfill used `--no-suites`). Add `(checkpoint, EVAL_ROW)` to the completed set.

### D5 — the resume key ignores `games`, `sims`, `max_plies`, `role` — Medium-low
`evaluate.py:600-602, 641-647`. A 20-game smoke row or a diagnostic row blocks the later 200-game protocol row for the same pair, and `scripts/eval_ladder.py` defaults `--out` to the campaign's own `ladder.jsonl`. Include the protocol parameters in the key, or refuse to skip when they differ.

### D6 — offline CLI defaults diverge from the inline pass — Low-med
`scripts/eval_ladder.py:79-81, 115` vs `phase2.py:379-389`: `--config` defaults to dataclass defaults (not the run's `config.yaml`), `--seed` to 0 (inline: the run seed, 42). The seed reaches the random/greedy move streams and the bootstrap RNG; rows record no config fingerprint, so an offline/inline mismatch is undetectable afterwards. Today every run's `mcts`/`eval` section equals the defaults and RUNLOGs say `--seed 42` was passed by hand. Default `--config` to `ckpt.parent/config.yaml`, `--seed` to `cfg.seed`; stamp `config_fingerprint` into rows.

### D7 — see C3 (plus `stockfish_path`, `openings_path` dead) — Medium-low
`config.py:268-270`. The YAML comment on `stockfish_path` describes behaviour that lives in `--stockfish` + `shutil.which`; `openings_path` is ignored for the hardcoded `OPENINGS` (`evaluate.py:46-47`). Probe: all three set to nonsense values, loaded with no error and no effect. Wire them or delete them.

### D8 — `configs/puct_ablation.yaml` is not "default but PUCT" — Low-med
Carries `model.dropout: 0.1` and `train.legality_loss_weight: 1.0` (default 0.0 / 0.0) from before `default.yaml` was retargeted; the M4 Gumbel-vs-PUCT comparison would confound root selection with dropout and the legality head. Not yet run. Regenerate from default with only `root_mode` changed; add a brief test.

### D9 — no type validation in `load_config` — Low
`config.py:317-342`. `root_noise: 'false'` is a truthy string: `resolved()` is the identity, noise stays on, the fingerprint changes. `'1e-4'` (YAML 1.1 string) crashes later and loudly; `mcts: 5` replaces the section. Coerce/validate leaves against the dataclass type hints.

### D10 — `load_config(overrides=...)` drops unknown leaf keys — Low (latent)
`config.py:390-398`: `{"model.n_layer": 4}` → no error, `n_layers` still 8. `_assert_known_keys` exists for YAML but is not applied to overrides. No caller uses overrides today.

### D11 — `config_fingerprint` is not regenerable once a field is added — Low
`config.py:433-436`. It hashes the dump of all *current* fields, so adding `proof_directed_expansion`/`proof_q_threshold` changed the fingerprint of every unchanged YAML: `config_fingerprint(load_config("runs/p2_a/config.yaml"))` = `67b1349a…` vs recorded `f8588828…`. The recorded values equal `sha256(config.yaml bytes)` (verified for all four rounds) because `save_config` and the fingerprint share one dump — but nothing documents or tests that. Document and test `config_fingerprint(cfg) == sha256(save_config(cfg))`, or hash only keys present in the source YAML.

### D12 — `close()` on a dead engine masks the original exception; nodes-only `play()` has no timeout — Low
`opponents.py:221, 227-230`; `evaluate.py:660-671`. After a SIGKILL'd engine, `choose()` raises `EngineTerminatedError`, the `finally` calls `close()`, and `quit()` raises again, replacing the traceback. python-chess 1.11.2 `SimpleEngine._timeout_for` returns `None` when `limit.time is None`, so a hung engine blocks forever (a second route to the 7-hour hang RUNLOG-p2_d describes). Guard `quit()`; add a time ceiling.

### D13 — `test_generator_scripts_are_runnable` uses a bare `python` — Low (test; the baseline failure)
`tests/test_evaluate.py:583`. Resolves from PATH, not the interpreter running pytest; on this machine only `python3` and the venv's `python` exist, so `make test` passes only in an activated venv. `tests/test_phase2.py:566` already uses `sys.executable`. Both scripts' `--help` run fine under the venv.

### D14 — `±Infinity` written to JSON for one-opening matches — Low
`evaluate.py:93-94`: `paired_bootstrap_elo` returns `±inf` for `n_pairs < 2`; `json.dumps` emits Python-only literals that strict parsers reject. The fields are already nullable — return `None`.

### D15 — a terminal opening crashes `RandomMover` — Low
`batch_runner.py:525-531` consults `move_fn` before `is_game_over()`; `opponents.py:59-61` calls `rng.integers(0)` → `ValueError`. Unreachable with the frozen book, reachable with custom `openings=`.

Minor (no action required): `_check` reports "missing pair_scores" before "version mismatch" on a v2 row; `_DeterministicGreedy.name = "det-greedy"` is overridden by the inherited dataclass `__init__`.

---

## E — scripts / benchmarks / build

### E1 — the Phase 1 anchor mixture is not the configured fraction — **High**
`scripts/train_phase1.py:65, 79-81` (`_build_loader`). Random-play samples get weight `1−a` (a = total anchor mix) **and** each anchor sample gets `(mix/(1−a))·(n_rand/n_a)`. Either normalisation alone yields exactly `mix`; both together yield `mix / ((1−a)² + a)`. **Verified by arithmetic:** total mass = `n_rand·[(1−a) + a/(1−a)]`, anchor share = `mix/(1−a)` over that = `mix/((1−a)²+a)`. The docstring (`:47`) claims "contributes exactly its configured fraction"; HOW_TO_TRAIN.md:108 and spec §5 say "fixed 5%". No test pins the realised fraction. **Actual record:** `phase1_mate1_mix: 0.05` → 5.25%; `configs/phase1_c.yaml` (0.10) → 10.99%; `configs/phase1_d.yaml` (0.08 + 0.07) → **9.17% + 8.02% = 17.2%** anchors instead of 15% — so the Phase 1 anchor v2 and every round's checkpoint config misstate the mixture they trained on. Deterministic and reproducible; just not what the record says. **Fix:** `np.full(n_rand, 1.0)` for random weights (keeping the anchor formula), or keep `1−a` and drop the `/(1−a)`; add a test asserting the realised fraction to 1e-6; erratum in TRAINING_STRATEGY/PROGRESS for the rounds.

### E2 — Phase 1 `--ckpt` resume restores neither the run's config nor its data order — Medium
`scripts/train_phase1.py:152-156, 199-200, 224-259`. On resume `cfg` comes from `--config`/dataclass defaults and CLI overrides, not from `ckpt["config"]` (used only for a model-shape check), so `phase1_epochs`, cosine `T_max`, `lr_start/end` and the mixes are whatever the resume command line says; `train_phase2.py:126` re-reads the run's `config.yaml` precisely so "a resumed run cannot silently change what it is". Separately, the `WeightedRandomSampler` generator and `torch.manual_seed` are reseeded from `--seed`, so the first resumed epoch draws exactly epoch 1's multiset (`resumed epoch3 == uninterrupted epoch1: True`; 37.8% of positions never drawn in epoch 1 stay undrawn). HOW_TO_TRAIN.md:187-199 documents a resume without `--config` and says the LR schedule "is restored automatically". **Failure:** resuming `runs/phase1_d/epoch_01.pt` (checkpoint config: lr 1e-4 constant, 2 epochs, mixes 0.08/0.07) with the documented command → cosine 3e-4→3e-5 over 3 epochs, mixes 0.05/0.0, fast-forwarded 9,463 steps. **Fix:** on `--ckpt`, build `cfg` from the checkpoint (error if `--config/--epochs/--batch` disagree); seed the sampler from `(seed, epoch)` or persist its state.

### E3 — run directories lack the AGENTS §4 artifacts; Phase 2's `--seed` is recorded only in a gitignored file — Medium (= C5)
`scripts/train_phase1.py:189-194`; `phase2.py:201-241`. AGENTS §4 / spec §9.2: every run dir gets the resolved `config.yaml`, the git SHA and the `check_env` output. Phase 1 writes only `metrics.jsonl` + `.pt` (`ls runs/phase1_d` → `epoch_01.pt epoch_02.pt final.pt metrics.jsonl`); a Phase 1 run can be tied to a SHA or config only by unpickling a 78 MB gitignored checkpoint. Phase 2 writes `config.yaml` + `provenance.json` but no `check_env` output (nothing in `src/` references it), and `save_config` writes the unoverridden `cfg.seed` while the run uses `state.seed` (scratch dry run: `state.json` 7, `config.yaml` 42; `provenance.json` has no seed). **Fix:** `train_phase1` calls `save_config` and writes provenance like Phase 2; `Phase2Run.create` writes the effective seed into both files; append `check_env()` to provenance.

### E4 — `make bench` is a silent no-op and `make lint` is red at HEAD — Medium
`Makefile:11-13, 21-23`; `benchmarks/bench_inference.py:103`. `bench` calls bare `python` (absent outside an activated venv) and `bench_inference.py --device cpu --batches 64` — there is no `--batches` flag and `--ckpt` is required, so the line always exits 2, hidden by `2>/dev/null || true`. `lint` fails on `search.py` formatting (P13's line), and `benchmarks/` + `bench_step.py` are outside its scope with 3 ruff errors (`E401`, `I001`, `W292`). So the AGENTS §7 pre-commit `make lint && make test` is red on a clean checkout. **Fix:** `PY := .venv/bin/python`; `bench_inference.py --ckpt runs/phase1_anchor.pt --device cpu --no-record`; drop `|| true`; widen the lint globs; format `search.py`.

### E5 — no ROCm torch index pin in `pyproject.toml` — Medium
`pyproject.toml:10-24`. AGENTS §2 rule 2 mandates `[tool.uv.sources] torch = {index = "pytorch-rocm"}` + `[[tool.uv.index]]`; neither exists (verified: `grep -n "tool.uv\|index"` → nothing), so `uv pip install -e ".[dev]"` (HOW_TO_TRAIN.md:11, `make install`) resolves `torch>=2.2.0` from PyPI — the "#1 recurring failure" AGENTS §2 describes, and exactly what rebuilding `.venv` per §2.3 would trigger. `gradio>=4.0` is a dev dependency with zero imports. **Fix:** add the two sections pinning `rocm7.2`; drop `gradio`.

### E6 — `benchmarks/results.jsonl` has no stable schema; `date` never stamped; sweep SHA is cwd-relative — Low
`bench_selfplay.py:92-93` (key `bench`, 7-char SHA, `"date": None  # stamped by the caller` — no caller exists); `bench_inference.py:128` and `sweep_eval_profile.py:53-60, 203-215` (key `benchmark`, 40-char SHA from `git rev-parse HEAD` without `-C repo` — the failure `train._git_sha`'s docstring warns about; `mate_in` not recorded, inferred from `"mate_in_2" in filename`, else 1). Census: 10 distinct key-sets in 31 rows; SHA lengths {40: 18, 7: 13}; `date` null in 26/31; a sweep launched from another cwd printed `sha=unknown`. **Fix:** one `benchmark_row()` helper used by all three writers; explicit `--mate-in`.

### E7 — `eval_phase1` gates on 200 freshly generated anchors when `--mate1` is omitted — Low
`scripts/eval_phase1.py:51, 62, 78-87`; `train.py:554-576`; `train_phase1.py:177-181`. `compute_metrics(mate1_dir=None)` generates `min(n_mate1, 200)` positions with `default_rng(0)`; the script prints PASS/FAIL for `wdl_anchor_mass` and `mate1_top8_masked` on them while announcing `n_mate1=1000` (dry run: 0.8333 with `--mate1`, 0.2450 without, same checkpoint). In `train_phase1`, `eval_mate1 = args.eval_mate1 or args.mate1` falls back to the *training* anchors with no warning (only `--eval-data` warns), so HOW_TO_TRAIN's "Custom config" command logs a gated metric over training data. **Fix:** require `--mate1` (or print "not measured"); warn on every training-dir fallback.

### E8 — HOW_TO_TRAIN contradicts the code — Low
`HOW_TO_TRAIN.md:107, 108, 217-237`; `train_phase1.py:11`. Step 3 evaluates on `runs/data/phase1` and `runs/data/mate1` — the training dirs — against the guide's own rule (line 39); the acceptance table lists the gates retired on 2026-08-06 (`top1_legal_rate`, `mate1_solve_rate`) instead of the four the script prints; line 107 says Phase 1 uses `configs/default.yaml`, but `load_config(None)` returns dataclass defaults (`legality_loss_weight 1.0, dropout 0.1`) and `configs/default.yaml` is now the Phase 2 config (`0.0 / 0.0`) — `train_phase1.py`'s own usage line suggests it, which would train Phase 1 with the legality auxiliary off and make `legality_auc ≥ 0.995` unmeetable; the 95/5 claim is 94.75/5.25 (E1).

### E9 — CLI defaults duplicated against config/library; `cfg.seed` dead for Phase 1 — Low
`train_phase1.py:138, 183-184` (`--seed` default 42; `cfg.seed` never read, so a Phase 1 YAML's `seed:` does nothing — `train_phase2` correctly uses `None → cfg.seed`); the anchor list `"random,greedy,phase1"` spelled in `train_phase2.py:97`, `eval_ladder.py:57`, `phase2.py:175`, `run_phase2`, `run_ladder_pass:555`; the anchor path in `eval_ladder.py:64`, `phase2.py:176`, `evaluate.py:587`.

### E10 — the mate-in-2 tuning suite is gitignored — Low
`.gitignore:5-41`. BLOCKED-2026-08-08 and 18 `results.jsonl` rows cite `runs/data/mate2_tune_500/mate_in_2.epd` as the disjoint tuning suite the sweep firewall rests on; `runs/**` ignores it and only the directory's `meta.json` is negated (`git status --ignored` → `!! runs/data/mate2_tune_500/mate_in_2.epd`); not in `MUST_TRACK`. Distinct from P7.

### E11 — banned `sys.path` mutation; printing `main()`s inside `src/` — Low
`tests/test_mixed_batch.py:124, 143` import `train_phase1._build_loader` via `sys.path.insert` (AGENTS §3) — the loader builder is library code living in a script. `selfplay.py:536-582`, `mcts/search.py:1009-1120` (a second suite-runner CLI beside `scripts/`), `encoding.py:344` carry argparse `main()`s with `print()` (AGENTS §4).

### E12 — inert generator flags; anchors silently unused — Low
`generate_phase1.py:111-112, 139-191`: `--max-plies/--min-plies` reach only the `--games` branch; `generate_mate_in_1` hard-codes 150, `generate_mate_in_2` 160 (help text does not say so). `train_phase1.py:67-72`: `--mate2 DIR` with a config whose `phase1_mate2_mix` is 0 (defaults, `mini.yaml`, `default.yaml`) is never sampled while `--eval-mate2` still gates on it — dry run printed `mate2_top16=0.6667` with zero mate-2 samples drawn.

### E13 — diagnostics cannot take the campaign config; "mate in 2" defined three times — Low (possible)
`triage_mate2.py:141`, `diagnose_mate2.py:53-58` hard-code `MCTSConfig(root_noise=False)`, i.e. the dataclass-default eval profile, not the campaign's `replace(cfg.mcts, root_noise=False)`; equal today (the GPU re-measure reproduced the ladder exactly), divergent on the first `eval_*`/`promo_prior_from_head`/`proof_directed_expansion` change. `generate_suites._is_mate_in_2` is the private copy `data.py:221` says should have been pointed at `mate_in_2_key_moves` "when those branches merge" — never done; both accept all 200 committed positions.

---

## Test-suite defects (collected)

Tests that cannot detect the bug they describe, leak process-global state, or depend on this machine. None changes a recorded number; each is a guard that is not guarding.

| # | Test | Problem |
|---|------|---------|
| T1 | `tests/test_data.py:71-104, 257-291` (B2) | value-target sign: alternation-only assertion + literal round-trips; a global sign flip passes |
| T2 | `tests/test_replay.py:202-247` (B8/C6) | root_q/outcome perspective: round-trips the test's own values; PREREG-p2_c cites it as the guard |
| T3 | `tests/test_mcts.py:326-342` (A5) | `chosen in (a, b)` on a two-edge node — any return passes |
| T4 | `tests/test_mcts.py:372-394` (A5) | repetition-on-history: asserts only the budget identity |
| T5 | `tests/test_mcts.py:1289` (P4) | determinism test with one seed |
| T6 | `tests/test_mcts.py:1251` (P8) | `set_num_threads(4)` leaks; depends on a gitignored checkpoint |
| T7 | `tests/test_rehearsal.py:345` (P11) | config-brief diff skips top-level scalars (`seed`) |
| T8 | `tests/test_evaluate.py:583` (D13) | bare `python` on PATH |
| T9 | `tests/test_train.py:233,603,625` (B9/C10) | `torch.manual_seed(0)` ×3, no restore |
| T10 | `tests/test_mixed_batch.py:120,136` (C10) | `sys.path.insert` — banned by AGENTS.md §3 |
| T11 | `tests/test_selfplay.py:210-229` (C10) | `max_plies=6` → every game a draw; decisive path through `_store` untested |
| T12 | `tests/test_contamination.py:221-230, 294-308` | `test_contamination_check_can_actually_fail` tests `set &` and `np.array_equal`, not the guard; `test_rehearsal_config_dirs_exist_when_enabled` ends in `assert True` |
| T13 | `tests/test_gitignore_musttrack.py` (P7) | four new tracked record kinds not in `MUST_TRACK` |
| T14 | `tests/test_evaluate.py` (P3) | `test_opponent_families_declare_whether_they_randomize` instantiates only Stockfish |
| T15 | — (E1) | no test pins the realised Phase 1 anchor fraction |
| T16 | — (E10) | the tuning suite `.epd` is not in `MUST_TRACK` |

---

## Verified sound (the ledger)

What was checked and found correct. This matters as much as the findings: it is what the fix pass must not break.

**Search core (A).** The August review's fixes hold: S1 (one implementation of considered-set formation/halving/force-include, consumed by both drivers), S2 (`scripts/parity_probe.py` 0 disagreements; an extended sweep over m ∈ {2,3,4,5,7,9,12,16,31} × sims ∈ {2,3,5,6,9,13,16,31,48,49} × 8 positions × both profiles = 1,440 combos, 0 divergences on root N, terminal, proven, proven_depth and acting move), S3 (real `base_reps` in trace tokens), S4 (promo-head priors behind the flag), and the proven-depth ordering fix. Sign convention consistent at every consumer (child-perspective exact values negated at `_select_edge`, `_root_completed_q`, `_gumbel_scores`, `force_include_proven_wins`, `_solve_edge`; backup flips once per ply; `root_q` root-perspective). Flat edge arrays, growth, zero-fill correct. Repetition semantics identical in-search and at game level. Terminal short-circuits never touch vloss or the buffer and still consume budget; `sims_run == num_sims`. Sequential Halving budgets partition exactly; zero-budget phases skipped identically in both drivers. Eval profile consumes no RNG and gives identical trees for seeds 1 vs 999. Virtual loss symmetric; with one generator per game, per-game search equals the synchronous one for any `batch_leaves` (A1 is the only B-dependence). Improved-policy target over all legal actions with exact values and `g` excluded. Solver algebra correct for win, loss, draw; `proven_depth` odd and ≥ 3 on real trees.

**Codecs and data (B).** 40,000 random reachable positions: `decode(encode(b))` reproduces placement, turn, castling rights, the *legal* EP square, halfmove bucket; legal-move sets equal; re-encode idempotent. All 919,060 legal moves round-trip through `encode_move`/`decode_move` with unique triples (6,908 promotions incl. 5,181 underpromotions, 73 EP, 372 castling). `legal_mask_ft` ≡ `legal_moves_packed` ≡ `train.unpack_legal_mask` (LSB-first). No mirroring anywhere (absolute coordinates + TURN token). REP token agrees with an independent transposition count on 3,000 shuffle-heavy positions and with self-play's `history + path_reps − 1`. Phase 1 outcome sign: 54 decisive games, 6,792 positions, 0 errors. Index conventions consistent across `replay.sample`, `train._promo_loss`, `rehearsal.py`, `search.py` (promo class c → head c−1; WDL `[loss, draw, win]` ↔ `outcome+1`; `value_scalar = P(win) − P(loss)`). Replay ring: uniform over live slots in partial/full/wrapped rings; `add` writes every field before advancing; `flush` msyncs before writing meta; `rewind_to` refuses to move forward; `state.json` committed after `flush` so the rewind target is never ahead of the durable head. `write_dataset` writes `meta.json` last (torn dataset fails loudly). `promo_logits` matches the dense form to 2.5e-7. bf16 autocast only on cuda, with softmax/CE/MSE in fp32. Checkpoints embed model, optimizer, config, SHA, step, phase. No global RNG in `src/`.

**Training loop (C).** Resume equivalence on CPU (mini config): uninterrupted vs resume-between-iterations vs resume-after-torn-generation (ring head rewound 1280 → 640) give **bit-identical** `ckpt_iter0002.pt`, identical iteration-2 rows (timings excluded) and identical ring contents. Outcome perspective end-to-end: White and Black mate-in-1 starts give `root_q=+1.000, outcome=+1` for the mover; a resigned game assigns −1 to the resigner. Resignation streaks per side on the mover's own `root_q`, strict `<`, holdout records `would_resign_*` and plays on; a resign cannot pre-empt an available immediate draw/win. Draw adjudication (threefold, fifty-move, stalemate/material, ply cap) uses the search's constants and arithmetic. Trace legal masks equal the real board's on 1,396/1,396 positions. Policy target = softmax(logits + σ(completed_q)) over all legal actions with exact values; promotions share a cell; >96-entry rows keep the largest and renormalise. Masked rows enter only the MSE; v2 rings and rehearsal rows carry `root_q_mask=False`. Config fields honoured: `masked_policy`, `legality_loss_weight`, `value_q_mse_weight`, `train_on_pool_moves`, `nan_abort_*`, `league_*` (χ² on the 80/20 marginal), rehearsal fractions (51/1024 rows; missing dir raises). AdamW constant LR (nothing to resume), clip then non-finite skip, `state.step` persisted. Write ordering ring → snapshot → row → atomic `state.json`. No thread/subprocess leaks.

**Scripts, suites and build (E).** All 14 `scripts/*.py` and both `benchmarks/*.py` `--help` cleanly; every library signature a script calls matches its call site. **The committed suites are what they claim:** an independent exhaustive solver (no repo code) accepts all 100 mate-in-1, all 200 mate-in-2 (forced in exactly 2, none in 1) and all 200 mate-in-3 (exactly 3, none in ≤ 2); ids and EPDs unique; zero cross-suite overlap; Stockfish `go mate N` confirms every position fixed-depth search disputed. The repo's own solvers (`_forces_mate_in_2/3`, `can_force_mate`, `mate_in_2_key_moves`, `generate_suites._is_mate_in_2`) agree with it on all 500. `openings_4ply.epd`: 100 unique, white to move, none in check or terminal, ≥ 31 pieces; regenerating with `--count 100 --seed 42` is byte-identical to the committed book. **`diagnose_mate2.py` on `runs/p2_c/ckpt_iter0020.pt` at 128 sims (GPU) reproduces the ladder eval row and RUNLOG-p2_c §4 to the digit** (coverage 199/200 = 0.995, conversion 186/199 = 0.935, gate 186/200 = 0.930); `triage_mate2.py` reports 0 mismatches vs `Search.run`. Dry runs in scratch of `generate_phase1` (all three modes; `--exclude` works), `generate_suites`, `generate_openings`, `generate_mate3`, `train_phase1` fresh + `--ckpt` + `--max-steps`, `eval_phase1`, `train_phase2 --out` then `--resume` (counter continues; `--out` on an existing run refused; `--init` with `--resume` refused; `config.yaml` re-read), `eval_ladder` (row passes `validate_ladder` with `role`/`produced`/`nondeterministic` stamped), both benchmarks with `--no-record`, `sweep_eval_profile`, `check_env --json`, `python -m tinyalphazero.encoding --fuzz 2000`. HOW_TO_TRAIN's "Reading the log"/"Reading a ladder row" snippets run unchanged. `.gitignore`: no tracked file is ignored; no `.pt/.dat/.npy` tracked; all cited p2_* records tracked. `benchmarks/results.jsonl` still 31 rows and no file under `runs/` was modified by any probe (checked by mtime).

**Evaluation and records (D).** Colour attribution reads the loser off `board.turn`; Fool's and Scholar's mate through `play_opponents` score correctly; ply cap and non-mate game-over score 0.5; checkmate precedes claim-draw and ply cap; both opponent kinds share one adjudication. Pairing `(white, black)` per opening, in order; every book opening white-to-move, non-terminal, distinct; the deterministic self-match null test is a live pairing proof. Elo `400·log10(s/(1−s))` with the `1/(2N)` clamp flagged; the bootstrap resamples opening *pairs* (the registered unit), percentile 2.5/97.5, rng `[seed, 99]`; the quantised Elo lattice in the record follows from the formula. **The p2_d verdict recomputed from the committed rep rows equals `p2_verdict_reps.json` and RUNLOG-p2_d §1 exactly** (paired difference 0.1033, CI [0.0600, 0.1467], SE 0.0220, rep σ 0.0328/0.0236); rep rows carry `nondeterministic: true`, `role: diagnostic`, and are genuinely different games. Every number in the RUNLOG-p2_a/b/c/d tables matches the committed rows; no `(checkpoint, anchor)` duplicates in any log. Suite scoring is mate verification, not `bm` matching; suites/watchlist run at `suite_sims` under `root_noise=False`. `resolved()` is applied by both search constructors and reached from every entry point in `evaluate.py` and the diagnose/triage scripts. All `configs/*.yaml` and `runs/p2_*/config.yaml` load, resolve and round-trip exactly; unknown YAML keys raise with a near-miss hint; `runs/p2_d/config.yaml` is byte-identical to p2_c's and to `configs/p2_d.yaml`; all four recorded fingerprints equal `sha256(config.yaml)`. `runs/p2_d/iterations.jsonl` 20/20 and the p2_d eval rows 4/4 validate; every rejected committed row falls under P1/P10 or pre-v2 rows lacking `role`/`produced`. Stockfish options configured; engine opened lazily; `close()` reaps the process; `nondeterministic` is `skill < 20`. `runs/phase1_anchor.pt` hashes to ANCHORS v2.

---

## Not covered

- GPU paths: ROCm/bf16 training and `torch_evaluate_batch` numerics were not executed; the sync/batched contract was verified with the stub on CPU. GPU nondeterminism as a resume-divergence source is untested (relevant to whether p2_a's crash-resume at iteration 24 regenerated identical games).
- No Stockfish match was played (option/kill/close probes only); no ladder row or suite was re-measured against a real checkpoint.
- `proof_directed_expansion` beyond unit semantics; `_forces_mate_in_2/3` and `can_force_mate` were read for consistency but not fuzzed against an independent oracle.
- `scripts/generate_phase1.py` multiprocessing shard seeding (ran `--workers 1` only); `track1_ckpt_iter*.json` and `competitor_q_trajectory.json` (produced by code not in the repo — P5).
- `bench_step.py` (hard-codes `cuda` and `configs/phase1_c.yaml`) read and linted, not run; Stockfish, `phase1` and `greedy` ladder rungs not run (only a 2-game `random` rung); `--seed-pool`, inline `--ladder-every`, `--compile` paths not exercised.
- Phase 1 gate numbers (the phase1_d acceptance table) were not re-measured; only the p2_c iteration-20 mate-in-2 decomposition was.
- The golden and SIGKILL tests in `tests/test_phase2.py` were not run in isolation (they ran in the baseline suite).

---

## Proposed fix order

1. **Record integrity first, code second.** Errata in RUNLOG-p2_c (C1: what the blend actually weighted), TRAINING_STRATEGY/PROGRESS (E1: the realised Phase 1 mixtures per round), RUNLOG-p2_b/p2_d (C2/D2: the rows' true launch SHAs, from `provenance.json`), and PREREG-p2_c (C8). Backfill the 17 rows for P1 in a commit that says so. These do not change any number; they make the numbers mean what they say.
2. **Schema and provenance plumbing:** P10 (per-family versions), C2/D2 (SHA captured once at launch), D11 (fingerprint contract documented and tested), D3 (anchor digest checked and recorded), B4 (one `_git_sha`), E3 (run-dir artifacts, effective seed persisted), E6 (one benchmark row writer).
3. **Commit the missing harness** (P5) and expose the lever on the CLI with eval-profile scoping (P2, P6 decided one way or the other).
4. **Training-affecting code:** C1 (per-row CE weighting — a *new* lever version; name it), E1 (correct mixture arithmetic — changes what a given `mix` means; name it), B3 (attention init — a new init; name it), C4 (store the played move), C5 (seeded init). None of these should be changed silently mid-campaign: each is a config-visible change for the next round's PREREG.
5. **Resumability:** B1/C9 (atomic writes), D1 (torn-tail guard), A1 (double root; add the `batch_leaves < games` test), E2 (Phase 1 resume from the checkpoint's config and a per-epoch sampler seed), D4/D5/D6 (ladder resume key and CLI defaults), C3/D7 (persist cadence; wire or delete the dead fields).
6. **Build and tests:** E5 (ROCm index pin — before anyone rebuilds `.venv`), E4 (`make bench`/`make lint`), E11 (move `_build_loader` into the package; `main()`s out of `src/`); T1–T16 in one pass; then `make lint` and the full suite green from a clean shell without the venv activated.
7. **Low/latent:** A2, A3, A4/C7, B5, B6, B7, D8, D9, D10, D12, D14, D15, E7, E8, E9, E10, E12, E13, P9, P12, A6.

Not fixed here: nothing. This document is the review; no code was changed.
