# Pre-registration — p2_x: the first public benchmark, a CCRL-anchored gauntlet

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-12** (see Sign-off), at Michael's instruction ("Ok given, go"; the network on
> cuda). Nothing under this file has run. The smoke games named in Sign-off tested the tooling at 10+0.1 and
> 5+0.05; they are not part of any measurement here.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

Every number since p2_a is a score against this repo's own fixed opponents (Stockfish at a skill level and 10,000
nodes, the phase 1 anchor), under this repo's own pairing. That ranks checkpoints, and it says nothing about where
the 6.5M network stands among chess programs. RUNLOG-p2_w §6 reading 5 ends the teacher rounds at a plateau, and
Michael chose the public benchmark as p2_x's lever (2026-11-12).

**What is p2_w's endpoint's rating on the CCRL scale, playing at its ladder configuration (48 simulations a move,
evaluation profile), against opponents whose CCRL ratings are published?**

This round trains nothing. It plays one checkpoint against five opponents and fits one rating.

## Contract

| | |
|---|---|
| Checkpoint | **`runs/p2_w_quarter_down/ckpt_iter0010.pt`** (sha256 `43494548…3857`), the standing protocol's latest endpoint and the best six-rep skill-5 mean so far (0.379; RUNLOG-p2_w). Config: the `config.yaml` beside it (fingerprint `036cd665…01da`) |
| The network as an engine | `scripts/uci_engine.py` (UCI over stdin/stdout), `--device cuda`: the evaluation profile (`root_noise=False` on the run's `mcts`), **48 simulations a move** (`eval.eval_match_sims`), the batched evaluator at batch 1 (bf16 autocast on cuda, as the ladder), the ladder's leaf masking (`masked_policy_probs`), and repetition history from the `position` command as `BatchRunner` counts it. Clock fields are ignored: the budget is simulations, not time, as on the ladder. No opening book, no tablebases, no pondering |
| Parity with the ladder (measured at draft, P1) | `scripts/uci_parity.py`: 20 ladder games (10 book openings × 2 colours vs `stockfish:5` at 10,000 nodes, `BatchRunner`, all games concurrent), every network decision replayed through the engine as a subprocess. **cpu (fp32): 1,108 / 1,108 moves identical. cuda (bf16): 1,399 / 1,475 (94.8%)**, the 76 differing moves spread through the games (the first at ply 10); 0.11 s a move over UCI on cuda, 0.52 on cpu `tests/test_uci.py` replays `BatchRunner` games through the session on cpu and requires every move |
| Opponents | Five, 200 games each (below). Stockfish 16 (`/usr/games/stockfish`, Ubuntu `16-1build1`, sha256 `00628bd9…5aca`) at `UCI_LimitStrength=true`, `Threads 1`, `Hash 16`, at **UCI_Elo 1400, 1600, 1800, 2000**; and **HoiChess 0.22.0** (Ubuntu `hoichess_0.22.0-3build1`, `.deb` sha256 `8112dec4…7f0b`, extracted to `runs/bench/engines/hoichess`, binary sha256 `cc9cbe8d…3fd8`), xboard, its own book not found at that path (plays without one) |
| Their ratings (inputs, not estimates) | Stockfish 16's `Skill` (src/search.cpp, tag `sf_16`) converts `UCI_Elo` to a fractional skill level "using anchoring to CCRL Elo (goldfish 1.13 = 2000) and a fit through Ordo derived Elo for match (TC 60+0.6)"; its range is 1320–3190. **HoiChess 0.22.0 64-bit: 1806 ± 20 (936 games) on CCRL Blitz**, the list of 2026-11-07 (2'+1" on an i7-4770K equivalent). Neither error is in the interval K1 reports (see "What the statistic does not cover") |
| Openings | `tests/suites/openings_4ply.epd` (100 positions, sha256 `ce5e9134…eaba`), the ladder's book: each opening twice, the network taking each colour once — 200 games an opponent, **1,000 games** in all |
| Time control | **60+0.6** (Fischer) for both sides: Stockfish's calibration control. This box's core runs Stockfish 16's `bench 16 1 13` at 1.70 Mnps; it is faster than an i7-4770K, so HoiChess at 60+0.6 here sits nearer CCRL Blitz's 2'+1" there than the clock alone says. The network uses ~0.1–0.5 s a move and cannot be short of time at this control |
| Game rules | `tinyalphazero.gauntlet.play_game`: mate, stalemate, insufficient material, and threefold repetition or the fifty-move count once it has occurred end a game; a flag fall loses unless the other side cannot mate; an engine 30 s past its own flag is closed and forfeits (`hung`); an illegal move or a crash loses (`error`). Hard stop at 600 plies (a draw, recorded). **No adjudication on engine scores**, no resignation |
| Runner | `scripts/gauntlet.py play --ckpt runs/p2_w_quarter_down/ckpt_iter0010.pt --device cuda --opponents sf1400,sf1600,sf1800,sf2000,hoichess --openings 100 --tc 60+0.6 --concurrency 3 --out runs/bench/p2_x`, then `scripts/gauntlet.py rate --out runs/bench/p2_x` (10,000 resamples, seed 0). Resumable: finished games in `games.jsonl` are skipped on restart. Every game is also written to `games.pgn` |
| Where | this box, `runs/p2_x_bench.sh` as the transient user unit `taz-bench-p2_x` (restarts on failure; every step resumes or is skipped once done), under `gpu hold --by tinyalphazero` with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTHONUNBUFFERED=1`, the environment of every ladder measurement since p2_k. **Concurrency 3** games: three single-threaded opponents and three network processes (one core-equivalent between them; torch 1 thread). Not more: all-core load has tripped this box's CPU thermal limit twice (2026-10-27, 2026-11-05) |
| Duration and cost | about 65 s a game (the opponent spends most of its 60 s; the network ~10 s), so **about 6 h** for 1,000 games at concurrency 3. No pod, no spend |

## Registered expectations

### P1 — the engine is the ladder's network
Measured at draft (Contract). **The cpu figure must be 1,108 / 1,108** — the gate the wrapper was built to; it
passed. The draft's cuda figure was measured without `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, which the ladder's
measurements set and which selects other attention kernels; **P1 cuda is re-measured under the launch environment
as the run's first step** (`runs/bench/uci_parity_cuda_launch.json`, same 10 openings) and that figure is the one
reported beside the rating. The cuda figure is reported, not judged: the ladder evaluates bf16 batches of many games and the engine
batches of one, so near-ties can fall either way (94.8% at draft). Unmeasured but likely: the ladder's own cuda
moves move with batch composition in the same way, so the 5% is the precision of "the ladder's network" on cuda,
not a second engine. It is reported beside the rating. The engine runs on cuda because cpu costs 0.52 s a move
(close to the 0.6 s increment) and three more busy cores; a cpu run is the alternative if Michael prefers exact
parity to load.

### G — the games are games
- **G1:** the network loses no game by `time_forfeit`, `hung` or `error`. Any such game stays in the data as a loss
  (the conservative reading) and is listed.
- **G2:** no game reaches the 600-ply stop.
- **G3:** every opponent has 100 complete pairs.

### K1 (primary) — the anchored rating
The maximum-likelihood rating over all 1,000 games against the five published ratings (draws as half points;
`tinyalphazero.gauntlet.anchored_rating`), with a 95% percentile interval resampling opening pairs within each
opponent (`bootstrap_rating`, 10,000 resamples, seed 0). **This is the headline: "CCRL ≈ R [lo, hi], at 48
simulations a move".**

### K2 — do the two anchor families agree?
The Stockfish-only rating (the four Stockfish opponents, same fit) against HoiChess's performance rating
(`performance`), each with its own percentile interval from the same resamples (`rating.json` holds both). They
**agree** if each point lies inside the other's interval.

### K3 — the shape (reported)
Per-opponent scores, wins/draws/losses and performance ratings. Expected to fall with UCI_Elo; a non-monotone
pair is reported, not judged.

### Reported, no decision
Per opponent: draw rate, terminations, plies, seconds a move for each side. The ladder's own skill-5 reading of the
checkpoint (0.379) beside K1, for the first mapping from the in-house scale to CCRL's.

## Predictions (written before launch)

- **K1: 1,450–1,750, point about 1,600.** From the smoke games only (not a measurement): 4/4 against UCI_Elo 1320
  at 10+0.1, and 0.5/4 against HoiChess at 10+0.1 and 5+0.05 combined, at controls that weaken HoiChess.
- K1's interval half-width 25–45 Elo (1,000 games, most opponents within 300 of the network).
- K2: agree, more likely than not; if not, HoiChess's performance falls below the Stockfish-only rating (the
  weakest Stockfish settings are where its fit has least data).
- G1–G3 hold. Draw rate 10–25% against each opponent (the network draws about 30% of its games against
  `stockfish:5` at 10,000 nodes, under a 300-ply cap that this round does not have).
- Against UCI_Elo 1400 the network scores 0.65–0.90; against 2000, 0.05–0.25.

## Decision rules

| K1 | K2 | G1–G3 | Decision |
|---|---|---|---|
| any | agree | hold | **Published as measured**: RUNLOG-p2_x gives "CCRL ≈ R [lo, hi] at 48 sims/move", with the PGN and `rating.json` committed. The ladder's skill-5 scale gets its first CCRL reading. p2_y is Michael's choice: a Lichess BOT (a human-pool rating; needs his account), a submission to CCRL's testers (needs a release), or back to training with an absolute yardstick |
| any | **disagree** | hold | **The scale is not settled at this level.** Both numbers are published, the Stockfish-only rating and HoiChess's performance, as a range, with no single headline. p2_y adds CCRL-listed engines near the network's level (TSCP 1.81, 1611; Micro-Max 4.8, 1854; both CCRL Blitz) before any rating is quoted alone |
| any | any | **fail** | Reported as measured, the failed check first. A G1 failure in more than 1% of games (≥ 11) makes the run **not a strength measurement** until its cause is found and the games are replayed under an addendum |
| clamped (a sweep against every opponent) | | | The bracket missed: reported as a bound, and the opponents re-chosen by addendum before any number is quoted |

## Measurement protocol

```bash
# runs/p2_x_bench.sh, as systemd-run --user --unit taz-bench-p2_x, from main at the registration commit,
# under gpu hold with the ladder's environment. First, P1 on cuda under that environment:
.venv/bin/python scripts/uci_parity.py --ckpt runs/p2_w_quarter_down/ckpt_iter0010.pt --device cuda --openings 10 \
    --out runs/bench/uci_parity_cuda_launch.json
# Then:
.venv/bin/python scripts/gauntlet.py play --ckpt runs/p2_w_quarter_down/ckpt_iter0010.pt --device cuda \
    --opponents sf1400,sf1600,sf1800,sf2000,hoichess --openings 100 --tc 60+0.6 --concurrency 3 \
    --out runs/bench/p2_x
.venv/bin/python scripts/gauntlet.py rate --out runs/bench/p2_x          # K1, K2, K3 -> rating.json
# Committed at close: runs/bench/p2_x/{games.jsonl,games.pgn,rating.json}, runs/bench/uci_parity_*.json
```

The task order interleaves nothing: opponents in the order above, each opening's two games adjacent. A restart
resumes where `games.jsonl` ends. Nothing is re-played that has finished.

## What would make p2_x uninterpretable

**The anchors' own error.** Stockfish's `UCI_Elo` is a fit, and at 1400–2000 it sits near the bottom of the range it
was fitted over; the fit's residuals are not published. HoiChess's ± 20 is CCRL's. K1's interval contains neither.
K2 is the one check on them this round can make; a second family would be stronger (the disagree row).

**Hardware and control.** CCRL rates at controls equivalent to a reference machine. This box is faster, and the
same clock buys every opponent more search than it had there, which plays them stronger than their ratings and
biases K1 low. Not measured here; stated with the number.

**One configuration.** The network plays 48 simulations a move whatever the clock. A rating at a time control
says how strong *this configuration* is, not how strong the network would be with the time its opponents use.
That second question is a different round (sims scaled to the clock).

**The book.** The ladder's 100 four-ply openings, not CCRL's 12-move book. Short openings leave more of the game to
the engines; both sides start from the same positions and swap colours.

## Sign-off

Checked for the draft, 2026-11-12, on this box: `make lint` and `make test` pass (1,212 passed, 4 skipped, with
`tests/test_uci.py` and `tests/test_gauntlet.py`). P1 on cpu 1,108 / 1,108; cuda 1,399 / 1,475 (`runs/bench/uci_parity_*.json`).
Tooling smoke games (not data): 8 games at 10+0.1 against UCI_Elo 1320 and HoiChess (`runs/bench/smoke`), 2 at
5+0.05 against HoiChess (`runs/bench/smoke2`); the first exposed a runner that left engines open after the last
game (the process never exited), fixed, and the second exited cleanly. Stockfish 16 reports `UCI_Elo` 1320–3190.
Disk 86 GB free; host RAM 15 GB available (three network processes take about 1.2 GB each on cuda). RunPod is not
used.

**Registered 2026-11-12** at Michael's instruction ("Ok given, go"), the network on **cuda** (his choice of the
two the draft offered). At registration: the `.gitignore` negations for `runs/bench/p2_x/{games.jsonl,games.pgn,
rating.json}` and `runs/bench/uci_parity_*.json` are added and red-checked through `MUST_REACH` (all five paths
failed without them, pass with them); the engine binaries and the smoke games are pinned ignored (`MUST_IGNORE`).
The launch environment is fixed as above, and P1 cuda is re-measured under it as the run's first step. The
repository is licensed **Apache-2.0** at Michael's choice (`LICENSE`, `pyproject.toml` `license = "Apache-2.0"`),
ahead of publishing the games; the code depends on python-chess (GPL-3.0), which a distributed combination
would carry, and no third-party code is vendored (`upstream/` holds this repo's own ROCm reproducer).
