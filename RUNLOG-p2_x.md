# RUNLOG — p2_x (the first public benchmark: a CCRL-anchored gauntlet of p2_w's endpoint at 48 simulations a move)

**Run:** `runs/bench/p2_x`: `p2_w_quarter_down/0010` (sha256 `43494548…3857`, config fingerprint `036cd665…01da`)
as a UCI engine (`scripts/uci_engine.py --device cuda`: the evaluation profile, 48 simulations a move, clocks
ignored) against five opponents of published CCRL rating, 200 games each: Stockfish 16 at `UCI_LimitStrength`,
`UCI_Elo` 1400, 1600, 1800 and 2000, and HoiChess 0.22.0 (CCRL Blitz 1806). The ladder's 100 four-ply openings,
each played twice with colours swapped; 60+0.6 for both sides; no adjudication on engine scores. 1,000 games on this
box, 2026-11-12 17:24 to 2026-11-13 01:06 UTC, under `gpu hold`, as the transient user unit `taz-bench-p2_x`
(`runs/p2_x_bench.sh`), code `1947bc0`. **Pre-registration:** `PREREG-p2_x.md` (`1947bc0`), with Addendum A
(`381977e`, how K2's two intervals are computed). Every verdict below was registered before it was measured, except
where marked.

**Verdict: the registered second row, "the scale is not settled at this level". The two anchor families disagree,
so both numbers are published as a range and no single rating is quoted: on the CCRL scale, at 48 simulations a
move, the network rates 1,674 [1,644, 1,704] against Stockfish 16's `UCI_Elo` (800 games) and 1,505 [1,447, 1,554]
against HoiChess 0.22.0 (200 games). K2: HoiChess − Stockfish = −169, each point outside the other's interval.
K1, the pooled fit over all 1,000 games: 1,642 [1,618, 1,666], not a headline under this row. G1–G3 hold: no
forfeit, hang or error by either side, no game near the 600-ply stop (longest 384 plies), 100 complete pairs per
opponent. K3: the score falls with UCI_Elo (0.800, 0.505, 0.368, 0.220), monotone. P1: on cpu the engine plays
the ladder's moves 1,108 / 1,108; on cuda, under the launch environment, 1,459 / 1,535 (95.0%). p2_y adds TSCP
1.81 and Micro-Max 4.8, two CCRL-listed engines near this level, before any rating is quoted alone (§6); Michael
agreed to drafting it on 2026-11-13.**

---

## 1. Before launch

The tooling went in at `4235eb0` (the UCI front end, `tinyalphazero.uci`; the gauntlet runner and rating fit,
`tinyalphazero.gauntlet` and `scripts/gauntlet.py`; `scripts/uci_parity.py`), the draft at `6cae103`, the Apache-2.0
licence at `d42549c` and the registration at `1947bc0` (2026-11-12, Michael: "Ok given, go", the network on cuda).
`make test` passed 1,212, skipped 4 on the draft's tree. Stockfish 16 (`/usr/games/stockfish`, Ubuntu
`16-1build1`) reports `UCI_Elo` 1320–3190 and runs `bench 16 1 13` at 1.70 Mnps on one core of this box. HoiChess
0.22.0 came from Ubuntu's `hoichess_0.22.0-3build1` package, extracted without installing to
`runs/bench/engines/hoichess` (binary sha256 `cc9cbe8d…3fd8`); it found no book at that path and played without one.

**Smoke games, not data.** Eight games at 10+0.1 (`runs/bench/smoke`: 4/4 against `UCI_Elo` 1320, 0.5/4 against
HoiChess) and two at 5+0.05 against HoiChess (`runs/bench/smoke2`: 0/2). The PREREG's predictions quote "0.5/4
against HoiChess at 10+0.1 and 5+0.05 combined"; combined it was 0.5/6, which only strengthens the reading it was
used for. The first exposed a runner whose engines were never quit after the last game, so the
process never exited; fixed (every engine is quit at the end, and a per-move watchdog closes an engine 30 s past
its own flag). The second exited cleanly. Neither enters any number below.

## 2. P1 — the engine is the ladder's network

`scripts/uci_parity.py`: 20 ladder games (10 book openings × 2 colours against `stockfish:5` at 10,000 nodes,
`BatchRunner` under the evaluation profile, all games concurrent), then every one of the network's decisions
replayed through `scripts/uci_engine.py` as a subprocess, over pipes, one position at a time.

| run | device | positions | identical | agreement | games with a difference | s a move over UCI |
|---|---|---|---|---|---|---|
| draft, `uci_parity_cpu.json` | cpu (fp32) | 1,108 | **1,108** | **1.0000** | 0 of 20 | 0.522 |
| draft, `uci_parity_cuda.json` | cuda (bf16) | 1,475 | 1,399 | 0.9485 | 17 of 20 | 0.114 |
| launch, `uci_parity_cuda_launch.json` | cuda (bf16), `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | 1,535 | **1,459** | **0.9505** | 17 of 20 | 0.113 |

**The cpu gate holds (1,108 / 1,108, as registered).** The cuda figure, reported and not judged, is the launch
run's: **95.0%**. The attention-kernel setting the ladder uses moved it by 0.2 points. The two cuda runs are not
the same games (the ladder half of each ran on cuda, so the games themselves differ), and in both the differences
sit in 17 of the 20 games. **Correction to the PREREG's Contract:** it says the draft's 76 differing moves had "the
first at ply 10". Ply 10 was game 0's first; the earliest over all 20 games was at ply 3 (in the launch run, ply 5).
Nothing registered depends on it.

## 3. G — the games are games

- **G1 holds.** No game ended by `time_forfeit`, `time_forfeit_draw`, `hung`, `hung_draw` or `error`, for either
  side. No side's total thinking time in any game exceeded its base plus increments (checked from the rows).
- **G2 holds.** No game reached the 600-ply stop; the longest was 384 plies (against `UCI_Elo` 1600).
- **G3 holds.** 100 complete pairs against each opponent; no unpaired game.

Every row carries `git_sha` `1947bc0`, the registered checkpoint and config fingerprint, `device` cuda, 48 sims and
60+0.6.

## 4. K — the rating

### K1 (primary) — the anchored rating, all 1,000 games

**1,642 [1,618, 1,666]**, se 12.3 (`runs/bench/p2_x/rating.json`; `anchored_rating`, draws as half points,
`bootstrap_rating` over opening pairs within each opponent, 10,000 resamples, seed 0). Not clamped. Under the
decision row K2 selects, this is not the headline (§6).

### K2 — do the two anchor families agree? (Addendum A)

| family | opponents | games | rating | 95% interval | se |
|---|---|---|---|---|---|
| Stockfish 16 `UCI_Elo` | sf1400, sf1600, sf1800, sf2000 | 800 | **1,674** | [1,644, 1,704] | 15.3 |
| HoiChess 0.22.0 | hoichess | 200 | **1,505** | [1,447, 1,554] | 26.8 |

`runs/bench/p2_x/k2.json`, by `scripts/gauntlet.py compare` (`a485d59`): each family fitted on its own opponents'
pairs alone, 10,000 resamples, seed 0. HoiChess − Stockfish = **−169**. Stockfish's point (1,674) lies above
HoiChess's interval and HoiChess's point (1,505) below Stockfish's, so **they disagree**.

### K3 — the shape (reported)

| opponent (rating) | W-D-L | score | performance | as White | as Black | draws | plies mean (median, max) | s a move: network / opponent | terminations |
|---|---|---|---|---|---|---|---|---|---|
| Stockfish `UCI_Elo` 1400 | 156-8-36 | 0.800 | 1,641 | 0.765 | 0.835 | 4.0% | 87.5 (79, 271) | 0.126 / 1.888 | mate 192, threefold 5, insufficient 2, fifty 1 |
| Stockfish `UCI_Elo` 1600 | 97-8-95 | 0.505 | 1,603 | 0.465 | 0.545 | 4.0% | 105.6 (94.5, 384) | 0.135 / 1.651 | mate 192, insufficient 4, threefold 4 |
| Stockfish `UCI_Elo` 1800 | 69-9-122 | 0.3675 | 1,706 | 0.435 | 0.300 | 4.5% | 103.3 (93, 303) | 0.122 / 1.672 | mate 191, threefold 7, insufficient 2 |
| Stockfish `UCI_Elo` 2000 | 40-8-152 | 0.220 | 1,780 | 0.310 | 0.130 | 4.0% | 108.4 (100.5, 251) | 0.120 / 1.616 | mate 192, threefold 4, insufficient 4 |
| HoiChess 0.22.0 (1806) | 10-40-150 | 0.150 | 1,505 | 0.140 | 0.160 | 20.0% | 84.5 (80, 195) | 0.124 / 0.970 | mate 160, threefold 40 |

The score falls with `UCI_Elo` at every step, so **K3 is monotone**. The performances are not flat: they rise from
1,641 and 1,603 to 1,706 and 1,780 (reading 1). The network's search took 0.125 s a move over the whole run.

**Reported beside K1, as registered:** the checkpoint's in-house skill-5 reading is 0.379 (six reps against
`stockfish:5` at 10,000 nodes; RUNLOG-p2_w). Its CCRL-scale reading is 1,505–1,674. That gives the first point of a
mapping from the ladder's scale to CCRL's, not a mapping.

## 5. The predictions

| prediction (written before launch) | result | |
|---|---|---|
| K1 1,450–1,750, point about 1,600 | 1,642 | held |
| K1's half-width 25–45 | 24.3 (1,617.7 to 1,666.3) | **missed**, just narrower |
| K2 agree, more likely than not; if not, HoiChess below the Stockfish-only rating | disagree; HoiChess 169 below | first part **missed**, second held |
| G1–G3 hold | hold | held |
| draw rate 10–25% against each opponent | 4.0–4.5% against each Stockfish setting; 20.0% against HoiChess | **missed** four of five |
| against `UCI_Elo` 1400: 0.65–0.90; against 2000: 0.05–0.25 | 0.800; 0.220 | held |

## 6. Decision

Per the table's second row (K2 disagree; G1–G3 hold): **the scale is not settled at this level. Both numbers are
published as a range, with no single headline: at 48 simulations a move, the network rates 1,674 [1,644, 1,704] on
the CCRL scale against Stockfish 16's `UCI_Elo` and 1,505 [1,447, 1,554] against HoiChess 0.22.0. p2_y adds
CCRL-listed engines near the network's level (TSCP 1.81, 1611; Micro-Max 4.8, 1854; both CCRL Blitz) before any
rating is quoted alone.** Michael agreed to drafting p2_y on 2026-11-13. The games (`games.pgn`, `games.jsonl`),
`rating.json` and `k2.json` are committed with this log.

Readings the table does not cover (all unregistered):

1. **Stockfish's 600 points play as about 460 against this network.** At the Stockfish-only fit (1,674), the
   logistic expects 0.829, 0.605, 0.326 and 0.133 against 1400, 1600, 1800 and 2000; the network scored 0.800,
   0.505, 0.368 and 0.220. Read the other way, the four settings behave like opponents of 1,433, 1,670, 1,768 and
   1,894: a span of 461 for a nominal 600. So the Stockfish-only rating depends on which settings a gauntlet uses
   (1,641 from 1400 alone, 1,780 from 2000 alone). Stockfish's `UCI_Elo` is fitted over 1320–3190 and anchored at
   goldfish 1.13 = 2000, so 1400–2000 is the low end of its fit. Whether the compression is in that fit, or in how
   this network meets a skill-limited opponent, the round cannot say.
2. **Two effects in opposite directions, neither measured.** HoiChess plays on a faster core than CCRL's reference
   machine, at a clock near its list control (§1), so it plays stronger than its 1806 and its implied performance
   for the network sits low. At the Stockfish-only fit HoiChess behaves like an opponent of 1,975, 169 above its
   list rating. Stockfish's limiter weakens it by picking, with a randomised choice, among moves it knows are worse.
   That is a different kind of weakness from a consistent engine's, and the CCRL fit behind `UCI_Elo` need not
   carry over to it. Short games are not where the wins come from: the network's wins inside 40 plies were 2 of
   156, 4 of 97, 2 of 69 and 0 of 40 against the four settings. The CCRL-scale figure plausibly lies between the two
   readings. p2_y's two extra anchors are the test.
3. **Draws were rare against Stockfish.** 33 of 800 (4.0–4.5% at each setting), against a predicted 10–25%; 767
   of those 800 games ended in mate. Against HoiChess, 40 of 200 were drawn, **every one by threefold repetition**.
   The ladder's ~30% draws against `stockfish:5` (under its 300-ply cap) did not carry over to games without a cap
   against time-controlled opponents.
4. **Colour.** Against 1400 and 1600 the network scored more as Black (0.835 and 0.545, against 0.765 and 0.465 as
   White); against 1800 and 2000 more as White (0.435 and 0.310, against 0.300 and 0.130); against HoiChess about
   evenly (0.140 White, 0.160 Black). 100 games per colour per opponent; reported, not interpreted.
5. **Time.** The network spent 0.120–0.135 s a move against each opponent (22.2 s in its longest game), as its
   fixed budget implies; Stockfish 1.62–1.89 s, HoiChess 0.97 s. This round rates the 48-simulation configuration
   only, not the network given its opponents' time (PREREG, "One configuration").
6. **The book contains lopsided positions.** The shortest game was 2 plies: from `open_013`
   (`rnbqk1nr/pppp1ppp/8/4p3/1b3PP1/8/PPPPP2P/RNBQKBNR w KQkq -`, White to move with …Qh4# threatened), Stockfish at
   1600 as White played fxe5 and was mated. The network as White lost the same opening to Stockfish at 1400 in 10
   plies. Colour-swapped pairs cancel such positions in expectation.

The open list, carried forward: **p2_y, TSCP 1.81 and Micro-Max 4.8 as further CCRL anchors (the registered next
step)**; a Lichess BOT (a human-pool rating; needs Michael's account); a submission to CCRL's testers (needs a
release); the network given its opponents' time (simulations scaled to the clock); CCRL's own 12-move book; and,
from p2_w, a deeper engine, the 2e-7 step-down, per-source loss weights, the calibration sample, the promotion loss
at 2e-6, L2's exclusion at storage time and the third line (p2_t §6, reading 4).

## 7. Mechanics

- **Unattended.** `taz-bench-p2_x` armed at 2026-11-12 17:20:12 UTC at `1947bc0` and took `gpu hold --by
  tinyalphazero` (ComfyUI not running, nothing stopped, 101.8 GB free) with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
  PYTHONUNBUFFERED=1`. It held the GPU for the whole run, to 01:06:44. P1 on cuda 17:20:12–17:24:19 (the ladder half
  in 72 s). Gauntlet from 17:24:19; the first game finished at 17:25:39 and the last at 01:06:31; `rate` at
  01:06:36; complete at 01:06:44. The unit armed once and never restarted (`NRestarts=0`, `Result=success`). No hand intervention.
- **Pace.** 7 h 42 min for 1,000 games at concurrency 3, against the registration's "about 6 h". Games against
  Stockfish took 88–94 s of thinking a game, against the 65 s assumed; games against HoiChess took 46 s. The
  console's per-game lines arrive in blocks, because the unit pipes them through `grep`; `games.jsonl` is the
  record, written as each game ends.
- **K2's code.** Addendum A was written at about 18:00 UTC on 2026-11-12, with 37 games in and no rating computed
  (`381977e`). The code it specifies went in at `a485d59` (2026-11-13 02:43:58 UTC), after the run had ended and
  before it was run on the games: `scripts/gauntlet.py compare` and the shared `pair_groups` helper. On the run's
  `games.jsonl`, `rate` through the helper reproduces the registered `rating.json` byte for byte. `k2.json` was then
  written. The result was scored at the 02:42 check-in and Michael was notified; he saw the verdict and agreed to
  closing.
- **The box** stayed up throughout (up since 2026-11-05 12:10); three games at a time did not trip the CPU's
  thermal limit.
- **Cost:** no spend. Nothing ran on RunPod.

## 8. Artefacts

| file | what |
|---|---|
| `runs/bench/p2_x/games.jsonl` | one row per game: opponent, opening, colour, result, termination, plies, each side's seconds and moves, start FEN, moves, checkpoint, config fingerprint, sha (G1–G3, K3) |
| `runs/bench/p2_x/games.pgn` | the same 1,000 games as PGN |
| `runs/bench/p2_x/rating.json` | K1, and each opponent's performance, W/D/L, terminations and seconds a move (K3) |
| `runs/bench/p2_x/k2.json` | K2: the Stockfish-only and HoiChess-only fits and the agreement (Addendum A) |
| `runs/bench/uci_parity_cpu.json`, `uci_parity_cuda.json`, `uci_parity_cuda_launch.json` | P1: the draft's cpu and cuda runs and the launch environment's cuda run, with every differing move |
| `runs/p2_x_bench.sh`, `runs/p2_x_bench.console.log` | the unit's script and console (untracked) |
| `runs/bench/smoke/`, `runs/bench/smoke2/` | tooling smoke games, not data (ignored) |
| `runs/bench/engines/hoichess/` | the extracted HoiChess package (ignored) |
