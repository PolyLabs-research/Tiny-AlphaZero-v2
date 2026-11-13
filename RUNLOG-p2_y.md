# RUNLOG — p2_y (two more CCRL-listed engines, to settle the scale p2_x left open)

**Run:** `runs/bench/p2_y`: `p2_w_quarter_down/0010` (sha256 `43494548…3857`, config fingerprint `036cd665…01da`)
as p2_x played it (`scripts/uci_engine.py --device cuda`, the evaluation profile, 48 simulations a move, clocks
ignored) against two new opponents of published CCRL Blitz rating, 200 games each: **Shallow Blue 2.0.0** (1578, UCI,
built from tag `v2.0.0`) and **Micro-Max 4.8** (1854, xboard protocol 1, behind `scripts/xboard1_adapter.py`). The
ladder's 100 four-ply openings, each played twice with colours swapped; 60+0.6 for both sides; no adjudication on
engine scores. 400 games on this box, 2026-11-13 16:55 to 19:21 UTC, under `gpu hold`, as the transient user unit
`taz-bench-p2_y` (`runs/p2_y_bench.sh`), code `7af10de`. With p2_x's 200 HoiChess games they make the **classical
family** (600 games); with all of p2_x, the analysis set (1,400 games). **Pre-registration:** `PREREG-p2_y.md`
(`7af10de`). Every verdict below was registered before it was measured, except where marked.

**Verdict: the registered fourth row, "no single number at this level". K2 fails: no classical engine's own fit
agrees with the other two's. HoiChess 1,505 [1,447, 1,554] against Micro-Max and Shallow Blue's 1,269 [1,204,
1,321]; Micro-Max 1,467 [1,376, 1,535] against 1,310 [1,261, 1,350]; Shallow Blue 1,154 [1,067, 1,224] against
1,489 [1,442, 1,529]. K1, the classical family's rating, is 1,347 [1,305, 1,385] and is not a headline under this
row. K3: it disagrees with Stockfish's 1,674 [1,644, 1,704] by −327. U: 4 Micro-Max pairs had a network
under-promotion, and the refit without them gives every verdict unchanged. G1–G3 hold: the network lost no game by
forfeit, hang or error, the longest game was 372 plies, and both new opponents have 100 complete pairs. Published,
as the row says, is each engine's performance as the spread: Shallow Blue 1,154, Micro-Max 1,467, HoiChess 1,505,
Stockfish's four settings 1,603–1,780 (fit 1,674). The registered next diagnosis is p2_z, the opponents at a clock
scaled to CCRL's reference machine. Michael chose the capacity-or-data diagnostic (DIAG-2026-11-13) to run next on
the training track, with p2_z kept for the benchmark track (§6). Two Micro-Max games ended in an unexplained harness
failure on its first move (§3); without them every verdict stands.**

---

## 1. Before launch

The tooling went in at `f13e03a` (`tinyalphazero.xboard1.Relay` and `scripts/xboard1_adapter.py`, the protocol-1
adapter; `gauntlet.py drop-pairs` and `network_underpromotes`, U's sensitivity set). The drafts are `d4402d2` and
`46bb89f` (revised after S), and the registration is `7af10de` (2026-11-13, Michael: "OK given, go"). Michael agreed
to building and running the two engines and to leaving TSCP 1.81 out ("Agree on one, leave 2 out"): its author
publishes 1.81 only as a Windows executable. He chose the adapter for Micro-Max ("A small adapter"). `make test`
passed 1,241, skipped 5 before the last tooling tests went in; the adapter's and `drop-pairs`' tests passed in their
own run.

**The builds.** Shallow Blue from `github.com/GunshipPenguin/shallow-blue` at `a04fbd9` (MIT), with its own Makefile
and two forced includes (`-include string -include cstdint`; the 2019 source relies on headers GCC 13 no longer
brings in; no source line changed), binary sha256 `0f9777ec…6353`. Micro-Max 4.8 from H.G. Muller's `umax4_8w.c`
(sha256 `b15f7530…a07d`), built against a three-line `compat/windows.h` that supplies its one Windows call,
`GetTickCount()`, binary sha256 `a4752e4e…8ced`. A read of both sources found no shell, process or network call.

**S, as recorded at registration (smoke games, not data, 10+0.1).** Shallow Blue: 4 games, all complete, network
0/4. Micro-Max through the first adapter: 2 of 4 games ended `error`, both at a Black pawn's promotion the engine
printed in four characters, which python-chess rejected. The adapter now completes such moves with `q` (the engine
promotes only to a queen). Through the extended adapter: 8 games, all complete, network 0/8. Micro-Max also found
`Ra8#` from a `setboard` position and played a legal move from one with all four castling rights. **S is met** by
the final adapter.

## 2. G — the games are games

- **G1 holds.** The network lost no game by `time_forfeit`, `time_forfeit_draw`, `hung`, `hung_draw` or `error`.
  Two games ended `error` on **Micro-Max's** side and are scored as network wins (§3); G1 is about the network and
  is not touched by them.
- **G2 holds.** No game reached the 600-ply stop; the longest was 372 plies (against Shallow Blue).
- **G3 holds.** 100 complete pairs against each new opponent; no unpaired game.

Every new row carries `git_sha` `7af10de`, the registered checkpoint and config fingerprint, `device` cuda, 48 sims and
60+0.6. The analysis set is exactly p2_x's `games.jsonl` followed by p2_y's (1,400 rows).

## 3. An unexplained harness failure (unregistered)

Two Micro-Max games ended `error` **after one ply**: `open_005` and `open_077`, both with the network as White. The
network played its move (`c1b2`, `b4b5`), and Micro-Max's first `play` call failed. They finished at 18:25:54 and
19:08:30. The runner scores an engine's failure as that engine's loss, so both are network wins: 2 of the network's
13 wins against Micro-Max (its 2 wins inside 40 plies are these two).

- **Neither reproduces in isolation.** Through the adapter, from the same positions, Micro-Max played `g8f6` and
  `c6d4`, both legal.
- **Not Micro-Max's under-promotion flag.** `UnderProm`, which makes it resign after an opponent's knight promotion,
  is reset by `InitGame` (`umax4_8w.c` line 279) at every `new`. The first failure, at 18:25, also came before every
  game in which the network under-promoted against Micro-Max.
- **The cause is not known.** `play_game` catches the engine's exception and records only `error`, so the message
  that would say what happened was discarded. **The runner should log the exception before the next gauntlet.**
- **The verdict does not depend on them.** With both pairs dropped (by `bootstrap_rating`, 10,000 resamples, seed 0;
  unregistered): Micro-Max 1,451 [1,355, 1,520] (196 games, score 0.089), K1 1,342 [1,299, 1,379], K2 disagree for
  all three (HoiChess 1,505 against 1,258; Micro-Max 1,451 against 1,310; Shallow Blue 1,154 against 1,482), K3
  disagree. Every registered verdict is unchanged.

## 4. K — the scale

### K1 (primary) — the classical family's rating

**1,347 [1,305, 1,385]**, se 20.4, over 600 games (`runs/bench/p2_y_all/k3.json`, family_b: HoiChess, Micro-Max and
Shallow Blue; `bootstrap_rating`, 10,000 resamples, seed 0). Under the decision row K2 selects, this is not the
headline (§6).

### K2 — do the classical engines agree among themselves?

| left out | its own fit | the other two | difference | agree |
|---|---|---|---|---|
| HoiChess (1806) | **1,505** [1,447, 1,554], se 26.8, 200 games | 1,269 [1,204, 1,321], 400 games | −236 | no |
| Micro-Max (1854) | **1,467** [1,376, 1,535], se 40.0, 200 games | 1,310 [1,261, 1,350], 400 games | −158 | no |
| Shallow Blue (1578) | **1,154** [1,067, 1,224], se 41.4, 200 games | 1,489 [1,442, 1,529], 400 games | +335 | no |

`runs/bench/p2_y_all/k2_{hoichess,micromax,shallowblue}.json`, by `gauntlet.py compare` and `families_agree`. In each
case at least one point lies outside the other's interval, so **K2 fails**.

### K3 — does the classical family agree with Stockfish?

Stockfish's four settings, 1,674 [1,644, 1,704] (800 games), against the classical family, 1,347 [1,305, 1,385]:
**−327, disagree** (`k3.json`).

### U — the network's under-promotions against Micro-Max (sensitivity)

In 4 of the 200 Micro-Max games the network promoted to something other than a queen: `open_013` (as Black, a
knight), `open_061` (as White, a rook), `open_072` (as White, a knight) and `open_081` (as White, a rook), 4 of its
17 promotions there (13 queen, 2 knight, 2 rook). `drop-pairs` dropped those **4 pairs (8 games)**. Refitted on the
remaining 1,392 games (`runs/bench/p2_y_sens/`):

| check | all games | U's refit |
|---|---|---|
| K1 | 1,347 [1,305, 1,385] | 1,344 [1,302, 1,382] |
| K2, HoiChess left out | 1,505 vs 1,269: disagree | 1,505 vs 1,262: disagree |
| K2, Micro-Max left out | 1,467 vs 1,310: disagree | 1,460 vs 1,310: disagree |
| K2, Shallow Blue left out | 1,154 vs 1,489: disagree | 1,154 vs 1,486: disagree |
| K3 | −327: disagree | −330: disagree |

**U's refit gives every verdict unchanged.** The resignation path PREREG-p2_y named did not fire: after both knight
promotions and one rook promotion, Micro-Max took the new piece on its next move. After the other rook (`open_081`)
it played on, with a queen on its board where the rook stood, and was mated.

### Reported, no decision

The analysis set's pooled rating, over all **seven** opponents (the PREREG says "eight"; it is four Stockfish
settings and three classical engines): **1,555 [1,534, 1,574]**, se 10.2, 1,400 games (`p2_y_all/rating.json`).

| opponent (rating) | W-D-L | score | performance | as White | as Black | draws | plies mean (median, max) | s a move: network / opponent | terminations |
|---|---|---|---|---|---|---|---|---|---|
| Stockfish `UCI_Elo` 1400 | 156-8-36 | 0.800 | 1,641 | 0.765 | 0.835 | 4.0% | 87.5 (79, 271) | 0.126 / 1.888 | mate 192, threefold 5, insufficient 2, fifty 1 |
| **Shallow Blue 2.0.0 (1578)** | 12-8-180 | **0.080** | **1,154** | 0.060 | 0.100 | 4.0% | 93.5 (88, 372) | 0.121 / 1.574 | mate 192, insufficient 6, threefold 2 |
| Stockfish `UCI_Elo` 1600 | 97-8-95 | 0.505 | 1,603 | 0.465 | 0.545 | 4.0% | 105.6 (94.5, 384) | 0.135 / 1.651 | mate 192, insufficient 4, threefold 4 |
| Stockfish `UCI_Elo` 1800 | 69-9-122 | 0.368 | 1,706 | 0.435 | 0.300 | 4.5% | 103.3 (93, 303) | 0.122 / 1.672 | mate 191, threefold 7, insufficient 2 |
| HoiChess 0.22.0 (1806) | 10-40-150 | 0.150 | 1,505 | 0.140 | 0.160 | 20.0% | 84.5 (80, 195) | 0.124 / 0.970 | mate 160, threefold 40 |
| **Micro-Max 4.8 (1854)** | 13-13-174 | **0.098** | **1,467** | 0.125 | 0.070 | 6.5% | 84.7 (78.5, 248) | 0.122 / 1.085 | mate 185, threefold 11, error 2, fifty 2 |
| Stockfish `UCI_Elo` 2000 | 40-8-152 | 0.220 | 1,780 | 0.310 | 0.130 | 4.0% | 108.4 (100.5, 251) | 0.120 / 1.616 | mate 192, threefold 4, insufficient 4 |

The Stockfish and HoiChess rows are p2_x's games, unchanged. Micro-Max's "as White" includes the two failed games;
without them it is 0.107 over 98 games.

## 5. The predictions

| prediction (written before launch) | result | |
|---|---|---|
| K1 1,350–1,550, point about 1,450 | 1,347 | **missed**, 3 below the range |
| against Shallow Blue 0.10–0.35 | 0.080 | **missed**, below |
| against Micro-Max 0.02–0.15 | 0.098 (0.089 without the two failed games) | held |
| K2 holds, about even odds | fails for all three | **missed** |
| K3 disagrees, more likely than not | disagrees, −327 | held |
| U: 5–15 Micro-Max pairs dropped | 4 | **missed**, 1 below |
| U's refit does not change a K2 or K3 verdict | unchanged | held |
| G1–G3 hold | hold | held |
| draw rates nearer HoiChess's 20% than Stockfish's 4% | Shallow Blue 4.0%, Micro-Max 6.5% | **missed** |

## 6. Decision

Per the table's fourth row (K2 fails; G1–G3 hold; U's refit agrees): **no single number at this level. Published:
each engine's performance as the spread. At 48 simulations a move, on the CCRL scale, the network performs at 1,154
[1,067, 1,224] against Shallow Blue 2.0.0, 1,467 [1,376, 1,535] against Micro-Max 4.8, 1,505 [1,447, 1,554] against
HoiChess 0.22.0, and 1,603–1,780 against Stockfish 16's `UCI_Elo` settings (their fit 1,674 [1,644, 1,704]). The
next diagnosis is the control: the opponents at a clock scaled to CCRL's reference machine (p2_z), before any rating
is quoted alone.** Michael chose, on 2026-11-13, to run the capacity-or-data diagnostic (DIAG-2026-11-13) next, on
the training track. p2_z stays the benchmark track's next step and is not scheduled. The games (`games.jsonl`,
`games.pgn`) and every fit (`p2_y_all/`, `p2_y_sens/`) are committed with this log.

Readings the table does not cover (all unregistered):

1. **The failure of K2 is Shallow Blue's.** HoiChess and Micro-Max agree with each other: 1,505 [1,447, 1,554]
   against 1,467 [1,376, 1,535] in the registered set, and against 1,451 [1,355, 1,520] with the two failed games
   out (`families_agree`: each point inside the other's interval, both ways). Together they give 1,489 [1,442,
   1,529] (1,482 [1,436, 1,522] without the failed games). Shallow Blue's 1,154 sits about 335 below them. Measured
   from their pair, Shallow Blue plays like an opponent of about 1,910 against this network, not 1,578. Why, the
   round cannot say. Its 2019 search on a fast core with `-march=native`, a CCRL rating measured on other builds and
   hardware, and how differently engines gain from extra time are all candidates, and none was measured. Every
   classical engine also sits far below Stockfish (§4), so the spread runs from about 1,150 to about 1,780 depending
   on whom the network plays.
2. **Micro-Max played with less time than its clock allowed**, as the PREREG foresaw. It reads the 0.6-second
   increment as 0 (PREREG-p2_y, S). It used on average 53% of its budget (base plus increments; at most 89% in
   any game) and 1.09 s a move. Shallow Blue used 83% (at most 97.5%) and 1.57 s a move. A Micro-Max with its full
   budget would play stronger, so the network's 1,467 against it is, if anything, high.
3. **Under-promotions.** Against Shallow Blue the network promoted 15 times: 10 queen, 4 rook, 1 knight. Against
   Micro-Max 17: 13 queen, 2 knight, 2 rook. The habit p2_x showed (103 of 284 under-promotions) holds at about the
   same rate here (9 of 32). An unregistered Stockfish screen of p2_x's 103 (2026-11-13, not committed) found it
   cost the result of no game.
4. **Draws were rare again.** 8 of 200 against Shallow Blue (6 by insufficient material, 2 by threefold), 13 of 200
   against Micro-Max (11 threefold, 2 fifty-move), against HoiChess's 40 of 200, all of them threefold.
5. **Time.** The network spent 0.121–0.122 s a move against both new opponents. The 400 games took 2 h 26 min of
   play (first finished 16:56:49, last 19:21:18) at concurrency 3, against "about 3 h" registered.

The open list, carried forward: **p2_z, the opponents at a clock scaled to CCRL's reference machine (the registered
next step, benchmark track)**; the runner logging an engine's exception (§3); TSCP 1.81 through Wine (needs sudo);
a Lichess BOT (needs Michael's account); a submission to CCRL's testers (needs a release); the network given its
opponents' time; CCRL's own 12-move book; and, from p2_w, a deeper engine, the 2e-7 step-down, per-source loss
weights, the calibration sample, the promotion loss at 2e-6, L2's exclusion at storage time and the third line
(p2_t §6, reading 4). **Training track next: DIAG-2026-11-13, capacity or data.**

## 7. Mechanics

- **Unattended.** `taz-bench-p2_y` armed at 2026-11-13 16:55:30 UTC at `7af10de` and took `gpu hold --by
  tinyalphazero` (ComfyUI not running, nothing stopped, 101.8 GB free) with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
  PYTHONUNBUFFERED=1`. Gauntlet from 16:55:31; the first game finished at 16:56:49 and the last at 19:21:18. Then the
  registered analysis at 19:21:37, U's refit at 19:22:22, complete at 19:23:07. The unit armed once and never
  restarted (`NRestarts=0`, `Result=success`). No hand intervention.
- **Health check.** The first six games, all against Shallow Blue, were read at about 16:59: complete, with no error,
  0.12–0.16 s a move for the network and 1.4–1.9 s for Shallow Blue.
- **Commits during the run.** Three commits went to `main` while the unit ran: `598d892` (17:28), `02a3cdb`
  (19:16) and `bee0fcf` (19:20). They carry the capacity-or-data diagnostic's plan and tooling. `git diff 7af10de
  HEAD` is empty over every path the run executes: `scripts/gauntlet.py`, `src/tinyalphazero/gauntlet.py`, the UCI
  front end and engine script, the adapter, `mcts/`, `model.py`, `train.py`, `encoding.py`, `config.py`. The
  analysis at 19:21 therefore ran the registration's code. They are still tracked edits made mid-run, which this
  project avoids, and are recorded here as such.
- **Scoring.** The result was scored at the 20:27 check-in, after the log's "p2_y complete". Michael was notified,
  saw the verdict and agreed to closing (2026-11-13).
- **The box** stayed up throughout; three games at a time did not trip the CPU's thermal limit.
- **Cost:** no spend. Nothing ran on RunPod.

## 8. Artefacts

| file | what |
|---|---|
| `runs/bench/p2_y/games.jsonl` | one row per new game: opponent, opening, colour, result, termination, plies, each side's seconds and moves, start FEN, moves, checkpoint, config fingerprint, sha (G1–G3, §3) |
| `runs/bench/p2_y/games.pgn` | the same 400 games as PGN |
| `runs/bench/p2_y_all/rating.json` | the pooled seven-opponent rating, and each opponent's performance, W/D/L, terminations and seconds a move |
| `runs/bench/p2_y_all/k3.json` | K1 (family_b) and K3: the classical family against Stockfish's four settings |
| `runs/bench/p2_y_all/k2_{hoichess,micromax,shallowblue}.json` | K2: each classical engine left out against the other two |
| `runs/bench/p2_y_sens/rating.json`, `k3.json`, `k2_*.json` | U: the same fits without the 4 Micro-Max pairs with a network under-promotion |
| `runs/bench/p2_y_all/games.jsonl`, `runs/bench/p2_y_sens/games.jsonl` | the analysis and sensitivity sets (regenerable concatenation and filter; ignored) |
| `runs/p2_y_bench.sh`, `runs/p2_y_bench.console.log` | the unit's script and console (untracked) |
| `runs/bench/smoke_p2_y/`, `runs/bench/smoke_p2_y_mm/` | S's smoke games, not data (ignored) |
| `runs/bench/engines/shallowblue-src/`, `runs/bench/engines/micromax-src/` | the two engines' sources and builds (ignored) |
