# Pre-registration — p2_y: two more CCRL-listed engines, to settle the scale p2_x left open

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: DRAFT 2026-11-13, revised after S, NOT signed.** Nothing under this file has run. Michael agreed to
> building and running the two engines and to leaving TSCP out ("Agree on one, leave 2 out"), and chose an
> adapter for Micro-Max ("A small adapter"); both are built, and S (smoke, not data) is done (Sign-off).

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_x (RUNLOG-p2_x) rated p2_w's endpoint against two anchor families, and they disagreed: **Stockfish 16's
UCI_Elo put it at 1,674 [1,644, 1,704]; HoiChess 0.22.0, rated by CCRL from games, at 1,505 [1,447, 1,554].** The
registered row for that outcome publishes both as a range and asks for more CCRL-listed engines near the
network's level before any single number is quoted. Michael said yes to drafting it (2026-11-13).

Stockfish's UCI_Elo is a fit through Ordo-derived results at the bottom of its range; HoiChess's 1806 is a rating
CCRL measured. A second and third engine that CCRL also rated from games say which reading the scale supports:
**do engines that CCRL rates directly agree on the network's rating, and does their rating agree with
Stockfish's?**

## Contract

| | |
|---|---|
| Everything p2_x fixed | **Unchanged**: checkpoint `runs/p2_w_quarter_down/ckpt_iter0010.pt` (sha256 `43494548…3857`), the engine (`scripts/uci_engine.py`, cuda, 48 sims, evaluation profile), the book (`tests/suites/openings_4ply.epd`, 100 openings, each from both sides), **60+0.6**, the game rules (`play_game`, no score adjudication), concurrency 3, this box, `gpu hold` with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. The `play` path is the one p2_x ran: since `1947bc0`, `git diff -- src scripts` touches only `src/tinyalphazero/gauntlet.py` (adds `families_agree`) and `scripts/gauntlet.py` (the `pair_groups` helper, `rate` through it, and `compare`; a485d59); `cmd_play` and `parse_opponent` are byte-identical, and `rate` reproduces p2_x's `rating.json` byte for byte |
| New opponent 1 | **Shallow Blue 2.0.0** — CCRL Blitz **1578 ± 19** (1,119 games), list of 2026-11-07. Source: `github.com/GunshipPenguin/shallow-blue`, tag `v2.0.0`, commit `a04fbd9861770c897eb566d83b0d2e3b17aa9fc0`, MIT licence, cloned to `runs/bench/engines/shallowblue-src/shallow-blue`. Built with its own Makefile and flags plus two forced includes, `make CC_FLAGS="-Wall -std=c++11 -O3 -march=native -flto -pthread -fno-exceptions -include string -include cstdint"`: the 2019 source relies on `<string>` and `<cstdint>` arriving through other headers, which GCC 13's library no longer does; no source line changes. Binary sha256 `0f9777ec…6353`. UCI; `OwnBook` defaults to false (`src/uci.cc:26`), so it plays without a book. Spec: `shallowblue=uci:runs/bench/engines/shallowblue-src/shallow-blue/shallowblue@1578` |
| New opponent 2 | **Micro-Max 4.8** — CCRL Blitz **1854 ± 19** (952 games). Source: H.G. Muller's WinBoard version `umax4_8w.c` from `home.hccnet.nl/h.g.muller/` (sha256 `b15f7530…a07d`, 569 lines), in `runs/bench/engines/micromax-src`. Its one Windows call is `GetTickCount()`: built with `gcc -O2 -I runs/bench/engines/micromax-src/compat` against a three-line `compat/windows.h` that supplies it from `CLOCK_MONOTONIC` (sha256 `73df363e…804f`); binary sha256 `a4752e4e…8ced`. It speaks xboard protocol 1, which python-chess will not drive (it requires `ping` and `setboard`), so it runs behind **`scripts/xboard1_adapter.py`** (`tinyalphazero.xboard1.Relay`; `tests/test_xboard1.py`): the adapter declares the protocol-2 features on the engine's behalf, turns `setboard` into `edit` mode (checked representable for all 100 book positions: no en-passant square, castling rights equal to what edit mode infers), answers `ping` through a marker the engine replies to in order, and completes the engine's four-character promotions with `q` (it promotes only to a queen); nothing else changes. No book. Spec: `micromax=xboard:.venv/bin/python scripts/xboard1_adapter.py -- runs/bench/engines/micromax-src/umax4_8w@1854` |
| Not included | **TSCP 1.81** (CCRL Blitz 1611), the engine the p2_x decision row named: its author publishes 1.81 only as a Windows executable (Wine is not installed and needs sudo), and the source on offer is 1.83b, which CCRL does not list. Shallow Blue takes its place below the network. Wine plus TSCP 1.81 can be added later by addendum if Michael wants it |
| Reused from p2_x | p2_x's **200 HoiChess games** (`runs/bench/p2_x/games.jsonl`): same checkpoint, engine, book, clock, rules, runner and box. The **classical family** is HoiChess (1806), Micro-Max (1854) and Shallow Blue (1578): **600 games**, 400 of them new |
| Games | 200 against each new opponent (100 openings × 2): **400 games**, `--out runs/bench/p2_y` |
| Analysis set | `runs/bench/p2_y_all/games.jsonl` = p2_x's `games.jsonl` then p2_y's, concatenated (1,400 games, eight opponents); every fit below runs on it |
| Duration and cost | about 3 h at concurrency 3 (p2_x averaged 27 s of wall time a game). No pod, no spend |

## Registered expectations

### S — the new engines work (at sign-off, not data)
Each new engine builds, answers its protocol, and plays two games at 10+0.1 against the network without an error,
an illegal move or a hang (smoke, into `runs/bench/smoke_p2_y`; not data). Micro-Max's `level` parser reads an
integer increment (`sscanf(line, "level %d %d %d", ...)`), so at 60+0.6 it may budget as if the increment were 0;
the smoke games record its seconds a move, and the reading is reported (see "What would make p2_y
uninterpretable"). Confirmed at draft: python-chess sends `level 0 1:00 0.6`, which Micro-Max's second `sscanf`
reads with an increment of 0.

### U — the network's under-promotions against Micro-Max (sensitivity)
Micro-Max represents only queen promotions. When the network promotes to a knight, Micro-Max resigns at its next
move (`umax4_8w.c` lines 151 and 330; scored as a Micro-Max loss, termination `error`); to a rook or bishop, it
plays on with a queen on its board in the piece's place (its moves stay legal; its judgement of the material
does not). In p2_x the network promoted to something other than a queen in 96 of 1,000 games (103 of its 284
promotions: 44 rook, 31 bishop, 28 knight). **K1–K3 are computed on all games as played (primary), and again with
every Micro-Max pair in which either game has a network under-promotion dropped** (`gauntlet.py drop-pairs
--opponent micromax`, then the same fits into `runs/bench/p2_y_sens/`). The dropped pairs are counted and
reported.

### G — the games are games
As p2_x, for the 400 new games: **G1** the network loses none by `time_forfeit`, `hung` or `error`; **G2** none
reaches 600 plies; **G3** both new opponents have 100 complete pairs.

### K1 (primary) — the classical-family rating
`bootstrap_rating` over the three classical groups (HoiChess, Micro-Max, Shallow Blue; 600 games), 10,000
resamples, seed 0: `scripts/gauntlet.py compare --out runs/bench/p2_y_all --family-a sf1400,sf1600,sf1800,sf2000
--family-b hoichess,micromax,shallowblue --file k3.json` (family_b). **This is the candidate headline: "CCRL ≈ R
[lo, hi], against three engines CCRL rates from games, at 48 simulations a move".**

### K2 — do the classical engines agree among themselves?
Leave one out, three times: each classical engine's own fit against the fit of the other two
(`compare --family-a <one> --family-b <other two> --file k2_<one>.json`), each with `families_agree`'s rule (each
point inside the other's 95% interval). **K2 holds if all three agree.**

### K3 — does the classical family agree with Stockfish?
`k3.json` above: family_a (the four Stockfish opponents, p2_x's K2 fit, 1,674 [1,644, 1,704]) against family_b
(K1), by the same rule.

### Reported, no decision
Per-opponent scores, W/D/L, terminations and seconds a move for all eight opponents (`scripts/gauntlet.py rate
--out runs/bench/p2_y_all`), and its pooled eight-opponent rating. Each new engine's performance beside its CCRL
rating.

## Predictions (written before launch)

- **K1: 1,350–1,550, point about 1,450.** HoiChess gave 1,505, and the smoke games (not data) went 0/4 against
  Shallow Blue and 0/8 against Micro-Max at 10+0.1, a control that weakens both. (The draft before S said
  1,480–1,620.)
- Against Shallow Blue (1578) the network scores 0.10–0.35; against Micro-Max (1854) 0.02–0.15.
- K2 holds, about even odds. K3 disagrees (Stockfish above the classical family), more likely than not.
- U: 5–15 Micro-Max pairs dropped; U's refit does not change a K2 or K3 verdict.
- G1–G3 hold. Draw rates nearer HoiChess's 20% than Stockfish's 4%.

## Decision rules

| K2 | K3 | G | Decision |
|---|---|---|---|
| **holds** | disagree | hold | **The classical family sets the number.** Published: "CCRL ≈ K1 [lo, hi] at 48 sims/move, against three CCRL-listed engines", with Stockfish's 1,674 beside it as a reading of an uncalibrated anchor at this level for this network. The README and RUNLOG quote K1 and nothing else as the rating |
| **holds** | agree | hold | **The scale is settled.** Published: the pooled eight-opponent rating over 1,400 games (`rate` on the analysis set), with K1 and Stockfish's fit beside it |
| any | any | hold, and U's refit gives a different K2 or K3 verdict | **Not robust to Micro-Max's promotion limit.** Both sets are reported; no single number is quoted alone, as the next row |
| **fails** | any | hold | **No single number at this level.** Published: each engine's performance as the spread. The next diagnosis is the control: the opponents at a clock scaled to CCRL's reference machine (p2_z), before any rating is quoted alone |
| any | any | **fail** | As p2_x: reported, the failed check first; G1 in more than 1% of the new games (≥ 5) makes the round not a strength measurement until its cause is found |

## Measurement protocol

```bash
# Built at draft (Sign-off):
make -C runs/bench/engines/shallowblue-src/shallow-blue CC_FLAGS="-Wall -std=c++11 -O3 -march=native -flto -pthread -fno-exceptions -include string -include cstdint"
gcc -O2 -I runs/bench/engines/micromax-src/compat -o runs/bench/engines/micromax-src/umax4_8w runs/bench/engines/micromax-src/umax4_8w.c
# runs/p2_y_bench.sh as the transient user unit taz-bench-p2_y, under gpu hold, the p2_x environment:
.venv/bin/python scripts/gauntlet.py play --ckpt runs/p2_w_quarter_down/ckpt_iter0010.pt --device cuda \
    --opponents "shallowblue=uci:runs/bench/engines/shallowblue-src/shallow-blue/shallowblue@1578,micromax=xboard:.venv/bin/python scripts/xboard1_adapter.py -- runs/bench/engines/micromax-src/umax4_8w@1854" \
    --openings 100 --tc 60+0.6 --concurrency 3 --out runs/bench/p2_y
# Analysis:
mkdir -p runs/bench/p2_y_all && cat runs/bench/p2_x/games.jsonl runs/bench/p2_y/games.jsonl > runs/bench/p2_y_all/games.jsonl
.venv/bin/python scripts/gauntlet.py rate --out runs/bench/p2_y_all
.venv/bin/python scripts/gauntlet.py compare --out runs/bench/p2_y_all --family-a sf1400,sf1600,sf1800,sf2000 \
    --family-b hoichess,micromax,shallowblue --file k3.json                                     # K1, K3
.venv/bin/python scripts/gauntlet.py compare --out runs/bench/p2_y_all --family-a hoichess \
    --family-b micromax,shallowblue --file k2_hoichess.json                                     # K2 (x3)
.venv/bin/python scripts/gauntlet.py compare --out runs/bench/p2_y_all --family-a micromax \
    --family-b hoichess,shallowblue --file k2_micromax.json
.venv/bin/python scripts/gauntlet.py compare --out runs/bench/p2_y_all --family-a shallowblue \
    --family-b hoichess,micromax --file k2_shallowblue.json
# U: the same fits with Micro-Max's under-promotion pairs dropped, into runs/bench/p2_y_sens (runs/p2_y_bench.sh).
.venv/bin/python scripts/gauntlet.py drop-pairs --src runs/bench/p2_y_all --dst runs/bench/p2_y_sens --opponent micromax
```

## What would make p2_y uninterpretable

**Builds are not CCRL's builds.** CCRL rated binaries compiled by someone else, at another time, perhaps 32-bit. A
different compiler and `-march=native` make these faster, not different: they search the same tree in less time,
which plays them a little above their ratings and biases the network's reading low (as with the hardware below).

**Hardware and control, as p2_x.** This box gives every opponent more search per second of clock than CCRL's
reference machine; it plays them above their ratings. Shared by all three classical engines and by Stockfish.

**Micro-Max and the fractional increment.** If it reads 60+0.6 as 60+0, it plays with less time than its clock
allows and below its rating, biasing the network's reading against it high; the smoke games' seconds a move
show whether it does. Recorded, not corrected: correcting it would change the clock for one opponent only.

**The adapter is ours.** CCRL played Micro-Max through WinBoard; here it plays through
`scripts/xboard1_adapter.py`. The adapter changes the protocol, not the chess: the positions it sets up are
checked, its promotions are the queen the engine plays, and its tests replay the translation end to end. U covers
the one place the engine and the real game can differ.

**Three engines are still three engines.** Each CCRL rating carries about ± 20. K2 tests whether they agree at the
resolution 600 games give; it cannot exclude a shared offset of the whole list at this level.

## Sign-off

Checked for the draft, 2026-11-13, on this box, without building or running either new engine: Shallow Blue
v2.0.0 cloned at `a04fbd9`, MIT; its Makefile compiles `src/*.cc` and does nothing else; a read of `src/` finds no
`system`, `popen`, `exec`, socket or network call, and one file read (`BookPath`, used only when `OwnBook` is
true). Micro-Max 4.8 WinBoard source downloaded (sha256
`b15f7530f430689b90b33155ebffd7bb70204d63953282b6718d934cd295a07d`); a read finds no `system`, `popen`, `exec`,
socket or file call. CCRL Blitz (list of 2026-11-07) checked for both ratings and for TSCP 1.81 (1611; Windows
executable only from its author). The opponent spec format `name=proto:cmd@rating` is the one `scripts/gauntlet.py`
already parses.

**S, 2026-11-13 (smoke, not data), at 10+0.1 from the book's first openings.** Shallow Blue: 4 games, all
complete, network 0/4, Shallow Blue 0.23–0.31 s a move. Micro-Max through the first adapter: 2 of 4 games ended
`error`, both at a Black pawn's promotion that the engine printed in four characters, rejected by python-chess and
scored as Micro-Max losses; the adapter was extended to complete such moves with `q` (`tests/test_xboard1.py`
replays that very position). Through the extended adapter: 8 games, all complete, network 0/8, Micro-Max 0.17–0.21
s a move; it also found `Ra8#` from a `setboard` position and played a legal move from one with all four castling
rights. **Pending: Michael's sign-off.**
