# Pre-registration — p2_ad: which half of p2_ac's new data paid, position for position

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-18** at Michael's instruction ("ok given, go"). Before registration only the data
> preparation the Sign-off lists had run (the two rings built and checked from rows already labelled; nothing
> trained, no rep played).

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_ac: 31,892,957 teacher positions beat E8's 8,000,000 at skill 5, **+0.0875 [+0.0694, +0.1055]** (E32, RUNLOG-p2_ac),
3.8 times p2_ab's +0.023 for the step before. The extra 23,892,957 rows changed two things at once: their number, and
their kind, 8,845,607 from this project's older self-play (networks two to five weeks before D4's data) and
15,047,350 from rated human games on Lichess (January–February 2016). RUNLOG-p2_ac §6 reading 1: the round cannot say
which half paid. Michael chose the separation for p2_ad ("proceed as suggested", 2026-11-18), at no cost: every row it
needs is already labelled.

**Two arms, equal in size, differing only in the source of their last 8,845,607 rows.** Each is E8's 8,000,000 rows
plus 8,845,607 more: **S17** adds all of pool 3's older self-play; **H17** adds the same number of pool 3's human-game
rows (the first 8,845,607 in pool order; pool 3 was shuffled when built, so they are a uniform draw of its 15,047,350).
Both are trained from the start as E8 and E32 were, at one seed, so they draw the same slots of every ring in the
same order (`distill.sample_union`, pinned by `test_sample_union_draws_the_same_slots_from_parts_of_the_same_sizes`).
What differs is which 8,845,607 rows sit in the fourth ring.

**Position for position, at 16,845,607 rows, do human-game positions train a stronger network at skill 5 than this
project's older self-play?**

The answer chooses the source of the next data. Older self-play is used up (pool 3 took every distinct position the
37 rings hold); more of it means generating fresh self-play with the standing network, and labelling it. Lichess is
effectively unlimited, about $1.19 per million positions labelled (RUNLOG-p2_ac).

## Contract

| | |
|---|---|
| Start | `runs/p2_w_quarter_down/ckpt_iter0010.pt` (sha256 `43494548…3857`), as D4, R, E8 and E32 |
| The two rings | `scripts/ring_by_source.py --ring runs/p2_ac/teacher --tags runs/p2_ac/pool` with `--source selfplay` → `runs/p2_ad/ring_selfplay`, and `--source lichess --limit 8845607` → `runs/p2_ad/ring_lichess`. The labelled ring's `iteration` field is the teacher tag on every row; each row's source is read from pool 3's (`lichess.source_rows`: −3 a Lichess row, ≥ 0 a self-play row, any other tag refused), after all 23,892,957 boards of the two are compared equal. Every field of the chosen rows is copied in ring order (`ReplayBuffer.from_rows`). Built before registration (Sign-off): **8,845,607 rows each**; the Lichess ring's last slot is 14,043,773 of pool 3 |
| S17 | `scripts/distill_curve.py --start <start> --d1 runs/p2_u_fresh/teacher --new runs/diag_cod/teacher --new runs/p2_ab/teacher --new runs/p2_ad/ring_selfplay --holdout runs/p2_u_fresh/teacher_holdout --sizes 16845607 --max-samples 67382428 --seed 4 --out runs/p2_ad/s17 --device cuda`: D1's 475,696, pool 1's 3,524,304, pool 2's 4,000,000 (E8's 8,000,000) and the self-play ring's 8,845,607; lr 2e-5, batch 1,024, teacher rows only, held-out every 250,000 samples, patience 4, the cap at four passes (as E8 and E32); the checkpoint best by held-out cross-entropy |
| H17 | the same command with `--new runs/p2_ad/ring_lichess --out runs/p2_ad/h17` |
| The references (on file) | **E8** (`runs/p2_ab/e8/size_8000000/best.pt`, 24 skill-5 reps: 12 from p2_ab, 12 from p2_ac) and **E32** (`runs/p2_ac/e32/size_31892957/best.pt`, 24 reps). No new reps of either |
| Reps | **24 skill-5 reps per arm**, played in blocks of 12 in the order S17, H17, S17, H17 (so drift in the instrument over the ~7 h falls on both arms alike). K1 compares 24 against 24 (about ±0.018 at skill 5, as p2_ac's K1) |
| Instrument | `eval_reps.py run --anchor stockfish:5 --config runs/p2_w_quarter_down/config.yaml`; verdicts by `eval_reps.py verdict` (per-opening rep-means, paired bootstrap over the 100 openings, 10,000 resamples, seed 0) |
| Where, time, cost | this box only, under `gpu hold`: two trainings (at ~1,680 samples a second, about 3.3–4.3 h each if they stop near E8's and E32's 19.75M–25.75M samples; 11 h each at the cap), 48 reps (about 7 h), two suite rows. **About 15 h end to end, 30 h at worst; $0** (no pod). As the transient user unit `taz-p2ad` (`runs/p2_ad_driver.sh`). Slower, not different, if another project's GPU work runs outside `gpu hold` |

## Registered expectations

### L — the data are what this file says
L1: `ring_selfplay` holds 8,845,607 rows, every self-play row of pool 3; `ring_lichess` holds 8,845,607 rows, the first
8,845,607 Lichess rows of pool 3 in pool order; each ring's every field equals `runs/p2_ac/teacher`'s at its slots; the
two are disjoint and, as rows of pool 3, share no position with D1, the held-out ring, pool 1 or pool 2 (p2_ac's L1).
Checked before registration (Sign-off). L2: each arm's `result.json` records 16,845,607 rows. If L fails, nothing
below is read until it is explained.

### C — the teacher measure (reported)
Each arm's best held-out cross-entropy, top-1 and value MSE beside E8's (2.2028, 0.3844, 0.0319) and E32's (2.1892,
0.3912, 0.0280), the samples at its best, whether it ended at the cap, and the two arms side by side at matched
samples. The held-out ring is self-play positions: S17's fourth ring is of its kind, H17's is not.

### K1 (primary) — H17 against S17, skill 5, 24 reps each
`p2ad_verdict_h17_vs_s17.json` (H17 − S17).

### K2 — mate-in-2
Each arm's suite row (128 sims, the start's config). **K2 passes if each arm solves at least 183 of 200.**

### Reported, no decision
S17 − E8 and H17 − E8 (each source's 8,845,607 rows added to E8's 8M: a doubling, beside p2_ab's +0.023), E32 − S17 and
E32 − H17 (the other source's rows added; E32 − H17 also adds 6,201,743 more human rows), mate-in-3, each arm's
self-play-held-out curve.

## Predictions (written before launch)

- L holds.
- C: S17's held-out CE 2.192–2.200, H17's 2.195–2.205 (the held-out ring is self-play); both stop on patience before
  the cap.
- **K1: point +0.00 to +0.05; the interval excludes zero, about even odds.** E32's gain was far above p2_ab's rate per
  doubling, and 63% of its new rows were human games; but the self-play held-out ring selects both checkpoints, which
  favours S17.
- K2 passes for both (186–195).
- S17 − E8: +0.00 to +0.03. H17 − E8: +0.02 to +0.06. E32 − the better arm: +0.02 to +0.06, excluding zero (size still
  pays above 17M rows).

## Decision rules

| L | K1 (H17 − S17) | K2 | Decision |
|---|---|---|---|
| holds | **entirely above zero** | both pass | **Human-game positions are worth more than the older self-play, position for position.** p2_ae: more Lichess positions (the size is Michael's choice); whether its held-out ring carries human positions is ruled in its PREREG before launch (RUNLOG-p2_ac §6 reading 2) |
| holds | includes zero | both pass | **No difference between the sources at this resolution**: p2_ac's gain reads as size. p2_ae: more positions from the cheaper source, Lichess, as the row above |
| holds | **entirely below zero** | both pass | **The older self-play is worth more, position for position**; the human games paid in E32 by number, if at all. p2_ae: Michael's choice between fresh self-play from E32, labelled (costed in its PREREG), and more Lichess at the lower worth measured here |
| holds | any | **either fails** | That arm's mate-in-2 cost is diagnosed first; K1 is reported |
| **fails** | | | Nothing is read until the data are explained |

E32 stays the standing network under every row: neither arm is a candidate to replace it (each holds about half
its rows).

## Measurement protocol

```bash
# Built before registration (Sign-off): the two rings, as the Contract says.
# runs/p2_ad_driver.sh as the transient user unit taz-p2ad, from main at the registration commit:
#  A. both rings hold 8,845,607 rows (subset.json)
#  B. S17, then H17 (the Contract's commands), under gpu hold
#  C. 12 reps of S17, 12 of H17, 12 of S17, 12 of H17, each
.venv/bin/python scripts/eval_reps.py run --ckpt runs/p2_ad/<arm>/size_16845607/best.pt \
    --config runs/p2_w_quarter_down/config.yaml --anchor stockfish:5 --reps 12 --out-dir runs/p2_ad/<arm>_sf5_<a|b>
#  D. each arm's suite row (eval_ladder.py --anchors random --games 20 --role diagnostic), then
.venv/bin/python scripts/eval_reps.py verdict --a s17 <S17's 24> --b h17 <H17's 24> \
    --out runs/p2_ad/p2ad_verdict_h17_vs_s17.json  # K1
#     and the reported: s17 and h17 against E8's 24; E32's 24 against s17 and h17 (exit 1 = includes zero, a reading)
```

## What would make p2_ad uninterpretable

**Selection reads self-play only.** Each checkpoint is the best by cross-entropy on a held-out ring of self-play
positions, the kind S17's fourth ring is and H17's is not; patience may stop H17 at a point that suits self-play
positions, not its own. If H17 stops much earlier or later than S17, K1 is read with that beside it. The fix (a
held-out ring with human positions) changes the selection rule and is a later round's single change.

**The sources differ in more than kind.** The older self-play comes from 37 rings of networks two to five weeks
older than D4's data, all of it; the human rows are 59% of pool 3's, a uniform draw, up to 10 positions a game from
1600+ games. "Source" means each source as this project holds it, not human play against engine play in general.

**Size is held at one point.** K1 is position for position at 16,845,607 rows. Which source wins at 32M or beyond is
not measured; E32 − each arm is reported, not ruled on.

**Resolution.** 24 reps against 24 resolve about ±0.018. The two arms' training draws are paired (same slots), their
games are not.

**Selection.** As before: the checkpoint best by held-out cross-entropy, never by skill.

## Sign-off

Checked for the draft, 2026-11-18, on this box, at `f0c598c` plus this draft's tooling (`ReplayBuffer.from_rows`,
`lichess.source_rows`, `scripts/ring_by_source.py`, with tests). **The two rings** (Contract) were built in 133 s
(self-play) and 97 s (Lichess) (`runs/p2_ad/rings_build.log`, each ring's `subset.json`): 8,845,607 rows each; the
self-play ring's slots run 5 … 23,892,952 of pool 3, the Lichess ring's 0 … 14,043,773; 8.9 GB each. **L1, checked
independently of the tooling** (`runs/p2_ad/rings_l1_check.txt`: raw memmaps read by `meta.json` alone, slots
recomputed from pool 3's tags): pool 3 holds 15,047,350 rows tagged −3 and 8,845,607 others, all tagged 1–60; each
ring's slots hash to its `subset.json`'s; **every one of the 13 fields of every row byte-equals `runs/p2_ac/teacher`'s
at its slot**; 8,845,607 distinct boards in each; the two share no slot. The start's config trains with dropout 0.0,
so the arms' only random input is the draw `--seed 4` makes, the same in both. Test suite: 1,337 passed, 6 skipped.
Disk: 45 GB free here after the rings. RunPod: not used (balance $14.16, unchanged). At registration: the round's
records (both `subset.json`, the build log, the L1 check, each arm's `result.json` and `log.jsonl`, the reps, the suite
rows, the verdicts) whitelisted in `.gitignore` and red-checked through `MUST_REACH`; the rings' arrays and the
checkpoints stay ignored. **Registered 2026-11-18** at Michael's instruction ("ok given, go"); the run is
`runs/p2_ad_driver.sh` as the transient user unit `taz-p2ad`, from `main` at this registration commit. Test suite at
registration, with the whitelist and `MUST_REACH` rows: 1,362 passed, 9 skipped.
