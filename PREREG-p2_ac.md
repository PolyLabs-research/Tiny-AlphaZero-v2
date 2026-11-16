# Pre-registration — p2_ac: about 32M teacher positions, the extra drawn from older self-play and Lichess games

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-16** at Michael's instruction ("ok given, go"). Before registration only the data
> preparation the Sign-off lists had run (the Lichess ring and pool 3 built and checked; no label taken, no pod
> launched).

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_ab: training on 8,000,000 teacher positions beat the 4,000,000-position protocol at skill 5, +0.023 [+0.001,
+0.045] (E8, RUNLOG-p2_ab), the first decision row: more data helps at this scale. Michael chose to go further than
the next doubling (option B, 2026-11-16): **about 32,000,000 positions**, four times E8's; in the event
**31,892,957**, because pool 3 takes every distinct position its sources hold and they hold 23,892,957, 0.4% short
of the 24,000,000 planned. The project's self-play rings hold only 8,845,607 distinct positions not yet labelled,
all from networks two to five weeks older than D4's data, so the remaining 15,047,350 come from **rated human games on Lichess** (January and February 2016), the source
DeepMind's searchless 9M model was trained on at a far larger scale.

**Two things change at once, by Michael's choice: the amount of data (8M → 31.9M) and its sources (the extra
23.9M is 37.0% older self-play and 63.0% human games).** The round answers whether 31.9M of this mix beats 8M; it does
not separate size from source (see "What would make p2_ac uninterpretable").

**Does training on 31,892,957 teacher positions play stronger at skill 5 than E8's 8,000,000?**

## Contract

| | |
|---|---|
| Start | `runs/p2_w_quarter_down/ckpt_iter0010.pt` (sha256 `43494548…3857`), as D4, R and E8 |
| Lichess positions | `scripts/lichess_positions.py --pgn-zst lichess_db_standard_rated_2016-01.pgn.zst --pgn-zst lichess_db_standard_rated_2016-02.pgn.zst --max-rows 16000000 --workers 4` (defaults: both players rated 1600+, base time 180 s or more, `Termination: Normal`, 30+ plies, up to 10 positions a game from ply 10, seed 0), into a ring on the expansion drive (`/media/michael/expansion/chess-p2ac/lichess_ring`). The two files' sha256 match Lichess's published list (2016-01 `faad7775…58d1`, 2016-02 `ea4eb457…681e`). Built 2026-11-16 at `4d13f0c` in 903 s: 7,151,004 games read (all of January, part of February), 1,723,136 kept by the filters, 1,600,000 gave positions, **16,000,000 rows** (stopped at the cap) |
| Pool 3 | `scripts/pool_positions.py --limit 24000000 --seed 0` over, in this order, the 15 local self-play rings (p2_w_wide_down … p2_t_teach, p2_s_cont), the 22 archived v4 rings in recency order (p2_s_sims … p2_e, through `runs/_ringview/`), then the Lichess ring (the list: `runs/p2_ac/pool_sources.txt`); excluding D1, the held-out ring, pool 1 and pool 2. The limit is above what the sources hold, so pool 3 is every one of their distinct positions, shuffled. With `--fields boards,outcomes,iteration`: only those fields are copied (about 1.8 GB on this box's disk, against 26 GB with every field); the legal masks are rebuilt from the boards when the ring is built (below). Built 2026-11-16 at `9728ada` in 81 s (`runs/p2_ac/pool_build.log`): **23,892,957 rows**, 8,845,607 from the 37 self-play rings (37.0%) and 15,047,350 from the Lichess ring (63.0%; of its 16,000,000 rows 851,308 were duplicates, 50,079 positions of the book, start or suites, 51,263 repeats within their game) |
| Labels | as the DIAG's and p2_ab's (Stockfish 16, 100,000 nodes, multipv 5, temperature 50 cp, D1's outcome model), on **six secure RTX 4090 pods at once**, one per slice of pool 3 (five of 4,000,000 rows, the last 3,892,957): each gets only its slice's boards, labels it with `--labels-only --row-offset <k·4M> --shard-prefix p<k>_`, and is pulled here. Each pod's deadline 10 h (worst $8.90 a pod, **$53.40 for six; about $40 expected**). The ring is built here with `--outcome-model-from runs/p2_u_fresh/teacher --legals-from-boards` (`build_teacher_ring_from_shards`: each row's legal mask written from its board, `legal_moves_packed(decode_board(...))`) into `runs/p2_ac/teacher` |
| E32 (the line) | `scripts/distill_curve.py --start <start> --d1 runs/p2_u_fresh/teacher --new runs/diag_cod/teacher --new runs/p2_ab/teacher --new runs/p2_ac/teacher --holdout runs/p2_u_fresh/teacher_holdout --sizes 31892957 --max-samples 127571828 --seed 3 --out runs/p2_ac/e32 --device cuda`: all 31,892,957 rows (D1's 475,696, pool 1's 3,524,304, pool 2's 4,000,000, pool 3's 23,892,957, read whole in that order); lr 2e-5, batch 1,024, teacher rows only, held-out every 250,000 samples, patience 4, the cap at four passes (E8 stopped on patience at about 2.5); the checkpoint best by held-out cross-entropy |
| The reference | **E8** (`runs/p2_ab/e8/size_8000000/best.pt`) with **24 skill-5 reps**: its 12 on file and 12 new. E32 gets **24 reps**. So K1 compares 24 against 24 (about ±0.016 at skill 5) |
| Instrument | `eval_reps.py run --anchor stockfish:5 --config runs/p2_w_quarter_down/config.yaml`; verdicts by `eval_reps.py verdict` (per-opening rep-means, paired bootstrap over the 100 openings, 10,000 resamples, seed 0) |
| Where, time, cost | pods: about 7.5 h of labelling, **about $40**; this box under `gpu hold`: the ring build (23.9M rows), E32's training (at ~1,680 samples a second: about 13 h if it stops near E8's 2.5 passes, 21 h at the cap), 36 reps (about 5.3 h), the suite row. **About 1.5–2 days** end to end, as the transient user unit `taz-p2ac` (`runs/p2_ac_driver.sh`) |

## Registered expectations

### L — the data are what this file says
L1: pool 3 holds 23,892,957 rows, all distinct, none a position of D1, the held-out ring, pool 1 or pool 2
(`pool.json`, with each source's rows taken; checked before registration, Sign-off). L2: every row is labelled, and the ring built here has 23,892,957 rows, D1's outcome
model (`fitted_here` false), 100,000 nodes, multipv 5, temperature 50, Stockfish 16, masks from the boards
(`legals_from_boards` true). If L fails, nothing below is
read until it is explained.

### C — the teacher measure (reported)
E32's best held-out cross-entropy beside E8's 2.2028, its top-1 and value MSE beside E8's (0.3844, 0.0319), the
samples at its best and whether it ended at the cap still improving. The held-out ring is self-play positions; the
human games, 47% of E32's rows, are not represented in it.

### K1 (primary) — E32 against E8, skill 5, 24 reps each
`p2ac_verdict_e32_vs_e8.json`.

### K2 — mate-in-2
E32's suite row (128 sims, the start's config). **K2 passes if E32 solves at least 183 of 200.**

### Reported, no decision
E32 − the 4M protocol (D4 and R, 24 reps), E32 − start (12 reps), E8's 12 new reps beside its 12 old (the
instrument's drift), each source's share of pool 3.

## Predictions (written before launch)

- L holds. Labelling 7–8.5 h on every pod.
- C: E32's held-out CE 2.19–2.21; it stops on patience before the cap.
- **K1: point +0.00 to +0.04; the interval excludes zero, about even odds.** A quadrupling at E8's rate per
  doubling (+0.023) would be about +0.046; returns are shrinking (4M→8M bought less than 0.5M→4M), and 63% of the
  new data (47% of E32's rows) is a different kind of position.
- K2 passes (E32 186–194).
- E32 − start: +0.05 to +0.10, excluding zero.

## Decision rules

| L | K1 (E32 − E8) | K2 | Decision |
|---|---|---|---|
| holds | **entirely above zero** | passes | **31.9M of this mix beats 8M.** E32 becomes the standing network. p2_ad is Michael's choice, data first: more human games (the Lichess database is effectively unlimited; labelling is the cost, about $1.6 per million positions), or the size/source separation (E8's data plus only the older self-play against E8's plus only the human games, if he wants to know which half paid) |
| holds | includes zero | passes | **No measurable gain over 8M** at ±0.016. Either the returns have flattened by 8M or the human-game positions do not transfer; the round does not say which. E8 stays standing. p2_ad: Michael's choice between separating size from source and leaving data for another lever (a step-down after the protocol; self-play rows mixed back in) |
| holds | **entirely below zero** | any | **The 31.9M mix costs skill.** Read first against the source change (the human games, 47% of E32's rows; the older self-play) before size; diagnosed before anything is built on it |
| holds | any | **fails** | The mate-in-2 cost is diagnosed first; K1 is reported |
| **fails** | | | Nothing is read until the data are explained |

## Measurement protocol

```bash
# Built before registration (Sign-off): the Lichess ring and pool 3, as the Contract says.
# runs/p2_ac_driver.sh as the transient user unit taz-p2ac, from main at the registration commit:
#  A. six pods, one per slice k of pool 3 (rows k·4M .. k·4M+3,999,999; the last to the pool's end), in parallel; each runs
python3 scripts/label_teacher.py --ring runs/p2_ac/part<k> --out runs/p2_ac/teacher --no-holdout --labels-only \
    --row-offset <k*4000000> --shard-prefix p<k>_ --stockfish /usr/games/stockfish
#  B. pull every pod's shards every 10 min; a slice complete here and DONE -> PULLED
#  C. pools 1 and 2's .dat files to the expansion drive; the ring, here:
.venv/bin/python scripts/label_teacher.py --ring runs/p2_ac/pool --out runs/p2_ac/teacher --no-holdout \
    --outcome-model-from runs/p2_u_fresh/teacher --legals-from-boards --stockfish /usr/games/stockfish --workers 2
#  D. E32 (the Contract's command), under gpu hold
#  E. 24 reps of E32, 12 more of E8, E32's suite row, then
.venv/bin/python scripts/eval_reps.py verdict --a e8 <E8's 24> --b e32 <E32's 24> --out runs/p2_ac/p2ac_verdict_e32_vs_e8.json  # K1
#     and the reported e32 vs the 4M protocol, e32 vs start (exit 1 = the interval includes zero, a reading)
```

## What would make p2_ac uninterpretable

**Size and source change together** (Michael's choice, B): the extra 23.9M is older self-play and human games. A
result for E32 is a result for this mix at this size. Separating them is a later round if Michael wants it.

**The held-out ring has no human-game positions.** C may move less than skill does; K1 is the measure.

**The positions' ages.** The older self-play (28% of E32's rows) comes from networks two to five weeks before D4's
data.

**Resolution.** 24 reps against 24 resolve about ±0.016.

**Selection.** As before: the checkpoint best by held-out cross-entropy, never by skill.

## Sign-off

Checked for the draft, 2026-11-16, on this box. **The Lichess ring** (Contract) was built at `4d13f0c`; both files'
sha256 match Lichess's published list. **Pool 3** was built at `9728ada` with only boards, outcomes and source tags
copied (`pool_positions.py --fields`; the first build, copying every field onto the expansion drive, was too slow
and was stopped and deleted before it wrote a row). **L1, checked independently of `pool_positions`**
(`runs/p2_ac/pool_l1_check.txt`): 23,892,957 rows, 23,892,957 distinct positions; none shared with D1 (475,696),
the held-out ring (5,000), pool 1 (3,524,304) or pool 2 (4,000,000); the source tags give 15,047,350 Lichess rows
and 8,845,607 self-play rows, as `pool.json`. **The rebuilt masks**: `legal_moves_packed(decode_board(board))`
equals the stored mask on 190,000 rows, 5,000 at random from each of the 38 sources, 0 mismatches
(`runs/p2_ac/legals_check.txt`). The test suite holds the ring built from a maskless pool byte-identical to the
one built from the full pool, positions with en passant, castling rights and promotions included (1,305 passed).
Disk: 60 GB free here; the ring takes about 26 GB. RunPod balance before registration $42.57 (expected spend
about $40, worst $53.40; Michael's auto-reload tops up at about $10). The run is `runs/p2_ac_driver.sh` as the
transient user unit `taz-p2ac` (each phase skipped once its output exists; a pod found gone, three tries 20 s
apart, with its slice incomplete is left for a human; verdict steps read `eval_reps verdict`'s exit 1 as an
interval that includes zero). The round's records (pool 3's `pool.json`, `pool_sources.txt`, the build log and
the two checks, a copy of the Lichess ring's `lichess.json`, the ring's `teacher.json` and `meta.json`, E32's
`result.json` and `log.jsonl`, the reps, the suite row, the verdicts) are reachable and red-checked through
`MUST_REACH`; the pool's arrays, the labels, the pod staging copies and E32's checkpoint stay ignored. Michael
accepted 31,892,957 for the planned 32,000,000 ("close enough"). **Registered 2026-11-16** at his instruction ("ok
given, go"); the balance at launch is recorded in the driver's log.

## Addendum A — 2026-11-16, before launch: RunPod's REST v1 is gone

At registration the pod tooling still called RunPod's REST v1, which RunPod retired on 2026-11-15 (its host no
longer resolves). The tooling was moved to REST v2 at `b1031db` (create body, SSH endpoint, lists, the pods'
self-delete; `cloud/README.md`), changing nothing this file measures: the labels, the ring, E32 and every reading
are as registered. The run is from `main` at this addendum's commit, not the registration commit. Before the six
pods, one short pod of the same spec goes through the v2 lifecycle (create, SSH, staging, self-delete after
PULLED); the run starts only if it passes. At the time of writing RunPod lists no secure RTX 4090 capacity (an
emergency maintenance across regions), so the launch waits for capacity (`runs/p2_ac_when_ready.sh`, as the
transient user unit `taz-p2ac-wait`, checking every 10 minutes); the driver retries each pod's launch for up
to 6 hours.
