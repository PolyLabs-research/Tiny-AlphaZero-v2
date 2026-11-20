# Pre-registration — p2_ae: fresh self-play from the standing network, labelled and added

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-19** at Michael's instruction ("ok given, go"). Before registration nothing had run
> for it but the tooling's tests and a 16-game local smoke of the generator (a scratch directory, not data of this
> round).

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_ad: position for position, this project's older self-play trained a stronger network at skill 5 than Lichess games,
H17 − S17 **−0.0301 [−0.0467, −0.0135]**; and S17 (E8's 8M rows plus the 8,845,607 older self-play rows) was not
measurably weaker than E32 on half its rows, E32 − S17 +0.0124 [−0.0042, +0.0283] (RUNLOG-p2_ad). The older self-play
is used up: pool 3 took every distinct position of the 37 rings. Michael chose the next data (2026-11-19): **fresh
self-play from E32, the standing network**, labelled by the teacher as every row since the DIAG.

**One change: 8,845,607 fresh positions from E32's self-play, added to S17's data.** Arm **F26** is S17's 16,845,607 rows
plus the fresh 8,845,607 (25,691,214 rows). A second arm, **F17**, puts the fresh rows *in place of* the older
self-play (E8's 8M plus the fresh 8,845,607: S17's size, S17's draws at the same seed), and answers a question the next
round's design depends on: do positions from the newest network train better than positions from older ones?

**Does adding 8,845,607 positions of E32's self-play to S17's data play stronger at skill 5 than S17?** (K1)
**Position for position, does E32's self-play train better than the older self-play?** (K3)

## Contract

| | |
|---|---|
| Generator | **E32** (`runs/p2_ac/e32/size_31892957/best.pt`, sha256 `a48b0b3c…a1ba`) playing itself under the start's config (`runs/p2_w_quarter_down/config.yaml`, fingerprint `036cd665…01da`): 48 simulations a move, Gumbel root (m 16) with root noise, temperature to ply 30, each game from 0–8 random plies off the start (the 4-ply book excluded), ply cap 160, resignation at −0.9 for 8 moves with 10% of games resign-disabled. `scripts/selfplay_generate.py` (`tinyalphazero.generate`): chunk `c` is 512 games, `wave_specs(chunk_seed(5, c), 512, …)`, each row tagged with its chunk; **chunks 0–179, 92,160 games**. The rows are a function of (E32, config, seed, chunk numbers) only: resuming, splitting chunks between pods, or the number of actor processes store the same bytes (`tests/test_generate.py`, `tests/test_actors.py`) |
| Generation pods | **six secure RTX 4090 pods at once**, pod k chunks 30k … 30k+29, actor processes min(40, CPU quota − 4) (throughput only). Each pod's deadline 9 h (worst $8.01 a pod, **$48.06 for six**). Only each ring's boards, outcomes and source tags come back (about 190 MB a pod); pooling from those three fields is byte-identical to pooling from the full ring (checked on the smoke ring) |
| Pool 4 | `scripts/pool_positions.py --limit 8845607 --seed 0 --fields boards,outcomes,iteration` over the six rings, excluding D1, the held-out ring and pools 1–3 (by their labelled rings `runs/diag_cod/teacher`, `runs/p2_ab/teacher`, `runs/p2_ac/teacher`, which hold the same positions; pools 1 and 2's arrays are on the expansion drive). **8,845,607 rows**, as many as the older self-play. If the six rings offer fewer, the driver stops and more chunks (180 on, same seed) are played on one pod until they do; recorded by addendum |
| Labels | as p2_ac's (Stockfish 16, 100,000 nodes, multipv 5, temperature 50 cp, D1's outcome model), on **six secure RTX 4090 pods**, one per slice of pool 4 (five of 1,474,268 rows, the last 1,474,267), `--labels-only --row-offset --shard-prefix`. Deadline 5 h a pod (worst **$26.70 for six**). The ring built here into `runs/p2_ae/teacher`, `--outcome-model-from runs/p2_u_fresh/teacher --legals-from-boards` |
| F26 | `scripts/distill_curve.py --start runs/p2_w_quarter_down/ckpt_iter0010.pt --d1 runs/p2_u_fresh/teacher --new runs/diag_cod/teacher --new runs/p2_ab/teacher --new runs/p2_ad/ring_selfplay --new runs/p2_ae/teacher --holdout runs/p2_u_fresh/teacher_holdout --sizes 25691214 --max-samples 102764856 --seed 4 --out runs/p2_ae/f26 --device cuda`: lr 2e-5, batch 1,024, teacher rows only, held-out every 250,000 samples, patience 4, the cap at four passes; the checkpoint best by held-out cross-entropy |
| F17 | the same with `--new runs/diag_cod/teacher --new runs/p2_ab/teacher --new runs/p2_ae/teacher --sizes 16845607 --max-samples 67382428 --out runs/p2_ae/f17`: S17's ring sizes, so seed 4 draws S17's slots; only the fourth ring's rows differ |
| The reference | **S17** (`runs/p2_ad/s17/size_16845607/best.pt`), re-measured here: **24 new skill-5 reps**, interleaved with the arms'. Its 24 reps from p2_ad are a drift reading only |
| Reps | 24 skill-5 reps each of F26, F17 and S17, in blocks of 12: F26, F17, S17, F26, F17, S17 (72 reps). K1 and K3 compare 24 against 24 (about ±0.018) |
| Instrument | `eval_reps.py run --anchor stockfish:5 --config runs/p2_w_quarter_down/config.yaml`; verdicts by `eval_reps.py verdict` (per-opening rep-means, paired bootstrap over the 100 openings, 10,000 resamples, seed 0) |
| Where, time, cost | pods: generation about 3–5.5 h a pod, labelling about 2–2.5 h a pod; **about $35–45 expected, $74.76 at worst**. This box under `gpu hold`: pool 4, the ring build, F26 (about 10–16 h), F17 (about 7–11 h), 72 reps (about 10.5 h), two suite rows. **About 1.5–2 days** end to end, as the transient user unit `taz-p2ae` (`runs/p2_ae_driver.sh`) |

## Registered expectations

### L — the data are what this file says
L1: each generation ring's `state.json` records its 30 chunks and its `provenance.json` E32's sha256, the config's
fingerprint, seed 5 and its chunk range; pool 4 holds 8,845,607 rows, all distinct, none a position of D1, the
held-out ring or pools 1–3, every tag a chunk number 0–179 (checked independently of `pool_positions`). L2: every row
labelled; the ring holds 8,845,607 rows, D1's outcome model (`fitted_here` false), 100,000 nodes, multipv 5,
temperature 50, Stockfish 16, `legals_from_boards` true; F26's `result.json` records 25,691,214 rows, F17's
16,845,607. If L fails, nothing below is read until it is explained.

### C — the teacher measure (reported)
Each arm's best held-out cross-entropy, top-1 and value MSE beside S17's (2.1770, 0.3884, 0.0275), the samples at its
best, whether it ended at the cap; F17 beside S17 at matched samples (same draws). Generation: positions a game,
results, games an hour.

### K1 (primary) — F26 against S17, skill 5, 24 reps each
`p2ae_verdict_f26_vs_s17.json` (F26 − S17).

### K2 — mate-in-2
Each arm's suite row (128 sims, the start's config). **K2 passes if each arm solves at least 183 of 200.**

### K3 — F17 against S17, skill 5, 24 reps each
`p2ae_verdict_f17_vs_s17.json` (F17 − S17). Decides how the next round's data is drawn, under K1's first row.

### Reported, no decision
F26 − F17; F26 − E32 (E32's 24 reps on file from p2_ac, not interleaved); S17's new reps against its p2_ad reps (the
instrument's drift over the days between).

## Predictions (written before launch)

- L holds. Generation 3–5.5 h a pod at 100–115 positions a game; the six rings offer at least 8,845,607 distinct
  positions after the exclusions (about 10M raw).
- C: F26's held-out CE 2.165–2.178; F17's 2.170–2.185; both stop on patience before the cap.
- **K1: point +0.01 to +0.05; the interval excludes zero, about 60%.** Doubling E8's rows with self-play gave +0.075;
  half as many again gives less, but these are positions from the strongest network yet.
- K2 passes for both (188–196).
- K3: point −0.01 to +0.03; includes zero more likely than not.
- F26 − E32: +0.01 to +0.06. S17's drift within ±0.02.

## Decision rules

| L | K1 (F26 − S17) | K2 | Decision |
|---|---|---|---|
| holds | **entirely above zero** | both pass | **Fresh self-play adds strength on top of S17's data.** F26 becomes the standing network (F26 − E32 reported beside). p2_af continues the loop: self-play from F26, labelled, added (size Michael's choice). **K3 sets how:** entirely above zero → each round's self-play comes from the newest network and may replace older rows; includes zero or below → rows accumulate, older self-play kept |
| holds | includes zero | both pass | **No measurable gain from 8.85M more self-play at 25.7M rows.** E32 stays standing. Before more data, a training-side change, Michael's choice: the selection rule (patience on the self-play held-out ring, or a fixed sample budget) or the 2e-6 step-down after the protocol. K3 reported |
| holds | **entirely below zero** | any | **Adding the fresh rows costs skill.** Diagnosed (labels, overlap with older rows, the games themselves) before anything is built on it |
| holds | any | **either fails** | That arm's mate-in-2 cost is diagnosed first; K1 and K3 are reported |
| **fails** | | | Nothing is read until the data are explained |

## Measurement protocol

```bash
# runs/p2_ae_driver.sh as the transient user unit taz-p2ae, from main at the registration commit:
#  G. six generation pods in parallel (60 s apart), pod k:
python3 scripts/selfplay_generate.py --ckpt runs/p2_ac/e32/size_31892957/best.pt \
    --config runs/p2_w_quarter_down/config.yaml --seed 5 --chunk-start <30k> --chunks 30 --out runs/p2_ae/gen<k> \
    --device cuda --actors <min(40, quota-4)>
#     state pulled every 10 min; at DONE with exit 0 and 30 chunks: boards, outcomes, tags pulled, PULLED written
#  P. pool 4 (the Contract's command); fewer than 8,845,607 rows -> stop for a human
#  L. six labelling pods, as p2_ac; pull; the ring built here
#  T. F26, then F17 (the Contract's commands), under gpu hold
#  R. 12 reps each of F26, F17, S17, twice; both suite rows; then
.venv/bin/python scripts/eval_reps.py verdict --a s17 <S17's 24 new> --b f26 <F26's 24> \
    --out runs/p2_ae/p2ae_verdict_f26_vs_s17.json  # K1
#     K3 (f17 vs s17) and the reported (f26 vs f17, f26 vs e32, s17 new vs p2_ad); exit 1 = includes zero, a reading
```

## What would make p2_ae uninterpretable

**The positions come from the network they are compared against, almost.** F26 and F17 are trained from the start
checkpoint, as every arm since the DIAG; E32 only chose the positions, and the teacher labelled them. A gain is a
gain from the positions, not from E32's weights.

**Selection reads a self-play ring.** Fresh self-play is the held-out ring's kind (p2_u's self-play), so this
round's arms are judged by patience on familiar ground, unlike H17 in p2_ad. F26 may train long (cap 102.8M samples).

**F26 changes size and adds a newer source at once** (25.7M rows, the extra all fresh). K1 is "adding these rows";
K3 is the source at equal size; size alone is not separated.

**Drift.** S17 is re-measured beside the arms; E32 is not, so F26 − E32 is reported, not ruled on.

**Resolution.** 24 reps against 24 resolve about ±0.018.

**Selection.** As before: the checkpoint best by held-out cross-entropy, never by skill.

## Sign-off

**Registered 2026-11-20** at Michael's instruction ("ok given, go", 2026-11-19; the session dropped before the
registration commit, and nothing had run or been spent). The run is `runs/p2_ae_driver.sh` as the transient user unit
`taz-p2ae`, from `main` at this registration commit. L1 is checked by `runs/p2_ae_l1_check.py`, which the driver runs
after pool 4 and before any label is bought, and which stops the run on a failure: each generation ring's state and
provenance against this file; pool 4's rows, its positions distinct and its tags 0–179; no position shared with D1,
the held-out ring or pools 1–3 (keys `tokens[:70]` as `teacher.position_key`, through a 64-bit splitmix hash with every
hit confirmed byte for byte; the hash gives 4,000,000 distinct values on 4,000,000 distinct positions of pool 3 and
finds planted repeats). The round's records (generation state, provenance, chunk logs and job logs; pool 4's record and
check; the ring's record; each arm's curve; the reps; the suite rows; the verdicts) are whitelisted and red-checked
through `MUST_REACH`; the rings, the pool's arrays, the labels and the checkpoints stay ignored. Test suite at
registration 1,394 passed, 13 skipped. RunPod balance at registration $14.16. Draft checks: `tests/test_generate.py` (resume, pod split and actor count store
identical rows; parameters checked on resume); a 16-game local smoke of E32 under the start's config (2 actors): 1,762
positions, 110.1 plies a game, 1,625 simulations a second on this box, 1,757 of 1,762 kept by pooling; pooling from
the three pulled fields byte-identical to pooling from the full ring. RunPod balance at draft $14.16; Michael's auto
top-up refills at about $10, and each launch needs $10.01 (a 9 h pod's $8.01 plus the $2 margin).
