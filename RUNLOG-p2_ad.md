# RUNLOG — p2_ad (size held at 16,845,607 rows, source split: older self-play against Lichess)

**Run:** `runs/p2_ad`. Two rings cut from p2_ac's labelled pool 3 (`scripts/ring_by_source.py`, each row's source read
from pool 3's tags after all 23,892,957 boards were compared equal): `ring_selfplay`, all 8,845,607 older self-play
rows; `ring_lichess`, the first 8,845,607 Lichess rows in pool order. Arms **S17** and **H17**: `scripts/distill_curve.py`
on D1 + pool 1 + pool 2 (E8's 8,000,000 rows) + one of the two rings, 16,845,607 rows each, teacher rows only, from
`p2_w_quarter_down/0010` (sha256 `43494548…3857`), lr 2e-5, batch 1,024, held-out every 250,000 samples, patience 4,
cap 67,382,428 (four passes), **seed 4 in both** (the same draws of every ring's slots; dropout 0.0). Then 24 skill-5
reps per arm in blocks of 12 (S17, H17, S17, H17), each arm's mate-in-2 suite row, five verdicts. Rings built and
checked 2026-11-18 before registration; run 2026-11-18 17:07:49 to 2026-11-19 10:33:30 UTC, all on this box under
`gpu hold`, driven by `runs/p2_ad_driver.sh` as the transient user unit `taz-p2ad`. **Pre-registration:**
`PREREG-p2_ad.md` (draft `98280c9`, registered `d6fb9bd`). Every verdict below was registered before it was measured,
except where marked.

**Verdict: the registered third row, "the older self-play is worth more, position for position". K1: H17 − S17 at
skill 5, −0.0301 [−0.0467, −0.0135] (H17 0.4855, S17 0.5156, 24 reps each): the whole interval below zero, the
opposite sign to the prediction (+0.00 to +0.05). K2 passes: S17 solves 192 of 200 mate-in-2 positions, H17 195
(floor 183). L holds. Reported: S17 − E8 +0.0751 [+0.0580, +0.0924]; H17 − E8 +0.0450 [+0.0294, +0.0604]; **E32 − S17
+0.0124 [−0.0042, +0.0283], including zero: S17, with half of E32's rows and no human games, is not measurably weaker
than E32**; E32 − H17 +0.0425 [+0.0229, +0.0616]. The caveat the PREREG named first is large: the self-play held-out ring
stopped H17 at 20.75M samples and S17 at 45.0M (§3). E32 stays the standing network. p2_ae is Michael's choice
between fresh self-play from E32, labelled, and more Lichess at the lower worth measured here. Cost $0.**

---

## 1. Before launch

Michael chose the size/source separation after p2_ac ("proceed as suggested", 2026-11-18; his auto top-up covers RunPod,
though this round used none). The design holds size equal, unlike the first sketch (E8 plus all 15,047,350 human rows
against E8 plus the 8,845,607 self-play rows), which would have confounded size again. The tooling is `ea4c4ca`:
`ReplayBuffer.from_rows` (a new ring of chosen slots, every field, the head written last), `lichess.source_rows`
(Lichess −3 or self-play ≥ 0 slots in ring order; any other tag refused; `LICHESS_ITERATION` moved into the library),
`scripts/ring_by_source.py`, and tests including `test_sample_union_draws_the_same_slots_from_parts_of_the_same_sizes`,
which pins the pairing. The labelled ring's `iteration` field holds the teacher tag (−2) on every row, so the source
had to come from pool 3, whose rows the ring holds in the same order (all boards compared). The draft is `98280c9`;
the registration `d6fb9bd` (Michael: "ok given, go"); suite at registration 1,362 passed, 9 skipped.

## 2. L — the data

**L1 holds** (before registration, `runs/p2_ad/rings_l1_check.txt`, independently of the tooling: raw memmaps read by
`meta.json` alone, slots recomputed from pool 3's tags). Pool 3: 15,047,350 rows tagged −3, 8,845,607 others, all
tagged 1–60. `ring_selfplay`: 8,845,607 rows, slots 5 … 23,892,952; `ring_lichess`: 8,845,607 rows, slots 0 …
14,043,773; each ring's slots hash to its `subset.json`'s; every one of the 13 fields of every row byte-equals
`runs/p2_ac/teacher`'s at its slot; 8,845,607 distinct boards in each; the two share no slot. As rows of pool 3 they
share no position with D1, the held-out ring, pool 1 or pool 2 (p2_ac's L1). Built in 133 s and 97 s
(`rings_build.log`), 8.9 GB each.

**L2 holds:** both `result.json` record 16,845,607 rows.

## 3. C — the teacher measure (reported)

| | best held-out CE | samples at best | stopped at | passes at stop | top-1 | value MSE |
|---|---|---|---|---|---|---|
| **S17** | **2.1770** | 44.0M | 45.0M (patience) | 2.67 | 0.3884 | 0.0275 |
| **H17** | 2.2006 | 19.75M | 20.75M (patience) | 1.23 | 0.3846 | 0.0288 |
| E8 | 2.2028 | 18.75M | 19.75M | 2.47 | 0.3844 | 0.0319 |
| E32 | 2.1892 | 24.75M | 25.75M | 0.81 | 0.3912 | 0.0280 |

Neither arm reached the cap (S17 181 measurements, H17 84). S17's 2.1770 is the lowest held-out cross-entropy this
project has recorded, and its value MSE the lowest. Training-probe CE at the best: S17 2.1721, H17 2.1951. At matched
samples (unregistered):

| samples | S17 | H17 | E8 | E32 |
|---|---|---|---|---|
| 4M | 2.2304 | 2.2323 | 2.2308 | 2.2319 |
| 8M | 2.2190 | 2.2220 | 2.2201 | 2.2187 |
| 12M | 2.2122 | 2.2139 | 2.2145 | 2.2110 |
| 16M | 2.2050 | 2.2063 | 2.2078 | 2.2027 |
| 20M | 2.2012 | 2.2014 | — | 2.1957 |
| 24M | 2.1962 | — | — | 2.1911 |
| 32M | 2.1872 | — | — | — |
| 40M | 2.1807 | — | — | — |
| 44M (S17's best) | 2.1770 | — | — | — |

H17 sat 0.001–0.003 above S17 on the self-play held-out ring to 16M and **level with it at 20M (2.2014 against
2.2012)**; then it set no new best (2.2010, 2.2006, 2.2014, 2.2014, 2.2012, 2.2008 over its last six measurements) and
patience stopped it, while S17, whose fourth ring is the held-out ring's kind, went on falling to 45.0M. So the arms'
training lengths differ because of their source, through the selection rule.

## 4. K — skill 5

The instrument as registered: `eval_reps.py run --anchor stockfish:5 --config runs/p2_w_quarter_down/config.yaml`, the
100-opening book, 200 games a rep, 48 simulations a move; verdicts by per-opening rep-means with a paired bootstrap
over the 100 openings (10,000 resamples, seed 0).

### K1 (primary) — H17 against S17, 24 reps each
**−0.0301 [−0.0467, −0.0135]**, se 0.0084 (`p2ad_verdict_h17_vs_s17.json`): H17 0.4855 against S17 0.5156. **Entirely
below zero**, by 0.0135 at the upper end. Re-run to a scratch file, byte-identical. All 24 rep files of each arm are
on that arm's `best.pt` against `stockfish_s5`, 4,800 games an arm. S17's reps run 0.445–0.5975, H17's 0.4175–0.535.

### K2 — mate-in-2
**S17 192 of 200, H17 195** (`s17_ladder.jsonl`, `h17_ladder.jsonl`, 128 sims, the start's config): **K2 passes** for
both (floor 183; E32 194, E8 192). Failures: S17 `mate2_124 125 126 127 128 129 138 196`; H17 `mate2_091 124 126 129
196` (E32: `091 124 126 127 129 196`). Mate-in-3: S17 169, H17 170 (E32 171, E8 167). The ladder's console prints
"REGRESSED" against the phase 1 baseline of 0.985; the registered floor is 183.

### Reported, no decision

| comparison | difference | interval | |
|---|---|---|---|
| S17 − E8 | +0.0751 | [+0.0580, +0.0924] | above zero |
| H17 − E8 | +0.0450 | [+0.0294, +0.0604] | above zero |
| E32 − S17 | +0.0124 | [−0.0042, +0.0283] | **includes zero** |
| E32 − H17 | +0.0425 | [+0.0229, +0.0616] | above zero |

E8's 24 reps (0.4405) and E32's 24 (0.5280) are those on file from p2_ab and p2_ac. **The blocks** (unregistered):
S17 0.5106 then 0.5206, H17 0.4771 then 0.4940; both rose by about 0.01–0.02 from the first block to the second, and
alternating put the drift on both arms.

## 5. The predictions

| prediction (written before launch) | result | |
|---|---|---|
| L holds | holds | held |
| C: S17's held-out CE 2.192–2.200 | 2.1770 | **missed** (lower) |
| C: H17's held-out CE 2.195–2.205 | 2.2006 | held |
| C: both stop on patience before the cap | 45.0M and 20.75M of 67.4M | held |
| K1 point +0.00 to +0.05 | −0.0301 | **missed** (opposite sign) |
| K1's interval excludes zero, about even odds | [−0.0467, −0.0135] | held (below zero, not above) |
| K2 passes for both (186–195) | 192, 195 | held |
| S17 − E8 +0.00 to +0.03 | +0.0751 | **missed** (far higher) |
| H17 − E8 +0.02 to +0.06 | +0.0450 | held |
| E32 − the better arm +0.02 to +0.06, excluding zero | E32 − S17 +0.0124, includes zero | **missed** |

## 6. Decision

Per the table's third row (L holds; K1 entirely below zero; K2 passes for both): **the older self-play is worth more,
position for position; the human games paid in E32 by number, if at all. E32 stays the standing network. p2_ae is
Michael's choice between fresh self-play from E32, labelled (costed in its PREREG), and more Lichess at the lower worth
measured here.**

Readings the table does not cover (all unregistered):

1. **The selection rule carries part of the result, and the round cannot say how much.** H17 trained 1.23 passes, S17
   2.67, because patience is read on a self-play held-out ring; at 20M samples the two fitted that ring equally. K1 is
   the worth of each source *under this project's protocol*, which is how any next round would train. Whether human
   positions trained longer, or selected on a held-out ring that includes them, would close the gap is a separate
   question: a change to the selection rule, one round's single change.
2. **p2_ac's gain was nearly all the self-play.** S17 − E8 (+0.075) is 86% of E32 − E8 (+0.0875), and E32 − S17 includes
   zero: on top of the self-play, 15M human rows added nothing measurable. E32 also stopped earlier than S17 (25.75M
   against 45.0M samples), under the same rule.
3. **Data still pays at fixed size, and so does training longer on it.** S17 doubled E8's rows and gained +0.075; its
   held-out CE was still falling at 28M samples and bottomed at 44M. The network shows no capacity limit at 16.8M rows.
4. **Why self-play might be worth more** (a hypothesis, not tested): the rung's games start from the book and run
   between engines; self-play positions are engine-play positions, human games from 1600+ players are not.

The open list, carried forward: p2_ae (data, Michael's choice above); a held-out ring with human positions, or a fixed
sample budget, before human data is judged again; a gauntlet of the standing network (p2_z) or a Lichess bot account
for a human-comparable number (lichess-bot setup awaiting Michael: the bot account, its token, his OK to install the
client); the 2e-6 step-down after the protocol; self-play rows mixed back in; per-source loss weights; the calibration
sample; the runner logging an engine's exception.

## 7. Mechanics

- **Schedule.** S17 trained 17:07:49–00:15:09 (about 1,755 samples a second); H17 00:15:09–03:31:22 (about 1,760); reps
  in blocks of 12, each about 1 h 44 min (S17 from 03:31:22, H17 05:15:51, S17 06:59:40, H17 08:44:48); the suite rows
  10:30:05 and 10:31:48; the verdicts and "p2_ad complete" at 10:33:30 on 11-19. Each step took `gpu hold --by
  tinyalphazero` with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTHONUNBUFFERED=1`; the GPU was not shared.
- **The timeline moved.** The PREREG's estimate (about 15 h) assumed both arms would stop near E8's and E32's
  19.75M–25.75M samples; S17 ran to 45.0M, and the run took 17.4 h, inside the 30 h worst case.
- **No commit went to `main` while the unit ran** (registration at `d6fb9bd` before it started).
- **Cost:** $0 (no pod; the RunPod balance stayed $14.16).
- **The box** stayed up throughout.

## 8. Artefacts

| file | what |
|---|---|
| `runs/p2_ad/ring_{selfplay,lichess}/subset.json`, `meta.json`, `teacher.json` | each ring's selection, layout and label provenance (L1) |
| `runs/p2_ad/rings_build.log`, `rings_l1_check.txt` | the rings' build and L1 checked independently |
| `runs/p2_ad/{s17,h17}/size_16845607/result.json`, `log.jsonl` | each arm's best measurement and every measurement (C) |
| `runs/p2_ad/{s17,h17}_sf5_{a,b}/reps_best_r{1..12}.jsonl` | the reps, in blocks (K1) |
| `runs/p2_ad/{s17,h17}_ladder.jsonl` | each arm's suite row (K2) and the 20-game random match that carried it |
| `runs/p2_ad/p2ad_verdict_{h17_vs_s17,s17_vs_e8,h17_vs_e8,e32_vs_s17,e32_vs_h17}.json` | K1 and the reported comparisons |
| `runs/p2_ad/ring_*/*.dat`, `{s17,h17}/size_16845607/best.pt`, `ce_rows.npy`, `summary.json` | the rings, the checkpoints, per-row CE (ignored; regenerable or large) |
| `runs/p2_ad_driver.sh`, `runs/p2_ad_driver.console.log` | the unit's script and console (untracked) |
