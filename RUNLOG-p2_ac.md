# RUNLOG — p2_ac (about 32M teacher positions: E8's 8M plus older self-play and Lichess games)

**Run:** `runs/p2_ac`. Pool 3, every distinct position the 38 sources hold that is not in D1, the held-out ring, pool 1
or pool 2: **23,892,957 rows**, 8,845,607 from 37 older self-play rings and 15,047,350 from 1,600,000 rated Lichess
games of January and February 2016 (`scripts/lichess_positions.py`, `scripts/pool_positions.py --fields
boards,outcomes,iteration`). Labelled by Stockfish 16 at 100,000 nodes, multipv 5, on six secure RTX 4090 pods, one
slice each (`--labels-only --row-offset`); the ring built here with D1's outcome model and the legal masks rebuilt
from the boards (`--legals-from-boards`). Line **E32**: `scripts/distill_curve.py` on D1 + pool 1 + pool 2 + pool 3
(31,892,957 rows), teacher rows only, from `p2_w_quarter_down/0010` (sha256 `43494548…3857`), lr 2e-5, batch 1,024,
held-out every 250,000 samples, patience 4, cap 127,571,828 (four passes), seed 3; the checkpoint best by held-out
cross-entropy. Then 24 skill-5 reps of E32, 12 more of E8, E32's mate-in-2 suite row, three verdicts. Data built
2026-11-16; run 2026-11-17 00:57 to 2026-11-18 01:38 UTC, the pods on RunPod and everything else on this box under
`gpu hold`, driven by `runs/p2_ac_driver.sh` as the transient user unit `taz-p2ac` (twice: §7). **Pre-registration:**
`PREREG-p2_ac.md` (registered `2406411`, Addendum A `2cc25d5`). Every verdict below was registered before it was
measured, except where marked.

**Verdict: the registered first row, "31.9M of this mix beats 8M". K1: E32 beats E8 at skill 5, +0.0875 [+0.0694,
+0.1055] (E32 0.5280 over 24 reps against E8's 0.4405 over 24), the largest gain between two rounds this project has
measured, more than twice the top of the predicted range (+0.00 to +0.04). K2 passes: E32 solves 194 of 200 mate-in-2
positions (floor 183; E8 192). L holds. C, reported: E32's best held-out cross-entropy 2.1892 (E8 2.2028), reached at
24.75M samples, less than one pass over its rows; it stopped on patience at 25.75M. Reported: E32 − the 4M protocol
+0.1067 [+0.0892, +0.1236]; E32 − start +0.1470 [+0.1252, +0.1692]. **E32 is the first network to score above 0.5
against skill 5, and becomes the standing network.** Size and source changed together (Michael's choice), so the
round does not say which paid. Pods cost about $28.41, under the $40 expected. RunPod retired the API the tooling used
the day before launch (Addendum A), and the ring build was killed once in a machine-wide out-of-memory squeeze (§7);
neither touched a measurement.**

---

## 1. Before launch

Michael chose option B after p2_ab (2026-11-16): about 32M teacher positions, the extra ~24M from the older self-play
rings and, because those hold only about 8.85M unlabelled positions, from human games on Lichess. The tooling for it
is `4d13f0c` (Lichess positions; labelling a slice of one pool per pod under the pool's row numbers; a compact ring
build that scales to tens of millions of rows) and `9728ada` (a pool of boards, outcomes and source tags only, the
legal masks rebuilt at the ring build; a first pool build copying every field onto the expansion drive was too slow,
was stopped by the 2 h limit on background shell jobs, and was deleted before it wrote a row). The Lichess files'
sha256 match Lichess's published list. Pool 3 held 23,892,957 positions, 0.4% short of the planned 24,000,000,
because that is every distinct one the sources hold; Michael accepted it ("close enough") at registration, and E32's
size and cap follow (31,892,957 rows, 127,571,828 samples). The draft is `c99b85c`; the registration `2406411`
(2026-11-16, Michael: "ok given, go").

**Addendum A (`2cc25d5`, before launch).** RunPod retired its REST API v1 on 2026-11-15; `rest.runpod.io` stopped
resolving, so launch, status, terminate, the sweeper and both of the pods' self-delete paths were dead (the sweeper had
been failing since 17:09 UTC on 11-16; no pod was running). The tooling moved to v2 at `b1031db`: the nested create
body, validated against RunPod's OpenAPI schema, SSH from `ssh.direct`, paginated lists, `DELETE /v2/pods/<id>` from
the pod's watchdog and `pod_job.sh`. The addendum records that the run is from that tooling, not the registration
commit, and that the six pods wait for a smoke pod of the same spec to pass the v2 lifecycle. `git diff 2406411
fa3ce5b -- src scripts cloud` touches `cloud/`, `scripts/runpod_pod.py` and, at `fa3ce5b`, `scripts/label_teacher.py`
(it frees a set before the ring build, §7; output unchanged, its byte-identity test passes). Nothing the round measures
changed.

## 2. L — the data

**L1 holds** (checked before registration, `runs/p2_ac/pool_l1_check.txt`, independently of `pool_positions`): 23,892,957
rows, 23,892,957 distinct positions; none shared with D1 (475,696), the held-out ring (5,000), pool 1 (3,524,304) or
pool 2 (4,000,000); the source tags give 15,047,350 Lichess rows (63.0%) and 8,845,607 self-play rows (37.0%), as
`pool.json`. Of the Lichess ring's 16,000,000 rows, 851,308 were duplicates of earlier positions, 50,079 positions of
the book, start or suites, and 51,263 repeats within their game. The rebuilt masks equal the stored ones on 190,000
rows, 5,000 from each of the 38 sources (`runs/p2_ac/legals_check.txt`).

**L2 holds** (`runs/p2_ac/teacher/teacher.json`): 23,892,957 rows; D1's outcome model (a 303.35, s 367.36, `fitted_here`
false); 100,000 nodes, multipv 5, temperature 50 cp, Stockfish 16; `legals_from_boards` true; every row labelled (the
build's scan labelled nothing). The draw rate on its own rows is 0.145 (pool 2's ring 0.296: human games end in draws
far less often than self-play). A spot check of 3,000 random ring rows: boards equal pool 3's, masks equal
`legal_moves_packed(decode_board(...))`, every policy move legal, every row tagged as a teacher row, 4.87 policy entries
a row on average.

## 3. C — the teacher measure (reported)

E32 ran 104 measurements, from 0 to 25,750,528 samples. **Its best: held-out cross-entropy 2.1892 at 24,750,080
samples** (top-1 0.3912, value MSE against the teacher 0.0280, training-probe cross-entropy 2.1934); the four after it
did not beat it, and **patience stopped the run at 25,750,528 samples**, far short of the 127.6M cap. That is 0.78
passes over 31,892,957 rows; draws are uniform with replacement (`distill.sample_union`), so about 54% of the rows had
been drawn at least once when it stopped (E8 at its best: 2.34 passes, about 90%). The start measured 2.3673, top-1
0.3658, as every run of this script.

E32 beside E8 at matched samples (unregistered):

| samples | E8 held-out | E32 held-out |
|---|---|---|
| 2M | 2.2376 | 2.2400 |
| 4M | 2.2308 | 2.2319 |
| 8M | 2.2201 | 2.2187 |
| 12M | 2.2145 | 2.2110 |
| 16M | 2.2078 | 2.2027 |
| 18.75M (E8's best) | 2.2028 | 2.1974 |
| 24.75M (E32's best) | — | 2.1892 |

The held-out ring is self-play positions; the human games, 47% of E32's rows, are not represented in it.

## 4. K — skill 5

The instrument as registered: `eval_reps.py run --anchor stockfish:5 --config runs/p2_w_quarter_down/config.yaml`, the
100-opening book, 200 games a rep, 48 simulations a move; verdicts by per-opening rep-means with a paired bootstrap
over the 100 openings (10,000 resamples, seed 0).

### K1 (primary) — E32 against E8, 24 reps each
**+0.0875 [+0.0694, +0.1055]**, se 0.0093 (`p2ac_verdict_e32_vs_e8.json`): E32 0.5280 against E8's 0.4405 (its 12
reps from p2_ab and 12 new). **Entirely above zero**, by 0.069 at the lower end.

### K2 — mate-in-2
**194 of 200** (`e32_ladder.jsonl`, the suite row at 128 sims under the start's config). **K2 passes** (floor 183; E8
192, the start 188). E32 solves `mate2_125` and `mate2_128`, which E8 fails, and fails none E8 solves; its six failures
are all E8's. Mate-in-3: 171 (E8 167, the start 164). The ladder's console prints "REGRESSED" against the phase 1
baseline of 0.985; the registered floor is 183.

### Reported, no decision
- **E32 − the 4M protocol: +0.1067 [+0.0892, +0.1236]** (`p2ac_verdict_e32_vs_4m.json`; D4's and R's 24 reps, 0.4214).
- **E32 − start: +0.1470 [+0.1252, +0.1692]** (`p2ac_verdict_e32_vs_start.json`; the start's 12 reps, 0.3810).
- **The instrument's drift:** E8's 12 new reps average 0.4365 (0.3775–0.4875), its 12 from p2_ab 0.4446 (0.3925–0.4875):
  −0.008, within the rep-to-rep spread.
- E32's 24 reps run 0.4775–0.5725. Each source's share of pool 3: §2.

## 5. The predictions

| prediction (written before launch) | result | |
|---|---|---|
| L holds | holds | held |
| Labelling 7–8.5 h on every pod | 4.4–6.2 h a pod (faster hosts than p2_ab's: 180–250 rows a second or more) | **missed** (faster) |
| C: E32's held-out CE 2.19–2.21 | 2.1892 | **missed**, by 0.0008 |
| C: stops on patience before the cap | at 25.75M samples | held |
| K1 point +0.00 to +0.04 | +0.0875 | **missed** (far higher) |
| K1's interval excludes zero, about even odds | [+0.0694, +0.1055] | held |
| K2 passes (E32 186–194) | 194 | held |
| E32 − start +0.05 to +0.10, excluding zero | +0.1470 [+0.1252, +0.1692] | **missed** (higher); excludes zero |

## 6. Decision

Per the table's first row (L holds; K1 entirely above zero; K2 passes): **31.9M of this mix beats 8M. E32 becomes the
standing network. p2_ad is Michael's choice, data first:** more human games (the Lichess database is effectively
unlimited; labelling cost about $1.19 per million positions this round), or the size/source separation (E8's data
plus only the older self-play against E8's plus only the human games), if he wants to know which half paid.

Readings the table does not cover (all unregistered):

1. **The gain is large and the round cannot say why.** +0.0875 is 3.8 times p2_ab's +0.0232 for the step before.
   Size (4×) and source (63% of the new rows from human games, 37% from networks two to five weeks older) changed
   together. Quadrupling at p2_ab's per-doubling rate predicted about +0.046; the rest may be the human games, or
   returns that had not flattened. The separation in the decision row would answer it.
2. **E32 stopped having drawn only about half its rows.** Patience is read on a self-play held-out ring. E32's
   held-out cross-entropy fell below E8's from 8M samples on and flattened near 2.189 by 24M, while 47% of its rows are
   of a kind that ring cannot see. Training past that point, or a held-out ring that includes human positions, might
   find more; this round's selection rule, as registered, stopped it. A question for a later round, not a result.
3. **On the rung's own scale** (unregistered): E8's 0.4405 and E32's 0.5280 against `stockfish:5` are about −42 and
   +19 Elo by the logistic relation, about +61. E32 is the first network here to score above 0.5 against skill 5. This
   is not a CCRL measurement; the gauntlets of p2_x and p2_y measured the start.
4. **The value head improved most.** E32's value MSE against the teacher at its best (0.0280) is the lowest yet
   (E8 0.0319), as top-1 (0.3912 against 0.3844).

The open list, carried forward: p2_ad (data, Michael's choice above); training past the self-play patience point, or
a held-out ring with human positions (reading 2); a gauntlet of the standing network (p2_z, the benchmark track's
clock-scaled gauntlet; the start's CCRL readings no longer describe it); the 2e-6 step-down after the protocol;
self-play rows mixed back in; from p2_y and p2_w: the runner logging an engine's exception, a deeper engine, per-source
loss weights, the calibration sample, the promotion loss at 2e-6, L2's exclusion at storage time.

## 7. Mechanics

- **Waiting for capacity.** RunPod listed no secure RTX 4090 capacity through the evening of 11-16 (an emergency
  maintenance across regions on its status page). A watcher (`runs/p2_ac_when_ready.sh`, unit `taz-p2ac-wait`, every
  10 minutes from 21:48 UTC) tried a smoke pod whenever the catalog read anything but NONE; most tries were refused
  for capacity. The first smoke pod that launched (22:48, `zymoufrr3y4vm4`) was created, listed and terminated
  correctly, but the smoke script's own check failed: `status | grep -q` under `pipefail` with unbuffered output reads
  a listed pod as missing (grep exits at the match; status's next print hits the closed pipe). The driver's `alive()`
  had the same pattern and would have read every live pod as gone; both now capture the output first. The second smoke
  pod (`s5a41eh2fezdul`, 00:03–00:09 on 11-17, 64 vCPU) passed the whole v2 lifecycle: create, SSH, staging, the pod's
  key on v2 (200), the watchdog, self-delete about 10 s after `PULLED`. The watcher then failed to start the run (the
  driver lacked its executable bit); it was started by hand at 00:57:14, armed at `2cc25d5`.
- **The pods.** Capacity came one pod at a time; each relaunch try waited 5 minutes (up to 72 tries):

  | slice | pod | launched | vCPU | job started | DONE seen, PULLED | rows a second, at least |
  |---|---|---|---|---|---|---|
  | 0 | `0dci9c0es08kmm` | 00:57:16 | 51 | 00:59:21 | 05:36:45 | 240 |
  | 2 | `ibgkwacp5erbnk` | 02:33:20 | 51 | 02:34:19 | 07:00:44 | 250 |
  | 4 | `0pax28j71eugk7` | 02:28:59 | 64 | 02:36:42 | 07:42:33 | 218 |
  | 3 | `2qncmk98j9cnh6` | 03:49:01 | 64 | 03:52:34 | 09:36:12 | 194 |
  | 5 | `nn5xllcw8seqnv` | 04:19:55 | 64 | 04:23:06 | 09:36:20 | 207 |
  | 1 | `qgg9011r36zppa` | 04:13:33 | 64 | 04:16:32 | 10:26:54 | 180 |

  (DONE is seen at a pull, every 10 minutes, so the rates are lower bounds.)

  Every slice's boards matched this box's sha256 on the pod; every slice came back complete (4,000,000 rows each,
  slice 5 3,892,957); every pod deleted itself after `PULLED`, none by its deadline or the sweeper.
- **The ring build, twice.** The first build (10:37 UTC) was killed at 11:05:22 by the kernel's out-of-memory killer
  in a machine-wide squeeze: RAM and all 8 GB of swap full, with Firefox about 6 GB, Experiment_Two's tests and
  experiment processes about 8 GB and gnome-system-monitor 2.7 GB resident; the build, at 4.4 GB, was the largest
  single process. `fa3ce5b` frees the labelled-rows set (about 1.7 GB at this size) before the build. The disk was
  then short of the ring's 26 GB (21 GB free: Experiment_Two's test suite keeps up to three pytest temp directories
  of 20–29 GB each in `/tmp`). Chess's own data made the room, as the 22 older rings were archived earlier: the 17
  local replay rings (pool sources, not needed by E32) and 270 intermediate checkpoints of those runs moved to
  `/media/michael/expansion/tinyalphazero-v2-runs/runs/<run>/`, each file compared byte for byte before removal, the
  rings readable through `runs/_ringview/<run>`; E32's start and each run's last checkpoint stay here. The driver was
  restarted at 12:31:18 (armed at `fa3ce5b`; phases A and B skipped, every slice pulled) with 58 GB free; the ring
  built 12:31–13:24.
- **E32, sharing the GPU.** E32 trained 13:23:58–20:21:23. Until about 16:00 two Experiment_Two experiment loops
  (`explore_weak_start.py`, 17 GB of VRAM each, started 12:51) shared the GPU, and E32 ran at
  about 400–476 samples a second; alone it ran about 1,336 (E8 about 1,680, on a ring a quarter the size). Speed does
  not enter any reading.
- **Reps and verdicts.** E32's 24 reps 20:21–23:50; E8's 12 23:50–01:37; the suite row and three verdicts by 01:38:23
  on 11-18 ("p2_ac complete"). Each step took `gpu hold --by tinyalphazero` with
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTHONUNBUFFERED=1`. No commit went to `main` while a unit ran.
- **Cost:** the balance read $42.57 before the smoke pods and $14.16 after the last pod: **$28.41** for both smoke
  pods and the six labelling pods (about 31.6 pod-hours at $0.89), against about $40 expected and $53.40 worst.
- **The box** stayed up throughout.

## 8. Artefacts

| file | what |
|---|---|
| `runs/p2_ac/lichess.json` | the Lichess ring's sources (sha256), filters and counts (a copy; the ring is on the expansion drive) |
| `runs/p2_ac/pool/pool.json`, `pool/meta.json`, `pool_sources.txt`, `pool_build.log` | pool 3's sources, counts, fields and build (L1) |
| `runs/p2_ac/pool_l1_check.txt`, `legals_check.txt` | L1 checked independently; the rebuilt masks against the stored ones |
| `runs/p2_ac/teacher/teacher.json`, `teacher/meta.json` | the ring's settings, outcome model and provenance (L2) |
| `runs/p2_ac/e32/size_31892957/result.json`, `log.jsonl` | E32's best measurement and all 104 (C; §3's table) |
| `runs/p2_ac/e32_sf5/reps_best_r{1..24}.jsonl`, `e8_sf5_more/reps_best_r{1..12}.jsonl` | the reps (K1) |
| `runs/p2_ac/e32_ladder.jsonl` | E32's suite row (K2) and the 20-game random match that carried it |
| `runs/p2_ac/p2ac_verdict_{e32_vs_e8,e32_vs_4m,e32_vs_start}.json` | K1 and the reported comparisons |
| `runs/p2_ac/launch_watch.log` | the capacity watch and both smoke pods (§7) |
| `runs/p2_ac/pool/*.dat`, `teacher/*.dat`, `teacher_labels/`, `e32/size_31892957/best.pt`, `ce_rows.npy` | the pool, the ring, the label shards, E32's checkpoint and per-row CE (ignored; regenerable or large) |
| `runs/p2_ac_driver.sh`, `p2_ac_smoke.sh`, `p2_ac_when_ready.sh`, `p2_ac_pool.sh`, `runs/p2_ac_driver.console.log` | the units' scripts and console (untracked) |
