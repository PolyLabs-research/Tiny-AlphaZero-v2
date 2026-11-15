# RUNLOG — p2_aa (does the diagnostic's D4 gain replicate?)

**Run:** `runs/p2_aa`. Line **R**: D4's protocol from DIAG-2026-11-13 run a second time
(`scripts/distill_curve.py`, teacher rows only, the same 4,000,000 rows, lr 2e-5 constant, batch 1,024, held-out
measured every 250,000 samples, patience 4, cap 16,000,000), from `p2_w_quarter_down/0010` (sha256
`43494548…3857`), with `--seed 1` in place of D4's 0: the batch order and the 5,000-row training probe change,
nothing else. Then twelve skill-5 reps of R, six more of D4 (`runs/diag_cod/curve/size_4000000/best.pt`, sha256
`7f827f08…`) and six more of the start, so each of the three stands on twelve; R's mate-in-2 suite row; three
verdicts. All on this box, 2026-11-15 09:35 to 15:46 UTC, under `gpu hold`, as the transient user unit `taz-p2aa`
(`runs/p2_aa_run.sh`), code `7e2cfcc`. **Pre-registration:** `PREREG-p2_aa.md` (`7e2cfcc`). Every verdict below
was registered before it was measured, except where marked.

**Verdict: the registered first row, "replicated". K1: R beats the start at skill 5, +0.0354 [+0.0112, +0.0596]
(R 0.4165 against 0.3810, twelve reps each). K0: D4 holds at twelve reps, +0.0452 [+0.0217, +0.0683] (0.4263). T
holds: R's best held-out cross-entropy is 2.2051 against D4's 2.2048, 0.0003 apart. K2 passes: R solves 189 of 200
mate-in-2 positions (floor 183; the start 188). Reported: R − D4 is −0.0098 [−0.0312, +0.0117]; two runs of the
protocol land within the instrument's resolution of each other. Training on many teacher positions, each seen a few
times, raises skill over the standing endpoint, and becomes the next lever. Michael chose its one change for p2_ab:
more data, 8,000,000 labelled positions (§6). The unit ended in a "failed" state and the log in "verdicts FAILED"
because of a bug in the run script, not in a measurement; every step ran and every record is complete (§7).**

---

## 1. Before launch

The draft is `2279a74` and the registration `7e2cfcc` (2026-11-15, Michael: "Ok given, go"). The round changes
nothing about D4's protocol but the seed: Michael's rule for this phase is one change at a time, and D4's +0.037
came from an aside of DIAG-2026-11-13 that decided nothing (its §5 and §7). At registration the `runs/p2_aa/` records
and every `reps_best_r*.jsonl` were negated in `.gitignore` and red-checked through `MUST_REACH` (9 of the 10 new
paths failed without the negations; the start's new reps were already reachable through `reps_iter*.jsonl`), and the
DIAG's twelve skill-5 rep files (D1's and D4's, six each), which no negation reached before, were committed. The
`MUST_REACH` and conventions tests passed (154, 4 skipped). No code changed between `58c6c11`, whose full suite
passed 1,257, and `7e2cfcc` (`git diff 58c6c11 7e2cfcc -- src scripts cloud` is empty).

## 2. T — the training replicated

R ran its 65 measurements, from 0 to 16,000,000 samples, without a patience stop: its longest run of measurements
without a new best was 3. Its best is its last: **held-out cross-entropy 2.2051 at 16,000,000 samples**, against D4's
2.2048 at 15,750,144. **T holds (0.0003 apart; the registered tolerance is 0.01).** The start measured 2.3673, top-1
0.3658 and value MSE 0.0378, the registered T1 reading to the digit, as in the DIAG.

R tracks D4 measurement by measurement, unregistered:

| samples | R held-out | D4 held-out | R top-1 | D4 top-1 | R value MSE | D4 value MSE |
|---|---|---|---|---|---|---|
| 0 | 2.3673 | 2.3673 | 0.3658 | 0.3658 | 0.0378 | 0.0378 |
| 1M | 2.2486 | 2.2476 | 0.3690 | 0.3682 | 0.0332 | 0.0321 |
| 4M | 2.2314 | 2.2314 | 0.3708 | 0.3744 | 0.0319 | 0.0316 |
| 8M | 2.2206 | 2.2205 | 0.3788 | 0.3786 | 0.0313 | 0.0319 |
| 12M | 2.2124 | 2.2122 | 0.3780 | 0.3764 | 0.0327 | 0.0337 |
| 16M | 2.2051 | 2.2051 | 0.3796 | 0.3842 | 0.0333 | 0.0355 |

At R's best: top-1 0.3796 (D4's best 0.3826), value MSE against the teacher 0.0333 (D4's best 0.0342; the start
0.0378), training-probe cross-entropy 2.1911 (a different 5,000 rows from D4's, by the seed; D4's 2.1935).

## 3. K — skill 5, twelve reps each

The instrument as registered: `eval_reps.py run --anchor stockfish:5 --config runs/p2_w_quarter_down/config.yaml`,
the 100-opening book, 200 games a rep, 48 simulations a move; the verdicts per-opening rep-means with a paired
bootstrap over the 100 openings (10,000 resamples, seed 0).

### K1 (primary) — R against the start
**+0.0354 [+0.0112, +0.0596]**, se 0.0123 (`p2aa_verdict_r_vs_start.json`): R 0.4165 against the start's 0.3810.
**Entirely above zero.**

### K0 — D4 against the start
**+0.0452 [+0.0217, +0.0683]**, se 0.0120 (`p2aa_verdict_d4_vs_start.json`): D4 0.4263 against 0.3810. **Entirely
above zero.** The DIAG's six reps of D4 read 0.4158; the six added here read 0.4367.

### K2 — mate-in-2
**189 of 200** at R's checkpoint (`r_ladder.jsonl`, the suite row at 128 sims under the start's config). **K2 passes**
(floor 183). The start solves 188. R solves `mate2_054` and `mate2_140`, which the start fails, and fails `mate2_165`,
which the start solves; the other ten failures are the start's. Mate-in-3: 163 against the start's 164. The ladder's
console prints "REGRESSED  M3 CLAUSE FAILED": that is the phase 1 milestone (0.95 of the suite, 190, against a
baseline of 0.985), which the start misses too (188) and no row of this round reads. The registered floor is 183.

### Reported, no decision
- **R − D4: −0.0098 [−0.0312, +0.0117]** (`p2aa_verdict_r_vs_d4.json`): the two runs of the protocol do not separate.
- Single reps overlap widely: the start's twelve run 0.3275–0.4300, R's 0.3500–0.4700, D4's 0.3825–0.4750. Only the
  rep-means over openings separate them.

## 4. The instrument between dates (reported)

The start's first six reps were measured by p2_w's finisher on 2026-11-11/12, the six added here on 2026-11-15:
**0.3788 against 0.3833, +0.0046 [−0.0300, +0.0396]** (`eval_reps.py verdict`, re-derived for this log into a
scratch path). The instrument did not drift between the two dates at this resolution. D4's two sets of six, on
2026-11-14 and 2026-11-15: 0.4158 against 0.4367, +0.0208 [−0.0133, +0.0546]; the second set did not regress toward
the start.

## 5. The predictions

| prediction (written before launch) | result | |
|---|---|---|
| T holds (R's held-out CE 2.20–2.21) | 2.2051 | held |
| K1 point +0.01 to +0.05 | +0.0354 | held |
| K1's interval excludes zero, more likely than not | [+0.0112, +0.0596] | held |
| K0 point +0.015 to +0.04 | +0.0452 | **missed**, 0.005 above the range |
| K0's interval excludes zero, about even odds | [+0.0217, +0.0683] | held as a direction |
| K2 passes (R 186–193) | 189 | held |
| R − D4 within ±0.04 | −0.0098 | held |

The K0 range rested on the view that D4's first six reps were a high draw that six more would pull toward the start.
They did the opposite (§4).

## 6. Decision

Per the table's first row (K1 entirely above zero; T holds; K2 passes): **replicated. Training on many teacher
positions, each seen a few times, raises skill over the standing endpoint. The protocol becomes the next lever, and
the round after, p2_ab, changes one thing about it. Michael chose more data (2026-11-15): 8,000,000 labelled
positions (about $6 of labelling), drafted separately as PREREG-p2_ab.** The other two candidates the row named, a
2e-6 step-down after the protocol and self-play rows mixed back in, stay on the open list.

Readings the table does not cover (all unregistered):

1. **Two runs, one curve.** R's held-out cross-entropy stays within about 0.001 of D4's at every matched sample
   count (§2): the batch order barely moves where this protocol ends. Both runs were still improving at the cap (R's
   best at its last measurement, D4's at its second-to-last), as DIAG Addendum A found; the 16M-sample cap, not the
   network, set where they stopped.
2. **The skill-5 means are the highest this project has recorded.** Start 0.3810, R 0.4165, D4 0.4263, each over
   twelve reps. Across the 85 skill-5 verdict files under `runs/`, the highest mean before the DIAG was 0.3917
   (`p2_w_quarter_down/0010`, three reps); the best six-rep mean was 0.379 (RUNLOG-p2_w). The value head is no worse than the
   start's against the teacher (R 0.0333, D4 0.0342, start 0.0378), unlike the DIAG's D1, whose value MSE rose to
   0.094 as it fell to 0.113 at skill 5.
3. **Why it works is still a hypothesis.** What this round shows is that the protocol replicates. That positions
   seen a few times help while a small ring drilled many times hurts (the DIAG's D1, p2_v's half dose) is the reading
   that fits; p2_ab's eight million rows test one side of it.

The open list, carried forward: the 2e-6 step-down after the protocol; self-play rows mixed back in; a cap above
16M samples for the larger sets (the DIAG's "not resolved (compute)"); **p2_z, the benchmark track's clock-scaled
gauntlet** (RUNLOG-p2_y); the runner logging an engine's exception (RUNLOG-p2_y §3); and from p2_w: a deeper engine,
the 2e-7 step-down, per-source loss weights, the calibration sample, the promotion loss at 2e-6, L2's exclusion at
storage time.

## 7. Mechanics

- **Unattended.** `taz-p2aa` armed at 2026-11-15 09:35:01 UTC at `7e2cfcc`. Each step took `gpu hold --by
  tinyalphazero` (ComfyUI not running, nothing stopped, 101.8 GB free) with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
  PYTHONUNBUFFERED=1`. R's training 09:35:01–12:09:14 (2 h 34 min; D4 took 2 h 32 min); R's twelve reps to 13:58:19
  (1 h 49 min); D4's six to 14:50:53; the start's six to 15:45:08; R's suite row, then the verdicts, logged at
  15:46:49. 6 h 12 min in all, against "about 6 h" registered. No commit went to `main` while the unit ran.
- **The "failure" was the script's.** `scripts/eval_reps.py verdict` exits 1 when its interval includes zero, by
  design (a verdict, not an error). `runs/p2_aa_run.sh` ran the three verdicts as one block and read a non-zero exit
  as a failure. The block's last verdict, R − D4, includes zero, so the script logged "verdicts FAILED" and exited 1,
  and the unit ended in `failed`, cleared with `systemctl --user reset-failed` at the check-in. All three verdict files
  were written in full first ("wrote …" for each in the log); only the script's closing "p2_aa complete" line is
  missing. **A run script must ignore `eval_reps.py verdict`'s exit status.**
- **Scoring.** The result was scored at the 16:00 check-in. Michael was notified, saw the verdict and agreed to
  closing (2026-11-15).
- **The box** stayed up throughout.
- **Cost:** no spend. Nothing ran on RunPod; R trained on the DIAG's labels.

## 8. Artefacts

| file | what |
|---|---|
| `runs/p2_aa/r/size_4000000/result.json`, `log.jsonl` | R's best measurement and all 65 (T; §2's table) |
| `runs/p2_aa/r_sf5/reps_best_r{1..12}.jsonl` | R's twelve skill-5 reps (K1) |
| `runs/p2_aa/d4_sf5_more/reps_best_r{1..6}.jsonl` | D4's six added reps (K0, with `runs/diag_cod/curve/size_4000000/sf5/`) |
| `runs/p2_aa/start_sf5_more/reps_iter0010_r{1..6}.jsonl` | the start's six added reps (K1, K0, §4, with `runs/p2_w_quarter_down/sf5/`) |
| `runs/p2_aa/r_ladder.jsonl` | R's suite row (K2) and the 20-game random match that carried it |
| `runs/p2_aa/p2aa_verdict_{r_vs_start,d4_vs_start,r_vs_d4}.json` | K1, K0, R − D4 |
| `runs/p2_aa/r/size_4000000/best.pt`, `ce_rows.npy` | R's checkpoint and per-row held-out CE (ignored) |
| `runs/p2_aa_run.sh`, `runs/p2_aa_run.console.log` | the unit's script and console (untracked) |
