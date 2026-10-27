# FINDING — 2026-10-27 — p2_i: where the `per_row` cost is not

**Context.** RUNLOG-p2_i's K1 failed: `per_row` − `batch_mean` at iteration 20, six reps per arm,
−0.040 [−0.070, −0.010] on skill-0 from the frozen book, with mate-in-2 unchanged (186 against 187).
The registered decision row says the cost is diagnosed before p2_j. This is that diagnosis, bounded
on purpose: a value-head probe of the two iteration-20 checkpoints, no search, no training, on this
box. The instrument is `scripts/diagnose_p2i_value.py`; the record is
`runs/p2_i_row/diag_value_probe.json` (seed 0, 5,000 positions per sampled set).

**Nets.** `cont` = `runs/p2_i_cont/ckpt_iter0020.pt`, `row` = `runs/p2_i_row/ckpt_iter0020.pt`,
`start` = `runs/p2_h/ckpt_iter0060.pt` (both arms' warm start). **Position sets**, shared across
nets so every comparison is paired: the book's 100 openings and the initial position; 5,000 rows
each from the two rehearsal sets (`mate1_100k_clean`, `mate2_50k_clean`: the rows whose value weight
the lever doubled); 5,000 rows from each arm's final replay ring, with the stored outcome and
`root_q`. The value read is P(win) − P(loss) from the side to move, the quantity search consumes.

## Readings

| set | reading | cont | row | start |
|---|---|---|---|---|
| mate-in-1 rows | CE on the (won) outcome | 0.0251 | **0.0136** | 0.0244 |
| mate-in-2 rows | CE on the outcome | 0.0141 | **0.0053** | 0.0172 |
| mate-in-2 rows | mean v | 0.984 | 0.994 | 0.983 |
| own ring | mean \|v\| | 0.353 | 0.349 | 0.362 (cont's ring) |
| own ring | fraction \|v\| > 0.9 | 0.176 | 0.164 | 0.142 |
| own ring | 10-bin ECE against the stored outcome | **0.023** | **0.041** | 0.043 / 0.053 |
| other arm's ring | ECE | 0.049 | 0.045 | |
| own ring | MSE against `root_q` / correlation | 0.066 / 0.88 | 0.064 / 0.88 | 0.065 / 0.88 |
| book, 100 openings | mean \|v\| | 0.048 | 0.040 | 0.041 |
| book | corr(v_row − v_cont, per-opening K1 difference) | | +0.07 | |

1. **The lever did exactly its mechanical thing.** Doubling the rehearsal rows' value weight
   roughly halved the value head's cross-entropy on them (mate-in-1 0.025 → 0.014, mate-in-2
   0.014 → 0.005); `start` sits with `cont`. That is the whole direct effect of `per_row`, and it
   bought nothing on the searched suite: K2 read 186 against 187. The mates the suite still misses
   are not missed for want of value confidence on mate positions.
2. **Not extremeness.** The cheap hypothesis — full-weight ±1 targets on 5% of rows push middlegame
   values toward ±1 — is not supported: on self-play positions the two arms' mean |v| and the share
   of |v| > 0.9 are the same (0.35 / 0.16–0.18), and both sit a little above `start`'s (0.36 / 0.14)
   in the share of extreme values, which is the restart's doing, not the lever's.
3. **Not the book.** Both nets read the 100 openings as level (|v| ≈ 0.04–0.05), and the per-opening
   difference in their value, signed or absolute, does not correlate with the per-opening K1
   difference (r = +0.07 and −0.12). The cost is not an opening-evaluation cost.
4. **A calibration gap, same direction as the ladder's ECE.** Each net is best calibrated on its own
   self-play, as expected, but `cont` on its own ring reads ECE 0.023 where `row` reads 0.041 on
   its own; across rings the two are 0.045–0.049; the ladder's ECE at 20 read 0.054 against 0.064.
   On its own ring the gap sits in the low bins: where `row` predicts P(win) 0.15–0.35, the realized
   rate is 0.10–0.13 lower (`cont` on its own: 0.03–0.12 lower in two of those bins, and 0.05–0.06
   *higher* in the 0.6–0.9 bins), so `row` under-calls losing positions in its own self-play. The
   ladder's table, on anchor-match positions, puts both arms' miscalibration in the 0.5–0.7 bins
   with `row`'s the larger; the two readings agree on the size (about 0.01–0.02 ECE) and not on
   the location, so the location is not a finding. Agreement with the stored search value is
   identical (MSE 0.064–0.068, r 0.88 for all three nets).

## Conclusion

The probe finds the lever's direct effect where the code puts it (the mate rows) and rules out the
two value-output mechanisms that would have been cheap to act on. What is left visible is a
calibration difference of about 0.02 ECE on self-play positions. Whether the −0.040 on skill-0 is
that, the training dynamics the reweighting changed (the same 400 steps per iteration, 5% of the
batch now at double value weight, and the self-play distribution those weights then produced), or
partly the draw (SE 0.0155), is not decidable from the value outputs; reaching it would mean
training-time instrumentation, i.e. another round on the lever. PREREG-p2_i's row has already
closed it: `batch_mean` stays, `per_row` is off the list, and p2_j is not conditioned on this
finding. **No further diagnosis is scheduled.**

Two things the record keeps from this: the mate-in-2 residual is a *search/policy* residual, not a
value-confidence one (the value head already reads +0.98 on those positions under both blends); and
the ladder's ECE moved with the lever (0.054 → 0.064) while staying far inside the 0.15 guard, so
ECE is a readable instrument at this scale, not only a guard.
