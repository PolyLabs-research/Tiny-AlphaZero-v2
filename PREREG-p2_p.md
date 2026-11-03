# Pre-registration — p2_p: p2_o's A/B replicated on a new seed (2e-6 against 2e-5, from p2_n/0040, judged on skill 3)

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-04** (see Sign-off), at Michael's instruction. Nothing under this file
> has run.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_o's K1 landed on its boundary: lr − cont +0.031, interval [0, +0.0625] with its lower end exactly zero
(RUNLOG-p2_o §3), ruled by Michael "no effect seen at this length", so 2e-5 stays. Every unregistered
reading leaned to the step-down (each arm against the start: lr +0.0825, cont +0.051; the initial
position lr +0.021; mate-in-3 162; the value-loss rise p2_k_lr had when it won). RUNLOG-p2_o §6 recommends
a registered replication, and p2_p is that: **the same A/B, from the same checkpoint, judged the same way,
on an independent seed.**

**H_lr2 (as PREREG-p2_o): a tenfold step-down to 2e-6 from p2_n/0040 improves skill-3 play from the frozen
book over a continuation arm at 2e-5 within twenty iterations, without costing mate-in-2.**

**Why a new seed.** Every stochastic part of a round — self-play games, league draws, training batches,
rehearsal draws — comes from `iteration_seed(seed, iteration, stream)`. Seed 43 in place of 42 makes all of
them independent of p2_o's, from the same weights, ring and league: a second draw of the same experiment,
not a re-run of the same draw. The seed is set at the invocation (`--seed 43`), which writes it into each
run's saved `config.yaml` before anything else is written; no config file changes.

## Contract

| | |
|---|---|
| Arms | **`p2_p_cont`**: `--config configs/p2_k_lr.yaml --seed 43` (2e-5; resolved fingerprint `d42a6d33…346e`). **`p2_p_lr`**: `--config configs/p2_o_lr.yaml --seed 43` (2e-6; resolved fingerprint `86fcc672…7d65`). Each resolved config differs from its file in `seed` only, and from the other arm in `lr` only |
| What differs from p2_o | the seed (43 against 42), in both arms; the instrument's exact-zero fix (`evaluate.CI_BOUND_TOL`, `f7779a8`): an interval end within 1e-9 of zero counts as on zero. Nothing else |
| Warm start | `runs/p2_n/ckpt_iter0040.pt`, both arms (p2_o's) |
| Continuity | `--seed-pool runs/p2_n`, `--seed-replay runs/p2_n`: committed head 2,649,005 = durable head, newest 500,000 rows, offset −40; both arms identical. `--init` in both arms |
| Length | `--iterations 20` per arm |
| Start reference | p2_o's: **skill 3**, p2_n/0040's six book reps (mean 0.6442; `runs/p2_n/sf3/reps_iter0040_r{1..6}.jsonl`); **skill 0**, its three book reps and three initial-position reps |
| Training | RunPod, **two pods in parallel**, each a secure RTX 4090 with ≥ 24 vCPU, the image of p2_g–p2_o; variants recorded. If none is offered, the registered launch is retried; nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | as PREREG-p2_o: this box, torch 2.14.1+rocm7.2, `gpu hold`, nothing until each arm's final pull; one `remote_campaign.sh` per arm with `MEASURE_AFTER_PULL=1` (judged `10 20`: ladder, ECE, N, three skill-0 reps; initial-position reps at 20); then a finisher: per arm three book reps against `stockfish:3` at 10 and six at 20 |
| Lifecycle | four guards; pod deadline **8 h** each; worst case **$7.12 per pod, $14.24 together** at $0.89/h; expected about **$7.7** together |

## Registered expectations

### L — the continuity applied
- **L1.** Each arm's `provenance.json`: `init_from` `runs/p2_n/ckpt_iter0040.pt`; `seed` 43;
  `seed_replay.rows` 500,000, `donor_committed` = `donor_head` = 2,649,005, `iteration_offset` −40;
  `git_dirty` false; `config_fingerprint` `d42a6d33…` (cont) and `86fcc672…` (lr); each pulled
  `config.yaml` differs from its file in `seed` only; `lr` on every row 2e-5 (cont) and 2e-6 (lr).
- **L2.** Book positions in each arm's final ring: reported, no bound.

### K1 (primary) — the step-down against the continuation, skill 3
`p2_p_lr` − `p2_p_cont` at iteration 20, against `stockfish:3`, **six reps per arm**, per-opening
rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed
0; `runs/p2_p_lr/sf3/p2p_verdict_lr_vs_cont.json`). **Expected: the 95% CI entirely above zero**, read by
`excludes_zero` as fixed in `f7779a8` (an end within 1e-9 of zero is on zero). Forecast point +0.02 to
+0.06 (p2_o read +0.031).

### K2 — mate-in-2
At 20: **`p2_p_lr` solves at least as many of the 200 as `p2_p_cont` minus 4**, and **both ≥ 183**.

### K3 — each arm against the start (informational)
Each arm at 10 and 20 against p2_n/0040's six skill-3 reps (three reps; six at 20).

### K4 — skill 0 and the initial position (continuity)
As PREREG-p2_o.

### K5 — p2_o and p2_p together (informational, registered now)
Per arm, the twelve skill-3 reps at 20 of the two seeds (p2_o's six and p2_p's six), lr − cont, the
same statistic (`runs/p2_p_lr/sf3/p2p_verdict_pooled_o_and_p.json`). It reads the two replicates'
combined estimate. **No decision row turns on it:** the bootstrap resamples openings, not runs, so
pooling two runs per arm narrows the interval without measuring the run-to-run spread, and p2_o's half
of it has been seen.

### ECE, H
- **ECE:** < 0.15 at 20, both arms; a break is handled in PREREG-p2_l's order.
- **H1–H8** as PREREG-p2_o, both arms. **H7** is read against p2_o: whether the step-down arm's value loss
  rises again (p2_o lr 0.358 → 0.395; p2_k_lr 0.319 → 0.357).

## Predictions (written before launch)
- K1 passes, point +0.02 to +0.06; K2 passes.
- K3: both arms above the start at 20, as in p2_o; lr above cont at both checkpoints.
- K5 above zero.
- The step-down arm's value loss rises over its twenty iterations (H7).
- ECE 0.08–0.12; N 185–195; L2 0–5 per arm.

## Decision rules

| K1 | K2 | Decision |
|---|---|---|
| **CI entirely above zero** | passes | **The second step-down replicates.** 2e-6 becomes standing protocol; p2_q continues `p2_p_lr/0020` at 2e-6 as the default round, judged at skill 3 (or the next rung, if 0020 reads above 0.75 there) |
| CI entirely above zero | fails | 2e-5 stays until the mate-in-2 cost is diagnosed |
| **CI includes zero** | any | **Not replicated at this length.** 2e-5 stays; the twenty-iteration step-down is closed as "no effect seen" twice. p2_q's lever is Michael's choice (a longer step-down arm is the candidate the two leaning readings point to) |
| **CI entirely below zero** | any | **The step-down costs skill.** 2e-5 stays; diagnosed before p2_q |
| N < 183 or ECE ≥ 0.15 in either arm | | The broken guard first |

## Measurement protocol

```bash
# Two pods, staged as p2_o's (the same inputs: runs/p2_n/ckpt_iter00{21..40}.pt, state.json, config.yaml, replay/,
# the mate data, phase1_anchor.pt; sha256 checked on each pod)
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2pc --vcpu 24 --ram 32 --disk 60 --yes
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2pl --vcpu 24 --ram 32 --disk 60 --yes

# On each pod
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --config configs/p2_k_lr.yaml --seed 43 \
   --init runs/p2_n/ckpt_iter0040.pt --seed-pool runs/p2_n --seed-replay runs/p2_n \
   --out runs/p2_p_cont --iterations 20" > /dev/null 2>&1 < /dev/null &
#   ... and on the other pod --config configs/p2_o_lr.yaml --seed 43 --out runs/p2_p_lr

# This box, one driver per arm (as p2_o: START_REPS/START_STARTPOS from runs/p2_n/…0040), MEASURE_AFTER_PULL=1,
# judged "10 20" final 20; then a finisher as runs/p2_o_finish.sh for the p2_p arms, and:
.venv/bin/python scripts/eval_reps.py verdict --a cont runs/p2_p_cont/sf3/reps_iter0020_r*.jsonl \
    --b lr runs/p2_p_lr/sf3/reps_iter0020_r*.jsonl --out runs/p2_p_lr/sf3/p2p_verdict_lr_vs_cont.json
.venv/bin/python scripts/eval_reps.py verdict \
    --a cont runs/p2_o_cont/sf3/reps_iter0020_r*.jsonl runs/p2_p_cont/sf3/reps_iter0020_r*.jsonl \
    --b lr runs/p2_o_lr/sf3/reps_iter0020_r*.jsonl runs/p2_p_lr/sf3/reps_iter0020_r*.jsonl \
    --out runs/p2_p_lr/sf3/p2p_verdict_pooled_o_and_p.json
```

## What would make p2_p uninterpretable
- A difference between the arms beyond `lr`, or a difference from p2_o beyond the seed (L1 checks both).
- A launch from anything but clean, pushed `main`; a measurement-instrument change beyond `f7779a8`.
- The arms on different GPU variants: recorded and reported, not a stop.
- Losing either arm's final pull: re-run by addendum.

---

## Sign-off

Checked for the draft, 2026-11-03, on this box: p2_n's state, pool, ring and start references as signed in
PREREG-p2_o (unchanged since); `--seed` writes the effective seed into the run's config before it is saved
(`Phase2Run.create`), and every stochastic stream is `iteration_seed(seed, iteration, stream)`; the
seed-43 fingerprints computed from `configs/p2_k_lr.yaml` and `configs/p2_o_lr.yaml`:
`d42a6d33037e6f7c763efd52e7ef22417303bbbd2c7a08389da0b62fbfe2346e` (cont),
`86fcc672d739acc598d681f279bed714aa845af3951cecdeb8add69892d17d65` (lr). Read for the draft: RunPod balance
$30.62 (auto-reload at $10), no pods of ours; secure RTX 4090 at $0.89/h (stock "None"); $14.24 leaves
$16.38, above the $2 margin. Read at registration, 2026-11-04 08:52 UTC: balance $30.24, no pods of ours,
secure RTX 4090 at $0.89/h (stock "None"; the registered launch is retried until offered); $14.24 leaves
$16.00, above the $2 margin. The 50 staged inputs' sha256 re-checked against p2_o's list: unchanged.
`make test` passed on this commit's tree (954 passed, 2 skipped).

**Signed: the Claude Code session, at Michael's instruction ("happy, go", 2026-11-03, after the draft and
its cost — expected about $7.7, worst case $14.24 — were put to him).** The commit carrying this line is
the registration.

---

## Addendum A — 2026-11-03: the registration's date

The status line ("REGISTERED 2026-11-04") and the sign-off's read ("2026-11-04 08:52 UTC") carry the wrong
day. The registration commit `0781889` is timed 2026-11-03 08:54:32 UTC, and the balance, price and stock
were read on 2026-11-03 at 08:52 UTC. Both lines mean 2026-11-03. Nothing else in this file changes; the
pods run the bundle of `0781889`.
