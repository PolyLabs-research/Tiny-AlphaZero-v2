# Pre-registration — p2_r: self-play at 128 simulations against 48, twenty iterations each from p2_q/0040, judged on skill 5

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: REGISTERED 2026-11-05** (see Sign-off), at Michael's instruction. Nothing under this file
> has run.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_q read **plateau at 2e-6** (RUNLOG-p2_q §6: K1 at skill 5 +0.009 [−0.018, +0.035]; flat at skill 3 too),
and its row left the next lever to Michael. He chose **more search in self-play** (2026-11-05), within the
project's aim: push the fixed 6.5M-parameter network as far as it goes.

The reading behind it (RUNLOG-p2_q §6 item 1): the policy loss keeps falling while play is flat, so the
network fits its self-play targets ever better and those targets — each move's search at 48 simulations
(`selfplay.sims_per_move`) — have stopped carrying new skill. More simulations per self-play move give
better search results to learn from (AlphaZero searched 800 per move in self-play) without touching the
network, the learning rate or the measuring instrument, which reads `eval.eval_match_sims` (48) and
`eval.suite_sims` (128) and stays where it is.

**H_sims: from p2_q/0040, twenty iterations whose self-play searches 128 simulations per move end
stronger at skill 5 than twenty at 48, without costing mate-in-2.**

**Why 128.** 2.7 times the search is a step large enough to show, at a cost one pod can carry: self-play
is about 90% of an iteration (9.6 of 10.5 minutes at 48), so an iteration at 128 runs near 27 minutes and
twenty near 9 hours. The root uses Gumbel sequential halving over `gumbel_m` 16 considered moves; more
simulations deepen the same 16 candidates' evaluations.

## Contract

| | |
|---|---|
| Arms | **`p2_r_cont`**: `--config configs/p2_o_lr.yaml --seed 43` (48 sims; fingerprint `86fcc672…7d65` = p2_q's). **`p2_r_sims`**: `--config configs/p2_r_sims.yaml --seed 43` — p2_o_lr's config with `selfplay.sims_per_move: 128` and nothing else (resolved fingerprint `1c284171…6a08`; `tests/test_rehearsal.py::test_p2r_sims_config_is_the_brief`, which also pins `eval_match_sims` 48 and `suite_sims` 128 in both) |
| What differs between the arms | the self-play search budget per move (128 against 48), and with it each iteration's self-play time. The games per iteration (512), the training steps (400 per iteration), the learning rate (2e-6), the seed (43), the network and the instrument are the same |
| Warm start | `runs/p2_q/ckpt_iter0040.pt`, both arms |
| Continuity | `--seed-pool runs/p2_q` (its league, 0021–0040), `--seed-replay runs/p2_q`: committed head 2,620,249 = durable head, newest 500,000 rows, offset −40; both arms identical. `--init` in both, as standing protocol |
| Length | `--iterations 20` per arm |
| Start reference | **skill 5:** p2_q/0040's six book reps (0.3075, 0.3125, 0.325, 0.295, 0.2675, 0.38; mean 0.3146; `runs/p2_q/sf5/reps_iter0040_r{1..6}.jsonl`). **skill 3:** its three reps (0.745, 0.7525, 0.785; `runs/p2_q/sf3/`). **skill 0:** its three book reps and three initial-position reps. Its ladder row at 40: mate-in-2 189, ECE 0.0966 |
| Training | RunPod, **two pods in parallel**, each a secure RTX 4090 with ≥ 24 vCPU, the image of p2_g–p2_q; variants recorded. If none is offered, the registered launch is retried; nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, `gpu hold`, nothing until each arm's final pull. One `remote_campaign.sh` per arm with `MEASURE_AFTER_PULL=1` (judged `10 20`: the ladder pass with ECE and N, three skill-0 reps; initial-position reps at 20). Then a finisher: per arm three book reps against `stockfish:5` at 10 and six at 20 (`runs/p2_r_{cont,sims}/sf5/`), and three against `stockfish:3` at 20 (`…/sf3/`) |
| Lifecycle | four guards. **`p2_r_cont`: deadline 8 h, worst $7.12. `p2_r_sims`: deadline 14 h, worst $12.46** (twenty iterations near 27 minutes ≈ 9 h, with room for a slower pace). **Worst case $19.58 together** at $0.89/h; expected about **$12.5** (cont ≈ 4.3 h, sims ≈ 9.5 h) |

## Registered expectations

### L — the continuity applied
- **L1.** Each arm's `provenance.json`: `init_from` `runs/p2_q/ckpt_iter0040.pt`; `seed` 43;
  `seed_replay.rows` 500,000, `donor_committed` = `donor_head` = 2,620,249, `iteration_offset` −40;
  `git_dirty` false; `config_fingerprint` `86fcc672…` (cont) and `1c284171…` (sims), each equal to its
  `config.yaml`'s sha256; the two pulled configs differ in `sims_per_move` only; `lr` 2e-6 on every row.
  **The lever applied:** each sims-arm iteration row's `sims` (simulations run) is about 128/48 of the cont
  arm's for the same games — reported per iteration.
- **L2.** Book positions in each arm's final ring: reported, no bound.

### K1 (primary) — 128 against 48, at skill 5
`p2_r_sims` − `p2_r_cont` at iteration 20, against `stockfish:5`, **six reps per arm**, per-opening
rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000 resamples, seed
0; `runs/p2_r_sims/sf5/p2r_verdict_sims_vs_cont.json`), read with the exact-zero rule. **Expected: the
95% CI entirely above zero.** Forecast point +0.02 to +0.08.

### K2 — mate-in-2
At 20: **`p2_r_sims` solves at least as many of the 200 as `p2_r_cont` minus 4**, and **both ≥ 183**.

### K3 — each arm against the start, and the last rung (informational)
Each arm at 10 and 20 against p2_q/0040's six skill-5 reps (three reps; six at 20); each arm's three
skill-3 reps at 20 against p2_q/0040's three; skill 0 and the initial position as before.

### ECE, H
- **ECE:** < 0.15 at 20, both arms; a break is handled in PREREG-p2_l's order.
- **H1–H9** as PREREG-p2_q, both arms, direction-free. **H10 (new):** the self-play itself under the lever —
  minutes per iteration, decisive rate, game length and the policy-target entropy (`H_target` in the
  console) by arm. **H7** is read for the reading behind the lever: whether the sims arm's policy loss
  stops falling (its targets now harder to fit) while it plays better.
  **H2:** `skipped_steps` < 10 per arm.

## Predictions (written before launch)
- K1 passes, point +0.02 to +0.08; K2 passes.
- K3: cont within noise of the start at 20 (the plateau holds a third time); sims above it.
- H10: sims-arm iterations 24–30 minutes; its policy loss higher than cont's (sharper targets fit less
  easily).
- ECE 0.08–0.12, N 185–195, L2 0–5 per arm.

## Decision rules

| K1 | K2 | Decision |
|---|---|---|
| **CI entirely above zero** | passes | **More self-play search supported.** 128 becomes standing for self-play; p2_s continues `p2_r_sims/0020` at 128 as the default round, judged at skill 5 (or the next rung, if 0020 reads above 0.75 there) |
| CI entirely above zero | fails | 48 stays until the mate-in-2 cost is diagnosed |
| **CI includes zero** | any | **No effect seen at this length.** 48 stays; the plateau stands; p2_s's lever is Michael's choice (the rest of the list: 2e-7, distillation, a longer search arm) |
| **CI entirely below zero** | any | **More search costs skill at this length.** 48 stays; diagnosed before p2_s |
| N < 183 or ECE ≥ 0.15 in either arm | | The broken guard first |

## Measurement protocol

```bash
# Two pods (launch, wait-ssh, stage; inputs' sha256 checked on each pod), the same inputs on both
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2rc --vcpu 24 --ram 32 --disk 60 --yes
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 14 --tag p2rs --vcpu 24 --ram 32 --disk 60 --yes
bash cloud/stage_pod.sh POD runs/p2_q/ckpt_iter00{21..40}.pt runs/p2_q/state.json runs/p2_q/config.yaml \
    runs/p2_q/replay/ runs/data/mate1_100k_clean/ runs/data/mate2_50k_clean/ runs/phase1_anchor.pt

# On each pod
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --config configs/p2_o_lr.yaml --seed 43 \
   --init runs/p2_q/ckpt_iter0040.pt --seed-pool runs/p2_q --seed-replay runs/p2_q \
   --out runs/p2_r_cont --iterations 20" > /dev/null 2>&1 < /dev/null &
#   ... and on the other pod --config configs/p2_r_sims.yaml --seed 43 --out runs/p2_r_sims

# This box, one driver per arm, MEASURE_AFTER_PULL=1, START_REPS/START_STARTPOS from runs/p2_q/…0040, judged
# "10 20" final 20; then a finisher like runs/p2_o_finish.sh at stockfish:5 (start runs/p2_q/sf5/reps_iter0040_r*),
# plus three stockfish:3 reps at 20 per arm (against runs/p2_q/sf3/), M4, L2, and:
.venv/bin/python scripts/eval_reps.py verdict --a cont runs/p2_r_cont/sf5/reps_iter0020_r*.jsonl \
    --b sims runs/p2_r_sims/sf5/reps_iter0020_r*.jsonl --out runs/p2_r_sims/sf5/p2r_verdict_sims_vs_cont.json
```

## What would make p2_r uninterpretable
- A difference between the arms beyond `sims_per_move` (the config test and L1 catch it).
- The lever not applied: the sims arm's per-iteration `sims` not near 128/48 of the cont arm's (L1).
- A launch from anything but clean, pushed `main`; a measurement-instrument change.
- The arms on different GPU variants: recorded and reported, not a stop.
- Losing either arm's final pull: re-run by addendum.

**What the statistic does not cover.** As PREREG-p2_k: each arm is one run.

---

## Sign-off

Checked for the draft, 2026-11-05, on this box: p2_q's `state.json` iteration 40, step 16,000, seed 43, 0
skipped steps, pool `ckpt_iter0021`–`0040` (all present), `replay_total_written` 2,620,249 = the ring's
`total_written`; its six skill-5 reps, three skill-3 reps, three skill-0 reps and three initial-position
reps present; self-play reads `selfplay.sims_per_move` (`phase2.py`) and the instrument
`eval.eval_match_sims`/`eval.suite_sims` (`evaluate.py`); `configs/p2_r_sims.yaml` differs from
`configs/p2_o_lr.yaml` in `sims_per_move` only (test passes); seed-43 fingerprints
`86fcc672d739acc598d681f279bed714aa845af3951cecdeb8add69892d17d65` (cont) and
`1c2841712f292f5b866a6be5a767c751eb21f16c8add71f871f3a09aa4d06a08` (sims). Read for the draft: RunPod
balance $17.34 (auto-reload at $10), no pods of ours; secure RTX 4090 at $0.89/h (stock "Medium"); each pod's
worst case passes the $15 ceiling and the $2 margin on its own; the combined $19.58 is above the balance and
is covered by the auto-reload. Read at registration, 2026-11-05 10:04 UTC: balance $17.18, no pods of
ours, secure RTX 4090 at $0.89/h (stock "Medium"); each pod's worst case passes the $2 margin on its own
($12.46 leaves $4.72, $7.12 leaves $10.06). `make test` passed on this commit's tree (993 passed, 2
skipped).

**Signed: the Claude Code session, at Michael's instruction ("ok given, go", 2026-11-05, after the draft
and its cost — expected about $12.5, worst case $19.58 — were put to him).** The commit carrying this line
is the registration.
