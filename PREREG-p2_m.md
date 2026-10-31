# Pre-registration — p2_m: the restart against a resume, twenty iterations each from p2_l/0040, judged on skill 3

**Written before launch. Not edited after, except by dated addendum.**

> **STATUS: DRAFT 2026-10-31 — not registered.** Nothing launches until Michael's cost OK; the
> registration commit flips this line and carries the sign-off.

**Amendment policy.** As PREREG-p2_h: a rule may be added after launch only for a measurement that has
not yet produced data, by dated addendum; never added, weakened or reworded for one already taken.

---

## The question

p2_l's registered row (RUNLOG-p2_l §6): **still climbing at 2e-5** — K1 at skill 3 +0.104 [+0.071,
+0.136] — so continuation at 2e-5 stays the default round and p2_m continues p2_l/0040, judged at skill 3.

The same RUNLOG records a shape its table did not cover, now seen twice: **the round's gain is in place
by iteration 10.** p2_k_lr took +0.110 of its +0.132 (skill 0) by 10; p2_l +0.070 of its +0.092 (skill 3,
three reps), and its four checkpoints are flat within noise (slope +0.005 per ten iterations). Every round
since p2_g starts with `--init`, which **restarts** two things: a fresh AdamW (zero moments, step 0) and
the per-iteration seed streams (iteration 1 again). A fresh Adam's first updates are close to `lr` per
parameter in size, against the much smaller steps of moments that have averaged a noisy gradient for
16,000 steps — a burst, at the start of every round. Whether that burst is the gain is the question, and
it decides what the default round should be: if the restart is the lever, rounds should restart more
often; if it is not, the early step is something else and longer rounds lose nothing.

The evidence so far does not separate it: p2_i_cont rose after a restart (+0.057 at 10) and p2_j did not
(−0.026 at 10), both at 2e-4. So p2_m continues p2_l/0040 — as the row says — **in two arms that differ
only in the restart**:

**H_restart: from p2_l/0040, twenty iterations that restart the optimizer and the seed streams end
stronger at skill 3 than twenty iterations that resume them.**

## Contract

| | |
|---|---|
| Arms | **`p2_m_init`** (the standing protocol, restarted): `--config configs/p2_k_lr.yaml --init runs/p2_l/ckpt_iter0040.pt --seed-pool runs/p2_l --seed-replay runs/p2_l --out runs/p2_m_init --iterations 20` → iterations 1–20. **`p2_m_resume`** (no restart): `runs/p2_m_resume` is a copy of p2_l's training state (below), continued by `--resume runs/p2_m_resume --iterations 20` → iterations 41–60 |
| What differs between the arms | **(1)** AdamW: fresh (step 0, zero moments) against p2_l's (step 16,000, its moments, loaded from `ckpt_iter0040.pt`); **(2)** seed streams: iterations 1–20 against 41–60 (`iteration_seed(seed, iteration, stream)`); **(3)** the restart's league: `--seed-pool` copies p2_l's 0021–0040 and the new run adds its own iteration 0 (p2_l/0040's weights again), so its first pool holds 0022–0040 plus 0040 twice, against the resumed 0021–0040; **(4)** the restart's ring rows are relabelled by −40 (labels only). Nothing else in training reads the iteration number. All four are what `--init` does; the round tests them as one, the restart |
| What does not differ | the config (`configs/p2_k_lr.yaml`, fingerprint `8ab654e2…298c`, lr 2e-5), the weights (p2_l/0040), the ring's contents (the newest 500,000 rows of p2_l's — its whole ring), `seed` 42, the code, the image and the GPU class |
| The resume arm's directory | `runs/p2_l`'s `state.json`, `config.yaml`, `provenance.json`, `README.md`, `iterations.jsonl` and `replay/` copied to `runs/p2_m_resume`, and its pool checkpoints `ckpt_iter0021`–`0040` hard-linked (never rewritten: the resume writes 0041–0060; pulls replace files by rename). `runs/p2_l` itself is not touched. `--resume` appends the launch SHA to `state.json`'s `resumed_with` and does not rewrite `provenance.json`, so the pod's GPU is recorded at staging (`nvidia-smi`) |
| Length | 20 iterations per arm. Both prior rounds' gains were in place by 10; 20 gives a second point at the cost of p2_l's one round |
| Start reference | **skill 3:** p2_l/0040's six book reps (0.575, 0.585, 0.57, 0.5675, 0.6125, 0.62; mean 0.588; `runs/p2_l/sf3/reps_iter0040_r{1..6}.jsonl`). **skill 0 (continuity):** its three book reps (0.9575, 0.9275, 0.9275; `runs/p2_l/reps_iter0040_r{1,2,3}.jsonl`) and three initial-position reps (`runs/p2_l/startpos/`). Its ladder row at 40: mate-in-2 189, ECE 0.0914 |
| Training | RunPod, **two pods in parallel**, each a secure RTX 4090 with ≥ 24 vCPU, the image of p2_g–p2_l; variants recorded. If none is offered, the registered launch is retried; nothing launches on another class without Michael's OK, by addendum. Code from a bundle of clean, pushed `main` |
| Measurement | this box, torch 2.14.1+rocm7.2, under `gpu hold`, nothing until each arm's final pull. One `cloud/remote_campaign.sh` per arm with `MEASURE_AFTER_PULL=1` (judged `10 20` for init, `50 60` for resume): the ladder pass (ECE, N) and three skill-0 book reps; at the final checkpoint three skill-0 reps from the initial position. Then a finisher: per arm, **three book reps against `stockfish:3` at the first judged checkpoint and six at the last** (`runs/p2_m_{init,resume}/sf3/`) |
| Lifecycle | four guards; pod deadline **8 h** each; worst case **$7.12 per pod, $14.24 together** at $0.89/h; expected about **$7.7** together (20 iterations at p2_l's 10.7 min = 3.6 h per pod, plus boot, staging and the final pull) |
| Disk | this box had 9.8 GB free at drafting (98% used). The round adds about 4.2 GB (each arm's twenty new checkpoints and ring; the resume arm's pool is hard-linked) |

## Registered expectations

### L — the continuity applied
- **L1, init:** `provenance.json` `config_fingerprint` `8ab654e2…`; `init_from`
  `runs/p2_l/ckpt_iter0040.pt`; `seed_replay.rows` 500,000, `donor_committed` = `donor_head` =
  2,644,005, `iteration_offset` −40; `git_dirty` false; `lr` 2e-5 on every row.
- **L1, resume:** `iterations.jsonl` rows 1–40 equal to `runs/p2_l`'s as parsed (`--resume` rewrites
  the log through `json.dumps`), then rows 41–60;
  `state.json` `resumed_with` = [the launch SHA]; `ckpt_iter0041.pt`'s `step` 16,400 (p2_l ended at
  16,000), so the optimizer was carried; the resolved config's fingerprint `8ab654e2…`; `lr` 2e-5 on rows
  41–60.
- **L2.** Book positions in each arm's final ring: reported, no bound.

### K1 (primary) — the restart against the resume at the endpoint, skill 3
`p2_m_init/0020` − `p2_m_resume/0060`, from the frozen book against `stockfish:3`, **six reps per arm**,
per-opening rep-means, paired bootstrap over the 100 openings (`scripts/eval_reps.py verdict`, 10,000
resamples, seed 0; `runs/p2_m_init/sf3/p2m_verdict_init_vs_resume.json`). **Expected: the 95% CI
entirely above zero.** Forecast point +0.02 to +0.06. Arm-against-arm SE at six reps was 0.012–0.016
(p2_i, p2_k), so a difference under about +0.03 will likely read "includes zero".

### K2 — the same at the first judged checkpoint (where the burst would show)
`p2_m_init/0010` − `p2_m_resume/0050`, three reps per arm, the same statistic
(`runs/p2_m_init/sf3/p2m_verdict_init_vs_resume_iter10.json`). Informational; expected above zero if
H_restart holds, and larger relative to its SE than at 20 if the burst fades.

### K3 — each arm against the start (informational)
Each arm at both judged checkpoints against p2_l/0040's six skill-3 reps (three reps; six at the last).
**Expected:** the init arm above the start by 10, as p2_k_lr and p2_l were; the resume arm within noise
of it. K3 decides the second and third decision rows.

### K4 — skill 0 and the initial position (continuity)
The three skill-0 reps per judged checkpoint against p2_l/0040's three, and the initial-position reps at
the last; reported. Skill 0 is at its ceiling (0.93); no row turns on it.

### N, ECE, H
- **N:** mate-in-2 ≥ 183/200 at the last judged checkpoint, **both arms**.
- **ECE:** < 0.15 at the last judged checkpoint, both arms, the unchanged measurement; a break is handled
  in PREREG-p2_l's order (the calibration sample first).
- **H1–H7** as PREREG-p2_l, both arms, direction-free. **H8 (new, direction-free):** the training losses
  over each arm's first five iterations against its last five — a fresh optimizer's burst would show as a
  move in the init arm's first iterations that the resume arm does not have. **H2:** `skipped_steps` < 10
  per arm.

## Predictions (written before launch)
- K1 passes, point +0.02 to +0.06.
- K2 above zero; K3: init above the start at 10, resume within noise of the start at 50 and 60.
- N 183–192 and ECE 0.07–0.11 in both arms.
- H8: the init arm's policy and value losses move more over its first five iterations than the resume
  arm's over its first five.

## Decision rules

| K1 (init − resume at 20/60) | K3 | Decision |
|---|---|---|
| **CI entirely above zero** | any | **The restart is a lever.** The standing protocol keeps `--init`; p2_n asks how often to restart (a ten-iteration cadence against twenty) from `p2_m_init/0020` |
| **CI includes zero** | either arm above the start at its last checkpoint | **The restart is not the lever at this length.** The standing protocol is unchanged (restart, the simpler record); p2_n continues `p2_m_init/0020` at 2e-5 as the default round, and the early-gain shape is read again there |
| **CI includes zero** | both arms within noise of the start | **Plateau at 2e-5.** p2_n is the next step-down (2e-6) as an A/B from `p2_m_init/0020` against a continuation arm |
| **CI entirely below zero** | any | **The resume is better.** The standing protocol becomes `--resume` between rounds; p2_n continues `p2_m_resume/0060` by `--resume` |
| N < 183 or ECE ≥ 0.15 in either arm | | The broken guard first (ECE in PREREG-p2_l's order); the round's reading is not taken as standing until it is resolved |

## Measurement protocol

```bash
# This box, before launch: the resume arm's directory (p2_l untouched)
mkdir -p runs/p2_m_resume
cp -al runs/p2_l/ckpt_iter00{21..40}.pt runs/p2_m_resume/
cp -a runs/p2_l/state.json runs/p2_l/config.yaml runs/p2_l/provenance.json runs/p2_l/README.md \
      runs/p2_l/iterations.jsonl runs/p2_m_resume/
cp -a runs/p2_l/replay runs/p2_m_resume/replay

# Two pods (launch, wait-ssh, stage; inputs' sha256 checked on each pod; nvidia-smi recorded)
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2mi --vcpu 24 --ram 32 --disk 60 --yes
python scripts/runpod_pod.py launch --gpu "RTX 4090" --cloud SECURE --hours 8 --tag p2mr --vcpu 24 --ram 32 --disk 60 --yes
bash cloud/stage_pod.sh POD_I runs/p2_l/ckpt_iter00{21..40}.pt runs/p2_l/state.json runs/p2_l/config.yaml \
    runs/p2_l/replay/ runs/data/mate1_100k_clean/ runs/data/mate2_50k_clean/ runs/phase1_anchor.pt
bash cloud/stage_pod.sh POD_R runs/p2_m_resume/ckpt_iter00{21..40}.pt runs/p2_m_resume/state.json \
    runs/p2_m_resume/config.yaml runs/p2_m_resume/provenance.json runs/p2_m_resume/README.md \
    runs/p2_m_resume/iterations.jsonl runs/p2_m_resume/replay/ runs/data/mate1_100k_clean/ \
    runs/data/mate2_50k_clean/ runs/phase1_anchor.pt

# On each pod (ssh redirects all three streams, so it returns)
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --config configs/p2_k_lr.yaml \
   --init runs/p2_l/ckpt_iter0040.pt --seed-pool runs/p2_l --seed-replay runs/p2_l \
   --out runs/p2_m_init --iterations 20" > /dev/null 2>&1 < /dev/null &
nohup setsid bash /root/taz/pod_job.sh /root/taz/work 14400 -- bash -c \
  "cd /root/chess && python3 scripts/train_phase2.py --resume runs/p2_m_resume --iterations 20" \
  > /dev/null 2>&1 < /dev/null &

# This box, one driver per arm, measuring only after its own final pull
MEASURE_AFTER_PULL=1 START_REPS='runs/p2_l/reps_iter0040_r*.jsonl' START_STARTPOS='runs/p2_l/startpos/reps_iter0040_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD_I p2_m_init "10 20" 20 > /dev/null 2>&1 < /dev/null &
MEASURE_AFTER_PULL=1 START_REPS='runs/p2_l/reps_iter0040_r*.jsonl' START_STARTPOS='runs/p2_l/startpos/reps_iter0040_r*.jsonl' \
setsid nohup bash cloud/remote_campaign.sh POD_R p2_m_resume "50 60" 60 > /dev/null 2>&1 < /dev/null &

# After both campaigns complete (a finisher like runs/p2_k_finish.sh and runs/p2_l_finish.sh): skill 3
for spec in "p2_m_init 0010 0020" "p2_m_resume 0050 0060"; do
  set -- $spec; arm=$1
  for nn in $2 $3; do
    reps=3; [ $nn = $3 ] && reps=6
    gpu hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
      .venv/bin/python scripts/eval_reps.py run --ckpt runs/$arm/ckpt_iter$nn.pt --anchor stockfish:3 --reps $reps --out-dir runs/$arm/sf3
    .venv/bin/python scripts/eval_reps.py verdict --a start runs/p2_l/sf3/reps_iter0040_r*.jsonl \
      --b ${arm}_$nn runs/$arm/sf3/reps_iter${nn}_r{1,2,3}.jsonl --out runs/$arm/sf3/p2_verdict_reps_iter$nn.json
  done
  .venv/bin/python scripts/eval_reps.py verdict --a start runs/p2_l/sf3/reps_iter0040_r*.jsonl \
    --b ${arm}_$3 runs/$arm/sf3/reps_iter$3_r*.jsonl --out runs/$arm/sf3/p2_verdict_reps.json
  .venv/bin/python scripts/ring_book_rows.py runs/$arm --json runs/$arm/book_rows.json
done
.venv/bin/python scripts/eval_reps.py verdict --a resume runs/p2_m_resume/sf3/reps_iter0060_r*.jsonl \
    --b init runs/p2_m_init/sf3/reps_iter0020_r*.jsonl --out runs/p2_m_init/sf3/p2m_verdict_init_vs_resume.json
.venv/bin/python scripts/eval_reps.py verdict --a resume runs/p2_m_resume/sf3/reps_iter0050_r{1,2,3}.jsonl \
    --b init runs/p2_m_init/sf3/reps_iter0010_r{1,2,3}.jsonl --out runs/p2_m_init/sf3/p2m_verdict_init_vs_resume_iter10.json
```

## What would make p2_m uninterpretable
- A difference between the arms beyond the four named (the config is one file; L1 checks both arms).
- The resume arm not resuming: a fresh optimizer in it (L1: `ckpt_iter0041.pt`'s step), or rows 1–40
  not p2_l's.
- A launch from anything but clean, pushed `main`; a measurement-instrument change.
- The arms on different GPU variants: recorded and reported, not a stop.
- Losing either arm's final pull: the round is re-run by addendum, not read from one arm.

**What the statistic does not cover.** As PREREG-p2_k: each arm is one run. And the round tests the
restart as a bundle (four parts); if it is the lever, which part carries it is a later question.

---

## Sign-off

Checked for the draft, 2026-10-31, on this box: p2_l's `state.json` iteration 40, step 16,000, 0 skipped
steps, pool `ckpt_iter0021`–`0040` (all present), `replay_total_written` 2,644,005 = the ring's
`total_written`, `resumed_with` empty; its six skill-3 book reps, three skill-0 book reps and three
initial-position reps present; `--resume` reloads the optimizer from the last checkpoint and continues
the iteration counter (`Phase2Run.resume`), and per-iteration seeds are `iteration_seed(seed, iteration,
stream)`. Read for the draft: RunPod balance $54.64, auto-top-up on (Michael, 2026-10-31), no pods of
ours; secure RTX 4090 at $0.89/h; 8 h × $0.89 = $7.12 per pod passes the $15 ceiling. Price and balance
are read again at registration.

**Not signed.** Registration waits for Michael's OK on the design and its cost — expected about $7.7,
worst case $14.24.
