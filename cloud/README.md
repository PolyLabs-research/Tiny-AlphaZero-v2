# Remote runs on RunPod

`scripts/runpod_pod.py` (library: `src/tinyalphazero/cloud.py`). Nothing it starts may
be left running; four guards, each independent of the others:

| guard | where | stops a pod when |
|---|---|---|
| job | pod: `cloud/pod_job.sh` | the job ends and this box has pulled (bounded wait) |
| deadline | pod: its start command | `TAZ_DEADLINE` passes, whatever the job is doing |
| sweeper | this box: `cloud/systemd/taz-cloud-sweep.timer`, every 5 min, lingering user | past deadline + 300 s, or no parseable deadline |
| budget | this box, before launch | rate x hours over the session ceiling or into the balance margin |

Pods are named `taz-<tag>-dl<unix deadline>`; only that prefix is ever touched.
Terminate, never stop; no pod volume; results are pulled back here over SSH.

Setup on this box: API key in `~/.config/runpod/api_key` (0600), SSH key
`~/.ssh/runpod_ed25519` (public half on the RunPod account), the timer installed with
`cp cloud/systemd/* ~/.config/systemd/user/ && systemctl --user enable --now taz-cloud-sweep.timer`.
Limits: `~/.config/taz-cloud/limits.json` (defaults: $10/session, 30 h, $5 margin).
Ledger of every launch and termination: `~/.local/state/taz-cloud/ledger.jsonl`.

## Lifecycle proof, 2026-10-24 (total $0.18)

| pod | guard isolated | result |
|---|---|---|
| `lhemicpao95oij` A5000 secure $0.27/h | job | pulled at 12:46:29, pod gone from the API at 12:46:46, deleted by itself with the injected key |
| `hti9p6t899f5do` A4000 community $0.17/h | deadline | untouched; deadline 13:02:05, gone 13:02:15; no sweeper action logged |
| `evljq2tyqojknf` A4000 secure $0.25/h, watchdog off | sweeper | deadline 12:59:33; the sweeper terminated it at 13:09:38 (604 s past) |
| — | budget | 30 h H100 ($80.70 > $10), 40 h (> 30 h), $68 into the $5 margin: all refused with `--yes`, nothing created |

## What the pods taught us

- **RunPod injects `RUNPOD_API_KEY` and `RUNPOD_POD_ID` into the container's environment**
  (pid 1). SSH sessions do not inherit it; `pod_job.sh` reads the two values from
  `/proc/1/environ`. The pod can therefore delete itself without our key on it.
- **The API's `allowedCudaVersions` filter matched no host** for 12.8+, though the host
  we got ran driver 595.91 / CUDA 13.2. It is opt-in (`--cuda`); the image's own
  `NVIDIA_REQUIRE_CUDA` is what binds.
- **`rest.runpod.io` refuses urllib's default User-Agent** (Cloudflare 1010); every
  client here sends `taz-cloud/1.0`.
- **`nproc` reports the host (96), not the pod**: the cgroup quota was 7.65 CPUs for a
  pod RunPod called 9 vCPU. Actor counts must be set from `RUNPOD_CPU_COUNT` / the
  quota, never from the CPU count.
- **CPU allocation varies by host**: 9, 14 and 25 vCPU on the three pods at $0.17–0.27/h.
- **The CUDA 12.4 image is Ubuntu 20.04, whose apt has Stockfish 11.** The skill-0
  instrument is Stockfish 16 (`16-1build1`, Ubuntu 24.04). Real runs use
  `runpod/pytorch:1.4.1-cu1281-torch2121-ubuntu2404` (the default image), which
  matches the local torch 2.12.1 and gets Stockfish 16 from apt.
- Boot to SSH, including a cold image pull: ~4 min.
