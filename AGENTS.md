# AGENTS.md — System & Build Conventions

Drop-in briefing for coding agents working on this machine. Read fully before writing code, installing packages, or running anything GPU-related. When these conventions conflict with an agent's defaults, these win.

---

## 1. The machine

| Item | Value |
|------|-------|
| Box | GMKtec NucBox EVO-X2 (mini PC) |
| APU | AMD Ryzen AI Max+ 395 "Strix Halo" |
| GPU | Integrated Radeon 8060S — **gfx1151**, RDNA 3.5 |
| Memory | 128GB total — **96GB dedicated to the GPU** (BIOS UMA carveout), **~32GB visible to the OS** for CPU-side work; ~8.6GB swap |
| OS | Ubuntu (Linux), ROCm 7.x userspace |
| GPU stack | **ROCm — there is no NVIDIA hardware here. CUDA wheels will never work.** |

Implications you must internalize:

- PyTorch on ROCm masquerades as CUDA: `torch.cuda.is_available()`, `device="cuda"`, `torch.cuda.get_device_name()` are the correct API calls and should return True / "AMD Radeon(TM) 8060S Graphics". Do not write ROCm-specific device strings.
- `nvidia-smi` does not exist. Use `rocminfo | grep gfx` (expect `gfx1151`) and `rocm-smi` for utilization.
- **The two memory pools have opposite pressure profiles.** GPU-side: 96GB dedicated VRAM — enormous for this class of hardware; huge inference batches and large resident models are cheap, though a running local-LLM server (LM Studio) can be holding most of it — check `rocm-smi` before assuming VRAM is free. Host-side: only ~32GB, of which the OS, browser, and file cache routinely consume half — **host RAM is the scarce resource on this box**. Size replay buffers, dataset loads, worker pools, and pinned staging tensors against ~15GB of realistically free host memory, not against "128GB". Prefer streaming/memmap over loading datasets into host RAM; swap is small (8.6GB), so an OOM-adjacent host allocation will thrash or die rather than degrade gracefully.
- gfx1151 support is recent (ROCm 7.1+). If GPU code misbehaves, suspect the wheel/stack version before suspecting the code. History on this box: the `amdgpu` driver was once blacklisted and silently forced CPU-only operation — if the GPU vanishes, check `lsmod | grep amdgpu` and `/etc/modprobe.d/` before anything else.

## 2. PyTorch installation — the #1 recurring failure

A bare `pip install torch` pulls CUDA wheels from PyPI, which import fine and then run CPU-only forever. This has burned real time more than once. Rules:

1. **Never install torch without an explicit ROCm index.**
   ```bash
   uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
   ```
   (Use the newest rocmX.Y index that pytorch.org offers; it must be ≥ rocm7.1 for gfx1151. AMD's own index `https://repo.amd.com/rocm/whl/gfx1151/` is the fallback if the official wheel misbehaves.)

2. **Every project that depends on torch must pin the index in `pyproject.toml`** so resolution can never drift back to CUDA:
   ```toml
   [tool.uv.sources]
   torch = { index = "pytorch-rocm" }

   [[tool.uv.index]]
   name = "pytorch-rocm"
   url = "https://download.pytorch.org/whl/rocm7.2"
   explicit = true
   ```

3. **Verify after any environment change**, and make projects self-verifying with a `scripts/check_env.py` that prints torch version, device, and a matmul TFLOPS number:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-ONLY')"
   ```
   A `+cuXXX` version suffix on this machine is always wrong; expect `+rocm7.x`. If an env is confused, delete `.venv` and rebuild — don't perform surgery on it.

4. No `HSA_OVERRIDE_GFX_VERSION` hacks. On ROCm 7.x, gfx1151 is natively supported; needing the override means an old wheel snuck in.

5. GPU numerics: bf16 autocast is the default precision for training/inference; fp32 master weights. `torch.compile` is opt-in behind a flag, default off, and only enabled after benchmarking eager vs compiled on this box.

## 3. Python tooling

- **`uv` for everything**: `uv venv`, `uv pip`, `uv lock`, `uv run`. Not conda, not poetry, not bare pip. Python 3.12.
- Projects are installable packages: `src/` layout, `pyproject.toml`, installed with `uv pip install -e ".[dev]"`. **`sys.path` mutation is banned** — if an import needs a path hack, the packaging is wrong; fix the packaging.
- Lint/format with `ruff` (check + format) before every commit. Type-hint all public interfaces.
- Tests: `pytest`, with `hypothesis` for property-based testing of codecs, invariants, and round-trips. All tests must pass on **CPU only** — GPU availability is never a test dependency. GPU-dependent checks live in benchmark scripts, not the test suite.
- Provide a `Makefile` with at least: `make test`, `make lint`, `make bench`.

## 4. Project conventions

- **One config system**: a nested dataclass tree, loadable from YAML, with a `config_fingerprint()` (sha256 of canonical YAML). CLI scripts are thin argparse wrappers that build a config and call library code. No config values duplicated between CLI defaults and code.
- **Run artifacts** live in `runs/<name>/` (gitignored, along with `data/`, `*.pt`, `__pycache__`, `.venv`). Every run directory gets the resolved `config.yaml`, the git SHA, and the `check_env.py` output.
- **Checkpoints embed their provenance**: model state, optimizer state, full config dict, git SHA, step count. A checkpoint that can't reproduce its own configuration is a bug.
- **Logging is JSONL**, one row per event/iteration, written into the run directory. Dashboards and analysis are pure readers of these files with zero imports from training code.
- **Determinism**: one seed in config fans out to all RNGs; library code takes explicit `rng`/`Generator` parameters, never global random state.
- **Large data**: prefer `numpy.memmap` binary layouts with a small `meta.json` sidecar over JSON/pickle/HDF5. Zero-copy and restartability beat convenience formats — and on this box memmap is a genuine memory-pressure tool, not just a convenience: host RAM is ~32GB, so multi-GB datasets must be memory-mapped or streamed, never fully loaded (`np.load`, `pd.read_*`, whole-file `json.load` on GB-scale files are all suspect).
- Library code raises; only `scripts/` prints and exits. No `print()` in `src/`.

## 5. How to build for me — working style

- **Work in chunks with objective gates.** Large builds are executed as an ordered chunk plan where each chunk ends in a DONE-WHEN block of shell commands. A chunk is done when those commands pass — not when the code "looks right". One commit per chunk: `chunk NN: <summary>`.
- **Never weaken a gate to pass it.** If a threshold or test fails after 3 distinct fix attempts, stop and write `BLOCKED.md` (failure, hypotheses tested, evidence). A relaxed acceptance criterion is a silent lie I will discover weeks later; a BLOCKED.md is useful information.
- **Interfaces are contracts.** Once a chunk's public interface (signatures, array layouts, file formats, constants) is established, don't change it without being asked — later chunks and other sessions depend on it.
- **Correctness before speed, and profile before optimizing.** Benchmark scripts append to `benchmarks/results.jsonl` with the git SHA. Never trade a correctness test for throughput; never optimize a bottleneck you haven't measured (`py-spy` for Python profiling).
- **Bias to minimal.** Small models, few dependencies, arrays over object graphs, one implementation of each concept. If a component exists twice, that's a defect. New dependencies need a reason; "it's popular" isn't one.
- **Prefer explicit structure in explanations**: precise interfaces, tables, worked numeric examples, and named invariants over visual metaphors or "imagine..." descriptions. State the structural relationship, don't gesture at it.
- Ask before: deleting data, force-pushing, changing an established interface, adding a heavyweight dependency, or anything touching system-level config (drivers, kernel modules, `/etc`). Everything else: proceed and report.

## 6. Domain defaults (ML projects on this box)

- Training runs are single-box, single-GPU. No distributed frameworks, no cloud assumptions.
- Throughput architecture pattern: many CPU-side actors/workers feeding **one** process that owns the GPU with large batched forwards. Per-worker model copies fighting over the GPU is an anti-pattern here.
- The GPU being underutilized while CPU-bound Python saturates is the *expected* profile for small-model RL/search workloads — confirm with profiling before "fixing" it.
- Long runs must be resumable: killing the process at any point and restarting from the run directory must not corrupt state. Write a crash-resume test for anything that runs for days.

## 7. Quick reference card

```bash
# New project bootstrap
uv venv --python 3.12 && uv pip install -e ".[dev]"

# Torch (ROCm, never bare)
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

# Sanity
python scripts/check_env.py            # expect Device: cuda, gfx1151 name, +rocm version
rocminfo | grep gfx                    # expect gfx1151
rocm-smi                               # GPU util / VRAM (96GB pool; check what LM Studio holds)
free -h                                # host RAM (~32GB pool — the scarce one)

# Hygiene before commit
make lint && make test
```
