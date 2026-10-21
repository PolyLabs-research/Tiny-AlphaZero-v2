.PHONY: test lint bench install golden

UV := $(shell command -v uv 2>/dev/null || echo ~/.local/bin/uv)
# Every Python call goes through the project venv (AGENTS §3): a bare `python`
# is whatever happens to be on PATH, which outside an activated venv has no
# torch and made `make bench` a silent no-op (REVIEW-2026-10-21 E4).
PY := .venv/bin/python
RUFF := .venv/bin/ruff
# Everything that is Python. benchmarks/ and bench_step.py were outside the lint
# scope and accumulated errors nobody saw.
LINT_PATHS := src/ tests/ scripts/ benchmarks/ bench_step.py
# The inference benchmark times a real network; the frozen Phase 1 anchor is the
# one every recorded row used. Override for another checkpoint.
BENCH_CKPT ?= runs/phase1_anchor.pt
# Extra flags passed to both benchmark scripts, e.g. BENCH_ARGS=--no-record for a
# dry run that must not append to benchmarks/results.jsonl.
BENCH_ARGS ?=

install:
	$(UV) pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/ -q

lint:
	$(RUFF) check $(LINT_PATHS)
	$(RUFF) format --check $(LINT_PATHS)

# The golden mini-run (spec §9.4): two real Phase 2 iterations on CPU in under
# three minutes. Separate from `make test` because it is the one target whose
# failure means "the loop is broken", not "a unit is wrong".
golden:
	$(PY) -m pytest tests/test_phase2.py -m golden -q

# Both benchmarks append a SHA- and date-stamped row to benchmarks/results.jsonl
# (AGENTS §5) unless BENCH_ARGS=--no-record. A benchmark that cannot run is a
# finding, not something to hide behind `2>/dev/null || true`.
bench:
	$(PY) benchmarks/bench_selfplay.py --games 256 --sims 48 $(BENCH_ARGS)
	$(PY) benchmarks/bench_inference.py --ckpt $(BENCH_CKPT) --device cpu $(BENCH_ARGS)
