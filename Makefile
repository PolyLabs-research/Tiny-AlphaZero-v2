.PHONY: test lint bench install golden

UV := $(shell command -v uv 2>/dev/null || echo ~/.local/bin/uv)

install:
	$(UV) pip install -e ".[dev]"

test:
	pytest tests/ -q

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

# The golden mini-run (spec §9.4): two real Phase 2 iterations on CPU in under
# three minutes. Separate from `make test` because it is the one target whose
# failure means "the loop is broken", not "a unit is wrong".
golden:
	pytest tests/test_phase2.py -m golden -q

bench:
	python benchmarks/bench_selfplay.py --games 256 --sims 48 2>/dev/null || true
	python benchmarks/bench_inference.py --device cpu --batches 64 2>/dev/null || true
