.PHONY: test lint bench install

UV := $(shell command -v uv 2>/dev/null || echo ~/.local/bin/uv)

install:
	$(UV) pip install -e ".[dev]"

test:
	pytest tests/ -q

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

bench:
	python benchmarks/bench_selfplay.py --games 256 --sims 48 2>/dev/null || true
	python benchmarks/bench_inference.py --device cpu --batches 64 2>/dev/null || true
