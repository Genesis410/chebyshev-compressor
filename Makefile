.PHONY: all setup test smoke bench clean lint docker-test

PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest

all: test

setup:
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e ".[dev]" --no-build-isolation

test:
	$(PYTEST) tests/ -v

smoke:
	$(PYTHON) -m chebyshev.cli --smoke-test --seed 42

bench:
	$(PYTHON) -m chebyshev.cli benchmark

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ __pycache__/ tests/__pycache__/ chebyshev/__pycache__/
	rm -f bench_harness bench_harness.exe

docker-test:
	docker compose run --rm test
