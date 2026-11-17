.PHONY: help install test test-verbose check lint fmt eval demo clean

help:
	@echo "install       pip install -e '.[dev]'"
	@echo "test          run the full unittest suite (no network, no keys)"
	@echo "test-verbose  same, one line per test"
	@echo "lint          ruff check (optional)"
	@echo "fmt           ruff format (optional)"
	@echo "check         byte-compile every module (works without ruff)"
	@echo "eval          run the 11-task eval suite offline (gated at 95%)"
	@echo "demo          end-to-end offline demo of the whole pipeline"
	@echo "clean         remove caches and runtime state"

install:
	python -m pip install -e ".[dev]"

test:
	python -m unittest discover -s tests -q

test-verbose:
	python -m unittest discover -s tests -v

lint:
	ruff check apex tests

fmt:
	ruff format apex tests

check:
	python -m compileall -q apex tests examples

eval:
	python -m apex --profile offline eval --min-pass-rate 0.95

demo:
	python examples/demo.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .apex .ruff_cache .pytest_cache build dist *.egg-info
