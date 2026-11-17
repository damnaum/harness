.PHONY: help install test test-verbose lint fmt eval demo clean

help:
	@echo "install       pip install -e '.[dev]'"
	@echo "test          run the full unittest suite (no network, no keys)"
	@echo "test-verbose  same, one line per test"
	@echo "lint          ruff check"
	@echo "fmt           ruff format"
	@echo "eval          run the 8-task eval suite offline"
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

eval:
	python -m apex --profile offline eval

demo:
	python examples/demo.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .apex .ruff_cache .pytest_cache build dist *.egg-info
