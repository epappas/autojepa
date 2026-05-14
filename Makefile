# Common workflows for autojepa. All targets are thin wrappers
# around `uv run` invocations — nothing here changes behavior, just ergonomics.
#
# Quick reference:
#   make test                  # full test suite
#   make test-fast             # skip the slow integration tests
#   make lint                  # ruff
#   make typecheck             # mypy
#   make check                 # test + lint + typecheck
#   make validate CONFIG=path  # run config_validate against any yaml
#   make smoke                 # in-tree examples smoke tests (post-Phase-2)
#   make help                  # this list

.DEFAULT_GOAL := help
.PHONY: help test test-fast lint typecheck check validate sync real-llm smoke

help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[1m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install runtime + dev dependencies via uv
	uv sync --extra dev

test: ## Run the full pytest suite (~95 s — includes showcase determinism)
	uv run pytest -q

test-fast: ## Run pytest excluding showcase determinism (~30 s)
	uv run pytest -q --ignore=tests/test_showcase_determinism.py

lint: ## Ruff lint check
	uv run ruff check src/ tests/

typecheck: ## Mypy on src/
	uv run mypy src/

check: lint typecheck test ## All three: lint + typecheck + tests

# NOTE: showcase / smoke targets pointing at examples/* are inherited from
# autoresearch-rl and will be re-pointed to AutoJEPA examples in Phase 2.
# They will fail until examples/ijepa-cifar10 lands.

validate: ## Validate a config: make validate CONFIG=path/to/config.yaml
	@if [ -z "$(CONFIG)" ]; then echo "usage: make validate CONFIG=path/to/config.yaml"; exit 2; fi
	uv run autojepa validate $(CONFIG)

real-llm: ## Run real-LLM prompt validation against Kimi K2.6 (needs MOONSHOT_API_KEY)
	@if [ -z "$(MOONSHOT_API_KEY)" ]; then echo "usage: MOONSHOT_API_KEY=sk-... make real-llm"; exit 2; fi
	uv run pytest tests/eval/test_real_llm.py -v

smoke: ## End-to-end smoke tests for in-tree CPU examples (~5 s/example)
	uv run pytest tests/test_examples_smoke.py -v
