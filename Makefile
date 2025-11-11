.PHONY: help install test clean build

help: ## Show available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package in development mode with dev dependencies
	pip install -e ".[dev]"

test: ## Run tests with coverage
	pytest

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/ coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build source and wheel distributions
	python -m build

version: ## Show package version
	@python -c "import srmbench; print(f'srmbench v{srmbench.__version__}')"

.DEFAULT_GOAL := help
