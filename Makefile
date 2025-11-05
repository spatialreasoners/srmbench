.PHONY: help install install-dev test clean build upload

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install the package in development mode with all dependencies
	pip install -e ".[dev,docs]"
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=srmbench --cov-report=term-missing

test-fast: ## Run tests without coverage
	pytest tests/ -v

lint: ## Run linting
	flake8 srmbench tests
	mypy srmbench

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build the package
	python -m build

upload-test: build ## Upload to test PyPI
	twine upload --repository testpypi dist/*

upload: build ## Upload to PyPI
	twine upload dist/*
	
ci: ## Run CI pipeline
	pre-commit run --all-files
	pytest tests/ --cov=srmbench --cov-report=xml
	flake8 srmbench tests
	mypy srmbench

version: ## Show version information
	@python -c "import srmbench; print(f'Version: {srmbench.__version__}')"

info: ## Show package information
	@python -c "import srmbench; print(f'Package: {srmbench.PACKAGE_NAME}'); print(f'Version: {srmbench.__version__}'); print(f'Author: {srmbench.AUTHOR}')"
