PYTHON ?= python3
VENV ?= .venv
PIP ?= $(VENV)/bin/pip
PY ?= $(VENV)/bin/python

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Commands:"
	@echo "  setup-dev       : Install dependencies and set up pre-commit hooks."
	@echo "  lint            : Run linters and type checkers."
	@echo "  lint-fix        : Run all pre-commit hooks to format and lint the codebase."
	@echo "  test            : Run tests."
	@echo "  test-cov        : Run tests and generate a coverage report."
	@echo "  docker-build    : Build the Docker image."
	@echo "  clean           : Clean up build artifacts."

.PHONY: setup
setup: ## Create environment and install dependencies with Poetry
	@echo "Setting up development environment with Poetry..."
	poetry install --with dev --with cloud
	poetry run pre-commit install

.PHONY: setup-pip
setup-pip: ## This target is now an alias for setup
	@$(MAKE) setup

.PHONY: setup-dev
setup-dev:
	poetry install
	poetry run pre-commit install

.PHONY: lint
lint:
	poetry run ruff check src tests
	poetry run mypy src
	poetry run black --check src tests

.PHONY: format
format: ## Format code with black and ruff
	poetry run ruff check --fix src tests
	poetry run black src tests

.PHONY: type-check
type-check: ## Run mypy type checking
	mypy src

.PHONY: test
test:
	poetry run pytest --cov=src --cov-report=term-missing --cov-fail-under=80

.PHONY: test-cov
test-cov: ## Run pytest with coverage
	poetry run pytest --cov=cytoflow_qc --cov-report=html --cov-report=term

.PHONY: test-integration
test-integration: ## Run integration tests only
	poetry run pytest -v -m integration

.PHONY: test-benchmark
test-benchmark: ## Run performance benchmarks
	poetry run pytest -v -m benchmark --benchmark-only

.PHONY: test-property
test-property: ## Run property-based tests
	poetry run pytest -v -m property

.PHONY: test-fast
test-fast: ## Run tests quickly (skip slow tests)
	poetry run pytest -v -m "not integration and not benchmark"

.PHONY: build
build: ## Build package (sdist/wheel)
	$(PY) -m build

.PHONY: clean
clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.PHONY: smoke
smoke: ## Run minimal smoke test
	cytoflow-qc run --samplesheet samplesheets/example_samplesheet.csv --config configs/example_config.yaml --out results/smoke

.PHONY: report
report: ## Build HTML report from results
	cytoflow-qc report --in results --template configs/report_template.html.j2 --out results/report.html

.PHONY: docker-build
docker-build: ## Build Docker image
	docker build -t cytoflow-qc .

.PHONY: docker-run
docker-run: ## Run Docker container
	docker-compose up

.PHONY: dashboard
dashboard: ## Launch interactive web dashboard
	cytoflow-qc dashboard --indir results/

.PHONY: api
api: ## Start REST API server
	python -m cytoflow_qc.api

.PHONY: serve
serve: ## Start both dashboard and API
	@echo "Starting cytoflow-qc services..."
	@echo "Dashboard: http://localhost:8501"
	@echo "API: http://localhost:8000"
	@echo "Press Ctrl+C to stop"
	@trap 'kill 0' INT; python -m cytoflow_qc.api & cytoflow-qc dashboard --indir results/ --port 8501

.PHONY: lint-fix
lint-fix:
	poetry run pre-commit run --all-files