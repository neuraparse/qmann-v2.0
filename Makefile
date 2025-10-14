# QMANN - Quantum Memory-Augmented Neural Networks
# Makefile for development, testing, and deployment

.PHONY: help install install-dev test test-quantum test-classical test-integration lint format type-check security clean build docker docs deploy

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
MYPY := mypy
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Project directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
EXAMPLES_DIR := examples

# Docker images
DEV_IMAGE := qmann:dev
PROD_IMAGE := qmann:prod
QUANTUM_IMAGE := qmann:quantum

# Help target
help: ## Show this help message
	@echo "QMANN - Quantum Memory-Augmented Neural Networks"
	@echo "================================================"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install the package
	$(PIP) install .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev,quantum]"
	pre-commit install

install-quantum: ## Install quantum computing dependencies
	$(PIP) install -e ".[quantum]"

install-all: ## Install all dependencies
	$(PIP) install -e ".[dev,quantum,visualization,optimization]"

# Testing targets
test: ## Run all tests
	$(PYTEST) $(TEST_DIR) -v --tb=short

test-unit: ## Run unit tests only
	$(PYTEST) $(TEST_DIR)/test_unit/ -v

test-integration: ## Run integration tests
	$(PYTEST) $(TEST_DIR)/test_integration/ -v

test-quantum: ## Run quantum-specific tests
	$(PYTEST) $(TEST_DIR) -k "quantum" -v

test-classical: ## Run classical ML tests
	$(PYTEST) $(TEST_DIR) -k "classical" -v

test-applications: ## Run application tests
	$(PYTEST) $(TEST_DIR) -k "application" -v

test-coverage: ## Run tests with coverage report
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR)/qmann --cov-report=html --cov-report=term-missing

test-performance: ## Run performance benchmarks
	$(PYTHON) -m pytest $(TEST_DIR)/test_performance/ -v --benchmark-only

# Code quality targets
lint: ## Run linting checks
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

format: ## Format code with Black
	$(BLACK) $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

format-check: ## Check code formatting
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

type-check: ## Run type checking with MyPy
	$(MYPY) $(SRC_DIR)/qmann

security: ## Run security checks
	bandit -r $(SRC_DIR)
	safety check

quality: lint format-check type-check security ## Run all code quality checks

# Development targets
dev-setup: install-dev ## Set up development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

dev-test: ## Run development tests (fast subset)
	$(PYTEST) $(TEST_DIR) -v -x --tb=short -k "not slow"

dev-quantum: ## Start quantum development environment
	$(DOCKER_COMPOSE) up qmann-dev

dev-jupyter: ## Start Jupyter Lab for development
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Build targets
build: ## Build the package
	$(PYTHON) -m build

build-wheel: ## Build wheel distribution
	$(PYTHON) -m build --wheel

build-sdist: ## Build source distribution
	$(PYTHON) -m build --sdist

# Docker targets
docker-build: ## Build all Docker images
	$(DOCKER) build -t $(DEV_IMAGE) --target development .
	$(DOCKER) build -t $(PROD_IMAGE) --target production .
	$(DOCKER) build -t $(QUANTUM_IMAGE) --target quantum-sim .

docker-dev: ## Start development environment with Docker
	$(DOCKER_COMPOSE) up qmann-dev

docker-prod: ## Start production environment with Docker
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml up

docker-test: ## Run tests in Docker
	$(DOCKER_COMPOSE) run --rm qmann-test

docker-quantum: ## Run quantum simulations in Docker
	$(DOCKER_COMPOSE) up qmann-quantum

docker-clean: ## Clean Docker images and containers
	$(DOCKER) system prune -f
	$(DOCKER) image prune -f

# Documentation targets
docs: ## Build documentation
	cd $(DOCS_DIR) && make html

docs-serve: ## Serve documentation locally
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	cd $(DOCS_DIR) && make clean

# Example targets
examples-healthcare: ## Run healthcare demo
	$(PYTHON) $(EXAMPLES_DIR)/healthcare_demo.py

examples-industrial: ## Run industrial maintenance demo
	$(PYTHON) $(EXAMPLES_DIR)/industrial_demo.py

examples-autonomous: ## Run autonomous systems demo
	$(PYTHON) $(EXAMPLES_DIR)/autonomous_demo.py

examples-all: examples-healthcare examples-industrial examples-autonomous ## Run all examples

# Quantum-specific targets
quantum-setup: ## Set up quantum computing environment
	@echo "Setting up quantum computing environment..."
	@echo "Please ensure you have IBM Quantum account and token"
	@echo "Set QISKIT_IBM_TOKEN environment variable"

quantum-test-real: ## Test on real quantum hardware (requires IBM Quantum access)
	$(PYTEST) $(TEST_DIR) -k "real_hardware" -v --tb=short

quantum-benchmark: ## Run quantum vs classical benchmarks
	$(PYTHON) -m qmann.utils.benchmarks

# Deployment targets
deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging..."
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.staging.yml up -d

deploy-prod: ## Deploy to production environment
	@echo "Deploying to production..."
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml up -d

deploy-stop: ## Stop deployment
	$(DOCKER_COMPOSE) down

# Monitoring targets
monitor-start: ## Start monitoring stack
	$(DOCKER_COMPOSE) up prometheus grafana -d

monitor-stop: ## Stop monitoring stack
	$(DOCKER_COMPOSE) stop prometheus grafana

logs: ## View application logs
	$(DOCKER_COMPOSE) logs -f qmann-prod

# Database targets
db-start: ## Start database services
	$(DOCKER_COMPOSE) up postgres redis -d

db-stop: ## Stop database services
	$(DOCKER_COMPOSE) stop postgres redis

db-reset: ## Reset database (WARNING: destroys data)
	$(DOCKER_COMPOSE) down -v postgres redis
	$(DOCKER_COMPOSE) up postgres redis -d

# Cleanup targets
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-cache: ## Clean quantum and ML caches
	rm -rf quantum_cache/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/

clean-all: clean clean-cache docker-clean ## Clean everything

# Release targets
version: ## Show current version
	$(PYTHON) -c "import qmann; print(qmann.__version__)"

release-check: quality test ## Check if ready for release
	@echo "Release readiness check complete!"

release-build: clean build ## Build release packages
	@echo "Release packages built successfully!"

# CI/CD targets
ci-install: ## Install dependencies for CI
	$(PIP) install -e ".[dev,quantum]"

ci-test: ## Run CI test suite
	$(PYTEST) $(TEST_DIR) -v --tb=short --cov=$(SRC_DIR)/qmann --cov-report=xml

ci-quality: ## Run CI quality checks
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	$(MYPY) $(SRC_DIR)/qmann

ci-security: ## Run CI security checks
	bandit -r $(SRC_DIR) -f json -o bandit-report.json
	safety check --json --output safety-report.json

# Performance targets
benchmark: ## Run performance benchmarks
	$(PYTHON) -m qmann.utils.benchmarks --output benchmarks.json

profile: ## Profile application performance
	$(PYTHON) -m cProfile -o profile.stats -m qmann.applications.healthcare
	$(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Research targets
research-setup: ## Set up research environment
	$(PIP) install -e ".[research]"
	jupyter labextension install @jupyter-widgets/jupyterlab-manager

research-notebooks: ## Start research notebooks
	jupyter lab --notebook-dir=notebooks

# Utility targets
check-deps: ## Check for dependency updates
	pip list --outdated

update-deps: ## Update dependencies (use with caution)
	pip install --upgrade -r requirements.txt

info: ## Show project information
	@echo "QMANN - Quantum Memory-Augmented Neural Networks"
	@echo "Version: $$($(PYTHON) -c 'import qmann; print(qmann.__version__)')"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Qiskit: $$($(PYTHON) -c 'import qiskit; print(qiskit.__version__)')"
	@echo "PyTorch: $$($(PYTHON) -c 'import torch; print(torch.__version__)')"
	@echo "CUDA Available: $$($(PYTHON) -c 'import torch; print(torch.cuda.is_available())')"

# Quick start target
quickstart: install-dev dev-test examples-healthcare ## Quick start for new developers
	@echo ""
	@echo "🎉 QMANN quickstart complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Set up IBM Quantum token: export QISKIT_IBM_TOKEN='your_token'"
	@echo "2. Run quantum tests: make test-quantum"
	@echo "3. Start development: make dev-jupyter"
	@echo "4. Read documentation: make docs-serve"
