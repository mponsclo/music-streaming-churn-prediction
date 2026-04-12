.PHONY: help install install-dev dbt-build dbt-run dbt-test dbt-docs train eval eval-temporal test lint format clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Install ---

install: ## Install runtime dependencies
	pip install -r requirements.txt

install-dev: ## Install dev dependencies (pytest, ruff, pre-commit)
	pip install -r requirements-dev.txt
	pre-commit install

# --- Feature Pipeline (dbt + DuckDB) ---

dbt-build: ## Run full dbt pipeline (run + test)
	dbt build --profiles-dir .

dbt-run: ## Run dbt models only
	dbt run --profiles-dir .

dbt-test: ## Run dbt schema tests
	dbt test --profiles-dir .

dbt-docs: ## Generate and serve dbt documentation (http://localhost:8080)
	dbt docs generate --profiles-dir . && dbt docs serve --profiles-dir . --port 8080

# --- Modeling ---

train: ## Train baseline + LightGBM with Optuna (random split)
	python src/modeling.py

eval: ## Generate SHAP + ROC/PR plots for trained model
	python src/evaluate.py

eval-temporal: ## Temporal holdout: train on Round 1, test on Round 2
	python src/temporal_eval.py

# --- Testing & Quality ---

test: ## Run pytest test suite
	pytest tests/ -v

lint: ## Ruff check + format check
	ruff check .
	ruff format --check .

format: ## Auto-format code (ruff fix + format)
	ruff check --fix .
	ruff format .

# --- Cleanup ---

clean: ## Remove build artifacts and caches
	rm -rf target/ logs/ dbt_packages/
	rm -rf .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
