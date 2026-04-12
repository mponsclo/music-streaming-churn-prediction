# CLAUDE.md

Internal project memory for AI-assisted development. Not published as part of the README, but kept in the repo so collaborators get the same context.

## Stack

- Python 3.13
- DuckDB for CSV ingestion and heavy aggregations (streams large files)
- dbt-duckdb for the feature engineering layer (staging -> intermediate -> marts)
- LightGBM for the main classifier (native categorical + NaN handling)
- Optuna for hyperparameter search (30 trials, 5-fold stratified CV)
- SHAP (TreeExplainer) for explainability
- scikit-learn for the logistic regression baseline and metrics
- pytest + ruff + pre-commit for quality gates

## Repo layout

- `src/modeling.py` - baseline + LightGBM + Optuna on random 80/20 split
- `src/temporal_eval.py` - Round 1 train -> Round 2 test temporal holdout (the canonical evaluation)
- `src/evaluate.py` - SHAP plots + ROC/PR curves
- `src/data_loader.py` - DuckDB query helpers used by notebooks and ad-hoc EDA
- `config/paths.py` - all data file paths (isolates the messy raw directory names)
- `models/` - dbt project (not Python models)
- `tests/` - pytest unit tests
- `docs/` - numbered guides 1-5 + `docs/blog/` for deep-dive posts
- `outputs/figures/` - tracked PNGs referenced from README and docs
- `outputs/models/` - gitignored model artifacts

## Running things

```bash
make install-dev      # install + pre-commit hooks
make dbt-build        # run + test dbt models (~16s)
make train            # baseline + LightGBM + Optuna (random split)
make eval-temporal    # honest temporal holdout
make eval             # SHAP + ROC/PR plots
make test             # pytest
make lint             # ruff check + format --check
```

## Data conventions

- Raw CSVs live under `data/` and are gitignored (32 GB total). Download from the Kaggle competition page.
- The canonical dataset is the v2 refresh (March 2017 expiry cohort). Round 1 uses the original files for the temporal holdout.
- `churn_pred.duckdb` is a 2.5 GB local database file, also gitignored. Regenerate via `make dbt-build`.
- `outputs/feature_table.parquet` is the dbt mart output and is gitignored (regenerate from dbt).
- `outputs/figures/*.png` ARE tracked because the README references them.

## Evaluation philosophy

**Temporal holdout is canonical.** The random-split 0.993 ROC-AUC is an upper bound under a no-drift assumption that does not hold in practice. The temporal 0.924 is what a deployed model would actually face. See [docs/blog/01-temporal-vs-random-split.md](docs/blog/01-temporal-vs-random-split.md).

Always report both numbers. Never screenshot the random-split number alone.

## Style

- No em dashes in prose or docs.
- Ruff with `select = ["E", "F", "I", "UP"]`, line length 120, quote style double.
- pre-commit runs ruff check with auto-fix + ruff-format on every commit.
- No comments that explain WHAT code does (identifiers should be clear). Comments only for WHY when it is non-obvious (hidden constraint, subtle invariant, workaround for a known bug).

## Quality gates

- `make lint` must pass before pushing.
- `make test` must pass before pushing. CI runs pytest on Python 3.13 against every PR.
- `make dbt-test` must pass after dbt model changes (16 schema tests).
