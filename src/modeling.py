"""
Churn prediction modeling pipeline.

Trains a logistic regression baseline and a LightGBM model with
Optuna hyperparameter tuning. Outputs a trained model, evaluation
metrics, and SHAP explanations.

Usage:
    python -m src.modeling            # run from project root
    python src/modeling.py            # also works
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import FEATURE_TABLE_PATH, FIGURES_DIR, MODELS_DIR, OUTPUT_DIR  # noqa: E402

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
# Columns to exclude from modeling (identifiers and target)
ID_COL = "msno"
TARGET = "is_churn"

# Categorical features: LightGBM handles these natively; logistic regression
# needs one-hot encoding.
CAT_FEATURES = ["gender", "city", "registered_via", "last_payment_method"]

# All other numeric columns are used as-is.
EXCLUDE_COLS = {ID_COL, TARGET}


def load_data():
    """Load feature table and split into train/validation sets."""
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols]
    y = df[TARGET]

    # Cast categoricals for LightGBM native handling
    for col in CAT_FEATURES:
        X[col] = X[col].astype("category")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")
    print(f"Train churn rate: {y_train.mean() * 100:.2f}% | Val churn rate: {y_val.mean() * 100:.2f}%")
    return X_train, X_val, y_train, y_val


def compute_metrics(y_true, y_prob, name=""):
    """Compute and print standard classification metrics."""
    ll = log_loss(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # F1 at optimal threshold (from precision-recall curve)
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1s)
    best_f1 = f1s[best_idx]
    best_thresh = thresholds[best_idx]

    metrics = {
        "log_loss": round(ll, 5),
        "roc_auc": round(roc, 5),
        "pr_auc": round(pr_auc, 5),
        "f1": round(best_f1, 5),
        "f1_threshold": round(best_thresh, 2),
    }

    if name:
        print(f"\n{'=' * 50}")
        print(f"  {name}")
        print(f"{'=' * 50}")
    for k, v in metrics.items():
        print(f"  {k:18s}: {v}")
    return metrics


# ---------------------------------------------------------------------------
# Baseline: Logistic Regression
# ---------------------------------------------------------------------------
def train_baseline(X_train, X_val, y_train, y_val):
    """Logistic regression baseline with one-hot categoricals + scaling."""
    numeric_cols = [c for c in X_train.columns if c not in CAT_FEATURES]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ],
    )

    pipe = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    # Fill NaN for sklearn (cannot handle missing values)
    X_tr = X_train.copy()
    X_vl = X_val.copy()
    for col in numeric_cols:
        median_val = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(median_val)
        X_vl[col] = X_vl[col].fillna(median_val)
    for col in CAT_FEATURES:
        # Convert to string first to ensure uniform type, then fill NaN
        X_tr[col] = X_tr[col].astype(str).replace("nan", "__missing__")
        X_vl[col] = X_vl[col].astype(str).replace("nan", "__missing__")

    pipe.fit(X_tr, y_train)
    y_prob = pipe.predict_proba(X_vl)[:, 1]
    metrics = compute_metrics(y_val, y_prob, name="Logistic Regression (baseline)")
    return pipe, metrics


# ---------------------------------------------------------------------------
# LightGBM with Optuna tuning
# ---------------------------------------------------------------------------
def tune_lgbm(X_train, y_train, n_trials=80):
    """Optuna hyperparameter search with 5-fold stratified CV."""
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / n_pos

    # Build LightGBM Dataset once (faster than re-creating each trial)
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES, free_raw_data=False)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "feature_pre_filter": False,
            "scale_pos_weight": scale_pos,
            # gbdt only: dart disables early stopping, making tuning 10x slower
            # for marginal improvement. Top competition solutions used gbdt as base.
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)]

        cv_result = lgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=5,
            stratified=True,
            callbacks=callbacks,
            return_cvbooster=False,
        )

        best_logloss = cv_result["valid binary_logloss-mean"][-1]
        best_round = len(cv_result["valid binary_logloss-mean"])
        trial.set_user_attr("best_round", best_round)
        return best_logloss

    # Run Optuna (minimize log loss)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name="lgbm_churn")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial: log_loss={study.best_value:.5f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")
    print(f"Best num_boost_round: {study.best_trial.user_attrs['best_round']}")

    return study


def train_final_lgbm(X_train, X_val, y_train, y_val, study):
    """Train final LightGBM model with best Optuna params."""
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / n_pos

    best_params = study.best_params.copy()
    best_round = study.best_trial.user_attrs["best_round"]

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "scale_pos_weight": scale_pos,
        **best_params,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=CAT_FEATURES, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=best_round,
        valid_sets=[dval],
        callbacks=[lgb.log_evaluation(period=50)],
    )

    # Save model
    model_path = MODELS_DIR / "lgbm_churn.txt"
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    # Evaluate
    y_prob = model.predict(X_val)
    metrics = compute_metrics(y_val, y_prob, name="LightGBM (tuned)")

    return model, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    X_train, X_val, y_train, y_val = load_data()

    # 1. Baseline
    print("\n--- Training baseline (Logistic Regression) ---")
    lr_model, lr_metrics = train_baseline(X_train, X_val, y_train, y_val)

    # 2. LightGBM with Optuna
    print("\n--- Tuning LightGBM with Optuna (30 trials, 5-fold CV) ---")
    study = tune_lgbm(X_train, y_train, n_trials=30)

    print("\n--- Training final LightGBM ---")
    lgbm_model, lgbm_metrics = train_final_lgbm(X_train, X_val, y_train, y_val, study)

    # Save study for later analysis
    study_path = MODELS_DIR / "optuna_study.json"
    with open(study_path, "w") as f:
        json.dump(
            {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "best_round": study.best_trial.user_attrs["best_round"],
                "n_trials": len(study.trials),
            },
            f,
            indent=2,
        )
    print(f"Optuna study saved to {study_path}")

    # Save validation predictions for evaluate.py
    X_val_lr = X_val.copy()
    for col in CAT_FEATURES:
        X_val_lr[col] = X_val_lr[col].astype(str).replace("nan", "__missing__")
    for col in X_val_lr.select_dtypes(include="number").columns:
        X_val_lr[col] = X_val_lr[col].fillna(X_train[col].median())
    y_prob_lr = lr_model.predict_proba(X_val_lr)[:, 1]
    y_prob_lgbm = lgbm_model.predict(X_val)

    val_preds = pd.DataFrame(
        {
            "y_true": y_val.values,
            "y_prob_lr": y_prob_lr,
            "y_prob_lgbm": y_prob_lgbm,
        },
        index=X_val.index,
    )
    val_preds.to_parquet(OUTPUT_DIR / "val_predictions.parquet")

    # Save validation features for SHAP
    X_val.to_parquet(OUTPUT_DIR / "X_val.parquet")
    print(f"\nValidation predictions and features saved to {OUTPUT_DIR}")

    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    print(f"  Baseline (LR)  log_loss={lr_metrics['log_loss']}, ROC-AUC={lr_metrics['roc_auc']}")
    print(f"  LightGBM       log_loss={lgbm_metrics['log_loss']}, ROC-AUC={lgbm_metrics['roc_auc']}")
    improvement = (lr_metrics["log_loss"] - lgbm_metrics["log_loss"]) / lr_metrics["log_loss"] * 100
    print(f"  Log loss improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
