"""
Model evaluation, SHAP explanations, and plot generation.

Reads the saved model and validation predictions from src/modeling.py,
then produces:
  - ROC and PR curves comparing baseline vs LightGBM
  - Confusion matrix at optimal F1 threshold
  - SHAP global importance (bar + beeswarm)
  - SHAP waterfall plots for individual predictions

Usage:
    python -m src.evaluate
    python src/evaluate.py
"""

import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    f1_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import FIGURES_DIR, MODELS_DIR, OUTPUT_DIR  # noqa: E402

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CAT_FEATURES = ["gender", "city", "registered_via", "last_payment_method"]


def load_artifacts():
    """Load model, validation predictions, and validation features."""
    model = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_churn.txt"))

    val_preds = pd.read_parquet(OUTPUT_DIR / "val_predictions.parquet")
    X_val = pd.read_parquet(OUTPUT_DIR / "X_val.parquet")

    for col in CAT_FEATURES:
        X_val[col] = X_val[col].astype("category")

    return model, val_preds, X_val


# ---------------------------------------------------------------------------
# Curve plots
# ---------------------------------------------------------------------------
def plot_roc_pr_curves(val_preds):
    """Side-by-side ROC and Precision-Recall curves."""
    y_true = val_preds["y_true"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC
    for name, col, color in [
        ("Logistic Regression", "y_prob_lr", "#8172B2"),
        ("LightGBM", "y_prob_lgbm", "#4C72B0"),
    ]:
        RocCurveDisplay.from_predictions(y_true, val_preds[col], name=name, ax=axes[0], color=color)
    axes[0].set_title("ROC Curve")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)

    # Precision-Recall
    for name, col, color in [
        ("Logistic Regression", "y_prob_lr", "#8172B2"),
        ("LightGBM", "y_prob_lgbm", "#4C72B0"),
    ]:
        PrecisionRecallDisplay.from_predictions(y_true, val_preds[col], name=name, ax=axes[1], color=color)
    axes[1].set_title("Precision-Recall Curve")
    prevalence = y_true.mean()
    axes[1].axhline(y=prevalence, color="k", linestyle="--", alpha=0.3, label=f"Baseline ({prevalence:.2f})")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved ROC and PR curves")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(val_preds):
    """Confusion matrix at optimal F1 threshold for LightGBM."""
    y_true = val_preds["y_true"]
    y_prob = val_preds["y_prob_lgbm"]

    # Find optimal F1 threshold
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    best_thresh = thresholds[np.argmax(f1s)]
    y_pred = (y_prob >= best_thresh).astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Retained", "Churned"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (threshold={best_thresh:.2f}, F1={max(f1s):.3f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix (threshold={best_thresh:.2f})")


# ---------------------------------------------------------------------------
# LightGBM native feature importance
# ---------------------------------------------------------------------------
def plot_lgbm_importance(model):
    """Built-in LightGBM feature importance (gain-based)."""
    fig, ax = plt.subplots(figsize=(8, 10))
    lgb.plot_importance(
        model,
        importance_type="gain",
        max_num_features=25,
        ax=ax,
        color="#4C72B0",
    )
    ax.set_title("LightGBM Feature Importance (gain)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_lgbm_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved LightGBM feature importance")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def run_shap(model, X_val, val_preds):
    """SHAP TreeExplainer: global bar, beeswarm, and waterfall plots."""
    # Subsample for SHAP (full dataset is too slow, especially if dart)
    n_shap = min(2000, len(X_val))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_val), size=n_shap, replace=False)
    X_shap = X_val.iloc[idx]

    print(f"Computing SHAP values for {n_shap} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Global bar plot
    fig, ax = plt.subplots(figsize=(8, 10))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (mean |SHAP value|)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14_shap_global_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved SHAP global importance bar plot")

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
    plt.title("SHAP Beeswarm Plot")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "15_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved SHAP beeswarm plot")

    # Waterfall: pick one churner and one retained user from the SHAP sample
    y_shap = val_preds["y_true"].iloc[idx].values
    y_prob_shap = val_preds["y_prob_lgbm"].iloc[idx].values

    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_shap.values,
        feature_names=X_shap.columns.tolist(),
    )

    # High-confidence churner
    churn_mask = (y_shap == 1) & (y_prob_shap > 0.8)
    if churn_mask.any():
        churn_idx = np.where(churn_mask)[0][0]
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation[churn_idx], show=False, max_display=12)
        plt.title("SHAP Waterfall: High-Confidence Churner")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "16_shap_waterfall_churner.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved SHAP waterfall (churner)")

    # High-confidence retained
    retained_mask = (y_shap == 0) & (y_prob_shap < 0.02)
    if retained_mask.any():
        retained_idx = np.where(retained_mask)[0][0]
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation[retained_idx], show=False, max_display=12)
        plt.title("SHAP Waterfall: High-Confidence Retained User")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "17_shap_waterfall_retained.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved SHAP waterfall (retained)")

    return shap_values, X_shap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading artifacts...")
    model, val_preds, X_val = load_artifacts()

    print("\n--- Generating evaluation plots ---")
    plot_roc_pr_curves(val_preds)
    plot_confusion_matrix(val_preds)
    plot_lgbm_importance(model)

    print("\n--- Running SHAP analysis ---")
    run_shap(model, X_val, val_preds)

    print(f"\nAll plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
