# 3. Modeling

Two models: a logistic regression baseline and a tuned LightGBM. The choice of LightGBM reflects three properties of the data that matter more than raw accuracy:

1. Native categorical handling (no one-hot explosion for `city` + `last_payment_method`)
2. Native NaN handling (60% of `age` values are missing)
3. Robust to the `engagement_depth` skew (trees are invariant to monotonic transforms)

## Baseline: logistic regression

Pipeline: `StandardScaler` on numerics + `OneHotEncoder(handle_unknown="ignore")` on categoricals, then `LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)`.

NaN handling for sklearn (which cannot accept them natively):
- Numeric columns: fill with train median
- Categorical columns: cast to string, replace `nan` with `"__missing__"`

This is an honest baseline. Results appear in [4. Evaluation](4-evaluation.md).

## LightGBM

### Imbalance handling
`scale_pos_weight = n_negative / n_positive` (~10:1). Chosen over class weighting because it directly adjusts the gradient without modifying the loss function.

### Categorical features
Declared at `lgb.Dataset` construction time:
```python
CAT_FEATURES = ["gender", "city", "registered_via", "last_payment_method"]
lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES, free_raw_data=False)
```

pandas `category` dtype is set before Dataset construction so LightGBM can partition categorical values optimally rather than using one-hot.

### Optuna hyperparameter search

30 trials, 5-fold stratified CV, minimizing binary log loss. Search space:

| Parameter | Range | Sampling |
|-----------|-------|----------|
| `num_leaves` | [20, 150] | Uniform int |
| `learning_rate` | [0.01, 0.3] | Log-uniform |
| `min_child_samples` | [5, 100] | Uniform int |
| `subsample` | [0.5, 1.0] | Uniform |
| `colsample_bytree` | [0.5, 1.0] | Uniform |
| `reg_alpha` | [1e-8, 10.0] | Log-uniform |
| `reg_lambda` | [1e-8, 10.0] | Log-uniform |
| `boosting_type` | `gbdt` (fixed) | Fixed |

**Why `gbdt` is fixed**: `dart` disables early stopping, making tuning ~10x slower for marginal improvement. Top WSDM 2018 solutions used `gbdt` as the base learner.

### Best hyperparameters landed on
- `num_leaves`: 150
- `learning_rate`: 0.096
- `num_boost_round`: 944 (early-stopped)

5-fold CV with `lgb.cv` and `early_stopping(50)` reports the best round internally, and that round is used when fitting the final model on the full training fold.

## Ablation studies

Three ablations run on the random 80/20 split of Round 2 data:

1. **Full model**: all 36 features. ROC-AUC 0.993.
2. **Without subscription metadata** (`last_cancel`, `last_auto_renew`, `has_transaction_data` removed): ROC-AUC 0.993. LightGBM recovers the signal from `last_plan_days` and transaction aggregates.
3. **Behavioral features only** (demographics + listening, no transaction history): 19 features. ROC-AUC 0.771. This is the model that would actually be useful for upstream retention interventions since it relies on signals available before a cancel event.

The 0.993 -> 0.771 drop is the most informative finding in the whole project. It says the easy 0.993 headline number is dominated by explicit user signals (cancel + auto-renew) that arrive too late to act on.

## Reproducibility

```bash
make train            # Trains baseline + LightGBM, saves to outputs/models/
make eval             # SHAP + ROC/PR curves on random-split validation
make eval-temporal    # Honest temporal holdout (Feb train -> Mar test)
```

Outputs:
- `outputs/models/lgbm_churn.txt` (LightGBM booster)
- `outputs/models/optuna_study.json` (best params + CV score)
- `outputs/val_predictions.parquet` (y_true + y_prob_lr + y_prob_lgbm)
- `outputs/X_val.parquet` (validation features for SHAP)
- `outputs/figures/11_roc_pr_curves.png`, `13_lgbm_importance.png`, `14-17_shap_*.png`

## What changed across iterations

See [5. Experiments](5-experiments.md) for the full run-by-run journal.
