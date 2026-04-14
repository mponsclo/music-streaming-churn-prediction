# 3. Modeling

Feature Parquet -> Logistic Regression baseline + LightGBM tuned with Optuna (30 trials, 5-fold stratified CV).

## Why This Layer Exists

A single tuned model tells you nothing about whether it is any good. You need a floor (what the simplest reasonable model achieves) and a ceiling (what a well-tuned strong model achieves), and you need those two numbers on the **same data split** so the comparison is honest.

This guide covers both models, why each was chosen, how they handle the data's awkward parts (class imbalance, 60% missing age, a handful of near-constant columns), and how the hyperparameter search is structured. Evaluation and final results live in [4. Evaluation](4-evaluation.md); ablation and iteration history live in [5. Experiments](5-experiments.md).

Run the full pipeline end-to-end:

```bash
make train          # fits baseline + LightGBM + Optuna, saves artifacts to outputs/models/
```

## Key Concepts

### The two-model setup

Logistic regression is the baseline because it is the simplest model that can in principle solve a binary classification problem, and because it establishes how much headroom a non-linear model has to earn. LightGBM is the main model because gradient boosting on tabular data is the current empirical standard, and because three specific properties of this dataset match its strengths:

1. Native categorical handling (no one-hot explosion for `city` + `last_payment_method`).
2. Native NaN handling (60% of `age` values are missing).
3. Robustness to the `engagement_depth` skew (trees are invariant to monotonic transforms).

A single-model project would leave the "is 0.993 ROC-AUC actually good?" question unanswered. The baseline anchors the answer.

### Binary log loss

Both models are trained to minimise binary cross-entropy. For a sample with true label $y \in \{0, 1\}$ and predicted probability $p$:

$$\text{LL}(y, p) = -\left[ y \log p + (1 - y) \log(1 - p) \right]$$

Log loss punishes confident wrong predictions much more than unconfident ones. A model that predicts $p = 0.99$ for a negative sample pays $-\log(0.01) \approx 4.6$ per sample, versus $-\log(0.5) = 0.69$ for a hedged prediction. This is what drives models toward well-calibrated probabilities rather than just correct rankings. Per-metric depth in the [glossary](0-glossary.md#evaluation-metrics).

### Class imbalance: `scale_pos_weight`

The data is roughly 10:1 retained vs churned. Without intervention, the model fits the majority class and ignores the minority. LightGBM's `scale_pos_weight` multiplies the gradient of positive-class examples by a constant, derived directly from the class counts:

$$\texttt{scale\_pos\_weight} = \frac{n_\text{neg}}{n_\text{pos}} \approx \frac{883{,}630}{87{,}330} \approx 10$$

The baseline uses scikit-learn's `class_weight="balanced"` instead, which achieves the same effect via sample reweighting. Both approaches shift the decision boundary toward the minority class without modifying the loss function or resampling the data.

## Baseline: logistic regression

Pipeline: `StandardScaler` on numerics + `OneHotEncoder(handle_unknown="ignore")` on categoricals, then `LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)`.

NaN handling for scikit-learn (which cannot accept NaN natively):
- Numeric columns: fill with train median.
- Categorical columns: cast to string, replace `nan` with `"__missing__"`.

This is an honest baseline. It cannot represent feature interactions unless you construct them by hand, so it establishes the floor for "linear, individually-weighted features." Results appear in [4. Evaluation](4-evaluation.md).

## LightGBM

### Categorical features
Declared at `lgb.Dataset` construction time:

```python
CAT_FEATURES = ["gender", "city", "registered_via", "last_payment_method"]
lgb.Dataset(X_train, label=y_train,
            categorical_feature=CAT_FEATURES, free_raw_data=False)
```

The pandas `category` dtype is set before `Dataset` construction so LightGBM can partition categorical values optimally. For a city feature with hundreds of distinct values, this matters: one-hot encoding would create hundreds of sparse binary columns, each with very little signal per split. Native handling lets the model find the optimal partition of cities in one step (see [glossary: LightGBM](0-glossary.md#lightgbm)).

### Optuna hyperparameter search

30 trials, 5-fold stratified CV, minimising binary log loss. The search space and its sampling distributions:

| Parameter | Range | Sampling |
|-----------|-------|----------|
| `num_leaves` | `[20, 150]` | Uniform int |
| `learning_rate` | `[0.01, 0.3]` | Log-uniform |
| `min_child_samples` | `[5, 100]` | Uniform int |
| `subsample` | `[0.5, 1.0]` | Uniform |
| `colsample_bytree` | `[0.5, 1.0]` | Uniform |
| `reg_alpha` | `[1e-8, 10.0]` | Log-uniform |
| `reg_lambda` | `[1e-8, 10.0]` | Log-uniform |
| `boosting_type` | `gbdt` (fixed) | Fixed |

**Why log-uniform on learning rate and regularisers**: `learning_rate=0.01` and `learning_rate=0.02` are meaningfully different; `learning_rate=0.20` and `learning_rate=0.21` are not. Log-uniform sampling spends proportional effort on each order of magnitude, which matches our prior about the parameter's effect on the optimiser. See [glossary: uniform vs log-uniform](0-glossary.md#uniform-vs-log-uniform-sampling).

**Why `gbdt` is fixed**: `dart` (dropout-regularised trees) disables early stopping, making tuning ~10x slower for marginal improvement on this dataset. Top WSDM 2018 solutions used `gbdt` as the base learner.

### Optuna and TPE

Optuna uses the Tree-structured Parzen Estimator sampler by default. TPE maintains two densities over the hyperparameter space: one over trials that performed well, and one over trials that performed poorly. It samples the next trial where the ratio of good density to bad density is highest. Concretely, this means 30 Optuna trials explore the search space much more efficiently than 30 random-search trials; the sampler concentrates on promising regions as it learns. More detail in the [glossary](0-glossary.md#tpe-tree-structured-parzen-estimator).

### Stratified k-fold CV

The CV split is stratified by `is_churn`, so each fold has exactly 9% churners. At 9% prevalence, a non-stratified 5-fold split could give one fold 7% churn and another 11%, introducing fold-to-fold noise that hides the signal we are trying to measure. Stratification is one line in scikit-learn (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) and has an outsized effect on tuning stability.

### Best hyperparameters landed on

- `num_leaves`: 150
- `learning_rate`: 0.096
- `num_boost_round`: 944 (early-stopped)

5-fold CV with `lgb.cv` and `early_stopping(50)` reports the best round internally, and that round is used when fitting the final model on the full training fold. Early stopping here means: stop adding trees if the validation log loss has not improved for 50 consecutive rounds. It prevents overfitting without requiring a separate max-iterations grid search.

## Ablation studies

Three ablations run on the random 80/20 split of Round 2 data. The exact numbers live in [5. Experiments](5-experiments.md); the framing lives here.

1. **Full model**: all 36 features. ROC-AUC 0.993.
2. **Without subscription metadata** (`last_cancel`, `last_auto_renew`, `has_transaction_data` removed): ROC-AUC 0.993. LightGBM recovers the signal from `last_plan_days` and transaction aggregates. Log loss does degrade (0.073 -> 0.097), which is consistent with the [glossary](0-glossary.md#discrimination-vs-calibration) point that ranking and calibration can come apart.
3. **Behavioural features only** (demographics + listening, no transaction history): 19 features. ROC-AUC 0.771. This is the model that would actually be useful for upstream retention interventions because it relies on signals available before a cancel event.

The 0.993 -> 0.771 drop is the most informative finding in the project. It says the easy 0.993 headline number is dominated by explicit user signals (cancel + auto-renew) that arrive too late to act on. Every reader should hold 0.771, not 0.993 or 0.924, as the "predict churn before the user signals intent" baseline.

## Artifacts

`make train` writes:

- `outputs/models/lgbm_churn.txt` (LightGBM booster, reloadable via `lgb.Booster(model_file=...)`).
- `outputs/models/optuna_study.json` (best params, best CV score, all trial history).
- `outputs/val_predictions.parquet` (`y_true`, `y_prob_lr`, `y_prob_lgbm` for the random-split validation set).
- `outputs/X_val.parquet` (validation features, used by the SHAP explainer in [4. Evaluation](4-evaluation.md)).

Separate entry points for SHAP and temporal evaluation:

```bash
make eval             # SHAP + ROC/PR curves on random-split validation
make eval-temporal    # Honest temporal holdout (Feb train -> Mar test)
```

## See also

- Evaluation numbers, SHAP analysis, limitations: [4. Evaluation](4-evaluation.md)
- Full run-by-run journal with `Config / Results / Diagnosis` per experiment: [5. Experiments](5-experiments.md)
- Glossary of LightGBM, Optuna, CV, and imbalance concepts: [0. Glossary](0-glossary.md#main-model)
