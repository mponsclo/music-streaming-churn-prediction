# 5. Experiments

The raw experiment journal: every model iteration, every metric, every diagnostic. Curated narrative lives in [3. Modeling](3-modeling.md) and [4. Evaluation](4-evaluation.md); this file is the unedited record.

**Current production model**: Run 5 (LightGBM, temporal holdout). ROC-AUC 0.924, log loss 0.267 on Round 2 test after training on Round 1. See [Run 5](#run-5-lightgbm-temporal-holdout) below.

## Summary table

Newest runs at the top. Temporal evaluation is canonical; random split is reported for the 0.993 vs 0.924 gap analysis in [4. Evaluation](4-evaluation.md).

| Run | Date | Model | Eval | ROC-AUC | Log Loss | PR-AUC | F1 |
|-----|------|-------|------|---------|----------|--------|-----|
| **5** | 2026-04-10 | LightGBM | Temporal (Feb -> Mar) | **0.9237** | **0.2667** | **0.6045** | **0.578** |
| 4 | 2026-04-10 | LightGBM (behavioural only) | Random 80/20 | 0.7705 | 0.2925 | -- | -- |
| 3 | 2026-04-10 | LightGBM (ablation, no cancel/auto-renew) | Random 80/20 | 0.9933 | 0.0972 | 0.9418 | 0.881 |
| 2 | 2026-04-10 | LightGBM (Optuna tuned) | Random 80/20 | 0.9930 | 0.0727 | 0.9466 | 0.881 |
| 1 | 2026-04-10 | Logistic Regression (baseline) | Random 80/20 | 0.9699 | 0.2549 | 0.7519 | 0.808 |

---

## Run 1: Logistic Regression (baseline)

**Config:**
- Model: `LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)`
- Preprocessing: `StandardScaler` on numerics, `OneHotEncoder(handle_unknown="ignore")` on categoricals
- NaN handling: train-median imputation for numerics, `"__missing__"` sentinel for categoricals
- Features: all 36 from `ml_churn_features`
- Split: random 80/20 stratified by `is_churn`

**Results:**
- ROC-AUC: 0.9699
- Log Loss: 0.2549
- PR-AUC: 0.7519
- F1 (optimal threshold): 0.808

**Diagnosis:**
Strong baseline. The model has no way to represent the `auto_renew x cancel` interaction explicitly, yet ROC-AUC already sits at 0.97 because both features are nearly additive in log-odds space. This is the floor any main model must clearly beat to justify complexity. It also confirms the one-hot + scaler + `class_weight="balanced"` recipe is not the bottleneck; anything LightGBM gains beyond this comes from interactions and non-linearity.

---

## Run 2: LightGBM (Optuna tuned)

**Config:**
- Model: LightGBM, `objective="binary"`, `scale_pos_weight = n_neg / n_pos` (~10)
- Categorical features (native): `gender`, `city`, `registered_via`, `last_payment_method`
- Hyperparameter search: Optuna, 30 trials, 5-fold stratified CV, minimise binary log loss
- Best params: `num_leaves=150`, `learning_rate=0.096`, `num_boost_round=944` (early-stopped at patience 50)
- Features: all 36
- Split: random 80/20 stratified by `is_churn`

**Results:**
- ROC-AUC: 0.9930
- Log Loss: 0.0727
- PR-AUC: 0.9466
- F1 (optimal threshold): 0.881

**Diagnosis:**
Headline number. Log loss improves 3.5x over the baseline (0.255 -> 0.073), confirming substantial non-linear signal on top of the linear floor. Optuna converged quickly: the best trial was found within the first half of the 30-trial budget, and the top 5 trials all sat within 0.003 log loss of each other. No obvious overfitting to the CV split (validation metrics match fold-CV metrics within noise). The caveat: this is the random-split number. Whether 0.993 survives temporal drift is Run 5.

---

## Run 3: LightGBM (ablation, subscription metadata dropped)

**Config:**
- Same as Run 2, but features dropped: `last_cancel`, `last_auto_renew`, `has_transaction_data`
- 33 features remain

**Results:**
- ROC-AUC: 0.9933 (vs Run 2: 0.9930)
- Log Loss: 0.0972 (vs Run 2: 0.0727, +34%)
- PR-AUC: 0.9418 (vs Run 2: 0.9466)
- F1 (optimal threshold): 0.881 (unchanged)

**Diagnosis:**
Removing the three explicit cancel/auto-renew features barely moves ROC-AUC but log loss rises 34%. LightGBM recovers the ranking information from correlated features (`last_plan_days`, transaction aggregates), but its probability estimates become less sharp because those proxies carry noisier signal. This is a clean example of discrimination and calibration decoupling: ranks survive, probabilities do not. It also means any production system that depends on calibrated probabilities (not just rankings) should not rely on feature-engineering tricks to "recover" removed signal.

---

## Run 4: LightGBM (behavioural only)

**Config:**
- Same tuning as Run 2, but features restricted to demographics + listening behaviour
- 19 features: age, gender, city, registered_via, tenure_days, has_member_data, has_age, active_days, total_secs, complete_plays, unique_songs, partial_plays, half_plays, three_quarter_plays, near_complete_plays, total_plays, completion_rate, engagement_depth, avg_daily_secs, mins_per_song
- No transaction features, no cancel/auto-renew flags

**Results:**
- ROC-AUC: 0.7705
- Log Loss: 0.2925
- PR-AUC, F1: not computed (ablation focus is discrimination)

**Diagnosis:**
This is the genuinely hard problem. Stripping to behaviour-only drops ROC-AUC from 0.993 to 0.771, confirming that most of the headline-number signal comes from explicit user actions (cancel, auto-renew toggle) that arrive too late to act on. **0.771 is the honest ceiling for "predict churn before the user signals intent."** Any upstream retention system aimed at early intervention should budget for this number, not 0.993 or even 0.924. The correlation analysis in [1. Data](1-data.md#feature-relevance-normal-subscribers) predicted exactly this: listening volume has near-zero correlation with churn within the normal-subscriber segment, leaving demographics and engagement-quality features doing most of the work.

---

## Run 5: LightGBM (temporal holdout)

**Current production model.**

**Config:**
- Same Run 2 hyperparameters (`num_leaves=150`, `learning_rate=0.096`, `num_boost_round=944`)
- Training set: Round 1 features built independently from the original (v1) files. 992,931 users, 6.4% churn.
- Test set: Round 2 features from v2 refresh. 970,960 users, 9.0% churn.
- Feature parity: same 36 columns, built via the same dbt logic, no shared intermediate artifacts across rounds.

**Results:**
- ROC-AUC: 0.9237
- Log Loss: 0.2667
- PR-AUC: 0.6045
- F1 (optimal threshold): 0.578

**Diagnosis:**
The honest evaluation. ROC-AUC drop of 0.069 vs Run 2 is modest given the +2.6 pp churn-rate shift from February to March. Log loss degradation (0.073 -> 0.267) is much larger, consistent with calibration breaking under drift while ranking ability holds. The model is production-usable for relative targeting (top-N at-risk users) but would need recalibration (Platt scaling or isotonic regression on a rolling window) for absolute-probability decisions. Full framing in [4. Evaluation](4-evaluation.md) and the blog post [blog/01-temporal-vs-random-split.md](blog/01-temporal-vs-random-split.md).

---

## Key takeaways

- **Temporal holdout is the canonical number** (0.924 ROC-AUC, 0.267 log loss). The random-split 0.993 is an upper bound under a no-drift assumption that does not hold in practice. Report both together, never random alone.
- **Behavioural-only ablation (0.771)** reveals what the model can do before a user signals intent. Any upstream retention system operates against this ceiling, not against 0.993 or 0.924.
- **Calibration degrades much faster than discrimination** under temporal drift. Log loss +266% while ROC-AUC -7% across the Feb->Mar boundary. Any production deployment should monitor both separately. See [4. Evaluation: Limitations](4-evaluation.md#limitations).

## See also

- Model configuration and search space: [3. Modeling](3-modeling.md)
- Evaluation philosophy and limitations: [4. Evaluation](4-evaluation.md)
- Glossary: [0. Glossary](0-glossary.md)
