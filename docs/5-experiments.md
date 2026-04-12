# 5. Experiments

Raw log of every trained model and each run's diagnostics. Newest runs at the top.

## Temporal holdout evaluation (Round 1 train -> Round 2 test)

| Run | Date | Model | Eval | ROC-AUC | Log Loss | PR-AUC | F1 | Notes |
|-----|------|-------|------|---------|----------|--------|-----|-------|
| 5 | 2026-04-10 | LightGBM | Temporal (Feb -> Mar) | 0.9237 | 0.2667 | 0.6045 | 0.578 | Honest evaluation. Train 993K users (6.4% churn), test 971K (9.0% churn). Distributional shift degrades calibration. |

### Diagnosis (Run 5)
ROC-AUC drop of 0.069 vs random split is modest given the +2.6 pp churn-rate shift. Log loss degradation (0.073 -> 0.267) is much larger, consistent with calibration breaking under drift while ranking ability holds. The model is production-usable for relative targeting but would need recalibration for absolute-probability decisions.

## Random split evaluation (within Round 2)

| Run | Date | Model | Eval | ROC-AUC | Log Loss | PR-AUC | F1 | Notes |
|-----|------|-------|------|---------|----------|--------|-----|-------|
| 4 | 2026-04-10 | LightGBM (behavioral) | Random 80/20 | 0.7705 | 0.2925 | -- | -- | Demographics + listening only (19 features). The genuinely hard problem. |
| 3 | 2026-04-10 | LightGBM (ablation) | Random 80/20 | 0.9933 | 0.0972 | 0.9418 | 0.881 | Dropped `last_cancel`, `last_auto_renew`, `has_transaction_data`. ROC-AUC unchanged. |
| 2 | 2026-04-10 | LightGBM (Optuna) | Random 80/20 | 0.9930 | 0.0727 | 0.9466 | 0.881 | 30 Optuna trials, 5-fold CV. 150 leaves, lr=0.096, 944 rounds. |
| 1 | 2026-04-10 | Logistic Regression | Random 80/20 | 0.9699 | 0.2549 | 0.7519 | 0.808 | Baseline. Balanced class weights, lbfgs. |

### Diagnosis (Run 3 vs Run 2)
Removing the three explicit cancel/auto-renew features barely moves ROC-AUC (0.9930 -> 0.9933) but log loss increases by 34% (0.0727 -> 0.0972). LightGBM recovers the ranking information from correlated features (`last_plan_days`, transaction aggregates) but its probability estimates become less sharp.

### Diagnosis (Run 4)
Stripping to behavioral features only (no transaction history, no cancel flags) drops ROC-AUC to 0.7705. This is the honest upper bound for "predict churn before the user signals intent." Any production retention system aimed at early intervention should budget for this number, not 0.993.

## Key takeaways

- **Temporal holdout is the canonical number** (0.924 ROC-AUC / 0.267 log loss). The random-split 0.993 is an upper bound under a no-drift assumption that does not hold in practice.
- **Behavioral-only ablation (0.771)** reveals what the model can do before a user signals intent. Any upstream retention system operates against this ceiling.
- **Calibration degrades much faster than discrimination** under temporal drift. Log loss +266% while ROC-AUC -7%.
