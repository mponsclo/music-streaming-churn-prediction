# 4. Evaluation

Trained model -> two protocols (temporal holdout + random split) -> metrics + SHAP + limitations.

## Why This Layer Exists

The evaluation protocol is where most ML projects quietly lie to themselves. A random 80/20 split is convenient, it is what tutorials use, and it produces the headline number. For a time-dependent problem like churn, it also assumes that the future looks exactly like the past, which is the assumption that breaks first in production.

This project reports two evaluations side by side. The temporal holdout is the **canonical** number: it mirrors the actual WSDM Cup 2018 setup and reflects the distribution drift the model would face if deployed. The random split is an **upper bound under no drift**, useful only for sanity checks and for quantifying the gap. Every metric in this project is reported for both protocols so the reader can see the difference.

Run the evaluations:

```bash
make eval-temporal    # canonical: train on Feb 2017, test on March 2017
make eval             # random-split metrics + SHAP plots on outputs/X_val.parquet
```

## Temporal holdout (the canonical evaluation)

Mirrors the actual WSDM Cup 2018 setup:

- Train on Round 1 features (users expiring February 2017).
- Test on Round 2 features (users expiring March 2017).
- Features for each round are built independently from the raw files, so there is no leakage across the temporal boundary.

```
Train Round 1: 992,931 users, 6.4% churn rate
Test  Round 2: 970,960 users, 9.0% churn rate
```

Churn rate shifts by +2.6 percentage points month over month. This is real distributional drift.

### Results

| Metric | Value |
|--------|-------|
| Log Loss | **0.267** |
| ROC-AUC | **0.924** |
| PR-AUC | **0.604** |
| F1 (optimal threshold) | **0.578** |

`python src/temporal_eval.py` (or `make eval-temporal`) reproduces these numbers end-to-end, including rebuilding Round 1 features from the original raw files.

### Why this is the honest number

Random cross-validation and a random 80/20 split both assume train and test are drawn from the same distribution. Churn prediction is a time-dependent problem, so that assumption is false. Temporal holdout enforces train-before-test by construction, which is the weakest reasonable fidelity to deployment. Any stricter protocol (e.g., rolling-window retraining) would require more data than the KKBox release provides.

## Random split (upper bound under no drift)

For completeness, the same LightGBM evaluated on a random 80/20 split of Round 2 only:

| Model | Log Loss | ROC-AUC | PR-AUC | F1 |
|-------|----------|---------|--------|-----|
| Logistic Regression (baseline) | 0.255 | 0.970 | 0.752 | 0.808 |
| **LightGBM (Optuna tuned)** | **0.073** | **0.993** | **0.947** | **0.881** |
| LightGBM (behavioural only) | 0.292 | 0.771 | -- | -- |

### The 0.993 vs 0.924 gap

Two ROC-AUC values for the same model, differing by 0.069. In terms of how much of the no-skill-to-perfect range the model covers, 0.924 closes 85% of the gap from 0.5 to 1.0, while 0.993 closes 99%.

See [blog/01-temporal-vs-random-split.md](blog/01-temporal-vs-random-split.md) for the full deep-dive.

The short version: random split inflates ROC-AUC because train and test share the same temporal context. Temporal holdout removes that context overlap and surfaces what a deployed model would actually face. The random-split number is an upper bound; the temporal number is the expected one.

## Key Concepts

### Discrimination vs calibration

ROC-AUC degrades from 0.993 to 0.924 (-7%). Log loss degrades from 0.073 to 0.267 (+266%). Why?

**Discrimination** is the model's ability to *rank* users by risk. ROC-AUC and PR-AUC measure this. Ranking is invariant to monotonic rescaling of the scores: if the model uniformly shifts every predicted probability by +0.1, ranks do not change and ROC-AUC does not change.

**Calibration** is whether the predicted probability matches reality. Log loss and the Brier score measure this:

$$\text{Brier} = \frac{1}{n}\sum_{i=1}^{n} (p_i - y_i)^2$$

Under temporal drift, the underlying churn rate changed from 6.4% to 9.0%. The model still ranks users correctly (most risky users in Round 2 are the same archetypes as in Round 1), but its absolute probabilities are calibrated to a 6.4% world. They systematically underpredict in a 9.0% world. That is the calibration failure showing up as a 3.7x log-loss increase with only a 7% ROC-AUC drop.

In production, this suggests:

- Relative targeting (top-N at-risk users) remains reliable.
- Decision thresholds calibrated on random-split probabilities would miscount at-risk users in absolute terms.
- Monitoring should track calibration error (Brier score, reliability diagrams) separately from ROC-AUC.

### SHAP and TreeExplainer

SHAP (SHapley Additive exPlanations) attributes each prediction to each input feature. For a sample $x$ with prediction $f(x)$, SHAP decomposes:

$$f(x) = \phi_0 + \sum_{j=1}^{d} \phi_j(x)$$

where $\phi_0$ is the baseline (mean prediction over the dataset) and $\phi_j(x)$ is the signed contribution of feature $j$ to this specific sample. The contributions are derived from cooperative game theory: each feature's contribution is its average marginal effect across all possible orderings of features. That averaging is what makes SHAP values "fair" in the game-theoretic sense (they are the unique attribution satisfying efficiency, symmetry, dummy, and additivity).

Computing exact Shapley values for a general model is exponential in the number of features. **TreeExplainer** exploits the tree structure to compute them exactly in polynomial time (roughly $O(TLD^2)$ where $T$ is the number of trees, $L$ the leaves per tree, $D$ the depth). That is why SHAP on LightGBM runs in seconds on 2,000 samples, where SHAP on a neural network would take hours.

All SHAP plots below use TreeExplainer on 2,000 random-split validation samples. See the [glossary](0-glossary.md#explainability) for a depth-one definition of every SHAP term.

## Feature importance (SHAP)

### Global importance

![SHAP global importance](../outputs/figures/14_shap_global_importance.png)

Top predictors are dominated by subscription metadata:
- `last_cancel`, `last_auto_renew`
- `days_until_expiry`
- `last_plan_days`
- `last_payment_method`

Listening behaviour features (`active_days`, `total_secs`, `completion_rate`) have measurable but modest effects. This is the same ordering suggested by the correlation analysis in [1. Data](1-data.md#feature-relevance-normal-subscribers), reassuring that the model is not learning something counterintuitive.

### Beeswarm plot

![SHAP beeswarm](../outputs/figures/15_shap_beeswarm.png)

One dot per sample per feature. Horizontal position is the SHAP value (contribution to log-odds). Colour is the raw feature value (red = high, blue = low). The separation of red vs blue shows the direction of the effect. For `last_cancel`, red dots (cancel = 1) cluster on the positive side, meaning cancellation pushes prediction upward toward churn. The opposite is true for `last_auto_renew`.

### Individual predictions

Waterfall plots decompose one sample's prediction, starting at the baseline and stacking contributions:

![SHAP waterfall: churner](../outputs/figures/16_shap_waterfall_churner.png)
![SHAP waterfall: retained](../outputs/figures/17_shap_waterfall_retained.png)

These are what you would show to a product owner or compliance reviewer asking "why did the model flag this specific user?"

## ROC and PR curves

![ROC and PR curves](../outputs/figures/11_roc_pr_curves.png)

Random-split evaluation. Both curves clearly above the no-skill baseline (diagonal for ROC, prevalence line for PR). Note that the PR curve's no-skill baseline sits at 0.09 (the positive-class prevalence), not at 0.5.

## Threshold selection

F1 is reported at the optimal threshold found by sweeping `np.arange(0.05, 0.95, 0.01)`. For production deployment, the threshold should instead be chosen by business constraints (e.g., retention campaign cost per contact vs expected revenue saved per retained user), not by F1 maximisation. A threshold that maximises F1 weights precision and recall equally, which almost never matches the real economic tradeoff.

## Limitations

Honest accounting of where this model would break in production:

1. **Calibration drift under temporal shift**. Log loss degrades 3.7x across one month of drift. In production, predicted probabilities would need recalibration against a rolling validation window (Platt scaling, isotonic regression, or a simple reliability-diagram-based shift). The ranking holds; the probabilities do not.

2. **Narrow 30-day churn definition**. The label is "did the user renew within 30 days of expiry?" A user who renews on day 35 is flagged as churned even though they effectively did not leave. A user who cancels on day 31 is flagged as retained because the label window closed. Any retention campaign using these predictions should define its own business-relevant churn horizon.

3. **One-month listening window**. `user_logs_v2` covers only March 2017. Multi-window behavioural features (trailing 30-day vs 90-day deltas, engagement trends, sessions-per-week growth) are not available from the public release. A production system with access to the full log would almost certainly extract more signal from listening behaviour than this model does.

4. **No resubscription modeling**. A churned user who re-subscribes the next month is treated as churned forever within this framing. For lifetime-value modelling or win-back campaigns, the target would need to be redefined (e.g., survival analysis or a competing-risks formulation).

5. **Cold-start users**. The 3.9% of users with no transaction history churn at 78.7%, and the model currently uses `has_transaction_data=0` as a strong feature. A production model deployed on genuinely new signups (not this dataset's "no transaction" cohort) would need special handling: either a separate model trained on early-signal features only, or a fallback rule until enough history accumulates.

6. **Subscription-feature dominance**. The full model's 0.993 ROC-AUC is dominated by `last_cancel` and `last_auto_renew`, which are late signals. For early intervention, use the 0.771 behavioural-only ablation as the honest ceiling. See [3. Modeling: Ablation studies](3-modeling.md#ablation-studies).

## See also

- Full experiment journal with per-run `Config / Results / Diagnosis`: [5. Experiments](5-experiments.md)
- Deep dive on the temporal vs random split gap: [blog post](blog/01-temporal-vs-random-split.md)
- Glossary of evaluation terms and protocols: [0. Glossary](0-glossary.md#evaluation-metrics)
