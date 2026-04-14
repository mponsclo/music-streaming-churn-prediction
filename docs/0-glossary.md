# 0. Glossary

Every technical term that appears in this project, defined in one place. Ordered by the stage of the pipeline where the concept first matters. If you are reading the numbered guides and hit a term you do not recognise, it is defined here.

## Data layer

### DuckDB
An embedded analytical database (the SQLite of OLAP). It runs inside the Python process, reads CSVs and Parquet files directly without a separate server, and streams data from disk rather than loading everything into RAM. This project uses DuckDB because `user_logs_v2.csv` is 1.3 GB (18.4M rows) and would not fit comfortably in a pandas DataFrame. Queries that would take minutes in pandas finish in seconds.

### dbt (data build tool)
A SQL compilation and orchestration framework. You write SQL files that reference each other via `{{ ref('model_name') }}`, and dbt builds a dependency graph, runs them in the right order, and runs schema tests against the outputs. The `dbt-duckdb` adapter lets dbt target a local DuckDB database instead of a cloud warehouse.

### Staging -> intermediate -> marts
A three-layer convention for dbt projects. **Staging** is one model per source file, doing only light cleanup (type casts, column renames). **Intermediate** models combine staging layers and compute per-entity aggregations. **Marts** are the final tables consumed by downstream systems (in this project, the ML pipeline). The layering keeps each SQL file small and testable, and makes the lineage easy to reason about.

### Parquet
A columnar file format that stores each column separately, compressed. Reading a subset of columns touches only those column files, so it is much faster than CSV for wide tables. The final feature table in this project is written as Parquet because modeling only reads a subset of features at a time.

## Target definition

### Churn (in this dataset)
A user churns if they do not renew their subscription within 30 days after it expires. The label is binary: `is_churn = 1` for users who failed to renew, `0` otherwise. Users can re-subscribe after churning, so churn is not a terminal state.

### Class imbalance ratio
The ratio of negative to positive examples. This project sits at roughly 10:1 (91% retained, 9% churned). Imbalance matters because a model that predicts "no churn" for every user reaches 91% accuracy while catching zero churners.

## Baseline model

### Logistic regression
A linear model that fits weights $w$ and a bias $b$ to predict probability $p = \sigma(w^T x + b)$ where $\sigma$ is the sigmoid function. It serves as the baseline because it is fast, interpretable, and well-calibrated when its assumptions hold. It cannot represent feature interactions unless you construct them manually.

### StandardScaler
Centres each numeric feature at zero and scales it to unit variance: $x' = (x - \mu) / \sigma$. Linear models need this because features with larger raw scales (e.g., `total_secs` at hundreds of thousands) would otherwise dominate the optimisation.

### One-hot encoding
Turns a categorical column with $k$ levels into $k$ binary columns, each indicating whether the row has that level. Logistic regression requires this because it only accepts numeric inputs. Tree models do not.

### `class_weight="balanced"`
A scikit-learn option that reweights samples so each class contributes equally to the loss. Equivalent to multiplying each class's gradient by `n_samples / (n_classes * n_class_samples)`. Used in the baseline to handle the 10:1 imbalance.

## Main model

### Gradient boosting
An ensemble of decision trees built sequentially. Each tree fits the gradient of the loss with respect to the current prediction, so the ensemble incrementally corrects its own errors. The final prediction is the sum of all trees' outputs passed through a sigmoid.

### LightGBM
Microsoft's gradient boosting framework. It differs from XGBoost in two ways that matter for this project. It grows trees **leaf-wise** (expand the leaf with the largest loss reduction) instead of level-wise, which is faster and slightly more accurate on medium-sized data. It handles **categorical features natively** by finding the optimal partition of categories at each split, avoiding the one-hot explosion that would hurt columns like `city`.

### `scale_pos_weight`
A LightGBM parameter that multiplies the gradient of positive-class examples by a constant. Set to $n_\text{neg} / n_\text{pos} \approx 10$ in this project. Directly tilts the gradient without modifying the loss function or resampling.

### Native NaN handling
LightGBM treats NaN as a separate value during split finding: for each split, it tries sending NaNs left and right and picks the direction with better loss reduction. This is why we do not impute `age` (60% missing). Imputation would introduce a synthetic value the model cannot distinguish from a real one.

### Early stopping
Stop adding trees when the validation metric has not improved for $k$ rounds. This project uses `early_stopping(50)` during 5-fold CV, which lets each fold find its own best round while preventing overfitting. The final model uses the mean best round across folds.

## Hyperparameter tuning

### Optuna
A Python library for hyperparameter optimisation. You write an objective function that takes a `trial` object, samples hyperparameters from distributions, trains and evaluates a model, and returns a score. Optuna chooses the next trial's parameters using a Bayesian strategy by default (TPE, below), which is more sample-efficient than grid or random search.

### TPE (Tree-structured Parzen Estimator)
Optuna's default sampler. It models the distribution of good trials and the distribution of bad trials separately, then samples the next trial where the ratio of "good" density to "bad" density is highest. Concretely, it is smart enough to avoid re-exploring regions it has already seen perform poorly.

### Uniform vs log-uniform sampling
`num_leaves` sampled uniformly over `[20, 150]` gives each integer equal probability. `learning_rate` sampled log-uniformly over `[0.01, 0.3]` gives each **order of magnitude** equal probability, which is the right prior for a parameter where 0.01 and 0.02 are meaningfully different but 0.20 and 0.21 are not.

### k-fold cross-validation
Split the training data into $k$ folds. Train on $k - 1$ folds and evaluate on the held-out fold. Repeat $k$ times so every sample is held out exactly once. The reported score is the mean across folds. This project uses $k = 5$ during Optuna search.

### Stratified k-fold
Like k-fold but each fold preserves the overall class ratio. At 9% churn, a random split could unluckily land a fold with 5% or 13% churn and give an unreliable score. Stratifying pins the churn rate per fold to 9% exactly.

## Evaluation metrics

### Binary log loss (cross-entropy)
The primary metric for probabilistic classifiers. For a sample with true label $y \in \{0, 1\}$ and predicted probability $p$:

$$\text{LL} = -\left[ y \log p + (1 - y) \log(1 - p) \right]$$

A correct, confident prediction ($p \to y$) yields LL $\to 0$. A wrong, confident prediction yields LL $\to \infty$. Log loss rewards well-calibrated probabilities, not just correct ranking.

### ROC-AUC (Area Under the ROC Curve)
The probability that the model ranks a random positive example higher than a random negative example. Ranges from 0.5 (random) to 1.0 (perfect). It measures **discrimination** (ranking) only and is invariant to class imbalance and calibration.

### PR-AUC (Area Under the Precision-Recall Curve)
The area under the curve of precision (y-axis) vs recall (x-axis) as the decision threshold varies. A random classifier scores equal to the positive-class prevalence (9% here), so any value materially above 0.09 represents learned signal. PR-AUC is more informative than ROC-AUC when the positive class is rare.

### F1 at the optimal threshold
$F_1 = 2 \cdot \text{precision} \cdot \text{recall} / (\text{precision} + \text{recall})$. F1 requires a hard 0/1 prediction, so it requires choosing a threshold. "Optimal" means the threshold that maximises F1 on the validation set (swept via `np.arange(0.05, 0.95, 0.01)`). In production you would choose the threshold by business cost, not by maximising F1.

### Brier score
Mean squared error between predicted probability and true label: $\frac{1}{n}\sum (p_i - y_i)^2$. Like log loss, it penalises miscalibration, but it is bounded on $[0, 1]$ and less punishing of extreme wrong predictions.

### Discrimination vs calibration
**Discrimination** is how well the model *ranks* users by churn risk. ROC-AUC and PR-AUC measure this. **Calibration** is how well the predicted probabilities match reality: if you take all users the model scored at 0.3, about 30% of them should actually churn. Log loss and Brier score measure this. The two can come apart sharply; this project's temporal holdout demonstrates that.

### Reliability diagram
Bin predictions into deciles of predicted probability. For each bin, plot the mean predicted probability (x) vs the actual fraction of churners (y). A perfectly calibrated model lies on the diagonal. Deviations above the diagonal mean the model underpredicts; below means it overpredicts.

## Evaluation protocols

### Random 80/20 split
Shuffle all rows and hold out 20% for testing. Assumes train and test are drawn from the same distribution. Almost always optimistic for time-dependent problems because it mixes past and future.

### Temporal holdout
Train on an earlier time window and test on a later one. In this project, train on Round 1 (users expiring February 2017, 6.4% churn) and test on Round 2 (March 2017, 9.0% churn). Removes the same-distribution assumption and exposes how the model handles drift.

### Distribution drift
When $P(X, y)$ changes between training and deployment. This project sees +2.6 percentage points of churn-rate drift in one month, which is the main reason log loss degrades 3.7x under temporal evaluation.

## Explainability

### SHAP (SHapley Additive exPlanations)
A framework for attributing a model's prediction to each input feature. For each sample, SHAP assigns every feature a signed contribution; the sum of contributions plus a baseline equals the prediction. Derived from Shapley values in cooperative game theory: each feature's contribution is its average marginal effect across all possible feature orderings.

### TreeExplainer
The tree-specific SHAP implementation. For a generic model, exact Shapley computation is exponential in the number of features. TreeExplainer exploits the tree structure to compute exact Shapley values in polynomial time (roughly $O(TLD^2)$ where $T$ is trees, $L$ leaves per tree, $D$ depth). This is why SHAP on LightGBM is fast enough to run on thousands of samples.

### Global vs local explanations
A **global** explanation describes feature behaviour across the entire dataset (e.g., "`last_cancel` is the most important feature overall"). A **local** explanation describes one individual prediction (e.g., "for this specific user, `last_cancel=1` pushed the score up by 0.28"). SHAP produces both.

### Beeswarm plot
One row per feature, one dot per sample. The horizontal axis is the SHAP value (signed contribution). Colour is the feature's raw value (red = high, blue = low). It compresses both importance and directional effect into one chart.

### Waterfall plot
Starts at the baseline prediction (average over the dataset) and stacks each feature's SHAP contribution until it reaches the final prediction for one sample. Useful for explaining an individual decision to a non-technical audience.

## See also

- Evaluation philosophy in practice: [4. Evaluation](4-evaluation.md)
- Why temporal holdout matters: [blog/01-temporal-vs-random-split.md](blog/01-temporal-vs-random-split.md)
