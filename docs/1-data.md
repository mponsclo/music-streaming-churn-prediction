# 1. Data

Raw CSVs -> DuckDB -> schema audit + class balance + quality issues uncovered.

## Why This Layer Exists

Before any feature engineering or modeling, the data has to be understood on its own terms: what the rows mean, how the tables join, where the gaps are, which signals are real and which are artefacts. The KKBox release is a messy public dataset with two refreshes (v1 and v2), overlapping tables, a narrow behavioural window, and silent failure modes like 60% missing age. Skipping the schema audit would mean silently inheriting every quality issue as a feature later.

This guide covers the v2 refresh files, the class balance, the three missingness groups discovered by joining train to the other tables, and a handful of data-quality issues that shape the feature design. Every number here is reproducible from the EDA notebook.

Source files live in `data/` (gitignored; 32 GB total). Download from the [Kaggle competition page](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data).

Interactive exploration: [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb).

## Key Concepts

### DuckDB as the CSV engine

`user_logs_v2.csv` is 1.3 GB with 18.4 million rows. Loading it into pandas eats 10+ GB of RAM and takes minutes. DuckDB reads the file column-by-column, streams it through the query, and never materialises the full table:

```python
import duckdb

duckdb.sql("""
    SELECT msno,
           COUNT(DISTINCT date) AS active_days,
           SUM(total_secs) AS total_secs
    FROM read_csv_auto('data/user_logs_v2.csv')
    GROUP BY msno
""").df()
```

The exact same aggregation that would OOM in pandas runs in about 12 seconds. This is why ingestion and aggregation both run through DuckDB throughout the project, and why the dbt layer (see [2. Features](2-features.md)) uses the `dbt-duckdb` adapter.

### The v1 vs v2 refresh

The WSDM Cup 2018 competition shipped original files ("Round 1") and later added a refresh ("Round 2"). This project uses Round 2 (users whose subscriptions expire in **March 2017**) as the canonical dataset, and uses Round 1 (February 2017 expiries) only for the temporal holdout in [4. Evaluation](4-evaluation.md). Mixing the two without care would silently leak test data into training.

## Schema overview

Four source files. The v2 refresh covers users whose subscriptions expired in March 2017.

| Table | Rows | Date range | Coverage vs train |
|-------|------|------------|-------------------|
| `train_v2.csv` | 970,960 | N/A (labels only) | 100% |
| `members_v3.csv` | 6,769,473 | Registration: 2004-2017 | 88.7% matched |
| `transactions_v2.csv` | 1,431,009 | 2015-01 to 2017-03 | 96.2% matched |
| `user_logs_v2.csv` | 18,396,362 | 2017-03 only | 77.7% matched |

**Critical constraint**: `user_logs_v2` covers exactly one month (March 2017). Multi-window behavioural features (30-day, 60-day, 90-day windows) are not available from this file. The "30-day aggregate" is the full listening history we can use.

## Class balance

- Retained: 883,630 (91.0%)
- Churned: 87,330 (9.0%)
- Imbalance ratio: ~10:1

Moderate imbalance. Handled by LightGBM's `scale_pos_weight` (see [3. Modeling](3-modeling.md) and the [glossary](0-glossary.md#scale_pos_weight)).

## The dominant signal: subscription status

The strongest single predictor is the `auto_renew x cancel` interaction. This is visible in the raw data before any feature engineering runs.

| `auto_renew` | `cancel` | Users | Churn rate |
|--------------|----------|-------|------------|
| 1 | 0 | 827,920 | 1.76% |
| 1 | 1 | 22,745 | 78.88% |
| 0 | 0 | 82,913 | 30.56% |
| No txn data | N/A | 37,382 | 78.74% |

85.3% of users are on auto-renew with no cancellation and churn at 1.76%. The remaining ~15% are at sharply higher risk.

**This is not data leakage**. `is_cancel` records a deliberate user action (cancelling auto-renew or the subscription itself). A user can cancel and re-subscribe, so cancellation does not deterministically cause churn. But it is the strongest behavioural signal available, and any honest model will lean on it heavily. The consequences of this for modeling (and why we ablate those features later) are discussed in [3. Modeling](3-modeling.md#ablation-studies).

## Plan duration pattern

97.9% of users are on 30-day plans. The remaining 2.1% are on 90/180/365/410/195-day plans and churn at 96.7%.

- 100% of non-30-day-plan users have `auto_renew=0`
- These are promotional or annual plans purchased once
- When they expire the user must actively re-subscribe
- `last_plan_days` correlates $r = 0.45$ with churn but is essentially a proxy for `auto_renew`

Expect LightGBM to use `last_plan_days` and `last_auto_renew` somewhat interchangeably during tree splits.

## Missing data patterns

Three distinct missing-data groups emerge from left-joining train to the other three tables. These are visible by running the join audit in the EDA notebook.

| Group | Users | % | Churn rate | Missing |
|-------|-------|---|------------|---------|
| Full data (all 4 tables) | 725,722 | 74.7% | 6.4% | Nothing |
| No member data | 109,993 | 11.3% | varies | age, gender, city, tenure |
| No transaction data | 37,382 | 3.9% | 78.7% | All txn features |
| No user logs | 216,409 | 22.3% | 9.1% | All listening features |

Two things stand out, and they drive two concrete feature decisions downstream:

- **The 37K users without transactions are critical**: their 78.7% churn rate means missingness itself is a strong feature. We surface this as a `has_transaction_data` flag in [2. Features](2-features.md#data-presence-flags-3).
- **The 216K users without logs churn at the baseline rate** (9.1% vs 9.0% overall). Listening-data absence is NOT a churn signal, so we `COALESCE` those aggregates to zero rather than adding a flag.

This is the kind of call that cannot be made without the missingness audit. Blindly imputing zeros everywhere would have hidden the transaction-missingness signal.

## Feature quality issues

### Negative `tenure_days` (758 users)
Users registered after the 2017-03-01 reference date. Minimum: -54 days. These are very recent signups. Churn rate: 63.2%. Keep as-is since negative tenure is meaningful (very new users) and LightGBM handles the sign.

### Negative `days_until_expiry` (2,178 users)
Last transaction date is after expire date. This is a renewal that happened after a brief lapse. Churn rate: 86.5%. Keep as-is.

### `last_has_discount` nearly constant
Only 1,707 users (0.18%) have a discount. Near-zero variance. LightGBM will effectively ignore it; we leave it in rather than curate the feature set manually.

### `engagement_depth` extremely skewed
Mean 8.17, median 3.45, max 9,525. Tree models are invariant to monotonic transforms, but extreme values can produce suboptimal splits. Optional `log1p` transform is available but not currently applied.

### 60% missing `age`
After capping `bd` outside `[10, 70]` to NULL, 584,828 users have no age. Add a `has_age` flag. LightGBM handles NaN natively for splits (see [glossary](0-glossary.md#native-nan-handling)), so imputation would hurt more than help.

## Feature relevance: "normal" subscribers

Among the 828K `auto_renew=1, cancel=0` users (1.76% churn), the top predictors by Pearson correlation:

| Feature | \|r\| with churn |
|---------|----------------|
| `days_until_expiry` | 0.342 |
| `txn_count` | 0.236 |
| `total_auto_renews` | 0.234 |
| `last_has_discount` | 0.124 |
| `total_discounts` | 0.079 |
| `last_list_price` | 0.071 |
| `total_cancels` | 0.057 |
| `tenure_days` | 0.055 |
| `active_days` | 0.012 |
| `total_secs` | 0.011 |

Listening volume has almost zero predictive power within this segment. Transaction history and subscription structure dominate even among retention-stable users.

Counterintuitively, among normal subscribers, churners listen *slightly more* than retained users (median 56,528 vs 44,924 total seconds). Listening volume does not indicate loyalty within the retention-stable segment. This observation motivates the behavioural-only ablation in [3. Modeling](3-modeling.md#ablation-studies), which quantifies exactly how much signal you give up by ignoring transaction history.

## See also

- Glossary of every term used above: [0. Glossary](0-glossary.md)
- EDA notebook (interactive plots and joins): [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb)
- Feature catalog and dbt pipeline: [2. Features](2-features.md)
- Modeling strategy and the behavioural-only ablation: [3. Modeling](3-modeling.md)
