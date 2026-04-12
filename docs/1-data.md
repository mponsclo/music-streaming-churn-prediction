# 1. Data

The dataset is the [WSDM Cup 2018 KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) v2 refresh. KKBox is Asia's largest music streaming service; the target is whether a user renews within 30 days of subscription expiry.

## Schema overview

Four source files. The v2 refresh covers users whose subscriptions expired in March 2017.

| Table | Rows | Date range | Coverage vs train |
|-------|------|------------|-------------------|
| `train_v2.csv` | 970,960 | N/A (labels only) | 100% |
| `members_v3.csv` | 6,769,473 | Registration: 2004-2017 | 88.7% matched |
| `transactions_v2.csv` | 1,431,009 | 2015-01 to 2017-03 | 96.2% matched |
| `user_logs_v2.csv` | 18,396,362 | 2017-03 only | 77.7% matched |

**Critical constraint**: `user_logs_v2` covers exactly one month (March 2017). Multi-window behavioral features (30-day, 60-day, 90-day windows) are not available from this file. The "30-day aggregate" is the full listening history we can use.

## Class balance

- Retained: 883,630 (91.0%)
- Churned: 87,330 (9.0%)
- Imbalance ratio: ~10:1

Moderate imbalance, handled by LightGBM's `scale_pos_weight` (see [3. Modeling](3-modeling.md)).

## The dominant signal: subscription status

The strongest single predictor is the `auto_renew x cancel` interaction.

| `auto_renew` | `cancel` | Users | Churn rate |
|--------------|----------|-------|------------|
| 1 | 0 | 827,920 | 1.76% |
| 1 | 1 | 22,745 | 78.88% |
| 0 | 0 | 82,913 | 30.56% |
| No txn data | N/A | 37,382 | 78.74% |

85.3% of users are on auto-renew with no cancellation and churn at 1.76%. The remaining ~15% are at sharply higher risk.

**This is not data leakage**. `is_cancel` records a deliberate user action (canceling auto-renew or the subscription itself). A user can cancel and re-subscribe, so cancellation does not deterministically cause churn. But it is the strongest behavioral signal available, and any honest model will lean on it heavily.

## Plan duration pattern

97.9% of users are on 30-day plans. The remaining 2.1% are on 90/180/365/410/195-day plans and churn at 96.7%.

- 100% of non-30-day-plan users have `auto_renew=0`
- These are promotional or annual plans purchased once
- When they expire the user must actively re-subscribe
- `last_plan_days` correlates r=0.45 with churn but is essentially a proxy for `auto_renew`

Expect LightGBM to use `last_plan_days` and `last_auto_renew` somewhat interchangeably.

## Missing data patterns

Three distinct missing-data groups emerge from left-joining train to the other three tables.

| Group | Users | % | Churn rate | Missing |
|-------|-------|---|------------|---------|
| Full data (all 4 tables) | 725,722 | 74.7% | 6.4% | Nothing |
| No member data | 109,993 | 11.3% | varies | age, gender, city, tenure |
| No transaction data | 37,382 | 3.9% | 78.7% | All txn features |
| No user logs | 216,409 | 22.3% | 9.1% | All listening features |

Two things stand out:

- **The 37K users without transactions are critical**: their 78.7% churn rate means the missingness itself is a strong feature. We add a `has_transaction_data` flag.
- **The 216K users without logs churn at the baseline rate** (9.1% vs 9.0% overall). Listening-data absence is NOT a churn signal, so COALESCE-to-zero is appropriate.

## Feature quality issues

### Negative `tenure_days` (758 users)
Users registered after the 2017-03-01 reference date. Minimum: -54 days. These are very recent signups. Churn rate: 63.2%. Keep as-is since negative tenure is meaningful (very new users) and LightGBM handles the sign.

### Negative `days_until_expiry` (2,178 users)
Last transaction date is after expire date. This is a renewal that happened after a brief lapse. Churn rate: 86.5%. Keep as-is.

### `last_has_discount` nearly constant
Only 1,707 users (0.18%) have a discount. Near-zero variance; consider dropping.

### `engagement_depth` extremely skewed
Mean 8.17, median 3.45, max 9,525. Tree models are invariant to monotonic transforms, but extreme values can produce suboptimal splits. Optional `log1p` transform.

### 60% missing `age`
After capping `bd` outside [10, 70] to NULL, 584,828 users have no age. Add a `has_age` flag; LightGBM handles NaN natively for splits.

## Feature relevance: "normal" subscribers

Among the 828K `auto_renew=1, cancel=0` users (1.76% churn), the top predictors by correlation:

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

Listening volume has almost zero predictive power for the normal-subscriber segment. Transaction history and subscription structure dominate even among retention-stable users.

Counterintuitively, among normal subscribers churners listen *slightly more* than retained users (median 56,528 vs 44,924 total seconds). Listening volume does not indicate loyalty within the retention-stable segment.

## See also

- EDA notebook: [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb)
- Feature catalog: [2. Features](2-features.md)
- Modeling strategy and the behavioral-only ablation: [3. Modeling](3-modeling.md)
