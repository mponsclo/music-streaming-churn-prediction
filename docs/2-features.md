# 2. Features

The feature engineering layer is a [dbt](https://www.getdbt.com/) project with the [dbt-duckdb](https://github.com/duckdb/dbt-duckdb) adapter. All transformations are SQL-native, version-controlled, and tested with dbt schema tests.

## Pipeline layering

```
staging (views)           -> intermediate (tables)         -> marts (parquet)

stg_train.sql             \
stg_members.sql            \
stg_transactions.sql        -> int_transaction_features.sql \
stg_user_logs.sql           -> int_user_log_features.sql     -> ml_churn_features.sql
```

- **Staging (views)**: one per source file. Column casting, date parsing, filtering to the v2 refresh window. Views keep staging cheap on a DuckDB backend because they are just query definitions.
- **Intermediate (tables)**: per-user aggregations that are expensive to recompute. Transactions aggregated to one row per `msno` (counts, sums, last-transaction snapshot). User logs aggregated to one row per `msno` with listening volume, completion rates, and engagement depth.
- **Marts (parquet)**: the single wide feature table `ml_churn_features` exported to `outputs/feature_table.parquet`. This is what Python modeling code consumes.

7 models total, 16 schema tests. Full build runs in ~16 seconds on a laptop.

## Feature catalog (36 features)

### Member features (5)
| Feature | Description |
|---------|-------------|
| `age` | Clipped to [10, 70], NULL otherwise |
| `gender` | `male` / `female` / `unknown` |
| `city` | Registration city ID (categorical) |
| `registered_via` | Registration channel ID (categorical) |
| `tenure_days` | Days from `registration_init_time` to 2017-03-01 |

### Transaction snapshot (8)
| Feature | Description |
|---------|-------------|
| `last_payment_method` | Payment method of the most recent transaction (categorical) |
| `last_auto_renew` | 0/1, auto-renewal status of latest transaction |
| `last_cancel` | 0/1, cancel flag of latest transaction |
| `last_plan_days` | Plan duration of latest transaction (30/90/180/365/410) |
| `last_amount_paid` | Amount paid in latest transaction |
| `last_list_price` | List price of latest plan |
| `last_has_discount` | 0/1, paid less than list price |
| `last_price_per_day` | `list_price / plan_days` |

### Transaction aggregates (5)
| Feature | Description |
|---------|-------------|
| `txn_count` | Total transactions per user |
| `total_cancels` | Sum of `is_cancel` across transactions |
| `total_auto_renews` | Sum of `is_auto_renew` across transactions |
| `total_discounts` | Count of discounted transactions |
| `days_until_expiry` | Days from last transaction to latest membership expiry |

### Listening behavior (12)
All aggregated over March 2017 (the only month available).

| Feature | Description |
|---------|-------------|
| `active_days` | Distinct days with listening activity |
| `total_secs` | Sum of seconds listened |
| `complete_plays` | `num_100` (plays at 100% completion) |
| `unique_songs` | Distinct songs played |
| `partial_plays` | `num_25` (quit before 25%) |
| `half_plays` | `num_50` |
| `three_quarter_plays` | `num_75` |
| `near_complete_plays` | `num_985` |
| `total_plays` | Sum across all play-percentage buckets |
| `completion_rate` | `complete_plays / total_plays` |
| `engagement_depth` | `unique_songs / partial_plays` (exploration vs skip-heavy) |
| `avg_daily_secs` | `total_secs / active_days` |
| `mins_per_song` | `(total_secs / 60) / unique_songs` |

### Data presence flags (3)
These emerged from the data review (see [1. Data](1-data.md)):

| Feature | Description |
|---------|-------------|
| `has_member_data` | User exists in `members_v3.csv` |
| `has_transaction_data` | User has any transaction (the 37K high-risk group) |
| `has_age` | Age is non-null after outlier handling |

## Macros

`macros/safe_divide.sql` provides a division-by-zero guard used throughout `int_user_log_features.sql` (completion_rate, engagement_depth, avg_daily_secs, mins_per_song).

```sql
{% macro safe_divide(numerator, denominator) %}
    CASE WHEN {{ denominator }} > 0
        THEN {{ numerator }}::double / {{ denominator }}::double
        ELSE 0
    END
{% endmacro %}
```

## Running the pipeline

```bash
make dbt-build     # dbt run + dbt test
make dbt-docs      # serve auto-generated docs at localhost:8080
```

Schema tests enforce non-null `msno`/`is_churn`, referential integrity between staging and intermediate layers, and bounded value ranges on key aggregates.
