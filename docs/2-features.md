# 2. Features

Raw CSVs -> staging views -> intermediate tables -> one wide Parquet mart (36 features).

## Why This Layer Exists

The modeling code should not be in the business of parsing dates, casting columns, computing per-user aggregates, or guarding against division by zero. All of that belongs in a transformation layer, and the transformation layer should be SQL-native, version-controlled, and tested independently.

A bag of ad-hoc SQL scripts or pandas notebooks would produce the same numbers, but two things break quickly: lineage (who depends on what) and testability (does the `mins_per_song` formula survive a schema change in `user_logs`?). `dbt` solves both. Each transformation is a standalone SQL file that declares its inputs via `{{ ref('...') }}`. dbt compiles those references into a DAG, runs the models in the right order, and runs schema tests against the outputs.

The layer also enforces a clean separation: staging is cheap (views over raw files), intermediate is where aggregation happens (materialised tables), and marts are the consumable outputs. Modeling reads one Parquet file, never the raw CSVs.

## Key Concepts

### dbt in one paragraph
You write SQL files under `models/`. Each file is one `SELECT` statement. When a file needs another model as input, you write `FROM {{ ref('stg_transactions') }}` instead of naming a table directly. dbt parses all files, builds a dependency DAG, runs `CREATE VIEW / CREATE TABLE / COPY TO` in the right order, and runs schema tests declared in sibling YAML files. The adapter (here `dbt-duckdb`) translates these into concrete SQL against the target database.

### Staging -> intermediate -> marts
A widely-used convention, borrowed from the dbt community. Keeping the three layers separate has a practical payoff:

- **Staging** is where "one column name per source file" problems live. Renames, type casts, date parsing. No joins, no aggregations.
- **Intermediate** is where "one row per user" problems live. Aggregations, window functions, conditional logic. Expensive, so materialised as tables.
- **Marts** is where "what downstream systems want" lives. Joins across intermediate models, feature flags, final column shaping. Exported to Parquet.

A reader trying to debug a bad feature can always follow the layering: "Is the bug in how the source was cleaned (staging), in how it was aggregated (intermediate), or in the final join (marts)?" Without the convention, that question has no home.

### `dbt-duckdb` and the lineage

DuckDB is the target database, so every `CREATE VIEW` / `CREATE TABLE` runs locally against `churn_pred.duckdb`. The marts model uses DuckDB's `COPY TO` to write `outputs/feature_table.parquet` directly, without going through pandas. Build the full graph once:

```bash
make dbt-build
```

This runs `dbt run` (builds all models) followed by `dbt test` (runs 16 schema tests). The full graph takes about 16 seconds on a laptop.

### `safe_divide` macro

Several listening-behaviour features are ratios. Any user who opened the app but played no songs hits divide-by-zero. Rather than sprinkle `CASE WHEN ... > 0` across every SQL file, we define it once as a Jinja macro:

```sql
{% macro safe_divide(numerator, denominator) %}
    CASE WHEN {{ denominator }} > 0
        THEN {{ numerator }}::double / {{ denominator }}::double
        ELSE 0
    END
{% endmacro %}
```

Called via `{{ safe_divide('complete_plays', 'total_plays') }}` in `int_user_log_features.sql`. The cast to `double` is defensive: DuckDB's integer division would truncate without it.

## Pipeline layering

```
staging (views)           -> intermediate (tables)         -> marts (parquet)

stg_train.sql             \
stg_members.sql            \
stg_transactions.sql        -> int_transaction_features.sql \
stg_user_logs.sql           -> int_user_log_features.sql     -> ml_churn_features.sql
```

- **Staging (views)**: one per source file. Column casting, date parsing, filtering to the v2 refresh window. Views keep staging cheap on a DuckDB backend because they are just query definitions, evaluated lazily.
- **Intermediate (tables)**: per-user aggregations that are expensive to recompute. Transactions aggregated to one row per `msno` (counts, sums, last-transaction snapshot via `ROW_NUMBER()`). User logs aggregated to one row per `msno` with listening volume, completion rates, and engagement depth.
- **Marts (parquet)**: the single wide feature table `ml_churn_features` exported to `outputs/feature_table.parquet`. This is what Python modeling code consumes via `pd.read_parquet(...)`.

7 models total, 16 schema tests. `dbt-docs` serves an interactive lineage graph:

```bash
make dbt-docs    # localhost:8080
```

## Feature catalog (36 features)

Every feature produced by the mart, grouped by source. All categoricals are kept as their raw IDs so LightGBM can partition them natively (see [glossary: LightGBM](0-glossary.md#lightgbm)).

### Member features (5)
| Feature | Description |
|---------|-------------|
| `age` | Clipped to `[10, 70]`, NULL otherwise |
| `gender` | `male` / `female` / `unknown` |
| `city` | Registration city ID (categorical) |
| `registered_via` | Registration channel ID (categorical) |
| `tenure_days` | Days from `registration_init_time` to 2017-03-01 |

### Transaction snapshot (8)
The "last transaction" features come from a `ROW_NUMBER() OVER (PARTITION BY msno ORDER BY transaction_date DESC)` window, taking row 1 per user.

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

### Listening behaviour (12)
All aggregated over March 2017 (the only month available, see [1. Data](1-data.md#schema-overview)).

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
| `completion_rate` | `complete_plays / total_plays` (via `safe_divide`) |
| `engagement_depth` | `unique_songs / partial_plays` (exploration vs skip-heavy) |
| `avg_daily_secs` | `total_secs / active_days` |
| `mins_per_song` | `(total_secs / 60) / unique_songs` |

### Data presence flags (3)
Derived directly from the missingness audit in [1. Data](1-data.md#missing-data-patterns):

| Feature | Description |
|---------|-------------|
| `has_member_data` | User exists in `members_v3.csv` |
| `has_transaction_data` | User has any transaction (the 37K high-risk group) |
| `has_age` | Age is non-null after outlier handling |

## Schema tests

Every schema test lives in a YAML file next to its model. Examples:

- `msno` is `not_null` and `unique` in every staging and intermediate model.
- `is_churn` accepts `[0, 1]` only.
- Intermediate aggregates have bounded ranges (e.g., `completion_rate` in `[0, 1]`).
- Foreign-key-style `relationships` tests check that every `msno` in `int_transaction_features` exists in `stg_members` or `stg_train`.

16 tests total. Run independently via:

```bash
make dbt-test
```

Failures block the build: if a schema test fails, the mart is not written.

## See also

- The missingness audit that motivates the presence flags: [1. Data](1-data.md#missing-data-patterns)
- How LightGBM consumes these features (and why categoricals stay raw): [3. Modeling](3-modeling.md)
- Glossary of dbt and DuckDB concepts: [0. Glossary](0-glossary.md#data-layer)
