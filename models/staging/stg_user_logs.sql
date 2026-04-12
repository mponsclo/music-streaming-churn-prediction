{{
    config(materialized='table')
}}

-- Materialized as a table (not view) because the source CSV is 1.3 GB / 18.3M rows.
-- Reading it once into DuckDB's columnar format avoids re-parsing on every
-- downstream reference and makes the aggregation in int_user_log_features fast.

select
    msno,
    date,
    num_25,
    num_50,
    num_75,
    num_985,
    num_100,
    num_unq,
    total_secs
from read_csv_auto('{{ var("project_root") }}/data/data 4/churn_comp_refresh/user_logs_v2.csv')
