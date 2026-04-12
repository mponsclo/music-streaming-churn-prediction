{{
    config(materialized='view')
}}

select
    msno,
    is_churn
from read_csv_auto('{{ var("project_root") }}/data/data 2/churn_comp_refresh/train_v2.csv')
