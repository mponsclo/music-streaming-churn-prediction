{{
    config(materialized='view')
}}

select
    msno,
    payment_method_id,
    payment_plan_days,
    plan_list_price,
    actual_amount_paid,
    is_auto_renew,
    is_cancel,
    -- Cast integer dates to proper DATE type (try_ variant to handle bad data)
    try_strptime(transaction_date::varchar, '%Y%m%d') as transaction_date,
    try_strptime(membership_expire_date::varchar, '%Y%m%d') as membership_expire_date,
    -- Derived: did the user pay less than list price?
    case
        when actual_amount_paid < plan_list_price then 1
        else 0
    end as has_discount
from read_csv_auto('{{ var("project_root") }}/data/data 3/churn_comp_refresh/transactions_v2.csv')
