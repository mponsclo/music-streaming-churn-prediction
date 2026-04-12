{{
    config(materialized='table')
}}

with txns as (
    select * from {{ ref('stg_transactions') }}
),

-- Identify each user's most recent transaction
last_txn as (
    select
        *,
        row_number() over (partition by msno order by transaction_date desc) as rn
    from txns
),

-- Aggregate transaction history per user
per_user_agg as (
    select
        msno,
        count(*) as txn_count,
        sum(is_cancel) as total_cancels,
        sum(is_auto_renew) as total_auto_renews,
        sum(has_discount) as total_discounts,
        max(membership_expire_date) as latest_expire_date,
        max(transaction_date) as latest_txn_date
    from txns
    group by msno
)

select
    lt.msno,

    -- Snapshot of the most recent transaction
    lt.payment_method_id as last_payment_method,
    lt.is_auto_renew as last_auto_renew,
    lt.is_cancel as last_cancel,
    lt.payment_plan_days as last_plan_days,
    lt.actual_amount_paid as last_amount_paid,
    lt.plan_list_price as last_list_price,
    lt.has_discount as last_has_discount,

    -- Derived: price per day on last plan (value signal from top solutions)
    {{ safe_divide('lt.plan_list_price', 'lt.payment_plan_days') }} as last_price_per_day,

    -- Lifetime aggregates
    agg.txn_count,
    agg.total_cancels,
    agg.total_auto_renews,
    agg.total_discounts,

    -- Gap between last transaction date and membership expiry
    datediff('day', agg.latest_txn_date, agg.latest_expire_date) as days_until_expiry

from last_txn lt
inner join per_user_agg agg on lt.msno = agg.msno
where lt.rn = 1
