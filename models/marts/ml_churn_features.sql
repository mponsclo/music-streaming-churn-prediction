{{
    config(
        materialized='external',
        location=var("project_root") ~ '/outputs/feature_table.parquet',
        format='parquet'
    )
}}

-- Final wide feature table for churn prediction.
-- Joins train labels with member demographics, transaction features,
-- and listening behavior. LEFT JOINs preserve all train users;
-- NULLs from missing joins are filled with 0 where appropriate.

with train as (
    select * from {{ ref('stg_train') }}
),

members as (
    select * from {{ ref('stg_members') }}
),

txn_features as (
    select * from {{ ref('int_transaction_features') }}
),

log_features as (
    select * from {{ ref('int_user_log_features') }}
)

select
    t.msno,
    t.is_churn,

    -- Data presence flags (missingness itself is predictive)
    case when m.msno is not null then 1 else 0 end as has_member_data,
    case when tf.msno is not null then 1 else 0 end as has_transaction_data,
    case when m.age is not null then 1 else 0 end as has_age,

    -- Member features
    m.age,
    m.gender,
    m.city,
    m.registered_via,
    m.tenure_days,

    -- Transaction features: last transaction snapshot
    tf.last_payment_method,
    tf.last_auto_renew,
    tf.last_cancel,
    tf.last_plan_days,
    tf.last_amount_paid,
    tf.last_list_price,
    tf.last_has_discount,
    tf.last_price_per_day,

    -- Transaction features: lifetime aggregates
    coalesce(tf.txn_count, 0) as txn_count,
    coalesce(tf.total_cancels, 0) as total_cancels,
    coalesce(tf.total_auto_renews, 0) as total_auto_renews,
    coalesce(tf.total_discounts, 0) as total_discounts,
    tf.days_until_expiry,

    -- Listening features
    coalesce(lf.active_days, 0) as active_days,
    coalesce(lf.total_secs, 0) as total_secs,
    coalesce(lf.complete_plays, 0) as complete_plays,
    coalesce(lf.unique_songs, 0) as unique_songs,
    coalesce(lf.partial_plays, 0) as partial_plays,
    coalesce(lf.half_plays, 0) as half_plays,
    coalesce(lf.three_quarter_plays, 0) as three_quarter_plays,
    coalesce(lf.near_complete_plays, 0) as near_complete_plays,
    coalesce(lf.total_plays, 0) as total_plays,
    coalesce(lf.completion_rate, 0) as completion_rate,
    coalesce(lf.engagement_depth, 0) as engagement_depth,
    coalesce(lf.avg_daily_secs, 0) as avg_daily_secs,
    coalesce(lf.mins_per_song, 0) as mins_per_song

from train t
left join members m on t.msno = m.msno
left join txn_features tf on t.msno = tf.msno
left join log_features lf on t.msno = lf.msno
