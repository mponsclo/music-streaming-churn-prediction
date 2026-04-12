{{
    config(materialized='table')
}}

-- Aggregate daily listening logs per user.
-- user_logs_v2 covers roughly one month (March 2017), so these
-- aggregates represent the full available listening window.

with logs as (
    select * from {{ ref('stg_user_logs') }}
)

select
    msno,
    count(distinct date) as active_days,
    coalesce(sum(total_secs), 0) as total_secs,
    coalesce(sum(num_100), 0) as complete_plays,
    coalesce(sum(num_unq), 0) as unique_songs,
    coalesce(sum(num_25), 0) as partial_plays,
    coalesce(sum(num_50), 0) as half_plays,
    coalesce(sum(num_75), 0) as three_quarter_plays,
    coalesce(sum(num_985), 0) as near_complete_plays,

    -- Total plays across all completion levels
    coalesce(sum(num_25 + num_50 + num_75 + num_985 + num_100), 0) as total_plays,

    -- Song completion rate: proportion played to 100% (engagement quality)
    {{ safe_divide(
        'sum(num_100)',
        'sum(num_25 + num_50 + num_75 + num_985 + num_100)'
    ) }} as completion_rate,

    -- Engagement depth: unique songs relative to partial listens
    {{ safe_divide('sum(num_unq)', 'sum(num_25)') }} as engagement_depth,

    -- Average daily listening time
    {{ safe_divide('sum(total_secs)', 'count(distinct date)') }} as avg_daily_secs,

    -- Minutes per unique song (listening intensity)
    {{ safe_divide('sum(total_secs) / 60.0', 'sum(num_unq)') }} as mins_per_song

from logs
group by msno
