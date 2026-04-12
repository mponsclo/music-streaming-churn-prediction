{{
    config(materialized='view')
}}

select
    msno,
    city,
    case
        when bd between 10 and 70 then bd
        else null
    end as age,
    case
        when gender = 'male' then 'male'
        when gender = 'female' then 'female'
        else 'unknown'
    end as gender,
    registered_via,
    registration_init_time,
    -- Tenure: days from registration to the prediction window (March 2017)
    datediff(
        'day',
        strptime(registration_init_time::varchar, '%Y%m%d'),
        date '2017-03-01'
    ) as tenure_days
from read_csv_auto('{{ var("project_root") }}/data/members_v3.csv')
