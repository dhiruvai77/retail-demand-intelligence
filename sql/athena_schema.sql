-- Retail Demand Intelligence
-- Athena / AWS Glue Data Catalog schema
--
-- Replace <bucket-name> with the real S3 bucket name later.
-- This table expects Hive-style partitions:
--
-- store_id=CA_1/year=2011/month=1/

CREATE DATABASE IF NOT EXISTS retail_demand_intelligence;


CREATE EXTERNAL TABLE IF NOT EXISTS
retail_demand_intelligence.curated_daily_demand (
    id string,
    item_id string,
    dept_id string,
    cat_id string,
    state_id string,
    d string,

    sales integer,
    wm_yr_wk integer,
    date timestamp,

    weekday string,
    wday integer,

    snap_CA integer,
    snap_TX integer,
    snap_WI integer,

    event_name_1 string,
    event_type_1 string,
    event_name_2 string,
    event_type_2 string,

    sell_price double,
    is_available integer
)
PARTITIONED BY (
    store_id string,
    year integer,
    month integer
)
STORED AS PARQUET
LOCATION 's3://<bucket-name>/curated/'
TBLPROPERTIES (
    'classification' = 'parquet'
);