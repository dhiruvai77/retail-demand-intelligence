-- Athena analytics queries for curated_daily_demand.


-- Load Hive-style S3 partitions into the Glue Data Catalog.
MSCK REPAIR TABLE
retail_demand_intelligence.curated_daily_demand;


-- 1. Total daily demand

SELECT
    CAST(date AS DATE) AS demand_date,
    store_id,
    SUM(sales) AS total_sales
FROM retail_demand_intelligence.curated_daily_demand
WHERE store_id = 'CA_1'
GROUP BY
    CAST(date AS DATE),
    store_id
ORDER BY demand_date;


-- 2. Demand by product category

SELECT
    cat_id,
    SUM(sales) AS total_sales,
    ROUND(
        100.0 * SUM(sales)
        / SUM(SUM(sales)) OVER (),
        2
    ) AS sales_share_pct
FROM retail_demand_intelligence.curated_daily_demand
WHERE store_id = 'CA_1'
GROUP BY cat_id
ORDER BY total_sales DESC;


-- 3. Availability and demand state

SELECT
    CASE
        WHEN is_available = 0
            THEN 'unavailable'
        WHEN sales = 0
            THEN 'available_zero_sales'
        ELSE 'available_positive_sales'
    END AS demand_state,
    COUNT(*) AS observations
FROM retail_demand_intelligence.curated_daily_demand
WHERE store_id = 'CA_1'
GROUP BY
    CASE
        WHEN is_available = 0
            THEN 'unavailable'
        WHEN sales = 0
            THEN 'available_zero_sales'
        ELSE 'available_positive_sales'
    END
ORDER BY observations DESC;


-- 4. Highest-demand products

SELECT
    item_id,
    dept_id,
    cat_id,
    SUM(sales) AS total_sales,
    ROUND(AVG(sales), 3) AS average_daily_sales
FROM retail_demand_intelligence.curated_daily_demand
WHERE store_id = 'CA_1'
GROUP BY
    item_id,
    dept_id,
    cat_id
ORDER BY total_sales DESC
LIMIT 20;


-- 5. Example partition-pruned query

SELECT
    CAST(date AS DATE) AS demand_date,
    SUM(sales) AS total_sales
FROM retail_demand_intelligence.curated_daily_demand
WHERE store_id = 'CA_1'
  AND year = 2016
  AND month = 5
GROUP BY CAST(date AS DATE)
ORDER BY demand_date;