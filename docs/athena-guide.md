# Athena Analytics Guide

## Purpose

This project stores curated retail demand data as partitioned Parquet.

The future AWS layout is:

```text
s3://<bucket-name>/curated/
  store_id=CA_1/
    year=2011/
      month=1/
```

## Athena setup

When AWS resources are ready:

1. Upload the local `data/curated` dataset to the S3 `curated/` prefix.
2. Replace `<bucket-name>` in `sql/athena_schema.sql`.
3. Configure the Athena query-results location under:

   `s3://<bucket-name>/athena-results/`

4. Run `sql/athena_schema.sql` in Athena.
5. Run:

```sql
MSCK REPAIR TABLE
retail_demand_intelligence.curated_daily_demand;
```

6. Run the analytical queries in `sql/athena_queries.sql`.

## Partition strategy

The dataset is partitioned by:

- `store_id`
- `year`
- `month`

The original `date` column remains available for filtering and grouping.

Monthly partitions avoid creating thousands of small daily partitions while still allowing Athena to reduce the amount of data scanned.

## Current status

The SQL contract is prepared and tested locally.

Athena execution will happen only after the S3 bucket and AWS access are intentionally configured.