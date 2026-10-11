from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DDL_PATH = (
    PROJECT_ROOT
    / "sql"
    / "athena_schema.sql"
)

QUERIES_PATH = (
    PROJECT_ROOT
    / "sql"
    / "athena_queries.sql"
)


def test_athena_schema_defines_partitioned_parquet_table():
    sql = DDL_PATH.read_text(
        encoding="utf-8"
    ).lower()

    assert "create database if not exists" in sql
    assert "create external table if not exists" in sql
    assert "retail_demand_intelligence.curated_daily_demand" in sql
    assert "partitioned by" in sql
    assert "store_id string" in sql
    assert "year integer" in sql
    assert "month integer" in sql
    assert "stored as parquet" in sql
    assert "s3://<bucket-name>/curated/" in sql


def test_athena_queries_use_catalog_table_and_partitions():
    sql = QUERIES_PATH.read_text(
        encoding="utf-8"
    ).lower()

    assert "msck repair table" in sql
    assert (
        "retail_demand_intelligence.curated_daily_demand"
        in sql
    )
    assert "store_id = 'ca_1'" in sql
    assert "year = 2016" in sql
    assert "month = 5" in sql