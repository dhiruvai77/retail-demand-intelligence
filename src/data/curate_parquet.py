from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.prepare_data import (
    DEFAULT_STORE_ID,
    prepare_model_data_from_files,
    validate_unavailable_sales,
)


CURATED_REQUIRED_COLUMNS = frozenset(
    {
        "item_id",
        "store_id",
        "date",
        "sales",
        "sell_price",
        "is_available",
    }
)

GRAIN_COLUMNS = (
    "item_id",
    "store_id",
    "date",
)

PARTITION_COLUMNS = (
    "store_id",
    "year",
    "month",
)


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: set[str] | frozenset[str],
) -> None:
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Curated data is missing required columns: "
            f"{missing_list}"
        )


def validate_curated_data(
    df: pd.DataFrame,
) -> None:
    """
    Validate the curated item-store-day data contract.
    """

    _validate_required_columns(
        df,
        CURATED_REQUIRED_COLUMNS,
    )

    if df.empty:
        raise ValueError("Curated data cannot be empty.")

    date_values = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    if date_values.isna().any():
        invalid_count = int(date_values.isna().sum())
        raise ValueError(
            f"Curated data contains {invalid_count:,} invalid dates."
        )

    grain = df.loc[
        :,
        list(GRAIN_COLUMNS),
    ].copy()

    grain["date"] = date_values

    if grain.isna().any(axis=None):
        raise ValueError(
            "Curated grain columns cannot contain null values."
        )

    duplicate_count = int(
        grain.duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "Curated data contains "
            f"{duplicate_count:,} duplicate item-store-date rows."
        )

    sales_values = pd.to_numeric(
        df["sales"],
        errors="coerce",
    )

    invalid_sales = (
        sales_values.isna()
        | sales_values.lt(0)
    )

    if invalid_sales.any():
        invalid_count = int(invalid_sales.sum())
        raise ValueError(
            f"Curated data contains {invalid_count:,} invalid sales values."
        )

    availability_values = pd.to_numeric(
        df["is_available"],
        errors="coerce",
    )

    invalid_availability = (
        availability_values.isna()
        | ~availability_values.isin([0, 1])
    )

    if invalid_availability.any():
        invalid_count = int(invalid_availability.sum())
        raise ValueError(
            "is_available must contain only 0 or 1; "
            f"found {invalid_count:,} invalid values."
        )

    price_values = pd.to_numeric(
        df["sell_price"],
        errors="coerce",
    )

    invalid_prices = (
        df["sell_price"].notna()
        & price_values.isna()
    )

    if invalid_prices.any():
        invalid_count = int(invalid_prices.sum())
        raise ValueError(
            f"Curated data contains {invalid_count:,} invalid prices."
        )

    negative_prices = (
        price_values.notna()
        & price_values.lt(0)
    )

    if negative_prices.any():
        invalid_count = int(negative_prices.sum())
        raise ValueError(
            f"Curated data contains {invalid_count:,} negative prices."
        )

    expected_availability = (
        df["sell_price"]
        .notna()
        .astype("uint8")
    )

    actual_availability = (
        availability_values
        .astype("uint8")
    )

    if not actual_availability.equals(
        expected_availability
    ):
        raise ValueError(
            "is_available must match whether sell_price is present."
        )

    validate_unavailable_sales(
        pd.DataFrame(
            {
                "sales": sales_values,
                "is_available": availability_values,
            }
        )
    )


def add_partition_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add year and month columns used for Parquet partitioning.
    """

    validate_curated_data(df)

    result = df.copy()

    result["date"] = pd.to_datetime(
        result["date"]
    )

    result["year"] = (
        result["date"]
        .dt.year
        .astype("int16")
    )

    result["month"] = (
        result["date"]
        .dt.month
        .astype("int8")
    )

    return result


def write_partitioned_parquet(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """
    Write curated data as a partitioned Parquet dataset.

    Existing output is never deleted automatically.
    """

    partitioned = add_partition_columns(df)

    output_path = Path(output_dir)

    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(
            f"Output path is not a directory: {output_path}"
        )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_entries = list(
        output_path.iterdir()
    )

    if existing_entries:
        raise FileExistsError(
            f"Output directory is not empty: {output_path}. "
            "Choose a new directory or remove the old output manually."
        )

    partitioned.to_parquet(
        output_path,
        engine="pyarrow",
        partition_cols=list(PARTITION_COLUMNS),
        index=False,
    )

    return output_path


def build_curated_parquet(
    raw_data_dir: str | Path,
    output_dir: str | Path,
    store_id: str = DEFAULT_STORE_ID,
) -> tuple[int, Path]:
    """
    Run existing preparation logic and write curated Parquet output.
    """

    curated_data = prepare_model_data_from_files(
        raw_data_dir=raw_data_dir,
        store_id=store_id,
    )

    output_path = write_partitioned_parquet(
        curated_data,
        output_dir=output_dir,
    )

    return len(curated_data), output_path