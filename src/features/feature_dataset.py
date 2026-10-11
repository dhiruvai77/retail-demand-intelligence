from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.curate_parquet import (
    validate_curated_data,
    write_partitioned_parquet,
)
from src.features.build_features import (
    build_features,
    validate_daily_product_history,
)


def load_curated_dataset(
    curated_dir: str | Path,
) -> pd.DataFrame:
    dataset_path = Path(curated_dir)

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Curated dataset directory not found: {dataset_path}"
        )

    parquet_files = list(
        dataset_path.rglob("*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in: {dataset_path}"
        )

    return pd.read_parquet(dataset_path)


def build_feature_dataset(
    curated_dir: str | Path,
    output_dir: str | Path,
    horizon: int = 28,
) -> tuple[int, int, Path]:
    """
    Read curated Parquet, build leakage-safe features,
    and write partitioned feature Parquet.
    """

    curated_data = load_curated_dataset(
        curated_dir
    )

    validate_curated_data(
        curated_data
    )

    validate_daily_product_history(
        curated_data
    )

    feature_data = build_features(
        curated_data,
        horizon=horizon,
    )

    output_path = write_partitioned_parquet(
        feature_data,
        output_dir,
    )

    return (
        len(feature_data),
        len(feature_data.columns),
        output_path,
    )