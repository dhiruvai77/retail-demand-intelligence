from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.curate_parquet import validate_curated_data


def load_curated_parquet(
    dataset_dir: str | Path,
) -> pd.DataFrame:
    dataset_path = Path(dataset_dir)

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


def _build_metrics(
    df: pd.DataFrame,
    dataset_dir: Path,
) -> dict[str, Any]:
    dates = pd.to_datetime(df["date"])
    sales = pd.to_numeric(df["sales"])
    availability = pd.to_numeric(df["is_available"])

    duplicate_grain_rows = int(
        df.assign(date=dates)
        .duplicated(
            subset=[
                "item_id",
                "store_id",
                "date",
            ]
        )
        .sum()
    )

    unavailable_positive_sales = int(
        (
            (availability == 0)
            & (sales > 0)
        ).sum()
    )

    return {
        "row_count": int(len(df)),
        "distinct_items": int(df["item_id"].nunique()),
        "distinct_stores": int(df["store_id"].nunique()),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
        "partition_file_count": len(
            list(dataset_dir.rglob("*.parquet"))
        ),
        "duplicate_item_store_date_rows": (
            duplicate_grain_rows
        ),
        "unavailable_positive_sales_rows": (
            unavailable_positive_sales
        ),
        "available_zero_sales_rows": int(
            (
                (availability == 1)
                & (sales == 0)
            ).sum()
        ),
        "positive_sales_rows": int(
            (sales > 0).sum()
        ),
        "null_counts": {
            str(column): int(count)
            for column, count in df.isna().sum().items()
            if count > 0
        },
    }


def write_quality_report(
    report: dict[str, Any],
    report_path: str | Path,
) -> Path:
    output_path = Path(report_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


def run_quality_check(
    dataset_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)

    report: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "generated_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
    }

    try:
        curated_data = load_curated_parquet(
            dataset_path
        )

        validate_curated_data(
            curated_data
        )

        report.update(
            _build_metrics(
                curated_data,
                dataset_path,
            )
        )

        report["status"] = "passed"

    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)

        write_quality_report(
            report,
            report_path,
        )

        raise

    write_quality_report(
        report,
        report_path,
    )

    return report