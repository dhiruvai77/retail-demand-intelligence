import json

import pandas as pd
import pytest

from src.data.curate_parquet import (
    write_partitioned_parquet,
)
from src.data.quality import (
    run_quality_check,
)


def make_quality_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [
                "ITEM_1",
                "ITEM_1",
                "ITEM_2",
            ],
            "store_id": [
                "CA_1",
                "CA_1",
                "CA_1",
            ],
            "date": [
                "2026-01-01",
                "2026-02-01",
                "2026-01-02",
            ],
            "sales": [
                2,
                0,
                0,
            ],
            "sell_price": [
                2.50,
                2.75,
                None,
            ],
            "is_available": [
                1,
                1,
                0,
            ],
        }
    )


def test_quality_check_writes_passed_report(
    tmp_path,
):
    dataset_dir = tmp_path / "curated"
    report_path = (
        tmp_path
        / "quality"
        / "report.json"
    )

    write_partitioned_parquet(
        make_quality_data(),
        dataset_dir,
    )

    report = run_quality_check(
        dataset_dir,
        report_path,
    )

    assert report["status"] == "passed"
    assert report["row_count"] == 3
    assert report["distinct_items"] == 2
    assert report["distinct_stores"] == 1
    assert report["partition_file_count"] == 2

    saved_report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved_report["status"] == "passed"


def test_quality_check_writes_failed_report(
    tmp_path,
):
    dataset_dir = tmp_path / "curated"
    report_path = (
        tmp_path
        / "quality"
        / "report.json"
    )

    dataset_dir.mkdir()

    invalid_data = make_quality_data()
    invalid_data.loc[2, "sales"] = 5

    invalid_data.to_parquet(
        dataset_dir / "part-0.parquet",
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="unavailable",
    ):
        run_quality_check(
            dataset_dir,
            report_path,
        )

    saved_report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved_report["status"] == "failed"
    assert "error" in saved_report