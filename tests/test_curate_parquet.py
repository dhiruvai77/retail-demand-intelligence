import pandas as pd
import pytest

from src.data.curate_parquet import (
    add_partition_columns,
    validate_curated_data,
    write_partitioned_parquet,
)


def make_curated_data() -> pd.DataFrame:
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


def test_validate_curated_data_accepts_valid_rows():
    validate_curated_data(
        make_curated_data()
    )


def test_add_partition_columns_creates_year_and_month():
    result = add_partition_columns(
        make_curated_data()
    )

    assert set(result["year"]) == {2026}
    assert set(result["month"]) == {1, 2}


def test_validate_curated_data_rejects_duplicate_grain():
    data = make_curated_data()

    duplicate_data = pd.concat(
        [
            data,
            data.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_curated_data(
            duplicate_data
        )


def test_validate_curated_data_rejects_unavailable_sales():
    data = make_curated_data()
    data.loc[2, "sales"] = 1

    with pytest.raises(
        ValueError,
        match="unavailable",
    ):
        validate_curated_data(
            data
        )


def test_write_partitioned_parquet_round_trip(tmp_path):
    output_dir = (
        tmp_path
        / "curated"
    )

    write_partitioned_parquet(
        make_curated_data(),
        output_dir,
    )

    assert (
        output_dir
        / "store_id=CA_1"
    ).is_dir()

    parquet_files = list(
        output_dir.rglob("*.parquet")
    )

    assert parquet_files

    loaded = pd.read_parquet(
        output_dir
    )

    assert len(loaded) == 3
    assert {
        "year",
        "month",
    }.issubset(
        loaded.columns
    )


def test_write_partitioned_parquet_rejects_existing_output(
    tmp_path,
):
    output_dir = (
        tmp_path
        / "curated"
    )

    output_dir.mkdir()
    (
        output_dir
        / "existing.txt"
    ).write_text(
        "existing output",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError
    ):
        write_partitioned_parquet(
            make_curated_data(),
            output_dir,
        )