import pandas as pd

from src.data.curate_parquet import (
    write_partitioned_parquet,
)
from src.features.feature_dataset import (
    build_feature_dataset,
)


def make_curated_input() -> pd.DataFrame:
    dates = pd.date_range(
        "2026-01-01",
        periods=70,
        freq="D",
    )

    return pd.DataFrame(
        {
            "item_id": ["ITEM_1"] * 70,
            "store_id": ["CA_1"] * 70,
            "date": dates,
            "sales": [
                index % 5
                for index in range(70)
            ],
            "sell_price": [2.50] * 70,
            "is_available": [1] * 70,
            "event_name_1": [None] * 70,
            "event_type_1": [None] * 70,
            "event_name_2": [None] * 70,
            "event_type_2": [None] * 70,
        }
    )


def test_build_feature_dataset_writes_features(
    tmp_path,
):
    curated_dir = tmp_path / "curated"
    features_dir = tmp_path / "features"

    write_partitioned_parquet(
        make_curated_input(),
        curated_dir,
    )

    row_count, column_count, output_path = (
        build_feature_dataset(
            curated_dir=curated_dir,
            output_dir=features_dir,
        )
    )

    assert row_count == 70
    assert column_count > 20
    assert output_path == features_dir

    feature_data = pd.read_parquet(
        features_dir
    )

    assert len(feature_data) == 70
    assert "sales_lag_28" in feature_data.columns
    assert "rolling_mean_7_lag28" in feature_data.columns
    assert "log_price_ratio_28" in feature_data.columns