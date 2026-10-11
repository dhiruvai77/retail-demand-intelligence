import pandas as pd
import pytest

from src.models.batch_inference import (
    prediction_column,
    prepare_batch_features,
    select_latest_horizon,
)
from src.models.train import (
    CATEGORICAL_FEATURES,
)


def test_select_latest_horizon():
    dates = pd.date_range(
        "2026-01-01",
        periods=5,
        freq="D",
    )

    df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "item_id": (
                ["ITEM_1"] * 5
                + ["ITEM_2"] * 5
            ),
        }
    )

    result = select_latest_horizon(
        df,
        horizon=2,
    )

    assert result["date"].nunique() == 2
    assert len(result) == 4
    assert result["date"].min() == dates[3]


def test_prediction_column():
    assert prediction_column(
        "poisson"
    ) == "poisson_prediction"

    assert prediction_column(
        "two_stage"
    ) == "two_stage_prediction"

    with pytest.raises(ValueError):
        prediction_column("unknown")


def test_prepare_batch_features_encodes_categories():
    data = {
        column: ["value"]
        for column in CATEGORICAL_FEATURES
    }

    data["event_name_1"] = [None]
    data["event_type_1"] = [None]
    data["event_name_2"] = [None]
    data["event_type_2"] = [None]
    data["date"] = ["2026-01-01"]
    data["is_available"] = [1]

    result = prepare_batch_features(
        pd.DataFrame(data)
    )

    for column in CATEGORICAL_FEATURES:
        assert str(
            result[column].dtype
        ) == "category"

    assert result.loc[
        0,
        "event_name_1",
    ] == "NoEvent"