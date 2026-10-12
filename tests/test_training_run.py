from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.train import MODEL_FEATURES
from src.models.training_run import (
    build_training_manifest,
    create_model_version,
    summarize_split,
)


def make_split(
    start: str,
    periods: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(
                start,
                periods=periods,
                freq="D",
            ),
        }
    )


def test_create_model_version_is_filesystem_safe():
    created_at = datetime(
        2026,
        10,
        11,
        18,
        30,
        tzinfo=timezone.utc,
    )

    assert create_model_version(
        created_at
    ) == "run-20261011T183000Z"


def test_summarize_split_returns_rows_and_dates():
    summary = summarize_split(
        make_split(
            "2026-01-01",
            3,
        )
    )

    assert summary == {
        "rows": 3,
        "date_min": "2026-01-01",
        "date_max": "2026-01-03",
    }


def test_build_training_manifest_records_metadata():
    feature_data = make_split(
        "2026-01-01",
        5,
    )

    training_result = {
        "train_df": make_split(
            "2026-01-01",
            2,
        ),
        "validation_df": make_split(
            "2026-01-03",
            2,
        ),
        "test_df": make_split(
            "2026-01-05",
            1,
        ),
        "validation_metrics": {
            "two_stage": {
                "MAE": 1.0,
                "WMAPE": 0.5,
            },
        },
    }

    manifest = build_training_manifest(
        model_version="run-test",
        created_at_utc=(
            "2026-10-11T18:30:00+00:00"
        ),
        feature_data=feature_data,
        training_result=training_result,
        source_dataset="data/features",
        validation_days=28,
        test_days=28,
        threshold=0.50,
        early_stopping_rounds=50,
        git_sha="abc123",
        artifact_paths={
            "poisson_model": Path(
                "run-test/poisson_model.joblib"
            ),
        },
    )

    assert manifest["model_version"] == "run-test"
    assert manifest["git_sha"] == "abc123"
    assert manifest[
        "model_feature_count"
    ] == len(MODEL_FEATURES)
    assert manifest["test_evaluated"] is False
    assert manifest[
        "validation_metrics"
    ]["two_stage"]["WMAPE"] == 0.5
    assert manifest[
        "artifacts"
    ]["poisson_model"] == (
        "poisson_model.joblib"
    )