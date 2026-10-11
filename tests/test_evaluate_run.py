import pandas as pd

from src.models.evaluate_run import (
    build_test_evaluation_report,
)


def test_build_test_evaluation_report():
    test_df = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-04-25",
                periods=3,
                freq="D",
            ),
            "is_available": [1, 0, 1],
        }
    )

    report = build_test_evaluation_report(
        model_version="run-test",
        evaluated_at_utc=(
            "2026-10-11T18:00:00+00:00"
        ),
        test_df=test_df,
        metrics={
            "poisson": {
                "MAE": 1.1,
            },
            "two_stage": {
                "MAE": 1.0,
            },
        },
        git_sha="abc123",
    )

    assert report["evaluation_type"] == (
        "untouched_test_holdout"
    )
    assert report["model_version"] == "run-test"
    assert report["test_rows"] == 3
    assert report["available_rows"] == 2
    assert report["test_date_min"] == "2026-04-25"
    assert report["test_date_max"] == "2026-04-27"
    assert report["git_sha"] == "abc123"