from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.persistence import load_model_bundle
from src.models.train import (
    DATE_COLUMN,
    FORECAST_HORIZON,
    MODEL_FEATURES,
    create_model_matrices,
    create_validation_forecasts,
    evaluate_validation_forecasts,
    prepare_training_splits,
)
from src.models.training_run import (
    get_git_sha,
    load_feature_dataset,
)


def build_test_evaluation_report(
    *,
    model_version: str,
    evaluated_at_utc: str,
    test_df: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
    git_sha: str,
) -> dict[str, object]:
    dates = pd.to_datetime(test_df[DATE_COLUMN])

    return {
        "evaluation_type": "untouched_test_holdout",
        "model_version": model_version,
        "evaluated_at_utc": evaluated_at_utc,
        "git_sha": git_sha,
        "test_rows": int(len(test_df)),
        "available_rows": int(
            test_df["is_available"].eq(1).sum()
        ),
        "test_date_min": dates.min().date().isoformat(),
        "test_date_max": dates.max().date().isoformat(),
        "metrics": metrics,
    }


def evaluate_model_run(
    *,
    features_dir: str | Path,
    model_dir: str | Path,
    validation_days: int = FORECAST_HORIZON,
    test_days: int = FORECAST_HORIZON,
) -> dict[str, object]:
    model_path = Path(model_dir)

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}"
        )

    feature_data = load_feature_dataset(
        features_dir
    )

    prepared = prepare_training_splits(
        feature_data,
        validation_days=validation_days,
        test_days=test_days,
    )

    bundle = load_model_bundle(model_path)

    if bundle["model_features"] != list(MODEL_FEATURES):
        raise ValueError(
            "Saved model feature contract does not match "
            "the current training feature contract."
        )

    _, _, X_test, _ = create_model_matrices(
        prepared["train_df"],
        prepared["test_df"],
    )

    test_forecasts = create_validation_forecasts(
        prepared["test_df"],
        X_test,
        bundle["poisson_model"],
        bundle["classifier_model"],
        threshold=bundle["threshold"],
    )

    metrics = evaluate_validation_forecasts(
        test_forecasts
    )

    evaluated_at = datetime.now(
        timezone.utc
    ).replace(microsecond=0)

    report = build_test_evaluation_report(
        model_version=model_path.name,
        evaluated_at_utc=evaluated_at.isoformat(),
        test_df=prepared["test_df"],
        metrics=metrics,
        git_sha=get_git_sha("."),
    )

    report_path = (
        model_path / "test_evaluation.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    return {
        "report": report,
        "report_path": report_path,
    }