from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.persistence import load_model_bundle
from src.models.predict import generate_forecasts
from src.models.train import (
    CATEGORICAL_FEATURES,
    encode_event_missingness,
)
from src.models.training_run import get_git_sha


PREDICTION_COLUMNS = {
    "poisson": "poisson_prediction",
    "two_stage": "two_stage_prediction",
}


def load_batch_dataset(
    features_dir: str | Path,
) -> pd.DataFrame:
    dataset_path = Path(features_dir)

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Feature dataset directory not found: {dataset_path}"
        )

    if not list(dataset_path.rglob("*.parquet")):
        raise FileNotFoundError(
            f"No Parquet files found in: {dataset_path}"
        )

    return pd.read_parquet(dataset_path)


def select_latest_horizon(
    df: pd.DataFrame,
    horizon: int = 28,
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError(
            "horizon must be greater than zero."
        )

    if "date" not in df.columns:
        raise ValueError(
            "Batch dataset is missing the date column."
        )

    result = df.copy()
    result["date"] = pd.to_datetime(
        result["date"]
    )

    unique_dates = (
        result["date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_dates) < horizon:
        raise ValueError(
            "Batch dataset does not contain enough "
            "unique dates for the requested horizon."
        )

    start_date = unique_dates.iloc[-horizon]

    return result.loc[
        result["date"] >= start_date
    ].copy()


def prepare_batch_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = encode_event_missingness(df)

    missing_columns = [
        column
        for column in CATEGORICAL_FEATURES
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            "Batch dataset is missing categorical features: "
            + ", ".join(missing_columns)
        )

    for column in CATEGORICAL_FEATURES:
        result[column] = result[
            column
        ].astype("category")

    return result


def prediction_column(
    strategy: str,
) -> str:
    if strategy not in PREDICTION_COLUMNS:
        raise ValueError(
            "Unsupported model strategy. Choose one of: "
            + ", ".join(PREDICTION_COLUMNS)
        )

    return PREDICTION_COLUMNS[strategy]


def build_prediction_manifest(
    *,
    model_version: str,
    strategy: str,
    source_dataset: str | Path,
    output_path: Path,
    predictions: pd.DataFrame,
    git_sha: str,
    created_at_utc: str,
) -> dict[str, object]:
    dates = pd.to_datetime(
        predictions["date"]
    )

    return {
        "model_version": model_version,
        "model_strategy": strategy,
        "source_dataset": str(source_dataset),
        "output_file": output_path.name,
        "created_at_utc": created_at_utc,
        "git_sha": git_sha,
        "row_count": int(len(predictions)),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
        "prediction_total": float(
            predictions["prediction"].sum()
        ),
    }


def run_batch_inference(
    *,
    features_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    strategy: str = "poisson",
    horizon: int = 28,
) -> dict[str, object]:
    model_path = Path(model_dir)

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}"
        )

    selected_column = prediction_column(
        strategy
    )

    feature_data = load_batch_dataset(
        features_dir
    )

    batch_data = select_latest_horizon(
        feature_data,
        horizon=horizon,
    )

    batch_data = prepare_batch_features(
        batch_data
    )

    bundle = load_model_bundle(
        model_path
    )

    forecasts = generate_forecasts(
        df=batch_data,
        poisson_model=bundle[
            "poisson_model"
        ],
        classifier_model=bundle[
            "classifier_model"
        ],
        model_features=bundle[
            "model_features"
        ],
        threshold=bundle[
            "threshold"
        ],
    )

    forecasts["prediction"] = forecasts[
        selected_column
    ]

    output_columns = [
        "id",
        "item_id",
        "store_id",
        "date",
        "is_available",
        "prediction",
        "poisson_prediction",
        "two_stage_prediction",
        "positive_demand_probability",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in forecasts.columns
    ]

    predictions = forecasts[
        output_columns
    ].copy()

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    dates = pd.to_datetime(
        predictions["date"]
    )

    file_stem = (
        f"predictions_{model_path.name}_"
        f"{dates.min().date()}_"
        f"{dates.max().date()}"
    )

    predictions_path = (
        output_path / f"{file_stem}.parquet"
    )

    predictions.to_parquet(
        predictions_path,
        index=False,
    )

    created_at = datetime.now(
        timezone.utc
    ).replace(microsecond=0)

    manifest = build_prediction_manifest(
        model_version=model_path.name,
        strategy=strategy,
        source_dataset=features_dir,
        output_path=predictions_path,
        predictions=predictions,
        git_sha=get_git_sha("."),
        created_at_utc=created_at.isoformat(),
    )

    manifest_path = (
        output_path / f"{file_stem}.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    return {
        "predictions_path": predictions_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }