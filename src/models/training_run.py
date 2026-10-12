from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.forecast_models import (
    DEFAULT_DEMAND_THRESHOLD,
)
from src.models.persistence import save_model_bundle
from src.models.train import (
    DATE_COLUMN,
    FORECAST_HORIZON,
    MODEL_FEATURES,
    TARGET_COLUMN,
    train_and_validate,
    validate_training_columns,
)


DEFAULT_FEATURES_DIR = Path("data/features")
DEFAULT_MODEL_ROOT = Path("artifacts/models")


def load_feature_dataset(
    features_dir: str | Path = DEFAULT_FEATURES_DIR,
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

    dataset = pd.read_parquet(dataset_path)
    validate_training_columns(dataset)

    return dataset


def get_git_sha(
    repo_dir: str | Path = ".",
) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return "unknown"

    return result.stdout.strip()


def create_model_version(
    created_at: datetime | None = None,
) -> str:
    timestamp = created_at or datetime.now(timezone.utc)

    return (
        f"run-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    )


def summarize_split(
    data: pd.DataFrame,
) -> dict[str, object]:
    dates = pd.to_datetime(data[DATE_COLUMN])

    return {
        "rows": int(len(data)),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
    }


def build_training_manifest(
    *,
    model_version: str,
    created_at_utc: str,
    feature_data: pd.DataFrame,
    training_result: dict[str, object],
    source_dataset: str | Path,
    validation_days: int,
    test_days: int,
    threshold: float,
    early_stopping_rounds: int,
    git_sha: str,
    artifact_paths: dict[str, Path],
) -> dict[str, object]:
    data_dates = pd.to_datetime(
        feature_data[DATE_COLUMN]
    )

    return {
        "model_version": model_version,
        "created_at_utc": created_at_utc,
        "git_sha": git_sha,
        "source_dataset": str(source_dataset),
        "row_count": int(len(feature_data)),
        "feature_column_count": int(
            len(feature_data.columns)
        ),
        "model_feature_count": int(
            len(MODEL_FEATURES)
        ),
        "model_features": list(MODEL_FEATURES),
        "target_column": TARGET_COLUMN,
        "data_date_min": (
            data_dates.min().date().isoformat()
        ),
        "data_date_max": (
            data_dates.max().date().isoformat()
        ),
        "validation_days": int(validation_days),
        "test_days": int(test_days),
        "threshold": float(threshold),
        "early_stopping_rounds": int(
            early_stopping_rounds
        ),
        "test_evaluated": False,
        "splits": {
            "train": summarize_split(
                training_result["train_df"]
            ),
            "validation": summarize_split(
                training_result["validation_df"]
            ),
            "test": summarize_split(
                training_result["test_df"]
            ),
        },
        "validation_metrics": training_result[
            "validation_metrics"
        ],
        "artifacts": {
            name: path.name
            for name, path in artifact_paths.items()
        },
    }


def run_training(
    *,
    features_dir: str | Path = DEFAULT_FEATURES_DIR,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    validation_days: int = FORECAST_HORIZON,
    test_days: int = FORECAST_HORIZON,
    threshold: float = DEFAULT_DEMAND_THRESHOLD,
    early_stopping_rounds: int = 50,
    repo_dir: str | Path = ".",
) -> dict[str, object]:
    feature_data = load_feature_dataset(
        features_dir
    )

    created_at = datetime.now(
        timezone.utc
    ).replace(microsecond=0)

    created_at_utc = created_at.isoformat()
    model_version = create_model_version(
        created_at
    )

    model_dir = (
        Path(model_root)
        / model_version
    )

    training_result = train_and_validate(
        feature_data,
        validation_days=validation_days,
        test_days=test_days,
        threshold=threshold,
        early_stopping_rounds=early_stopping_rounds,
    )

    artifact_paths = save_model_bundle(
        poisson_model=training_result[
            "poisson_model"
        ],
        classifier_model=training_result[
            "classifier_model"
        ],
        threshold=threshold,
        output_dir=model_dir,
    )

    manifest = build_training_manifest(
        model_version=model_version,
        created_at_utc=created_at_utc,
        feature_data=feature_data,
        training_result=training_result,
        source_dataset=features_dir,
        validation_days=validation_days,
        test_days=test_days,
        threshold=threshold,
        early_stopping_rounds=early_stopping_rounds,
        git_sha=get_git_sha(repo_dir),
        artifact_paths=artifact_paths,
    )

    manifest_path = (
        model_dir / "training_manifest.json"
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
        "model_version": model_version,
        "model_dir": model_dir,
        "manifest_path": manifest_path,
        "validation_metrics": training_result[
            "validation_metrics"
        ],
    }