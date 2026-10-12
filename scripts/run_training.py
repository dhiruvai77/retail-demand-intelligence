from __future__ import annotations

import argparse
from pathlib import Path

from src.models.training_run import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and version a retail demand model."
        )
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/features"),
    )

    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("artifacts/models"),
    )

    parser.add_argument(
        "--validation-days",
        type=int,
        default=28,
    )

    parser.add_argument(
        "--test-days",
        type=int,
        default=28,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_training(
        features_dir=args.features_dir,
        model_root=args.model_root,
        validation_days=args.validation_days,
        test_days=args.test_days,
        threshold=args.threshold,
        early_stopping_rounds=(
            args.early_stopping_rounds
        ),
    )

    print(
        f"Model version: {result['model_version']}"
    )

    print(
        "Model directory: "
        f"{Path(result['model_dir']).resolve()}"
    )

    print(
        "Manifest: "
        f"{Path(result['manifest_path']).resolve()}"
    )

    print("Validation metrics:")

    for model_name, metrics in (
        result["validation_metrics"].items()
    ):
        print(f"  {model_name}: {metrics}")


if __name__ == "__main__":
    main()