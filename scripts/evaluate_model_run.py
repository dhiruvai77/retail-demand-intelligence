from __future__ import annotations

import argparse
from pathlib import Path

from src.models.evaluate_run import evaluate_model_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved model on the untouched "
            "test holdout."
        )
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/features"),
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = evaluate_model_run(
        features_dir=args.features_dir,
        model_dir=args.model_dir,
        validation_days=args.validation_days,
        test_days=args.test_days,
    )

    report = result["report"]

    print(
        f"Model version: {report['model_version']}"
    )

    print(
        f"Test period: {report['test_date_min']} "
        f"to {report['test_date_max']}"
    )

    print(f"Test rows: {report['test_rows']}")
    print("Test metrics:")

    for model_name, metrics in (
        report["metrics"].items()
    ):
        print(f"  {model_name}: {metrics}")

    print(
        "Report: "
        f"{Path(result['report_path']).resolve()}"
    )


if __name__ == "__main__":
    main()