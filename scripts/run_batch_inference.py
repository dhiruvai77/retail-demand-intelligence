from __future__ import annotations

import argparse
from pathlib import Path

from src.models.batch_inference import (
    run_batch_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate versioned batch demand forecasts."
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
        "--output-dir",
        type=Path,
        default=Path("data/predictions"),
    )

    parser.add_argument(
        "--strategy",
        choices=["poisson", "two_stage"],
        default="poisson",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=28,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_batch_inference(
        features_dir=args.features_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        strategy=args.strategy,
        horizon=args.horizon,
    )

    print(
        "Predictions: "
        f"{Path(result['predictions_path']).resolve()}"
    )

    print(
        "Manifest: "
        f"{Path(result['manifest_path']).resolve()}"
    )

    print(
        f"Rows written: "
        f"{result['manifest']['row_count']:,}"
    )

    print(
        f"Prediction total: "
        f"{result['manifest']['prediction_total']:.2f}"
    )


if __name__ == "__main__":
    main()