from __future__ import annotations

import argparse
from pathlib import Path

from src.features.feature_dataset import (
    build_feature_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe features from curated Parquet."
        )
    )

    parser.add_argument(
        "--curated-dir",
        type=Path,
        default=Path("data/curated"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=28,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    row_count, column_count, output_path = (
        build_feature_dataset(
            curated_dir=args.curated_dir,
            output_dir=args.output_dir,
            horizon=args.horizon,
        )
    )

    print(
        f"Feature rows written: {row_count:,}"
    )

    print(
        f"Feature columns: {column_count}"
    )

    print(
        f"Output directory: {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()