from __future__ import annotations

import argparse
from pathlib import Path

from src.data.curate_parquet import (
    build_curated_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the local curated Parquet dataset."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/curated"),
    )

    parser.add_argument(
        "--store-id",
        default="CA_1",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    row_count, output_path = (
        build_curated_parquet(
            raw_data_dir=args.raw_dir,
            output_dir=args.output_dir,
            store_id=args.store_id,
        )
    )

    print(
        f"Curated rows written: {row_count:,}"
    )

    print(
        f"Output directory: {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()