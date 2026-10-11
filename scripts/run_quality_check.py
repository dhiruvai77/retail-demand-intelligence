from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.quality import (
    run_quality_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run quality checks on curated Parquet data."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/curated"),
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(
            "data/quality/curated_quality_report.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = run_quality_check(
        dataset_dir=args.dataset_dir,
        report_path=args.report_path,
    )

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()