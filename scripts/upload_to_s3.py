from __future__ import annotations

import argparse
from pathlib import Path

from src.storage.s3_upload import upload_path_to_s3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local file or directory to Amazon S3."
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Local file or directory to upload.",
    )

    parser.add_argument(
        "--bucket",
        required=True,
        help="Destination S3 bucket name.",
    )

    parser.add_argument(
        "--prefix",
        required=True,
        help="Destination key prefix, such as features or models/run-id.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = upload_path_to_s3(
        source=args.source,
        bucket=args.bucket,
        prefix=args.prefix,
    )

    print(f"S3 destination: {result['s3_uri']}")
    print(f"Files uploaded: {result['file_count']:,}")
    print(f"Total bytes: {result['total_bytes']:,}")


if __name__ == "__main__":
    main()