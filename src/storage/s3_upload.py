from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

import boto3


def _normalise_prefix(prefix: str) -> str:
    normalised = prefix.strip().replace("\\", "/").strip("/")

    if not normalised:
        raise ValueError("S3 prefix must not be empty.")

    if any(part in {".", ".."} for part in normalised.split("/")):
        raise ValueError("S3 prefix cannot contain '.' or '..' path parts.")

    return normalised


def _files_to_upload(source_path: Path) -> list[tuple[Path, str]]:
    if not source_path.exists():
        raise FileNotFoundError(
            f"Upload source does not exist: {source_path}"
        )

    if source_path.is_file():
        return [(source_path, source_path.name)]

    if not source_path.is_dir():
        raise ValueError(
            f"Upload source must be a file or directory: {source_path}"
        )

    files = sorted(
        (
            path
            for path in source_path.rglob("*")
            if path.is_file() and not path.is_symlink()
        ),
        key=lambda path: path.as_posix(),
    )

    if not files:
        raise ValueError(
            f"Upload source contains no files: {source_path}"
        )

    return [
        (
            path,
            PurePosixPath(
                path.relative_to(source_path).as_posix()
            ).as_posix(),
        )
        for path in files
    ]


def upload_path_to_s3(
    *,
    source: str | Path,
    bucket: str,
    prefix: str,
    s3_client: Any | None = None,
) -> dict[str, object]:
    source_path = Path(source).expanduser()

    if not bucket.strip():
        raise ValueError("S3 bucket name must not be empty.")

    normalised_prefix = _normalise_prefix(prefix)
    files = _files_to_upload(source_path)
    client = s3_client or boto3.client("s3")

    uploaded_bytes = 0
    uploaded_keys: list[str] = []

    for file_path, relative_path in files:
        key = f"{normalised_prefix}/{relative_path}"
        client.upload_file(
            str(file_path),
            bucket,
            key,
        )
        uploaded_bytes += file_path.stat().st_size
        uploaded_keys.append(key)

    return {
        "bucket": bucket,
        "prefix": normalised_prefix,
        "s3_uri": f"s3://{bucket}/{normalised_prefix}/",
        "file_count": len(uploaded_keys),
        "total_bytes": uploaded_bytes,
        "keys": uploaded_keys,
    }