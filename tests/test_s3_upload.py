from pathlib import Path

import pytest

from src.storage.s3_upload import upload_path_to_s3


class FakeS3Client:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str]] = []

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
    ) -> None:
        self.uploads.append((filename, bucket, key))


def test_upload_directory_preserves_relative_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "features"
    nested = source / "store_id=1"
    nested.mkdir(parents=True)

    first_file = nested / "part-000.parquet"
    second_file = source / "manifest.json"
    first_file.write_bytes(b"parquet-data")
    second_file.write_text("{}", encoding="utf-8")

    client = FakeS3Client()

    result = upload_path_to_s3(
        source=source,
        bucket="retail-demand-test",
        prefix="features/run-1",
        s3_client=client,
    )

    assert result["file_count"] == 2
    assert result["total_bytes"] == (
        first_file.stat().st_size + second_file.stat().st_size
    )
    assert client.uploads == [
        (
            str(second_file),
            "retail-demand-test",
            "features/run-1/manifest.json",
        ),
        (
            str(first_file),
            "retail-demand-test",
            "features/run-1/store_id=1/part-000.parquet",
        ),
    ]


def test_upload_single_file_uses_its_name(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.joblib"
    source.write_bytes(b"model")

    client = FakeS3Client()

    result = upload_path_to_s3(
        source=source,
        bucket="retail-demand-test",
        prefix="models/run-1",
        s3_client=client,
    )

    assert result["file_count"] == 1
    assert client.uploads == [
        (
            str(source),
            "retail-demand-test",
            "models/run-1/model.joblib",
        ),
    ]


def test_upload_rejects_missing_source(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        upload_path_to_s3(
            source=tmp_path / "missing",
            bucket="retail-demand-test",
            prefix="features",
            s3_client=FakeS3Client(),
        )


def test_upload_rejects_empty_prefix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data.txt"
    source.write_text("data", encoding="utf-8")

    with pytest.raises(ValueError, match="prefix"):
        upload_path_to_s3(
            source=source,
            bucket="retail-demand-test",
            prefix=" / ",
            s3_client=FakeS3Client(),
        )