from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, Tuple

import pandas as pd
import io

logger = logging.getLogger(__name__)

# Optional dependencies for cloud connectors
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore
    ClientError = None # type: ignore
    S3_AVAILABLE = False

try:
    from google.cloud import storage
    from google.oauth2 import service_account
    GCS_AVAILABLE = True
except ImportError:
    storage = None # type: ignore
    service_account = None # type: ignore
    GCS_AVAILABLE = False


class DataSourceConnector(ABC):
    """Abstract base class for all data source connectors."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def list_files(self, path: str, pattern: str = "*") -> Generator[str, None, None]:
        """List files in the specified path that match a pattern.

        Args:
            path: The base path (e.g., directory, S3 bucket prefix).
            pattern: A glob-style pattern to filter files (e.g., "*.fcs").

        Yields:
            Absolute paths or URIs of matching files.
        """
        pass

    @abstractmethod
    def read_file(self, file_path_or_uri: str) -> bytes:
        """Read the content of a file from the data source.

        Args:
            file_path_or_uri: The path or URI of the file.

        Returns:
            The content of the file as bytes.
        """
        pass

    @abstractmethod
    def write_file(self, file_path_or_uri: str, data: bytes) -> None:
        """Write data to a file in the data source.

        Args:
            file_path_or_uri: The path or URI of the file.
            data: The content to write as bytes.
        """
        pass

    def read_dataframe(self, file_path_or_uri: str, **kwargs) -> pd.DataFrame:
        """Read a file and attempt to load it as a pandas DataFrame.

        This method handles common file types (CSV, Parquet) by reading raw bytes
        and then using pandas.

        Args:
            file_path_or_uri: Path or URI to the file.
            **kwargs: Additional keyword arguments passed to pandas read functions.

        Returns:
            A pandas DataFrame.

        Raises:
            ValueError: If the file type is unsupported.
        """
        file_bytes = self.read_file(file_path_or_uri)
        file_suffix = Path(file_path_or_uri).suffix.lower()

        if file_suffix == ".csv":
            return pd.read_csv(io.BytesIO(file_bytes), **kwargs)
        elif file_suffix in {".parquet", ".pq"}:
            return pd.read_parquet(io.BytesIO(file_bytes), **kwargs)
        else:
            raise ValueError(f"Unsupported file type for DataFrame: {file_suffix}")

    def write_dataframe(self, file_path_or_uri: str, df: pd.DataFrame, **kwargs) -> None:
        """Write a pandas DataFrame to the data source.

        Args:
            file_path_or_uri: Path or URI for the output file.
            df: The DataFrame to write.
            **kwargs: Additional keyword arguments passed to pandas write functions.

        Raises:
            ValueError: If the file type is unsupported.
        """
        file_suffix = Path(file_path_or_uri).suffix.lower()
        buffer = io.BytesIO()

        if file_suffix == ".csv":
            df.to_csv(buffer, index=False, **kwargs)
        elif file_suffix in {".parquet", ".pq"}:
            df.to_parquet(buffer, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported file type for DataFrame: {file_suffix}")

        self.write_file(file_path_or_uri, buffer.getvalue())


class LocalFileConnector(DataSourceConnector):
    """A connector for local filesystem operations."""

    def list_files(self, path: str, pattern: str = "*") -> Generator[str, None, None]:
        base_path = Path(path)
        if not base_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        for f in base_path.rglob(pattern):
            if f.is_file():
                yield str(f.absolute())

    def read_file(self, file_path: str) -> bytes:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path.read_bytes()

    def write_file(self, file_path: str, data: bytes) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


class S3Connector(DataSourceConnector):
    """A connector for Amazon S3 bucket operations."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        if not S3_AVAILABLE:
            raise ImportError("boto3 library not found. Install with: pip install boto3")

        self.s3_client = boto3.client("s3", **self.config.get("client_kwargs", {}))

    def _parse_s3_uri(self, uri: str) -> Tuple[str, str]:
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}. Must start with s3://")
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def list_files(self, path: str, pattern: str = "*") -> Generator[str, None, None]:
        bucket, prefix = self._parse_s3_uri(path)
        paginator = self.s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Simple glob matching for S3 keys
                if Path(key).match(pattern):
                    yield f"s3://{bucket}/{key}"

    def read_file(self, file_uri: str) -> bytes:
        bucket, key = self._parse_s3_uri(file_uri)
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"File not found on S3: {file_uri}") from e
            raise # Re-raise other S3 errors

    def write_file(self, file_uri: str, data: bytes) -> None:
        bucket, key = self._parse_s3_uri(file_uri)
        try:
            self.s3_client.put_object(Bucket=bucket, Key=key, Body=data)
        except ClientError as e:
            raise IOError(f"Failed to write file to S3: {file_uri}") from e


class GCSConnector(DataSourceConnector):
    """A connector for Google Cloud Storage bucket operations."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage library not found. Install with: pip install google-cloud-storage")

        credentials_path = self.config.get("credentials_path")
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.gcs_client = storage.Client(credentials=credentials)
        else:
            self.gcs_client = storage.Client()

    def _parse_gcs_uri(self, uri: str) -> Tuple[str, str]:
        if not uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {uri}. Must start with gs://")
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def list_files(self, path: str, pattern: str = "*") -> Generator[str, None, None]:
        bucket_name, prefix = self._parse_gcs_uri(path)
        bucket = self.gcs_client.get_bucket(bucket_name)

        for blob in bucket.list_blobs(prefix=prefix):
            # Simple glob matching for GCS blob names
            if Path(blob.name).match(pattern):
                yield f"gs://{bucket_name}/{blob.name}"

    def read_file(self, file_uri: str) -> bytes:
        bucket_name, key = self._parse_gcs_uri(file_uri)
        bucket = self.gcs_client.get_bucket(bucket_name)
        blob = bucket.blob(key)
        if not blob.exists():
            raise FileNotFoundError(f"File not found on GCS: {file_uri}")
        return blob.download_as_bytes()

    def write_file(self, file_uri: str, data: bytes) -> None:
        bucket_name, key = self._parse_gcs_uri(file_uri)
        bucket = self.gcs_client.get_bucket(bucket_name)
        blob = bucket.blob(key)
        blob.upload_from_string(data)


class DataSourceError(Exception):
    """Custom exception for data source related errors."""
    pass


def get_connector(uri: str, config: Dict[str, Any] | None = None) -> DataSourceConnector:
    """Factory function to get the appropriate data source connector based on URI.

    Args:
        uri: The URI of the data source (e.g., "file:///local/path", "s3://bucket/path", "gs://bucket/path").
        config: Optional configuration for the connector.

    Returns:
        An instance of a DataSourceConnector subclass.

    Raises:
        ValueError: If the URI scheme is unsupported.
    """
    if uri.startswith("file://") or "://" not in uri: # Assume local file if no scheme or file://
        return LocalFileConnector(config)
    elif uri.startswith("s3://"):
        return S3Connector(config)
    elif uri.startswith("gs://"):
        return GCSConnector(config)
    else:
        raise ValueError(f"Unsupported data source URI scheme: {uri}")


if __name__ == "__main__":
    # Example Usage
    logger.info("--- Local File Connector Example ---")
    local_connector = get_connector("file:///tmp/my_data")
    test_dir = Path("/tmp/my_data/sub_dir")
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "file1.csv").write_text("col1,col2\n1,A")
    (test_dir / "file2.parquet").touch()
    (test_dir / "another.txt").write_text("hello")

    logger.info("Local files:")
    for f in local_connector.list_files("/tmp/my_data", pattern="*.csv"):
        logger.info(f"  {f}")
        content = local_connector.read_file(f)
        logger.info(f"  Content: {content.decode()}")
        df = local_connector.read_dataframe(f)
        logger.info(f"  DataFrame:\n{df}")

    output_path = test_dir / "output.csv"
    local_connector.write_dataframe(str(output_path), pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    logger.info(f"DataFrame written to: {output_path}")
    logger.info(f"Content read back: {local_connector.read_file(str(output_path)).decode()}")

    # Clean up
    import shutil
    shutil.rmtree("/tmp/my_data")

    print("\n--- S3 Connector Example (requires AWS credentials and a bucket) ---")
    # To run this, ensure you have AWS credentials configured (e.g., via ~/.aws/credentials)
    # and replace 'your-s3-bucket' with an actual S3 bucket name.
    # try:
    #     s3_connector = get_connector("s3://your-s3-bucket")
    #     bucket_name = "your-s3-bucket"
    #     test_s3_key = "test_cytoflow_qc/test_file.txt"
    #     test_s3_uri = f"s3://{bucket_name}/{test_s3_key}"
    #
    #     # Write a file
    #     s3_connector.write_file(test_s3_uri, b"Hello from S3!")
    #     print(f"Wrote to {test_s3_uri}")
    #
    #     # Read a file
    #     read_content = s3_connector.read_file(test_s3_uri)
    #     print(f"Read from S3: {read_content.decode()}")
    #
    #     # List files
    #     print(f"Files in s3://{bucket_name}/test_cytoflow_qc/")
    #     for f in s3_connector.list_files(f"s3://{bucket_name}/test_cytoflow_qc/", pattern="*.txt"):
    #         print(f"  {f}")
    #
    #     # Clean up
    #     s3_connector.s3_client.delete_object(Bucket=bucket_name, Key=test_s3_key)
    #     print(f"Cleaned up {test_s3_uri}")
    # except ImportError:
    #     print("Skipping S3 example: boto3 not installed.")
    # except ClientError as e:
    #     print(f"Skipping S3 example due to AWS error: {e}")
    #
    print("\n--- GCS Connector Example (requires Google Cloud credentials and a bucket) ---")
    # To run this, ensure you have Google Cloud credentials configured
    # (e.g., GOOGLE_APPLICATION_CREDENTIALS env var or gcloud auth application-default login)
    # and replace 'your-gcs-bucket' with an actual GCS bucket name.
    # try:
    #     gcs_connector = get_connector("gs://your-gcs-bucket")
    #     bucket_name = "your-gcs-bucket"
    #     test_gcs_key = "test_cytoflow_qc/test_file.txt"
    #     test_gcs_uri = f"gs://{bucket_name}/{test_gcs_key}"
    #
    #     # Write a file
    #     gcs_connector.write_file(test_gcs_uri, b"Hello from GCS!")
    #     print(f"Wrote to {test_gcs_uri}")
    #
    #     # Read a file
    #     read_content = gcs_connector.read_file(test_gcs_uri)
    #     print(f"Read from GCS: {read_content.decode()}")
    #
    #     # List files
    #     print(f"Files in gs://{bucket_name}/test_cytoflow_qc/")
    #     for f in gcs_connector.list_files(f"gs://{bucket_name}/test_cytoflow_qc/", pattern="*.txt"):
    #         print(f"  {f}")
    #
    #     # Clean up
    #     blob = gcs_connector.gcs_client.get_bucket(bucket_name).blob(test_gcs_key)
    #     blob.delete()
    #     print(f"Cleaned up {test_gcs_uri}")
    # except ImportError:
    #     print("Skipping GCS example: google-cloud-storage not installed.")
    # except Exception as e:
    #     print(f"Skipping GCS example due to Google Cloud error: {e}")















