from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from cytoflow_qc.data_connectors import (
    DataSourceConnector, LocalFileConnector, S3Connector, GCSConnector, get_connector, DataSourceError
)

# --- Fixtures ---

@pytest.fixture
def local_test_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "local_data"
    data_dir.mkdir()
    (data_dir / "file1.csv").write_text("col1,col2\n1,A\n2,B")
    (data_dir / "sub_dir").mkdir()
    (data_dir / "sub_dir" / "file2.parquet").write_bytes(pd.DataFrame({"x": [10], "y": [20]}).to_parquet())
    (data_dir / "file3.fcs").touch() # Dummy FCS file
    yield data_dir
    shutil.rmtree(data_dir)

@pytest.fixture
def mock_s3_client():
    with patch('boto3.client') as mock_client:
        s3_mock = MagicMock()
        mock_client.return_value = s3_mock
        yield s3_mock

@pytest.fixture
def mock_gcs_client():
    with patch('google.cloud.storage.Client') as mock_client:
        gcs_mock = MagicMock()
        mock_client.return_value = gcs_mock
        yield gcs_mock

# --- Test LocalFileConnector ---

class TestLocalFileConnector:
    def test_list_files(self, local_test_dir: Path):
        connector = LocalFileConnector()
        files = list(connector.list_files(str(local_test_dir), pattern="*.csv"))
        assert len(files) == 1
        assert Path(files[0]).name == "file1.csv"

        all_files = list(connector.list_files(str(local_test_dir), pattern="*.*"))
        assert len(all_files) == 3 # file1.csv, file2.parquet, file3.fcs

    def test_read_file(self, local_test_dir: Path):
        connector = LocalFileConnector()
        content = connector.read_file(str(local_test_dir / "file1.csv"))
        assert content == b"col1,col2\n1,A\n2,B"

    def test_write_file(self, tmp_path: Path):
        connector = LocalFileConnector()
        output_file = tmp_path / "output.txt"
        connector.write_file(str(output_file), b"test content")
        assert output_file.exists()
        assert output_file.read_bytes() == b"test content"

    def test_read_dataframe_csv(self, local_test_dir: Path):
        connector = LocalFileConnector()
        df = connector.read_dataframe(str(local_test_dir / "file1.csv"))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["col1", "col2"]

    def test_read_dataframe_parquet(self, local_test_dir: Path):
        connector = LocalFileConnector()
        df = connector.read_dataframe(str(local_test_dir / "sub_dir" / "file2.parquet"))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 2)
        assert list(df.columns) == ["x", "y"]

    def test_write_dataframe_csv(self, tmp_path: Path):
        connector = LocalFileConnector()
        output_file = tmp_path / "output.csv"
        df_to_write = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        connector.write_dataframe(str(output_file), df_to_write)
        assert output_file.exists()
        read_df = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(read_df, df_to_write)


# --- Test S3Connector ---

class TestS3Connector:
    # Patch boto3.client for all tests in this class
    @pytest.fixture(autouse=True)
    def setup_s3_mocks(self, mock_s3_client):
        self.mock_s3_client = mock_s3_client

    def test_parse_s3_uri(self):
        connector = S3Connector(config={"region_name": "us-east-1"})
        bucket, key = connector._parse_s3_uri("s3://my-bucket/path/to/object.txt")
        assert bucket == "my-bucket"
        assert key == "path/to/object.txt"

        bucket, key = connector._parse_s3_uri("s3://my-bucket/")
        assert bucket == "my-bucket"
        assert key == ""

        with pytest.raises(ValueError, match="Invalid S3 URI"):
            connector._parse_s3_uri("http://not-s3.com")

    def test_list_files(self):
        # Mock the paginator for list_objects_v2
        paginator_mock = MagicMock()
        self.mock_s3_client.get_paginator.return_value = paginator_mock
        paginator_mock.paginate.return_value = [
            {"Contents": [{"Key": "prefix/file1.csv"}, {"Key": "prefix/sub/file2.parquet"}, {"Key": "another/file.txt"}]}
        ]

        connector = S3Connector(config={"region_name": "us-east-1"})
        files = list(connector.list_files("s3://test-bucket/prefix/", pattern="*.csv"))
        assert len(files) == 1
        assert files[0] == "s3://test-bucket/prefix/file1.csv"

    def test_read_file(self):
        self.mock_s3_client.get_object.return_value = {"Body": io.BytesIO(b"S3 file content")}
        connector = S3Connector(config={"region_name": "us-east-1"})
        content = connector.read_file("s3://test-bucket/test.txt")
        assert content == b"S3 file content"

    def test_read_file_not_found(self):
        self.mock_s3_client.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject") # type: ignore
        connector = S3Connector(config={"region_name": "us-east-1"})
        with pytest.raises(FileNotFoundError):
            connector.read_file("s3://test-bucket/non-existent.txt")

    def test_write_file(self):
        connector = S3Connector(config={"region_name": "us-east-1"})
        connector.write_file("s3://test-bucket/output.txt", b"output data")
        self.mock_s3_client.put_object.assert_called_once_with(Bucket="test-bucket", Key="output.txt", Body=b"output data")


# --- Test GCSConnector ---

class TestGCSConnector:
    # Patch google.cloud.storage.Client for all tests in this class
    @pytest.fixture(autouse=True)
    def setup_gcs_mocks(self, mock_gcs_client):
        self.mock_gcs_client = mock_gcs_client

    def test_parse_gcs_uri(self):
        connector = GCSConnector()
        bucket, key = connector._parse_gcs_uri("gs://my-gcs-bucket/path/to/blob.json")
        assert bucket == "my-gcs-bucket"
        assert key == "path/to/blob.json"

        bucket, key = connector._parse_gcs_uri("gs://my-gcs-bucket/")
        assert bucket == "my-gcs-bucket"
        assert key == ""

        with pytest.raises(ValueError, match="Invalid GCS URI"):
            connector._parse_gcs_uri("http://not-gcs.com")

    def test_list_files(self):
        mock_bucket = MagicMock()
        mock_blob1 = MagicMock(name="prefix/fileA.csv")
        mock_blob2 = MagicMock(name="prefix/sub/fileB.parquet")
        mock_blob3 = MagicMock(name="another/fileC.txt")
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]
        self.mock_gcs_client.return_value.get_bucket.return_value = mock_bucket

        connector = GCSConnector()
        files = list(connector.list_files("gs://test-gcs-bucket/prefix/", pattern="*.csv"))
        assert len(files) == 1
        assert files[0] == "gs://test-gcs-bucket/prefix/fileA.csv"

    def test_read_file(self):
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = b"GCS file content"
        mock_bucket.blob.return_value = mock_blob
        self.mock_gcs_client.return_value.get_bucket.return_value = mock_bucket

        connector = GCSConnector()
        content = connector.read_file("gs://test-gcs-bucket/test.txt")
        assert content == b"GCS file content"

    def test_read_file_not_found(self):
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob
        self.mock_gcs_client.return_value.get_bucket.return_value = mock_bucket

        connector = GCSConnector()
        with pytest.raises(FileNotFoundError):
            connector.read_file("gs://test-gcs-bucket/non-existent.txt")

    def test_write_file(self):
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        self.mock_gcs_client.return_value.get_bucket.return_value = mock_bucket

        connector = GCSConnector()
        connector.write_file("gs://test-gcs-bucket/output.txt", b"output data")
        mock_blob.upload_from_string.assert_called_once_with(b"output data")


# --- Test get_connector factory ---

class TestGetConnector:
    def test_get_local_connector(self):
        connector = get_connector("/tmp/data")
        assert isinstance(connector, LocalFileConnector)
        connector = get_connector("file:///tmp/data")
        assert isinstance(connector, LocalFileConnector)

    def test_get_s3_connector(self, mock_s3_client):
        # Mock S3_AVAILABLE to be True for this test
        with patch('cytoflow_qc.data_connectors.S3_AVAILABLE', True):
            connector = get_connector("s3://bucket/path")
            assert isinstance(connector, S3Connector)

    def test_get_gcs_connector(self, mock_gcs_client):
        # Mock GCS_AVAILABLE to be True for this test
        with patch('cytoflow_qc.data_connectors.GCS_AVAILABLE', True):
            connector = get_connector("gs://bucket/path")
            assert isinstance(connector, GCSConnector)

    def test_unsupported_uri(self):
        with pytest.raises(ValueError, match="Unsupported data source URI scheme"):
            get_connector("ftp://server/file")

    def test_s3_import_error(self):
        with patch('cytoflow_qc.data_connectors.S3_AVAILABLE', False):
            with pytest.raises(ImportError, match="boto3 library not found"):
                S3Connector()

    def test_gcs_import_error(self):
        with patch('cytoflow_qc.data_connectors.GCS_AVAILABLE', False):
            with pytest.raises(ImportError, match="google-cloud-storage library not found"):
                GCSConnector()

# --- Test DataSourceError ---

def test_data_source_error():
    with pytest.raises(DataSourceError):
        raise DataSourceError("Test error")



