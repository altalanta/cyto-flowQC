# Extensible Data Source Connectors and ETL Framework

CytoFlow-QC now features an extensible data source connector and ETL (Extract, Transform, Load) framework. This allows seamless integration with various data storage systems, including local files, AWS S3, and Google Cloud Storage, simplifying data ingestion and management.

## Data Source Connectors

The `DataSourceConnector` abstract base class provides a unified interface for interacting with different data sources. Concrete implementations (e.g., `LocalFileConnector`, `S3Connector`, `GCSConnector`) handle the specifics of each storage system. All custom connectors must inherit from this base class and implement its abstract methods.

### `DataSourceConnector` Abstract Base Class

The `DataSourceConnector` defines the essential contract for any data source. It ensures that all connectors provide a consistent way to list, read, and write files, as well as handle common data formats like CSV and Parquet.

**Key Methods:**

*   `__init__(self, config: Dict[str, Any] | None = None)`:
    Initializes the connector with an optional configuration dictionary. Subclasses should call `super().__init__(config)`.

*   `list_files(self, path: str, pattern: str = "*") -> Generator[str, None, None]`:
    An abstract method that must be implemented by subclasses. It should list files within a given `path` that match a `glob-style pattern`.
    *   `path` (str): The base path or URI (e.g., local directory, S3 bucket prefix).
    *   `pattern` (str): A glob-style pattern (e.g., `*.fcs`, `sample_*.csv`). Defaults to `*`.
    *   **Yields**: Absolute paths or URIs of matching files as strings.

*   `read_file(self, file_path_or_uri: str) -> bytes`:
    An abstract method to read the raw binary content of a file from the data source.
    *   `file_path_or_uri` (str): The full path or URI of the file.
    *   **Returns**: The file content as a `bytes` object.

*   `write_file(self, file_path_or_uri: str, data: bytes) -> None`:
    An abstract method to write raw binary data to a file in the data source.
    *   `file_path_or_uri` (str): The full path or URI where the file should be written.
    *   `data` (bytes): The binary content to write.

*   `read_dataframe(self, file_path_or_uri: str, **kwargs) -> pd.DataFrame`:
    A convenience method (non-abstract) that reads a file and attempts to load it into a pandas DataFrame. It supports `.csv` and `.parquet` files by default. Custom connectors can override this for specific formats.
    *   `file_path_or_uri` (str): Path or URI to the file.
    *   `**kwargs`: Additional keyword arguments passed to `pd.read_csv` or `pd.read_parquet`.
    *   **Returns**: A `pandas.DataFrame`.
    *   **Raises**: `ValueError` if the file type is unsupported.

*   `write_dataframe(self, file_path_or_uri: str, df: pd.DataFrame, **kwargs) -> None`:
    A convenience method (non-abstract) that writes a pandas DataFrame to the data source. It supports `.csv` and `.parquet` files by default. Custom connectors can override this for specific formats.
    *   `file_path_or_uri` (str): Path or URI for the output file.
    *   `df` (pd.DataFrame): The DataFrame to write.
    *   `**kwargs`: Additional keyword arguments passed to `df.to_csv` or `df.to_parquet`.
    *   **Raises**: `ValueError` if the file type is unsupported.

### Implementing a Custom Connector

To create a new data source connector, you need to:
1.  Inherit from `cytoflow_qc.data_connectors.DataSourceConnector`.
2.  Implement the abstract methods: `list_files`, `read_file`, and `write_file`.
3.  (Optional) Override `read_dataframe` or `write_dataframe` if your connector handles specific DataFrame formats more efficiently.
4.  Register your connector with the `get_connector` factory function (this is implicitly handled by the `if/elif` structure in `get_connector`).

**Example: Simple HTTP/S Connector (conceptual)**

```python
import requests
from urllib.parse import urlparse
from cytoflow_qc.data_connectors import DataSourceConnector, get_connector, DataSourceError
from typing import Generator, Dict, Any

class HTTPConnector(DataSourceConnector):
    def list_files(self, path: str, pattern: str = "*") -> Generator[str, None, None]:
        # For a simple HTTP connector, listing files might involve parsing an HTML directory listing
        # or relying on an API. This example is simplified.
        print(f"Warning: HTTPConnector.list_files is highly simplified and may not work for all HTTP servers.")
        try:
            response = requests.get(path, timeout=5)
            response.raise_for_status()
            # A real implementation would parse the response to find file links
            # For this example, we assume `path` is a direct file URL if pattern is default
            if pattern == "*" and urlparse(path).path.split("/")[-1]:
                yield path
            # More complex logic would be needed for directory listing
        except requests.exceptions.RequestException as e:
            raise DataSourceError(f"Failed to list files from {path}: {e}") from e

    def read_file(self, file_url: str) -> bytes:
        try:
            response = requests.get(file_url, timeout=10)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            raise DataSourceError(f"Failed to read file from {file_url}: {e}") from e

    def write_file(self, file_url: str, data: bytes) -> None:
        # HTTP/S typically does not support direct file writing via GET/POST for arbitrary paths
        # This method would usually raise an error or require a specific API endpoint.
        raise NotImplementedError("Writing files via HTTP is not supported by this connector example.")

# To make this connector discoverable, you would add an `elif` condition in `get_connector`:
# ```python
# def get_connector(uri: str, config: Dict[str, Any] | None = None) -> DataSourceConnector:
#     if uri.startswith("http://") or uri.startswith("https://"):
#         return HTTPConnector(config)
#     # ... existing logic ...
# ```
```

### URI Scheme and Factory Function

Connectors are selected and initialized based on a URI scheme. The `get_connector` factory function automatically returns the appropriate connector instance.

*   **Local Files**: Use `file:///path/to/data` or simply `/path/to/data`
*   **AWS S3**: Use `s3://your-bucket-name/path/to/data`
*   **Google Cloud Storage**: Use `gs://your-bucket-name/path/to/data`

### CLI Usage

Use the `cytoflow-qc data-source` command to manage data sources and ingest data:

```bash
# List files in a local directory
cytoflow-qc data-source list /path/to/local/data --pattern "*.fcs"

# List files in an S3 bucket (requires AWS credentials configured)
cytoflow-qc data-source list s3://your-s3-bucket/my_project/ --pattern "*.csv"

# Configure a data source (e.g., with credentials or region)
# (Configuration is typically passed via a YAML file for cloud services)
cytoflow-qc data-source configure s3://your-s3-bucket/ --config my_s3_config.yaml

# Ingest files from a data source to a local directory
cytoflow-qc data-source ingest s3://your-s3-bucket/raw_data/ \
    --pattern "*.fcs" \
    --output-dir ./data/raw_ingested
```

**Arguments:**

*   `<action>`: The action to perform (`list`, `configure`, `ingest`).
*   `--uri`, `-u`: The base URI for the data source.
*   `--config`, `-c`: (Optional) Path to a YAML configuration file for the connector (e.g., AWS credentials, GCS project ID).
*   `--pattern`, `-p`: (Optional) A glob pattern to filter files (e.g., `*.fcs`, `sample_*.csv`). Defaults to `*.fcs`.
*   `--output-dir`, `-o`: (Required for `ingest` action) The local directory to save ingested files.

### Programmatic Usage

You can interact with data connectors directly in your Python scripts:

```python
from cytoflow_qc.data_connectors import get_connector, DataSourceError
import pandas as pd
from pathlib import Path
import os
import shutil

# --- Local File Connector Example ---
print("\n--- Local File Connector Example ---")
local_base_dir = Path("/tmp/cytoflow_qc_data")
local_uri = f"file://{local_base_dir}"
local_connector = get_connector(local_uri)

local_dir = local_base_dir / "my_local_data"
local_dir.mkdir(parents=True, exist_ok=True)
(local_dir / "sample1.csv").write_text("marker1,marker2\n100,200\n150,220")
(local_dir / "sample2.fcs").touch() # Simulate a dummy FCS file

print(f"Listing files in {local_uri} matching *.csv:")
for file_uri in local_connector.list_files(str(local_dir), pattern="*.csv"):
    print(f"- {file_uri}")
    df = local_connector.read_dataframe(file_uri)
    print("  DataFrame head:\n", df.head())

output_csv_path = local_dir / "output.csv"
local_connector.write_dataframe(str(output_csv_path), pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
print(f"DataFrame written to: {output_csv_path}")
print(f"Content read back: {local_connector.read_file(str(output_csv_path)).decode()}")

# Clean up local dummy data
shutil.rmtree(local_base_dir)
print(f"Cleaned up local directory: {local_base_dir}")

# --- AWS S3 Connector (requires boto3 and AWS credentials) ---
print("\n--- AWS S3 Connector Example (requires boto3 and AWS credentials) ---")
print("To run this example, ensure you have:")
print("1. The 'boto3' library installed (`pip install boto3`).")
print("2. AWS credentials configured (e.g., via `~/.aws/credentials` or environment variables).")
print("3. An existing S3 bucket. Replace 'your-s3-bucket' below with your actual bucket name.")

S3_BUCKET_NAME = os.environ.get("CYTOFLOW_QC_S3_BUCKET", "your-s3-bucket-name") # IMPORTANT: Change this!
S3_TEST_PREFIX = "cytoflow-qc-test-data/"
s3_uri_base = f"s3://{S3_BUCKET_NAME}/{S3_TEST_PREFIX}"

try:
    s3_connector = get_connector(s3_uri_base)

    # Example: Write a DataFrame to S3
    df_to_s3 = pd.DataFrame({"col_a": [1, 2], "col_b": ["X", "Y"]})
    s3_output_key = f"{S3_TEST_PREFIX}output.parquet"
    s3_output_uri = f"s3://{S3_BUCKET_NAME}/{s3_output_key}"
    s3_connector.write_dataframe(s3_output_uri, df_to_s3)
    print(f"DataFrame written to S3: {s3_output_uri}")

    # Example: List files in S3
    print(f"Listing S3 files matching *.parquet in {s3_uri_base}:")
    found_s3_files = list(s3_connector.list_files(s3_uri_base, pattern="*.parquet"))
    if found_s3_files:
        for file_uri in found_s3_files:
            print(f"- {file_uri}")
            df_read_s3 = s3_connector.read_dataframe(file_uri)
            print("  DataFrame head:\n", df_read_s3.head())
    else:
        print("No .parquet files found in S3.")

    # Clean up S3 dummy data
    s3_connector.s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_output_key)
    print(f"Cleaned up S3 object: {s3_output_uri}")

except ImportError:
    print("Skipping S3 example: boto3 library not installed.")
except DataSourceError as e:
    print(f"Skipping S3 example due to configuration/connection error: {e}")
except Exception as e:
    print(f"An unexpected error occurred during S3 example: {e}")

# --- Google Cloud Storage Connector (requires google-cloud-storage and GCS credentials) ---
print("\n--- Google Cloud Storage Connector Example (requires google-cloud-storage and GCS credentials) ---")
print("To run this example, ensure you have:")
print("1. The 'google-cloud-storage' library installed (`pip install google-cloud-storage`).")
print("2. Google Cloud credentials configured (e.g., via `GOOGLE_APPLICATION_CREDENTIALS` environment variable or `gcloud auth application-default login`).")
print("3. An existing GCS bucket. Replace 'your-gcs-bucket' below with your actual bucket name.")

GCS_BUCKET_NAME = os.environ.get("CYTOFLOW_QC_GCS_BUCKET", "your-gcs-bucket-name") # IMPORTANT: Change this!
GCS_TEST_PREFIX = "cytoflow-qc-test-data/"
gcs_uri_base = f"gs://{GCS_BUCKET_NAME}/{GCS_TEST_PREFIX}"

try:
    gcs_connector = get_connector(gcs_uri_base)

    # Example: Write a file to GCS
    gcs_output_key = f"{GCS_TEST_PREFIX}hello.txt"
    gcs_output_uri = f"gs://{GCS_BUCKET_NAME}/{gcs_output_key}"
    gcs_connector.write_file(gcs_output_uri, b"Hello from GCS!")
    print(f"Wrote to GCS: {gcs_output_uri}")

    # Example: Read a file from GCS
    read_content_gcs = gcs_connector.read_file(gcs_output_uri)
    print(f"Read from GCS: {read_content_gcs.decode()}")

    # Example: List files in GCS
    print(f"Listing GCS files matching *.txt in {gcs_uri_base}:")
    found_gcs_files = list(gcs_connector.list_files(gcs_uri_base, pattern="*.txt"))
    if found_gcs_files:
        for file_uri in found_gcs_files:
            print(f"- {file_uri}")
    else:
        print("No .txt files found in GCS.")

    # Clean up GCS dummy data
    bucket = gcs_connector.gcs_client.get_bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(gcs_output_key)
    blob.delete()
    print(f"Cleaned up GCS object: {gcs_output_uri}")

except ImportError:
    print("Skipping GCS example: google-cloud-storage library not installed.")
except DataSourceError as e:
    print(f"Skipping GCS example due to configuration/connection error: {e}")
except Exception as e:
    print(f"An unexpected error occurred during GCS example: {e}")

```

### Error Handling

Data source connectors can raise `DataSourceError` (defined in `cytoflow_qc.data_connectors`) for issues specific to data access or operations. This custom exception allows for more granular error handling compared to generic `IOError` or `ValueError`.

**Common Scenarios and Errors:**

*   `FileNotFoundError`: Raised by `read_file` or `read_dataframe` if the specified file does not exist on the data source.
*   `NotADirectoryError`: Raised by `LocalFileConnector.list_files` if the provided path is not a directory.
*   `ValueError`: Raised by `get_connector` for unsupported URI schemes or by `read/write_dataframe` for unsupported file types.
*   `ImportError`: Raised by `S3Connector` or `GCSConnector` if their respective underlying libraries (`boto3`, `google-cloud-storage`) are not installed.
*   `DataSourceError`: A general exception for issues like network connectivity problems, permission denied errors, or invalid configurations specific to a data source (e.g., incorrect S3 bucket name, invalid GCS credentials).
*   `NotImplementedError`: Raised if a connector method is called that is intentionally not supported (e.g., `HTTPConnector.write_file`).

**Best Practices for Error Handling:**

*   **Catch specific exceptions**: Use `try-except` blocks to catch `DataSourceError` and other specific exceptions to provide informative feedback to the user.
*   **Provide user guidance**: Error messages should clearly explain what went wrong and how to fix it (e.g., "Install `boto3`", "Check AWS credentials").
*   **Log errors**: For robust applications, log detailed error information for debugging.

**Example:**

```python
from cytoflow_qc.data_connectors import get_connector, DataSourceError
from pathlib import Path

try:
    # Attempt to get a connector for an unsupported scheme
    connector = get_connector("ftp://example.com/data")
except ValueError as e:
    print(f"Caught expected error: {e}")

try:
    # Attempt to read a non-existent local file
    local_connector = get_connector("file:///tmp/non_existent_path")
    local_connector.read_file(str(Path("/tmp/non_existent_path/file.fcs")))
except FileNotFoundError as e:
    print(f"Caught expected error: {e}")
except NotADirectoryError as e:
    print(f"Caught expected error: {e}")
except DataSourceError as e:
    print(f"Caught expected data source error: {e}")
except Exception as e:
    print(f"Caught unexpected error: {e}")

# Example with S3 (conceptual, requires setup)
# try:
#     s3_connector = get_connector("s3://non-existent-bucket")
#     s3_connector.list_files("s3://non-existent-bucket/", pattern="*")
# except DataSourceError as e:
#     print(f"Caught S3 related error: {e}")
# except ImportError:
#     print("boto3 not installed, cannot test S3 error handling.")
```

## ETL Framework (Planned Enhancements)

The current data connector framework lays the groundwork for a more comprehensive ETL system. Future enhancements will include:

*   **Automated Data Validation**: Define schema expectations and validate incoming data against these schemas.
*   **Data Transformation Pipelines**: Implement flexible pipelines for common data transformations (e.g., unit conversion, feature engineering).
*   **Schema Inference**: Automatically infer data schemas from various input formats.
*   **Data Quality Monitoring**: Tools to monitor data quality metrics as data flows through the ETL process.

These planned features will further streamline the data preparation phase, ensuring high-quality and consistent data for downstream analysis in CytoFlow-QC.
