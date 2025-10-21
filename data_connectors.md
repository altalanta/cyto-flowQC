# Extensible Data Source Connectors and ETL Framework

CytoFlow-QC now features an extensible data source connector and ETL (Extract, Transform, Load) framework. This allows seamless integration with various data storage systems, including local files, AWS S3, and Google Cloud Storage, simplifying data ingestion and management.

## Data Source Connectors

The `DataSourceConnector` abstract base class provides a unified interface for interacting with different data sources. Concrete implementations (e.g., `LocalFileConnector`, `S3Connector`, `GCSConnector`) handle the specifics of each storage system.

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
from cytoflow_qc.data_connectors import get_connector
import pandas as pd
from pathlib import Path

# --- Local File Connector ---
local_uri = "file:///tmp/my_local_data"
local_connector = get_connector(local_uri)

# Create some dummy local files
local_dir = Path("/tmp/my_local_data")
local_dir.mkdir(parents=True, exist_ok=True)
(local_dir / "sample1.csv").write_text("marker1,marker2\n100,200\n150,220")
(local_dir / "sample2.fcs").touch() # Dummy FCS file

print("\nLocal files matching *.csv:")
for file_uri in local_connector.list_files(local_uri, pattern="*.csv"):
    print(f"- {file_uri}")
    df = local_connector.read_dataframe(file_uri)
    print(df.head())

# --- AWS S3 Connector (requires boto3 and AWS credentials) ---
# s3_uri = "s3://your-s3-bucket/my_experiment"
# s3_connector = get_connector(s3_uri)
# 
# # Example: Write a DataFrame to S3
# df_to_s3 = pd.DataFrame({"col_a": [1, 2], "col_b": ["X", "Y"]})
# s3_connector.write_dataframe(f"{s3_uri}/output.parquet", df_to_s3)
# print(f"DataFrame written to S3: {s3_uri}/output.parquet")
# 
# # Example: List files in S3
# print("\nS3 files matching *.parquet:")
# for file_uri in s3_connector.list_files(s3_uri, pattern="*.parquet"):
#     print(f"- {file_uri}")

# --- Google Cloud Storage Connector (requires google-cloud-storage and GCS credentials) ---
# gcs_uri = "gs://your-gcs-bucket/project_data"
# gcs_connector = get_connector(gcs_uri)
# 
# # Example: Read a file from GCS
# # Ensure a file named 'metadata.json' exists in gs://your-gcs-bucket/project_data/
# # metadata_content = gcs_connector.read_file(f"{gcs_uri}/metadata.json")
# # print(f"Metadata from GCS: {metadata_content.decode()}")

# Clean up local dummy data
# import shutil
# shutil.rmtree(local_dir)
```

## ETL Framework (Planned Enhancements)

The current data connector framework lays the groundwork for a more comprehensive ETL system. Future enhancements will include:

*   **Automated Data Validation**: Define schema expectations and validate incoming data against these schemas.
*   **Data Transformation Pipelines**: Implement flexible pipelines for common data transformations (e.g., unit conversion, feature engineering).
*   **Schema Inference**: Automatically infer data schemas from various input formats.
*   **Data Quality Monitoring**: Tools to monitor data quality metrics as data flows through the ETL process.

These planned features will further streamline the data preparation phase, ensuring high-quality and consistent data for downstream analysis in CytoFlow-QC.
