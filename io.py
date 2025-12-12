"""I/O operations for FCS and Parquet files, including streaming readers."""
import logging
from pathlib import Path
from typing import Dict, Generator, Tuple

import pandas as pd
import flowkit as fk
import pyarrow.parquet as pq

from cytoflow_qc.exceptions import FileOperationError

logger = logging.getLogger(__name__)

def load_samplesheet(path: str) -> pd.DataFrame:
    """Load a samplesheet, validating required columns and checking file existence."""
    try:
        sheet = pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileOperationError(f"Samplesheet not found at {path}") from e

    required_cols = {"sample_id", "file_path"}
    if not required_cols.issubset(sheet.columns):
        raise FileOperationError(f"Samplesheet missing required columns: {required_cols - set(sheet.columns)}")

    sheet["missing_file"] = sheet["file_path"].apply(lambda p: not Path(p).exists())
    return sheet

def stream_events(
    file_path: Path, chunk_size: int = 100_000, columns: list[str] | None = None
) -> Generator[pd.DataFrame, None, None]:
    """
    Read events from an FCS or Parquet file in chunks.
    
    Yields:
        Pandas DataFrame chunks.
    """
    if file_path.suffix.lower() == ".parquet":
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
            yield batch.to_pandas()
    elif file_path.suffix.lower() in [".fcs", ".lmd"]:
        sample = fk.Sample(str(file_path))
        num_events = sample.event_count
        
        # flowkit Sample.read_events doesn't support chunking directly, so we manually iterate
        for offset in range(0, num_events, chunk_size):
            count = min(chunk_size, num_events - offset)
            df = sample.read_events(event_count=count, offset=offset)
            if columns:
                df = df[columns]
            yield df
    else:
        raise FileOperationError(f"Unsupported file type for streaming: {file_path.suffix}")

def get_fcs_metadata(path: str) -> dict:
    """Read metadata from an FCS file."""
    try:
        sample = fk.Sample(path)
        return sample.get_metadata()
    except Exception as e:
        raise FileOperationError(f"Failed to read metadata from FCS file {path}") from e


def read_fcs(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Read all events and metadata from an FCS file.
    
    Returns:
        Tuple of (events DataFrame, metadata dict)
    """
    try:
        sample = fk.Sample(path)
        events = sample.as_dataframe(source='raw')
        metadata = sample.get_metadata()
        return events, metadata
    except Exception as e:
        raise FileOperationError(f"Failed to read FCS file {path}") from e

def standardize_channels(
    df: pd.DataFrame, metadata: dict, channel_map: dict[str, str]
) -> pd.DataFrame:
    """Rename channels based on a mapping, handling missing channels."""
    rename_map = {}
    for standard_name, fcs_name in channel_map.items():
        if fcs_name in df.columns:
            rename_map[fcs_name] = standard_name
    return df.rename(columns=rename_map)
