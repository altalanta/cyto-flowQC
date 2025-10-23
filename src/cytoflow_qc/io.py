"""I/O helpers for flow cytometry inputs and manifests."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Import the new data connector factory
from cytoflow_qc.data_connectors import get_connector, DataSourceConnector

try:  # pragma: no cover - optional dependency
    import flowkit as fk
except ImportError:  # pragma: no cover
    fk = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import fcsparser
except ImportError:  # pragma: no cover
    fcsparser = None  # type: ignore


REQUIRED_SAMPLE_COLUMNS = {
    "sample_id",
    "file_path",
    "batch",
    "condition",
}


def load_samplesheet(uri: str) -> pd.DataFrame:
    """Read and validate the experiment samplesheet from a given URI.

    The loader enforces required columns, normalises path columns to absolute
    paths (if local), and preserves any extra metadata columns for downstream joins.
    Missing files are tolerated but reported via a ``missing_file`` column so the
    CLI can surface a warning while still allowing dry runs with placeholder
    paths (useful for CI tests with synthetic data).
    """

    connector = get_connector(uri)

    try:
        # Read samplesheet using the connector
        df = connector.read_dataframe(uri)
    except FileNotFoundError:
        raise FileNotFoundError(f"Samplesheet not found: {uri}")
    except ValueError as e:
        raise ValueError(f"Error reading samplesheet from {uri}: {e}")

    missing_cols = REQUIRED_SAMPLE_COLUMNS.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Samplesheet missing columns: {sorted(missing_cols)}")

    if df["sample_id"].duplicated().any():
        dupes = df[df["sample_id"].duplicated()]["sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id values: {dupes}")

    # Normalize file_path column. If local, make absolute. Otherwise, keep as is.
    if uri.startswith("file://") or "://" not in uri:
        # Assuming local file system for now, will need a better way to get base path
        # for relative paths within a remote URI if that becomes a requirement.
        base_path = Path(uri).parent if "://" in uri else Path(uri).parent
        df["file_path"] = df["file_path"].apply(lambda p: str((base_path / p).resolve()))
        df["missing_file"] = df["file_path"].apply(lambda p: not Path(p).exists())
    else:
        # For remote URIs, assume file_path is already a valid URI or relative to base_uri
        # No easy way to check existence without another connector call per file, which could be slow.
        # For now, mark as not missing and rely on read_fcs to handle FileNotFoundError.
        df["missing_file"] = False

    return df


def read_fcs(file_uri: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load event-level data from an FCS (or CSV/Parquet surrogate) file via connector.

    The function accepts URIs for various data sources. When real FCS
    parsing libraries are installed it will prefer ``flowkit`` and fall back to
    ``fcsparser``. Metadata always includes a ``channels`` list and the number of
    events; spillover matrices are propagated when present.
    """

    connector = get_connector(file_uri)

    file_bytes = connector.read_file(file_uri)

    suffix = Path(file_uri).suffix.lower()
    metadata: Dict[str, Any] = {
        "source_path": file_uri,
    }

    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(io.BytesIO(file_bytes))
    else:
        df = _read_fcs_binary(file_uri, file_bytes, metadata)

    metadata["n_events"] = len(df)
    metadata["channels"] = df.columns.tolist()
    metadata.setdefault("sample_id", Path(file_uri).stem)
    return df, metadata


def standardize_channels(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    channel_map: Dict[str, str],
) -> pd.DataFrame:
    """Rename channels according to a configuration map.

    The map keys are canonical names (``fsc_a``, ``viability``...) and values are
    the column headers present in the raw data. Only channels present in the
    DataFrame are renamed; unknown entries are ignored. Metadata is updated in
    place by replacing the ``channels`` list when a rename occurs.
    """

    rename_pairs = {
        raw_name: canonical
        for canonical, raw_name in channel_map.items()
        if raw_name in df.columns
    }
    if not rename_pairs:
        return df

    renamed = df.rename(columns=rename_pairs)
    metadata["channels"] = renamed.columns.tolist()
    return renamed


def _read_fcs_binary(file_uri: str, file_bytes: bytes, metadata: Dict[str, Any]) -> pd.DataFrame:
    """Internal helper that loads a binary FCS file via flowkit/fcsparser from bytes."""
    # flowkit and fcsparser typically expect file paths or file-like objects, not raw bytes directly.
    # We'll write the bytes to a temporary file for these libraries.

    # Use tempfile to create a temporary file to write the bytes into
    with tempfile.NamedTemporaryFile(delete=True, suffix=".fcs") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = Path(tmp_file.name)

        if fk is not None:  # pragma: no branch
            try:
                sample = fk.Sample(str(tmp_file_path))
                metadata.update(
                    {
                        "sample_id": sample.id,
                        "acquisition_date": getattr(sample, "acquisition_date", None),
                    }
                )
                if sample.compensation is not None:
                    metadata["spillover_matrix"] = sample.compensation.matrix
                    metadata["spillover_channels"] = sample.compensation.fluorochrome_labels
                return sample.as_dataframe(source="raw")
            except Exception:  # pragma: no cover - fall back to fcsparser
                pass

        if fcsparser is None:
            raise ImportError(
                "No FCS reader available. Install flowkit or fcsparser to read binary FCS files."
            )

        meta, data = fcsparser.parse(str(tmp_file_path), reformat_meta=True)
        metadata["keywords"] = meta
        if "$SPILLOVER" in meta:
            spill = _parse_spillover_keyword(meta["$SPILLOVER"])
            if spill is not None:
                matrix, channels = spill
                metadata["spillover_matrix"] = matrix
                metadata["spillover_channels"] = channels
        return pd.DataFrame(data)


def _parse_spillover_keyword(value: Any) -> Optional[Tuple[np.ndarray, list[str]]]:
    """Parse the `$SPILLOVER` keyword when present."""

    if not isinstance(value, str):
        return None

    parts = [p.strip() for p in value.replace("\n", ",").split(",") if p.strip()]
    if not parts:
        return None

    try:
        n_channels = int(parts[0])
    except ValueError:
        return None

    channel_names = parts[1 : 1 + n_channels]
    values = parts[1 + n_channels :]
    if len(values) != n_channels * n_channels:
        return None

    matrix = np.array(values, dtype=float).reshape(n_channels, n_channels)
    return matrix, channel_names
