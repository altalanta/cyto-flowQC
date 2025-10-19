"""Shared utility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import yaml


def load_config(path: str) -> Dict[str, object]:
    """Load a YAML configuration file, returning an empty dict if absent."""

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing and return the ``Path``."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a DataFrame as Parquet for compact storage."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame saved with :func:`save_dataframe`."""

    return pd.read_parquet(Path(path))


def write_manifest(manifest: pd.DataFrame, path: str | Path) -> None:
    """Persist manifest CSVs with consistent ordering."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.sort_values("sample_id").to_csv(output, index=False)


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Load a previously saved manifest CSV."""

    return pd.read_csv(Path(path))


def list_stage_events(stage_dir: str | Path) -> Dict[str, Path]:
    """Return mapping of sample_id -> event parquet within a stage directory."""

    mapping: Dict[str, Path] = {}
    stage = Path(stage_dir)
    for file in stage.glob("events/*.parquet"):
        try:
            mapping[file.stem] = file.relative_to(stage)
        except ValueError:
            mapping[file.stem] = file
    return mapping


def timestamp() -> str:
    """UTC timestamp used for logs and reports."""

    return pd.Timestamp.utcnow().isoformat()
