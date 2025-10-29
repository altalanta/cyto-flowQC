"""Shared utility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import yaml


def ensure_dir(path: Path) -> Path:
    """Convenience for mkdir -p."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe to Parquet or CSV."""
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load dataframe from Parquet or CSV."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
        return pd.read_csv(path)


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    """Persist manifest CSVs with consistent ordering."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("sample_id").to_csv(output, index=False)


def read_manifest(path: Path) -> pd.DataFrame:
    """Load a previously saved manifest CSV."""

    return pd.read_csv(path)


def list_stage_events(stage_dir: Path) -> dict[str, str]:
    """Return mapping of sample_id -> event parquet within a stage directory."""

    return {p.stem: str(p.relative_to(stage_dir)) for p in events_dir.glob("*.parquet")}


def timestamp() -> str:
    """UTC timestamp used for logs and reports."""

    return pd.Timestamp.utcnow().isoformat()
