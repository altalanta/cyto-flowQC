"""Shared utility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file, returning an empty dict if absent."""
    try:
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with cfg_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
            config = config if config is not None else {}
            validate_config(config)
            return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config {path}: {e}")
    except (OSError, IOError) as e:
        raise RuntimeError(f"Cannot read config {path}: {e}")


def validate_config(config: dict[str, Any]) -> None:
    """Validate configuration structure and required fields."""
    required_keys = ["channels"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Validate channels structure
    channels = config.get("channels", {})
    if not isinstance(channels, dict):
        raise ValueError("Configuration 'channels' must be a dictionary")

    # Validate markers if present
    markers = channels.get("markers")
    if markers is not None and not isinstance(markers, list):
        raise ValueError("Configuration 'channels.markers' must be a list")


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


def list_stage_events(stage_dir: str | Path) -> dict[str, Path]:
    """Return mapping of sample_id -> event parquet within a stage directory."""

    mapping: dict[str, Path] = {}
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
