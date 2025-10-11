"""Spillover compensation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def get_spillover(
    metadata: dict[str, Any],
    override: str | None = None,
) -> tuple[np.ndarray | None, list[str] | None]:
    """Return a spillover matrix and channel order if one is available.

    Priority order:

    1. explicit CSV override supplied by the user
    2. matrix embedded in ``metadata['spillover_matrix']``
    3. keywords containing a `$SPILLOVER` string
    """

    if override:
        return _load_spillover_from_csv(Path(override))

    if "spillover_matrix" in metadata and "spillover_channels" in metadata:
        matrix = np.asarray(metadata["spillover_matrix"], dtype=float)
        channels = list(metadata["spillover_channels"])  # type: ignore[arg-type]
        if matrix.shape[0] == matrix.shape[1] == len(channels):
            return matrix, channels

    keywords = metadata.get("keywords")
    if isinstance(keywords, dict) and "$SPILLOVER" in keywords:
        parsed = _parse_spillover_string(keywords["$SPILLOVER"])
        if parsed is not None:
            return parsed

    return None, None


def apply_compensation(
    df: pd.DataFrame,
    spill_matrix: np.ndarray,
    spill_channels: Any,
) -> pd.DataFrame:
    """Apply spillover compensation to fluorescence channels.

    Compensation only runs on channels present in the DataFrame; missing channels
    are silently ignored so that partial matrices can still be used in tests.
    ``spill_matrix`` must be invertible. A copy of ``df`` is returned to avoid
    mutating upstream data.
    """

    matrix = np.asarray(spill_matrix, dtype=float)
    channels = list(spill_channels)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Spillover matrix must be square")
    if matrix.shape[0] != len(channels):
        raise ValueError("Channel list length does not match matrix dimensions")

    present = [ch for ch in channels if ch in df.columns]
    MIN_CHANNELS_FOR_COMPENSATION = 2
    if len(present) < MIN_CHANNELS_FOR_COMPENSATION:
        return df.copy()

    sub_idx = [channels.index(ch) for ch in present]
    sub_matrix = matrix[np.ix_(sub_idx, sub_idx)]
    try:
        inverse = np.linalg.inv(sub_matrix)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise ValueError("Spillover matrix is singular") from exc

    compensated = df.copy()
    original_values = df[present].to_numpy(dtype=float)
    corrected = original_values @ inverse.T
    compensated.loc[:, present] = corrected
    return compensated


def _load_spillover_from_csv(path: Path) -> tuple[np.ndarray | None, list[str] | None]:
    """Load a square spillover matrix from a CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"Spillover matrix not found: {path}")

    df = pd.read_csv(path, index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError("Spillover matrix CSV must be square")
    if not df.index.equals(df.columns):
        raise ValueError("CSV index/columns must match channel names")
    return df.to_numpy(dtype=float), df.index.tolist()


def _parse_spillover_string(raw: object) -> tuple[np.ndarray, list[str]] | None:
    """Parse the `$SPILLOVER` keyword string into an array and channel list."""

    if not isinstance(raw, str):
        return None

    parts = [token.strip() for token in raw.replace("\n", ",").split(",") if token.strip()]
    if not parts:
        return None
    try:
        n = int(parts[0])
    except ValueError:
        return None

    names = parts[1 : 1 + n]
    values = parts[1 + n :]
    if len(values) != n * n:
        return None

    matrix = np.array(values, dtype=float).reshape(n, n)
    return matrix, names
