"""Quality-control heuristics for event level data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_QC_CONFIG: dict[str, dict[str, float]] = {
    "debris": {"fsc_percentile": 2.0, "ssc_percentile": 2.0},
    "doublets": {"tolerance": 0.08},
    "saturation": {"threshold": 0.995},
}


def add_qc_flags(
    df: pd.DataFrame, config: dict[str, dict[str, float]] | None = None
) -> pd.DataFrame:
    """Annotate the provided events table with QC flag columns.

    Columns added:
      * ``qc_debris`` – low FSC/SSC events
      * ``qc_doublets`` – FSC-H deviating from FSC-A
      * ``qc_saturated`` – fluorescence at detector max
      * ``qc_pass`` – inverse OR of the above flags
    """

    cfg = DEFAULT_QC_CONFIG.copy()
    if config:
        cfg = {**cfg, **config}

    result = df.copy()
    result["qc_debris"] = _flag_debris(result, cfg["debris"])
    result["qc_doublets"] = _flag_doublets(result, cfg["doublets"])
    result["qc_saturated"] = _flag_saturation(result, cfg["saturation"])
    result["qc_pass"] = ~(result[["qc_debris", "qc_doublets", "qc_saturated"]].any(axis=1))
    return result


def qc_summary(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarise QC metrics for a dict of ``sample_id -> DataFrame``."""

    records = []
    for sample_id, frame in samples.items():
        total = len(frame)
        if total == 0:
            continue
        record = {
            "sample_id": sample_id,
            "total_events": total,
            "qc_pass_fraction": frame["qc_pass"].mean() if "qc_pass" in frame else np.nan,
            "debris_fraction": frame.get("qc_debris", pd.Series(False)).mean(),
            "doublet_fraction": frame.get("qc_doublets", pd.Series(False)).mean(),
            "saturated_fraction": frame.get("qc_saturated", pd.Series(False)).mean(),
        }
        channel_stats = _channel_metrics(frame, exclude_flags=True)
        record.update(channel_stats)
        records.append(record)
    return pd.DataFrame(records)


def _flag_debris(df: pd.DataFrame, config: dict[str, float]) -> pd.Series:
    """Flag events with low forward/side scatter (debris)."""
    fsc = _find_channel(df, ("FSC-A", "FSC_A", "fsc_a", "fsc-a"))
    ssc = _find_channel(df, ("SSC-A", "SSC_A", "ssc_a", "ssc-a"))
    if fsc is None or ssc is None:
        return pd.Series(False, index=df.index)
    fsc_cut = np.percentile(df[fsc], config.get("fsc_percentile", 2.0))
    ssc_cut = np.percentile(df[ssc], config.get("ssc_percentile", 2.0))
    return (df[fsc] <= fsc_cut) | (df[ssc] <= ssc_cut)


def _flag_doublets(df: pd.DataFrame, config: dict[str, float]) -> pd.Series:
    """Flag doublets based on FSC-H vs FSC-A deviation."""
    fsc_a = _find_channel(df, ("FSC-A", "FSC_A", "fsc_a", "fsc-a"))
    fsc_h = _find_channel(df, ("FSC-H", "FSC_H", "fsc_h", "fsc-h"))
    if fsc_a is None or fsc_h is None:
        return pd.Series(False, index=df.index)
    expected = df[fsc_a].replace(0, np.nan)
    ratio = (df[fsc_h] / expected).fillna(1.0)
    tol = config.get("tolerance", 0.1)
    return ratio.sub(1.0).abs() > tol


def _flag_saturation(df: pd.DataFrame, config: dict[str, float]) -> pd.Series:
    """Flag saturated fluorescence events."""
    threshold = config.get("threshold", 0.995)
    fluor_channels = [
        col
        for col in df.columns
        if not col.lower().startswith("fsc")
        and not col.lower().startswith("ssc")
        and not col.startswith("qc_")
    ]
    if not fluor_channels:
        return pd.Series(False, index=df.index)
    saturated = pd.Series(False, index=df.index)
    for col in fluor_channels:
        max_val = df[col].max()
        if max_val == 0:
            continue
        saturated |= df[col] >= (max_val * threshold)
    return saturated


def _channel_metrics(df: pd.DataFrame, exclude_flags: bool = False) -> dict[str, float]:
    """Compute median signal and mean IQR across numeric channels."""
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if exclude_flags:
        numeric_cols = [col for col in numeric_cols if not col.startswith("qc_")]
    if not numeric_cols:
        return {}

    # Use more efficient operations
    subset = df[numeric_cols]
    medians = subset.median()
    iqr = subset.quantile(0.75) - subset.quantile(0.25)
    mean_iqr = float(iqr.mean()) if not iqr.empty else float("nan")
    return {
        "median_signal": float(medians.mean()),
        "mean_iqr": mean_iqr,
    }


def _find_channel(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Find first matching channel name from candidates, case-insensitive."""
    for name in candidates:
        if name in df.columns:
            return name
    lower_map = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None
