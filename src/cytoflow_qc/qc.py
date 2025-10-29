"""Quality-control heuristics for event level data."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, List

import numpy as np
import pandas as pd
from cytoflow_qc.exceptions import QCError

DEFAULT_QC_CONFIG: Dict[str, Dict[str, float]] = {
    "debris": {"fsc_percentile": 2.0, "ssc_percentile": 2.0},
    "doublets": {"tolerance": 0.08},
    "saturation": {"threshold": 0.995},
}


def add_qc_flags(df: pd.DataFrame, qc_config: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Add QC flags to the event DataFrame based on configuration."""

    df["qc_debris"] = _flag_debris(df, **qc_config.get("debris", {}))
    df["qc_doublet"] = _flag_doublets(df, **qc_config.get("doublets", {}))
    return df


def qc_summary(sample_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Generate a summary DataFrame of QC metrics for all samples."""
    records = []
    for sample_id, df in sample_tables.items():
        total = len(df)
        if total == 0:
            records.append({"sample_id": sample_id, "total_events": 0, "pass_qc": 0, "pass_qc_pct": 0.0})
            continue

        debris = df["qc_debris"].sum()
        doublet = df["qc_doublet"].sum()
        passed = total - debris - doublet
        records.append({
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


def _flag_debris(df: pd.DataFrame, method: str = "percentile", **kwargs) -> pd.Series:
    """Flag debris based on FSC/SSC properties."""
    if method == "percentile":
        fsc_a_pct = kwargs.get("fsc_a_pct", 2)
        ssc_a_pct = kwargs.get("ssc_a_pct", 2)
        fsc_low = np.percentile(df["FSC-A"], fsc_a_pct)
        ssc_low = np.percentile(df["SSC-A"], ssc_a_pct)
        return (df["FSC-A"] < fsc_low) | (df["SSC-A"] < ssc_low)
    else:
        raise QCError(f"Unsupported debris removal method: {method}")


def _flag_doublets(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Flag doublets based on FSC-A vs FSC-H."""
    fsc_a = _find_channel(df, ("FSC-A", "FSC_A", "fsc_a", "fsc-a"))
    fsc_h = _find_channel(df, ("FSC-H", "FSC_H", "fsc_h", "fsc-h"))
    if fsc_a is None or fsc_h is None:
        return pd.Series(False, index=df.index)
    expected_fsc_h = df["FSC-A"] * kwargs.get("slope", 1.0) + kwargs.get("intercept", 0.0)
    deviation = np.abs(df["FSC-H"] - expected_fsc_h) / (expected_fsc_h + 1e-6)
    return deviation > kwargs.get("tol", 0.02)


def _flag_saturation(df: pd.DataFrame, config: Dict[str, float]) -> pd.Series:
    threshold = config.get("threshold", 0.995)
    fluor_channels = [
        col
        for col in df.columns
        if not col.lower().startswith("fsc") and not col.lower().startswith("ssc") and not col.startswith("qc_")
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


def _channel_metrics(df: pd.DataFrame, exclude_flags: bool = False) -> Dict[str, float]:
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if exclude_flags:
        numeric_cols = [col for col in numeric_cols if not col.startswith("qc_")]
    if not numeric_cols:
        return {}

    medians = df[numeric_cols].median()
    iqr = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
    mean_iqr = float(iqr.mean()) if not iqr.empty else float("nan")
    return {
        "median_signal": float(medians.mean()),
        "mean_iqr": mean_iqr,
    }


def _find_channel(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    lower_map = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None
