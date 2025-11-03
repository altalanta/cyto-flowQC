"""Automated gating primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from cytoflow_qc.exceptions import GatingError
from cytoflow_qc.plugins.registry import get_plugin_registry


@dataclass
class GateResult:
    mask: pd.Series
    params: Dict[str, float]


DEFAULT_GATE_CONFIG: Dict[str, Dict[str, float]] = {
    "debris": {"min_percentile": 5.0},
    "singlets": {"tolerance": 0.07},
    "lymphocytes": {"low_percentile": 10.0, "high_percentile": 80.0},
    "viability": {"direction": "below"},
}


def auto_gate(
    df: pd.DataFrame, strategy: str, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Apply the selected gating strategy and return gated data + parameters."""

    if "qc_debris" not in df.columns or "qc_doublet" not in df.columns:
        raise GatingError("QC flags must be computed before gating. Run 'qc' stage first.")

    # Apply QC filters first
    df_clean = df[~df["qc_debris"] & ~df["qc_doublet"]].copy()

    if strategy == "default":
        return _default_gating_strategy(df_clean, config)
    
    # Try to load a plugin for the given strategy
    try:
        registry = get_plugin_registry()
        gating_plugin = registry.load_plugin("gating", strategy, config)
        return gating_plugin.gate(df_clean)
    except Exception as e:
        raise GatingError(f"Failed to load or execute gating strategy '{strategy}': {e}") from e


def _default_gating_strategy(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Default gating: density-based lymphocyte gate."""
    channels = config.get("channels", {})
    fsc_col = channels.get("fsc_a", "FSC-A")
    ssc_col = channels.get("ssc_a", "SSC-A")

    if fsc_col not in df.columns or ssc_col not in df.columns:
        raise GatingError(f"Required channels for gating not found: {fsc_col}, {ssc_col}")

    # Density-based gating for lymphocytes
    gate_params = config.get("lymphocytes", {})
    percentile = gate_params.get("percentile", 90)

    # Subset to avoid memory issues with KDE on large data
    sample_df = df if len(df) < 50000 else df.sample(n=50000)

    kde = gaussian_kde(sample_df[[fsc_col, ssc_col]].T)
    density = kde(df[[fsc_col, ssc_col]].T)
    density_threshold = np.percentile(density, 100 - percentile)

    df["gate_lymphocyte"] = density > density_threshold

    gated_df = df[df["gate_lymphocyte"]].copy()
    params = {"lymphocyte_density_threshold": density_threshold}

    return gated_df, params


def _debris_gate(df: pd.DataFrame, channels: Dict[str, Optional[str]], config: Dict[str, float]) -> GateResult:
    fsc = channels.get("fsc_a")
    ssc = channels.get("ssc_a")
    if fsc not in df or ssc not in df:
        return GateResult(pd.Series(True, index=df.index), {"applied": 0.0})
    percentile = config.get("min_percentile", 5.0)
    fsc_cut = np.percentile(df[fsc], percentile)
    ssc_cut = np.percentile(df[ssc], percentile)
    mask = (df[fsc] >= fsc_cut) & (df[ssc] >= ssc_cut)
    return GateResult(mask, {"fsc_cut": float(fsc_cut), "ssc_cut": float(ssc_cut)})


def _singlet_gate(df: pd.DataFrame, channels: Dict[str, Optional[str]], config: Dict[str, float]) -> GateResult:
    fsc_a = channels.get("fsc_a")
    fsc_h = channels.get("fsc_h")
    if fsc_a not in df or fsc_h not in df:
        return GateResult(pd.Series(True, index=df.index), {"applied": 0.0})
    tol = config.get("tolerance", 0.07)
    ratio = (df[fsc_h] / df[fsc_a].replace(0, np.nan)).fillna(1.0)
    mask = ratio.sub(1.0).abs() <= tol
    return GateResult(mask, {"tolerance": float(tol)})


def _lymph_gate(df: pd.DataFrame, channels: Dict[str, Optional[str]], config: Dict[str, float]) -> GateResult:
    fsc = channels.get("fsc_a")
    ssc = channels.get("ssc_a")
    if fsc not in df or ssc not in df:
        return GateResult(pd.Series(True, index=df.index), {"applied": 0.0})
    lo = config.get("low_percentile", 10.0)
    hi = config.get("high_percentile", 80.0)
    fsc_bounds = np.percentile(df[fsc], [lo, hi])
    ssc_bounds = np.percentile(df[ssc], [lo, hi])
    mask = (
        (df[fsc] >= fsc_bounds[0])
        & (df[fsc] <= fsc_bounds[1])
        & (df[ssc] >= ssc_bounds[0])
        & (df[ssc] <= ssc_bounds[1])
    )
    return GateResult(
        mask,
        {
            "fsc_min": float(fsc_bounds[0]),
            "fsc_max": float(fsc_bounds[1]),
            "ssc_min": float(ssc_bounds[0]),
            "ssc_max": float(ssc_bounds[1]),
        },
    )


def _viability_gate(df: pd.DataFrame, channels: Dict[str, Optional[str]], config: Dict[str, float]) -> GateResult:
    channel = channels.get("viability")
    if channel not in df:
        return GateResult(pd.Series(True, index=df.index), {"applied": 0.0})
    direction = config.get("direction", "below")
    threshold = config.get("threshold")
    if threshold is None:
        threshold = float(df[channel].median())
    if direction == "below":
        mask = df[channel] <= threshold
    else:
        mask = df[channel] >= threshold
    return GateResult(mask, {"threshold": float(threshold), "direction": direction})
