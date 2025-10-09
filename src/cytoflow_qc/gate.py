"""Automated gating primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GateResult:
    mask: pd.Series
    params: dict[str, float]


DEFAULT_GATE_CONFIG: dict[str, dict[str, float]] = {
    "debris": {"min_percentile": 5.0},
    "singlets": {"tolerance": 0.07},
    "lymphocytes": {"low_percentile": 10.0, "high_percentile": 80.0},
    "viability": {"direction": "below"},
}


def auto_gate(
    df: pd.DataFrame,
    strategy: str = "default",
    config: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Apply the selected gating strategy and return gated data + parameters."""

    if strategy != "default":
        raise ValueError(f"Unsupported gating strategy: {strategy}")

    cfg = {**DEFAULT_GATE_CONFIG, **(config or {})}
    chan_cfg = config.get("channels", {}) if config else {}
    channels = {
        "fsc_a": chan_cfg.get("fsc_a", "FSC-A"),
        "ssc_a": chan_cfg.get("ssc_a", "SSC-A"),
        "fsc_h": chan_cfg.get("fsc_h", "FSC-H"),
        "viability": chan_cfg.get("viability"),
    }

    working = df.copy()
    gate_log: dict[str, dict[str, float]] = {}

    # Debris gate (use QC flags if present)
    if "qc_debris" in working:
        mask = ~working["qc_debris"]
        params = {"source": "qc_flag"}
    else:
        result = _debris_gate(working, channels, cfg["debris"])
        mask, params = result.mask, result.params
    working = working.loc[mask].copy()
    gate_log["debris"] = params | {"remaining": float(mask.mean())}

    # Singlets
    result = _singlet_gate(working, channels, cfg["singlets"])
    working = working.loc[result.mask].copy()
    gate_log["singlets"] = result.params | {"remaining": float(result.mask.mean())}

    # Lymphocytes
    result = _lymph_gate(working, channels, cfg["lymphocytes"])
    working = working.loc[result.mask].copy()
    gate_log["lymphocytes"] = result.params | {"remaining": float(result.mask.mean())}

    # Viability (optional)
    if channels.get("viability") and channels["viability"] in working.columns:
        result = _viability_gate(working, channels, cfg["viability"])
        working = working.loc[result.mask].copy()
        gate_log["viability"] = result.params | {"remaining": float(result.mask.mean())}

    return working, gate_log


def _debris_gate(
    df: pd.DataFrame, channels: dict[str, str | None], config: dict[str, float]
) -> GateResult:
    fsc = channels.get("fsc_a")
    ssc = channels.get("ssc_a")
    if fsc not in df or ssc not in df:
        return GateResult(pd.Series(True, index=df.index), {"applied": 0.0})
    percentile = config.get("min_percentile", 5.0)
    fsc_cut = np.percentile(df[fsc], percentile)
    ssc_cut = np.percentile(df[ssc], percentile)
    mask = (df[fsc] >= fsc_cut) & (df[ssc] >= ssc_cut)
    return GateResult(mask, {"fsc_cut": float(fsc_cut), "ssc_cut": float(ssc_cut)})


def _singlet_gate(
    df: pd.DataFrame, channels: dict[str, str | None], config: dict[str, float]
) -> GateResult:
    fsc_a = channels.get("fsc_a")
    fsc_h = channels.get("fsc_h")
    if fsc_a not in df or fsc_h not in df:
        return GateResult(pd.Series(True, index=df.index), {"applied": 0.0})
    tol = config.get("tolerance", 0.07)
    ratio = (df[fsc_h] / df[fsc_a].replace(0, np.nan)).fillna(1.0)
    mask = ratio.sub(1.0).abs() <= tol
    return GateResult(mask, {"tolerance": float(tol)})


def _lymph_gate(
    df: pd.DataFrame, channels: dict[str, str | None], config: dict[str, float]
) -> GateResult:
    fsc = channels.get("fsc_a")
    ssc = channels.get("ssc_a")
    if fsc not in df or ssc not in df:
        return GateResult(pd.Series(True, index=df.index), {"applied": 0.0})

    lo = config.get("low_percentile", 10.0)
    hi = config.get("high_percentile", 80.0)

    # Compute percentiles more efficiently
    percentiles = [lo, hi]
    fsc_bounds = np.percentile(df[fsc], percentiles)
    ssc_bounds = np.percentile(df[ssc], percentiles)

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


def _viability_gate(
    df: pd.DataFrame, channels: dict[str, str | None], config: dict[str, float]
) -> GateResult:
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
