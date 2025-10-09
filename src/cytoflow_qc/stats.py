"""Effect-size utilities for gated populations."""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy import stats


def effect_sizes(
    df: pd.DataFrame,
    group_col: str,
    value_cols: Iterable[str],
    control_group: str | None = None,
) -> pd.DataFrame:
    """Compute non-parametric effect sizes relative to a control group."""

    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not present")

    groups = df[group_col].dropna().unique().tolist()
    if not groups:
        return pd.DataFrame()
    groups.sort()
    control = control_group or groups[0]
    if control not in groups:
        raise ValueError(f"Control group '{control}' not available")

    results: list[dict[str, Any]] = []
    control_values = df[df[group_col] == control]
    for group in groups:
        if group == control:
            continue
        treatment = df[df[group_col] == group]
        for column in value_cols:
            if column not in df.columns:
                continue
            ctrl_series = control_values[column].dropna()
            trt_series = treatment[column].dropna()
            if ctrl_series.empty or trt_series.empty:
                continue

            hedges = _hedges_g(ctrl_series.to_numpy(), trt_series.to_numpy())
            cliff = _cliffs_delta(ctrl_series.to_numpy(), trt_series.to_numpy())
            try:
                u_stat, p_value = stats.mannwhitneyu(
                    trt_series, ctrl_series, alternative="two-sided"
                )
            except ValueError:
                u_stat, p_value = np.nan, np.nan

            results.append(
                {
                    "parameter": column,
                    "comparison": f"{group}_vs_{control}",
                    "group": group,
                    "control": control,
                    "hedges_g": hedges,
                    "cliffs_delta": cliff,
                    "u_statistic": u_stat,
                    "p_value": p_value,
                    "control_n": len(ctrl_series),
                    "treatment_n": len(trt_series),
                    "control_mean": float(ctrl_series.mean()),
                    "treatment_mean": float(trt_series.mean()),
                }
            )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df
    results_df["adj_p_value"] = _holm_correction(results_df["p_value"].to_numpy())
    SIGNIFICANCE_THRESHOLD = 0.05
    results_df["significant"] = results_df["adj_p_value"] < SIGNIFICANCE_THRESHOLD
    return results_df


def _hedges_g(control: np.ndarray, treatment: np.ndarray) -> float:
    n1, n2 = len(control), len(treatment)
    MIN_SAMPLE_SIZE = 2
    if n1 < MIN_SAMPLE_SIZE or n2 < MIN_SAMPLE_SIZE:
        return float("nan")
    mean_diff = treatment.mean() - control.mean()
    pooled_var = ((n1 - 1) * control.var(ddof=1) + (n2 - 1) * treatment.var(ddof=1)) / (n1 + n2 - 2)
    if pooled_var == 0:
        return float("nan")
    cohen_d = mean_diff / np.sqrt(pooled_var)
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return float(cohen_d * correction)


def _cliffs_delta(control: np.ndarray, treatment: np.ndarray) -> float:
    dominance = 0
    for a in treatment:
        dominance += np.sum(a > control)
        dominance -= np.sum(a < control)
    denom = len(control) * len(treatment)
    if denom == 0:
        return float("nan")
    return float(dominance / denom)


def _holm_correction(p_values: np.ndarray) -> np.ndarray:
    cleaned = np.nan_to_num(p_values, nan=1.0)
    order = np.argsort(cleaned)
    adjusted = np.empty_like(cleaned)
    m = len(cleaned)
    prev = 0.0
    for rank, idx in enumerate(order, start=1):
        value = min(1.0, cleaned[idx] * (m - rank + 1))
        prev = max(prev, value)
        adjusted[idx] = prev
    return adjusted
