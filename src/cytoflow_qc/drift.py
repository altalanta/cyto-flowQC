"""Batch drift feature engineering and statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - optional dependency
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None


def extract_sample_features(
    samples: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    marker_channels: list[str] | None = None,
) -> pd.DataFrame:
    """Convert per-sample events into a feature matrix for drift analysis."""

    records = []
    for sample_id, frame in samples.items():
        record: dict[str, float] = {
            "sample_id": sample_id,
            "n_events": float(len(frame)),
            "qc_pass_fraction": float(frame.get("qc_pass", pd.Series(True)).mean()),
        }
        numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
        if marker_channels:
            numeric_cols = [col for col in numeric_cols if col in marker_channels]
        for col in numeric_cols:
            record[f"median_{col}"] = float(frame[col].median())
            record[f"iqr_{col}"] = float(frame[col].quantile(0.75) - frame[col].quantile(0.25))
        records.append(record)

    features = pd.DataFrame(records)
    if features.empty:
        return features
    merged = features.merge(metadata, on="sample_id", how="left")
    return merged


def compute_batch_drift(features_df: pd.DataFrame, by: str = "batch") -> dict[str, Any]:
    """Run simple statistical tests and embeddings to flag drift across batches."""

    if by not in features_df.columns:
        raise ValueError(f"Batch column '{by}' missing from features table")

    feature_cols = [
        col
        for col in features_df.columns
        if col not in {by, "sample_id"} and pd.api.types.is_numeric_dtype(features_df[col])
    ]
    if not feature_cols:
        raise ValueError("No numeric features available for drift analysis")

    batches = features_df[by].unique().tolist()
    test_rows = []
    for col in feature_cols:
        groups = [features_df.loc[features_df[by] == batch, col].dropna() for batch in batches]
        groups = [g for g in groups if len(g) > 0]
        MIN_GROUPS_FOR_TEST = 2
        if len(groups) < MIN_GROUPS_FOR_TEST:
            continue
        try:
            stat, p_value = stats.kruskal(*groups)
        except ValueError:
            continue
        test_rows.append(
            {"feature": col, "test": "kruskal", "statistic": float(stat), "p_value": float(p_value)}
        )

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests["adj_p_value"] = _holm_correction(tests["p_value"].to_numpy())
    else:
        tests["adj_p_value"] = []

    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        features_df[feature_cols].fillna(features_df[feature_cols].median())
    )
    pca = PCA(n_components=min(3, scaled.shape[1]))
    pca_coords = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(pca_coords, columns=[f"PC{i+1}" for i in range(pca_coords.shape[1])])
    pca_df["sample_id"] = features_df["sample_id"].values
    pca_df[by] = features_df[by].values

    umap_df = None
    MIN_SAMPLES_FOR_UMAP = 2
    if umap is not None and scaled.shape[0] > MIN_SAMPLES_FOR_UMAP:  # pragma: no cover - optional path
        reducer = umap.UMAP(random_state=42)
        coords = reducer.fit_transform(scaled)
        umap_df = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"])
        umap_df["sample_id"] = features_df["sample_id"].values
        umap_df[by] = features_df[by].values

    SIGNIFICANCE_THRESHOLD = 0.05
    significant = (
        tests.loc[tests["adj_p_value"] < SIGNIFICANCE_THRESHOLD, "feature"].tolist() if not tests.empty else []
    )
    return {
        "tests": tests,
        "pca": pca_df,
        "umap": umap_df,
        "significant_features": significant,
        "drift_detected": bool(significant),
        "feature_columns": feature_cols,
        "batch_column": by,
    }


def _holm_correction(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    m = len(p_values)
    prev = 0.0
    for rank, idx in enumerate(order, start=1):
        value = min(1.0, p_values[idx] * (m - rank + 1))
        prev = max(prev, value)
        adjusted[idx] = prev
    return adjusted
