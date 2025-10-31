"""Matplotlib plotting helpers used across the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd


def plot_qc_summary(summary_df: pd.DataFrame, output_path: Optional[str] = None, ax: Optional[plt.Axes] = None) -> Figure:
    """Bar chart of QC pass fractions per sample."""
    if summary_df.empty:
        return Figure()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    ax.bar(summary_df["sample_id"], summary_df["pass_qc_pct"], color="#2E86AB")
    ax.set_ylabel("QC pass fraction")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(summary_df["sample_id"], rotation=45, ha="right")
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    return fig


def plot_gating_scatter(
    original: pd.DataFrame,
    gated: pd.DataFrame,
    fsc_channel: str,
    ssc_channel: str,
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Overlay scatter plot showing gated vs ungated FSC/SSC events."""
    if fsc_channel not in original or ssc_channel not in original:
        return Figure()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    ax.scatter(original[fsc_channel], original[ssc_channel], s=5, alpha=0.2, label="All")
    if fsc_channel in gated and ssc_channel in gated:
        ax.scatter(gated[fsc_channel], gated[ssc_channel], s=5, alpha=0.5, label="Gated")
    ax.set_xlabel(fsc_channel)
    ax.set_ylabel(ssc_channel)
    ax.legend()
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    return fig


def plot_batch_drift_pca(pca_df: pd.DataFrame, output_path: Optional[str] = None, batch_col: str = 'batch', ax: Optional[plt.Axes] = None) -> Figure:
    """Scatter plot of the first two PCA components coloured by batch."""
    if pca_df.empty or "PC1" not in pca_df or "PC2" not in pca_df:
        return Figure()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    for batch, subset in pca_df.groupby(batch_col):
        ax.scatter(subset["PC1"], subset["PC2"], s=30, label=batch)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    return fig


def plot_batch_drift_umap(umap_df: Optional[pd.DataFrame], output_path: Optional[str] = None, batch_col: str = 'batch', ax: Optional[plt.Axes] = None) -> Figure:
    """Scatter plot of UMAP embedding if available."""
    if umap_df is None or umap_df.empty:
        return Figure()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    for batch, subset in umap_df.groupby(batch_col):
        ax.scatter(subset["UMAP1"], subset["UMAP2"], s=30, label=batch)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend()
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    return fig


def plot_effect_sizes(effects_df: pd.DataFrame, output_path: Optional[str] = None, ax: Optional[plt.Axes] = None) -> Figure:
    """Dot plot of Hedges' g effect sizes with significance highlighting."""
    if effects_df.empty:
        return Figure()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    ordered = effects_df.sort_values("hedges_g")
    colors = ordered["significant"].map({True: "#C0392B", False: "#7F8C8D"})
    ax.scatter(ordered["hedges_g"], ordered["parameter"], c=colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Hedges' g")
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    return fig
