"""Visual regression tests for Matplotlib plots."""
import pytest
from cytoflow_qc.viz import (
    plot_qc_summary,
    plot_gating_scatter,
    plot_batch_drift_pca,
    plot_batch_drift_umap,
    plot_effect_sizes,
)

pytestmark = pytest.mark.mpl_image_compare

def test_qc_summary_plot(pipeline_results):
    return plot_qc_summary(pipeline_results["qc_summary"])

def test_gating_scatter_plot(pipeline_results):
    cfg = pipeline_results["config"]
    return plot_gating_scatter(
        pipeline_results["unfiltered_events"],
        pipeline_results["gated_events"],
        cfg.channels.fsc_a,
        cfg.channels.ssc_a,
    )

def test_pca_plot(pipeline_results):
    return plot_batch_drift_pca(pipeline_results["pca"])

def test_umap_plot(pipeline_results):
    return plot_batch_drift_umap(pipeline_results["umap"])

def test_effect_sizes_plot(pipeline_results):
    return plot_effect_sizes(pipeline_results["effects"])




