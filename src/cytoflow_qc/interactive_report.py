"""Interactive HTML report generator for CytoFlow-QC."""
import panel as pn
import hvplot.pandas
import holoviews as hv
import pandas as pd
from pathlib import Path

pn.extension("tabulator", sizing_mode="stretch_width")

def load_data(results_dir: Path) -> dict:
    """Load all necessary data from a results directory."""
    data = {}
    files_to_load = {
        "qc_summary": "qc/summary.csv",
        "gate_summary": "gate/summary.csv",
        "drift_pca": "drift/pca.csv",
        "drift_umap": "drift/umap.csv",
        "effect_sizes": "stats/effect_sizes.csv",
    }
    for key, rel_path in files_to_load.items():
        file_path = results_dir / rel_path
        if file_path.exists():
            data[key] = pd.read_csv(file_path)
    return data

def create_interactive_report(results_dir: Path):
    """Create an interactive report dashboard from pipeline results."""
    data = load_data(results_dir)

    # --- QC Tab ---
    qc_tab = pn.Column("## Quality Control Summary")
    if 'qc_summary' in data:
        qc_summary_table = pn.widgets.Tabulator(data['qc_summary'], pagination='local', page_size=10, frozen_columns=['sample_id'])
        qc_summary_plot = data['qc_summary'].hvplot.bar(
            x='sample_id', y='pass_qc_pct', title='QC Pass Percentage', ylim=(0, 1.05), rot=45
        ).opts(responsive=True, min_height=400, yformatter='%.2f')
        qc_tab.extend([qc_summary_plot, qc_summary_table])

    # --- Gating Tab ---
    gating_tab = pn.Column("## Gating Summary")
    if 'gate_summary' in data:
        gating_summary_table = pn.widgets.Tabulator(data['gate_summary'], pagination='local', page_size=10, frozen_columns=['sample_id'])
        gate_summary_plot = data['gate_summary'].hvplot.bar(
            x='sample_id', y='gated_events', title='Gated Events per Sample', rot=45
        ).opts(responsive=True, min_height=400)
        gating_tab.extend([gate_summary_plot, gating_summary_table])

    # --- Drift Tab ---
    drift_tab_content = pn.Tabs()
    if 'drift_pca' in data:
        pca_plot = data['drift_pca'].hvplot.scatter(
            x='PC1', y='PC2', by='batch', title='PCA of Sample Features', hover_cols=['sample_id']
        ).opts(responsive=True, min_height=500)
        drift_tab_content.append(("PCA", pca_plot))
    if 'drift_umap' in data:
        umap_plot = data['drift_umap'].hvplot.scatter(
            x='UMAP1', y='UMAP2', by='batch', title='UMAP of Sample Features', hover_cols=['sample_id']
        ).opts(responsive=True, min_height=500)
        drift_tab_content.append(("UMAP", umap_plot))
    drift_tab = pn.Column("## Batch Drift Analysis", drift_tab_content)

    # --- Stats Tab ---
    stats_tab = pn.Column("## Statistical Analysis")
    if 'effect_sizes' in data:
        effect_sizes_df = data['effect_sizes']
        effect_plot = effect_sizes_df.hvplot.scatter(
            x='hedges_g', y='parameter', c='significant', cmap={'True': 'red', 'False': 'grey'},
            title="Effect Sizes (Hedges' g)"
        ).opts(responsive=True, min_height=400) * hv.VLine(0).opts(color='black', line_dash='dashed')
        stats_tab.extend([effect_plot, pn.widgets.Tabulator(effect_sizes_df, pagination='local', page_size=10)])
    
    # --- Assemble Dashboard ---
    dashboard = pn.Tabs(
        ("Quality Control", qc_tab),
        ("Gating", gating_tab),
        ("Batch Drift", drift_tab),
        ("Statistics", stats_tab),
        dynamic=True
    )
    
    template = pn.template.FastListTemplate(
        site="CytoFlow-QC Report",
        title=f"Analysis Report for: {results_dir.name}",
        main=[dashboard],
    )
    return template

def create_and_save_report(results_dir: Path, output_path: Path):
    """Generate and save the interactive HTML report."""
    dashboard = create_interactive_report(results_dir)
    dashboard.save(filename=output_path, embed=True, embed_json=True, save_path='./')


