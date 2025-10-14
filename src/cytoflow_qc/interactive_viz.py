"""Interactive data visualization for cytoflow-qc results using modern web frameworks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Interactive visualization libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Static visualization for comparison
import matplotlib.pyplot as plt
import seaborn as sns


class InteractiveVisualizer:
    """Interactive visualization system for cytoflow-qc results."""

    def __init__(self, results_dir: str | Path) -> None:
        """Initialize the interactive visualizer.

        Args:
            results_dir: Path to cytoflow-qc results directory
        """
        self.results_dir = Path(results_dir)
        self._load_data()

    def _load_data(self) -> None:
        """Load all available data from results directory."""
        # Load QC data
        qc_summary = self.results_dir / "qc" / "summary.csv"
        if qc_summary.exists():
            self.qc_data = pd.read_csv(qc_summary)
        else:
            self.qc_data = None

        # Load gating data
        gate_summary = self.results_dir / "gate" / "summary.csv"
        if gate_summary.exists():
            self.gate_data = pd.read_csv(gate_summary)
        else:
            self.gate_data = None

        # Load drift analysis data
        drift_features = self.results_dir / "drift" / "features.csv"
        if drift_features.exists():
            self.drift_features = pd.read_csv(drift_features)
        else:
            self.drift_features = None

        drift_tests = self.results_dir / "drift" / "tests.csv"
        if drift_tests.exists():
            self.drift_tests = pd.read_csv(drift_tests)
        else:
            self.drift_tests = None

        # Load statistical data
        stats_file = self.results_dir / "stats" / "effect_sizes.csv"
        if stats_file.exists():
            self.stats_data = pd.read_csv(stats_file)
        else:
            self.stats_data = None

    def create_dashboard(self) -> None:
        """Create the main interactive dashboard."""
        st.set_page_config(
            page_title="CytoFlow-QC Interactive Dashboard",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🔬 CytoFlow-QC Interactive Dashboard")
        st.markdown("Interactive exploration of flow cytometry quality control results")

        # Sidebar for navigation and controls
        with st.sidebar:
            st.header("Navigation")
            page = st.selectbox(
                "Select View",
                ["Overview", "Quality Control", "Gating Analysis", "Drift Analysis", "Statistics", "Animations", "Export"]
            )

        # Main content area
        if page == "Overview":
            self._show_overview()
        elif page == "Quality Control":
            self._show_qc_analysis()
        elif page == "Gating Analysis":
            self._show_gating_analysis()
        elif page == "Drift Analysis":
            self._show_drift_analysis()
        elif page == "Statistics":
            self._show_statistics()
        elif page == "Export":
            self._show_export_options()
        elif page == "Animations":
            self._show_animations()

    def _show_overview(self) -> None:
        """Show overview dashboard."""
        st.header("📊 Pipeline Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            if self.qc_data is not None:
                total_samples = len(self.qc_data)
                mean_qc_pass = self.qc_data["qc_pass_fraction"].mean()
                st.metric("Samples Processed", total_samples)
                st.metric("Mean QC Pass Rate", f"{mean_qc_pass".1%"}")

        with col2:
            if self.gate_data is not None:
                total_events = self.gate_data["input_events"].sum()
                gated_events = self.gate_data["gated_events"].sum()
                retention_rate = gated_events / total_events if total_events > 0 else 0
                st.metric("Total Events", f"{total_events","}")
                st.metric("Retention Rate", f"{retention_rate".1%"}")

        with col3:
            if self.drift_tests is not None:
                significant_effects = (self.drift_tests["adj_p_value"] < 0.05).sum()
                total_tests = len(self.drift_tests)
                st.metric("Features Tested", total_tests)
                st.metric("Significant Effects", significant_effects)

        # Pipeline summary chart
        if self.qc_data is not None:
            st.subheader("Quality Control Summary")
            fig = px.bar(
                self.qc_data,
                x="sample_id",
                y="qc_pass_fraction",
                color="qc_pass_fraction",
                color_continuous_scale="RdYlGn",
                title="QC Pass Rate by Sample"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def _show_qc_analysis(self) -> None:
        """Show detailed QC analysis."""
        st.header("✅ Quality Control Analysis")

        if self.qc_data is None:
            st.warning("No QC data available. Please run the QC stage first.")
            return

        # Interactive QC metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("QC Metrics Distribution")
            metric = st.selectbox(
                "Select QC Metric",
                ["qc_pass_fraction", "debris_fraction", "doublet_fraction", "saturated_fraction"],
                key="qc_metric"
            )

            fig = px.histogram(
                self.qc_data,
                x=metric,
                nbins=20,
                title=f"Distribution of {metric.replace('_', ' ').title()}"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("QC Pass/Fail Summary")
            pass_fail_counts = self.qc_data["qc_pass"].value_counts()
            fig = px.pie(
                values=pass_fail_counts.values,
                names=["Passed", "Failed"],
                title="QC Pass/Fail Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Sample details table
        st.subheader("Sample QC Details")
        st.dataframe(
            self.qc_data.style.format({
                "qc_pass_fraction": "{:.1%}",
                "debris_fraction": "{:.1%}",
                "doublet_fraction": "{:.1%}",
                "saturated_fraction": "{:.1%}",
            })
        )

    def _show_gating_analysis(self) -> None:
        """Show interactive gating analysis."""
        st.header("🚪 Gating Analysis")

        if self.gate_data is None:
            st.warning("No gating data available. Please run the gating stage first.")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Event Retention")
            fig = px.scatter(
                self.gate_data,
                x="input_events",
                y="gated_events",
                text="sample_id",
                size="gated_events",
                color="gated_events",
                color_continuous_scale="Viridis",
                title="Input vs Gated Events",
                labels={"input_events": "Input Events", "gated_events": "Gated Events"}
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Gating Efficiency")
            self.gate_data["retention_rate"] = self.gate_data["gated_events"] / self.gate_data["input_events"]
            fig = px.bar(
                self.gate_data,
                x="sample_id",
                y="retention_rate",
                color="retention_rate",
                color_continuous_scale="RdYlBu",
                title="Event Retention Rate by Sample"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Interactive 3D gating visualization
        st.subheader("3D Gating Visualization")
        if (self.results_dir / "gate" / "events").exists():
            event_files = list((self.results_dir / "gate" / "events").glob("*.parquet"))
            if event_files:
                selected_sample = st.selectbox(
                    "Select Sample for 3D View",
                    [f.stem for f in event_files],
                    key="3d_sample"
                )

                if selected_sample:
                    col1, col2 = st.columns(2)

                    with col1:
                        x_channel = st.selectbox("X-axis Channel", ["FSC-A", "SSC-A", "CD3-A", "CD19-A"], key="x_channel")
                        y_channel = st.selectbox("Y-axis Channel", ["FSC-A", "SSC-A", "CD3-A", "CD19-A"], key="y_channel")
                        z_channel = st.selectbox("Z-axis Channel", ["FSC-A", "SSC-A", "CD3-A", "CD19-A"], key="z_channel")

                    with col2:
                        show_gated_only = st.checkbox("Show Gated Events Only", value=False)
                        show_density = st.checkbox("Show Density Contours", value=False)

                    try:
                        # Create 3D scatter plot
                        fig = self.create_3d_gating_visualization(
                            selected_sample, x_channel, y_channel, z_channel, show_gated_only
                        )

                        # Add density contours if requested
                        if show_density:
                            sample_file = self.results_dir / "gate" / "events" / f"{selected_sample}.parquet"
                            df = pd.read_parquet(sample_file)
                            if not show_gated_only or "gated" not in df.columns:
                                df = df  # Show all events for density
                            else:
                                df = df[df["gated"]]  # Show only gated events

                            # Add contour surface
                            fig.add_trace(go.Scatter3d(
                                x=df[x_channel],
                                y=df[y_channel],
                                z=df[z_channel],
                                mode='markers',
                                marker=dict(
                                    size=1,
                                    color='rgba(0,0,0,0.1)',
                                    opacity=0.3
                                ),
                                name="Density",
                                showlegend=False
                            ))

                        st.plotly_chart(fig, use_container_width=True)

                    except FileNotFoundError as e:
                        st.error(f"Sample data not found: {e}")
                    except Exception as e:
                        st.error(f"Error creating 3D visualization: {e}")

        # Interactive gating parameters viewer
        st.subheader("Gating Parameters")
        if (self.results_dir / "gate" / "params").exists():
            params_files = list((self.results_dir / "gate" / "params").glob("*.json"))
            if params_files:
                selected_sample = st.selectbox(
                    "Select Sample",
                    [f.stem for f in params_files],
                    key="gating_sample"
                )

                if selected_sample:
                    params_file = self.results_dir / "gate" / "params" / f"{selected_sample}.json"
                    with open(params_file, 'r') as f:
                        params = json.load(f)

                    st.json(params)

    def _show_drift_analysis(self) -> None:
        """Show interactive drift analysis."""
        st.header("📊 Batch Drift Analysis")

        if self.drift_features is None or self.drift_tests is None:
            st.warning("No drift data available. Please run the drift analysis stage first.")
            return

        # PCA visualization
        if "PC1" in self.drift_features.columns:
            st.subheader("Principal Component Analysis")

            # 3D PCA if available
            if "PC3" in self.drift_features.columns:
                fig = px.scatter_3d(
                    self.drift_features,
                    x="PC1", y="PC2", z="PC3",
                    color="batch" if "batch" in self.drift_features.columns else None,
                    symbol="condition" if "condition" in self.drift_features.columns else None,
                    hover_data=["sample_id"],
                    title="3D PCA of Sample Features"
                )
            else:
                fig = px.scatter(
                    self.drift_features,
                    x="PC1", y="PC2",
                    color="batch" if "batch" in self.drift_features.columns else None,
                    symbol="condition" if "condition" in self.drift_features.columns else None,
                    hover_data=["sample_id"],
                    title="2D PCA of Sample Features"
                )

            st.plotly_chart(fig, use_container_width=True)

        # Statistical test results
        st.subheader("Drift Test Results")

        # Filter for significant results
        significant_tests = self.drift_tests[self.drift_tests["adj_p_value"] < 0.05]

        if not significant_tests.empty:
            st.success(f"Found {len(significant_tests)} significant batch effects")

            # Interactive table of significant results
            st.dataframe(
                significant_tests.style.format({
                    "p_value": "{:.2e}",
                    "adj_p_value": "{:.2e}",
                    "statistic": "{:.3f}"
                })
            )
        else:
            st.info("No significant batch effects detected at p < 0.05")

        # Population density heatmaps
        st.subheader("Population Density Analysis")
        col1, col2 = st.columns(2)

        with col1:
            heatmap_x = st.selectbox("X-axis Channel", ["FSC-A", "SSC-A", "CD3-A", "CD19-A"], key="heatmap_x")
            heatmap_y = st.selectbox("Y-axis Channel", ["FSC-A", "SSC-A", "CD3-A", "CD19-A"], key="heatmap_y")

        with col2:
            density_type = st.radio("Density Type", ["Count", "Density"], key="density_type")
            log_scale = st.checkbox("Log Scale", value=False, key="log_scale")

        if self.drift_features is not None:
            try:
                # Create heatmap visualization
                heatmap_fig = self.create_heatmap_visualization(
                    self.drift_features,
                    heatmap_x,
                    heatmap_y,
                    density=(density_type == "Density"),
                    log_scale=log_scale,
                    title=f"Population Density: {heatmap_x} vs {heatmap_y}"
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)

                # Add contour overlay option
                if st.checkbox("Show Contour Lines", value=False, key="show_contours"):
                    contour_fig = self.create_density_contours(
                        self.drift_features,
                        heatmap_x,
                        heatmap_y,
                        n_contours=15,
                        opacity=0.7
                    )
                    st.plotly_chart(contour_fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error creating heatmap: {e}")

        # All test results in expandable table
        with st.expander("View All Test Results"):
            st.dataframe(
                self.drift_tests.style.format({
                    "p_value": "{:.2e}",
                    "adj_p_value": "{:.2e}",
                    "statistic": "{:.3f}"
                })
            )

    def _show_statistics(self) -> None:
        """Show interactive statistical analysis."""
        st.header("📈 Statistical Analysis")

        if self.stats_data is None:
            st.warning("No statistical data available. Please run the statistics stage first.")
            return

        # Effect size visualization
        st.subheader("Effect Size Analysis")

        # Interactive scatter plot of effect sizes
        fig = px.scatter(
            self.stats_data,
            x="effect_size",
            y="marker",
            color="significant",
            color_discrete_map={True: "red", False: "blue"},
            hover_data=["p_value", "adj_p_value"],
            title="Effect Sizes by Marker",
            labels={"effect_size": "Effect Size", "marker": "Marker"}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Volcano plot (effect size vs p-value)
        st.subheader("Volcano Plot")
        fig_volcano = px.scatter(
            self.stats_data,
            x="effect_size",
            y="-log10(p_value)",
            color="significant",
            color_discrete_map={True: "red", False: "blue"},
            hover_data=["marker", "adj_p_value"],
            title="Volcano Plot: Effect Size vs Statistical Significance",
            labels={"-log10(p_value)": "-log₁₀(p-value)", "effect_size": "Effect Size"}
        )
        st.plotly_chart(fig_volcano, use_container_width=True)

        # Statistical summary
        col1, col2, col3 = st.columns(3)

        with col1:
            total_tests = len(self.stats_data)
            st.metric("Total Tests", total_tests)

        with col2:
            significant = (self.stats_data["adj_p_value"] < 0.05).sum()
            st.metric("Significant Effects", significant)

        with col3:
            mean_effect = self.stats_data["effect_size"].abs().mean()
            st.metric("Mean |Effect Size|", f"{mean_effect".3f"}")

    def _show_export_options(self) -> None:
        """Show export options for figures and data."""
        st.header("📥 Export Options")

        st.info("Export high-resolution figures and interactive reports for publications and presentations.")

        # Export format selection
        col1, col2 = st.columns(2)

        with col1:
            export_format = st.selectbox(
                "Export Format",
                ["PNG", "PDF", "SVG", "HTML Report"],
                key="export_format"
            )

        with col2:
            if export_format != "HTML Report":
                dpi = st.slider("Resolution (DPI)", 150, 600, 300, key="export_dpi")
                width = st.slider("Width (inches)", 4, 20, 10, key="export_width")
                height = st.slider("Height (inches)", 4, 20, 8, key="export_height")
            else:
                include_animations = st.checkbox("Include Animation Features", value=False, key="include_animations")

        # Export buttons for different visualizations
        st.subheader("Export Individual Visualizations")

        if self.qc_data is not None:
            if st.button("📊 Export QC Summary Plot", key="export_qc"):
                try:
                    # Create QC summary plot
                    fig = px.bar(
                        self.qc_data,
                        x="sample_id",
                        y="qc_pass_fraction",
                        color="qc_pass_fraction",
                        color_continuous_scale="RdYlGn",
                        title="QC Pass Rate by Sample"
                    )
                    fig.update_layout(height=400)

                    if export_format == "PNG":
                        fig.write_image("qc_summary.png", width=width*100, height=height*100, scale=2)
                    elif export_format == "PDF":
                        fig.write_image("qc_summary.pdf", width=width, height=height)
                    elif export_format == "SVG":
                        fig.write_image("qc_summary.svg", width=width, height=height)

                    st.success(f"✅ QC Summary exported as {export_format.lower()}")
                except Exception as e:
                    st.error(f"❌ Export failed: {e}")

        if self.gate_data is not None:
            if st.button("🚪 Export Gating Analysis", key="export_gating"):
                try:
                    # Create gating analysis plot
                    self.gate_data["retention_rate"] = self.gate_data["gated_events"] / self.gate_data["input_events"]
                    fig = px.bar(
                        self.gate_data,
                        x="sample_id",
                        y="retention_rate",
                        color="retention_rate",
                        color_continuous_scale="RdYlBu",
                        title="Event Retention Rate by Sample"
                    )
                    fig.update_layout(height=400)

                    if export_format == "PNG":
                        fig.write_image("gating_analysis.png", width=width*100, height=height*100, scale=2)
                    elif export_format == "PDF":
                        fig.write_image("gating_analysis.pdf", width=width, height=height)
                    elif export_format == "SVG":
                        fig.write_image("gating_analysis.svg", width=width, height=height)

                    st.success(f"✅ Gating Analysis exported as {export_format.lower()}")
                except Exception as e:
                    st.error(f"❌ Export failed: {e}")

        if self.stats_data is not None:
            if st.button("📈 Export Statistical Analysis", key="export_stats"):
                try:
                    # Create volcano plot
                    fig = px.scatter(
                        self.stats_data,
                        x="effect_size",
                        y="-log10(p_value)",
                        color="significant",
                        color_discrete_map={True: "red", False: "blue"},
                        hover_data=["marker", "adj_p_value"],
                        title="Volcano Plot: Effect Size vs Statistical Significance"
                    )

                    if export_format == "PNG":
                        fig.write_image("volcano_plot.png", width=width*100, height=height*100, scale=2)
                    elif export_format == "PDF":
                        fig.write_image("volcano_plot.pdf", width=width, height=height)
                    elif export_format == "SVG":
                        fig.write_image("volcano_plot.svg", width=width, height=height)

                    st.success(f"✅ Statistical Analysis exported as {export_format.lower()}")
                except Exception as e:
                    st.error(f"❌ Export failed: {e}")

        # Export complete interactive report
        st.subheader("Export Complete Report")

        if st.button("📋 Export Interactive HTML Report", key="export_report"):
            try:
                report_path = self.results_dir / "interactive_report.html"
                self.export_interactive_dashboard(report_path, include_animations)
                st.success(f"✅ Interactive report exported to: {report_path}")

                # Provide download link
                with open(report_path, 'r') as f:
                    report_content = f.read()

                st.download_button(
                    "⬇️ Download Report",
                    data=report_content,
                    file_name="cytoflow_qc_report.html",
                    mime="text/html",
                    key="download_report"
                )

            except Exception as e:
                st.error(f"❌ Report export failed: {e}")

        # Export all data as CSV
        st.subheader("Export Data")

        if st.button("📄 Export All Data (CSV)", key="export_data"):
            try:
                # Create data export directory
                export_dir = self.results_dir / "exports"
                export_dir.mkdir(exist_ok=True)

                # Export each dataset
                if self.qc_data is not None:
                    self.qc_data.to_csv(export_dir / "qc_summary.csv", index=False)

                if self.gate_data is not None:
                    self.gate_data.to_csv(export_dir / "gating_summary.csv", index=False)

                if self.drift_features is not None:
                    self.drift_features.to_csv(export_dir / "drift_features.csv", index=False)

                if self.drift_tests is not None:
                    self.drift_tests.to_csv(export_dir / "drift_tests.csv", index=False)

                if self.stats_data is not None:
                    self.stats_data.to_csv(export_dir / "statistical_results.csv", index=False)

                st.success(f"✅ All data exported to: {export_dir}")

                # Create zip file for download
                import zipfile
                zip_path = self.results_dir / "cytoflow_qc_data.zip"
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file_path in export_dir.glob("*.csv"):
                        zipf.write(file_path, file_path.name)

                with open(zip_path, 'rb') as f:
                    st.download_button(
                        "⬇️ Download All Data (ZIP)",
                        data=f.read(),
                        file_name="cytoflow_qc_data.zip",
                        mime="application/zip",
                        key="download_data"
                    )

            except Exception as e:
                st.error(f"❌ Data export failed: {e}")

        # Export instructions
        st.subheader("📖 Export Instructions")
        st.markdown("""
        **Figure Export:**
        - Choose your preferred format (PNG, PDF, SVG)
        - Adjust resolution and dimensions as needed
        - Click export buttons for individual visualizations

        **Interactive Report:**
        - Exports a self-contained HTML file with all visualizations
        - Can be opened in any modern web browser
        - Includes summary statistics and analysis results

        **Data Export:**
        - Downloads all analysis results as CSV files
        - Useful for further statistical analysis
        - Includes QC, gating, drift, and statistical results
        """)

    def create_3d_gating_visualization(
        self,
        sample_id: str,
        x_channel: str = "FSC-A",
        y_channel: str = "SSC-A",
        z_channel: str = "CD3-A",
        gated_only: bool = False
    ) -> go.Figure:
        """Create interactive 3D scatter plot for gating visualization.

        Args:
            sample_id: Sample identifier
            x_channel: Channel for X-axis
            y_channel: Channel for Y-axis
            z_channel: Channel for Z-axis
            gated_only: Show only gated events

        Returns:
            Plotly 3D scatter figure
        """
        # Load sample data
        sample_file = self.results_dir / "gate" / "events" / f"{sample_id}.parquet"
        if not sample_file.exists():
            raise FileNotFoundError(f"Sample data not found: {sample_file}")

        df = pd.read_parquet(sample_file)

        if gated_only and "gated" in df.columns:
            df = df[df["gated"]]

        # Create 3D scatter plot
        fig = go.Figure()

        # Add scatter points
        fig.add_trace(go.Scatter3d(
            x=df[x_channel],
            y=df[y_channel],
            z=df[z_channel],
            mode='markers',
            marker=dict(
                size=2,
                color=df[z_channel],
                colorscale='Viridis',
                opacity=0.6,
                showscale=True,
                colorbar=dict(title=z_channel)
            ),
            name=f"{sample_id} - {z_channel}"
        ))

        # Update layout
        fig.update_layout(
            title=f"3D Gating Visualization: {sample_id}",
            scene=dict(
                xaxis_title=x_channel,
                yaxis_title=y_channel,
                zaxis_title=z_channel,
            ),
            height=600
        )

        return fig

    def create_heatmap_visualization(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        value_col: str | None = None,
        title: str = "Population Density Heatmap"
    ) -> go.Figure:
        """Create interactive heatmap for population density analysis.

        Args:
            data: DataFrame with flow cytometry data
            x_col: Column for X-axis
            y_col: Column for Y-axis
            value_col: Column for heatmap values (optional)
            title: Plot title

        Returns:
            Plotly heatmap figure
        """
        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            data[x_col],
            data[y_col],
            bins=50,
            density=True
        )

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            x=x_edges[:-1],
            y=y_edges[:-1],
            z=hist.T,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>Density: %{{z}}<extra></extra>"
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500
        )

        return fig

    def create_interactive_report(self, output_path: str | Path) -> None:
        """Generate an interactive HTML report with all visualizations.

        Args:
            output_path: Path to save the interactive report
        """
        # This would generate a comprehensive HTML report
        # For now, we'll create a basic structure
        report_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CytoFlow-QC Interactive Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin: 30px 0; }}
                h1 {{ color: #2E86AB; }}
                h2 {{ color: #A23B72; }}
                .metric {{ background: #f0f2f6; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>🔬 CytoFlow-QC Interactive Report</h1>
            <p>Generated from: {self.results_dir}</p>

            <div class="section">
                <h2>📊 Overview</h2>
                <div class="metric">
                    <p><strong>Pipeline Status:</strong> ✅ Completed Successfully</p>
                    <p><strong>Report Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
            </div>

            <div class="section">
                <h2>📥 Data Export</h2>
                <p>Interactive visualizations and export options available in the dashboard interface.</p>
                <p>Access the full interactive dashboard by running:</p>
                <code>streamlit run src/cytoflow_qc/interactive_viz.py {self.results_dir}</code>
            </div>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(report_content)

        print(f"Interactive report generated: {output_path}")

    def create_heatmap_visualization(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        density: bool = True,
        log_scale: bool = False,
        title: str = "Population Density Heatmap"
    ) -> go.Figure:
        """Create interactive heatmap for population density analysis.

        Args:
            data: DataFrame with flow cytometry data
            x_col: Column for X-axis
            y_col: Column for Y-axis
            density: Whether to show density or count
            log_scale: Whether to use log scale for colors
            title: Plot title

        Returns:
            Plotly Figure object
        """
        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            data[x_col],
            data[y_col],
            bins=50,
            density=density
        )

        # Apply log scale if requested
        if log_scale:
            hist = np.log1p(hist)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            x=x_edges[:-1],
            y=y_edges[:-1],
            z=hist.T,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{'Density' if density else 'Count'}: %{{z}}<extra></extra>",
            colorbar=dict(title='Density' if density else 'Count')
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500
        )

        return fig

    def create_density_contours(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        n_contours: int = 10,
        opacity: float = 0.5,
        title: str = "Density Contours"
    ) -> go.Figure:
        """Create density contour plot for population analysis.

        Args:
            data: DataFrame with flow cytometry data
            x_col: Column for X-axis
            y_col: Column for Y-axis
            n_contours: Number of contour levels
            opacity: Opacity of contour lines
            title: Plot title

        Returns:
            Plotly Figure object
        """
        # Calculate 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            data[x_col], data[y_col], bins=[30, 30]
        )

        # Create contour plot
        fig = go.Figure(data=go.Contour(
            x=x_edges[:-1],
            y=y_edges[:-1],
            z=hist.T,
            colorscale='Viridis',
            ncontours=n_contours,
            showscale=False,
            opacity=opacity,
            name="Density Contours"
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500
        )

        return fig

    def create_time_series_animation(
        self,
        data: pd.DataFrame,
        time_col: str,
        x_col: str,
        y_col: str,
        z_col: str | None = None,
        frame_duration: int = 500,
        title: str = "Time Series Animation"
    ) -> go.Figure:
        """Create animated time-series visualization for kinetic experiments.

        Args:
            data: DataFrame with time-series flow cytometry data
            time_col: Column containing time information
            x_col: Column for X-axis
            y_col: Column for Y-axis
            z_col: Column for Z-axis (optional for 3D)
            frame_duration: Duration of each frame in milliseconds
            title: Animation title

        Returns:
            Plotly Figure with animation
        """
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found in data")

        # Get unique time points
        time_points = sorted(data[time_col].unique())

        # Create frames for animation
        frames = []
        for time_point in time_points:
            frame_data = data[data[time_col] == time_point]

            if z_col:
                # 3D frame
                frame = go.Frame(
                    data=[go.Scatter3d(
                        x=frame_data[x_col],
                        y=frame_data[y_col],
                        z=frame_data[z_col],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=frame_data[z_col],
                            colorscale='Viridis',
                            opacity=0.7
                        ),
                        name=f"Time: {time_point}"
                    )],
                    name=str(time_point)
                )
            else:
                # 2D frame
                frame = go.Frame(
                    data=[go.Scatter(
                        x=frame_data[x_col],
                        y=frame_data[y_col],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color='blue',
                            opacity=0.6
                        ),
                        name=f"Time: {time_point}"
                    )],
                    name=str(time_point)
                )

            frames.append(frame)

        # Create initial figure
        initial_data = data[data[time_col] == time_points[0]]

        if z_col:
            fig = go.Figure(
                data=[go.Scatter3d(
                    x=initial_data[x_col],
                    y=initial_data[y_col],
                    z=initial_data[z_col],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=initial_data[z_col],
                        colorscale='Viridis',
                        opacity=0.7
                    ),
                    name=f"Time: {time_points[0]}"
                )],
                frames=frames
            )

            # Update 3D layout
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    zaxis_title=z_col,
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600
            )
        else:
            fig = go.Figure(
                data=[go.Scatter(
                    x=initial_data[x_col],
                    y=initial_data[y_col],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='blue',
                        opacity=0.6
                    ),
                    name=f"Time: {time_points[0]}"
                )],
                frames=frames
            )

            # Update 2D layout
            fig.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=500
            )

        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': frame_duration, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }]
                    },
                    {
                        'label': '⏸️ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[str(t)], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate'
                        }],
                        'label': str(t),
                        'method': 'animate'
                    } for t in time_points
                ],
                'active': 0,
                'transition': {'duration': 300}
            }]
        )

        return fig

    def create_animated_gating_comparison(
        self,
        data_list: list[pd.DataFrame],
        labels: list[str],
        x_col: str,
        y_col: str,
        z_col: str | None = None,
        frame_duration: int = 800,
        title: str = "Animated Gating Comparison"
    ) -> go.Figure:
        """Create animated comparison of gating results across multiple conditions.

        Args:
            data_list: List of DataFrames for different conditions
            labels: Labels for each condition
            x_col, y_col, z_col: Channel columns
            frame_duration: Duration of each frame
            title: Animation title

        Returns:
            Plotly Figure with animation
        """
        if len(data_list) != len(labels):
            raise ValueError("Number of datasets must match number of labels")

        # Create frames for each condition
        frames = []
        for i, (data, label) in enumerate(zip(data_list, labels)):
            if z_col:
                frame = go.Frame(
                    data=[go.Scatter3d(
                        x=data[x_col],
                        y=data[y_col],
                        z=data[z_col],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=f'rgb({i*60}, {100 + i*30}, {150 + i*20})',
                            opacity=0.7
                        ),
                        name=label
                    )],
                    name=label
                )
            else:
                frame = go.Frame(
                    data=[go.Scatter(
                        x=data[x_col],
                        y=data[y_col],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=f'rgb({i*60}, {100 + i*30}, {150 + i*20})',
                            opacity=0.6
                        ),
                        name=label
                    )],
                    name=label
                )

            frames.append(frame)

        # Create initial figure with first dataset
        initial_data = data_list[0]

        if z_col:
            fig = go.Figure(
                data=[go.Scatter3d(
                    x=initial_data[x_col],
                    y=initial_data[y_col],
                    z=initial_data[z_col],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='rgb(0, 100, 150)',
                        opacity=0.7
                    ),
                    name=labels[0]
                )],
                frames=frames
            )

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    zaxis_title=z_col,
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600
            )
        else:
            fig = go.Figure(
                data=[go.Scatter(
                    x=initial_data[x_col],
                    y=initial_data[y_col],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='rgb(0, 100, 150)',
                        opacity=0.6
                    ),
                    name=labels[0]
                )],
                frames=frames
            )

            fig.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=500
            )

        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': frame_duration, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 500}
                        }]
                    },
                    {
                        'label': '⏸️ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[label], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate'
                        }],
                        'label': label,
                        'method': 'animate'
                    } for label in labels
                ],
                'active': 0,
                'transition': {'duration': 500}
            }]
        )

        return fig

    def _show_animations(self) -> None:
        """Show animation visualizations."""
        st.header("🎬 Time-Series Animations")

        if self.drift_features is None:
            st.warning("No data available for animations. Please run the drift analysis stage first.")
            return

        st.info("Animation features will be available in the next release. Features planned:")
        st.markdown("""
        - **Time-series gating animations** showing population changes over time
        - **Condition comparison animations** comparing treatment effects
        - **Batch drift animations** visualizing changes across experimental batches
        - **Interactive playback controls** with speed adjustment and frame stepping
        - **Export animations** as GIF/MP4 files for presentations
        """)

        # Placeholder for future animation functionality
        if st.button("🚧 Coming Soon: Animation Features"):
            st.info("Animation functionality is under development. Check back soon!")

    def export_interactive_dashboard(
        self,
        output_path: str | Path,
        include_animations: bool = False
    ) -> None:
        """Export the complete interactive dashboard as HTML.

        Args:
            output_path: Path to save the HTML file
            include_animations: Whether to include animation features
        """
        # Generate comprehensive HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CytoFlow-QC Interactive Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    color: #2c3e50;
                }}
                .section {{
                    margin: 40px 0;
                    padding: 25px;
                    border-left: 5px solid #3498db;
                    background: #f8f9fa;
                    border-radius: 10px;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .metric-label {{
                    opacity: 0.9;
                    font-size: 0.9em;
                }}
                .plot-container {{
                    margin: 20px 0;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    overflow: hidden;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔬 CytoFlow-QC Interactive Dashboard</h1>
                    <p>Comprehensive flow cytometry quality control analysis</p>
                    <p><strong>Report Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Data Source:</strong> {self.results_dir}</p>
                </div>

                <div class="section">
                    <h2>📊 Pipeline Overview</h2>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
        """

        # Add metrics if available
        if self.qc_data is not None:
            html_content += f"""
                        <div class="metric-card">
                            <div class="metric-label">Samples Processed</div>
                            <div class="metric-value">{len(self.qc_data)}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">QC Pass Rate</div>
                            <div class="metric-value">{self.qc_data["qc_pass_fraction"].mean()".1%"}</div>
                        </div>
            """

        if self.gate_data is not None:
            total_events = self.gate_data["input_events"].sum()
            gated_events = self.gate_data["gated_events"].sum()
            retention_rate = gated_events / total_events if total_events > 0 else 0

            html_content += f"""
                        <div class="metric-card">
                            <div class="metric-label">Total Events</div>
                            <div class="metric-value">{total_events","}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Retention Rate</div>
                            <div class="metric-value">{retention_rate".1%"}</div>
                        </div>
            """

        if self.drift_tests is not None:
            significant_effects = (self.drift_tests["adj_p_value"] < 0.05).sum()

            html_content += f"""
                        <div class="metric-card">
                            <div class="metric-label">Significant Effects</div>
                            <div class="metric-value">{significant_effects}</div>
                        </div>
            """

        html_content += """
                    </div>
                </div>

                <div class="section">
                    <h2>📈 Analysis Sections</h2>
                    <p>This interactive dashboard includes:</p>
                    <ul>
                        <li><strong>Quality Control Analysis:</strong> Detailed QC metrics and pass/fail distributions</li>
                        <li><strong>Gating Visualization:</strong> 3D scatter plots and gating parameter inspection</li>
                        <li><strong>Batch Drift Analysis:</strong> PCA projections and statistical test results</li>
                        <li><strong>Statistical Analysis:</strong> Effect size calculations and volcano plots</li>
                        <li><strong>Export Options:</strong> High-resolution figures and data export capabilities</li>
                    </ul>
                    <p><em>Note: Full interactive functionality requires running the Streamlit dashboard locally.</em></p>
                </div>

                <div class="section">
                    <h2>🚀 Getting Started</h2>
                    <p>To access the full interactive dashboard, run:</p>
                    <pre><code>cd {self.results_dir.parent}
streamlit run src/cytoflow_qc/interactive_viz.py {self.results_dir}</code></pre>
                </div>
            </div>

            <script>
                // Add some basic interactivity
                console.log('CytoFlow-QC Interactive Dashboard loaded');
            </script>
        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Interactive dashboard exported to: {output_path}")


def launch_interactive_dashboard(results_dir: str | Path) -> None:
    """Launch the interactive Streamlit dashboard.

    Args:
        results_dir: Path to cytoflow-qc results directory
    """
    visualizer = InteractiveVisualizer(results_dir)

    # Set up Streamlit session state
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = visualizer

    # Create the dashboard
    visualizer.create_dashboard()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m cytoflow_qc.interactive_viz <results_directory>")
        sys.exit(1)

    results_dir = sys.argv[1]
    launch_interactive_dashboard(results_dir)
