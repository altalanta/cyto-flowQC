"""3D and advanced visualization capabilities for cytoflow-qc."""

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


class ThreeDVisualizer:
    """Advanced 3D visualization tools for flow cytometry data."""

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        """Initialize the 3D visualizer.

        Args:
            data: DataFrame with flow cytometry data (optional)
        """
        self.data = data

    def create_3d_scatter(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        color_col: str | None = None,
        size_col: str | None = None,
        opacity: float = 0.7,
        title: str = "3D Scatter Plot",
        show_legend: bool = True
    ) -> go.Figure:
        """Create an interactive 3D scatter plot.

        Args:
            data: DataFrame with the data to visualize
            x_col: Column name for X-axis
            y_col: Column name for Y-axis
            z_col: Column name for Z-axis
            color_col: Column name for color mapping (optional)
            size_col: Column name for point size (optional)
            opacity: Opacity of points (0-1)
            title: Plot title
            show_legend: Whether to show legend

        Returns:
            Plotly Figure object
        """
        # Prepare data
        if color_col and color_col not in data.columns:
            color_col = None
        if size_col and size_col not in data.columns:
            size_col = None

        # Create figure
        fig = go.Figure()

        # Add scatter points
        scatter_data = {
            'x': data[x_col],
            'y': data[y_col],
            'z': data[z_col],
            'mode': 'markers',
            'marker': {
                'size': 3,
                'opacity': opacity,
                'showscale': color_col is not None
            },
            'name': title
        }

        # Add color mapping if specified
        if color_col:
            scatter_data['marker']['color'] = data[color_col]
            scatter_data['marker']['colorbar'] = {'title': color_col}

        # Add size mapping if specified
        if size_col:
            # Normalize size for better visualization
            size_values = data[size_col]
            if size_values.max() > size_values.min():
                normalized_size = 3 + 10 * (size_values - size_values.min()) / (size_values.max() - size_values.min())
            else:
                normalized_size = 5
            scatter_data['marker']['size'] = normalized_size

        fig.add_trace(go.Scatter3d(**scatter_data))

        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=show_legend,
            height=600
        )

        return fig

    def create_density_contours(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str | None = None,
        n_contours: int = 10,
        colorscale: str = "Viridis",
        opacity: float = 0.5,
        show_points: bool = False
    ) -> go.Figure:
        """Create density contour surfaces for 3D data.

        Args:
            data: DataFrame with the data
            x_col: Column name for X-axis
            y_col: Column name for Y-axis
            z_col: Column name for Z-axis (optional, for 3D contours)
            n_contours: Number of contour levels
            colorscale: Color scheme for contours
            opacity: Opacity of contour surfaces
            show_points: Whether to show original data points

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        if z_col:
            # 3D contour surface
            # Create a grid for contour calculation
            x_range = np.linspace(data[x_col].min(), data[x_col].max(), 50)
            y_range = np.linspace(data[y_col].min(), data[y_col].max(), 50)
            X, Y = np.meshgrid(x_range, y_range)

            # Interpolate Z values
            from scipy.interpolate import griddata
            Z = griddata(
                (data[x_col], data[y_col]),
                data[z_col],
                (X, Y),
                method='cubic',
                fill_value=0
            )

            # Create contour surface
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale=colorscale,
                opacity=opacity,
                showscale=False,
                name="Density Surface"
            ))

            # Set camera angle for better 3D view
            camera = dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        else:
            # 2D contour plot
            # Create contour lines
            x_range = np.linspace(data[x_col].min(), data[x_col].max(), 30)
            y_range = np.linspace(data[y_col].min(), data[y_col].max(), 30)

            # Calculate 2D histogram
            hist, x_edges, y_edges = np.histogram2d(
                data[x_col], data[y_col], bins=[30, 30]
            )

            # Create contour plot
            fig.add_trace(go.Contour(
                x=x_edges[:-1],
                y=y_edges[:-1],
                z=hist.T,
                colorscale=colorscale,
                ncontours=n_contours,
                showscale=False,
                opacity=opacity,
                name="Density Contours"
            ))

            camera = None

        # Add original data points if requested
        if show_points:
            fig.add_trace(go.Scatter3d(
                x=data[x_col],
                y=data[y_col],
                z=data[z_col] if z_col else np.zeros(len(data)),
                mode='markers',
                marker=dict(
                    size=2,
                    color='black',
                    opacity=0.3
                ),
                name="Data Points",
                showlegend=True
            ))

        # Update layout
        title = f"Density Contours: {x_col} vs {y_col}"
        if z_col:
            title += f" vs {z_col}"

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col if z_col else None,
                camera=camera
            ) if z_col else dict(
                xaxis_title=x_col,
                yaxis_title=y_col
            ),
            height=600 if z_col else 500
        )

        return fig

    def create_multi_panel_3d_view(
        self,
        data: pd.DataFrame,
        channel_pairs: list[tuple[str, str, str]],
        title: str = "Multi-Panel 3D Visualization"
    ) -> go.Figure:
        """Create a multi-panel view showing multiple 3D scatter plots.

        Args:
            data: DataFrame with the data
            channel_pairs: List of (x, y, z) channel tuples for each panel
            title: Overall plot title

        Returns:
            Plotly Figure with subplots
        """
        n_panels = len(channel_pairs)
        cols = min(3, n_panels)  # Max 3 columns
        rows = (n_panels + cols - 1) // cols  # Ceiling division

        subplot_titles = [f"{x}-{y}-{z}" for x, y, z in channel_pairs]

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "scene"}] * cols] * rows
        )

        for i, (x, y, z) in enumerate(channel_pairs):
            row = i // cols + 1
            col = i % cols + 1

            # Add scatter plot to subplot
            fig.add_trace(
                go.Scatter3d(
                    x=data[x],
                    y=data[y],
                    z=data[z],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=data[z],
                        colorscale='Viridis',
                        opacity=0.6,
                        showscale=i == 0  # Only show colorbar for first panel
                    ),
                    showlegend=False,
                    name=f"{x}-{y}-{z}"
                ),
                row=row, col=col
            )

            # Update subplot camera
            fig.update_scenes(
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                row=row, col=col
            )

        fig.update_layout(
            title=title,
            height=400 * rows,
            showlegend=False
        )

        return fig

    def create_gating_comparison_3d(
        self,
        before_data: pd.DataFrame,
        after_data: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        title: str = "Before vs After Gating (3D)"
    ) -> go.Figure:
        """Create a side-by-side 3D comparison of data before and after gating.

        Args:
            before_data: DataFrame before gating
            after_data: DataFrame after gating
            x_col, y_col, z_col: Channel columns for visualization
            title: Plot title

        Returns:
            Plotly Figure with comparison
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Before Gating", "After Gating"],
            specs=[[{"type": "scene"}, {"type": "scene"}]]
        )

        # Before gating
        fig.add_trace(
            go.Scatter3d(
                x=before_data[x_col],
                y=before_data[y_col],
                z=before_data[z_col],
                mode='markers',
                marker=dict(
                    size=2,
                    color='blue',
                    opacity=0.5
                ),
                name="Before",
                showlegend=False
            ),
            row=1, col=1
        )

        # After gating
        fig.add_trace(
            go.Scatter3d(
                x=after_data[x_col],
                y=after_data[y_col],
                z=after_data[z_col],
                mode='markers',
                marker=dict(
                    size=2,
                    color='red',
                    opacity=0.7
                ),
                name="After",
                showlegend=False
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title=title,
            height=500
        )

        # Update both scenes
        for col in [1, 2]:
            fig.update_scenes(
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                row=1, col=col
            )

        return fig


class HeatmapVisualizer:
    """Heatmap and contour plot visualization tools."""

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        """Initialize the heatmap visualizer.

        Args:
            data: DataFrame with flow cytometry data (optional)
        """
        self.data = data

    def create_population_heatmap(
        self,
        data: pd.DataFrame,
        x_channel: str,
        y_channel: str,
        bins: int = 50,
        density: bool = True,
        log_scale: bool = False,
        title: str = "Population Density Heatmap"
    ) -> go.Figure:
        """Create a 2D density heatmap for population analysis.

        Args:
            data: DataFrame with flow cytometry data
            x_channel: Channel for X-axis
            y_channel: Channel for Y-axis
            bins: Number of bins for histogram
            density: Whether to show density or count
            log_scale: Whether to use log scale for colors
            title: Plot title

        Returns:
            Plotly Figure object
        """
        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            data[x_channel],
            data[y_channel],
            bins=bins,
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
            hovertemplate=f"{x_channel}: %{{x}}<br>{y_channel}: %{{y}}<br>{'Density' if density else 'Count'}: %{{z}}<extra></extra>",
            colorbar=dict(title='Density' if density else 'Count')
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_channel,
            yaxis_title=y_channel,
            height=500
        )

        return fig

    def create_multi_channel_heatmap(
        self,
        data: pd.DataFrame,
        channels: list[str],
        title: str = "Multi-Channel Correlation Heatmap"
    ) -> go.Figure:
        """Create a correlation heatmap for multiple channels.

        Args:
            data: DataFrame with flow cytometry data
            channels: List of channel names to include
            title: Plot title

        Returns:
            Plotly Figure object
        """
        # Filter to numeric channels only
        numeric_channels = [col for col in channels if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]

        if len(numeric_channels) < 2:
            raise ValueError("Need at least 2 numeric channels for correlation heatmap")

        # Calculate correlation matrix
        corr_matrix = data[numeric_channels].corr()

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=numeric_channels,
            y=numeric_channels,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            hoverongaps=False,
            hovertemplate="Channel X: %{x}<br>Channel Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title=title,
            height=max(400, len(numeric_channels) * 30)
        )

        return fig


def create_interactive_gating_dashboard(
    results_dir: str | Path,
    sample_id: str | None = None,
    output_file: str | Path | None = None
) -> None:
    """Create an interactive gating dashboard for a specific sample.

    Args:
        results_dir: Path to cytoflow-qc results directory
        sample_id: Specific sample to visualize (optional)
        output_file: Path to save HTML file (optional)
    """
    results_path = Path(results_dir)

    # Load sample data
    sample_files = list((results_path / "gate" / "events").glob("*.parquet"))
    if not sample_files:
        print("No gated sample data found in results directory")
        return

    # Select sample
    if sample_id:
        sample_file = results_path / "gate" / "events" / f"{sample_id}.parquet"
        if not sample_file.exists():
            print(f"Sample {sample_id} not found")
            return
    else:
        sample_file = sample_files[0]
        sample_id = sample_file.stem

    # Load data
    df = pd.read_parquet(sample_file)

    # Create comprehensive dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "FSC-A vs SSC-A (2D)",
            "FSC-A vs CD3-A (2D)",
            "FSC-A vs SSC-A vs CD3-A (3D)",
            "Channel Correlations"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "scene", "colspan": 2}, None],
        ]
    )

    # 2D scatter plots
    fig.add_trace(
        go.Scatter(
            x=df["FSC-A"], y=df["SSC-A"],
            mode='markers',
            marker=dict(size=2, opacity=0.6, color=df["SSC-A"], colorscale="Viridis"),
            name="FSC-A vs SSC-A"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["FSC-A"], y=df["CD3-A"],
            mode='markers',
            marker=dict(size=2, opacity=0.6, color=df["CD3-A"], colorscale="Plasma"),
            name="FSC-A vs CD3-A"
        ),
        row=1, col=2
    )

    # 3D scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=df["FSC-A"], y=df["SSC-A"], z=df["CD3-A"],
            mode='markers',
            marker=dict(
                size=2,
                color=df["CD3-A"],
                colorscale="Viridis",
                opacity=0.6,
                showscale=True,
                colorbar=dict(title="CD3-A")
            ),
            name="3D View"
        ),
        row=2, col=1
    )

    # Channel correlation heatmap
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != "gated"]
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=numeric_cols,
                y=numeric_cols,
                colorscale="RdBu",
                zmin=-1, zmax=1,
                showscale=True,
                colorbar=dict(title="Correlation")
            ),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        title=f"Interactive Gating Dashboard: {sample_id}",
        height=800,
        showlegend=False
    )

    # Update subplot titles and axes
    fig.update_xaxes(title_text="FSC-A", row=1, col=1)
    fig.update_yaxes(title_text="SSC-A", row=1, col=1)
    fig.update_xaxes(title_text="FSC-A", row=1, col=2)
    fig.update_yaxes(title_text="CD3-A", row=1, col=2)
    fig.update_scenes(
        xaxis_title="FSC-A",
        yaxis_title="SSC-A",
        zaxis_title="CD3-A",
        row=2, col=1
    )

    # Save or display
    if output_file:
        fig.write_html(output_file)
        print(f"Interactive dashboard saved to: {output_file}")
    else:
        fig.show()


def create_publication_ready_figure(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str | None = None,
    output_path: str | Path | None = None,
    format: str = "png",
    dpi: int = 300,
    figsize: tuple[int, int] = (10, 8)
) -> None:
    """Create publication-ready figures with proper styling.

    Args:
        data: DataFrame with flow cytometry data
        x_col: X-axis channel
        y_col: Y-axis channel
        z_col: Z-axis channel (optional for 3D)
        output_path: Path to save figure
        format: Output format (png, pdf, svg, eps)
        dpi: Resolution for raster formats
        figsize: Figure size in inches
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set publication-ready style
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if z_col:
        # 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D

        ax = fig.add_subplot(111, projection='3d')

        # Create 3D scatter
        scatter = ax.scatter(
            data[x_col], data[y_col], data[z_col],
            c=data[z_col], cmap='viridis', alpha=0.6, s=1
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(z_col, rotation=270, labelpad=15)

        # Set labels
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)

        # Set equal aspect ratio
        max_range = max(
            data[x_col].max() - data[x_col].min(),
            data[y_col].max() - data[y_col].min(),
            data[z_col].max() - data[z_col].min()
        )
        ax.set_xlim(data[x_col].min(), data[x_col].min() + max_range)
        ax.set_ylim(data[y_col].min(), data[y_col].min() + max_range)
        ax.set_zlim(data[z_col].min(), data[z_col].min() + max_range)

    else:
        # 2D scatter plot with density
        # Create hexbin plot for density
        hb = ax.hexbin(
            data[x_col], data[y_col],
            gridsize=50, cmap='viridis', alpha=0.8
        )

        # Add colorbar
        cbar = plt.colorbar(hb, ax=ax, shrink=0.8)
        cbar.set_label('Density', rotation=270, labelpad=15)

        # Set labels
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        # Set equal aspect ratio
        ax.set_aspect('equal')

    # Add title
    title = f"{x_col} vs {y_col}"
    if z_col:
        title += f" vs {z_col}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Improve layout
    plt.tight_layout()

    # Save with high quality if output_path provided
    if output_path is not None:
        output_path = Path(output_path)
        if format.lower() == 'pdf':
            fig.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight')
        elif format.lower() == 'svg':
            fig.savefig(output_path, format='svg', dpi=dpi, bbox_inches='tight')
        elif format.lower() == 'eps':
            fig.savefig(output_path, format='eps', dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')

        print(f"Publication-ready figure saved to: {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m cytoflow_qc.viz_3d <data_file> <output_file> [format]")
        sys.exit(1)

    data_file = sys.argv[1]
    output_file = sys.argv[2]
    format_type = sys.argv[3] if len(sys.argv) > 3 else "png"

    # Load data
    df = pd.read_parquet(data_file)

    # Create publication-ready figure
    create_publication_ready_figure(
        df, "FSC-A", "SSC-A", "CD3-A",
        output_file, format_type, dpi=300, figsize=(12, 10)
    )
