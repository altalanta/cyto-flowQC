"""Typer-powered CLI for the CytoFlow-QC pipeline.

This module defines the command-line interface for `cytoflow-qc`, allowing users to
execute various stages of the flow cytometry quality control and analysis pipeline
from the terminal. It leverages the `typer` library for creating a user-friendly
and robust CLI.

The `app` instance is the main entry point for all commands.

Commands:
- `ingest`: Loads samplesheet and ingests raw FCS files, converting them to
  standardized Parquet format and extracting metadata.
  Arguments:
    - `samplesheet` (Path): Path to the samplesheet CSV file.
    - `out` (Path): Output directory for ingested data.
    - `--config`, `-c` (Path, optional): Path to an optional YAML configuration file.

- `compensate`: Applies compensation to raw event data using a spillover matrix.
  Arguments:
    - `indir` (Path): Input directory containing ingested (uncompensated) events.
    - `out` (Path): Output directory for compensated events.
    - `--spill` (Path, optional): Path to a custom spillover matrix CSV file.

- `qc`: Applies quality control flags to events and generates a QC summary.
  Arguments:
    - `indir` (Path): Input directory containing compensated events.
    - `out` (Path): Output directory for QC-annotated events and summary.
    - `--config`, `-c` (Path, optional): Path to a YAML configuration file containing QC parameters.

- `gate`: Applies gating strategies to identify cell populations. Supports custom
  gating strategies via the plugin system.
  Arguments:
    - `indir` (Path): Input directory containing QC-annotated events.
    - `out` (Path): Output directory for gated events and gating parameters.
    - `--strategy` (str, optional): Name of the gating strategy to use (default: "default").
    - `--config`, `-c` (Path, optional): Path to a YAML configuration file containing gating parameters.

- `drift`: Performs batch drift analysis to identify shifts in populations over time.
  Arguments:
    - `indir` (Path): Input directory containing gated events.
    - `out` (Path): Output directory for drift analysis results and plots.
    - `--by` (str, optional): Metadata column to use for batch grouping (default: "batch").
    - `--config`, `-c` (Path, optional): Path to a YAML configuration file.

- `stats`: Calculates effect sizes and other statistics on gated populations.
  Arguments:
    - `indir` (Path): Input directory containing gated events.
    - `out` (Path): Output directory for statistical analysis results.
    - `--groups` (str, optional): Metadata column to use for grouping samples (default: "condition").
    - `--values` (str, optional): Comma-separated list of marker columns for analysis.
    - `--config`, `-c` (Path, optional): Path to a YAML configuration file.

- `report`: Generates an HTML report summarizing the entire pipeline's results.
  Arguments:
    - `indir` (Path): Input directory containing all pipeline stage results.
    - `out` (Path): Output HTML file path for the report.
    - `--template` (Path, optional): Path to a custom Jinja2 HTML report template.

- `dashboard`: Launches an interactive Streamlit visualization dashboard for exploring results.
  Arguments:
    - `indir` (Path): Results directory from a `cytoflow-qc run`.
    - `--sample`, `-s` (str, optional): Specific sample to visualize.
    - `--port`, `-p` (int, optional): Port for the Streamlit server (default: 8501).

- `viz3d`: Creates an interactive 3D gating visualization (HTML output).
  Arguments:
    - `indir` (Path): Results directory from a `cytoflow-qc run`.
    - `--sample`, `-s` (str, optional): Specific sample to visualize.
    - `--output`, `-o` (Path, optional): Output HTML file.
    - `--x` (str, optional): X-axis channel (default: "FSC-A").
    - `--y` (str, optional): Y-axis channel (default: "SSC-A").
    - `--z` (str, optional): Z-axis channel (default: "CD3-A").

- `export`: Exports publication-ready 2D or 3D figures from processed data.
  Arguments:
    - `data` (Path): Data file (CSV or Parquet).
    - `output` (Path): Output file path.
    - `--x` (str, optional): X-axis channel (default: "FSC-A").
    - `--y` (str, optional): Y-axis channel (default: "SSC-A").
    - `--z` (str, optional): Z-axis channel (optional for 3D).
    - `--format` (str, optional): Output format (png, pdf, svg, eps) (default: "png").
    - `--dpi` (int, optional): Resolution for raster formats (default: 300).
    - `--width` (int, optional): Figure width in inches (default: 10).
    - `--height` (int, optional): Figure height in inches (default: 8).

- `export-dashboard`: Exports the interactive dashboard as a standalone HTML file.
  Arguments:
    - `indir` (Path): Results directory from a `cytoflow-qc run`.
    - `output` (Path): Output HTML file path.
    - `--animations` (bool, optional): Include animation features (default: False).

- `export-3d`: Exports a 3D gating visualization as a standalone HTML file.
  Arguments:
    - `indir` (Path): Results directory from a `cytoflow-qc run`.
    - `--sample`, `-s` (str): Sample to visualize.
    - `output` (Path): Output HTML file path.
    - `--x` (str, optional): X-axis channel (default: "FSC-A").
    - `--y` (str, optional): Y-axis channel (default: "SSC-A").
    - `--z` (str, optional): Z-axis channel (default: "CD3-A").

- `plugins`: Manages and interacts with `cytoflow-qc` plugins.
  Arguments:
    - `action` (str): Plugin action ("list", "info", "load").
    - `--type`, `-t` (str, optional): Plugin type filter (e.g., "gating_strategy").
    - `--name`, `-n` (str, optional): Specific plugin name.
    - `--config`, `-c` (str, optional): Path to plugin configuration file.

- `cloud`: Manages cloud deployment and scaling for `cytoflow-qc` workloads.
  Arguments:
    - `provider` (str): Cloud provider ("kubernetes", "serverless").
    - `action` (str): Cloud action ("deploy", "scale", "status").
    - `--config`, `-c` (str, optional): Path to cloud configuration file.

- `realtime`: Manages real-time data processing and monitoring.
  Arguments:
    - `action` (str): Real-time action ("start", "monitor").
    - `--ws-url` (str, optional): WebSocket URL for data source.
    - `--config`, `-c` (str, optional): Path to real-time configuration.

- `anonymize`: Anonymizes sensitive data in specified columns of dataframes.
  Arguments:
    - `indir` (Path): Input directory containing dataframes.
    - `outdir` (Path): Output directory for anonymized dataframes.
    - `--columns`, `-c` (str): Comma-separated list of columns to anonymize.
    - `--identifier`, `-i` (str, optional): Column to use as a stable identifier.

- `encrypt`: Encrypts a file using a symmetric encryption key.
  Arguments:
    - `infile` (Path): Input file to encrypt.
    - `outfile` (Path): Output file for encrypted data.
    - `--key-path`, `-k` (Path, optional): Path to encryption key file.

- `decrypt`: Decrypts an encrypted file.
  Arguments:
    - `infile` (Path): Input encrypted file.
    - `outfile` (Path): Output file for decrypted data.
    - `--key-path`, `-k` (Path, optional): Path to encryption key file.

- `rbac`: Checks role-based access control permissions.
  Arguments:
    - `--roles`, `-r` (str): Comma-separated list of user roles.
    - `action` (str): Action to check (e.g., "read", "write").
    - `resource` (str): Resource to access (e.g., "data_raw", "reports").
    - `--policy-file`, `-p` (Path, optional): Path to custom RBAC policy JSON file.

- `data-source`: Manages data source connectors and ingests data.
  Arguments:
    - `action` (str): Action to perform ("configure", "list", "ingest").
    - `--uri`, `-u` (str, optional): Base URI for the data source (default: "file:///").
    - `--config`, `-c` (Path, optional): Path to data source configuration YAML file.
    - `--pattern`, `-p` (str, optional): Glob pattern for listing/ingesting files (default: "*.fcs").
    - `--output-dir`, `-o` (Path, optional): Output directory for ingested files.

- `run`: Executes the full `cytoflow-qc` pipeline from ingestion to report generation.
  Arguments:
    - `--samplesheet` (Path): Path to the samplesheet CSV file.
    - `--config` (Path): Path to the main YAML configuration file.
    - `--out` (Path): Output root directory for all generated artifacts.
    - `--spill` (Path, optional): Path to override the spillover matrix.
    - `--batch` (str, optional): Metadata column for batch grouping (default: "batch").
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Any

import pandas as pd
import typer
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import matplotlib.pyplot as plt

from cytoflow_qc import __version__
from cytoflow_qc.config import AppConfig, load_and_validate_config
from cytoflow_qc.exceptions import CytoflowQCError, ValidationError
from cytoflow_qc.log_config import setup_logging
from cytoflow_qc.pipeline import (
    CompensationStage, 
    GatingStage, 
    QCStage,
    IngestionStage,
    DriftStage,
    StatsStage,
    ReportStage,
)
from cytoflow_qc.utils import (
    _read_json,
    _write_json,
    ensure_dir,
    list_stage_events,
    load_dataframe,
    read_manifest,
    save_dataframe,
    timestamp,
    write_manifest,
)
from cytoflow_qc.io import load_samplesheet, standardize_channels, get_fcs_metadata, read_fcs
from cytoflow_qc.compensate import get_spillover, apply_compensation
from cytoflow_qc.qc import add_qc_flags, qc_summary
from cytoflow_qc.gate import auto_gate
from cytoflow_qc.drift import extract_sample_features, compute_batch_drift
from cytoflow_qc.stats import effect_sizes
from cytoflow_qc.viz import (
    plot_batch_drift_pca,
    plot_batch_drift_umap,
    plot_effect_sizes,
    plot_gating_scatter,
    plot_qc_summary,
)
from cytoflow_qc.interactive_viz import launch_interactive_dashboard
from cytoflow_qc.viz_3d import create_interactive_gating_dashboard, create_publication_ready_figure
from cytoflow_qc.plugins import get_plugin_registry, load_plugin
from cytoflow_qc.cloud import KubernetesDeployment, CloudStorage
from cytoflow_qc.realtime import WebSocketProcessor, RealTimeMonitor
from cytoflow_qc.security import DataAnonymizer, DataEncryptor, RBACManager, SecurityError
from cytoflow_qc.experiment_design import ExperimentManager, CohortManager
from cytoflow_qc.data_connectors import get_connector, DataSourceError
from cytoflow_qc.configure import generate_config_interactive
from cytoflow_qc.scaffold import create_plugin_scaffold
from cytoflow_qc.validation import validate_inputs
from cytoflow_qc.provenance import generate_provenance_report

app = typer.Typer(add_completion=False, help="Flow cytometry QC and gating pipeline")
logger = logging.getLogger("cytoflow_qc")


@app.callback()
def _version(ctx: typer.Context, version: bool = typer.Option(False, "--version", help="Show version and exit")) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def configure():
    """Launch an interactive tool to generate a config.yaml file."""
    try:
        generate_config_interactive()
    except Exception as e:
        logger.error(f"Failed to generate configuration: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def validate(
    samplesheet: Path = typer.Option(..., "--samplesheet", exists=True, help="Path to the samplesheet CSV file."),
    config: Path = typer.Option(..., "--config", exists=True, help="Path to the main YAML configuration file."),
):
    """Validate input files and configuration without running the full pipeline."""
    setup_logging(Path.cwd())
    try:
        validate_inputs(samplesheet, config)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def ingest(
    samplesheet: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Argument(...),
    config: dict[str, object] | None = typer.Option(None, "--config", "-c", help="Optional YAML config"),
) -> None:
    cfg = load_and_validate_config(config) if config else AppConfig()
    stage_ingest(samplesheet, out, cfg)
    logger.info(f"Ingested samples -> {out}")


@app.command()
def compensate(
    indir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
    spill: Path | None = typer.Option(None, "--spill", help="Override spillover CSV"),
    workers: int = typer.Option(os.cpu_count(), "--workers", "-w", help="Number of worker processes to use."),
) -> None:
    stage_compensate(indir, out, spill, workers)
    logger.info(f"Compensated events -> {out}")


@app.command()
def qc(
    indir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workers: int = typer.Option(os.cpu_count(), "--workers", "-w", help="Number of worker processes to use."),
) -> None:
    cfg = load_and_validate_config(config) if config else AppConfig()
    stage_qc(indir, out, cfg.qc, workers)
    logger.info(f"QC annotations -> {out}")


@app.command()
def gate(
    indir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
    strategy: str = typer.Option("default", "--strategy"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workers: int = typer.Option(os.cpu_count(), "--workers", "-w", help="Number of worker processes to use."),
) -> None:
    cfg = load_and_validate_config(config) if config else AppConfig()
    stage_gate(indir, out, strategy, cfg, workers)
    logger.info(f"Gated populations -> {out}")


@app.command()
def drift(
    indir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
    by: str = typer.Option("batch", "--by", help="Metadata column for batch grouping"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    cfg = load_and_validate_config(config) if config else AppConfig()
    stage_drift(indir, out, by, cfg)
    logger.info(f"Batch drift analysis -> {out}")


@app.command()
def stats(
    indir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
    group_col: str = typer.Option("condition", "--groups"),
    values: str | None = typer.Option(None, "--values", help="Comma-separated marker columns"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    cfg = load_and_validate_config(config) if config else AppConfig()
    marker_columns = _resolve_marker_columns(values, cfg)
    stage_stats(indir, out, group_col, marker_columns)
    logger.info(f"Effect-size statistics -> {out}")


@app.command()
def report(
    indir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
) -> None:
    ReportStage(indir, out).run()
    logger.info(f"Report written to {out}")


@app.command()
def launch():
    """Launch the interactive pipeline launcher GUI."""
    try:
        import streamlit.web.cli as stcli
        import sys

        launcher_path = Path(__file__).parent / "launcher.py"
        sys.argv = ["streamlit", "run", str(launcher_path)]
        stcli.main()
    except ImportError:
        logger.error("Streamlit is not installed. Please run 'pip install streamlit'.")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Failed to launch GUI: {e}")
        raise typer.Exit(1)


@app.command()
def dashboard(
    indir: Path = typer.Argument(..., exists=True, help="Results directory from cytoflow-qc run"),
    sample: str | None = typer.Option(None, "--sample", "-s", help="Specific sample to visualize"),
    port: int = typer.Option(8501, "--port", "-p", help="Port for Streamlit server"),
) -> None:
    """Launch interactive visualization dashboard."""
    logger.info(f"🚀 Launching interactive dashboard for results in: {indir}")

    # Import streamlit here to avoid issues if not installed
    try:
        import streamlit.web.cli as stcli

        # Set up the script to run
        script_path = Path(__file__).parent / "interactive_viz.py"

        # Launch streamlit
        import sys
        sys.argv = [
            "streamlit", "run", str(script_path), str(indir),
            "--server.port", str(port),
            "--server.address", "0.0.0.0"
        ]

        stcli.main()

    except ImportError:
        logger.error("❌ Streamlit not installed. Install with: pip install streamlit")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"❌ Error launching dashboard: {e}")
        raise typer.Exit(1)


@app.command()
def viz3d(
    indir: Path = typer.Argument(..., exists=True, help="Results directory from cytoflow-qc run"),
    sample: str | None = typer.Option(None, "--sample", "-s", help="Specific sample to visualize"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output HTML file"),
    x: str = typer.Option("FSC-A", "--x", help="X-axis channel"),
    y: str = typer.Option("SSC-A", "--y", help="Y-axis channel"),
    z: str = typer.Option("CD3-A", "--z", help="Z-axis channel"),
) -> None:
    """Create interactive 3D gating visualization."""
    try:
        if output:
            create_interactive_gating_dashboard(indir, sample, output)
        else:
            # Just show the visualization (this would need plotly display setup)
            logger.info("💡 Tip: Use --output to save HTML file, or run 'cytoflow-qc dashboard' for full interface")
            logger.info(f"📁 Results directory: {indir}")
    except Exception as e:
        logger.error(f"❌ Error creating 3D visualization: {e}")
        raise typer.Exit(1)


@app.command()
def export(
    data: Path = typer.Argument(..., exists=True, help="Data file (CSV or Parquet)"),
    output: Path = typer.Argument(..., help="Output file path"),
    x: str = typer.Option("FSC-A", "--x", help="X-axis channel"),
    y: str = typer.Option("SSC-A", "--y", help="Y-axis channel"),
    z: str | None = typer.Option(None, "--z", help="Z-axis channel (optional for 3D)"),
    format: str = typer.Option("png", "--format", help="Output format (png, pdf, svg, eps)"),
    dpi: int = typer.Option(300, "--dpi", help="Resolution for raster formats"),
    width: int = typer.Option(10, "--width", help="Figure width in inches"),
    height: int = typer.Option(8, "--height", help="Figure height in inches"),
) -> None:
    """Export publication-ready figures."""
    try:
        import pandas as pd

        # Load data
        if data.suffix.lower() == '.parquet':
            df = pd.read_parquet(data)
        else:
            df = pd.read_csv(data)

        # Create publication-ready figure
        create_publication_ready_figure(
            df, x, y, z,
            output, format, dpi, (width, height)
        )

        logger.info(f"✅ Publication-ready figure exported to: {output}")

    except Exception as e:
        logger.error(f"❌ Error exporting figure: {e}")
        raise typer.Exit(1)


@app.command()
def export_dashboard(
    indir: Path = typer.Argument(..., exists=True, help="Results directory from cytoflow-qc run"),
    output: Path = typer.Argument(..., help="Output HTML file path"),
    animations: bool = typer.Option(False, "--animations", help="Include animation features"),
) -> None:
    """Export interactive dashboard as HTML file."""
    try:
        from cytoflow_qc.interactive_viz import InteractiveVisualizer

        visualizer = InteractiveVisualizer(indir)
        visualizer.export_interactive_dashboard(output, animations)

        logger.info(f"✅ Interactive dashboard exported to: {output}")

    except Exception as e:
        logger.error(f"❌ Error exporting dashboard: {e}")
        raise typer.Exit(1)


@app.command()
def export_3d(
    indir: Path = typer.Argument(..., exists=True, help="Results directory from cytoflow-qc run"),
    sample: str = typer.Option(..., "--sample", "-s", help="Sample to visualize"),
    output: Path = typer.Argument(..., help="Output HTML file path"),
    x: str = typer.Option("FSC-A", "--x", help="X-axis channel"),
    y: str = typer.Option("SSC-A", "--y", help="Y-axis channel"),
    z: str = typer.Option("CD3-A", "--z", help="Z-axis channel"),
) -> None:
    """Export 3D gating visualization as HTML file."""
    try:
        from cytoflow_qc.viz_3d import create_interactive_gating_dashboard

        create_interactive_gating_dashboard(indir, sample, output)

        logger.info(f"✅ 3D visualization exported to: {output}")

    except Exception as e:
        logger.error(f"❌ Error exporting 3D visualization: {e}")
        raise typer.Exit(1)


@app.command()
def plugins(
    action: str = typer.Argument(..., help="Plugin action (list, info, create)"),
    plugin_type: str | None = typer.Option(None, "--type", "-t", help="Plugin type filter"),
    plugin_name: str | None = typer.Option(None, "--name", "-n", help="Specific plugin name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Plugin configuration"),
) -> None:
    """Manage and interact with cytoflow-qc plugins."""
    if action == "create":
        create_plugin_scaffold()
        return

    registry = get_plugin_registry()

    if action == "list":
        # List available plugins
        available = registry.get_available_plugins(plugin_type)
        logger.info("Available plugins:")

        for ptype, plugins in available.items():
            if plugins:
                logger.info(f"  {ptype}:")
                for plugin in plugins:
                    logger.info(f"    • {plugin}")
    elif action == "info":
        # Get info about specific plugin
        if not plugin_name:
            logger.error("Error: --name required for info action")
            raise typer.Exit(1)

        try:
            info = registry.get_plugin_info(plugin_type or "gating_strategy", plugin_name)
            logger.info(f"Plugin: {info['name']}")
            logger.info(f"Version: {info['version']}")
            logger.info(f"Description: {info['description']}")
            logger.info(f"Author: {info['author']}")
            logger.info(f"Type: {info['plugin_type']}")
            logger.info("Default config:")
            import json
            logger.info(json.dumps(info['default_config'], indent=2))
        except Exception as e:
            logger.error(f"Error getting plugin info: {e}")
    elif action == "load":
        # Load and test plugin
        if not plugin_name:
            logger.error("Error: --name required for load action")
            raise typer.Exit(1)

        try:
            plugin_config: dict[str, Any] = {}
            if config:
                import yaml
                with open(config, 'r') as f:
                    plugin_config = yaml.safe_load(f) or {}

            plugin = load_plugin(plugin_type or "gating_strategy", plugin_name, plugin_config)
            logger.info(f"✅ Loaded plugin: {plugin.name} v{plugin.version}")
            logger.info(f"Description: {plugin.description}")
        except Exception as e:
            logger.error(f"❌ Error loading plugin: {e}")
    else:
        logger.error(f"Unknown action: {action}")
        logger.info("Available actions: list, info, load")


@app.command()
def cloud(
    provider: str = typer.Argument(..., help="Cloud provider (aws, gcp, azure)"),
    action: str = typer.Argument(..., help="Cloud action (deploy, scale, status)"),
    config: str | None = typer.Option(None, "--config", "-c", help="Cloud configuration file"),
) -> None:
    """Manage cloud deployment and scaling."""
    if provider.lower() == "kubernetes":
        if action == "deploy":
            k8s = KubernetesDeployment()
            if config:
                import yaml
                with open(config, 'r') as f:
                    k8s_config = yaml.safe_load(f)
                # Apply configuration to k8s deployment
                logger.info("Kubernetes deployment configured")
            else:
                logger.info("Use --config to specify Kubernetes configuration")
        elif action == "status":
            k8s = KubernetesDeployment()
            status = k8s.get_deployment_status()
            logger.info("Kubernetes deployment status:")
            import json
            logger.info(json.dumps(status, indent=2))
        else:
            logger.error(f"Unknown Kubernetes action: {action}")
    elif provider.lower() == "serverless":
        if action == "deploy":
            logger.warning("Serverless deployment not implemented yet")
        else:
            logger.error(f"Unknown serverless action: {action}")
    else:
        logger.error(f"Unsupported provider: {provider}")


@app.command()
def realtime(
    action: str = typer.Argument(..., help="Real-time action (start, monitor)"),
    ws_url: str | None = typer.Option(None, "--ws-url", help="WebSocket URL for data source"),
    config: str | None = typer.Option(None, "--config", "-c", help="Real-time configuration"),
) -> None:
    """Manage real-time processing."""
    if action == "start":
        if not ws_url:
            logger.error("Error: --ws-url required for start action")
            raise typer.Exit(1)

        logger.info(f"Starting real-time processing from: {ws_url}")
        # This would start the real-time processing
        logger.info("Real-time processing started (placeholder)")
    elif action == "monitor":
        # Start monitoring dashboard
        monitor = RealTimeMonitor()
        logger.info("Starting real-time monitoring dashboard...")
        # This would start the monitoring dashboard
        logger.info("Monitoring dashboard started (placeholder)")
    else:
        logger.error(f"Unknown real-time action: {action}")


@app.command()
def anonymize(
    indir: Path = typer.Argument(..., exists=True, help="Input directory containing dataframes"),
    outdir: Path = typer.Argument(..., help="Output directory for anonymized dataframes"),
    columns: str = typer.Option(..., "--columns", "-c", help="Comma-separated list of columns to anonymize"),
    identifier_col: str | None = typer.Option(None, "--identifier", "-i", help="Column to use as a stable identifier for consistent anonymization"),
) -> None:
    """Anonymize sensitive data in specified columns of dataframes."""
    ensure_dir(outdir)
    anonymizer = DataAnonymizer()
    cols_to_anon = [c.strip() for c in columns.split(",") if c.strip()]

    logger.info(f"Anonymizing data in {indir} and saving to {outdir}...")

    try:
        for sample_id, events_file in list_stage_events(indir).items():
            df = load_dataframe(indir / events_file)
            anonymized_df = anonymizer.anonymize_dataframe(df, cols_to_anon, identifier_col)
            save_dataframe(anonymized_df, outdir / Path(events_file).name)
            logger.info(f"✅ Anonymized {sample_id}")
        logger.info("🎉 All specified dataframes anonymized successfully!")
    except Exception as e:
        logger.error(f"❌ Error during anonymization: {e}")
        raise typer.Exit(code=1)

@app.command()
def encrypt(
    infile: Path = typer.Argument(..., exists=True, readable=True, help="Input file to encrypt"),
    outfile: Path = typer.Argument(..., help="Output file for encrypted data"),
    key_path: Path | None = typer.Option(None, "--key-path", "-k", help="Path to encryption key file"),
) -> None:
    """Encrypt a file using a symmetric encryption key."""
    try:
        encryptor = DataEncryptor(key_path=key_path)
        encryptor.encrypt_file(infile, outfile)
        logger.info(f"✅ File '{infile}' encrypted to '{outfile}' successfully!")
    except SecurityError as e:
        logger.error(f"❌ Encryption Error: {e}")
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        logger.error(f"❌ Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during encryption: {e}")
        raise typer.Exit(code=1)

@app.command()
def decrypt(
    infile: Path = typer.Argument(..., exists=True, readable=True, help="Input encrypted file"),
    outfile: Path = typer.Argument(..., help="Output file for decrypted data"),
    key_path: Path | None = typer.Option(None, "--key-path", "-k", help="Path to encryption key file"),
) -> None:
    """Decrypt an encrypted file."""
    try:
        encryptor = DataEncryptor(key_path=key_path)
        encryptor.decrypt_file(infile, outfile)
        logger.info(f"✅ File '{infile}' decrypted to '{outfile}' successfully!")
    except SecurityError as e:
        logger.error(f"❌ Decryption Error: {e}")
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        logger.error(f"❌ Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during decryption: {e}")
        raise typer.Exit(code=1)

@app.command()
def rbac(
    roles: str = typer.Option(..., "--roles", "-r", help="Comma-separated list of user roles"),
    action: str = typer.Argument(..., help="Action to check (e.g., 'read', 'write')"),
    resource: str = typer.Argument(..., help="Resource to access (e.g., 'data_raw', 'reports')"),
    policy_file: Path | None = typer.Option(None, "--policy-file", "-p", help="Path to custom RBAC policy JSON file"),
) -> None:
    """Check role-based access control permissions."""
    rbac_manager = RBACManager(policy_file=policy_file)
    user_roles = [r.strip() for r in roles.split(",") if r.strip()]

    logger.info(f"Checking if roles {user_roles} can '{action}' resource '{resource}'...")
    if rbac_manager.check_permission(user_roles, action, resource):
        logger.info(f"✅ Permission granted for roles {user_roles} to {action} {resource}.")
    else:
        logger.error(f"❌ Permission denied for roles {user_roles} to {action} {resource}.")
        raise typer.Exit(code=1)

@app.command(name="data-source")
def data_source_cmd(
    action: str = typer.Argument(..., help="Action to perform (configure, list, ingest)"),
    uri: str = typer.Option("file:///", "--uri", "-u", help="Base URI for the data source"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to data source configuration YAML file"),
    pattern: str = typer.Option("*.fcs", "--pattern", "-p", help="Glob pattern for listing/ingesting files"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="Output directory for ingested files"),
) -> None:
    """Manage data source connectors and ingest data."""
    connector_config: dict[str, Any] = {}
    if config:
        import yaml
        with open(config, 'r') as f:
            connector_config = yaml.safe_load(f)

    try:
        connector = get_connector(uri, connector_config)

        if action == "list":
            logger.info(f"Listing files in '{uri}' with pattern '{pattern}':")
            found_files = list(connector.list_files(uri, pattern))
            if found_files:
                for f in found_files:
                    logger.info(f"  - {f}")
            else:
                logger.info("No files found.")
        elif action == "configure":
            logger.info(f"Configured data source for URI: {uri}")
            if connector_config:
                logger.info(f"  with configuration: {json.dumps(connector_config, indent=2)}")
            else:
                logger.info("  (using default configuration)")
        elif action == "ingest":
            if not output_dir:
                logger.error("Error: --output-dir is required for ingest action.")
                raise typer.Exit(1)
            ensure_dir(output_dir)

            logger.info(f"Ingesting files from '{uri}' (pattern: '{pattern}') to '{output_dir}'...")
            ingested_count = 0
            for remote_file_uri in connector.list_files(uri, pattern):
                local_file_path = output_dir / Path(remote_file_uri).name
                try:
                    file_content = connector.read_file(remote_file_uri)
                    local_file_path.write_bytes(file_content)
                    logger.info(f"  ✅ Ingested {remote_file_uri} to {local_file_path}")
                    ingested_count += 1
                except Exception as e:
                    logger.error(f"  ❌ Failed to ingest {remote_file_uri}: {e}")
            logger.info(f"🎉 Ingestion complete: {ingested_count} files successfully ingested.")
        else:
            logger.error(f"Unknown action: {action}. Available actions: configure, list, ingest")
            raise typer.Exit(1)
    except (ValueError, ImportError, DataSourceError) as e:
        logger.error(f"❌ Data Source Error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred with data source: {e}")
        raise typer.Exit(1)


@app.command()
def run(
    samplesheet: Path = typer.Option(..., "--samplesheet", exists=True),
    config: Path = typer.Option(..., "--config", exists=True),
    out: Path = typer.Option(..., "--out"),
    spill: Path | None = typer.Option(None, "--spill"),
    batch: str = typer.Option(..., "--batch"),
    workers: int = typer.Option(
        os.cpu_count(),
        "--workers",
        "-w",
        help="Number of worker processes to use."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs and exit without running the pipeline."),
) -> None:
    setup_logging(out)
    
    # Perform validation and obtain the validated config
    try:
        cfg = validate_inputs(samplesheet, config)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}", exc_info=True)
        raise typer.Exit(1)

    if dry_run:
        logger.info("Dry run successful. Exiting without running the pipeline.")
        return

    try:
        root = ensure_dir(out)

        # Generate and save provenance information
        logger.info("Generating provenance report...")
        generate_provenance_report(cfg, samplesheet, root)

        # Define directories
        ingest_dir = root / "ingest"
        compensate_dir = root / "compensate"
        qc_dir = root / "qc"
        gate_dir = root / "gate"
        drift_dir = root / "drift"
        stats_dir = root / "stats"

        # Execute pipeline
        ingestion_result = IngestionStage(ingest_dir, samplesheet, cfg).run()
        compensation_result = CompensationStage(compensate_dir, workers, spill).run(ingestion_result)
        qc_result = QCStage(qc_dir, workers, cfg.qc).run(compensation_result)
        gating_result = GatingStage(gate_dir, workers, "default", cfg).run(qc_result)
        drift_result = DriftStage(drift_dir, batch, cfg).run(gating_result)

        markers = cfg.channels.markers
        stats_result = StatsStage(stats_dir, "condition", markers).run(gating_result)

        # Build final report
        ReportStage(root, root / "report.html").run(stats_result)

    except CytoflowQCError as e:
        logger.error(f"A pipeline error occurred: {e}", exc_info=True)
        raise typer.Exit(code=1)

# ---------------------------------------------------------------------------
# Stage implementations (shared by commands and run())


def stage_ingest(samplesheet: Path, out_dir: Path, config: AppConfig) -> None:
    ensure_dir(out_dir)
    events_dir = ensure_dir(out_dir / "events")
    meta_dir = ensure_dir(out_dir / "metadata")

    sheet = load_samplesheet(str(samplesheet))
    channel_map = config.channels.model_dump()

    records = []
    for row in sheet.to_dict(orient="records"):
        if row.get("missing_file"):
            logger.warning(f"Skipping missing file {row['file_path']}")
            continue
        events, metadata = read_fcs(row["file_path"])
        if channel_map:
            events = standardize_channels(events, metadata, channel_map)
        sample_id = row["sample_id"]
        save_dataframe(events, events_dir / f"{sample_id}.parquet")
        _write_json(meta_dir / f"{sample_id}.json", metadata)
        record = dict(row)
        record.pop("missing_file", None)
        record["events_file"] = f"events/{sample_id}.parquet"
        record["metadata_file"] = f"metadata/{sample_id}.json"
        records.append(record)

    manifest = pd.DataFrame(records)
    manifest["stage"] = "ingest"
    manifest["timestamp"] = timestamp()
    write_manifest(manifest, out_dir / "manifest.csv")


def stage_compensate(indir: Path, out_dir: Path, spill: Path | None, workers: int) -> None:
    ensure_dir(out_dir)
    events_dir = ensure_dir(out_dir / "events")
    meta_dir = ensure_dir(out_dir / "metadata")
    manifest = read_manifest(indir / "manifest.csv")

    compensated_records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_compensate_sample, record, indir, events_dir, meta_dir, spill): record
            for record in manifest.to_dict(orient="records")
        }
        for future in as_completed(futures):
            compensated_records.append(future.result())

    out_manifest = pd.DataFrame(compensated_records)
    out_manifest["stage"] = "compensate"
    out_manifest["timestamp"] = timestamp()
    write_manifest(out_manifest, out_dir / "manifest.csv")


def stage_qc(indir: Path, out_dir: Path, qc_config: "QCConfig", workers: int) -> None:
    ensure_dir(out_dir)
    events_dir = ensure_dir(out_dir / "events")
    meta_dir = ensure_dir(out_dir / "metadata")
    manifest = read_manifest(indir / "manifest.csv")

    sample_tables: dict[str, pd.DataFrame] = {}
    updated_records = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_qc_sample, record, indir, events_dir, meta_dir, qc_config): record
            for record in manifest.to_dict(orient="records")
        }
        for future in as_completed(futures):
            updated_record, qc_df = future.result()
            updated_records.append(updated_record)
            sample_tables[updated_record["sample_id"]] = qc_df

    summary = qc_summary(sample_tables)
    summary.to_csv(out_dir / "summary.csv", index=False)

    out_manifest = pd.DataFrame(updated_records)
    out_manifest["stage"] = "qc"
    out_manifest["timestamp"] = timestamp()
    write_manifest(out_manifest, out_dir / "manifest.csv")

    plot_qc_summary(summary, str(out_dir / "figures" / "qc_pass.png"))


def stage_gate(indir: Path, out_dir: Path, strategy: str, config: AppConfig, workers: int) -> None:
    ensure_dir(out_dir)
    events_dir = ensure_dir(out_dir / "events")
    params_dir = ensure_dir(out_dir / "params")
    figures_dir = ensure_dir(out_dir / "figures")
    manifest = read_manifest(indir / "manifest.csv")

    gate_config = config.gating.model_dump()
    gate_config["channels"] = config.channels.model_dump()
    channels = gate_config.get("channels", {})

    summary_rows = []
    updated_records = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _gate_sample,
                record,
                indir,
                events_dir,
                params_dir,
                figures_dir,
                strategy,
                gate_config,
                channels
            ): record
            for record in manifest.to_dict(orient="records")
        }
        for future in as_completed(futures):
            updated_record, summary_row = future.result()
            updated_records.append(updated_record)
            summary_rows.append(summary_row)

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    manifest_out = pd.DataFrame(updated_records)
    manifest_out["stage"] = "gate"
    manifest_out["timestamp"] = timestamp()
    write_manifest(manifest_out, out_dir / "manifest.csv")


def stage_drift(indir: Path, out_dir: Path, batch_col: str, config: AppConfig) -> None:
    ensure_dir(out_dir)
    figures_dir = ensure_dir(out_dir / "figures")
    manifest = read_manifest(indir / "manifest.csv")
    sample_events = {sid: load_dataframe(indir / path) for sid, path in list_stage_events(indir).items()}
    meta_cols = ["sample_id", batch_col]
    if "condition" in manifest.columns:
        meta_cols.append("condition")
    metadata = manifest[meta_cols].drop_duplicates()
    marker_channels = config.channels.markers
    if isinstance(marker_channels, list):
        markers = marker_channels
    else:
        markers = None
    features = extract_sample_features(sample_events, metadata, marker_channels=markers)
    features.to_csv(out_dir / "features.csv", index=False)

    drift_res = compute_batch_drift(features, by=batch_col)
    drift_res["tests"].to_csv(out_dir / "tests.csv", index=False)
    drift_res["pca"].to_csv(out_dir / "pca.csv", index=False)
    if drift_res.get("umap") is not None:
        drift_res["umap"].to_csv(out_dir / "umap.csv", index=False)

    fig1 = plot_batch_drift_pca(drift_res["pca"], str(figures_dir / "pca.png"), batch_col)
    fig2 = plot_batch_drift_umap(drift_res.get("umap"), str(figures_dir / "umap.png"), batch_col)
    plt.close(fig1)
    plt.close(fig2)


def stage_stats(indir: Path, out_dir: Path, group_col: str, value_cols: Iterable[str]) -> None:
    ensure_dir(out_dir)
    manifest = read_manifest(indir / "manifest.csv")
    records = []
    columns: list[str] | None = None
    if value_cols is None:
        value_cols = []
    for record in manifest.to_dict(orient="records"):
        df = load_dataframe(indir / record["events_file"])
        if columns is None:
            columns = [col for col in value_cols if col in df.columns]
        summary = df[columns].mean().to_dict() if columns else {}
        summary[group_col] = record.get(group_col)
        summary["sample_id"] = record["sample_id"]
        records.append(summary)

    aggregated = pd.DataFrame(records)
    aggregated.to_csv(out_dir / "per_sample_summary.csv", index=False)
    if aggregated.empty or not columns:
        effects = pd.DataFrame()
    else:
        effects = effect_sizes(aggregated, group_col, columns)
    effects.to_csv(out_dir / "effect_sizes.csv", index=False)
    fig = plot_effect_sizes(effects, str(out_dir / "figures" / "effect_sizes.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers


def _compensate_sample(
    record: dict, indir: Path, events_dir: Path, meta_dir: Path, spill: Path | None
) -> dict:
    """Compensate a single sample's events."""
    sample_id = record["sample_id"]
    events = load_dataframe(indir / record["events_file"])
    metadata = _read_json(indir / record["metadata_file"])
    
    matrix, channels = get_spillover(metadata, str(spill) if spill else None)
    if matrix is not None and channels is not None:
        events = apply_compensation(events, matrix, channels)
        metadata["compensated"] = True
    else:
        metadata["compensated"] = False
    
    save_dataframe(events, events_dir / f"{sample_id}.parquet")
    _write_json(meta_dir / f"{sample_id}.json", metadata)
    
    record = dict(record)
    record["events_file"] = f"events/{sample_id}.parquet"
    record["metadata_file"] = f"metadata/{sample_id}.json"
    return record


def _qc_sample(
    record: dict, indir: Path, events_dir: Path, meta_dir: Path, qc_config: "QCConfig"
) -> tuple[dict, pd.DataFrame]:
    """Apply QC flags to a single sample's events."""
    sample_id = record["sample_id"]
    events = load_dataframe(indir / record["events_file"])
    
    qc_events = add_qc_flags(events, qc_config.model_dump())
    save_dataframe(qc_events, events_dir / f"{sample_id}.parquet")
    
    record = dict(record)
    record["events_file"] = f"events/{sample_id}.parquet"
    return record, qc_events


def _gate_sample(
    record: dict,
    indir: Path,
    events_dir: Path,
    params_dir: Path,
    figures_dir: Path,
    strategy: str,
    gate_config: dict,
    channels: dict,
) -> tuple[dict, dict]:
    """Apply gating to a single sample's events."""
    sample_id = record["sample_id"]
    events = load_dataframe(indir / record["events_file"])
    
    gated_events, params = auto_gate(events, strategy, gate_config)
    
    save_dataframe(gated_events, events_dir / f"{sample_id}.parquet")
    _write_json(params_dir / f"{sample_id}.json", params)
    
    # Generate scatter plot
    fig = plot_gating_scatter(
        events,
        gated_events,
        fsc_channel=channels.get("fsc_a", "FSC-A"),
        ssc_channel=channels.get("ssc_a", "SSC-A"),
    )
    fig.savefig(figures_dir / f"{sample_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    summary_row = {
        "sample_id": sample_id,
        "input_events": len(events),
        "gated_events": len(gated_events),
        "gated_fraction": len(gated_events) / len(events) if len(events) > 0 else 0,
    }
    summary_row.update(params)
    
    record = dict(record)
    record["events_file"] = f"events/{sample_id}.parquet"
    record["params_file"] = f"params/{sample_id}.json"
    return record, summary_row


def _resolve_marker_columns(values: str | None, cfg: AppConfig) -> Iterable[str]:
    if values:
        return [v.strip() for v in values.split(",") if v.strip()]
    markers = cfg.channels.markers
    if isinstance(markers, list) and markers:
        return markers
    raise typer.BadParameter("No marker columns provided via --values or config channels.markers")
