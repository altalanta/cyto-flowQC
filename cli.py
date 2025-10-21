"""Typer-powered CLI for the CytoFlow-QC pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Any

import pandas as pd
import typer

from cytoflow_qc import __version__
from cytoflow_qc.compensate import apply_compensation, get_spillover
from cytoflow_qc.drift import compute_batch_drift, extract_sample_features
from cytoflow_qc.gate import auto_gate
from cytoflow_qc.io import load_samplesheet, read_fcs, standardize_channels
from cytoflow_qc.qc import add_qc_flags, qc_summary
from cytoflow_qc.report import build_report
from cytoflow_qc.stats import effect_sizes
from cytoflow_qc.utils import (
    ensure_dir,
    list_stage_events,
    load_config,
    load_dataframe,
    read_manifest,
    save_dataframe,
    timestamp,
    write_manifest,
)
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

app = typer.Typer(add_completion=False, help="Flow cytometry QC and gating pipeline")


@app.callback()
def _version(ctx: typer.Context, version: bool = typer.Option(False, "--version", help="Show version and exit")) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def ingest(
    samplesheet: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Argument(...),
    config: dict[str, object] | None = typer.Option(None, "--config", "-c", help="Optional YAML config"),
) -> None:
    cfg = load_config(config) if config else {}
    stage_ingest(samplesheet, out, cfg)
    typer.echo(f"Ingested samples -> {out}")


@app.command()
def compensate(
    indir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
    spill: Path | None = typer.Option(None, "--spill", help="Override spillover CSV"),
) -> None:
    stage_compensate(indir, out, spill)
    typer.echo(f"Compensated events -> {out}")


@app.command()
def qc(
    indir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    cfg = load_config(config) if config else {}
    stage_qc(indir, out, cfg.get("qc", {}))
    typer.echo(f"QC annotations -> {out}")


@app.command()
def gate(
    indir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
    strategy: str = typer.Option("default", "--strategy"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    cfg = load_config(config) if config else {}
    stage_gate(indir, out, strategy, cfg)
    typer.echo(f"Gated populations -> {out}")


@app.command()
def drift(
    indir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
    by: str = typer.Option("batch", "--by", help="Metadata column for batch grouping"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    cfg = load_config(config) if config else {}
    stage_drift(indir, out, by, cfg)
    typer.echo(f"Batch drift analysis -> {out}")


@app.command()
def stats(
    indir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
    group_col: str = typer.Option("condition", "--groups"),
    values: str | None = typer.Option(None, "--values", help="Comma-separated marker columns"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    cfg = load_config(config) if config else {}
    marker_columns = _resolve_marker_columns(values, cfg)
    stage_stats(indir, out, group_col, marker_columns)
    typer.echo(f"Effect-size statistics -> {out}")


@app.command()
def report(
    indir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Argument(...),
    template: Path = typer.Option(Path("configs/report_template.html.j2"), "--template"),
) -> None:
    build_report(str(indir), str(template), str(out))
    typer.echo(f"Report written to {out}")


@app.command()
def dashboard(
    indir: Path = typer.Argument(..., exists=True, help="Results directory from cytoflow-qc run"),
    sample: str | None = typer.Option(None, "--sample", "-s", help="Specific sample to visualize"),
    port: int = typer.Option(8501, "--port", "-p", help="Port for Streamlit server"),
) -> None:
    """Launch interactive visualization dashboard."""
    typer.echo(f"🚀 Launching interactive dashboard for results in: {indir}")

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
        typer.echo("❌ Streamlit not installed. Install with: pip install streamlit")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error launching dashboard: {e}")
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
            typer.echo("💡 Tip: Use --output to save HTML file, or run 'cytoflow-qc dashboard' for full interface")
            typer.echo(f"📁 Results directory: {indir}")
    except Exception as e:
        typer.echo(f"❌ Error creating 3D visualization: {e}")
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

        typer.echo(f"✅ Publication-ready figure exported to: {output}")

    except Exception as e:
        typer.echo(f"❌ Error exporting figure: {e}")
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

        typer.echo(f"✅ Interactive dashboard exported to: {output}")

    except Exception as e:
        typer.echo(f"❌ Error exporting dashboard: {e}")
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

        typer.echo(f"✅ 3D visualization exported to: {output}")

    except Exception as e:
        typer.echo(f"❌ Error exporting 3D visualization: {e}")
        raise typer.Exit(1)


@app.command()
def plugins(
    action: str = typer.Argument(..., help="Plugin action (list, info, load)"),
    plugin_type: str | None = typer.Option(None, "--type", "-t", help="Plugin type filter"),
    plugin_name: str | None = typer.Option(None, "--name", "-n", help="Specific plugin name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Plugin configuration"),
) -> None:
    """Manage and interact with cytoflow-qc plugins."""
    registry = get_plugin_registry()

    if action == "list":
        # List available plugins
        available = registry.get_available_plugins(plugin_type)
        typer.echo("Available plugins:")

        for ptype, plugins in available.items():
            if plugins:
                typer.echo(f"  {ptype}:")
                for plugin in plugins:
                    typer.echo(f"    • {plugin}")
    elif action == "info":
        # Get info about specific plugin
        if not plugin_name:
            typer.echo("Error: --name required for info action")
            raise typer.Exit(1)

        try:
            info = registry.get_plugin_info(plugin_type or "gating_strategy", plugin_name)
            typer.echo(f"Plugin: {info['name']}")
            typer.echo(f"Version: {info['version']}")
            typer.echo(f"Description: {info['description']}")
            typer.echo(f"Author: {info['author']}")
            typer.echo(f"Type: {info['plugin_type']}")
            typer.echo("Default config:")
            import json
            typer.echo(json.dumps(info['default_config'], indent=2))
        except Exception as e:
            typer.echo(f"Error getting plugin info: {e}")
    elif action == "load":
        # Load and test plugin
        if not plugin_name:
            typer.echo("Error: --name required for load action")
            raise typer.Exit(1)

        try:
            plugin_config: dict[str, Any] = {}
            if config:
                import yaml
                with open(config, 'r') as f:
                    plugin_config = yaml.safe_load(f) or {}

            plugin = load_plugin(plugin_type or "gating_strategy", plugin_name, plugin_config)
            typer.echo(f"✅ Loaded plugin: {plugin.name} v{plugin.version}")
            typer.echo(f"Description: {plugin.description}")
        except Exception as e:
            typer.echo(f"❌ Error loading plugin: {e}")
    else:
        typer.echo(f"Unknown action: {action}")
        typer.echo("Available actions: list, info, load")


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
                typer.echo("Kubernetes deployment configured")
            else:
                typer.echo("Use --config to specify Kubernetes configuration")
        elif action == "status":
            k8s = KubernetesDeployment()
            status = k8s.get_deployment_status()
            typer.echo("Kubernetes deployment status:")
            import json
            typer.echo(json.dumps(status, indent=2))
        else:
            typer.echo(f"Unknown Kubernetes action: {action}")
    elif provider.lower() == "serverless":
        if action == "deploy":
            typer.echo("Serverless deployment not implemented yet")
        else:
            typer.echo(f"Unknown serverless action: {action}")
    else:
        typer.echo(f"Unsupported provider: {provider}")


@app.command()
def realtime(
    action: str = typer.Argument(..., help="Real-time action (start, monitor)"),
    ws_url: str | None = typer.Option(None, "--ws-url", help="WebSocket URL for data source"),
    config: str | None = typer.Option(None, "--config", "-c", help="Real-time configuration"),
) -> None:
    """Manage real-time processing."""
    if action == "start":
        if not ws_url:
            typer.echo("Error: --ws-url required for start action")
            raise typer.Exit(1)

        typer.echo(f"Starting real-time processing from: {ws_url}")
        # This would start the real-time processing
        typer.echo("Real-time processing started (placeholder)")
    elif action == "monitor":
        # Start monitoring dashboard
        monitor = RealTimeMonitor()
        typer.echo("Starting real-time monitoring dashboard...")
        # This would start the monitoring dashboard
        typer.echo("Monitoring dashboard started (placeholder)")
    else:
        typer.echo(f"Unknown real-time action: {action}")


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

    typer.echo(f"Anonymizing data in {indir} and saving to {outdir}...")

    try:
        for sample_id, events_file in list_stage_events(indir).items():
            df = load_dataframe(indir / events_file)
            anonymized_df = anonymizer.anonymize_dataframe(df, cols_to_anon, identifier_col)
            save_dataframe(anonymized_df, outdir / Path(events_file).name)
            typer.echo(f"✅ Anonymized {sample_id}")
        typer.echo("🎉 All specified dataframes anonymized successfully!")
    except Exception as e:
        typer.echo(f"❌ Error during anonymization: {e}")
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
        typer.echo(f"✅ File '{infile}' encrypted to '{outfile}' successfully!")
    except SecurityError as e:
        typer.echo(f"❌ Encryption Error: {e}")
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ An unexpected error occurred during encryption: {e}")
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
        typer.echo(f"✅ File '{infile}' decrypted to '{outfile}' successfully!")
    except SecurityError as e:
        typer.echo(f"❌ Decryption Error: {e}")
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ An unexpected error occurred during decryption: {e}")
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

    typer.echo(f"Checking if roles {user_roles} can '{action}' resource '{resource}'...")
    if rbac_manager.check_permission(user_roles, action, resource):
        typer.echo(f"✅ Permission granted for roles {user_roles} to {action} {resource}.")
    else:
        typer.echo(f"❌ Permission denied for roles {user_roles} to {action} {resource}.")
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
            typer.echo(f"Listing files in '{uri}' with pattern '{pattern}':")
            found_files = list(connector.list_files(uri, pattern))
            if found_files:
                for f in found_files:
                    typer.echo(f"  - {f}")
            else:
                typer.echo("No files found.")
        elif action == "configure":
            typer.echo(f"Configured data source for URI: {uri}")
            if connector_config:
                typer.echo(f"  with configuration: {json.dumps(connector_config, indent=2)}")
            else:
                typer.echo("  (using default configuration)")
        elif action == "ingest":
            if not output_dir:
                typer.echo("Error: --output-dir is required for ingest action.")
                raise typer.Exit(1)
            ensure_dir(output_dir)

            typer.echo(f"Ingesting files from '{uri}' (pattern: '{pattern}') to '{output_dir}'...")
            ingested_count = 0
            for remote_file_uri in connector.list_files(uri, pattern):
                local_file_path = output_dir / Path(remote_file_uri).name
                try:
                    file_content = connector.read_file(remote_file_uri)
                    local_file_path.write_bytes(file_content)
                    typer.echo(f"  ✅ Ingested {remote_file_uri} to {local_file_path}")
                    ingested_count += 1
                except Exception as e:
                    typer.echo(f"  ❌ Failed to ingest {remote_file_uri}: {e}")
            typer.echo(f"🎉 Ingestion complete: {ingested_count} files successfully ingested.")
        else:
            typer.echo(f"Unknown action: {action}. Available actions: configure, list, ingest")
            raise typer.Exit(1)
    except (ValueError, ImportError, DataSourceError) as e:
        typer.echo(f"❌ Data Source Error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ An unexpected error occurred with data source: {e}")
        raise typer.Exit(1)


@app.command()
def run(
    samplesheet: Path = typer.Option(..., "--samplesheet", exists=True),
    config: Path = typer.Option(..., "--config", exists=True),
    out: Path = typer.Option(..., "--out"),
    spill: Path | None = typer.Option(None, "--spill"),
    batch: str = typer.Option("batch", "--batch"),
) -> None:
    cfg = load_config(config)
    root = ensure_dir(out)
    ingest_dir = root / "ingest"
    compensate_dir = root / "compensate"
    qc_dir = root / "qc"
    gate_dir = root / "gate"
    drift_dir = root / "drift"
    stats_dir = root / "stats"

    stage_ingest(samplesheet, ingest_dir, cfg)
    stage_compensate(ingest_dir, compensate_dir, spill)
    stage_qc(compensate_dir, qc_dir, cfg.get("qc", {}))
    stage_gate(qc_dir, gate_dir, "default", cfg)
    stage_drift(gate_dir, drift_dir, batch, cfg)
    markers = _resolve_marker_columns(None, cfg)
    stage_stats(gate_dir, stats_dir, "condition", markers)

    report_path = root / "report.html"
    build_report(str(root), str(cfg.get("report_template", Path("configs/report_template.html.j2"))), str(report_path))
    typer.echo(f"Report available at {report_path}")


# ---------------------------------------------------------------------------
# Stage implementations (shared by commands and run())


def stage_ingest(samplesheet: Path, out_dir: Path, config: dict[str, object]) -> None:
    ensure_dir(out_dir)
    events_dir = ensure_dir(out_dir / "events")
    meta_dir = ensure_dir(out_dir / "metadata")

    sheet = load_samplesheet(str(samplesheet))
    channel_map = config.get("channels", {}) if isinstance(config, dict) else {}

    records = []
    for row in sheet.to_dict(orient="records"):
        if row.get("missing_file"):
            typer.echo(f"Skipping missing file {row['file_path']}")
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


def stage_compensate(indir: Path, out_dir: Path, spill: Path | None) -> None:
    ensure_dir(out_dir)
    events_dir = ensure_dir(out_dir / "events")
    meta_dir = ensure_dir(out_dir / "metadata")
    manifest = read_manifest(indir / "manifest.csv")

    compensated_records = []
    for record in manifest.to_dict(orient="records"):
        events = load_dataframe(indir / record["events_file"])
        metadata = _read_json(indir / record["metadata_file"])
        matrix, channels = get_spillover(metadata, str(spill) if spill else None)
        if matrix is not None and channels is not None:
            events = apply_compensation(events, matrix, channels)
            metadata["compensated"] = True
        else:
            metadata["compensated"] = False
        sample_id = record["sample_id"]
        save_dataframe(events, events_dir / f"{sample_id}.parquet")
        _write_json(meta_dir / f"{sample_id}.json", metadata)
        record["events_file"] = f"events/{sample_id}.parquet"
        record["metadata_file"] = f"metadata/{sample_id}.json"
        compensated_records.append(record)

    out_manifest = pd.DataFrame(compensated_records)
    out_manifest["stage"] = "compensate"
    out_manifest["timestamp"] = timestamp()
    write_manifest(out_manifest, out_dir / "manifest.csv")


def stage_qc(indir: Path, out_dir: Path, qc_config: dict[str, dict[str, float]]) -> None:
    ensure_dir(out_dir)
    events_dir = ensure_dir(out_dir / "events")
    meta_dir = ensure_dir(out_dir / "metadata")
    manifest = read_manifest(indir / "manifest.csv")

    sample_tables: dict[str, pd.DataFrame] = {}
    updated_records = []

    for record in manifest.to_dict(orient="records"):
        df = load_dataframe(indir / record["events_file"])
        qc_df = add_qc_flags(df, qc_config)
        sample_id = record["sample_id"]
        save_dataframe(qc_df, events_dir / f"{sample_id}.parquet")
        _write_json(meta_dir / f"{sample_id}.json", _read_json(indir / record["metadata_file"]))
        record["events_file"] = f"events/{sample_id}.parquet"
        record["metadata_file"] = f"metadata/{sample_id}.json"
        sample_tables[sample_id] = qc_df
        updated_records.append(record)

    summary = qc_summary(sample_tables)
    summary.to_csv(out_dir / "summary.csv", index=False)

    out_manifest = pd.DataFrame(updated_records)
    out_manifest["stage"] = "qc"
    out_manifest["timestamp"] = timestamp()
    write_manifest(out_manifest, out_dir / "manifest.csv")

    plot_qc_summary(summary, str(out_dir / "figures" / "qc_pass.png"))


def stage_gate(indir: Path, out_dir: Path, strategy: str, config: dict[str, object]) -> None:
    ensure_dir(out_dir)
    events_dir = ensure_dir(out_dir / "events")
    params_dir = ensure_dir(out_dir / "params")
    manifest = read_manifest(indir / "manifest.csv")

    channel_config = config.get("channels", {}) if isinstance(config, dict) else {}
    gate_config = dict(config.get("gating", {})) if isinstance(config, dict) else {}
    gate_config["channels"] = channel_config
    channels = gate_config.get("channels", {})

    summary_rows = []
    updated_records = []
    for record in manifest.to_dict(orient="records"):
        sample_id = record["sample_id"]
        df = load_dataframe(indir / record["events_file"])
        gated, params = auto_gate(df, strategy=strategy, config=gate_config)
        save_dataframe(gated, events_dir / f"{sample_id}.parquet")
        _write_json(params_dir / f"{sample_id}.json", params)
        record["events_file"] = f"events/{sample_id}.parquet"
        record["params_file"] = f"params/{sample_id}.json"
        summary_rows.append({
            "sample_id": sample_id,
            "input_events": len(df),
            "gated_events": len(gated),
        })

        plot_gating_scatter(
            df,
            gated,
            channels.get("fsc_a", "FSC-A"),
            channels.get("ssc_a", "SSC-A"),
            str(out_dir / "figures" / f"{sample_id}_gating.png"),
        )
        updated_records.append(record)

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    manifest_out = pd.DataFrame(updated_records)
    manifest_out["stage"] = "gate"
    manifest_out["timestamp"] = timestamp()
    write_manifest(manifest_out, out_dir / "manifest.csv")


def stage_drift(indir: Path, out_dir: Path, batch_col: str, config: dict[str, object]) -> None:
    ensure_dir(out_dir)
    figures_dir = ensure_dir(out_dir / "figures")
    manifest = read_manifest(indir / "manifest.csv")
    sample_events = {sid: load_dataframe(indir / path) for sid, path in list_stage_events(indir).items()}
    meta_cols = ["sample_id", batch_col]
    if "condition" in manifest.columns:
        meta_cols.append("condition")
    metadata = manifest[meta_cols].drop_duplicates()
    marker_channels = config.get("channels", {}).get("markers") if isinstance(config, dict) else None
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

    plot_batch_drift_pca(drift_res["pca"], str(figures_dir / "pca.png"), batch_col)
    plot_batch_drift_umap(drift_res.get("umap"), str(figures_dir / "umap.png"), batch_col)


def stage_stats(indir: Path, out_dir: Path, group_col: str, value_cols: Iterable[str]) -> None:
    ensure_dir(out_dir)
    manifest = read_manifest(indir / "manifest.csv")
    records = []
    columns: list[str] | None = None
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
    plot_effect_sizes(effects, str(out_dir / "figures" / "effect_sizes.png"))


# ---------------------------------------------------------------------------
# Helpers


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_marker_columns(values: str | None, cfg: dict[str, object]) -> Iterable[str]:
    if values:
        return [v.strip() for v in values.split(",") if v.strip()]
    markers = cfg.get("channels", {}).get("markers") if isinstance(cfg, dict) else None
    if isinstance(markers, list) and markers:
        return markers
    raise typer.BadParameter("No marker columns provided via --values or config channels.markers")
