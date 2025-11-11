"""REST API for programmatic access to cytoflow-qc."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .cli import stage_ingest, stage_compensate, stage_qc, stage_gate, stage_drift, stage_stats
from .utils import load_config, validate_config


app = FastAPI(
    title="CytoFlow-QC API",
    description="REST API for programmatic access to cytoflow-qc pipeline",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Get API information."""
    return {
        "name": "CytoFlow-QC API",
        "version": "1.0.0",
        "description": "REST API for flow cytometry quality control pipeline",
        "endpoints": {
            "POST /process": "Run complete pipeline",
            "POST /ingest": "Data ingestion",
            "POST /compensate": "Compensation",
            "POST /qc": "Quality control",
            "POST /gate": "Automated gating",
            "POST /drift": "Batch drift analysis",
            "POST /stats": "Statistical analysis",
            "GET /health": "Health check",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}


@app.post("/process")
async def run_full_pipeline(
    samplesheet: UploadFile = File(...),
    config: UploadFile = File(...),
    output_dir: str = Form(...),
    spill: UploadFile | None = None,
):
    """Run complete cytoflow-qc pipeline."""
    try:
        # Save uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save samplesheet
            samplesheet_path = temp_path / "samplesheet.csv"
            with open(samplesheet_path, "wb") as f:
                f.write(await samplesheet.read())

            # Save config
            config_path = temp_path / "config.yaml"
            with open(config_path, "wb") as f:
                f.write(await config.read())

            # Save spillover matrix if provided
            spill_path = None
            if spill:
                spill_path = temp_path / "spillover.csv"
                with open(spill_path, "wb") as f:
                    f.write(await spill.read())

            # Load and validate configuration
            cfg = load_config(str(config_path))
            validate_config(cfg)

            # Run pipeline
            results_dir = Path(output_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: Ingest
            ingest_dir = results_dir / "ingest"
            stage_ingest(samplesheet_path, ingest_dir, cfg)

            # Stage 2: Compensate
            compensate_dir = results_dir / "compensate"
            stage_compensate(ingest_dir, compensate_dir, spill_path)

            # Stage 3: QC
            qc_dir = results_dir / "qc"
            stage_qc(compensate_dir, qc_dir, cfg.get("qc", {}))

            # Stage 4: Gate
            gate_dir = results_dir / "gate"
            stage_gate(qc_dir, gate_dir, "default", cfg)

            # Stage 5: Drift
            drift_dir = results_dir / "drift"
            stage_drift(gate_dir, drift_dir, "batch", cfg)

            # Stage 6: Stats
            stats_dir = results_dir / "stats"
            markers = cfg.get("channels", {}).get("markers", [])
            stage_stats(gate_dir, stats_dir, "condition", markers)

            return {
                "status": "success",
                "results_dir": str(results_dir),
                "stages_completed": ["ingest", "compensate", "qc", "gate", "drift", "stats"]
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_data(
    samplesheet: UploadFile = File(...),
    config: UploadFile | None = None,
    output_dir: str = Form(...),
):
    """Run data ingestion stage."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save samplesheet
            samplesheet_path = temp_path / "samplesheet.csv"
            with open(samplesheet_path, "wb") as f:
                f.write(await samplesheet.read())

            # Load config if provided
            cfg = {}
            if config:
                config_path = temp_path / "config.yaml"
                with open(config_path, "wb") as f:
                    f.write(await config.read())
                cfg = load_config(str(config_path))

            # Run ingestion
            results_dir = Path(output_dir)
            stage_ingest(samplesheet_path, results_dir / "ingest", cfg)

            return {"status": "success", "output_dir": str(results_dir / "ingest")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qc")
async def run_quality_control(
    input_dir: str = Form(...),
    output_dir: str = Form(...),
    config: UploadFile | None = None,
):
    """Run quality control stage."""
    try:
        # Load config if provided
        qc_config = {}
        if config:
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "config.yaml"
                with open(config_path, "wb") as f:
                    f.write(await config.read())
                full_config = load_config(str(config_path))
                qc_config = full_config.get("qc", {})

        # Run QC
        stage_qc(Path(input_dir), Path(output_dir), qc_config)

        return {"status": "success", "output_dir": output_dir}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gate")
async def run_gating(
    input_dir: str = Form(...),
    output_dir: str = Form(...),
    strategy: str = Form("default"),
    config: UploadFile | None = None,
):
    """Run automated gating stage."""
    try:
        # Load config if provided
        gate_config = {}
        if config:
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "config.yaml"
                with open(config_path, "wb") as f:
                    f.write(await config.read())
                full_config = load_config(str(config_path))
                gate_config = full_config

        # Run gating
        stage_gate(Path(input_dir), Path(output_dir), strategy, gate_config)

        return {"status": "success", "output_dir": output_dir}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drift")
async def run_drift_analysis(
    input_dir: str = Form(...),
    output_dir: str = Form(...),
    batch_column: str = Form("batch"),
    config: UploadFile | None = None,
):
    """Run batch drift analysis."""
    try:
        # Load config if provided
        drift_config = {}
        if config:
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "config.yaml"
                with open(config_path, "wb") as f:
                    f.write(await config.read())
                drift_config = load_config(str(config_path))

        # Run drift analysis
        stage_drift(Path(input_dir), Path(output_dir), batch_column, drift_config)

        return {"status": "success", "output_dir": output_dir}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stats")
async def run_statistics(
    input_dir: str = Form(...),
    output_dir: str = Form(...),
    group_column: str = Form("condition"),
    value_columns: str = Form("CD3-A,CD19-A,CD56-A"),
    config: UploadFile | None = None,
):
    """Run statistical analysis."""
    try:
        # Load config if provided
        stats_config = {}
        if config:
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "config.yaml"
                with open(config_path, "wb") as f:
                    f.write(await config.read())
                stats_config = load_config(str(config_path))

        # Parse value columns
        value_cols = [col.strip() for col in value_columns.split(",")]

        # Run statistics
        stage_stats(Path(input_dir), Path(output_dir), group_column, value_cols)

        return {"status": "success", "output_dir": output_dir}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{results_dir}")
async def get_results_summary(results_dir: str):
    """Get summary of pipeline results."""
    try:
        results_path = Path(results_dir)

        # Check if results exist
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Results directory not found")

        summary = {
            "results_dir": results_dir,
            "stages": {},
            "metrics": {}
        }

        # Check each stage
        stages = ["ingest", "compensate", "qc", "gate", "drift", "stats"]
        for stage in stages:
            stage_dir = results_path / stage
            if stage_dir.exists():
                summary["stages"][stage] = {
                    "exists": True,
                    "files": [f.name for f in stage_dir.glob("*") if f.is_file()]
                }
            else:
                summary["stages"][stage] = {"exists": False}

        # Calculate metrics if data available
        qc_summary = results_path / "qc" / "summary.csv"
        if qc_summary.exists():
            qc_df = pd.read_csv(qc_summary)
            summary["metrics"]["qc_pass_rate"] = float(qc_df["qc_pass_fraction"].mean())

        gate_summary = results_path / "gate" / "summary.csv"
        if gate_summary.exists():
            gate_df = pd.read_csv(gate_summary)
            summary["metrics"]["retention_rate"] = float(
                gate_df["gated_events"].sum() / gate_df["input_events"].sum()
            )

        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{results_dir}/{filename}")
async def download_file(results_dir: str, filename: str):
    """Download a specific file from results."""
    try:
        file_path = Path(results_dir) / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            file_path,
            media_type='application/octet-stream',
            filename=filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/{results_dir}")
async def get_dashboard(results_dir: str):
    """Get interactive dashboard for results."""
    try:
        results_path = Path(results_dir)

        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Results directory not found")

        # Generate dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CytoFlow-QC Dashboard - {results_dir}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                         color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ background: #f0f2f6; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                iframe {{ width: 100%; height: 600px; border: none; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 CytoFlow-QC Dashboard</h1>
                <p>Results for: {results_dir}</p>
            </div>

            <div class="section">
                <h2>📊 Pipeline Overview</h2>
                <p>Interactive dashboard for exploring flow cytometry quality control results.</p>
                <p><strong>Dashboard URL:</strong> <code>/dashboard/{results_dir}/interactive</code></p>
            </div>

            <div class="section">
                <h2>📁 Available Files</h2>
                <ul>
        """

        # List available files
        for file_path in sorted(results_path.rglob("*")):
            if file_path.is_file():
                rel_path = file_path.relative_to(results_path)
                dashboard_html += f'<li><a href="/download/{results_dir}/{rel_path}">{rel_path}</a></li>\n'

        dashboard_html += """
                </ul>
            </div>

            <div class="section">
                <h2>🚀 Quick Actions</h2>
                <ul>
                    <li><a href="/dashboard/{results_dir}/interactive">Launch Interactive Dashboard</a></li>
                    <li><a href="/results/{results_dir}">View Results Summary</a></li>
                </ul>
            </div>
        </body>
        </html>
        """

        return HTMLResponse(content=dashboard_html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/{results_dir}/interactive")
async def get_interactive_dashboard(results_dir: str):
    """Get interactive dashboard HTML."""
    try:
        from .interactive_viz import InteractiveVisualizer

        results_path = Path(results_dir)
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Results directory not found")

        # Create visualizer
        visualizer = InteractiveVisualizer(results_path)

        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CytoFlow-QC Interactive Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                         color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .sidebar {{ width: 250px; float: left; padding: 20px; background: #f8f9fa; }}
                .main {{ margin-left: 270px; padding: 20px; }}
                .section {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 CytoFlow-QC Interactive Dashboard</h1>
                <p>Real-time exploration of flow cytometry results</p>
            </div>

            <div class="content">
                <div class="sidebar">
                    <h3>Navigation</h3>
                    <ul>
                        <li><a href="#overview">Overview</a></li>
                        <li><a href="#qc">Quality Control</a></li>
                        <li><a href="#gating">Gating Analysis</a></li>
                        <li><a href="#drift">Drift Analysis</a></li>
                        <li><a href="#stats">Statistics</a></li>
                    </ul>
                </div>

                <div class="main">
                    <div id="overview" class="section">
                        <h2>📊 Pipeline Overview</h2>
        """

        # Add overview metrics
        if visualizer.qc_data is not None:
            total_samples = len(visualizer.qc_data)
            mean_qc_pass = visualizer.qc_data["qc_pass_fraction"].mean()
            html_content += f"""
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                            <div style="background: #e3f2fd; padding: 15px; border-radius: 5px;">
                                <h3>Samples Processed</h3>
                                <p style="font-size: 2em; margin: 0;">{total_samples}</p>
                            </div>
                            <div style="background: #f3e5f5; padding: 15px; border-radius: 5px;">
                                <h3>QC Pass Rate</h3>
                                <p style="font-size: 2em; margin: 0;">{mean_qc_pass".1%"}</p>
                            </div>
                        </div>
            """

        html_content += """
                    </div>

                    <div id="qc" class="section">
                        <h2>✅ Quality Control Analysis</h2>
                        <p>Interactive quality control metrics and visualizations will be displayed here.</p>
                        <p><em>Full interactive functionality requires running the Streamlit dashboard locally.</em></p>
                    </div>

                    <div id="gating" class="section">
                        <h2>🚪 Gating Analysis</h2>
                        <p>Interactive gating visualizations and parameter inspection.</p>
                    </div>

                    <div id="drift" class="section">
                        <h2>📊 Batch Drift Analysis</h2>
                        <p>Interactive PCA plots and statistical test results.</p>
                    </div>

                    <div id="stats" class="section">
                        <h2>📈 Statistical Analysis</h2>
                        <p>Interactive effect size plots and volcano plots.</p>
                    </div>
                </div>
            </div>

            <script>
                // Add smooth scrolling for navigation
                document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                    anchor.addEventListener('click', function (e) {
                        e.preventDefault();
                        const target = document.querySelector(this.getAttribute('href'));
                        if (target) {
                            target.scrollIntoView({ behavior: 'smooth' });
                        }
                    });
                });
            </script>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)






