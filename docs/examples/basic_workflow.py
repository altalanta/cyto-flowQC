#!/usr/bin/env python3
"""
Basic workflow example for cytoflow-qc.

This example demonstrates the complete pipeline from data ingestion
through analysis and reporting.
"""

import tempfile
from pathlib import Path

import pandas as pd

from cytoflow_qc.cli import (
    stage_ingest, stage_compensate, stage_qc, stage_gate,
    stage_drift, stage_stats
)


def main() -> None:
    """Run a complete cytoflow-qc workflow."""
    # Create temporary directory for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Configuration for the analysis
        config = {
            "channels": {
                "fsc_a": "FSC-A",
                "fsc_h": "FSC-H",
                "ssc_a": "SSC-A",
                "viability": "Zombie_Aqua-A",
                "markers": ["CD3-A", "CD19-A", "CD56-A", "CD4-A", "CD8-A"],
            },
            "qc": {
                "debris": {"fsc_percentile": 5.0, "ssc_percentile": 5.0},
                "doublets": {"tolerance": 0.1},
                "saturation": {"threshold": 0.99},
            },
            "gating": {
                "strategy": "default",
                "lymphocytes": {"low_percentile": 10.0, "high_percentile": 80.0},
            },
        }

        # Create sample data
        sample_data = create_sample_data()

        # Create samplesheet
        samplesheet = create_samplesheet(sample_data, temp_path)

        print("🔬 Starting cytoflow-qc pipeline...")

        # Stage 1: Data Ingestion
        print("📥 Stage 1: Data Ingestion")
        ingest_dir = temp_path / "ingest"
        stage_ingest(samplesheet, ingest_dir, config)

        # Stage 2: Compensation
        print("⚖️ Stage 2: Compensation")
        compensate_dir = temp_path / "compensate"
        stage_compensate(ingest_dir, compensate_dir, None)  # No spillover matrix

        # Stage 3: Quality Control
        print("✅ Stage 3: Quality Control")
        qc_dir = temp_path / "qc"
        stage_qc(compensate_dir, qc_dir, config["qc"])

        # Stage 4: Automated Gating
        print("🚪 Stage 4: Automated Gating")
        gate_dir = temp_path / "gate"
        stage_gate(qc_dir, gate_dir, "default", config)

        # Stage 5: Batch Drift Analysis
        print("📊 Stage 5: Batch Drift Analysis")
        drift_dir = temp_path / "drift"
        stage_drift(gate_dir, drift_dir, "batch", config)

        # Stage 6: Statistical Analysis
        print("📈 Stage 6: Statistical Analysis")
        stats_dir = temp_path / "stats"
        stage_stats(gate_dir, stats_dir, "condition", ["CD3-A", "CD19-A", "CD56-A"])

        print("🎉 Pipeline completed successfully!")
        print(f"📁 Results available in: {temp_path}")

        # Show summary of results
        show_results_summary(temp_path)


def create_sample_data() -> dict[str, pd.DataFrame]:
    """Create synthetic FCS-like data for the example."""
    import numpy as np

    data = {}

    # Create 6 samples across 2 batches and 3 conditions
    conditions = ["control", "treatment", "vehicle"]
    batches = ["batch_1", "batch_2"]

    for i, (batch, condition) in enumerate(zip(
        batches * 3,  # Alternate batches
        conditions * 2  # Repeat conditions
    )):
        sample_id = f"sample_{i+1"03d"}"

        # Generate realistic FCS data
        n_events = 10000
        events = pd.DataFrame({
            "FSC-A": np.random.lognormal(6, 0.5, n_events),
            "FSC-H": np.random.lognormal(6, 0.5, n_events),
            "SSC-A": np.random.lognormal(5, 0.5, n_events),
            "CD3-A": np.random.lognormal(4, 0.8, n_events),
            "CD19-A": np.random.lognormal(3, 0.8, n_events),
            "CD56-A": np.random.lognormal(3.5, 0.8, n_events),
            "CD4-A": np.random.lognormal(3.8, 0.8, n_events),
            "CD8-A": np.random.lognormal(3.2, 0.8, n_events),
        })

        # Add realistic correlations
        events["FSC-H"] = events["FSC-A"] * (1 + np.random.normal(0, 0.05, n_events))

        # Add condition-specific effects
        if condition == "treatment":
            # Treatment increases CD3 expression
            events["CD3-A"] *= 1.2
        elif condition == "vehicle":
            # Vehicle has slight increase in CD19
            events["CD19-A"] *= 1.1

        data[sample_id] = events

    return data


def create_samplesheet(sample_data: dict[str, pd.DataFrame], temp_dir: Path) -> Path:
    """Create a samplesheet CSV for the sample data."""
    samplesheet_data = []

    for sample_id, events in sample_data.items():
        # Save sample data as CSV
        sample_file = temp_dir / f"{sample_id}.csv"
        events.to_csv(sample_file, index=False)

        # Determine metadata based on sample ID
        sample_num = int(sample_id.split("_")[1])
        batch = "batch_1" if sample_num <= 3 else "batch_2"
        condition = ["control", "treatment", "vehicle"][(sample_num - 1) % 3]

        samplesheet_data.append({
            "sample_id": sample_id,
            "file_path": str(sample_file),
            "batch": batch,
            "condition": condition,
            "timepoint": "T0",
            "replicate": 1,
        })

    samplesheet_path = temp_dir / "samplesheet.csv"
    pd.DataFrame(samplesheet_data).to_csv(samplesheet_path, index=False)

    return samplesheet_path


def show_results_summary(results_dir: Path) -> None:
    """Display a summary of the pipeline results."""
    print("\n📋 Pipeline Results Summary:")

    # Check QC summary
    qc_summary = results_dir / "qc" / "summary.csv"
    if qc_summary.exists():
        df = pd.read_csv(qc_summary)
        print(f"✅ QC: Processed {len(df)} samples")
        mean_pass_rate = df["qc_pass_fraction"].mean()
        print(f"   Average QC pass rate: {mean_pass_rate".2%"}")

    # Check gating summary
    gate_summary = results_dir / "gate" / "summary.csv"
    if gate_summary.exists():
        df = pd.read_csv(gate_summary)
        print(f"🚪 Gating: Retained {df['gated_events'].sum():,","events across all samples")

    # Check drift analysis
    drift_tests = results_dir / "drift" / "tests.csv"
    if drift_tests.exists():
        df = pd.read_csv(drift_tests)
        significant = (df["adj_p_value"] < 0.05).sum()
        print(f"📊 Drift: Found {significant} significant batch effects")

    # Check statistics
    stats_file = results_dir / "stats" / "effect_sizes.csv"
    if stats_file.exists():
        df = pd.read_csv(stats_file)
        print(f"📈 Statistics: Computed {len(df)} effect sizes")

    print(f"\n📂 All results saved in: {results_dir}")


if __name__ == "__main__":
    main()






