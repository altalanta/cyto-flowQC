#!/usr/bin/env python3
"""
Advanced configuration example for cytoflow-qc.

This example demonstrates custom gating strategies, advanced QC settings,
and multi-batch experimental designs.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cytoflow_qc.cli import (
    stage_ingest, stage_compensate, stage_qc, stage_gate,
    stage_drift, stage_stats
)


def main() -> None:
    """Run advanced cytoflow-qc configuration example."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Advanced configuration with multiple gating strategies
        config = {
            "channels": {
                "fsc_a": "FSC-A",
                "fsc_h": "FSC-H",
                "fsc_w": "FSC-W",
                "ssc_a": "SSC-A",
                "ssc_h": "SSC-H",
                "viability": "Zombie_Aqua-A",
                "markers": [
                    "CD3-A", "CD19-A", "CD56-A", "CD4-A", "CD8-A",
                    "CD45-A", "CD14-A", "CD16-A"
                ],
            },
            "qc": {
                "debris": {
                    "fsc_percentile": 2.0,
                    "ssc_percentile": 2.0,
                },
                "doublets": {
                    "tolerance": 0.08,
                },
                "saturation": {
                    "threshold": 0.995,
                },
                "channel_qc": {
                    "saturation_threshold": 0.95,
                    "min_dynamic_range": 2.0,
                    "min_snr": 3.0,
                },
            },
            "gating": {
                "strategy": "default",
                "lymphocytes": {
                    "method": "density",
                    "percentile": 90,
                },
                "singlets": {
                    "method": "linear",
                    "slope_tolerance": 0.05,
                },
                "viability": {
                    "threshold": 5000,
                    "direction": "below",
                },
            },
            "compensation": {
                "method": "auto",
            },
            "transforms": {
                "method": "logicle",
                "parameters": {
                    "T": 262144,
                    "W": 0.5,
                    "M": 4.5,
                    "A": 0,
                },
            },
        }

        # Create multi-batch experimental data
        sample_data = create_multi_batch_data()

        # Create samplesheet with complex experimental design
        samplesheet = create_complex_samplesheet(sample_data, temp_path)

        print("🔬 Running advanced cytoflow-qc pipeline...")

        # Execute pipeline with advanced configuration
        run_advanced_pipeline(samplesheet, temp_path, config)

        print("🎉 Advanced pipeline completed!")
        show_advanced_results(temp_path)


def create_multi_batch_data() -> dict[str, pd.DataFrame]:
    """Create data simulating a complex multi-batch experiment."""
    import numpy as np

    data = {}

    # Simulate 4 batches with different characteristics
    batch_configs = [
        {"name": "batch_1", "fsc_shift": 0.0, "cd3_shift": 0.0, "n_samples": 3},
        {"name": "batch_2", "fsc_shift": 0.1, "cd3_shift": 0.05, "n_samples": 3},
        {"name": "batch_3", "fsc_shift": -0.05, "cd3_shift": -0.1, "n_samples": 2},
        {"name": "batch_4", "fsc_shift": 0.15, "cd3_shift": 0.2, "n_samples": 2},
    ]

    conditions = ["control", "treatment_A", "treatment_B"]
    timepoints = ["T0", "T24", "T48"]

    sample_counter = 1

    for batch_config in batch_configs:
        batch_name = batch_config["name"]
        fsc_shift = batch_config["fsc_shift"]
        cd3_shift = batch_config["cd3_shift"]

        for sample_idx in range(batch_config["n_samples"]):
            # Vary conditions and timepoints
            condition = conditions[sample_counter % len(conditions)]
            timepoint = timepoints[sample_counter % len(timepoints)]

            sample_id = f"{batch_name}_sample_{sample_counter"02d"}"

            # Base parameters with batch-specific shifts
            n_events = 15000 + np.random.randint(-2000, 2000)

            events = pd.DataFrame({
                "FSC-A": np.random.lognormal(6.0 + fsc_shift, 0.5, n_events),
                "FSC-H": np.random.lognormal(6.0 + fsc_shift, 0.5, n_events),
                "SSC-A": np.random.lognormal(5.0, 0.5, n_events),
                "CD3-A": np.random.lognormal(4.0 + cd3_shift, 0.8, n_events),
                "CD19-A": np.random.lognormal(3.0, 0.8, n_events),
                "CD56-A": np.random.lognormal(3.5, 0.8, n_events),
                "CD4-A": np.random.lognormal(3.8, 0.8, n_events),
                "CD8-A": np.random.lognormal(3.2, 0.8, n_events),
                "CD45-A": np.random.lognormal(5.5, 0.6, n_events),
                "CD14-A": np.random.lognormal(3.0, 0.9, n_events),
                "CD16-A": np.random.lognormal(2.8, 0.9, n_events),
                "Zombie_Aqua-A": np.random.lognormal(2.5, 0.7, n_events),
            })

            # Add correlations and condition-specific effects
            events = add_realistic_characteristics(events, condition, timepoint)

            data[sample_id] = events
            sample_counter += 1

    return data


def add_realistic_characteristics(events: pd.DataFrame, condition: str, timepoint: str) -> pd.DataFrame:
    """Add realistic biological and technical characteristics to the data."""
    import numpy as np

    # FSC-A and FSC-H correlation
    events["FSC-H"] = events["FSC-A"] * (1 + np.random.normal(0, 0.05, len(events)))

    # SSC correlations
    events["SSC-H"] = events["SSC-A"] * (1 + np.random.normal(0, 0.08, len(events)))

    # Condition-specific effects
    if condition == "treatment_A":
        # Treatment A increases T cell markers
        events["CD3-A"] *= 1.3
        events["CD4-A"] *= 1.2
        events["CD8-A"] *= 1.15

    elif condition == "treatment_B":
        # Treatment B increases NK cells
        events["CD56-A"] *= 1.4
        events["CD16-A"] *= 1.25

    # Timepoint effects
    if timepoint == "T24":
        # Some activation at 24h
        events["CD69-A"] = np.random.lognormal(2.5, 0.8, len(events)) if "CD69-A" not in events.columns else events["CD69-A"] * 1.1

    elif timepoint == "T48":
        # More activation at 48h
        if "CD69-A" not in events.columns:
            events["CD69-A"] = np.random.lognormal(3.0, 0.8, len(events))
        else:
            events["CD69-A"] *= 1.2

    # Add some technical variation (batch effects)
    for col in events.columns:
        if col.startswith("CD") or col.startswith("SSC"):
            # Technical noise
            technical_noise = np.random.normal(0, 0.02, len(events))
            events[col] = events[col] * (1 + technical_noise)

    return events


def create_complex_samplesheet(sample_data: dict[str, pd.DataFrame], temp_dir: Path) -> Path:
    """Create a complex samplesheet with detailed experimental metadata."""
    samplesheet_data = []

    condition_counter = {"control": 1, "treatment_A": 1, "treatment_B": 1}

    for sample_id, events in sample_data.items():
        # Save sample data
        sample_file = temp_dir / f"{sample_id}.csv"
        events.to_csv(sample_file, index=False)

        # Extract metadata from sample ID
        batch = sample_id.split("_")[0]
        sample_num = int(sample_id.split("_")[-1])

        # Assign conditions in a balanced design
        if sample_num % 3 == 1:
            condition = "control"
        elif sample_num % 3 == 2:
            condition = "treatment_A"
        else:
            condition = "treatment_B"

        # Assign timepoints
        timepoint = ["T0", "T24", "T48"][(sample_num - 1) % 3]

        # Generate replicate numbers
        replicate = condition_counter[condition]
        condition_counter[condition] += 1

        samplesheet_data.append({
            "sample_id": sample_id,
            "file_path": str(sample_file),
            "batch": batch,
            "condition": condition,
            "timepoint": timepoint,
            "replicate": replicate,
            "treatment_dose": "10uM" if condition != "control" else "vehicle",
            "cell_line": "Jurkat" if "CD3" in str(events.columns) else "Unknown",
            "notes": f"Sample {sample_num} from {batch} with {condition} treatment",
        })

    samplesheet_path = temp_dir / "complex_samplesheet.csv"
    df = pd.DataFrame(samplesheet_data)
    df.to_csv(samplesheet_path, index=False)

    return samplesheet_path


def run_advanced_pipeline(samplesheet: Path, output_dir: Path, config: dict) -> None:
    """Run the complete pipeline with advanced configuration."""
    print("📥 Stage 1: Data Ingestion with advanced configuration")
    stage_ingest(samplesheet, output_dir / "ingest", config)

    print("⚖️ Stage 2: Compensation with auto-detection")
    stage_compensate(output_dir / "ingest", output_dir / "compensate", None)

    print("✅ Stage 3: Advanced Quality Control")
    stage_qc(output_dir / "compensate", output_dir / "qc", config["qc"])

    print("🚪 Stage 4: Custom Gating Strategy")
    stage_gate(output_dir / "qc", output_dir / "gate", "default", config)

    print("📊 Stage 5: Multi-batch Drift Analysis")
    stage_drift(output_dir / "gate", output_dir / "drift", "batch", config)

    print("📈 Stage 6: Complex Statistical Analysis")
    stage_stats(
        output_dir / "gate",
        output_dir / "stats",
        "condition",
        config["channels"]["markers"]
    )


def show_advanced_results(results_dir: Path) -> None:
    """Display detailed results from the advanced pipeline."""
    print("\n📋 Advanced Pipeline Results:")

    # QC Analysis
    qc_summary = results_dir / "qc" / "summary.csv"
    if qc_summary.exists():
        df = pd.read_csv(qc_summary)
        print("✅ Quality Control Summary:")
        print(f"   Samples processed: {len(df)}")
        print(f"   Mean QC pass rate: {df['qc_pass_fraction'].mean()".2%"}")
        print(f"   Samples with high debris (>20%): {(df['debris_fraction'] > 0.2).sum()}")

    # Gating Analysis
    gate_summary = results_dir / "gate" / "summary.csv"
    if gate_summary.exists():
        df = pd.read_csv(gate_summary)
        print("🚪 Gating Summary:")
        print(f"   Total events before gating: {df['input_events'].sum():,","}")
        print(f"   Total events after gating: {df['gated_events'].sum():,","}")
        print(f"   Mean retention rate: {df['gated_events'].sum() / df['input_events'].sum()".2%"}")

    # Drift Analysis
    drift_tests = results_dir / "drift" / "tests.csv"
    if drift_tests.exists():
        df = pd.read_csv(drift_tests)
        significant = (df["adj_p_value"] < 0.05).sum()
        print("📊 Batch Drift Analysis:")
        print(f"   Features tested: {len(df)}")
        print(f"   Significant batch effects: {significant}")

    # Statistical Analysis
    stats_file = results_dir / "stats" / "effect_sizes.csv"
    if stats_file.exists():
        df = pd.read_csv(stats_file)
        print("📈 Statistical Analysis:")
        print(f"   Markers analyzed: {len(df)}")
        print(f"   Significant effects (p<0.05): {(df['adj_p_value'] < 0.05).sum()}")

        # Show top effects
        top_effects = df.nlargest(3, 'effect_size')
        print("   Top 3 effect sizes:")
        for _, row in top_effects.iterrows():
            print(f"     {row['marker']}: {row['effect_size']".3f"}")

    print(f"\n📂 Detailed results in: {results_dir}")


if __name__ == "__main__":
    main()








