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

        # Create multi-batch experimental data.
        # This function simulates FCS files and returns a dictionary of dataframes.
        # Each dataframe represents a sample with various channels.
        sample_data = create_multi_batch_data()

        # Save the simulated dataframes as FCS files for ingestion.
        # This step is crucial for simulating the real-world scenario where
        # cytoflow-qc ingests actual FCS files.
        fcs_files_dir = temp_path / "raw_fcs_files"
        fcs_files_dir.mkdir(exist_ok=True)
        create_fcs_files(sample_data, fcs_files_dir)

        # Create samplesheet with complex experimental design.
        # The samplesheet links sample IDs to their corresponding FCS file paths
        # and includes various metadata columns like batch, condition, and timepoint.
        samplesheet = create_complex_samplesheet(sample_data, fcs_files_dir, temp_path)

        print("🔬 Running advanced cytoflow-qc pipeline...")

        # Execute the pipeline with the advanced configuration.
        # This involves sequential execution of ingestion, compensation, QC,
        # gating, drift analysis, and statistical analysis stages.
        run_advanced_pipeline(samplesheet, temp_path, config)

        print("🎉 Advanced pipeline completed!")
        # Display a summary of the results to the console.
        show_advanced_results(temp_path)


def create_fcs_files(sample_data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """
    Save pandas DataFrames as simulated FCS files.

    This function takes a dictionary of DataFrames (where keys are sample IDs)
    and saves each DataFrame as a CSV file in the specified output directory.
    These CSVs simulate FCS files for the purpose of this example.

    Args:
        sample_data (dict[str, pd.DataFrame]): Dictionary where keys are sample IDs
                                                and values are pandas DataFrames
                                                representing flow cytometry events.
        output_dir (Path): The directory where the simulated FCS (CSV) files will be saved.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_id, events_df in sample_data.items():
        # Save each DataFrame as a CSV, simulating an FCS file
        file_path = output_dir / f"{sample_id}.fcs.csv"  # Using .fcs.csv for identification
        events_df.to_csv(file_path, index=False)
        print(f"Generated simulated FCS file: {file_path}")


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
            # Vary conditions and timepoints for each sample
            condition = conditions[sample_counter % len(conditions)]
            timepoint = timepoints[sample_counter % len(timepoints)]

            sample_id = f"{batch_name}_sample_{sample_counter:02d}"

            # Base parameters for event generation, with batch-specific shifts
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

            # Add realistic biological and technical characteristics to the events
            events = add_realistic_characteristics(events, condition, timepoint)

            data[sample_id] = events
            sample_counter += 1

    return data


def add_realistic_characteristics(events: pd.DataFrame, condition: str, timepoint: str) -> pd.DataFrame:
    """
    Add realistic biological and technical characteristics to the simulated data.

    This function modifies the event data to introduce correlations between channels,
    condition-specific marker expression changes, timepoint-dependent effects,
    and some general technical variation.

    Args:
        events (pd.DataFrame): DataFrame of simulated flow cytometry events.
        condition (str): The experimental condition (e.g., "control", "treatment_A").
        timepoint (str): The experimental timepoint (e.g., "T0", "T24").

    Returns:
        pd.DataFrame: Modified DataFrame with added realistic characteristics.
    """
    import numpy as np

    # FSC-A and FSC-H correlation: Simulate some spread around a linear relationship
    events["FSC-H"] = events["FSC-A"] * (1 + np.random.normal(0, 0.05, len(events)))

    # SSC correlations: Introduce a similar correlation for SSC channels
    events["SSC-H"] = events["SSC-A"] * (1 + np.random.normal(0, 0.08, len(events)))

    # Condition-specific effects: Simulate changes in marker expression based on treatment
    if condition == "treatment_A":
        # Treatment A increases T cell markers (CD3, CD4, CD8)
        events["CD3-A"] *= 1.3
        events["CD4-A"] *= 1.2
        events["CD8-A"] *= 1.15

    elif condition == "treatment_B":
        # Treatment B increases NK cell markers (CD56, CD16)
        events["CD56-A"] *= 1.4
        events["CD16-A"] *= 1.25

    # Timepoint effects: Simulate changes in activation markers over time
    if timepoint == "T24":
        # Some activation at 24h (e.g., increase in CD69 expression)
        if "CD69-A" not in events.columns:
            events["CD69-A"] = np.random.lognormal(2.5, 0.8, len(events))
        else:
            events["CD69-A"] *= 1.1

    elif timepoint == "T48":
        # More pronounced activation at 48h
        if "CD69-A" not in events.columns:
            events["CD69-A"] = np.random.lognormal(3.0, 0.8, len(events))
        else:
            events["CD69-A"] *= 1.2

    # Add some technical variation (batch effects/noise) to all marker channels
    for col in events.columns:
        if col.startswith("CD") or col.startswith("SSC") or col.startswith("FSC"):
            # Apply a small amount of technical noise as a multiplicative factor
            technical_noise = np.random.normal(0, 0.02, len(events))
            events[col] = events[col] * (1 + technical_noise)

    return events


def create_complex_samplesheet(
    sample_data: dict[str, pd.DataFrame],
    fcs_files_dir: Path,
    temp_dir: Path
) -> Path:
    """
    Create a complex samplesheet with detailed experimental metadata.

    This function generates a samplesheet DataFrame that includes metadata
    such as sample ID, file path (pointing to the simulated FCS files), batch,
    experimental condition, timepoint, replicate number, and other relevant
    experimental details. The conditions and timepoints are assigned in a
    way that simulates a balanced experimental design.

    Args:
        sample_data (dict[str, pd.DataFrame]): Dictionary of simulated sample data.
        fcs_files_dir (Path): Directory where the simulated FCS files are stored.
        temp_dir (Path): Temporary directory for saving the samplesheet.

    Returns:
        Path: The file path to the generated samplesheet CSV.
    """
    samplesheet_data = []

    # Counters to help assign unique replicate numbers within conditions
    condition_replicate_counter = {"control": 1, "treatment_A": 1, "treatment_B": 1}

    for sample_id, events in sample_data.items():
        # The file path will point to the simulated FCS file (CSV in this case)
        sample_file = fcs_files_dir / f"{sample_id}.fcs.csv"

        # Extract batch name from sample_id (e.g., "batch_1" from "batch_1_sample_01")
        batch = sample_id.split("_")[0]
        # Extract sample number (e.g., 1 from "batch_1_sample_01")
        sample_num = int(sample_id.split("_")[-1])

        # Assign conditions in a balanced, repeating pattern for demonstration
        if sample_num % 3 == 1:
            condition = "control"
        elif sample_num % 3 == 2:
            condition = "treatment_A"
        else:
            condition = "treatment_B"

        # Assign timepoints in a repeating pattern
        timepoint = ["T0", "T24", "T48"][(sample_num - 1) % 3]

        # Get and increment replicate number for the current condition
        replicate = condition_replicate_counter[condition]
        condition_replicate_counter[condition] += 1

        samplesheet_data.append({
            "sample_id": sample_id,
            "file_path": str(sample_file),
            "batch": batch,
            "condition": condition,
            "timepoint": timepoint,
            "replicate": replicate,
            "treatment_dose": "10uM" if condition != "control" else "vehicle",
            "cell_line": "Jurkat" if "CD3-A" in events.columns else "Unknown", # Check for specific marker
            "notes": f"Sample {sample_num} from {batch} with {condition} treatment at {timepoint}",
        })

    # Create a DataFrame from the collected samplesheet data
    df = pd.DataFrame(samplesheet_data)
    samplesheet_path = temp_dir / "complex_samplesheet.csv"
    df.to_csv(samplesheet_path, index=False)

    print(f"Generated samplesheet: {samplesheet_path}")
    print(df.head()) # Display first few rows of the samplesheet

    return samplesheet_path


def run_advanced_pipeline(samplesheet: Path, output_dir: Path, config: dict) -> None:
    """
    Execute the complete cytoflow-qc pipeline with the advanced configuration.

    This function orchestrates the sequential execution of various `cytoflow-qc`
    stages, passing the output of one stage as the input to the next.
    It also applies the provided configuration at each relevant stage.

    Args:
        samplesheet (Path): Path to the generated samplesheet CSV.
        output_dir (Path): The root output directory for all pipeline artifacts.
        config (dict): The advanced configuration dictionary for the pipeline.
    """
    # Define paths for intermediate and final outputs
    ingest_dir = output_dir / "ingest"
    compensate_dir = output_dir / "compensate"
    qc_dir = output_dir / "qc"
    gate_dir = output_dir / "gate"
    drift_dir = output_dir / "drift"
    stats_dir = output_dir / "stats"
    report_path = output_dir / "report.html"

    print("\n📥 Stage 1: Data Ingestion with advanced configuration")
    # Ingest raw FCS data, convert to standardized format (Parquet), and extract metadata.
    # The full configuration is passed to capture any channel mapping settings.
    stage_ingest(samplesheet, ingest_dir, config)

    print("\n⚖️ Stage 2: Compensation with auto-detection")
    # Apply compensation to correct for spectral overlap.
    # Here, spill is set to None, implying auto-detection from FCS metadata or default.
    stage_compensate(ingest_dir, compensate_dir, None)

    print("\n✅ Stage 3: Advanced Quality Control")
    # Apply quality control flags based on configured QC parameters (debris, doublets, etc.).
    # The QC-specific part of the configuration is passed.
    stage_qc(compensate_dir, qc_dir, config["qc"])

    print("\n🚪 Stage 4: Custom Gating Strategy")
    # Apply gating strategies to identify cell populations.
    # "default" strategy is used, but custom strategies can be specified and configured.
    stage_gate(qc_dir, gate_dir, "default", config)

    print("\n📊 Stage 5: Multi-batch Drift Analysis")
    # Perform analysis to detect and quantify batch effects or drift over time.
    # Uses the "batch" column from the samplesheet for grouping.
    stage_drift(gate_dir, drift_dir, "batch", config)

    print("\n📈 Stage 6: Complex Statistical Analysis")
    # Calculate effect sizes and other statistics on gated populations.
    # Groups by "condition" and uses the specified marker channels.
    markers = config["channels"]["markers"]
    stage_stats(gate_dir, stats_dir, "condition", markers)

    print("\n📝 Stage 7: Generate Comprehensive Report")
    # Build the final HTML report aggregating all results and visualizations.
    # Uses the default report template or a custom one specified in config.
    # Assuming build_report is defined elsewhere or will be added.
    # For now, we'll just print a placeholder message.
    print(f"Report generation skipped. Placeholder for build_report function.")
    # Example of how build_report might be called:
    # from jinja2 import Environment, FileSystemLoader
    # from pathlib import Path
    #
    # def build_report(output_dir: str, template_path: str, report_path: str) -> None:
    #     env = Environment(loader=FileSystemLoader(template_path))
    #     template = env.get_template("report.html.j2")
    #     report_content = template.render(
    #         output_dir=output_dir,
    #         qc_summary=results_dir / "qc" / "summary.csv",
    #         gate_summary=results_dir / "gate" / "summary.csv",
    #         drift_tests=results_dir / "drift" / "tests.csv",
    #         effect_sizes=results_dir / "stats" / "effect_sizes.csv",
    #         # Add other relevant data for the report
    #     )
    #     with open(report_path, "w") as f:
    #         f.write(report_content)
    #     print(f"Generated report: {report_path}")


def show_advanced_results(results_dir: Path) -> None:
    """
    Display detailed results from the advanced pipeline execution.

    This function reads key summary CSVs generated by the pipeline and prints
    relevant statistics to the console, providing a quick overview of the results.

    Args:
        results_dir (Path): The root directory containing all pipeline results.
    """
    print("\n\n📋 Advanced Pipeline Results Summary:")

    # --- QC Analysis Summary ---
    qc_summary_file = results_dir / "qc" / "summary.csv"
    if qc_summary_file.exists():
        df = pd.read_csv(qc_summary_file)
        print("\n  ✅ Quality Control Summary:")
        print(f"     - Samples processed: {len(df)}")
        print(f"     - Mean QC pass rate: {df['qc_pass_fraction'].mean():.2%}")
        # Example of deeper insight: samples with high debris
        print(f"     - Samples with high debris (>20%): {(df['debris_fraction'] > 0.2).sum()}")

    # Gating Analysis
    gate_summary = results_dir / "gate" / "summary.csv"
    if gate_summary.exists():
        df = pd.read_csv(gate_summary)
        print("🚪 Gating Summary:")
        print(f"   Total events before gating: {df['input_events'].sum():,","}")
        print(f"   Total events after gating: {df['gated_events'].sum():,","}")
        print(f"   Mean retention rate: {df['gated_events'].sum() / df['input_events'].sum():.2%}")

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
            print(f"     {row['marker']}: {row['effect_size']:.3f}")

    print(f"\n📂 Detailed results in: {results_dir}")


if __name__ == "__main__":
    main()











