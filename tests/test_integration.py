"""Integration tests for the complete cytoflow-qc pipeline."""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from cytoflow_qc.cli import stage_ingest, stage_compensate, stage_qc, stage_gate, stage_drift, stage_stats


class TestFullPipelineIntegration:
    """Test the complete pipeline from ingest to report generation."""

    def setup_method(self) -> None:
        """Set up temporary directories for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            "channels": {
                "fsc_a": "FSC-A",
                "fsc_h": "FSC-H",
                "ssc_a": "SSC-A",
                "markers": ["CD3-A", "CD19-A", "CD56-A"],
            },
            "qc": {
                "debris": {"fsc_percentile": 5.0, "ssc_percentile": 5.0},
                "doublets": {"tolerance": 0.1},
                "saturation": {"threshold": 0.99},
            },
        }

    def teardown_method(self) -> None:
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_pipeline_execution(self) -> None:
        """Test complete pipeline execution with sample data."""
        # Create sample data
        sample_data = self._create_sample_data()

        # Stage 1: Ingest
        ingest_dir = self.temp_dir / "ingest"
        samplesheet = self._create_samplesheet(sample_data)
        stage_ingest(samplesheet, ingest_dir, self.config)

        # Verify ingest output
        assert (ingest_dir / "manifest.csv").exists()
        assert (ingest_dir / "events").exists()
        assert (ingest_dir / "metadata").exists()

        # Stage 2: Compensate
        compensate_dir = self.temp_dir / "compensate"
        stage_compensate(ingest_dir, compensate_dir, None)

        # Verify compensate output
        assert (compensate_dir / "manifest.csv").exists()
        assert (compensate_dir / "events").exists()

        # Stage 3: QC
        qc_dir = self.temp_dir / "qc"
        stage_qc(compensate_dir, qc_dir, self.config["qc"])

        # Verify QC output
        assert (qc_dir / "manifest.csv").exists()
        assert (qc_dir / "events").exists()
        assert (qc_dir / "summary.csv").exists()

        # Stage 4: Gate
        gate_dir = self.temp_dir / "gate"
        stage_gate(qc_dir, gate_dir, "default", self.config)

        # Verify gate output
        assert (gate_dir / "manifest.csv").exists()
        assert (gate_dir / "events").exists()
        assert (gate_dir / "params").exists()
        assert (gate_dir / "summary.csv").exists()

        # Stage 5: Drift analysis
        drift_dir = self.temp_dir / "drift"
        stage_drift(gate_dir, drift_dir, "batch", self.config)

        # Verify drift output
        assert (drift_dir / "features.csv").exists()
        assert (drift_dir / "tests.csv").exists()
        assert (drift_dir / "pca.csv").exists()

        # Stage 6: Statistics
        stats_dir = self.temp_dir / "stats"
        stage_stats(gate_dir, stats_dir, "condition", ["CD3-A", "CD19-A"])

        # Verify stats output
        assert (stats_dir / "effect_sizes.csv").exists()

    def test_pipeline_error_handling(self) -> None:
        """Test pipeline behavior with problematic data."""
        # Create problematic sample data (empty file)
        problematic_data = self._create_problematic_sample_data()

        # Test should handle errors gracefully
        samplesheet = self._create_samplesheet(problematic_data)
        ingest_dir = self.temp_dir / "ingest"

        # Should not raise exception but handle errors appropriately
        stage_ingest(samplesheet, ingest_dir, self.config)

        # Pipeline should continue with valid samples
        manifest = pd.read_csv(ingest_dir / "manifest.csv")
        assert len(manifest) > 0  # Should have at least one valid sample

    def _create_sample_data(self) -> dict[str, pd.DataFrame]:
        """Create sample FCS-like data for testing."""
        import numpy as np

        # Create synthetic flow cytometry data
        n_events = 1000
        data = {}

        for sample_id in ["sample_001", "sample_002", "sample_003"]:
            # Generate synthetic FCS data
            events = pd.DataFrame({
                "FSC-A": np.random.lognormal(6, 0.5, n_events),
                "FSC-H": np.random.lognormal(6, 0.5, n_events),
                "SSC-A": np.random.lognormal(5, 0.5, n_events),
                "CD3-A": np.random.lognormal(4, 0.8, n_events),
                "CD19-A": np.random.lognormal(3, 0.8, n_events),
                "CD56-A": np.random.lognormal(3.5, 0.8, n_events),
            })

            # Add some realistic noise and patterns
            events["FSC-H"] = events["FSC-A"] * (1 + np.random.normal(0, 0.05, n_events))
            events["CD3-A"] = events["CD3-A"] * (1 + np.random.normal(0, 0.1, n_events))

            data[sample_id] = events

        return data

    def _create_problematic_sample_data(self) -> dict[str, pd.DataFrame]:
        """Create problematic sample data for error testing."""
        import numpy as np

        data = {}
        # Create one valid sample
        data["sample_001"] = pd.DataFrame({
            "FSC-A": np.random.lognormal(6, 0.5, 100),
            "FSC-H": np.random.lognormal(6, 0.5, 100),
            "SSC-A": np.random.lognormal(5, 0.5, 100),
            "CD3-A": np.random.lognormal(4, 0.8, 100),
        })

        # Create empty sample (should be handled gracefully)
        data["sample_002"] = pd.DataFrame()

        return data

    def _create_samplesheet(self, sample_data: dict[str, pd.DataFrame]) -> Path:
        """Create a samplesheet CSV for the test data."""
        samplesheet_data = []
        for sample_id, events in sample_data.items():
            # Create temporary CSV file for each sample
            sample_file = self.temp_dir / f"{sample_id}.csv"
            events.to_csv(sample_file, index=False)

            samplesheet_data.append({
                "sample_id": sample_id,
                "file_path": str(sample_file),
                "batch": "batch_1" if sample_id in ["sample_001", "sample_002"] else "batch_2",
                "condition": "control" if sample_id == "sample_001" else "treatment",
            })

        samplesheet_path = self.temp_dir / "samplesheet.csv"
        pd.DataFrame(samplesheet_data).to_csv(samplesheet_path, index=False)

        return samplesheet_path






