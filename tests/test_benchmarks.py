"""Performance benchmarks for cytoflow-qc pipeline stages."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytoflow_qc.cli import stage_ingest, stage_compensate, stage_qc, stage_gate, stage_drift, stage_stats


class TestPipelinePerformance:
    """Benchmark performance of each pipeline stage."""

    @pytest.fixture
    def large_sample_data(self) -> dict[str, pd.DataFrame]:
        """Create large synthetic dataset for performance testing."""
        n_events = 50000  # Large dataset for benchmarking

        data = {}
        for sample_id in [f"sample_{i"03d"}" for i in range(1, 11)]:  # 10 samples
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

            data[sample_id] = events

        return data

    @pytest.fixture
    def samplesheet_path(self, large_sample_data: dict[str, pd.DataFrame], tmp_path: Path) -> Path:
        """Create samplesheet for the large dataset."""
        samplesheet_data = []
        for sample_id, events in large_sample_data.items():
            sample_file = tmp_path / f"{sample_id}.csv"
            events.to_csv(sample_file, index=False)

            samplesheet_data.append({
                "sample_id": sample_id,
                "file_path": str(sample_file),
                "batch": f"batch_{int(sample_id.split('_')[1]) // 5 + 1}",
                "condition": "control" if sample_id.endswith("1") or sample_id.endswith("2") else "treatment",
            })

        samplesheet_path = tmp_path / "samplesheet.csv"
        pd.DataFrame(samplesheet_data).to_csv(samplesheet_path, index=False)

        return samplesheet_path

    def test_ingest_performance(self, benchmark, samplesheet_path: Path, tmp_path: Path) -> None:
        """Benchmark the ingest stage performance."""
        config = {
            "channels": {
                "fsc_a": "FSC-A",
                "fsc_h": "FSC-H",
                "ssc_a": "SSC-A",
                "markers": ["CD3-A", "CD19-A", "CD56-A", "CD4-A", "CD8-A"],
            }
        }

        def run_ingest() -> None:
            stage_ingest(samplesheet_path, tmp_path / "ingest", config)

        # Run benchmark
        result = benchmark(run_ingest)

        # Verify output exists
        assert (tmp_path / "ingest" / "manifest.csv").exists()
        assert (tmp_path / "ingest" / "events").exists()

        # Check that it completed in reasonable time (adjust based on system)
        # This is a basic sanity check - actual thresholds would depend on system specs
        assert result < 60  # Should complete in under 60 seconds

    def test_qc_performance(self, benchmark, samplesheet_path: Path, tmp_path: Path) -> None:
        """Benchmark the QC stage performance."""
        # First run ingest to get data
        config = {
            "channels": {
                "fsc_a": "FSC-A",
                "fsc_h": "FSC-H",
                "ssc_a": "SSC-A",
                "markers": ["CD3-A", "CD19-A", "CD56-A", "CD4-A", "CD8-A"],
            }
        }

        stage_ingest(samplesheet_path, tmp_path / "ingest", config)

        def run_qc() -> None:
            stage_qc(tmp_path / "ingest", tmp_path / "qc", config["qc"])

        result = benchmark(run_qc)

        # Verify output
        assert (tmp_path / "qc" / "summary.csv").exists()

        # Should complete quickly for QC operations
        assert result < 30

    def test_gate_performance(self, benchmark, samplesheet_path: Path, tmp_path: Path) -> None:
        """Benchmark the gating stage performance."""
        # Run ingest and QC first
        config = {
            "channels": {
                "fsc_a": "FSC-A",
                "fsc_h": "FSC-H",
                "ssc_a": "SSC-A",
                "markers": ["CD3-A", "CD19-A", "CD56-A", "CD4-A", "CD8-A"],
            }
        }

        stage_ingest(samplesheet_path, tmp_path / "ingest", config)
        stage_qc(tmp_path / "ingest", tmp_path / "qc", config["qc"])

        def run_gate() -> None:
            stage_gate(tmp_path / "qc", tmp_path / "gate", "default", config)

        result = benchmark(run_gate)

        # Verify output
        assert (tmp_path / "gate" / "summary.csv").exists()

        # Gating should be reasonably fast
        assert result < 45

    def test_drift_performance(self, benchmark, samplesheet_path: Path, tmp_path: Path) -> None:
        """Benchmark the drift analysis stage performance."""
        # Run full pipeline up to gating
        config = {
            "channels": {
                "fsc_a": "FSC-A",
                "fsc_h": "FSC-H",
                "ssc_a": "SSC-A",
                "markers": ["CD3-A", "CD19-A", "CD56-A", "CD4-A", "CD8-A"],
            }
        }

        stage_ingest(samplesheet_path, tmp_path / "ingest", config)
        stage_qc(tmp_path / "ingest", tmp_path / "qc", config["qc"])
        stage_gate(tmp_path / "qc", tmp_path / "gate", "default", config)

        def run_drift() -> None:
            stage_drift(tmp_path / "gate", tmp_path / "drift", "batch", config)

        result = benchmark(run_drift)

        # Verify output
        assert (tmp_path / "drift" / "features.csv").exists()
        assert (tmp_path / "drift" / "tests.csv").exists()

        # Drift analysis should complete in reasonable time
        assert result < 60

    def test_stats_performance(self, benchmark, samplesheet_path: Path, tmp_path: Path) -> None:
        """Benchmark the statistics stage performance."""
        # Run full pipeline up to gating
        config = {
            "channels": {
                "fsc_a": "FSC-A",
                "fsc_h": "FSC-H",
                "ssc_a": "SSC-A",
                "markers": ["CD3-A", "CD19-A", "CD56-A", "CD4-A", "CD8-A"],
            }
        }

        stage_ingest(samplesheet_path, tmp_path / "ingest", config)
        stage_qc(tmp_path / "ingest", tmp_path / "qc", config["qc"])
        stage_gate(tmp_path / "qc", tmp_path / "gate", "default", config)

        def run_stats() -> None:
            stage_stats(tmp_path / "gate", tmp_path / "stats", "condition", ["CD3-A", "CD19-A"])

        result = benchmark(run_stats)

        # Verify output
        assert (tmp_path / "stats" / "effect_sizes.csv").exists()

        # Statistics should be fast
        assert result < 15


class TestMemoryUsage:
    """Test memory usage patterns for large datasets."""

    def test_memory_efficiency_with_large_dataset(self, samplesheet_path: Path, tmp_path: Path) -> None:
        """Test that pipeline doesn't have memory leaks with large datasets."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run full pipeline
        config = {
            "channels": {
                "fsc_a": "FSC-A",
                "fsc_h": "FSC-H",
                "ssc_a": "SSC-A",
                "markers": ["CD3-A", "CD19-A", "CD56-A", "CD4-A", "CD8-A"],
            }
        }

        stage_ingest(samplesheet_path, tmp_path / "ingest", config)
        stage_qc(tmp_path / "ingest", tmp_path / "qc", config["qc"])
        stage_gate(tmp_path / "qc", tmp_path / "gate", "default", config)
        stage_drift(tmp_path / "gate", tmp_path / "drift", "batch", config)
        stage_stats(tmp_path / "gate", tmp_path / "stats", "condition", ["CD3-A", "CD19-A"])

        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 1GB for this dataset)
        # This is a rough check - actual limits would depend on system
        assert memory_increase < 1000  # MB

        # Force garbage collection and check memory is released
        import gc
        gc.collect()
        post_gc_memory = process.memory_info().rss / 1024 / 1024

        # Memory should decrease after GC (allowing for some overhead)
        assert post_gc_memory < final_memory + 100  # Allow 100MB overhead


class TestScalability:
    """Test how pipeline performance scales with dataset size."""

    @pytest.mark.parametrize("n_samples", [1, 5, 10, 20])
    def test_scalability_with_sample_count(self, benchmark, n_samples: int, tmp_path: Path) -> None:
        """Test how performance scales with number of samples."""
        # Create dataset with varying sample counts
        n_events_per_sample = 1000

        data = {}
        for i in range(n_samples):
            sample_id = f"sample_{i"03d"}"
            events = pd.DataFrame({
                "FSC-A": np.random.lognormal(6, 0.5, n_events_per_sample),
                "FSC-H": np.random.lognormal(6, 0.5, n_events_per_sample),
                "SSC-A": np.random.lognormal(5, 0.5, n_events_per_sample),
                "CD3-A": np.random.lognormal(4, 0.8, n_events_per_sample),
            })
            data[sample_id] = events

        # Create samplesheet
        samplesheet_data = []
        for sample_id, events in data.items():
            sample_file = tmp_path / f"{sample_id}.csv"
            events.to_csv(sample_file, index=False)

            samplesheet_data.append({
                "sample_id": sample_id,
                "file_path": str(sample_file),
                "batch": "batch_1",
                "condition": "control",
            })

        samplesheet_path = tmp_path / "samplesheet.csv"
        pd.DataFrame(samplesheet_data).to_csv(samplesheet_path, index=False)

        # Run full pipeline
        config = {
            "channels": {
                "fsc_a": "FSC-A",
                "fsc_h": "FSC-H",
                "ssc_a": "SSC-A",
                "markers": ["CD3-A"],
            }
        }

        def run_pipeline() -> None:
            stage_ingest(samplesheet_path, tmp_path / "ingest", config)
            stage_qc(tmp_path / "ingest", tmp_path / "qc", config["qc"])
            stage_gate(tmp_path / "qc", tmp_path / "gate", "default", config)
            stage_drift(tmp_path / "gate", tmp_path / "drift", "batch", config)
            stage_stats(tmp_path / "gate", tmp_path / "stats", "condition", ["CD3-A"])

        result = benchmark(run_pipeline)

        # Verify all outputs exist
        assert (tmp_path / "ingest" / "manifest.csv").exists()
        assert (tmp_path / "qc" / "summary.csv").exists()
        assert (tmp_path / "gate" / "summary.csv").exists()
        assert (tmp_path / "drift" / "features.csv").exists()
        assert (tmp_path / "stats" / "effect_sizes.csv").exists()

        # Return the result for potential analysis
        return result








