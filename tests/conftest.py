from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def samplesheet_path(project_root: Path) -> Path:
    return project_root / "samplesheets" / "example_samplesheet.csv"


@pytest.fixture(scope="session")
def config_path(project_root: Path) -> Path:
    return project_root / "configs" / "example_config.yaml"


@pytest.fixture(scope="session")
def sample_event_table(project_root: Path) -> pd.DataFrame:
    sample_file = project_root / "sample_data" / "sample_001.csv"
    return pd.read_csv(sample_file)


@pytest.fixture(scope="session")
def pipeline_results(tmpdir_factory):
    """Run the pipeline on sample data and return a dictionary of result paths."""
    tmp_path = Path(tmpdir_factory.mktemp("pipeline_run"))
    root_path = Path(__file__).parent.parent

    samplesheet = root_path / "samplesheets" / "example_samplesheet.csv"
    config_file = root_path / "configs" / "example_config.yaml"

    # Manually run pipeline stages
    cfg = load_and_validate_config(config_file)

    ingest_dir = tmp_path / "ingest"
    compensate_dir = tmp_path / "compensate"
    qc_dir = tmp_path / "qc"
    gate_dir = tmp_path / "gate"
    drift_dir = tmp_path / "drift"
    stats_dir = tmp_path / "stats"

    stage_ingest(samplesheet, ingest_dir, cfg)
    stage_compensate(ingest_dir, compensate_dir, None, 1)
    stage_qc(compensate_dir, qc_dir, cfg.qc, 1)
    stage_gate(qc_dir, gate_dir, "default", cfg, 1)
    stage_drift(gate_dir, drift_dir, "batch", cfg)
    markers = _resolve_marker_columns(None, cfg)
    stage_stats(gate_dir, stats_dir, "condition", markers)

    return {
        "qc_summary": pd.read_csv(qc_dir / "summary.csv"),
        "gated_events": load_dataframe(next((gate_dir / "events").glob("*.parquet"))),
        "unfiltered_events": load_dataframe(next((qc_dir / "events").glob("*.parquet"))),
        "pca": pd.read_csv(drift_dir / "pca.csv"),
        "umap": pd.read_csv(drift_dir / "umap.csv"),
        "effects": pd.read_csv(stats_dir / "effect_sizes.csv"),
        "config": cfg,
    }
