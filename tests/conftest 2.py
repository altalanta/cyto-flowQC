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
