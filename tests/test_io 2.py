from __future__ import annotations

from pathlib import Path

import pandas as pd

from cytoflow_qc.io import load_samplesheet, read_fcs, standardize_channels


def test_load_samplesheet_resolves_paths(samplesheet_path: Path) -> None:
    df = load_samplesheet(str(samplesheet_path))
    assert len(df) == 4
    assert df["missing_file"].sum() == 0
    assert Path(df.loc[0, "file_path"]).is_absolute()


def test_read_fcs_csv(project_root: Path) -> None:
    data_path = project_root / "sample_data" / "sample_001.csv"
    events, metadata = read_fcs(str(data_path))
    assert not events.empty
    assert metadata["n_events"] == len(events)
    assert "channels" in metadata


def test_standardize_channels(sample_event_table: pd.DataFrame) -> None:
    metadata = {"channels": list(sample_event_table.columns)}
    channel_map = {"fsc_a": "FSC-A", "ssc_a": "SSC-A"}
    renamed = standardize_channels(sample_event_table, metadata, channel_map)
    assert "fsc_a" in renamed.columns
    assert "ssc_a" in renamed.columns
    assert metadata["channels"][0] == renamed.columns[0]
