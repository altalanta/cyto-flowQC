from __future__ import annotations

from typing import Dict

import pandas as pd

from cytoflow_qc.qc import add_qc_flags, qc_summary
from cytoflow_qc.io import read_fcs


def test_add_qc_flags_creates_expected_columns(project_root) -> None:
    events, _ = read_fcs(str(project_root / "sample_data" / "sample_001.csv"))
    qc_df = add_qc_flags(events)
    for col in ["qc_debris", "qc_doublets", "qc_saturated", "qc_pass"]:
        assert col in qc_df
        assert qc_df[col].dtype == bool


def test_qc_summary_returns_metrics(project_root) -> None:
    events, _ = read_fcs(str(project_root / "sample_data" / "sample_001.csv"))
    qc_df = add_qc_flags(events)
    summary = qc_summary({"sample_001": qc_df})
    assert not summary.empty
    assert summary.loc[0, "sample_id"] == "sample_001"
    assert 0 <= summary.loc[0, "qc_pass_fraction"] <= 1
