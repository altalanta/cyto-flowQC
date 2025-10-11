from __future__ import annotations

from cytoflow_qc.gate import auto_gate
from cytoflow_qc.io import read_fcs
from cytoflow_qc.qc import add_qc_flags

CONFIG = {
    "channels": {
        "fsc_a": "FSC-A",
        "ssc_a": "SSC-A",
        "fsc_h": "FSC-H",
        "viability": "Zombie_Aqua-A",
    }
}


def test_auto_gate_reduces_event_count(project_root) -> None:
    events, _ = read_fcs(str(project_root / "sample_data" / "sample_001.csv"))
    qc_df = add_qc_flags(events)
    gated, params = auto_gate(qc_df, config=CONFIG)
    assert len(gated) < len(qc_df)
    assert "debris" in params
    assert "singlets" in params
    assert "lymphocytes" in params
