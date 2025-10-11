from __future__ import annotations

import pandas as pd

from cytoflow_qc.drift import compute_batch_drift, extract_sample_features
from cytoflow_qc.gate import auto_gate
from cytoflow_qc.io import load_samplesheet, read_fcs
from cytoflow_qc.qc import add_qc_flags

GATE_CONFIG = {
    "channels": {
        "fsc_a": "FSC-A",
        "ssc_a": "SSC-A",
        "fsc_h": "FSC-H",
        "viability": "Zombie_Aqua-A",
    }
}


def test_extract_features_and_compute_drift(samplesheet_path) -> None:
    sheet = load_samplesheet(str(samplesheet_path))
    sample_tables = {}
    metadata_rows = []
    for row in sheet.to_dict(orient="records"):
        events, _ = read_fcs(row["file_path"])
        qc_df = add_qc_flags(events)
        gated, _ = auto_gate(qc_df, config=GATE_CONFIG)
        sample_tables[row["sample_id"]] = gated
        metadata_rows.append(
            {"sample_id": row["sample_id"], "batch": row["batch"], "condition": row["condition"]}
        )

    metadata = pd.DataFrame(metadata_rows)
    features = extract_sample_features(sample_tables, metadata)
    assert "sample_id" in features.columns
    assert features["sample_id"].nunique() == len(sample_tables)

    drift = compute_batch_drift(features, by="batch")
    assert "tests" in drift
    assert not drift["pca"].empty
    assert isinstance(drift["drift_detected"], bool)
