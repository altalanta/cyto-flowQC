from __future__ import annotations

import numpy as np
import pandas as pd

from cytoflow_qc.compensate import apply_compensation, get_spillover


def test_get_spillover_prefers_metadata() -> None:
    metadata = {
        "spillover_matrix": [[1.0, 0.05], [0.02, 1.0]],
        "spillover_channels": ["CD3-A", "CD19-A"],
    }
    matrix, channels = get_spillover(metadata)
    assert matrix is not None and channels is not None
    assert matrix.shape == (2, 2)
    assert channels == ["CD3-A", "CD19-A"]


def test_apply_compensation_removes_crosstalk() -> None:
    df = pd.DataFrame(
        {
            "CD3-A": [100.0, 120.0, 80.0],
            "CD19-A": [50.0, 40.0, 45.0],
        }
    )
    spill = np.array([[1.0, 0.2], [0.1, 1.0]])
    channels = ["CD3-A", "CD19-A"]
    compensated = apply_compensation(df, spill, channels)
    assert np.allclose(compensated.mean().values, df.mean().values, atol=50)
    assert not compensated.equals(df)
