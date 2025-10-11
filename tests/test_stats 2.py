from __future__ import annotations

import pandas as pd

from cytoflow_qc.stats import effect_sizes


def test_effect_sizes_returns_adjusted_pvalues() -> None:
    data = pd.DataFrame(
        {
            "condition": ["control"] * 5 + ["treatment"] * 5,
            "CD3-A": [1.0, 1.2, 0.9, 1.1, 1.0, 1.8, 1.7, 1.6, 1.9, 1.8],
            "CD56-A": [0.5, 0.4, 0.55, 0.45, 0.5, 0.9, 0.85, 0.88, 0.92, 0.95],
        }
    )
    results = effect_sizes(data, "condition", ["CD3-A", "CD56-A"])
    assert set(results["parameter"]) == {"CD3-A", "CD56-A"}
    assert (results["adj_p_value"] <= 1).all()
