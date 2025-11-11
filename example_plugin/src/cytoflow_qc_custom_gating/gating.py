"""A custom gating strategy example for CytoFlow-QC."""
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from cytoflow_qc.plugins.base import GatingStrategyPlugin

class CustomDensityGating(GatingStrategyPlugin):
    """A custom gating strategy that uses a density-based method."""

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "fsc_channel": "FSC-A",
            "ssc_channel": "SSC-A",
            "percentile": 95,
        }

    def gate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        fsc_col = self.config["fsc_channel"]
        ssc_col = self.config["ssc_channel"]
        percentile = self.config["percentile"]

        if fsc_col not in df.columns or ssc_col not in df.columns:
            raise ValueError(f"Required channels for gating not found: {fsc_col}, {ssc_col}")

        sample_df = df if len(df) < 50000 else df.sample(n=50000)
        
        kde = gaussian_kde(sample_df[[fsc_col, ssc_col]].T)
        density = kde(df[[fsc_col, ssc_col]].T)
        density_threshold = np.percentile(density, 100 - percentile)
        
        df["gate_custom_density"] = density > density_threshold
        
        gated_df = df[df["gate_custom_density"]].copy()
        params = {"custom_density_threshold": density_threshold}
        
        return gated_df, params





