"""Example custom gating strategy plugin."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..gating import GatingStrategyPlugin


class CustomGatingExample(GatingStrategyPlugin):
    """Example custom gating strategy that demonstrates the plugin interface.

    This plugin implements a simple gating strategy that:
    1. Uses FSC-A and SSC-A to identify main cell populations
    2. Applies a custom density-based filtering
    3. Demonstrates proper parameter validation and error handling
    """

    PLUGIN_NAME = "custom_gating_example"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Example custom gating strategy for demonstration"
    PLUGIN_AUTHOR = "CytoFlow-QC Team"
    PLUGIN_EMAIL = "cytoflow-qc@example.com"

    REQUIRES_CYTOFLOW_VERSION = ">=0.1.0"

    def validate_gate_parameters(self, channels: dict[str, str]) -> None:
        """Validate that required channels are available."""
        required_channels = ["fsc_a", "ssc_a"]
        for channel in required_channels:
            if channel not in channels:
                raise ValueError(f"Required channel '{channel}' not found")

        # Validate custom parameters
        density_threshold = self.config.get("density_threshold", 0.1)
        if not 0 < density_threshold < 1:
            raise ValueError("density_threshold must be between 0 and 1")

    def get_gate_description(self) -> str:
        """Get human-readable description of this gating strategy."""
        return (
            "Custom gating strategy that uses density-based filtering "
            "to identify main cell populations based on FSC-A and SSC-A parameters"
        )

    def get_required_parameters(self) -> list[str]:
        """Get list of required configuration parameters."""
        return ["channels"]

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration parameters."""
        return {
            "density_threshold": 0.1,
            "min_events_per_gate": 100,
            "use_adaptive_threshold": True,
            "outlier_removal": True,
        }

    def apply_gate(
        self,
        data: pd.DataFrame,
        channels: dict[str, str],
        **kwargs
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply custom gating strategy to flow cytometry data.

        Args:
            data: Input DataFrame with flow cytometry events
            channels: Dictionary mapping canonical channel names to data columns
            **kwargs: Additional gating parameters

        Returns:
            Tuple of (gated_data, gating_parameters)
        """
        # Validate parameters
        self.validate_gate_parameters(channels)

        # Extract channel names
        fsc_channel = channels.get("fsc_a")
        ssc_channel = channels.get("ssc_a")

        if not fsc_channel or not ssc_channel:
            raise ValueError("FSC-A and SSC-A channels required for custom gating")

        if fsc_channel not in data.columns or ssc_channel not in data.columns:
            raise ValueError("Required channels not found in data")

        # Get configuration
        density_threshold = self.config.get("density_threshold", 0.1)
        min_events = self.config.get("min_events_per_gate", 100)
        use_adaptive = self.config.get("use_adaptive_threshold", True)
        remove_outliers = self.config.get("outlier_removal", True)

        # Apply outlier removal if requested
        if remove_outliers:
            data = self._remove_outliers(data, fsc_channel, ssc_channel)

        # Calculate density-based gating
        gated_data, gating_params = self._apply_density_gating(
            data, fsc_channel, ssc_channel, density_threshold, use_adaptive
        )

        # Ensure minimum events per gate
        if len(gated_data) < min_events:
            # Fall back to simple percentile-based gating
            gated_data, gating_params = self._apply_percentile_gating(
                data, fsc_channel, ssc_channel
            )

        # Add metadata
        gating_params.update({
            "method": "custom_density",
            "density_threshold": density_threshold,
            "adaptive_threshold": use_adaptive,
            "outlier_removal": remove_outliers,
            "total_events": len(data),
            "gated_events": len(gated_data),
        })

        return gated_data, gating_params

    def _remove_outliers(self, data: pd.DataFrame, fsc_col: str, ssc_col: str) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        # Calculate IQR for each channel
        fsc_q1, fsc_q3 = data[fsc_col].quantile([0.25, 0.75])
        ssc_q1, ssc_q3 = data[ssc_col].quantile([0.25, 0.75])

        fsc_iqr = fsc_q3 - fsc_q1
        ssc_iqr = ssc_q3 - ssc_q1

        # Define bounds
        fsc_lower = fsc_q1 - 1.5 * fsc_iqr
        fsc_upper = fsc_q3 + 1.5 * fsc_iqr
        ssc_lower = ssc_q1 - 1.5 * ssc_iqr
        ssc_upper = ssc_q3 + 1.5 * ssc_iqr

        # Filter data
        mask = (
            (data[fsc_col] >= fsc_lower) &
            (data[fsc_col] <= fsc_upper) &
            (data[ssc_col] >= ssc_lower) &
            (data[ssc_col] <= ssc_upper)
        )

        return data[mask].copy()

    def _apply_density_gating(
        self,
        data: pd.DataFrame,
        fsc_col: str,
        ssc_col: str,
        density_threshold: float,
        use_adaptive: bool
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply density-based gating strategy."""
        # Create 2D histogram to estimate density
        hist, x_edges, y_edges = np.histogram2d(
            data[fsc_col], data[ssc_col],
            bins=50, density=True
        )

        # Find regions with high density
        if use_adaptive:
            # Use adaptive threshold based on data distribution
            density_threshold = self._calculate_adaptive_threshold(hist, density_threshold)

        # Create mask for high-density regions
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # Simple density-based gating: keep points in high-density regions
        mask = np.zeros(len(data), dtype=bool)

        for i in range(len(x_centers)):
            for j in range(len(y_centers)):
                if hist[i, j] >= density_threshold:
                    # Find points in this bin
                    x_mask = (data[fsc_col] >= x_edges[i]) & (data[fsc_col] < x_edges[i + 1])
                    y_mask = (data[ssc_col] >= y_edges[j]) & (data[ssc_col] < y_edges[j + 1])
                    mask |= (x_mask & y_mask)

        gated_data = data[mask].copy()

        gating_params = {
            "density_threshold_used": density_threshold,
            "high_density_bins": int((hist >= density_threshold).sum()),
            "total_bins": hist.size,
        }

        return gated_data, gating_params

    def _apply_percentile_gating(
        self,
        data: pd.DataFrame,
        fsc_col: str,
        ssc_col: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply simple percentile-based gating as fallback."""
        # Use 10th-90th percentile for gating
        fsc_lower = data[fsc_col].quantile(0.1)
        fsc_upper = data[fsc_col].quantile(0.9)
        ssc_lower = data[ssc_col].quantile(0.1)
        ssc_upper = data[ssc_col].quantile(0.9)

        mask = (
            (data[fsc_col] >= fsc_lower) &
            (data[fsc_col] <= fsc_upper) &
            (data[ssc_col] >= ssc_lower) &
            (data[ssc_col] <= ssc_upper)
        )

        gated_data = data[mask].copy()

        gating_params = {
            "method": "percentile_fallback",
            "fsc_percentile_range": (0.1, 0.9),
            "ssc_percentile_range": (0.1, 0.9),
        }

        return gated_data, gating_params

    def _calculate_adaptive_threshold(self, hist: np.ndarray, base_threshold: float) -> float:
        """Calculate adaptive density threshold based on data distribution."""
        # Use the median density as adaptive threshold
        median_density = np.median(hist[hist > 0])

        # Adaptive threshold is base_threshold * median_density
        # This ensures we adapt to the overall density level
        adaptive_threshold = base_threshold * median_density

        # Don't let adaptive threshold go too low
        min_threshold = np.percentile(hist[hist > 0], 25)
        adaptive_threshold = max(adaptive_threshold, min_threshold)

        return adaptive_threshold










