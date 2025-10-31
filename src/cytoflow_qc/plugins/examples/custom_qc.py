"""Example custom quality control method plugin."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..qc import QCMethodPlugin


class CustomQCMethod(QCMethodPlugin):
    """Example custom quality control method that demonstrates the plugin interface.

    This plugin implements a simple machine learning-based QC method that:
    1. Uses Isolation Forest to detect anomalous events (outliers).
    2. Flags samples with a high percentage of outliers.
    3. Demonstrates proper parameter validation and error handling.
    """

    PLUGIN_NAME = "custom_qc_method"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Machine learning-based QC method using Isolation Forest"
    PLUGIN_AUTHOR = "CytoFlow-QC Team"
    PLUGIN_EMAIL = "cytoflow-qc@example.com"

    REQUIRES_CYTOFLOW_VERSION = ">=0.1.0"

    def validate_qc_parameters(self, channels: dict[str, str]) -> None:
        """Validate that required channels are available and config parameters are valid."""
        required_channels = self.config.get("features", [])
        if not required_channels:
            raise ValueError("At least one feature channel must be specified in config.features")

        for channel_key in required_channels:
            if channel_key not in channels:
                raise ValueError(f"Required channel '{channel_key}' not found in provided channels.")

        contamination = self.config.get("contamination", 0.05)
        if not 0 < contamination < 0.5:
            raise ValueError("Contamination parameter must be between 0 and 0.5")

        n_estimators = self.config.get("n_estimators", 100)
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")

    def get_qc_description(self) -> str:
        """Get human-readable description of this QC method."""
        return (
            "Custom QC method using Isolation Forest to identify and flag anomalous events "
            "within samples. Samples with a high proportion of anomalous events are marked as low quality."
        )

    def get_required_parameters(self) -> list[str]:
        """Get list of required configuration parameters."""
        return ["features"]

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration parameters."""
        return {
            "features": [],  # List of canonical channel names to use as features
            "contamination": 0.05,  # Expected proportion of outliers in the data
            "n_estimators": 100,  # Number of base estimators in Isolation Forest
            "max_samples": "auto",  # The number of samples to draw from X to train each base estimator
            "random_state": 42,
            "min_outlier_fraction": 0.1,  # Minimum fraction of outliers to flag a sample
        }

    def apply_qc(
        self,
        data: pd.DataFrame,
        channels: dict[str, str],
        **kwargs
    ) -> pd.DataFrame:
        """Apply custom QC method to flow cytometry data.

        Args:
            data: Input DataFrame with flow cytometry events.
            channels: Dictionary mapping canonical channel names to data columns.
            **kwargs: Additional QC parameters.

        Returns:
            DataFrame with an additional 'qc_flag_custom_qc_method' column.
        """
        self.validate_qc_parameters(channels)

        feature_keys = self.config.get("features", [])
        if not feature_keys:
            # Fallback or raise error if no features specified despite validation
            return data.assign(qc_flag_custom_qc_method=False)

        feature_cols = [channels[key] for key in feature_keys if key in channels]
        
        missing_data_cols = [col for col in feature_cols if col not in data.columns]
        if missing_data_cols:
            raise ValueError(f"Feature columns not found in data: {missing_data_cols}")

        if data.empty or len(data) < 2:
            return data.assign(qc_flag_custom_qc_method=False)

        # Prepare data for Isolation Forest
        X = data[feature_cols].values

        # Get Isolation Forest parameters from config
        contamination = self.config.get("contamination", 0.05)
        n_estimators = self.config.get("n_estimators", 100)
        max_samples = self.config.get("max_samples", "auto")
        random_state = self.config.get("random_state", 42)

        # Initialize and fit Isolation Forest
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,  # Use all available CPU cores
        )
        model.fit(X)

        # Predict outliers (-1 for outliers, 1 for inliers)
        outlier_predictions = model.predict(X)

        # Convert predictions to boolean flags (True for outlier/bad event)
        data["qc_flag_custom_qc_method"] = outlier_predictions == -1

        return data

    def get_qc_metrics(self, qc_flags: pd.Series) -> dict[str, Any]:
        """Calculate and return QC metrics from the generated flags.

        Args:
            qc_flags: A pandas Series of boolean flags where True indicates a flagged event.

        Returns:
            A dictionary of QC metrics, e.g., outlier_fraction.
        """
        if qc_flags.empty:
            return {"outlier_fraction": 0.0, "sample_flagged": False}

        outlier_fraction = qc_flags.sum() / len(qc_flags)
        min_outlier_fraction = self.config.get("min_outlier_fraction", 0.1)

        return {
            "outlier_fraction": outlier_fraction,
            "sample_flagged": outlier_fraction >= min_outlier_fraction,
        }







