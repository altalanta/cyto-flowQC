"""Plugin interface for custom quality control methods."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import pandas as pd

try:
    import packaging.version
except ImportError:
    packaging = None

from .base import PluginBase, PluginVersionError


class QCMethodPlugin(PluginBase):
    """Plugin interface for custom quality control methods."""

    PLUGIN_TYPE = "qc_method"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize QC method plugin.

        Args:
            config: Plugin configuration parameters
        """
        super().__init__(config)
        self._qc_results: dict[str, pd.Series] | None = None

    @abstractmethod
    def apply_qc(
        self,
        data: pd.DataFrame,
        channels: dict[str, str],
        **kwargs
    ) -> dict[str, pd.Series]:
        """Apply quality control method to flow cytometry data.

        Args:
            data: Input DataFrame with flow cytometry events
            channels: Dictionary mapping canonical channel names to data columns
            **kwargs: Additional QC parameters

        Returns:
            Dictionary mapping QC flag names to boolean Series
        """
        pass

    @abstractmethod
    def get_qc_description(self) -> str:
        """Get human-readable description of this QC method."""
        pass

    @abstractmethod
    def validate_qc_parameters(self, channels: dict[str, str]) -> None:
        """Validate that required channels are available for QC.

        Args:
            channels: Dictionary mapping canonical channel names to data columns

        Raises:
            ValueError: If required channels are missing
        """
        pass

    @abstractmethod
    def get_qc_metrics(self, qc_flags: dict[str, pd.Series]) -> dict[str, float]:
        """Calculate QC metrics from flag results.

        Args:
            qc_flags: Dictionary of QC flag Series from apply_qc()

        Returns:
            Dictionary of QC metric names to values
        """
        pass

    def _validate_config(self) -> None:
        """Validate QC method configuration."""
        required_params = self.get_required_parameters()
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Required parameter '{param}' not found in QC method config")

    @abstractmethod
    def get_required_parameters(self) -> list[str]:
        """Get list of required configuration parameters for this QC method."""
        pass

    @abstractmethod
    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration parameters for this QC method."""
        pass

    def validate_compatibility(self) -> None:
        """Validate compatibility with current cytoflow-qc version."""
        try:
            from .._version import __version__ as current_version

            if packaging:
                required_version = packaging.version.parse(self.REQUIRES_CYTOFLOW_VERSION.replace(">=", ""))
                current = packaging.version.parse(current_version)

                if current < required_version:
                    raise PluginVersionError(
                        f"Plugin {self.name} v{self.version} requires cytoflow-qc {self.REQUIRES_CYTOFLOW_VERSION}, "
                        f"but current version is {current_version}"
                    )
        except ImportError:
            # If version info not available, assume compatibility
            pass

    def get_qc_results(self) -> dict[str, pd.Series] | None:
        """Get the most recent QC results."""
        return self._qc_results

    def reset_qc_state(self) -> None:
        """Reset internal QC state."""
        self._qc_results = None


class QCMethodResult:
    """Container for QC method operation results."""

    def __init__(
        self,
        qc_flags: dict[str, pd.Series],
        qc_metrics: dict[str, float],
        method_name: str,
        method_version: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Initialize QC method result.

        Args:
            qc_flags: Dictionary of QC flag Series
            qc_metrics: Dictionary of computed QC metrics
            method_name: Name of QC method used
            method_version: Version of QC method
            metadata: Additional metadata about the QC operation
        """
        self.qc_flags = qc_flags
        self.qc_metrics = qc_metrics
        self.method_name = method_name
        self.method_version = method_version
        self.metadata = metadata or {}
        self.timestamp = pd.Timestamp.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "qc_flags": {name: series.sum() for name, series in self.qc_flags.items()},
            "qc_metrics": self.qc_metrics,
            "method_name": self.method_name,
            "method_version": self.method_version,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        """String representation of QC result."""
        return (
            f"QCMethodResult(method='{self.method_name} v{self.method_version}', "
            f"flags={len(self.qc_flags)}, metrics={len(self.qc_metrics)})"
        )
