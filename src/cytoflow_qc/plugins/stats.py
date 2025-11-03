"""Plugin interface for custom statistical methods."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import pandas as pd

try:
    import packaging.version
except ImportError:
    packaging = None

from .base import PluginBase, PluginVersionError


class StatsMethodPlugin(PluginBase):
    """Plugin interface for custom statistical analysis methods."""

    PLUGIN_TYPE = "stats_method"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize statistical method plugin.

        Args:
            config: Plugin configuration parameters
        """
        super().__init__(config)
        self._stats_results: pd.DataFrame | None = None

    @abstractmethod
    def apply_stats(
        self,
        data: pd.DataFrame,
        group_col: str,
        value_cols: list[str],
        **kwargs
    ) -> pd.DataFrame:
        """Apply statistical analysis to grouped flow cytometry data.

        Args:
            data: Input DataFrame with flow cytometry data
            group_col: Column name for grouping (e.g., 'condition', 'batch')
            value_cols: List of column names to analyze statistically
            **kwargs: Additional statistical parameters

        Returns:
            DataFrame with statistical results (effect sizes, p-values, etc.)
        """
        pass

    @abstractmethod
    def get_stats_description(self) -> str:
        """Get human-readable description of this statistical method."""
        pass

    @abstractmethod
    def validate_stats_parameters(
        self,
        data: pd.DataFrame,
        group_col: str,
        value_cols: list[str]
    ) -> None:
        """Validate that statistical analysis can be performed.

        Args:
            data: Input DataFrame
            group_col: Grouping column name
            value_cols: Value column names

        Raises:
            ValueError: If statistical analysis cannot be performed
        """
        pass

    @abstractmethod
    def get_stats_columns(self) -> list[str]:
        """Get list of column names that this method will add to results."""
        pass

    def _validate_config(self) -> None:
        """Validate statistical method configuration."""
        required_params = self.get_required_parameters()
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Required parameter '{param}' not found in stats method config")

    @abstractmethod
    def get_required_parameters(self) -> list[str]:
        """Get list of required configuration parameters for this statistical method."""
        pass

    @abstractmethod
    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration parameters for this statistical method."""
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

    def get_stats_results(self) -> pd.DataFrame | None:
        """Get the most recent statistical results."""
        return self._stats_results

    def reset_stats_state(self) -> None:
        """Reset internal statistical state."""
        self._stats_results = None


class StatsMethodResult:
    """Container for statistical method operation results."""

    def __init__(
        self,
        stats_data: pd.DataFrame,
        method_name: str,
        method_version: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Initialize statistical method result.

        Args:
            stats_data: DataFrame with statistical results
            method_name: Name of statistical method used
            method_version: Version of statistical method
            metadata: Additional metadata about the statistical operation
        """
        self.stats_data = stats_data
        self.method_name = method_name
        self.method_version = method_version
        self.metadata = metadata or {}
        self.timestamp = pd.Timestamp.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "stats_shape": self.stats_data.shape,
            "columns": list(self.stats_data.columns),
            "method_name": self.method_name,
            "method_version": self.method_version,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        """String representation of statistical result."""
        return (
            f"StatsMethodResult(method='{self.method_name} v{self.method_version}', "
            f"shape={self.stats_data.shape})"
        )

