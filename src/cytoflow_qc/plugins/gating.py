"""Plugin interface for custom gating strategies."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import pandas as pd

try:
    import packaging.version
except ImportError:
    packaging = None

from .base import PluginBase


class GatingStrategyPlugin(PluginBase):
    """Plugin interface for custom gating strategies."""

    PLUGIN_TYPE = "gating_strategy"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize gating strategy plugin.

        Args:
            config: Plugin configuration parameters
        """
        super().__init__(config)
        self._gated_data: pd.DataFrame | None = None
        self._gating_params: dict[str, Any] | None = None

    @abstractmethod
    def apply_gate(
        self,
        data: pd.DataFrame,
        channels: dict[str, str],
        **kwargs
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply gating strategy to flow cytometry data.

        Args:
            data: Input DataFrame with flow cytometry events
            channels: Dictionary mapping canonical channel names to data columns
            **kwargs: Additional gating parameters

        Returns:
            Tuple of (gated_data, gating_parameters)
            - gated_data: DataFrame with gated events
            - gating_parameters: Dictionary of gating parameters used
        """
        pass

    @abstractmethod
    def get_gate_description(self) -> str:
        """Get human-readable description of this gating strategy."""
        pass

    @abstractmethod
    def validate_gate_parameters(self, channels: dict[str, str]) -> None:
        """Validate that required channels are available for gating.

        Args:
            channels: Dictionary mapping canonical channel names to data columns

        Raises:
            ValueError: If required channels are missing
        """
        pass

    def _validate_config(self) -> None:
        """Validate gating strategy configuration."""
        required_params = self.get_required_parameters()
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Required parameter '{param}' not found in gating strategy config")

    @abstractmethod
    def get_required_parameters(self) -> list[str]:
        """Get list of required configuration parameters for this gating strategy."""
        pass

    @abstractmethod
    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration parameters for this gating strategy."""
        pass

    def validate_compatibility(self) -> None:
        """Validate compatibility with current cytoflow-qc version."""
        # Import current version dynamically to avoid circular imports
        try:
            from .._version import __version__ as current_version
            import packaging.version

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

    def get_gated_data(self) -> pd.DataFrame | None:
        """Get the most recently gated data."""
        return self._gated_data

    def get_gating_params(self) -> dict[str, Any] | None:
        """Get the gating parameters from the last gating operation."""
        return self._gating_params

    def reset_gating_state(self) -> None:
        """Reset internal gating state."""
        self._gated_data = None
        self._gating_params = None


class GatingResult:
    """Container for gating operation results."""

    def __init__(
        self,
        gated_data: pd.DataFrame,
        gating_params: dict[str, Any],
        strategy_name: str,
        strategy_version: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Initialize gating result.

        Args:
            gated_data: Gated DataFrame
            gating_params: Parameters used for gating
            strategy_name: Name of gating strategy used
            strategy_version: Version of gating strategy
            metadata: Additional metadata about the gating operation
        """
        self.gated_data = gated_data
        self.gating_params = gating_params
        self.strategy_name = strategy_name
        self.strategy_version = strategy_version
        self.metadata = metadata or {}
        self.timestamp = pd.Timestamp.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "gated_events": len(self.gated_data),
            "gating_params": self.gating_params,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        """String representation of gating result."""
        return (
            f"GatingResult(strategy='{self.strategy_name} v{self.strategy_version}', "
            f"events={len(self.gated_data)}, params={len(self.gating_params)})"
        )
