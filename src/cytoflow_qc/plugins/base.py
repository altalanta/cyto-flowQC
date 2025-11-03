"""Base plugin classes and error definitions for cytoflow-qc plugin system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import pandas as pd


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginLoadError(PluginError):
    """Raised when a plugin cannot be loaded."""
    pass


class PluginBase(ABC):
    """Base class for all cytoflow-qc plugins."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize plugin with configuration."""
        self.config = self.get_default_config()
        if config:
            self.config.update(config)

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters for this plugin."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """String representation of the plugin."""
        return f"{self.__class__.__name__}()"


class GatingStrategyPlugin(PluginBase):
    """Base class for gating strategy plugins."""

    @abstractmethod
    def gate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Apply the gating strategy to the dataframe.

        Returns:
            A tuple containing the gated dataframe and a dictionary of gating parameters.
        """
        raise NotImplementedError

