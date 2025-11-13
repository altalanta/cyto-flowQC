"""Base plugin classes and error definitions for cytoflow-qc plugin system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type

import pandas as pd
from pydantic import BaseModel, ValidationError


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginLoadError(PluginError):
    """Raised when a plugin cannot be loaded."""
    pass


class PluginBase(ABC):
    """Base class for all cytoflow-qc plugins."""
    config: BaseModel

    @property
    @abstractmethod
    def config_model(self) -> Type[BaseModel]:
        """The Pydantic model for the plugin's configuration."""
        raise NotImplementedError

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize plugin with configuration and validate it."""
        try:
            self.config = self.config_model(**(config or {}))
        except ValidationError as e:
            raise PluginError(f"Invalid configuration for plugin {self.__class__.__name__}:\n{e}") from e

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

