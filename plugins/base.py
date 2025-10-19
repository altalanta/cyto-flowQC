"""Base plugin classes and error definitions for cytoflow-qc plugin system."""

from __future__ import annotations

import importlib.metadata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginLoadError(PluginError):
    """Raised when a plugin cannot be loaded."""
    pass


class PluginVersionError(PluginError):
    """Raised when plugin version compatibility issues occur."""
    pass


class PluginBase(ABC):
    """Base class for all cytoflow-qc plugins."""

    # Plugin metadata
    PLUGIN_NAME: str = "unnamed_plugin"
    PLUGIN_VERSION: str = "1.0.0"
    PLUGIN_DESCRIPTION: str = "Base plugin class"
    PLUGIN_AUTHOR: str = "Unknown"
    PLUGIN_EMAIL: str = ""

    # Required cytoflow-qc version compatibility
    REQUIRES_CYTOFLOW_VERSION: str = ">=0.1.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize plugin with configuration.

        Args:
            config: Plugin-specific configuration parameters
        """
        self.config = config or {}
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate plugin configuration parameters."""
        # Subclasses should override this method to validate their specific config
        pass

    @property
    def name(self) -> str:
        """Get plugin name."""
        return self.PLUGIN_NAME

    @property
    def version(self) -> str:
        """Get plugin version."""
        return self.PLUGIN_VERSION

    @property
    def description(self) -> str:
        """Get plugin description."""
        return self.PLUGIN_DESCRIPTION

    @abstractmethod
    def validate_compatibility(self) -> None:
        """Validate that this plugin is compatible with current cytoflow-qc version."""
        pass

    @abstractmethod
    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration parameters for this plugin."""
        pass

    def get_config_schema(self) -> dict[str, Any]:
        """Get JSON schema for plugin configuration validation."""
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True
        }

    def __repr__(self) -> str:
        """String representation of the plugin."""
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"







