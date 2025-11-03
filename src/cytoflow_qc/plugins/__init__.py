"""Plugin system for extending cytoflow-qc functionality."""

from .base import PluginBase, PluginError, PluginLoadError, PluginVersionError
from .gating import GatingStrategyPlugin
from .qc import QCMethodPlugin
from .stats import StatsMethodPlugin
from .registry import PluginRegistry, get_plugin_registry

__all__ = [
    "PluginBase",
    "PluginError",
    "PluginLoadError",
    "PluginVersionError",
    "GatingStrategyPlugin",
    "QCMethodPlugin",
    "StatsMethodPlugin",
    "PluginRegistry",
    "get_plugin_registry",
]
