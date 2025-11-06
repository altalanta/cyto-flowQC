"""Unit tests for the plugin system."""
import unittest.mock
import pytest
from cytoflow_qc.plugins.registry import PluginRegistry
from cytoflow_qc.plugins.base import GatingStrategyPlugin, PluginLoadError

class MockGatingPlugin(GatingStrategyPlugin):
    """A mock gating plugin for testing."""
    def get_default_config(self):
        return {}
    def gate(self, df):
        return df, {}

class BadPlugin:
    """A class that does not inherit from PluginBase."""
    pass

@unittest.mock.patch("importlib.metadata.entry_points")
def test_plugin_discovery(mock_entry_points):
    """Test that the registry correctly discovers plugins from entry points."""
    # Mock entry points
    mock_entry_points.return_value = [
        unittest.mock.Mock(name="mock_gate", value="mock_gate:MockGatingPlugin", group="cytoflow_qc.gating_strategies"),
    ]
    
    # Mock the load method of the entry point
    mock_entry_points.return_value[0].load.return_value = MockGatingPlugin

    registry = PluginRegistry()
    available = registry.get_available_plugins("gating")
    assert "mock_gate" in available["gating"]

@unittest.mock.patch("importlib.metadata.entry_points")
def test_plugin_loading(mock_entry_points):
    """Test that the registry can successfully load a valid plugin."""
    mock_entry_points.return_value = [
        unittest.mock.Mock(name="mock_gate", value="mock_gate:MockGatingPlugin", group="cytoflow_qc.gating_strategies"),
    ]
    mock_entry_points.return_value[0].load.return_value = MockGatingPlugin

    registry = PluginRegistry()
    plugin_instance = registry.load_plugin("gating", "mock_gate")
    assert isinstance(plugin_instance, MockGatingPlugin)

@unittest.mock.patch("importlib.metadata.entry_points")
def test_plugin_load_error(mock_entry_points):
    """Test that the registry raises an error for a non-existent plugin."""
    mock_entry_points.return_value = []

    registry = PluginRegistry()
    with pytest.raises(PluginLoadError):
        registry.load_plugin("gating", "non_existent_plugin")

@unittest.mock.patch("importlib.metadata.entry_points")
def test_plugin_discovery_handles_bad_plugins(mock_entry_points):
    """Test that discovery skips plugins that don't inherit from the base class."""
    mock_entry_points.return_value = [
        unittest.mock.Mock(name="bad_plugin", value="bad:BadPlugin", group="cytoflow_qc.gating_strategies"),
    ]
    mock_entry_points.return_value[0].load.return_value = BadPlugin

    registry = PluginRegistry()
    available = registry.get_available_plugins("gating")
    assert "bad_plugin" not in available["gating"]
