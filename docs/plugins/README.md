# CytoFlow-QC Plugin Development Guide

This guide explains how to develop custom plugins for cytoflow-qc to extend its functionality with custom gating strategies, quality control methods, and statistical analysis techniques.

## Overview

CytoFlow-QC uses a plugin architecture that allows you to:

- **Custom Gating Strategies**: Implement specialized gating algorithms for specific cell types or experimental conditions
- **Advanced QC Methods**: Create sophisticated quality control algorithms beyond the built-in methods
- **Statistical Analysis**: Add new statistical tests and effect size calculations
- **Data Processing**: Extend the pipeline with custom data transformations

## Plugin Types

### 1. Gating Strategy Plugins
Implement custom gating algorithms for identifying cell populations.

### 2. Quality Control Plugins
Create specialized quality control methods for detecting problematic samples.

### 3. Statistical Method Plugins
Add new statistical analysis techniques and effect size calculations.

## Getting Started

### 1. Create a Plugin Class

All plugins inherit from base classes in the `cytoflow_qc.plugins` module:

```python
from cytoflow_qc.plugins.gating import GatingStrategyPlugin

class MyCustomGating(GatingStrategyPlugin):
    PLUGIN_NAME = "my_custom_gating"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "My custom gating strategy"

    def validate_gate_parameters(self, channels):
        # Validate required channels and parameters
        pass

    def apply_gate(self, data, channels, **kwargs):
        # Implement your gating logic
        return gated_data, gating_params
```

### 2. Define Plugin Metadata

Set the required class attributes:

```python
PLUGIN_NAME = "my_plugin_name"           # Unique identifier
PLUGIN_VERSION = "1.0.0"                 # Version string
PLUGIN_DESCRIPTION = "Brief description" # Human-readable description
PLUGIN_AUTHOR = "Your Name"              # Plugin author
PLUGIN_EMAIL = "your.email@example.com"  # Contact information
REQUIRES_CYTOFLOW_VERSION = ">=0.1.0"   # Version compatibility
```

### 3. Implement Required Methods

Each plugin type has specific abstract methods that must be implemented:

#### Gating Strategy Plugins
- `validate_gate_parameters(channels)`: Validate required channels
- `apply_gate(data, channels, **kwargs)`: Apply gating logic
- `get_gate_description()`: Return human-readable description
- `get_required_parameters()`: List required config parameters
- `get_default_config()`: Default configuration values

#### QC Method Plugins
- `validate_qc_parameters(channels)`: Validate QC parameters
- `apply_qc(data, channels, **kwargs)`: Apply QC method
- `get_qc_description()`: Return description
- `get_qc_metrics(qc_flags)`: Calculate QC metrics
- `get_required_parameters()`: Required config parameters
- `get_default_config()`: Default configuration

#### Statistical Method Plugins
- `validate_stats_parameters(data, group_col, value_cols)`: Validate parameters
- `apply_stats(data, group_col, value_cols, **kwargs)`: Apply statistics
- `get_stats_description()`: Return description
- `get_stats_columns()`: List result columns
- `get_required_parameters()`: Required config parameters
- `get_default_config()`: Default configuration

## Configuration Integration

### Plugin Configuration

Plugins are configured through the main cytoflow-qc configuration YAML file:

```yaml
plugins:
  gating_strategy:
    my_custom_gating:
      parameter1: value1
      parameter2: value2
  qc_method:
    my_custom_qc:
      threshold: 0.05
```

### Configuration Validation

Your plugin should validate its configuration:

```python
def _validate_config(self):
    """Validate plugin configuration."""
    required_params = self.get_required_parameters()
    for param in required_params:
        if param not in self.config:
            raise ValueError(f"Required parameter '{param}' not found")

    # Additional validation logic
    threshold = self.config.get("threshold", 0.1)
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")
```

## Error Handling

### Plugin Errors

Use the provided error classes for consistent error handling:

```python
from cytoflow_qc.plugins.base import PluginError, PluginLoadError, PluginVersionError

def apply_gate(self, data, channels, **kwargs):
    try:
        # Your gating logic here
        if some_condition_fails:
            raise PluginError("Custom error message")
    except Exception as e:
        raise PluginError(f"Gating failed: {e}") from e
```

### Graceful Degradation

Plugins should handle edge cases gracefully:

```python
def apply_gate(self, data, channels, **kwargs):
    try:
        # Try advanced method first
        result = self._advanced_gating(data, channels)
    except Exception:
        # Fall back to simpler method
        result = self._simple_gating(data, channels)

    return result
```

## Examples

### Example 1: Custom Gating Strategy

See `src/cytoflow_qc/plugins/examples/custom_gating.py` for a complete example of a custom gating strategy that implements density-based filtering.

### Example 2: Custom Statistical Method

See `src/cytoflow_qc/plugins/examples/custom_stats.py` for an example of advanced statistical analysis with multiple effect size measures and bootstrap confidence intervals.

### Example 3: Custom QC Method

See `src/cytoflow_qc/plugins/examples/custom_qc.py` for an example of machine learning-based quality control.

## Testing Your Plugin

### Unit Testing

Create comprehensive tests for your plugin:

```python
import pytest
from cytoflow_qc.plugins.gating import GatingStrategyPlugin

def test_my_plugin():
    plugin = MyCustomGating({"param": "value"})

    # Test configuration validation
    with pytest.raises(ValueError):
        plugin = MyCustomGating({})  # Missing required params

    # Test gating functionality
    test_data = create_test_data()
    gated_data, params = plugin.apply_gate(test_data, {"fsc_a": "FSC-A"})

    assert len(gated_data) <= len(test_data)
    assert "method" in params
```

### Integration Testing

Test your plugin with the full cytoflow-qc pipeline:

```python
def test_plugin_integration():
    # Create test configuration with your plugin
    config = {
        "channels": {"fsc_a": "FSC-A", "ssc_a": "SSC-A"},
        "plugins": {
            "gating_strategy": {
                "my_custom_gating": {"density_threshold": 0.1}
            }
        }
    }

    # Run pipeline with your plugin
    # Verify results
```

## Best Practices

### 1. Performance Considerations
- **Memory efficiency**: Process data in chunks for large datasets
- **Computational complexity**: Document expected performance characteristics
- **Caching**: Cache expensive computations when appropriate

### 2. Robustness
- **Input validation**: Validate all inputs thoroughly
- **Error handling**: Provide meaningful error messages
- **Edge cases**: Handle empty data, single samples, etc.

### 3. Documentation
- **Comprehensive docstrings**: Document all methods and parameters
- **Usage examples**: Provide clear examples of how to use your plugin
- **Parameter descriptions**: Explain what each configuration parameter does

### 4. Versioning
- **Semantic versioning**: Follow semantic versioning for plugin releases
- **Breaking changes**: Clearly document any breaking changes
- **Deprecation warnings**: Use warnings for deprecated features

## Deployment

### Plugin Distribution

1. **Package your plugin**: Create a Python package with your plugin
2. **Include metadata**: Provide clear installation and usage instructions
3. **Documentation**: Include comprehensive documentation

### Installation

Users can install your plugin and register it:

```bash
pip install my-cytoflow-plugin

# Configure in YAML
plugins:
  gating_strategy:
    my_custom_gating:
      parameter1: value1
```

### Discovery

The plugin registry automatically discovers plugins in registered directories:

```python
from cytoflow_qc.plugins import get_plugin_registry

registry = get_plugin_registry()
registry.register_plugin_path("/path/to/my/plugins")

# Discover available plugins
plugins = registry.discover_plugins("gating_strategy")
```

## Support and Community

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check plugin development documentation
- **Examples**: Study existing plugin examples
- **Community**: Share your plugins with the cytoflow-qc community

## Advanced Topics

### Custom Data Types

If your plugin needs custom data structures:

```python
from dataclasses import dataclass

@dataclass
class CustomGateResult:
    gated_data: pd.DataFrame
    confidence_score: float
    gate_boundaries: dict[str, tuple[float, float]]

    def to_dict(self):
        return {
            "confidence": self.confidence_score,
            "boundaries": self.gate_boundaries,
        }
```

### Async Processing

For computationally intensive plugins:

```python
import asyncio
from typing import Awaitable

class AsyncPlugin(GatingStrategyPlugin):
    async def apply_gate_async(self, data, channels, **kwargs):
        # Async implementation
        pass
```

### Plugin Dependencies

If your plugin requires additional dependencies:

```python
def get_plugin_dependencies(self):
    """Return list of required dependencies."""
    return ["scikit-learn", "scipy", "numpy"]
```

This plugin development guide should help you create robust, well-documented plugins that extend cytoflow-qc's capabilities while maintaining compatibility and following best practices.







