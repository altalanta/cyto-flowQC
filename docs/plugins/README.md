# Extending CytoFlow-QC with Plugins

CytoFlow-QC is designed to be extensible through a powerful and flexible plugin system. You can create your own custom components for various stages of the analysis pipeline, such as gating strategies, quality control methods, and statistical analyses.

## How Plugins Work

The plugin system is built on Python's standard `entry_points` mechanism. This allows you to develop your plugin as a completely separate, installable Python package. When your package is installed in the same environment as `cytoFlow-qc`, it automatically discovers and integrates your custom components.

## Creating a Gating Strategy Plugin

Here's a step-by-step guide to creating a custom gating strategy plugin.

### 1. Set Up Your Project Structure

Create a new directory for your plugin. The structure should look something like this:

```
cytoflow-qc-my-gating/
├── pyproject.toml
├── README.md
└── src/
    └── my_gating_strategy/
        ├── __init__.py
        └── gating.py
```

### 2. Define the Plugin Class

In `gating.py`, create a class that inherits from `GatingStrategyPlugin` and implements the required `gate` method.

```python
# src/my_gating_strategy/gating.py
from typing import Any, Dict, Tuple
import pandas as pd
from cytoflow_qc.plugins.base import GatingStrategyPlugin

class MyCustomGate(GatingStrategyPlugin):
    """A brief description of your custom gate."""

    def get_default_config(self) -> Dict[str, Any]:
        """Define default parameters for your gate."""
        return {"my_parameter": 1.0}

    def gate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Implement your gating logic here."""
        # Your gating logic goes here...
        gated_df = df[df["FSC-A"] > self.config["my_parameter"]]
        params = {"parameter_used": self.config["my_parameter"]}
        return gated_df, params
```

### 3. Register the Plugin with an Entry Point

In your `pyproject.toml` file, you need to register your new class as a plugin using a special entry point.

```toml
# pyproject.toml
[project.entry-points."cytoflow_qc.gating_strategies"]
my-custom-gate = "my_gating_strategy.gating:MyCustomGate"
```

-   `cytoflow_qc.gating_strategies` is the entry point group for gating plugins.
-   `my-custom-gate` is the name you will use to refer to your strategy in the `cytoflow-qc` CLI (e.g., `--strategy my-custom-gate`).
-   `my_gating_strategy.gating:MyCustomGate` is the import path to your plugin class.

### 4. Install and Use Your Plugin

To use your new plugin, install it in the same Python environment as `cytoflow-qc`:

```bash
pip install ./cytoflow-qc-my-gating
```

Once installed, `cytoflow-qc` will automatically discover your plugin, and you can use it in your pipeline:

```bash
cytoflow-qc gate --strategy my-custom-gate ...
```

For a complete working example, see the `example_plugin` directory in the main `cytoflow-qc` repository.

