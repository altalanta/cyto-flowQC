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

### 2. Define a Pydantic Model for Configuration

Create a Pydantic `BaseModel` to define the schema, default values, and validation rules for your plugin's configuration. This makes your plugin robust and user-friendly.

```python
# src/my_gating_strategy/gating.py
from pydantic import BaseModel, Field

class MyGateConfig(BaseModel):
    my_parameter: float = Field(1.0, description="An example parameter.")
    some_threshold: int = Field(10, gt=0)
```

### 3. Implement the Plugin Class

In `gating.py`, create your main plugin class. It must inherit from `GatingStrategyPlugin` and implement two things:
1.  The `config_model` property, which returns your Pydantic model class.
2.  The `gate` method, which contains your core logic.

```python
# src/my_gating_strategy/gating.py
from typing import Any, Dict, Tuple, Type
import pandas as pd
from cytoflow_qc.plugins.base import GatingStrategyPlugin
from pydantic import BaseModel

# ... (Your MyGateConfig model from above) ...

class MyCustomGate(GatingStrategyPlugin):
    """A brief description of your custom gate."""

    @property
    def config_model(self) -> Type[BaseModel]:
        return MyGateConfig

    def gate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Implement your gating logic here."""
        # Access validated config through self.config
        threshold = self.config.some_threshold
        
        gated_df = df[df["FSC-A"] > threshold]
        params = {"threshold_used": threshold}
        return gated_df, params
```

### 4. Register the Plugin with an Entry Point

In your `pyproject.toml` file, register your new class as a plugin using a special entry point.

```toml
# pyproject.toml
[project.entry-points."cytoflow_qc.gating_strategies"]
my-custom-gate = "my_gating_strategy.gating:MyCustomGate"
```

-   `cytoflow_qc.gating_strategies` is the entry point group for gating plugins.
-   `my-custom-gate` is the name you will use to refer to your strategy in the `cytoflow-qc` CLI (e.g., `--strategy my-custom-gate`).
-   `my_gating_strategy.gating:MyCustomGate` is the import path to your plugin class.

### 5. Install and Use Your Plugin

Install your plugin in the same Python environment as `cytoflow-qc`:

```bash
pip install ./cytoflow-qc-my-gating
```

Once installed, you can configure and use your plugin in your main `config.yaml`:

```yaml
gating:
  strategy: 'my-custom-gate'  # Tell the pipeline to use your plugin
  my-custom-gate:             # Configuration for your plugin (will be validated)
    some_threshold: 15
```

When the pipeline runs, `cytoflow-qc` will automatically find, load, and validate the configuration for your plugin.

For a complete working example, see the `example_plugin` directory in the main `cytoflow-qc` repository.

