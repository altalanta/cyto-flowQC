from typing import Type
from pydantic import BaseModel, Field
from cytoflow_qc.plugins.base import GatingStrategyPlugin
import pandas as pd

class {{ cookiecutter.plugin_name.replace(' ', '') }}Config(BaseModel):
    """Configuration for {{ cookiecutter.plugin_name }}."""
    example_parameter: float = Field(0.5, description="An example parameter for the plugin.")

class {{ cookiecutter.plugin_name.replace(' ', '') }}Plugin(GatingStrategyPlugin):
    """{{ cookiecutter.plugin_name }}"""

    @property
    def config_model(self) -> Type[BaseModel]:
        return {{ cookiecutter.plugin_name.replace(' ', '') }}Config

    def gate(self, df: pd.DataFrame):
        # Your gating logic here
        threshold = self.config.example_parameter
        gated_df = df[df['FSC-A'] > threshold].copy()
        params = {"threshold_used": threshold}
        return gated_df, params

