"""Pydantic models for configuration validation."""
from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, ValidationError

class ChannelConfig(BaseModel):
    fsc_a: str = 'FSC-A'
    fsc_h: str = 'FSC-H'
    fsc_w: str = 'FSC-W'
    ssc_a: str = 'SSC-A'
    ssc_h: str = 'SSC-H'
    viability: str | None = None
    markers: list[str] = []

class DebrisConfig(BaseModel):
    method: str = 'percentile'
    fsc_a_pct: int = 2
    ssc_a_pct: int = 2

class DoubletsConfig(BaseModel):
    slope: float = 1.0
    intercept: float = 0.0
    tol: float = 0.02

class ChannelQCConfig(BaseModel):
    saturation_threshold: float = Field(0.95, gt=0, le=1)
    min_dynamic_range: int = 2
    min_snr: int = 3

class QCConfig(BaseModel):
    debris: DebrisConfig = Field(default_factory=DebrisConfig)
    doublets: DoubletsConfig = Field(default_factory=DoubletsConfig)
    channel_qc: ChannelQCConfig = Field(default_factory=ChannelQCConfig)

class ViabilityConfig(BaseModel):
    threshold: int = 5000
    direction: str = 'below'

class LymphocytesConfig(BaseModel):
    method: str = 'density'
    percentile: int = 90

class SingletsConfig(BaseModel):
    method: str = 'linear'
    slope_tolerance: float = 0.05

class GatingConfig(BaseModel):
    strategy: str = 'default'
    lymphocytes: LymphocytesConfig = Field(default_factory=LymphocytesConfig)
    singlets: SingletsConfig = Field(default_factory=SingletsConfig)

class CompensationConfig(BaseModel):
    method: str = 'auto'
    matrix_file: Path | None = None

class TransformParameters(BaseModel):
    T: int = 262144
    W: float = 0.5
    M: float = 4.5
    A: float = 0

class TransformsConfig(BaseModel):
    method: str = 'logicle'
    parameters: TransformParameters = Field(default_factory=TransformParameters)

class ReportConfig(BaseModel):
    include_umap: bool = True
    include_drift_plots: bool = True
    figure_format: str = 'png'
    dpi: int = 300

class AppConfig(BaseModel):
    channels: ChannelConfig = Field(default_factory=ChannelConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    viability: ViabilityConfig = Field(default_factory=ViabilityConfig)
    gating: GatingConfig = Field(default_factory=GatingConfig)
    compensation: CompensationConfig = Field(default_factory=CompensationConfig)
    transforms: TransformsConfig = Field(default_factory=TransformsConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

def load_and_validate_config(config_path: Path) -> AppConfig:
    """Loads and validates the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        return AppConfig.parse_obj(config_data)
    except FileNotFoundError:
        raise
    except (yaml.YAMLError, ValidationError) as e:
        raise ValueError(f"Error parsing or validating config file {config_path}:\\n{e}") from e
