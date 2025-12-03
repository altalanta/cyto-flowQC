"""Pydantic models for pipeline data artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Sample(BaseModel):
    """Represents a single sample and its associated files and metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_id: str
    file_path: Path
    events_file: Path
    metadata_file: Path
    params_file: Optional[Path] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class StageResult(BaseModel):
    """Represents the output of a pipeline stage."""
    stage_name: str
    samples: List[Sample]

    def get_sample(self, sample_id: str) -> Optional[Sample]:
        """Find a sample by its ID."""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None


