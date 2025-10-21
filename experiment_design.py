from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from cytoflow_qc.io import load_samplesheet # Import for samplesheet validation


class ExperimentManager:
    """Manages the lifecycle of an experiment, including metadata, samples, and analysis plans."""

    def __init__(self, experiment_dir: str | Path):
        """Initialize ExperimentManager.

        Args:
            experiment_dir: Directory where experiment data and metadata are stored.
        """
        self.experiment_dir = Path(experiment_dir)
        self.metadata_file = self.experiment_dir / "experiment_metadata.json"
        self.samplesheet_file = self.experiment_dir / "samplesheet.csv"
        self.analysis_plan_file = self.experiment_dir / "analysis_plan.json"
        self.experiment_id: str | None = None
        self.experiment_metadata: Dict[str, Any] = {}
        self.samplesheet: pd.DataFrame = pd.DataFrame()
        self.analysis_plan: Dict[str, Any] = {}

        self._load_experiment()

    def _load_experiment(self) -> None:
        """Load existing experiment metadata, samplesheet, and analysis plan if available."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.experiment_metadata = json.load(f)
            self.experiment_id = self.experiment_metadata.get("experiment_id")

        if self.samplesheet_file.exists():
            self.samplesheet = pd.read_csv(self.samplesheet_file)

        if self.analysis_plan_file.exists():
            with open(self.analysis_plan_file, 'r') as f:
                self.analysis_plan = json.load(f)

        if not self.experiment_id:
            self.experiment_id = str(uuid.uuid4())
            self.experiment_metadata["experiment_id"] = self.experiment_id
            self.experiment_metadata["created_at"] = datetime.now().isoformat()
            self._save_experiment_metadata()

    def _save_experiment_metadata(self) -> None:
        """Save experiment metadata to file."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=4)

    def update_metadata(self, key: str, value: Any) -> None:
        """Update a single metadata field."""
        self.experiment_metadata[key] = value
        self._save_experiment_metadata()

    def set_samplesheet(self, samplesheet_df: pd.DataFrame) -> None:
        """Set the samplesheet for the experiment and save it."""
        # Validate the samplesheet using io.load_samplesheet (requires saving to temp file or adapting load_samplesheet)
        # For now, let's just do basic validation here or assume pre-validated
        # A more robust solution would be to make io.load_samplesheet accept a DataFrame directly.
        # For simplicity and to avoid modifying io.py's core logic for now:
        temp_samplesheet_path = self.experiment_dir / ".temp_samplesheet.csv"
        samplesheet_df.to_csv(temp_samplesheet_path, index=False)
        validated_df = load_samplesheet(str(temp_samplesheet_path))
        temp_samplesheet_path.unlink() # Delete temp file

        self.samplesheet = validated_df
        self.samplesheet.to_csv(self.samplesheet_file, index=False)
        self.update_metadata("last_samplesheet_update", datetime.now().isoformat())

    def update_analysis_plan(self, analysis_plan_dict: Dict[str, Any]) -> None:
        """Update the analysis plan and save it."""
        self.analysis_plan = analysis_plan_dict
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(self.analysis_plan_file, 'w') as f:
            json.dump(self.analysis_plan, f, indent=4)
        self.update_metadata("last_analysis_plan_update", datetime.now().isoformat())

    def get_experiment_info(self) -> Dict[str, Any]:
        """Return comprehensive information about the experiment."""
        info = {
            "experiment_id": self.experiment_id,
            "metadata": self.experiment_metadata,
            "samples_count": len(self.samplesheet) if not self.samplesheet.empty else 0,
            "analysis_plan_defined": bool(self.analysis_plan),
            "samplesheet_path": str(self.samplesheet_file.absolute()),
            "analysis_plan_path": str(self.analysis_plan_file.absolute()),
        }
        return info

    def add_sample(self, sample_data: Dict[str, Any]) -> None:
        """Add a new sample record to the samplesheet."""
        new_sample_df = pd.DataFrame([sample_data])
        self.samplesheet = pd.concat([self.samplesheet, new_sample_df], ignore_index=True)
        self.set_samplesheet(self.samplesheet) # Re-saves the updated samplesheet

    def remove_sample(self, sample_id_col: str, sample_id_value: Any) -> None:
        """Remove a sample from the samplesheet by ID."""
        if self.samplesheet.empty:
            print("No samples to remove.")
            return
        initial_len = len(self.samplesheet)
        self.samplesheet = self.samplesheet[self.samplesheet[sample_id_col] != sample_id_value]
        if len(self.samplesheet) < initial_len:
            self.set_samplesheet(self.samplesheet)
            print(f"Sample with {sample_id_col}={sample_id_value} removed.")
        else:
            print(f"Sample with {sample_id_col}={sample_id_value} not found.")


class CohortManager:
    """Manages patient/sample cohorts within an experiment for subgroup analysis."""

    def __init__(self, experiment_manager: ExperimentManager):
        """Initialize CohortManager.

        Args:
            experiment_manager: An instance of ExperimentManager.
        """
        self.experiment_manager = experiment_manager
        self.cohorts_file = self.experiment_manager.experiment_dir / "cohorts.json"
        self.cohorts: Dict[str, Dict[str, Any]] = {}
        self._load_cohorts()

    def _load_cohorts(self) -> None:
        """Load existing cohort definitions."""
        if self.cohorts_file.exists():
            with open(self.cohorts_file, 'r') as f:
                self.cohorts = json.load(f)

    def _save_cohorts(self) -> None:
        """Save cohort definitions to file."""
        self.experiment_manager.experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cohorts_file, 'w') as f:
            json.dump(self.cohorts, f, indent=4)

    def create_cohort(
        self, cohort_name: str, description: str, filters: Dict[str, Any]
    ) -> None:
        """Create a new cohort based on sample filters.

        Args:
            cohort_name: Name of the cohort.
            description: Description of the cohort.
            filters: Dictionary of column: value pairs to filter samples.
        """
        if cohort_name in self.cohorts:
            print(f"Cohort '{cohort_name}' already exists. Use update_cohort to modify.")
            return

        self.cohorts[cohort_name] = {
            "description": description,
            "filters": filters,
            "created_at": datetime.now().isoformat(),
            "sample_ids": self._get_samples_in_cohort(filters),
        }
        self._save_cohorts()
        print(f"Cohort '{cohort_name}' created with {len(self.cohorts[cohort_name]['sample_ids'])} samples.")

    def update_cohort(self, cohort_name: str, description: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> None:
        """Update an existing cohort.

        Args:
            cohort_name: Name of the cohort to update.
            description: New description for the cohort.
            filters: New filters for the cohort.
        """
        if cohort_name not in self.cohorts:
            print(f"Cohort '{cohort_name}' not found. Use create_cohort to add.")
            return

        if description is not None:
            self.cohorts[cohort_name]["description"] = description
        if filters is not None:
            self.cohorts[cohort_name]["filters"] = filters
            self.cohorts[cohort_name]["sample_ids"] = self._get_samples_in_cohort(filters)
        self.cohorts[cohort_name]["last_updated"] = datetime.now().isoformat()
        self._save_cohorts()
        print(f"Cohort '{cohort_name}' updated. Now contains {len(self.cohorts[cohort_name]['sample_ids'])} samples.")

    def delete_cohort(self, cohort_name: str) -> None:
        """Delete a cohort.

        Args:
            cohort_name: Name of the cohort to delete.
        """
        if cohort_name in self.cohorts:
            del self.cohorts[cohort_name]
            self._save_cohorts()
            print(f"Cohort '{cohort_name}' deleted.")
        else:
            print(f"Cohort '{cohort_name}' not found.")

    def get_cohort_details(self, cohort_name: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific cohort.

        Args:
            cohort_name: Name of the cohort.

        Returns:
            Cohort details or None if not found.
        """
        return self.cohorts.get(cohort_name)

    def list_cohorts(self) -> List[str]:
        """List all defined cohort names."""
        return list(self.cohorts.keys())

    def _get_samples_in_cohort(self, filters: Dict[str, Any]) -> List[str]:
        """Apply filters to the experiment's samplesheet to get matching sample IDs."""
        if self.experiment_manager.samplesheet.empty:
            return []

        filtered_df = self.experiment_manager.samplesheet.copy()
        for col, value in filters.items():
            if col in filtered_df.columns:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[col].isin(value)]
                elif isinstance(value, dict) and "operator" in value and "value" in value:
                    # Handle more complex filter conditions (e.g., >, <, >=, <=)
                    operator = value["operator"]
                    filter_val = value["value"]
                    if operator == ">":
                        filtered_df = filtered_df[filtered_df[col] > filter_val]
                    elif operator == "<":
                        filtered_df = filtered_df[filtered_df[col] < filter_val]
                    elif operator == ">=":
                        filtered_df = filtered_df[filtered_df[col] >= filter_val]
                    elif operator == "<=":
                        filtered_df = filtered_df[filtered_df[col] <= filter_val]
                    elif operator == "!=":
                        filtered_df = filtered_df[filtered_df[col] != filter_val]
                    else:
                        print(f"Warning: Unsupported operator '{operator}' for column '{col}'. Ignoring filter.")
                else:
                    filtered_df = filtered_df[filtered_df[col] == value]
            else:
                print(f"Warning: Filter column '{col}' not found in samplesheet. Ignoring filter.")

        return filtered_df["sample_id"].tolist() if "sample_id" in filtered_df.columns else []


if __name__ == "__main__":
    # Example Usage
    print("--- Experiment Manager Example ---")
    exp_dir = Path("./test_experiment_data")
    exp_dir.mkdir(exist_ok=True)

    # Initialize experiment manager
    exp_manager = ExperimentManager(exp_dir)
    print(f"Experiment ID: {exp_manager.experiment_id}")

    # Update metadata
    exp_manager.update_metadata("project_name", "Cancer Biomarker Study")
    exp_manager.update_metadata("principal_investigator", "Dr. Jane Doe")
    print(f"Metadata: {exp_manager.get_experiment_info()['metadata']}")

    # Set samplesheet
    sample_data = [
        {"sample_id": "S001", "patient_id": "P001", "treatment": "A", "visit": 1, "age": 30},
        {"sample_id": "S002", "patient_id": "P002", "treatment": "B", "visit": 1, "age": 45},
        {"sample_id": "S003", "patient_id": "P001", "treatment": "A", "visit": 2, "age": 30},
        {"sample_id": "S004", "patient_id": "P003", "treatment": "C", "visit": 1, "age": 60},
    ]
    samplesheet_df = pd.DataFrame(sample_data)
    exp_manager.set_samplesheet(samplesheet_df)
    print("\nSamplesheet:")
    print(exp_manager.samplesheet)

    # Add a new sample
    exp_manager.add_sample({"sample_id": "S005", "patient_id": "P004", "treatment": "A", "visit": 1, "age": 25})
    print("\nSamplesheet after adding S005:")
    print(exp_manager.samplesheet)

    # Remove a sample
    exp_manager.remove_sample("sample_id", "S002")
    print("\nSamplesheet after removing S002:")
    print(exp_manager.samplesheet)

    # Update analysis plan
    analysis_plan = {
        "qc_methods": ["debris_filter", "doublet_removal"],
        "gating_strategy": "lymphocyte_subset",
        "statistical_tests": [{"name": "t-test", "group_by": "treatment"}],
    }
    exp_manager.update_analysis_plan(analysis_plan)
    print("\nAnalysis Plan:")
    print(exp_manager.analysis_plan)

    print("\n--- Cohort Manager Example ---")
    cohort_manager = CohortManager(exp_manager)

    # Create cohorts
    cohort_manager.create_cohort(
        "Treatment_A_Cohort",
        "All samples receiving Treatment A",
        {"treatment": "A"}
    )
    cohort_manager.create_cohort(
        "Young_Patients",
        "Patients under 40 years old",
        {"age": {"operator": "<", "value": 40}}
    )
    cohort_manager.create_cohort(
        "Visit_1_Treatment_A",
        "Treatment A samples at Visit 1",
        {"treatment": "A", "visit": 1}
    )

    print("\nDefined Cohorts:", cohort_manager.list_cohorts())
    print("\nDetails for Treatment_A_Cohort:", cohort_manager.get_cohort_details("Treatment_A_Cohort"))

    # Update a cohort
    cohort_manager.update_cohort(
        "Young_Patients",
        description="Patients aged 18-39",
        filters={
            "age": {"operator": "<", "value": 40},
            "treatment": "A" # Add another filter
        }
    )
    print("\nDetails for Young_Patients (updated):", cohort_manager.get_cohort_details("Young_Patients"))

    # Delete a cohort
    cohort_manager.delete_cohort("Visit_1_Treatment_A")
    print("\nDefined Cohorts after deletion:", cohort_manager.list_cohorts())

    # Clean up
    import shutil
    shutil.rmtree(exp_dir)
