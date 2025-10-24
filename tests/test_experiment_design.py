from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from cytoflow_qc.experiment_design import ExperimentManager, CohortManager
from cytoflow_qc.io import load_samplesheet


@pytest.fixture
def experiment_directory(tmp_path: Path) -> Path:
    exp_dir = tmp_path / "test_experiment"
    exp_dir.mkdir()

    # Create a dummy samplesheet for validation
    dummy_samplesheet_path = exp_dir / "samplesheet.csv"
    pd.DataFrame({
        "sample_id": ["S1", "S2", "S3"],
        "file_path": ["data/f1.fcs", "data/f2.fcs", "data/f3.fcs"],
        "batch": ["B1", "B1", "B2"],
        "condition": ["C1", "C2", "C1"],
        "age": [20, 30, 25],
    }).to_csv(dummy_samplesheet_path, index=False)

    # Create dummy FCS files for load_samplesheet to not complain
    (exp_dir / "data").mkdir()
    (exp_dir / "data" / "f1.fcs").touch()
    (exp_dir / "data" / "f2.fcs").touch()
    (exp_dir / "data" / "f3.fcs").touch()

    yield exp_dir
    shutil.rmtree(exp_dir)


@pytest.fixture
def initialized_experiment_manager(experiment_directory: Path) -> ExperimentManager:
    exp_manager = ExperimentManager(experiment_directory)
    exp_manager.update_metadata("project", "Test Project")
    exp_manager.set_samplesheet(load_samplesheet(str(experiment_directory / "samplesheet.csv")))
    return exp_manager


class TestExperimentManager:
    def test_init_new_experiment(self, tmp_path: Path):
        exp_dir = tmp_path / "new_experiment"
        exp_manager = ExperimentManager(exp_dir)

        assert exp_dir.exists()
        assert exp_manager.experiment_id is not None
        assert (exp_dir / "experiment_metadata.json").exists()
        assert "created_at" in exp_manager.experiment_metadata

    def test_init_load_existing_experiment(self, experiment_directory: Path):
        exp_manager = ExperimentManager(experiment_directory)
        initial_id = exp_manager.experiment_id
        exp_manager.update_metadata("description", "Initial description")

        # Load again
        reloaded_exp_manager = ExperimentManager(experiment_directory)
        assert reloaded_exp_manager.experiment_id == initial_id
        assert reloaded_exp_manager.experiment_metadata["description"] == "Initial description"
        assert not reloaded_exp_manager.samplesheet.empty

    def test_update_metadata(self, initialized_experiment_manager: ExperimentManager):
        exp_manager = initialized_experiment_manager
        exp_manager.update_metadata("investigator", "Dr. Test")
        assert exp_manager.experiment_metadata["investigator"] == "Dr. Test"

        reloaded_exp_manager = ExperimentManager(exp_manager.experiment_dir)
        assert reloaded_exp_manager.experiment_metadata["investigator"] == "Dr. Test"

    def test_set_samplesheet(self, initialized_experiment_manager: ExperimentManager, tmp_path: Path):
        exp_manager = initialized_experiment_manager
        new_samplesheet_data = pd.DataFrame({
            "sample_id": ["S4", "S5"],
            "file_path": ["data/f4.fcs", "data/f5.fcs"],
            "batch": ["B3", "B3"],
            "condition": ["C3", "C4"],
            "age": [35, 40],
        })
        # Create dummy FCS files for load_samplesheet to not complain
        (tmp_path / "test_experiment" / "data" / "f4.fcs").touch()
        (tmp_path / "test_experiment" / "data" / "f5.fcs").touch()

        exp_manager.set_samplesheet(new_samplesheet_data)

        assert len(exp_manager.samplesheet) == 2
        assert "last_samplesheet_update" in exp_manager.experiment_metadata

        reloaded_exp_manager = ExperimentManager(exp_manager.experiment_dir)
        assert len(reloaded_exp_manager.samplesheet) == 2

    def test_add_sample(self, initialized_experiment_manager: ExperimentManager, tmp_path: Path):
        exp_manager = initialized_experiment_manager
        initial_sample_count = len(exp_manager.samplesheet)

        new_sample = {"sample_id": "S6", "patient_id": "P4", "treatment": "D", "visit": 1, "age": 50, "file_path": "data/f6.fcs", "batch": "B4", "condition": "C4"}
        (tmp_path / "test_experiment" / "data" / "f6.fcs").touch()

        exp_manager.add_sample(new_sample)
        assert len(exp_manager.samplesheet) == initial_sample_count + 1
        assert exp_manager.samplesheet.iloc[-1]["sample_id"] == "S6"

    def test_remove_sample(self, initialized_experiment_manager: ExperimentManager):
        exp_manager = initialized_experiment_manager
        initial_sample_count = len(exp_manager.samplesheet)

        exp_manager.remove_sample("sample_id", "S1")
        assert len(exp_manager.samplesheet) == initial_sample_count - 1
        assert "S1" not in exp_manager.samplesheet["sample_id"].tolist()

        exp_manager.remove_sample("sample_id", "NonExistent") # Remove non-existent
        assert len(exp_manager.samplesheet) == initial_sample_count - 1 # Should not change

    def test_update_analysis_plan(self, initialized_experiment_manager: ExperimentManager):
        exp_manager = initialized_experiment_manager
        analysis_plan = {"gating": {"method": "manual"}, "qc": {"threshold": 0.9}}
        exp_manager.update_analysis_plan(analysis_plan)

        assert exp_manager.analysis_plan["gating"]["method"] == "manual"
        assert "last_analysis_plan_update" in exp_manager.experiment_metadata

        reloaded_exp_manager = ExperimentManager(exp_manager.experiment_dir)
        assert reloaded_exp_manager.analysis_plan["qc"]["threshold"] == 0.9


class TestCohortManager:
    def test_create_cohort(self, initialized_experiment_manager: ExperimentManager):
        cohort_manager = CohortManager(initialized_experiment_manager)
        cohort_manager.create_cohort(
            "Cohort_B1_C1",
            "Samples from Batch B1 and Condition C1",
            {"batch": "B1", "condition": "C1"}
        )

        assert "Cohort_B1_C1" in cohort_manager.list_cohorts()
        details = cohort_manager.get_cohort_details("Cohort_B1_C1")
        assert details["description"] == "Samples from Batch B1 and Condition C1"
        assert "S1" in details["sample_ids"]
        assert "S2" not in details["sample_ids"]

    def test_update_cohort(self, initialized_experiment_manager: ExperimentManager):
        cohort_manager = CohortManager(initialized_experiment_manager)
        cohort_manager.create_cohort(
            "Cohort_B1",
            "Samples from Batch B1",
            {"batch": "B1"}
        )

        cohort_manager.update_cohort(
            "Cohort_B1",
            description="Updated description for Batch B1 samples",
            filters={"batch": "B1", "age": {"operator": ">=", "value": 25}}
        )

        details = cohort_manager.get_cohort_details("Cohort_B1")
        assert details["description"] == "Updated description for Batch B1 samples"
        assert "S1" not in details["sample_ids"]
        assert "S2" in details["sample_ids"]
        assert "S3" in details["sample_ids"]

    def test_delete_cohort(self, initialized_experiment_manager: ExperimentManager):
        cohort_manager = CohortManager(initialized_experiment_manager)
        cohort_manager.create_cohort(
            "Cohort_ToDelete",
            "Temporary cohort",
            {"batch": "B1"}
        )
        assert "Cohort_ToDelete" in cohort_manager.list_cohorts()

        cohort_manager.delete_cohort("Cohort_ToDelete")
        assert "Cohort_ToDelete" not in cohort_manager.list_cohorts()

    def test_get_samples_in_cohort_complex_filters(self, initialized_experiment_manager: ExperimentManager):
        cohort_manager = CohortManager(initialized_experiment_manager)
        # Samplesheet: S1 (B1, C1, 20), S2 (B1, C2, 30), S3 (B2, C1, 25)

        # Test with age > 20 and batch B1
        cohort_manager.create_cohort(
            "Adult_B1",
            "Adults in Batch B1",
            {"age": {"operator": ">", "value": 20}, "batch": "B1"}
        )
        details = cohort_manager.get_cohort_details("Adult_B1")
        assert sorted(details["sample_ids"]) == sorted(["S2"])

        # Test with age <= 25 and condition C1
        cohort_manager.create_cohort(
            "Young_C1",
            "Young in Condition C1",
            {"age": {"operator": "<=", "value": 25}, "condition": "C1"}
        )
        details = cohort_manager.get_cohort_details("Young_C1")
        assert sorted(details["sample_ids"]) == sorted(["S1", "S3"])

        # Test with multiple values for a column
        cohort_manager.create_cohort(
            "Batch_B1_B2",
            "Samples in Batch B1 or B2",
            {"batch": ["B1", "B2"]}
        )
        details = cohort_manager.get_cohort_details("Batch_B1_B2")
        assert sorted(details["sample_ids"]) == sorted(["S1", "S2", "S3"])



