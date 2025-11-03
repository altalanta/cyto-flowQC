# Interactive Experiment Design and Cohort Management

CytoFlow-QC now provides tools for interactive experiment design and cohort management, allowing users to define experimental metadata, manage samples, and create patient/sample cohorts for subgroup analysis. This enhances reproducibility and simplifies complex study designs.

## Experiment Manager

The `ExperimentManager` class helps you organize your experiment by managing metadata, samplesheets, and analysis plans. Each experiment is stored in a dedicated directory, ensuring all related files are kept together.

### CLI Usage

There are no direct CLI commands for `ExperimentManager` as its functionality is primarily integrated into the main pipeline stages and other tools. However, you can initialize and interact with it programmatically.

### Programmatic Usage

You can use the `ExperimentManager` class directly in your Python scripts:

```python
from cytoflow_qc.experiment_design import ExperimentManager
from pathlib import Path
import pandas as pd
import shutil

# Define an experiment directory
exp_dir = Path("./my_first_experiment")
exp_dir.mkdir(exist_ok=True) # Create if it doesn't exist

# Initialize ExperimentManager (loads existing or creates new experiment)
exp_manager = ExperimentManager(exp_dir)
print(f"Experiment ID: {exp_manager.experiment_id}")

# Update experiment metadata
exp_manager.update_metadata("project_name", "Immunotherapy Trial Phase I")
exp_manager.update_metadata("principal_investigator", "Dr. Alex Chen")
print("Current Metadata:", exp_manager.get_experiment_info()['metadata'])

# Define and set a samplesheet
sample_data = [
    {"sample_id": "TRIAL-001", "patient_id": "P101", "treatment_group": "A", "day": 0, "age": 55, "file_path": "data/TRIAL-001.fcs", "batch": "B1", "condition": "Treated"},
    {"sample_id": "TRIAL-002", "patient_id": "P102", "treatment_group": "B", "day": 0, "age": 62, "file_path": "data/TRIAL-002.fcs", "batch": "B1", "condition": "Control"},
    {"sample_id": "TRIAL-003", "patient_id": "P101", "treatment_group": "A", "day": 7, "age": 55, "file_path": "data/TRIAL-003.fcs", "batch": "B2", "condition": "Treated"},
    {"sample_id": "TRIAL-004", "patient_id": "P103", "treatment_group": "A", "day": 0, "age": 48, "file_path": "data/TRIAL-004.fcs", "batch": "B2", "condition": "Treated"},
]
# Note: In a real scenario, file_path should point to actual FCS files
samplesheet_df = pd.DataFrame(sample_data)
exp_manager.set_samplesheet(samplesheet_df)
print("\nUpdated Samplesheet:")
print(exp_manager.samplesheet)

# Add a new sample
exp_manager.add_sample({"sample_id": "TRIAL-005", "patient_id": "P104", "treatment_group": "C", "day": 0, "age": 38, "file_path": "data/TRIAL-005.fcs", "batch": "B3", "condition": "Control"})
print("\nSamplesheet after adding TRIAL-005:")
print(exp_manager.samplesheet)

# Update analysis plan
analysis_plan = {
    "qc_thresholds": {"min_events": 1000, "max_debris": 0.1},
    "gating_strategies": [{"name": "T_cells", "channels": ["CD3"]}, {"name": "B_cells", "channels": ["CD19"]}],
    "statistical_comparisons": [{"group1": "A", "group2": "B", "markers": ["CD4", "CD8"]}],
}
exp_manager.update_analysis_plan(analysis_plan)
print("\nAnalysis Plan:")
print(exp_manager.analysis_plan)

# Clean up example directory
# shutil.rmtree(exp_dir)
```

## Cohort Manager

The `CohortManager` works in conjunction with `ExperimentManager` to define and manage patient/sample cohorts. This is useful for performing subgroup analyses or for defining specific groups of interest within your experiment.

### CLI Usage

Similar to `ExperimentManager`, the `CohortManager` is primarily used programmatically for building complex cohort definitions. However, its integration into future CLI commands for specific analyses (e.g., `cytoflow-qc analyze --cohort <name>`) is planned.

### Programmatic Usage

You can use the `CohortManager` class to define and query cohorts:

```python
from cytoflow_qc.experiment_design import ExperimentManager, CohortManager
from pathlib import Path
import pandas as pd
import shutil

exp_dir = Path("./my_first_experiment")
exp_dir.mkdir(exist_ok=True)

# Ensure ExperimentManager has a samplesheet loaded
exp_manager = ExperimentManager(exp_dir)
sample_data = [
    {"sample_id": "S001", "patient_id": "P001", "treatment": "A", "visit": 1, "age": 30},
    {"sample_id": "S002", "patient_id": "P002", "treatment": "B", "visit": 1, "age": 45},
    {"sample_id": "S003", "patient_id": "P001", "treatment": "A", "visit": 2, "age": 30},
    {"sample_id": "S004", "patient_id": "P003", "treatment": "C", "visit": 1, "age": 60},
]
samplesheet_df = pd.DataFrame(sample_data)
exp_manager.set_samplesheet(samplesheet_df)

cohort_manager = CohortManager(exp_manager)

# Create cohorts using filters
cohort_manager.create_cohort(
    "Treatment_A_Group",
    "Samples from patients in Treatment Group A",
    {"treatment": "A"}
)

cohort_manager.create_cohort(
    "Elderly_Patients",
    "Patients older than 50 years",
    {"age": {"operator": ">=", "value": 50}}
)

cohort_manager.create_cohort(
    "Treatment_A_Visit_1",
    "Samples from Treatment A at Visit 1",
    {"treatment": "A", "visit": 1}
)

print("\nAll Defined Cohorts:", cohort_manager.list_cohorts())
print("\nDetails for 'Treatment_A_Group':")
print(cohort_manager.get_cohort_details("Treatment_A_Group"))

# Get sample IDs in a cohort
print("\nSample IDs in 'Elderly_Patients':")
print(cohort_manager.get_cohort_details("Elderly_Patients")['sample_ids'])

# Update a cohort
cohort_manager.update_cohort(
    "Elderly_Patients",
    description="Patients aged 50 and above in Treatment C",
    filters={
        "age": {"operator": ">=", "value": 50},
        "treatment": "C"
    }
)
print("\nUpdated 'Elderly_Patients' details:")
print(cohort_manager.get_cohort_details("Elderly_Patients"))

# Delete a cohort
cohort_manager.delete_cohort("Treatment_A_Group")
print("\nCohorts after deletion:", cohort_manager.list_cohorts())

# Clean up example directory
# shutil.rmtree(exp_dir)
```

