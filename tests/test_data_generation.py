"""Test data generation utilities for cytoflow-qc testing."""

import numpy as np
import pandas as pd
from faker import Faker
from pathlib import Path


class TestDataGenerator:
    """Generate realistic test data for cytoflow-qc testing."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize with optional seed for reproducible data."""
        self.seed = seed
        self.faker = Faker()
        Faker.seed(seed)
        np.random.seed(seed)

    def generate_fcs_like_dataframe(
        self,
        n_events: int = 10000,
        channels: list[str] | None = None,
        add_noise: bool = True,
        add_correlations: bool = True,
    ) -> pd.DataFrame:
        """Generate a DataFrame that resembles FCS data.

        Args:
            n_events: Number of events to generate
            channels: List of channel names (defaults to common FCS channels)
            add_noise: Whether to add realistic noise patterns
            add_correlations: Whether to add realistic correlations between channels

        Returns:
            DataFrame with FCS-like data
        """
        if channels is None:
            channels = [
                "FSC-A", "FSC-H", "FSC-W",
                "SSC-A", "SSC-H", "SSC-W",
                "CD3-A", "CD19-A", "CD56-A",
                "CD4-A", "CD8-A", "CD45-A",
            ]

        data = {}

        # Generate each channel with appropriate distributions
        for channel in channels:
            if channel.startswith(("FSC", "SSC")):
                # Forward/side scatter - log-normal distribution
                data[channel] = np.random.lognormal(
                    mean=6 if "FSC" in channel else 5,
                    sigma=0.5,
                    size=n_events
                )
            else:
                # Fluorescence markers - log-normal with lower means
                data[channel] = np.random.lognormal(
                    mean=3.5 if "CD" in channel else 4.0,
                    sigma=0.8,
                    size=n_events
                )

        df = pd.DataFrame(data)

        if add_correlations:
            self._add_realistic_correlations(df)

        if add_noise:
            self._add_realistic_noise(df)

        return df

    def generate_samplesheet(
        self,
        n_samples: int = 5,
        output_dir: Path | None = None,
        batches: list[str] | None = None,
        conditions: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate a samplesheet CSV with realistic experimental metadata.

        Args:
            n_samples: Number of samples to generate
            output_dir: Directory to save CSV files (optional)
            batches: List of batch names (defaults to batch_1, batch_2, etc.)
            conditions: List of condition names (defaults to control, treatment)

        Returns:
            DataFrame with samplesheet data
        """
        if batches is None:
            batches = [f"batch_{i+1}" for i in range(max(2, n_samples // 3))]

        if conditions is None:
            conditions = ["control", "treatment", "vehicle"]

        samplesheet_data = []

        for i in range(n_samples):
            sample_id = f"sample_{i+1"03d"}"

            # Create sample data if output_dir is provided
            if output_dir:
                sample_file = output_dir / f"{sample_id}.csv"
                sample_data = self.generate_fcs_like_dataframe(n_events=1000)
                sample_data.to_csv(sample_file, index=False)
                file_path = str(sample_file)
            else:
                file_path = f"/path/to/{sample_id}.csv"

            # Generate metadata
            samplesheet_data.append({
                "sample_id": sample_id,
                "file_path": file_path,
                "batch": self.faker.random_element(batches),
                "condition": self.faker.random_element(conditions),
                "timepoint": self.faker.random_element(["T0", "T24", "T48", "T72"]),
                "replicate": self.faker.random_int(min=1, max=3),
                "notes": self.faker.sentence(nb_words=6) if self.faker.boolean(chance_of_getting_true=30) else "",
            })

        return pd.DataFrame(samplesheet_data)

    def generate_problematic_data(
        self,
        problem_types: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Generate datasets with various problematic characteristics for testing error handling.

        Args:
            problem_types: List of problem types to include

        Returns:
            Dictionary of problematic datasets
        """
        if problem_types is None:
            problem_types = ["empty", "missing_channels", "corrupted_values", "outliers"]

        datasets = {}

        if "empty" in problem_types:
            datasets["empty_sample"] = pd.DataFrame()

        if "missing_channels" in problem_types:
            # Missing required channels
            datasets["missing_fsc"] = pd.DataFrame({
                "SSC-A": np.random.lognormal(5, 0.5, 1000),
                "CD3-A": np.random.lognormal(4, 0.8, 1000),
            })

        if "corrupted_values" in problem_types:
            # Data with NaN, inf, and negative values
            data = self.generate_fcs_like_dataframe(n_events=1000)
            # Introduce problematic values
            data.iloc[0, 0] = np.nan
            data.iloc[1, 1] = np.inf
            data.iloc[2, 2] = -1
            datasets["corrupted_values"] = data

        if "outliers" in problem_types:
            # Data with extreme outliers
            data = self.generate_fcs_like_dataframe(n_events=1000)
            # Add extreme outliers
            data.loc[0, "FSC-A"] = 1e10
            data.loc[1, "CD3-A"] = 1e-10
            datasets["extreme_outliers"] = data

        return datasets

    def generate_batch_drift_data(
        self,
        n_batches: int = 3,
        n_samples_per_batch: int = 5,
        drift_magnitude: float = 0.2,
    ) -> dict[str, pd.DataFrame]:
        """Generate data with realistic batch-to-batch drift.

        Args:
            n_batches: Number of batches to generate
            n_samples_per_batch: Number of samples per batch
            drift_magnitude: Magnitude of drift between batches

        Returns:
            Dictionary of sample data with batch drift
        """
        datasets = {}

        for batch_idx in range(n_batches):
            batch_name = f"batch_{batch_idx + 1}"

            # Base parameters for this batch
            base_fsc_mean = 6.0 + (batch_idx * drift_magnitude)
            base_cd3_mean = 4.0 + (batch_idx * drift_magnitude * 0.5)

            for sample_idx in range(n_samples_per_batch):
                sample_id = f"{batch_name}_sample_{sample_idx + 1"02d"}"

                # Add some sample-to-sample variation
                sample_fsc_mean = base_fsc_mean + np.random.normal(0, 0.1)
                sample_cd3_mean = base_cd3_mean + np.random.normal(0, 0.1)

                n_events = 1000 + np.random.randint(-100, 100)  # Vary event count

                data = pd.DataFrame({
                    "FSC-A": np.random.lognormal(sample_fsc_mean, 0.5, n_events),
                    "FSC-H": np.random.lognormal(sample_fsc_mean, 0.5, n_events),
                    "SSC-A": np.random.lognormal(5.0, 0.5, n_events),
                    "CD3-A": np.random.lognormal(sample_cd3_mean, 0.8, n_events),
                    "CD19-A": np.random.lognormal(3.0, 0.8, n_events),
                })

                # Add correlations
                data["FSC-H"] = data["FSC-A"] * (1 + np.random.normal(0, 0.05, n_events))

                datasets[sample_id] = data

        return datasets

    def _add_realistic_correlations(self, df: pd.DataFrame) -> None:
        """Add realistic correlations between FCS channels."""
        # FSC-A and FSC-H should be highly correlated
        if "FSC-A" in df.columns and "FSC-H" in df.columns:
            correlation_factor = 1 + np.random.normal(0, 0.05, len(df))
            df["FSC-H"] = df["FSC-A"] * correlation_factor

        # SSC channels should correlate with each other
        if "SSC-A" in df.columns and "SSC-H" in df.columns:
            correlation_factor = 1 + np.random.normal(0, 0.08, len(df))
            df["SSC-H"] = df["SSC-A"] * correlation_factor

        # Some markers might correlate (e.g., CD4 and CD8 in T cells)
        if "CD4-A" in df.columns and "CD8-A" in df.columns:
            # Create negative correlation for some samples
            if np.random.random() > 0.5:
                df["CD8-A"] = df["CD8-A"] * (1 - df["CD4-A"] / df["CD4-A"].max() * 0.3)

    def _add_realistic_noise(self, df: pd.DataFrame) -> None:
        """Add realistic noise patterns to FCS data."""
        # Add multiplicative noise to fluorescence channels
        for col in df.columns:
            if col.startswith("CD") or col.startswith("SSC"):
                # Add small amount of multiplicative noise
                noise_factor = 1 + np.random.normal(0, 0.02, len(df))
                df[col] = df[col] * noise_factor

        # Add background noise
        for col in df.columns:
            if col.startswith("CD"):
                # Add small background level
                background = np.random.exponential(0.1, len(df))
                df[col] = df[col] + background


def create_test_dataset(
    dataset_type: str = "standard",
    n_samples: int = 5,
    n_events_per_sample: int = 1000,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Convenience function to create common test datasets.

    Args:
        dataset_type: Type of dataset to create ("standard", "problematic", "drift")
        n_samples: Number of samples to generate
        n_events_per_sample: Events per sample
        output_dir: Directory for samplesheet generation

    Returns:
        Dictionary of sample data
    """
    generator = TestDataGenerator()

    if dataset_type == "standard":
        data = {}
        for i in range(n_samples):
            sample_id = f"sample_{i+1"03d"}"
            data[sample_id] = generator.generate_fcs_like_dataframe(n_events_per_sample)
        return data

    elif dataset_type == "problematic":
        return generator.generate_problematic_data()

    elif dataset_type == "drift":
        return generator.generate_batch_drift_data(n_batches=3, n_samples_per_batch=n_samples)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_test_samplesheet(
    sample_data: dict[str, pd.DataFrame],
    output_path: Path | None = None,
) -> Path:
    """Create a samplesheet for test data.

    Args:
        sample_data: Dictionary of sample data
        output_path: Path to save samplesheet (optional)

    Returns:
        Path to created samplesheet
    """
    generator = TestDataGenerator()
    samplesheet_df = generator.generate_samplesheet(len(sample_data))

    if output_path:
        samplesheet_df.to_csv(output_path, index=False)
        return output_path
    else:
        # Return a temporary path
        import tempfile
        temp_path = Path(tempfile.mktemp(suffix=".csv"))
        samplesheet_df.to_csv(temp_path, index=False)
        return temp_path

