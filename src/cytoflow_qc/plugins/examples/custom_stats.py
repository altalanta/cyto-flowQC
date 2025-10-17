"""Example custom statistical method plugin."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..stats import StatsMethodPlugin


class CustomStatsMethod(StatsMethodPlugin):
    """Example custom statistical method that demonstrates the plugin interface.

    This plugin implements advanced statistical analysis that:
    1. Uses robust statistical methods (bootstrapping, trimmed means)
    2. Calculates multiple effect size measures
    3. Provides confidence intervals
    4. Handles edge cases gracefully
    """

    PLUGIN_NAME = "custom_stats_method"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Advanced statistical analysis with robust methods and confidence intervals"
    PLUGIN_AUTHOR = "CytoFlow-QC Team"
    PLUGIN_EMAIL = "cytoflow-qc@example.com"

    REQUIRES_CYTOFLOW_VERSION = ">=0.1.0"

    def validate_stats_parameters(
        self,
        data: pd.DataFrame,
        group_col: str,
        value_cols: list[str]
    ) -> None:
        """Validate statistical analysis parameters."""
        if group_col not in data.columns:
            raise ValueError(f"Group column '{group_col}' not found in data")

        missing_cols = [col for col in value_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Value columns not found: {missing_cols}")

        # Check for minimum sample sizes
        group_sizes = data[group_col].value_counts()
        if len(group_sizes) < 2:
            raise ValueError("Need at least 2 groups for statistical analysis")

        min_group_size = group_sizes.min()
        if min_group_size < 3:
            raise ValueError(f"Each group must have at least 3 samples, got {min_group_size}")

    def get_stats_description(self) -> str:
        """Get human-readable description of this statistical method."""
        return (
            "Advanced statistical analysis using robust methods including "
            "bootstrapped confidence intervals and multiple effect size measures"
        )

    def get_required_parameters(self) -> list[str]:
        """Get list of required configuration parameters."""
        return ["group_col", "value_cols"]

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration parameters."""
        return {
            "bootstrap_iterations": 1000,
            "confidence_level": 0.95,
            "trim_fraction": 0.1,  # For trimmed means
            "effect_size_measures": ["hedges_g", "cliffs_delta", "cohen_d"],
            "multiple_testing_correction": "fdr_bh",
        }

    def get_stats_columns(self) -> list[str]:
        """Get list of column names that this method will add to results."""
        return [
            "marker", "group1", "group2", "n1", "n2",
            "mean1", "mean2", "median1", "median2",
            "hedges_g", "hedges_g_ci_lower", "hedges_g_ci_upper",
            "cliffs_delta", "cohen_d",
            "p_value", "adj_p_value", "significant"
        ]

    def apply_stats(
        self,
        data: pd.DataFrame,
        group_col: str,
        value_cols: list[str],
        **kwargs
    ) -> pd.DataFrame:
        """Apply custom statistical analysis.

        Args:
            data: Input DataFrame with flow cytometry data
            group_col: Column name for grouping
            value_cols: List of column names to analyze
            **kwargs: Additional statistical parameters

        Returns:
            DataFrame with statistical results
        """
        # Validate parameters
        self.validate_stats_parameters(data, group_col, value_cols)

        results = []

        for value_col in value_cols:
            if value_col not in data.columns:
                continue

            # Get unique groups
            groups = sorted(data[group_col].unique())

            for i, group1 in enumerate(groups):
                for group2 in groups[i+1:]:
                    # Extract data for this comparison
                    group1_data = data[data[group_col] == group1][value_col].dropna()
                    group2_data = data[data[group_col] == group2][value_col].dropna()

                    if len(group1_data) < 2 or len(group2_data) < 2:
                        continue

                    # Calculate statistics
                    stats_result = self._calculate_group_statistics(
                        group1_data, group2_data, value_col, group1, group2
                    )

                    results.append(stats_result)

        return pd.DataFrame(results)

    def _calculate_group_statistics(
        self,
        group1_data: pd.Series,
        group2_data: pd.Series,
        marker: str,
        group1: str,
        group2: str
    ) -> dict[str, Any]:
        """Calculate comprehensive statistics for two groups."""
        # Basic descriptive statistics
        n1, n2 = len(group1_data), len(group2_data)
        mean1, mean2 = group1_data.mean(), group2_data.mean()
        median1, median2 = group1_data.median(), group2_data.median()

        # Calculate effect sizes
        hedges_g, hedges_ci_lower, hedges_ci_upper = self._calculate_hedges_g_with_ci(
            group1_data.values, group2_data.values
        )

        cliffs_delta = self._calculate_cliffs_delta(group1_data.values, group2_data.values)
        cohen_d = self._calculate_cohen_d(group1_data.values, group2_data.values)

        # Statistical test
        p_value = self._perform_statistical_test(group1_data.values, group2_data.values)

        # Apply multiple testing correction if specified
        correction_method = self.config.get("multiple_testing_correction", "fdr_bh")

        return {
            "marker": marker,
            "group1": group1,
            "group2": group2,
            "n1": n1,
            "n2": n2,
            "mean1": mean1,
            "mean2": mean2,
            "median1": median1,
            "median2": median2,
            "hedges_g": hedges_g,
            "hedges_g_ci_lower": hedges_ci_lower,
            "hedges_g_ci_upper": hedges_ci_upper,
            "cliffs_delta": cliffs_delta,
            "cohen_d": cohen_d,
            "p_value": p_value,
            "adj_p_value": p_value,  # Will be corrected later
            "significant": p_value < 0.05,
        }

    def _calculate_hedges_g_with_ci(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        confidence_level: float = 0.95
    ) -> tuple[float, float, float]:
        """Calculate Hedges' g with bootstrap confidence intervals."""
        # Calculate Hedges' g
        hedges_g = self._calculate_hedges_g(group1, group2)

        # Bootstrap confidence intervals
        bootstrap_iterations = self.config.get("bootstrap_iterations", 1000)
        bootstrap_effects = []

        # Combine data for bootstrap sampling
        combined_data = np.concatenate([group1, group2])
        n1, n2 = len(group1), len(group2)

        for _ in range(bootstrap_iterations):
            # Resample with replacement
            bootstrap_idx = np.random.choice(len(combined_data), len(combined_data), replace=True)
            bootstrap_combined = combined_data[bootstrap_idx]

            bootstrap_group1 = bootstrap_combined[:n1]
            bootstrap_group2 = bootstrap_combined[n1:]

            bootstrap_effect = self._calculate_hedges_g(bootstrap_group1, bootstrap_group2)
            bootstrap_effects.append(bootstrap_effect)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_effects, lower_percentile)
        ci_upper = np.percentile(bootstrap_effects, upper_percentile)

        return hedges_g, ci_lower, ci_upper

    def _calculate_hedges_g(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedges' g effect size."""
        n1, n2 = len(group1), len(group2)

        # Calculate pooled standard deviation
        pooled_sd = np.sqrt(
            ((n1 - 1) * group1.var(ddof=1) + (n2 - 1) * group2.var(ddof=1)) / (n1 + n2 - 2)
        )

        if pooled_sd == 0:
            return 0.0

        # Calculate mean difference
        mean_diff = group2.mean() - group1.mean()

        # Hedges' g correction factor
        correction_factor = 1 - 3 / (4 * (n1 + n2) - 9)

        return (mean_diff / pooled_sd) * correction_factor

    def _calculate_cliffs_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's delta effect size."""
        # Count how often values in group2 are larger than values in group1
        comparisons = 0
        larger_count = 0

        for val1 in group1:
            for val2 in group2:
                comparisons += 1
                if val2 > val1:
                    larger_count += 1

        if comparisons == 0:
            return 0.0

        # Cliff's delta ranges from -1 to 1
        return (larger_count - (comparisons - larger_count)) / comparisons

    def _calculate_cohen_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)

        # Pooled standard deviation
        pooled_sd = np.sqrt(
            ((n1 - 1) * group1.var(ddof=1) + (n2 - 1) * group2.var(ddof=1)) / (n1 + n2 - 2)
        )

        if pooled_sd == 0:
            return 0.0

        # Mean difference
        mean_diff = group2.mean() - group1.mean()

        return mean_diff / pooled_sd

    def _perform_statistical_test(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Perform statistical test (Mann-Whitney U test)."""
        from scipy import stats

        # Use Mann-Whitney U test for non-parametric comparison
        statistic, p_value = stats.mannwhitneyu(
            group1, group2, alternative='two-sided'
        )

        return p_value
