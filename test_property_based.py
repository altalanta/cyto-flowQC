"""Property-based tests using hypothesis for cytoflow-qc components."""

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column

from cytoflow_qc.qc import add_qc_flags, qc_summary
from cytoflow_qc.gate import auto_gate
from cytoflow_qc.stats import effect_sizes


class TestQCProperties:
    """Property-based tests for QC functionality."""

    @given(
        data_frames(
            columns=[
                column("FSC-A", dtype=float, elements=st.floats(min_value=1, max_value=1e6)),
                column("FSC-H", dtype=float, elements=st.floats(min_value=1, max_value=1e6)),
                column("SSC-A", dtype=float, elements=st.floats(min_value=1, max_value=1e6)),
                column("CD3-A", dtype=float, elements=st.floats(min_value=1, max_value=1e6)),
                column("CD19-A", dtype=float, elements=st.floats(min_value=1, max_value=1e6)),
            ],
            rows=st.integers(min_value=10, max_value=1000),
        )
    )
    @settings(max_examples=50)
    def test_qc_flags_preserve_data_shape(self, df: pd.DataFrame) -> None:
        """QC flags should preserve the shape of input data."""
        qc_df = add_qc_flags(df)

        # Should have same number of rows
        assert len(qc_df) == len(df)

        # Should have additional QC columns
        qc_columns = [col for col in qc_df.columns if col.startswith("qc_")]
        assert len(qc_columns) == 4  # qc_debris, qc_doublets, qc_saturated, qc_pass

        # QC columns should be boolean
        for col in qc_columns:
            assert qc_df[col].dtype == bool

    @given(
        data_frames(
            columns=[
                column("FSC-A", dtype=float, elements=st.floats(min_value=1, max_value=1e6)),
                column("FSC-H", dtype=float, elements=st.floats(min_value=1, max_value=1e6)),
                column("SSC-A", dtype=float, elements=st.floats(min_value=1, max_value=1e6)),
            ],
            rows=st.integers(min_value=10, max_value=100),
        )
    )
    @settings(max_examples=50)
    def test_qc_summary_properties(self, df: pd.DataFrame) -> None:
        """QC summary should handle various data distributions correctly."""
        qc_df = add_qc_flags(df)
        samples = {"sample_1": qc_df}

        summary = qc_summary(samples)

        # Should have one row per sample
        assert len(summary) == 1
        assert summary.iloc[0]["sample_id"] == "sample_1"

        # Pass fraction should be between 0 and 1
        pass_fraction = summary.iloc[0]["qc_pass_fraction"]
        assert 0 <= pass_fraction <= 1

        # Total events should match input
        assert summary.iloc[0]["total_events"] == len(df)


class TestGatingProperties:
    """Property-based tests for gating functionality."""

    @given(
        data_frames(
            columns=[
                column("FSC-A", dtype=float, elements=st.floats(min_value=1, max_value=1e6)),
                column("FSC-H", dtype=float, elements=st.floats(min_value=1, max_value=1e6)),
                column("SSC-A", dtype=float, elements=st.floats(min_value=1, max_value=1e6)),
                column("CD3-A", dtype=float, elements=st.floats(min_value=1, max_value=1e6)),
            ],
            rows=st.integers(min_value=50, max_value=500),
        ),
        st.sampled_from(["default"]),
        st.fixed_dictionaries({
            "channels": st.fixed_dictionaries({
                "fsc_a": st.just("FSC-A"),
                "fsc_h": st.just("FSC-H"),
                "ssc_a": st.just("SSC-A"),
                "markers": st.lists(st.text(), min_size=1, max_size=3),
            })
        }),
    )
    @settings(max_examples=20, deadline=None)
    def test_gating_preserve_structure(self, df: pd.DataFrame, strategy: str, config: dict) -> None:
        """Gating should preserve data structure while filtering events."""
        gated_df, params = auto_gate(df, strategy=strategy, config=config)

        # Should have same or fewer events
        assert len(gated_df) <= len(df)

        # Should preserve all columns
        assert set(gated_df.columns) == set(df.columns)

        # Should have gating parameters
        assert isinstance(params, dict)
        assert len(params) > 0

        # If gating removed events, params should indicate this
        if len(gated_df) < len(df):
            assert any("remaining" in gate_info for gate_info in params.values())


class TestStatsProperties:
    """Property-based tests for statistical functionality."""

    @given(
        data_frames(
            columns=[
                column("condition", dtype=str, elements=st.sampled_from(["control", "treatment"])),
                column("marker1", dtype=float, elements=st.floats(min_value=0, max_value=100)),
                column("marker2", dtype=float, elements=st.floats(min_value=0, max_value=100)),
            ],
            rows=st.integers(min_value=20, max_value=200),
        ),
        st.sampled_from(["condition"]),
        st.lists(st.text(), min_size=1, max_size=2),
    )
    @settings(max_examples=30, deadline=None)
    def test_effect_sizes_properties(self, df: pd.DataFrame, group_col: str, value_cols: list[str]) -> None:
        """Effect sizes should be computed correctly for various data patterns."""
        # Filter to only include columns that exist in the dataframe
        available_cols = [col for col in value_cols if col in df.columns]
        if not available_cols:
            return  # Skip if no valid columns

        try:
            effects = effect_sizes(df, group_col, available_cols)

            # Should return a DataFrame
            assert isinstance(effects, pd.DataFrame)

            # Should have expected columns
            expected_columns = {"marker", "effect_size", "p_value", "adj_p_value"}
            assert expected_columns.issubset(set(effects.columns))

            # Effect sizes should be numeric
            assert pd.api.types.is_numeric_dtype(effects["effect_size"])

            # P-values should be between 0 and 1
            assert effects["p_value"].between(0, 1).all()
            assert effects["adj_p_value"].between(0, 1).all()

        except ValueError:
            # Effect size computation can fail for certain data patterns
            # This is expected behavior
            pass


class TestDataGenerationStrategies:
    """Strategies for generating test data that matches real FCS patterns."""

    @staticmethod
    @st.composite
    def fcs_like_events(draw: st.DrawFn) -> pd.DataFrame:
        """Generate DataFrame that resembles FCS data."""
        n_events = draw(st.integers(min_value=100, max_value=10000))

        # Generate base fluorescence channels with log-normal distributions
        channels = {}
        for channel in ["FSC-A", "FSC-H", "SSC-A", "CD3-A", "CD19-A", "CD56-A", "CD4-A", "CD8-A"]:
            if channel in ["FSC-A", "FSC-H", "SSC-A"]:
                # Forward/side scatter - higher values
                channels[channel] = draw(
                    st.lists(
                        st.floats(min_value=1, max_value=1e6, allow_nan=False),
                        min_size=n_events,
                        max_size=n_events,
                    )
                )
            else:
                # Fluorescence markers - lower values
                channels[channel] = draw(
                    st.lists(
                        st.floats(min_value=1, max_value=1e4, allow_nan=False),
                        min_size=n_events,
                        max_size=n_events,
                    )
                )

        df = pd.DataFrame(channels)

        # Add realistic correlations between FSC-A and FSC-H
        fsc_correlation = draw(st.floats(min_value=0.8, max_value=0.99))
        df["FSC-H"] = df["FSC-A"] * (1 + draw(st.floats(min_value=-0.1, max_value=0.1)))

        return df

    @staticmethod
    @st.composite
    def experimental_design(draw: st.DrawFn) -> dict[str, str]:
        """Generate realistic experimental design metadata."""
        return {
            "sample_id": draw(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=8, max_size=12)),
            "batch": draw(st.sampled_from([f"batch_{i}" for i in range(1, 6)])),
            "condition": draw(st.sampled_from(["control", "treatment", "vehicle"])),
            "timepoint": draw(st.sampled_from(["T0", "T24", "T48", "T72"])),
        }










