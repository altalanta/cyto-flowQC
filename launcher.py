"""Interactive Streamlit-based dashboard for CytoFlow-QC."""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import logging

from cytoflow_qc.pipeline import IngestionStage, CompensationStage, QCStage
from cytoflow_qc.gate import _default_gating_strategy
from cytoflow_qc.viz import plot_gating_scatter
from cytoflow_qc.config import load_and_validate_config, AppConfig
from cytoflow_qc.model import StageResult, Sample
from cytoflow_qc.io import load_samplesheet, stream_events

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def load_data_for_sample(sample: Sample) -> pd.DataFrame:
    """Loads all events for a single sample into a DataFrame."""
    return pd.concat(list(stream_events(sample.events_file)))

@st.cache_data
def run_preprocessing_pipeline(_samplesheet_bytes, _config_bytes, output_dir):
    """
    Runs the initial stages of the pipeline (ingest, compensate, qc)
    and returns the result. Caching to avoid re-running on every interaction.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ss_path = temp_path / "samplesheet.csv"
        ss_path.write_bytes(_samplesheet_bytes)
        
        cfg_path = temp_path / "config.yml"
        cfg_path.write_bytes(_config_bytes)
        
        config = load_and_validate_config(cfg_path)
        
        ingest_dir = output_dir / "ingest"
        compensate_dir = output_dir / "compensate"
        qc_dir = output_dir / "qc"
        
        st.write("Running Ingestion...")
        ingest_stage = IngestionStage(outdir=ingest_dir, samplesheet_path=ss_path, config=config)
        ingest_result = ingest_stage.run()

        st.write("Running Compensation...")
        compensate_stage = CompensationStage(outdir=compensate_dir, workers=1, spill=config.compensation.matrix_file)
        compensate_result = compensate_stage.run(ingest_result)

        st.write("Running QC...")
        qc_stage = QCStage(outdir=qc_dir, workers=1, qc_config=config.qc)
        qc_result = qc_stage.run(compensate_result)
        
        st.success("Preprocessing complete!")
        return qc_result, config

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="CytoFlow-QC Dashboard", layout="wide")
    st.title("🔬 CytoFlow-QC Interactive Dashboard")

    # --- Session State Initialization ---
    if 'pipeline_result' not in st.session_state:
        st.session_state.pipeline_result = None
    if 'app_config' not in st.session_state:
        st.session_state.app_config = None

    # --- Sidebar for Setup ---
    st.sidebar.header("Setup")
    samplesheet_file = st.sidebar.file_uploader("Upload Samplesheet (CSV)", type=["csv"])
    config_file = st.sidebar.file_uploader("Upload Configuration (YAML)", type=["yaml", "yml"])

    if st.sidebar.button("Process Data", disabled=(not samplesheet_file or not config_file)):
        output_dir = Path(tempfile.mkdtemp(prefix="cytoflow-qc-interactive-"))
        with st.spinner("Processing initial pipeline stages..."):
            result, config = run_preprocessing_pipeline(
                samplesheet_file.getvalue(), 
                config_file.getvalue(), 
                output_dir
            )
            st.session_state.pipeline_result = result
            st.session_state.app_config = config
    
    # --- Main Content ---
    if not st.session_state.pipeline_result:
        st.info("Upload your samplesheet and configuration, then click 'Process Data' to begin.")
        return

    st.header("Interactive Gating")
    
    result: StageResult = st.session_state.pipeline_result
    config: AppConfig = st.session_state.app_config
    
    sample_options = {s.sample_id: s for s in result.samples}
    selected_sample_id = st.selectbox("Choose a sample to analyze:", list(sample_options.keys()))
    
    if selected_sample_id:
        sample = sample_options[selected_sample_id]
        
        # Load data for the selected sample
        with st.spinner(f"Loading data for {selected_sample_id}..."):
            original_df = load_data_for_sample(sample)
        
        st.subheader("Gating Parameters")
        percentile = st.slider(
            "Density Gate Percentile",
            min_value=1,
            max_value=100,
            value=config.gating.lymphocytes.percentile,
            help="Determines the density threshold. A higher percentile creates a more stringent gate."
        )

        # Re-run gating and plotting on parameter change
        gate_config = config.gating.model_dump()
        gate_config["channels"] = config.channels.model_dump()
        gate_config["lymphocytes"]["percentile"] = percentile

        # Apply QC filters before gating
        df_clean = original_df[~original_df["qc_debris"] & ~original_df["qc_doublet"]].copy()

        gated_df, params = _default_gating_strategy(df_clean, gate_config)

        st.subheader("Gating Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Events Passing QC",
                f"{len(df_clean):,}",
                f"{(len(df_clean) / len(original_df)):.2%}"
            )
            st.metric(
                "Events in Gate",
                f"{len(gated_df):,}",
                f"{(len(gated_df) / len(df_clean) if len(df_clean) > 0 else 0):.2%} of QC'd events"
            )
            st.write("Gating Parameters:")
            st.json(params)
            
        with col2:
            fig = plot_gating_scatter(
                df_clean,
                gated_df,
                fsc_channel=config.channels.fsc_a,
                ssc_channel=config.channels.ssc_a
            )
            st.pyplot(fig)
            plt.close(fig) # Avoid keeping figures in memory

if __name__ == "__main__":
    main()





