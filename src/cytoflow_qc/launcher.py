"""Interactive Streamlit-based pipeline launcher for CytoFlow-QC."""
import streamlit as st
import subprocess
import tempfile
from pathlib import Path

def run_pipeline(samplesheet_path, config_path, out_dir, workers):
    """Runs the cytoflow-qc pipeline and streams the output."""
    command = [
        "poetry", "run", "cytoflow-qc", "run",
        "--samplesheet", str(samplesheet_path),
        "--config", str(config_path),
        "--out", str(out_dir),
        "--workers", str(workers),
    ]
    
    st.info(f"Running command: `{' '.join(command)}`")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    log_container = st.empty()
    log_content = ""
    
    while True:
        line = process.stdout.readline()
        if not line:
            break
        log_content += line
        log_container.code(log_content, language="log")
        
    process.wait()
    
    if process.returncode == 0:
        st.success("Pipeline completed successfully!")
    else:
        st.error(f"Pipeline failed with exit code {process.returncode}.")

def main():
    st.set_page_config(page_title="CytoFlow-QC Launcher", layout="wide")
    st.title("🔬 CytoFlow-QC Pipeline Launcher")

    st.sidebar.header("Pipeline Configuration")
    
    # --- File Uploads ---
    samplesheet_file = st.sidebar.file_uploader(
        "Upload Samplesheet (CSV)", type=["csv"]
    )
    config_file = st.sidebar.file_uploader(
        "Upload Configuration (YAML)", type=["yaml", "yml"]
    )
    
    # --- Parameters ---
    out_dir = st.sidebar.text_input("Output Directory", value="results/interactive_run")
    workers = st.sidebar.slider("Number of Workers", min_value=1, max_value=16, value=4)
    
    # --- Run Button ---
    run_button = st.sidebar.button("🚀 Run Pipeline", disabled=(not samplesheet_file or not config_file))
    
    if run_button:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            samplesheet_path = temp_path / samplesheet_file.name
            with open(samplesheet_path, "wb") as f:
                f.write(samplesheet_file.getvalue())
                
            config_path = temp_path / config_file.name
            with open(config_path, "wb") as f:
                f.write(config_file.getvalue())
            
            run_pipeline(samplesheet_path, config_path, out_dir, workers)

if __name__ == "__main__":
    main()
