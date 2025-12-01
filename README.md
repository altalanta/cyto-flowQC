plea# CytoFlow-QC

Automated, reproducible quality control, compensation, and gating for flow cytometry experiments. CytoFlow-QC replaces fragile manual FlowJo pipelines with a scriptable workflow that ingests FCS batches, applies spillover compensation, performs automated QC + gating, detects batch drift, runs basic effect-size statistics, and emits a publication-ready HTML report.

## Installation

For the most reliable and reproducible results, we recommend running the pipeline using Docker.

### Docker (Recommended)

1.  **Build the Docker image:**
    ```bash
    docker build -t cytoflow-qc .
    ```

2.  **Run the pipeline:**
    Mount your local data, configuration, and results directories into the container.

    ```bash
    docker run --rm \
      -v "$(pwd)/data:/app/data" \
      -v "$(pwd)/configs:/app/configs" \
      -v "$(pwd)/results:/app/results" \
      cytoflow-qc run --samplesheet /app/data/samplesheet.csv --config /app/configs/config.yaml --out /app/results
    ```

### Local Installation with Poetry

For development or if you prefer a local installation:

1.  **Install Poetry:**
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
2.  **Install dependencies:**
    ```bash
    poetry install
    ```
3.  **Run the pipeline:**
    ```bash
    poetry run cytoflow-qc run ...
    ```

## Quickstart

1. Drop FCS files (or the provided synthetic CSV surrogates) into `data/raw/`.
2. Copy/modify `samplesheets/example_samplesheet.csv` for your experiment.
3. Tune `configs/example_config.yaml` for channel names and QC heuristics.
4. Run the end-to-end pipeline:
   ```bash
   cytoflow-qc run --samplesheet samplesheets/example_samplesheet.csv \
     --config configs/example_config.yaml --out results
   ```
5. Open `results/report.html` to review the interactive report, which includes sortable tables and explorable plots for QC, gating, drift, and effect-size outputs.
6. **Optional:** Launch the live analysis dashboard for deeper exploration:
   ```bash
   cytoflow-qc dashboard --indir results/
   ```

## Validating Inputs

Before running a full analysis, you can validate your `samplesheet.csv` and `config.yaml` to catch common errors early:

```bash
cytoflow-qc validate --samplesheet path/to/samplesheet.csv --config path/to/config.yaml
```

You can also perform a "dry run" of the main command, which will run the same validation checks and then exit:

```bash
cytoflow-qc run --dry-run --samplesheet ... --config ...
```

## Interactive Configuration

To simplify the setup process, you can use the interactive configuration generator. This tool will ask you a series of questions and create a `config.yaml` file based on your answers.

```bash
cytoflow-qc configure
```

## Interactive Pipeline Launcher

For a user-friendly graphical interface, you can use the interactive launcher:

```bash
cytoflow-qc launch
```

This will open a web application in your browser where you can upload your samplesheet and configuration files, adjust parameters, and run the pipeline with real-time log output.

## Extending CytoFlow-QC with Plugins

CytoFlow-QC can be extended with custom plugins for gating strategies, QC methods, and more. To simplify the process of creating a new plugin, you can use the built-in scaffolding tool:

```bash
cytoflow-qc plugins create
```

This command will launch an interactive questionnaire that guides you through the process of creating a new, installable plugin package with all the necessary boilerplate. For more details on the plugin architecture, see the [Plugin Development Guide](plugins/README.md).

## Reproducibility with DVC

This project uses [DVC](https://dvc.org) to version data and models, ensuring that all results are reproducible. The `dvc.yaml` file defines the pipeline, and you can reproduce the results by running:

```bash
dvc repro
```

### Data Provenance

For every pipeline run executed via `cytoflow-qc run`, a `provenance.json` file is generated in the root of the output directory. This file serves as a detailed audit trail for the analysis, capturing a complete snapshot of the execution environment to ensure full reproducibility. It includes:

-   **Timestamp:** The exact time the run was initiated.
-   **Platform Information:** Details about the operating system and Python version.
-   **Git Status:** The exact Git commit hash of the code that was executed, including whether the repository was "dirty" (had uncommitted changes).
-   **DVC Status:** Hashes of all DVC-tracked data files and parameters, ensuring the inputs are fully versioned.
-   **Inputs:** The complete content of the samplesheet and the configuration file used for the run.
-   **Dependencies:** A frozen list of all Python packages and their exact versions as captured from the environment.

This file is crucial for debugging, auditing, and faithfully reproducing a specific analysis, even years later.

## Methods (brief)

- **Compensation:** auto-detect `$SPILLOVER` matrices or accept external CSV overrides. Compensation is applied only to matched fluorescence channels.
- **QC:** debris removal via low-percentile FSC/SSC thresholds, doublet detection via FSC-H vs FSC-A deviation, saturation + dynamic range metrics, and aggregate per-sample summaries.
- **Gating:** default strategy composes debris removal, singlet selection, lymphocyte density gate, and optional viability dye thresholding. Additional strategies can plug into `gate.auto_gate`.
- **Batch drift:** per-sample feature tables (medians, IQRs, gated fractions) feed ANOVA/Kruskal tests and PCA/UMAP projections to flag drifted batches.
- **Statistics:** effect sizes (Hedges' g, Cliff's delta) and Mann–Whitney tests across experimental conditions with Holm–Bonferroni correction.
- **Reporting:** static plots (Matplotlib) and tables rendered into HTML via Jinja2 templates or nbconvert workflows.

## Outputs

A typical single-run layout under `results/`:

```
results/
├─ ingest/
│  ├─ manifest.csv
│  └─ events/<sample>.parquet
├─ compensate/
│  └─ events/<sample>.parquet
├─ qc/
│  ├─ events/<sample>.parquet
│  └─ summary.csv
├─ gate/
│  ├─ events/<sample>.parquet
│  ├─ params/<sample>.json
│  └─ summary.csv
├─ drift/
│  ├─ features.csv
│  ├─ tests.csv
│  ├─ pca.csv
│  └─ figures/*.png
├─ stats/
│  └─ effect_sizes.csv
├─ exports/                    # New: Exported data and figures
│  ├─ qc_summary.png
│  ├─ gating_analysis.pdf
│  └─ cytoflow_qc_data.zip
├─ interactive_report.html     # New: Interactive dashboard
└─ report.html
```

## Interactive Visualization

CytoFlow-QC now includes powerful interactive visualization capabilities:

### Web Dashboard

Launch an interactive web dashboard for exploring results:

```bash
cytoflow-qc dashboard --indir results/
```

**Features:**
- **Overview Dashboard**: Key metrics and pipeline summary
- **Quality Control Analysis**: Interactive QC metrics and pass/fail distributions
- **3D Gating Visualization**: Multi-parameter scatter plots with gating inspection
- **Batch Drift Analysis**: Interactive PCA plots and statistical test results
- **Statistical Analysis**: Effect size plots and volcano plots
- **Export Options**: High-resolution figures and data export

### 3D Visualizations

Create interactive 3D gating visualizations:

```bash
cytoflow-qc viz3d --indir results/ --sample sample_001 --output 3d_gating.html
```

### Export Capabilities

Export publication-ready figures:

```bash
# Export QC summary as PDF
cytoflow-qc export --data results/qc/summary.csv --output qc_summary.pdf --format pdf

# Export 3D gating visualization
cytoflow-qc export_3d --indir results/ --sample sample_001 --output gating_3d.html

# Export complete interactive dashboard
cytoflow-qc export_dashboard --indir results/ --output full_report.html

## Advanced Features

### Machine Learning Integration

CytoFlow-QC now includes machine learning-based gating strategies:

```bash
# Use ML-based gating
cytoflow-qc gate --indir results/qc --out results/gate --strategy lymphocyte_gating --config config.yaml

# Use anomaly detection
cytoflow-qc gate --indir results/qc --out results/gate --strategy anomaly_detection --config config.yaml

# Use Bayesian optimization for parameter tuning
cytoflow-qc gate --indir results/qc --out results/gate --strategy bayesian_optimization --config config.yaml
```

### Cloud-Native Deployment

Deploy cytoflow-qc to cloud platforms for scalability:

```bash
# Deploy to Kubernetes
cytoflow-qc cloud deploy --provider kubernetes --config k8s-config.yaml

# Use serverless processing
cytoflow-qc cloud process --provider aws --function my-function --data s3://bucket/data/
```

### Real-Time Processing

Process data in real-time as it's acquired:

```bash
# Start real-time processing
cytoflow-qc realtime start --ws-url ws://instrument:8080 --output results/

# Monitor processing in real-time
cytoflow-qc realtime monitor --dashboard-port 8081
```

### High-Performance Computing

Optimize for large datasets:

```bash
# Use streaming processing for large files
cytoflow-qc process --streaming --chunk-size 50000 --input large_file.csv

# Enable parallel processing
cytoflow-qc process --parallel --workers 8 --input data_directory/
```

### Custom Plugin Development

Extend cytoflow-qc with custom plugins:

```bash
# List available plugins
cytoflow-qc plugins list

# Use custom gating strategy
cytoflow-qc gate --strategy my_custom_plugin --config plugins.yaml

# Develop new plugins
cytoflow-qc plugins create --template gating --name my_plugin
```

## Extending

- Custom gates: add new primitives in `src/cytoflow_qc/gate.py` and expose them via the Typer CLI options.
- Additional QC heuristics: extend `src/cytoflow_qc/qc.py` and accompanying config schema.
- Alternate reporting: adapt `configs/report_template.html.j2` or drive a Quarto/nbconvert notebook via `notebooks/report_notebook.ipynb`.

## Citation

If you use CytoFlow-QC in a publication, please cite this repository and include the MIT license notice.

## License

MIT License – see `LICENSE` for details.
