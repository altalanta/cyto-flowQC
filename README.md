plea# CytoFlow-QC

Automated, reproducible quality control, compensation, and gating for flow cytometry experiments. CytoFlow-QC replaces fragile manual FlowJo pipelines with a scriptable workflow that ingests FCS batches, applies spillover compensation, performs automated QC + gating, detects batch drift, runs basic effect-size statistics, and emits a publication-ready HTML report.

## Install

### Poetry (Recommended)

1. Install Poetry:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
2. Configure Poetry to create virtual environments in the project's root:
   ```bash
   poetry config virtualenvs.in-project true
   ```
3. Install dependencies:
   ```bash
   poetry install --with dev
   ```

### Docker

```bash
docker build -t cytoflow-qc .
docker run --rm -v $(pwd):/workspace cytoflow-qc cytoflow-qc --help
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

## Reproducibility with DVC

This project uses [DVC](https://dvc.org) to version data and create a reproducible pipeline. To use it:

1.  **Install DVC:** If you haven't already, install DVC with the necessary cloud provider support:
    ```bash
    pip install "dvc[s3,gcs]"
    ```
2.  **Configure a remote:** Set up a remote storage location for your data (e.g., S3, GCS, or a local directory).
    ```bash
    dvc remote add -d myremote s3://my-bucket/cytoflow-qc
    ```
3.  **Retrieve data:** Pull the versioned data from the remote storage.
    ```bash
    dvc pull
    ```
4.  **Reproduce the pipeline:** Run the full pipeline as defined in `dvc.yaml`.
    ```bash
    dvc repro
    ```
5.  **Push new data:** If you update the data or results, push them to the remote.
    ```bash
    dvc push
    ```

Reusable Make targets:

- `make setup` – create the poetry env and install pre-commit hooks.
- `make lint` / `make format` – ruff + black.
- `make test` – pytest suite with synthetic data.
- `make smoke` – one-shot CLI run that checks the HTML report exists.
- `make report` – rebuild the HTML report from existing artifacts.
- `make dashboard` – launch the interactive web dashboard (requires results/ directory).

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
```

## Extending

- Custom gates: add new primitives in `src/cytoflow_qc/gate.py` and expose them via the Typer CLI options.
- Additional QC heuristics: extend `src/cytoflow_qc/qc.py` and accompanying config schema.
- Alternate reporting: adapt `configs/report_template.html.j2` or drive a Quarto/nbconvert notebook via `notebooks/report_notebook.ipynb`.

## Citation

If you use CytoFlow-QC in a publication, please cite this repository and include the MIT license notice.

## License

MIT License – see `LICENSE` for details.
