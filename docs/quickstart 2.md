# Quickstart

This walkthrough uses the bundled synthetic CSV surrogates to exercise the full pipeline without large FCS assets.

## 1. Setup

```bash
mamba env create -f env/environment.yml
mamba activate cytoflow-qc
pip install -e .[dev]
pre-commit install
```

## 2. Inspect inputs

```bash
ls samplesheets/
cat samplesheets/example_samplesheet.csv
```

Update `configs/example_config.yaml` if your channel names differ.

## 3. Run the CLI

```bash
cytoflow-qc run \
  --samplesheet samplesheets/example_samplesheet.csv \
  --config configs/example_config.yaml \
  --out results
```

## 4. Review artifacts

- `results/qc/summary.csv` – event-level QC metrics
- `results/gate/summary.csv` – gated counts
- `results/drift/` – batch drift statistics & PCA plot
- `results/stats/effect_sizes.csv`
- `results/report.html`

## 5. Re-run with updated parameters

The pipeline is deterministic – re-running with the same inputs produces identical outputs. Edit YAML configs or supply different samplesheets to explore new gates or heuristics.
