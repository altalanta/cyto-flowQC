# Methods Overview

## Data ingestion

FCS files are read with `flowkit` when available and fall back to `fcsparser`. For testing and documentation the project also accepts CSV surrogates with the same channel headers. Channel canonicalisation is handled through the `channels` map in the YAML configuration.

## Compensation

CytoFlow-QC extracts spillover matrices from FCS `$SPILLOVER` keywords or an explicit CSV path. Compensation is applied as:

```
C = X · S⁻¹
```

where `X` is the matrix of fluorescence intensities and `S` is the spillover matrix restricted to matched channels. Matrices must be square and invertible; otherwise compensation is skipped and a warning is logged.

## Quality control

Event-level QC flags include:

- **Debris:** low-percentile FSC/SSC thresholds remove small particles.
- **Doublets:** deviation from the FSC-H ~ FSC-A linear relationship.
- **Saturation:** proportion of events hitting detector maximum.

Per-sample summaries aggregate these metrics plus signal statistics (median, IQR, dynamic range). Samples can be flagged as outliers for manual review.

## Automated gating

The default gating chain is debris → singlets → lymphocytes → optional viability. Density-based rules rely on quantiles and Gaussian mixture models to identify the dominant population. Gate parameters are recorded for reproducibility.

## Batch drift

Each gated sample contributes a feature vector (population fractions, medians, IQRs). CytoFlow-QC runs ANOVA/Kruskal tests across batches, projects samples with PCA, and optionally computes a 2D UMAP embedding. Significant features (Holm-adjusted `p < 0.05`) indicate drift.

## Statistics

Group comparisons compute effect sizes (Hedges' g, Cliff's delta, Glass' delta) and Mann–Whitney tests with Holm–Bonferroni correction. Outputs are tidy tables suitable for downstream plotting or reporting.

## Reporting

Results are rendered through a Jinja2-driven HTML template that collects QC tables, gating plots, drift figures, and stats summaries. The template can be swapped or extended and optionally driven through the reference notebook under `notebooks/report_notebook.ipynb`.
