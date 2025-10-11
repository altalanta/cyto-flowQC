# Validation Notes

A lightweight synthetic dataset accompanies the test suite. It encodes four samples across two batches with known gated fractions. The pipeline smoke test asserts:

- deterministic QC flag counts per sample
- stable gated event counts across runs
- drift detection correctly flags the deliberately shifted batch
- effect-size tables exist and contain Holm-adjusted p-values

For real-world validation, compare CytoFlow-QC outputs against manual FlowJo gates on a representative experiment:

1. Export event tables for a small subset of files.
2. Run the `cytoflow-qc` pipeline and collect the gated CSVs from `results/gate/events/`.
3. Compute agreement metrics (precision/recall on gated populations). A concordance above 0.9 indicates the heuristics align well.
4. Inspect the PCA plots in `results/drift/figures/` to ensure batches cluster as expected.

Future work includes a larger validation set covering high-parameter panels and alternative cytometers.
