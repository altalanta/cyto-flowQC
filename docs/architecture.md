# CytoFlow-QC Architecture Overview

This document provides a high-level overview of the `cytoflow-qc` architecture, outlining its core components, their responsibilities, and how they interact to facilitate quality control and analysis of flow cytometry data.

## 1. Core Concepts

`cytoflow-qc` is designed around a modular and extensible architecture, primarily focused on processing and analyzing flow cytometry data for quality control. Key concepts include:

*   **Data Connectors**: Modules responsible for ingesting flow cytometry data from various sources (e.g., FCS files, CSVs).
*   **Pipeline**: A sequence of operations (compensation, gating, QC, statistics, reporting) applied to the data.
*   **Plugins**: Extensible components that allow users to customize gating strategies, QC methods, and statistical analyses.
*   **Configuration**: YAML-based configuration files define the pipeline steps, parameters, and plugin settings.
*   **Reporting**: Generation of comprehensive HTML reports summarizing QC and analysis results.

## 2. Main Components

The `cytoflow-qc` codebase is structured into several key directories and modules:

### `src/cytoflow_qc/`

This directory contains the core Python modules of the `cytoflow-qc` library.

*   **`cli.py`**: Command-Line Interface. Handles parsing command-line arguments and orchestrating the execution of the main pipeline.
*   **`io.py`**: Input/Output operations. Manages reading and writing of data (e.g., CSV, FCS) and configuration files.
*   **`data_connectors.py`**: Defines interfaces and implementations for connecting to different data sources.
*   **`compensate.py`**: Implements algorithms for compensation, correcting for spectral overlap in flow cytometry data.
*   **`gate.py`**: Contains logic for applying gating strategies to identify cell populations. This module heavily leverages the plugin system.
*   **`qc.py`**: Implements various quality control (QC) methods to assess data quality and identify problematic samples. This module also uses the plugin system.
*   **`stats.py`**: Provides statistical analysis functions, including calculations for drift and effect sizes. Integrates with statistical plugins.
*   **`report.py` and `report_generator.py`**: Responsible for generating the final HTML reports, aggregating results from various pipeline stages.
*   **`viz.py` and `viz_3d.py`**: Visualization utilities for plotting flow cytometry data and QC results.
*   **`plugins/`**: A sub-directory dedicated to the plugin architecture.
    *   **`base.py`**: Defines abstract base classes for different plugin types (gating, QC, stats).
    *   **`registry.py`**: Manages the discovery, loading, and registration of custom plugins.
    *   **`gating.py`**: Specific plugin interface for gating strategies.
    *   **`qc.py`**: Specific plugin interface for QC methods.
    *   **`stats.py`**: Specific plugin interface for statistical methods.
    *   **`examples/`**: Contains example implementations of custom plugins.
*   **`utils.py`**: General utility functions used across the codebase.
*   **`api.py`**: Defines the public API for programmatic interaction with `cytoflow-qc`.
*   **`experiment_design.py`**: Logic related to experimental design and metadata handling.
*   **`security.py`**: Handles security aspects, potentially including data access control or sensitive information handling.
*   **`high_performance.py`**: Contains code optimized for performance, likely for processing large datasets efficiently.
*   **`realtime.py`**: Functionality related to real-time data processing or monitoring.
*   **`interactive_viz.py`**: Interactive visualization components, possibly for Jupyter notebooks or web interfaces.

### `configs/`

Contains example YAML configuration files that define pipeline parameters, channel mappings, gating rules, and plugin settings.

### `data/`

Typically used for storing raw, interim, and processed data files.

### `notebooks/`

Includes Jupyter notebooks for examples, tutorials, and report generation (e.g., `report_notebook.ipynb`).

### `docs/`

Comprehensive project documentation, including guides, API references, and conceptual explanations.

### `tests/`

Unit and integration tests to ensure the correctness and reliability of the codebase.

## 3. Data Flow and Execution

The typical data flow in `cytoflow-qc` involves the following steps:

1.  **Configuration Loading**: The `cli.py` or `api.py` loads the primary configuration (`example_config.yaml`) and gating rules (`gating_rules.yaml`).
2.  **Data Ingestion**: `io.py` and `data_connectors.py` read raw flow cytometry data based on the provided samplesheet.
3.  **Preprocessing**: Data undergoes compensation (`compensate.py`) and potentially other transformations.
4.  **Gating**: `gate.py`, leveraging registered `gating` plugins, applies gating strategies to define cell populations.
5.  **Quality Control**: `qc.py`, utilizing registered `qc` plugins, assesses data quality.
6.  **Statistical Analysis**: `stats.py`, with `stats` plugins, performs statistical tests and drift analysis.
7.  **Reporting**: `report_generator.py` compiles results, plots, and summaries into a human-readable HTML report (`report.html`) using `report_notebook.ipynb` as a template.

## 4. Plugin System

The plugin system (`src/cytoflow_qc/plugins/`) is a cornerstone of `cytoflow-qc`'s extensibility. It allows users to:

*   **Register Custom Logic**: Implement their own `GatingStrategyPlugin`, `QCMethodPlugin`, or `StatisticalMethodPlugin` classes.
*   **Integrate Seamlessly**: Registered plugins are automatically discovered and can be specified in the configuration files to be incorporated into the pipeline.

This modular design promotes flexibility, allowing `cytoflow-qc` to adapt to diverse experimental needs and analytical approaches.




