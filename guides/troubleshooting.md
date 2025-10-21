# Troubleshooting Guide

This guide helps diagnose and resolve common issues when using cytoflow-qc.

## Common Issues and Solutions

### Installation Problems

#### Conda Environment Issues

**Problem**: `conda env create -f env/environment.yml` fails

**Solutions**:
1. **Update conda**: `conda update conda`
2. **Clean conda cache**: `conda clean --all`
3. **Try mamba**: `mamba env create -f env/environment.yml`
4. **Manual installation**:
   ```bash
   conda create -n cytoflow-qc python=3.11
   conda activate cytoflow-qc
   pip install -e .
   ```

#### Package Installation Fails

**Problem**: `pip install -e .` fails with dependency conflicts

**Solutions**:
1. **Use conda-forge channel**:
   ```bash
   conda config --add channels conda-forge
   conda install --file env/requirements.txt
   ```
2. **Install dependencies manually**:
   ```bash
   pip install flowkit flowutils fcsparser
   pip install -e .
   ```

### Data Ingestion Issues

#### FCS Files Not Recognized

**Problem**: `stage_ingest` fails to read FCS files

**Solutions**:
1. **Check file format**: Ensure files are valid FCS 3.0 or 3.1 format
2. **Verify file paths**: Use absolute paths in samplesheet
3. **Check file permissions**: Ensure read access to FCS files

#### CSV Files with Wrong Columns

**Problem**: CSV files don't match expected FCS column names

**Solutions**:
1. **Check column names**: FCS columns should match your configuration
2. **Use channel mapping**: Configure `channels` section in config YAML
3. **Verify samplesheet**: Ensure `file_path` points to correct CSV files

### Quality Control Issues

#### High Debris Fraction

**Problem**: Many samples fail QC due to high debris fraction

**Solutions**:
1. **Adjust debris thresholds** in config:
   ```yaml
   qc:
     debris:
       fsc_percentile: 5.0  # Increase from default 2.0
       ssc_percentile: 5.0  # Increase from default 2.0
   ```
2. **Check FCS data quality**: High debris may indicate poor sample preparation
3. **Visualize data**: Use `plot_qc_summary` to inspect gating decisions

#### Doublet Detection Too Strict

**Problem**: Too many events flagged as doublets

**Solutions**:
1. **Adjust tolerance** in config:
   ```yaml
   qc:
     doublets:
       tolerance: 0.15  # Increase from default 0.08
   ```
2. **Verify FSC-A vs FSC-H correlation**: Poor correlation may indicate instrument issues

### Gating Issues

#### Poor Gating Performance

**Problem**: Automated gating removes too many or too few events

**Solutions**:
1. **Adjust gating parameters**:
   ```yaml
   gating:
     lymphocytes:
       low_percentile: 15.0   # Increase from default 10.0
       high_percentile: 75.0  # Decrease from default 80.0
   ```
2. **Check data distribution**: Use scatter plots to verify gating regions
3. **Manual inspection**: Review gating plots in results

#### Memory Issues with Large Datasets

**Problem**: Out of memory errors with large FCS files

**Solutions**:
1. **Process in batches**: Split large experiments into smaller batches
2. **Use sampling**: Enable event sampling in configuration (if supported)
3. **Increase system memory**: Ensure adequate RAM for dataset size
4. **Monitor memory usage**: Use `htop` or similar tools

### Batch Drift Analysis Issues

#### No Significant Drift Detected

**Problem**: Drift analysis shows no significant batch effects

**Solutions**:
1. **Check batch grouping**: Ensure samples are correctly grouped by batch
2. **Verify feature extraction**: Check that marker channels are properly configured
3. **Increase sensitivity**: More samples per batch may be needed for detection
4. **Biological relevance**: Some batch effects may be too small to detect

#### False Positive Drift Detection

**Problem**: Drift analysis flags non-problematic batch differences

**Solutions**:
1. **Adjust significance threshold**: Increase p-value threshold if needed
2. **Verify batch composition**: Ensure batches are comparable
3. **Check for confounding factors**: Time, operator, or other variables may cause apparent drift

### Statistical Analysis Issues

#### Effect Size Calculation Fails

**Problem**: `stage_stats` fails with statistical errors

**Solutions**:
1. **Check sample sizes**: Ensure adequate samples per condition
2. **Verify data distribution**: Non-parametric tests may fail with extreme distributions
3. **Check for missing data**: Remove or impute missing values
4. **Use alternative grouping**: Try different grouping variables

#### Multiple Testing Corrections

**Problem**: All p-values become non-significant after correction

**Solutions**:
1. **Understand correction**: Holm-Bonferroni is conservative but appropriate
2. **Increase sample size**: More statistical power reduces need for correction
3. **Focus on effect sizes**: Large effects may be biologically meaningful even with p>0.05
4. **Use alternative corrections**: Consider Benjamini-Hochberg if appropriate

### Performance Issues

#### Slow Processing Times

**Problem**: Pipeline takes too long to complete

**Solutions**:
1. **Optimize configuration**:
   ```yaml
   qc:
     debris:
       fsc_percentile: 5.0  # Reduce computation
   ```
2. **Use faster algorithms**: Some QC methods have faster alternatives
3. **Parallel processing**: Run multiple samples simultaneously where possible
4. **Hardware optimization**: Use SSD storage and adequate RAM

#### High Memory Usage

**Problem**: System runs out of memory during processing

**Solutions**:
1. **Process samples individually**: Use `cytoflow-qc ingest` for single samples
2. **Reduce batch sizes**: Process fewer samples at once
3. **Use memory-efficient formats**: Parquet files are more memory-efficient than CSV
4. **Monitor memory usage**: Use system monitoring tools

### Configuration Issues

#### YAML Syntax Errors

**Problem**: Configuration file has YAML syntax errors

**Solutions**:
1. **Validate YAML**: Use online YAML validators
2. **Check indentation**: YAML is sensitive to indentation
3. **Use simple editor**: Avoid complex editors that may add hidden characters
4. **Start with example**: Use provided example configs as starting point

#### Configuration Validation Errors

**Problem**: Pipeline fails with configuration validation errors

**Solutions**:
1. **Check required sections**: Ensure `channels` section is present
2. **Verify channel names**: Match FCS file column names exactly
3. **Check data types**: Ensure numeric values are numbers, not strings
4. **Use validation**: Run `cytoflow-qc --help` to see expected formats

### Output and Reporting Issues

#### Empty Results Files

**Problem**: Output files are created but contain no data

**Solutions**:
1. **Check input data**: Verify FCS files contain expected data
2. **Review logs**: Look for error messages during processing
3. **Test with sample data**: Use provided sample data to verify pipeline
4. **Check file permissions**: Ensure write access to output directories

#### Report Generation Fails

**Problem**: HTML report cannot be generated

**Solutions**:
1. **Check template**: Ensure `report_template.html.j2` exists and is valid Jinja2
2. **Verify data availability**: All pipeline stages must complete successfully
3. **Check dependencies**: Ensure Jinja2 and nbconvert are installed
4. **Template syntax**: Verify template syntax is correct

### Platform-Specific Issues

#### Windows Path Issues

**Problem**: File paths not recognized on Windows

**Solutions**:
1. **Use forward slashes**: `/` works on all platforms in Python
2. **Escape backslashes**: `\\` for Windows paths in strings
3. **Use pathlib**: `Path` objects handle platform differences automatically
4. **Check path separators**: Use `os.path.sep` for platform-specific separators

#### macOS Permission Issues

**Problem**: Permission denied errors on macOS

**Solutions**:
1. **Check file permissions**: Ensure read/write access to files and directories
2. **Use user directories**: Avoid system directories for output
3. **Check antivirus**: Some antivirus software may interfere
4. **Use absolute paths**: Avoid relative path issues

### Getting Help

#### Debug Mode

Enable verbose logging:
```bash
export CYTOFLOW_QC_DEBUG=1
cytoflow-qc run --config config.yaml --samplesheet samples.csv --out results
```

#### Log Files

Check for log files in output directories:
- `results/qc/summary.csv` - QC results
- `results/gate/summary.csv` - Gating results
- `results/drift/tests.csv` - Drift analysis results

#### Community Support

1. **GitHub Issues**: Report bugs and request features
2. **Documentation**: Check updated docs and examples
3. **Discussions**: Use GitHub Discussions for questions

#### Professional Support

For commercial or urgent support needs, consider:
1. **Flow cytometry core facilities**
2. **Bioinformatics support teams**
3. **Commercial flow cytometry software vendors**

## Common Patterns

### Debugging Pipeline Failures

1. **Run stages individually**:
   ```bash
   cytoflow-qc ingest --samplesheet samples.csv --config config.yaml --out results/ingest
   cytoflow-qc qc --indir results/ingest --out results/qc --config config.yaml
   ```

2. **Check intermediate results**:
   ```python
   import pandas as pd
   manifest = pd.read_csv("results/ingest/manifest.csv")
   print(manifest.head())
   ```

3. **Visualize problematic data**:
   ```python
   from cytoflow_qc.viz import plot_qc_summary
   plot_qc_summary("results/qc/summary.csv", "qc_plot.png")
   ```

### Performance Optimization

1. **Profile memory usage**:
   ```python
   import psutil
   process = psutil.Process()
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
   ```

2. **Time individual stages**:
   ```bash
   time cytoflow-qc ingest --samplesheet samples.csv --config config.yaml --out results/ingest
   ```

### Configuration Tuning

1. **Start conservative**:
   ```yaml
   qc:
     debris:
       fsc_percentile: 5.0  # Less strict
       ssc_percentile: 5.0
   ```

2. **Adjust based on data**:
   ```python
   import pandas as pd
   df = pd.read_csv("sample.csv")
   print(df["FSC-A"].describe())
   ```

## Advanced Troubleshooting

### Custom Error Handling

For production environments, implement custom error handling:

```python
from cytoflow_qc.cli import stage_ingest, stage_compensate, stage_qc

def robust_pipeline(samplesheet, config, output_dir):
    try:
        stage_ingest(samplesheet, output_dir / "ingest", config)
        stage_compensate(output_dir / "ingest", output_dir / "compensate", None)
        stage_qc(output_dir / "compensate", output_dir / "qc", config["qc"])
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        # Implement recovery logic
```

### Monitoring and Alerting

Set up monitoring for production pipelines:

```python
import time
import logging

def monitored_pipeline(samplesheet, config, output_dir):
    start_time = time.time()

    try:
        # Pipeline execution with timing
        stage_ingest(samplesheet, output_dir / "ingest", config)

        # Send success notification
        logger.info(f"Pipeline completed in {time.time() - start_time:.1f}s")

    except Exception as e:
        logger.error(f"Pipeline failed after {time.time() - start_time:.1f}s: {e}")
        # Send failure alert
```

This troubleshooting guide should help resolve most common issues with cytoflow-qc. For additional help, please consult the full documentation or open an issue on GitHub.










