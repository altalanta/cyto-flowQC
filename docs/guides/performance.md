# Performance Optimization Guide

This guide provides strategies for optimizing cytoflow-qc performance when working with large datasets or in production environments.

## Understanding Performance Bottlenecks

### Common Performance Issues

1. **Memory Usage**: Large FCS files can exhaust system memory
2. **I/O Operations**: Frequent file reads/writes slow down processing
3. **CPU-intensive Operations**: Statistical computations and gating algorithms
4. **Storage Space**: Multiple output directories accumulate over time

## Optimization Strategies

### Memory Optimization

#### 1. Process Samples Individually

For very large datasets, process samples one at a time:

```bash
# Instead of processing all samples together
cytoflow-qc run --samplesheet samples.csv --config config.yaml --out results

# Process samples individually
for sample in $(cut -d',' -f1 samples.csv | tail -n +2); do
    cytoflow-qc ingest --samplesheet <(grep "$sample" samples.csv) --config config.yaml --out results/ingest
    cytoflow-qc compensate --indir results/ingest --out results/compensate
    cytoflow-qc qc --indir results/compensate --out results/qc --config config.yaml
    # Continue with remaining stages...
done
```

#### 2. Use Memory-Efficient Data Types

Configure pandas to use more memory-efficient data types:

```python
import pandas as pd

# Read FCS data with optimized dtypes
def read_fcs_optimized(file_path):
    df = pd.read_csv(file_path)

    # Downcast numeric columns where possible
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    return df
```

#### 3. Enable Garbage Collection

Force garbage collection between processing stages:

```python
import gc

def process_with_gc_optimization():
    # Process stage
    result = stage_ingest(samplesheet, output_dir, config)

    # Force garbage collection
    gc.collect()

    return result
```

### I/O Optimization

#### 1. Use Fast Storage

- **SSDs over HDDs**: Significantly faster for large file operations
- **Local storage**: Avoid network-attached storage when possible
- **RAID arrays**: Use RAID 0 or 10 for better I/O performance

#### 2. Optimize Parquet Compression

Configure Parquet writing for better compression/performance balance:

```python
import pyarrow as pa
import pyarrow.parquet as pq

def save_optimized_parquet(df, path):
    table = pa.Table.from_pandas(df)

    # Use Snappy compression (good balance of speed/compression)
    pq.write_table(
        table,
        path,
        compression='snappy',
        use_dictionary=True,
        row_group_size=100000
    )
```

#### 3. Batch File Operations

Process multiple files together when possible:

```python
def batch_process_files(file_list, output_dir):
    """Process multiple FCS files in a single operation."""

    # Read all files at once if memory allows
    dfs = []
    for file_path in file_list:
        df = read_fcs_optimized(file_path)
        dfs.append(df)

    # Combine and process
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save combined results
    combined_df.to_parquet(output_dir / "combined.parquet")
```

### CPU Optimization

#### 1. Parallel Processing

Use multiprocessing for independent operations:

```python
from multiprocessing import Pool
import numpy as np

def process_sample_parallel(sample_file):
    """Process a single sample (for parallel execution)."""
    df = read_fcs_optimized(sample_file)
    qc_df = add_qc_flags(df, qc_config)
    gated_df, params = auto_gate(qc_df, strategy="default", config=gate_config)

    return gated_df, params

def parallel_pipeline(samplesheet):
    """Run pipeline with parallel sample processing."""
    sample_files = samplesheet['file_path'].tolist()

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_sample_parallel, sample_files)

    return results
```

#### 2. Vectorized Operations

Replace loops with vectorized pandas/numpy operations:

```python
# Instead of this (slow):
for i, row in df.iterrows():
    df.loc[i, 'normalized'] = row['value'] / row['max_value']

# Use this (fast):
df['normalized'] = df['value'] / df['max_value']
```

#### 3. Algorithm Selection

Choose the most efficient algorithms for your data:

```python
# For large datasets, consider faster alternatives
from sklearn.neighbors import NearestNeighbors

def optimized_drift_analysis(features_df, batch_col):
    """Use faster algorithms for drift detection."""

    # Use BallTree for faster neighbor searches
    if len(features_df) > 10000:
        # Use approximate methods for large datasets
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=5)
        clusters = clustering.fit_predict(features_df.drop(columns=[batch_col]))
    else:
        # Use exact methods for smaller datasets
        pass
```

### Configuration Optimization

#### 1. Adjust QC Parameters

Reduce computational load by adjusting QC sensitivity:

```yaml
# Less strict QC (faster processing)
qc:
  debris:
    fsc_percentile: 5.0  # Increased from 2.0
    ssc_percentile: 5.0
  doublets:
    tolerance: 0.15      # Increased from 0.08
  saturation:
    threshold: 0.99      # Slightly less strict
```

#### 2. Optimize Gating Parameters

Adjust gating to be less computationally intensive:

```yaml
gating:
  lymphocytes:
    method: "percentile"  # Faster than density-based
    low_percentile: 15.0
    high_percentile: 75.0
  singlets:
    method: "linear"      # Faster than complex methods
    slope_tolerance: 0.1
```

#### 3. Statistical Computation Optimization

Reduce statistical computation overhead:

```python
def optimized_effect_sizes(df, group_col, value_cols):
    """Optimized effect size computation."""

    # Use vectorized operations
    results = []
    for col in value_cols:
        if col in df.columns:
            # Compute effect sizes in batches
            grouped = df.groupby(group_col)[col]
            control_values = grouped.get_group(grouped.groups.keys()[0])

            for group_name in grouped.groups.keys():
                if group_name != grouped.groups.keys()[0]:
                    treatment_values = grouped.get_group(group_name)

                    # Compute effect size efficiently
                    effect_size = compute_hedges_g(control_values, treatment_values)
                    results.append({
                        'marker': col,
                        'effect_size': effect_size,
                        'group': group_name
                    })

    return pd.DataFrame(results)
```

## Monitoring and Profiling

### Performance Profiling

#### 1. Profile Individual Stages

```python
import cProfile
import pstats

def profile_stage():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the stage you want to profile
    stage_ingest(samplesheet, output_dir, config)

    profiler.disable()

    # Save profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

profile_stage()
```

#### 2. Memory Profiling

```python
import tracemalloc
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function to profile
    large_df = pd.read_csv("large_file.csv")
    return process_dataframe(large_df)

# Run with profiling
memory_intensive_function()
```

#### 3. Time Profiling with context managers

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name="operation"):
    start = time.time()
    yield
    end = time.time()
    print(f"{name} took {end - start:.3f} seconds")

def timed_pipeline():
    with timer("ingest"):
        stage_ingest(samplesheet, ingest_dir, config)

    with timer("qc"):
        stage_qc(ingest_dir, qc_dir, config)

    with timer("gate"):
        stage_gate(qc_dir, gate_dir, config)
```

### Production Monitoring

#### 1. Set Up Performance Alerts

```python
import time
import smtplib
from email.mime.text import MimeText

def monitor_pipeline_performance(samplesheet, config, output_dir):
    """Monitor pipeline performance and send alerts."""

    start_time = time.time()
    start_memory = get_memory_usage()

    try:
        # Run pipeline
        run_full_pipeline(samplesheet, config, output_dir)

        # Check performance metrics
        end_time = time.time()
        end_memory = get_memory_usage()

        duration = end_time - start_time
        memory_delta = end_memory - start_memory

        # Alert if performance degrades
        if duration > EXPECTED_DURATION * 1.5:  # 50% slower than expected
            send_performance_alert("slow_execution", duration)

        if memory_delta > EXPECTED_MEMORY * 1.5:  # 50% more memory than expected
            send_performance_alert("high_memory", memory_delta)

    except Exception as e:
        send_error_alert(e)

def get_memory_usage():
    """Get current memory usage."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def send_performance_alert(alert_type, value):
    """Send performance alert via email or logging."""
    # Implementation depends on your alerting system
    print(f"PERFORMANCE ALERT: {alert_type} = {value}")
```

#### 2. Resource Usage Logging

```python
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_performance.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_resource_usage(stage_name):
    """Log resource usage for a pipeline stage."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()

    logger.info(
        f"{stage_name}: Memory={memory_mb:.1f}MB, CPU={cpu_percent:.1f}%"
    )
```

## Hardware Optimization

### System Requirements

**Minimum Requirements:**
- **RAM**: 8GB (16GB recommended for large datasets)
- **CPU**: 4 cores (8+ cores recommended for parallel processing)
- **Storage**: SSD with at least 50GB free space
- **OS**: Linux/macOS/Windows with Python 3.10+

**Recommended for Large Datasets:**
- **RAM**: 32GB+ for datasets > 1M events
- **CPU**: 16+ cores for parallel processing
- **Storage**: NVMe SSD for fastest I/O
- **Network**: Avoid network storage for large files

### Cloud Deployment

For very large datasets or production workloads:

```yaml
# Docker configuration for cloud deployment
services:
  cytoflow-qc:
    image: cytoflow-qc:latest
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '4.0'
        reservations:
          memory: 8G
          cpus: '2.0'
    volumes:
      - ./data:/data:ro
      - ./results:/results
```

## Performance Benchmarks

### Expected Performance

**Small Dataset** (1-10 samples, 10K events each):
- **Ingest**: 5-15 seconds
- **QC**: 2-8 seconds
- **Gating**: 3-10 seconds
- **Drift**: 5-15 seconds
- **Stats**: 1-5 seconds
- **Total**: 16-53 seconds

**Medium Dataset** (50 samples, 50K events each):
- **Ingest**: 30-90 seconds
- **QC**: 15-45 seconds
- **Gating**: 20-60 seconds
- **Drift**: 30-90 seconds
- **Stats**: 5-15 seconds
- **Total**: 100-300 seconds

**Large Dataset** (200+ samples, 100K+ events each):
- Use batch processing or cloud deployment
- Expected: 10-30 minutes depending on hardware

### Performance Tuning Checklist

- [ ] Profile memory usage before optimization
- [ ] Identify slowest pipeline stages
- [ ] Check I/O bottlenecks (disk speed, file sizes)
- [ ] Verify CPU utilization during processing
- [ ] Test with representative dataset sizes
- [ ] Monitor garbage collection impact
- [ ] Validate optimizations don't break functionality
- [ ] Document performance requirements for future changes

## Troubleshooting Performance Issues

### Slow Processing

1. **Check system resources**: Monitor CPU, memory, and disk usage
2. **Profile individual stages**: Use `cProfile` or `line_profiler`
3. **Optimize data structures**: Use appropriate pandas dtypes
4. **Reduce I/O**: Minimize file reads/writes where possible
5. **Enable parallel processing**: Use multiprocessing for independent operations

### High Memory Usage

1. **Monitor memory patterns**: Use `memory_profiler` or `psutil`
2. **Process in chunks**: Split large files into smaller pieces
3. **Use streaming**: Process data without loading everything into memory
4. **Optimize data types**: Use smaller numeric types where possible
5. **Force garbage collection**: Call `gc.collect()` between stages

### I/O Bottlenecks

1. **Use faster storage**: SSD vs HDD makes significant difference
2. **Optimize file formats**: Parquet vs CSV for large datasets
3. **Batch operations**: Read/write multiple files together
4. **Compression**: Use appropriate compression for your use case
5. **Network storage**: Avoid for large files if possible

This performance optimization guide should help you achieve optimal throughput with cytoflow-qc. For dataset-specific optimizations, consider profiling your actual data and adjusting parameters accordingly.
