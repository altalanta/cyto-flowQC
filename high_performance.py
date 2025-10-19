"""High-performance computing utilities for processing large flow cytometry datasets."""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Iterator

import numpy as np
import pandas as pd

# Try to import optional performance libraries
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


class MemoryManager:
    """Memory management utilities for large dataset processing."""

    def __init__(self, max_memory_gb: float = 8.0):
        """Initialize memory manager.

        Args:
            max_memory_gb: Maximum memory to use in GB
        """
        self.max_memory_gb = max_memory_gb
        self._initial_memory = self._get_memory_usage()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024**3)
        except ImportError:
            # Fallback for systems without psutil
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)

    def should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        current_memory = self._get_memory_usage()
        return current_memory - self._initial_memory > self.max_memory_gb

    def force_garbage_collection(self) -> None:
        """Force garbage collection and clear caches."""
        gc.collect()
        if hasattr(pd, 'core'):
            # Clear pandas cache if available
            try:
                pd.core.common.cleanup()
            except AttributeError:
                pass

    @contextmanager
    def memory_checkpoint(self, checkpoint_name: str = "unknown"):
        """Context manager for memory checkpoints."""
        initial_memory = self._get_memory_usage()
        try:
            yield
        finally:
            final_memory = self._get_memory_usage()
            memory_delta = final_memory - initial_memory
            if memory_delta > 0.1:  # Only log significant changes
                print(f"Memory checkpoint '{checkpoint_name}': +{memory_delta:.2f} GB")


class StreamingProcessor:
    """Process large FCS files in chunks to handle datasets larger than memory."""

    def __init__(self, chunk_size: int = 100000):
        """Initialize streaming processor.

        Args:
            chunk_size: Number of events to process at once
        """
        self.chunk_size = chunk_size

    def process_fcs_streaming(
        self,
        file_path: str | Path,
        processing_function: Callable[[pd.DataFrame], pd.DataFrame],
        output_dir: str | Path | None = None,
        **kwargs
    ) -> pd.DataFrame | None:
        """Process an FCS file in streaming chunks.

        Args:
            file_path: Path to FCS file
            processing_function: Function to apply to each chunk
            output_dir: Directory to save intermediate results
            **kwargs: Additional arguments for processing function

        Returns:
            Combined processed DataFrame or None if output_dir specified
        """
        file_path = Path(file_path)
        output_dir = Path(output_dir) if output_dir else None

        # Try to read file in chunks
        try:
            # First, try to read the entire file to get structure
            sample_df = pd.read_csv(file_path, nrows=1)
            columns = list(sample_df.columns)

            # Process in chunks
            chunk_results = []
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                # Ensure chunk has all expected columns
                chunk = chunk.reindex(columns=columns)

                # Apply processing function
                processed_chunk = processing_function(chunk, **kwargs)

                if output_dir:
                    # Save chunk to disk
                    chunk_file = output_dir / f"chunk_{len(chunk_results)}.parquet"
                    processed_chunk.to_parquet(chunk_file, index=False)
                    chunk_results.append(chunk_file)
                else:
                    # Keep in memory
                    chunk_results.append(processed_chunk)

            if output_dir:
                # Return list of output files
                return chunk_results
            else:
                # Combine results
                return pd.concat(chunk_results, ignore_index=True)

        except Exception as e:
            print(f"Error in streaming processing: {e}")
            # Fallback to normal processing
            df = pd.read_csv(file_path)
            return processing_function(df, **kwargs)

    def process_directory_streaming(
        self,
        input_dir: str | Path,
        processing_function: Callable[[pd.DataFrame], pd.DataFrame],
        output_dir: str | Path | None = None,
        file_pattern: str = "*.csv",
        **kwargs
    ) -> list[Path] | None:
        """Process all files in a directory using streaming.

        Args:
            input_dir: Directory containing input files
            processing_function: Function to apply to each file
            output_dir: Directory to save results
            file_pattern: Pattern to match input files
            **kwargs: Additional arguments for processing function

        Returns:
            List of output file paths if output_dir specified, None otherwise
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else None

        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        results = []

        for file_path in input_path.glob(file_pattern):
            print(f"Processing {file_path.name}...")

            try:
                chunk_results = self.process_fcs_streaming(
                    file_path, processing_function, output_path, **kwargs
                )

                if output_path and chunk_results:
                    results.extend(chunk_results)
                elif chunk_results is not None:
                    results.append(chunk_results)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        return results if output_path else None


class ParallelProcessor:
    """Parallel processing utilities for CPU-intensive operations."""

    def __init__(self, n_workers: int | None = None):
        """Initialize parallel processor.

        Args:
            n_workers: Number of worker processes (defaults to CPU count - 1)
        """
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self._pool = None

    def __enter__(self):
        """Enter context manager."""
        self._pool = mp.Pool(self.n_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self._pool:
            self._pool.close()
            self._pool.join()

    def map_parallel(
        self,
        func: Callable,
        items: Iterable[Any],
        chunksize: int | None = None
    ) -> list[Any]:
        """Apply function to items in parallel.

        Args:
            func: Function to apply
            items: Items to process
            chunksize: Size of chunks for parallel processing

        Returns:
            List of results
        """
        if not self._pool:
            raise RuntimeError("ParallelProcessor must be used as context manager")

        return self._pool.map(func, items, chunksize)

    def starmap_parallel(
        self,
        func: Callable,
        items: Iterable[tuple],
        chunksize: int | None = None
    ) -> list[Any]:
        """Apply function with arguments to items in parallel.

        Args:
            func: Function to apply
            items: Items with arguments to process
            chunksize: Size of chunks for parallel processing

        Returns:
            List of results
        """
        if not self._pool:
            raise RuntimeError("ParallelProcessor must be used as context manager")

        return self._pool.starmap(func, items, chunksize)

    @staticmethod
    def process_sample_parallel(sample_data: tuple) -> dict[str, Any]:
        """Process a single sample in parallel (helper function).

        Args:
            sample_data: Tuple of (sample_id, sample_path, config)

        Returns:
            Dictionary with processing results
        """
        sample_id, sample_path, config = sample_data

        try:
            # Load sample data
            df = pd.read_parquet(sample_path)

            # Apply QC
            from .qc import add_qc_flags
            qc_df = add_qc_flags(df, config.get("qc", {}))

            # Apply gating
            from .gate import auto_gate
            gated_df, gate_params = auto_gate(qc_df, config=config.get("gating", {}))

            # Extract features
            from .drift import extract_sample_features
            features = extract_sample_features(
                {sample_id: gated_df},
                pd.DataFrame([{"sample_id": sample_id}]),
                config.get("channels", {}).get("markers")
            )

            return {
                "sample_id": sample_id,
                "success": True,
                "features": features,
                "gated_events": len(gated_df),
                "gate_params": gate_params
            }

        except Exception as e:
            return {
                "sample_id": sample_id,
                "success": False,
                "error": str(e)
            }


class MemoryMappedProcessor:
    """Memory-mapped processing for very large datasets."""

    def __init__(self):
        """Initialize memory-mapped processor."""
        self._temp_files = []

    def __del__(self):
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                temp_file.unlink()
            except (OSError, AttributeError):
                pass

    def create_memory_map(self, data: pd.DataFrame, temp_dir: str | Path | None = None) -> str:
        """Create a memory-mapped file from DataFrame.

        Args:
            data: DataFrame to memory map
            temp_dir: Directory for temporary file

        Returns:
            Path to memory-mapped file
        """
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()

        temp_path = Path(temp_dir)

        # Create temporary file
        temp_file = temp_path / f"mmap_{id(data)}.dat"
        self._temp_files.append(temp_file)

        # Save data to memory-mappable format
        np_array = data.values.astype(np.float64)
        np_array.tofile(str(temp_file))

        return str(temp_file)

    def load_memory_mapped(
        self,
        file_path: str | Path,
        shape: tuple[int, int],
        dtype: np.dtype = np.float64
    ) -> np.memmap:
        """Load data as memory-mapped array.

        Args:
            file_path: Path to memory-mapped file
            shape: Shape of the array (rows, columns)
            dtype: Data type

        Returns:
            Memory-mapped numpy array
        """
        return np.memmap(str(file_path), dtype=dtype, mode='r', shape=shape)


class CompressedProcessor:
    """Processing utilities for compressed FCS data."""

    def __init__(self):
        """Initialize compressed processor."""
        pass

    def estimate_compression_ratio(self, data: pd.DataFrame) -> float:
        """Estimate compression ratio for the data.

        Args:
            data: DataFrame to analyze

        Returns:
            Estimated compression ratio
        """
        # Simple heuristic based on data types and patterns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        compression_ratio = 1.0

        # Float64 -> Float32 can compress by ~2x
        for col in numeric_cols:
            if data[col].dtype == np.float64:
                compression_ratio *= 2.0

        # Integer columns compress well
        int_cols = data.select_dtypes(include=[np.integer]).columns
        if len(int_cols) > 0:
            compression_ratio *= 1.5

        return min(compression_ratio, 5.0)  # Cap at 5x compression

    def optimize_dataframe_for_compression(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for better compression.

        Args:
            data: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        optimized = data.copy()

        # Downcast numeric types where possible
        for col in optimized.select_dtypes(include=[np.number]).columns:
            if optimized[col].dtype == np.float64:
                # Check if we can downcast to float32
                if (optimized[col].max() < 3.4e38 and
                    optimized[col].min() > -3.4e38 and
                    optimized[col].notna().all()):
                    try:
                        optimized[col] = optimized[col].astype(np.float32)
                    except (ValueError, OverflowError):
                        pass  # Keep as float64

            elif optimized[col].dtype in [np.int64, np.uint64]:
                # Try to downcast integers
                try:
                    optimized[col] = pd.to_numeric(optimized[col], downcast='integer')
                except (ValueError, TypeError):
                    pass  # Keep original dtype

        return optimized

    def process_with_compression_optimization(
        self,
        data: pd.DataFrame,
        processing_function: Callable[[pd.DataFrame], pd.DataFrame],
        use_compression: bool = True
    ) -> pd.DataFrame:
        """Process data with compression optimization.

        Args:
            data: Input DataFrame
            processing_function: Function to apply
            use_compression: Whether to optimize for compression

        Returns:
            Processed DataFrame
        """
        if use_compression:
            optimized_data = self.optimize_dataframe_for_compression(data)
            result = processing_function(optimized_data)

            # Optimize result as well
            return self.optimize_dataframe_for_compression(result)
        else:
            return processing_function(data)


class OutOfCoreProcessor:
    """Out-of-core processing for datasets larger than available RAM."""

    def __init__(self, temp_dir: str | Path | None = None):
        """Initialize out-of-core processor.

        Args:
            temp_dir: Directory for temporary storage
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "cytoflow_ooc"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def process_large_dataset(
        self,
        file_paths: list[str | Path],
        processing_function: Callable[[pd.DataFrame], pd.DataFrame],
        output_file: str | Path,
        chunk_size: int = 100000
    ) -> pd.DataFrame:
        """Process multiple large files using out-of-core techniques.

        Args:
            file_paths: List of file paths to process
            processing_function: Function to apply to each chunk
            output_file: Path to save final result
            chunk_size: Size of chunks to process

        Returns:
            Final processed DataFrame
        """
        temp_files = []

        for file_path in file_paths:
            print(f"Processing {file_path}...")

            # Process each file in chunks
            file_path = Path(file_path)
            chunks_processed = 0

            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Apply processing function
                processed_chunk = processing_function(chunk)

                # Save to temporary file
                temp_file = self.temp_dir / f"{file_path.stem}_chunk_{chunks_processed}.parquet"
                processed_chunk.to_parquet(temp_file, index=False)
                temp_files.append(temp_file)
                chunks_processed += 1

        # Combine all results
        print("Combining results...")
        combined_df = pd.concat(
            [pd.read_parquet(f) for f in temp_files],
            ignore_index=True
        )

        # Save final result
        output_path = Path(output_file)
        combined_df.to_parquet(output_path, index=False)

        # Clean up temporary files
        for temp_file in temp_files:
            temp_file.unlink()

        return combined_df

    def create_sampled_dataset(
        self,
        input_file: str | Path,
        sample_fraction: float = 0.1,
        output_file: str | Path | None = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Create a sampled subset of a large dataset.

        Args:
            input_file: Path to input file
            sample_fraction: Fraction of data to sample (0-1)
            output_file: Path to save sampled data
            random_state: Random seed for reproducibility

        Returns:
            Sampled DataFrame
        """
        input_path = Path(input_file)

        # Get total number of rows
        with open(input_path, 'r') as f:
            total_rows = sum(1 for _ in f) - 1  # Subtract header

        # Calculate sample size
        sample_size = max(1, int(total_rows * sample_fraction))

        # Sample rows
        sampled_df = pd.read_csv(
            input_path,
            skiprows=lambda x: x > 0 and np.random.random() > sample_fraction,
            random_state=random_state
        )

        if output_file:
            output_path = Path(output_file)
            sampled_df.to_csv(output_path, index=False)

        return sampled_df


# Performance utilities
def get_optimal_chunk_size(data_size_gb: float, available_memory_gb: float = 4.0) -> int:
    """Calculate optimal chunk size for processing.

    Args:
        data_size_gb: Estimated size of dataset in GB
        available_memory_gb: Available memory for processing

    Returns:
        Optimal chunk size in rows
    """
    # Assume ~8 bytes per float64 value, ~10 columns average
    bytes_per_row = 8 * 10  # 80 bytes per row
    rows_per_gb = (1024**3) / bytes_per_row  # ~13M rows per GB

    # Use 50% of available memory for chunks
    usable_memory_gb = available_memory_gb * 0.5

    # Calculate chunk size to fit in usable memory
    chunk_rows = int(usable_memory_gb * rows_per_gb)

    # Cap at reasonable sizes
    return min(max(chunk_rows, 10000), 1000000)


def optimize_pandas_memory_usage() -> None:
    """Optimize pandas memory usage settings."""
    # Set pandas options for better memory efficiency
    pd.set_option('mode.chained_assignment', None)  # Disable warning
    pd.set_option('mode.use_inf_as_na', True)  # Handle inf values

    # Enable string dtype for better memory usage (pandas 1.0+)
    try:
        pd.set_option('mode.string_storage', 'python')
    except KeyError:
        pass  # Option may not be available in older pandas versions


def benchmark_function(func: Callable, *args, iterations: int = 3, **kwargs) -> dict[str, float]:
    """Benchmark a function's performance.

    Args:
        func: Function to benchmark
        *args: Positional arguments for function
        iterations: Number of iterations to run
        **kwargs: Keyword arguments for function

    Returns:
        Dictionary with timing statistics
    """
    import time

    times = []
    for _ in range(iterations):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "iterations": iterations
    }


# Context managers for performance optimization
@contextmanager
def memory_optimized_processing(max_memory_gb: float = 4.0):
    """Context manager for memory-optimized processing."""
    memory_manager = MemoryManager(max_memory_gb)
    optimize_pandas_memory_usage()

    with memory_manager:
        yield memory_manager


@contextmanager
def parallel_processing(n_workers: int | None = None):
    """Context manager for parallel processing."""
    with ParallelProcessor(n_workers) as processor:
        yield processor


def enable_dask_if_available(data: pd.DataFrame, npartitions: int | None = None) -> Any:
    """Convert pandas DataFrame to Dask DataFrame if available.

    Args:
        data: Input pandas DataFrame
        npartitions: Number of partitions for Dask

    Returns:
        Dask DataFrame if available, otherwise pandas DataFrame
    """
    if DASK_AVAILABLE:
        if npartitions is None:
            npartitions = max(1, mp.cpu_count() // 2)
        return dd.from_pandas(data, npartitions=npartitions)
    else:
        return data


def enable_polars_if_available(data: pd.DataFrame) -> Any:
    """Convert pandas DataFrame to Polars DataFrame if available.

    Args:
        data: Input pandas DataFrame

    Returns:
        Polars DataFrame if available, otherwise pandas DataFrame
    """
    if POLARS_AVAILABLE:
        return pl.from_pandas(data)
    else:
        return data


# Utility functions for high-performance processing
def chunked_dataframe_processing(
    data: pd.DataFrame,
    processing_function: Callable[[pd.DataFrame], pd.DataFrame],
    chunk_size: int | None = None
) -> pd.DataFrame:
    """Process DataFrame in chunks for memory efficiency.

    Args:
        data: Input DataFrame
        processing_function: Function to apply to each chunk
        chunk_size: Size of chunks (defaults to optimal size)

    Returns:
        Processed DataFrame
    """
    if chunk_size is None:
        # Estimate optimal chunk size based on data size
        data_size_gb = data.memory_usage(deep=True).sum() / (1024**3)
        chunk_size = get_optimal_chunk_size(data_size_gb)

    # Process in chunks
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size].copy()
        processed_chunk = processing_function(chunk)
        results.append(processed_chunk)

    return pd.concat(results, ignore_index=True)


def create_performance_report(
    timings: dict[str, float],
    memory_usage: dict[str, float],
    output_file: str | Path | None = None
) -> str:
    """Create a performance report from timing and memory data.

    Args:
        timings: Dictionary of operation names to execution times
        memory_usage: Dictionary of checkpoint names to memory usage
        output_file: Path to save report (optional)

    Returns:
        Formatted performance report
    """
    report = []
    report.append("🔬 CytoFlow-QC Performance Report")
    report.append("=" * 50)
    report.append("")

    # Timing information
    report.append("⏱️  Execution Times:")
    for operation, time_taken in timings.items():
        report.append(f"  {operation}: {time_taken:.3f}s")

    report.append("")

    # Memory information
    report.append("🧠 Memory Usage:")
    for checkpoint, memory in memory_usage.items():
        report.append(f"  {checkpoint}: {memory:.2f} GB")

    report.append("")

    # Summary statistics
    if timings:
        total_time = sum(timings.values())
        report.append("📊 Summary:")
        report.append(f"  Total execution time: {total_time:.3f}s")
        report.append(f"  Average operation time: {total_time / len(timings):.3f}s")

    if memory_usage:
        max_memory = max(memory_usage.values())
        report.append(f"  Peak memory usage: {max_memory:.2f} GB")

    full_report = "\n".join(report)

    if output_file:
        Path(output_file).write_text(full_report)

    return full_report






