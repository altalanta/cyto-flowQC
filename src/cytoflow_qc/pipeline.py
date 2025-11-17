"""Class-based pipeline stage architecture for CytoFlow-QC."""
from __future__ import annotations

import hashlib
import inspect
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from cytoflow_qc.config import AppConfig, QCConfig
from cytoflow_qc.io import (
    stream_events, 
    get_fcs_metadata, 
    standardize_channels, 
    load_samplesheet
)
from cytoflow_qc.model import Sample, StageResult
from cytoflow_qc.utils import ensure_dir, _read_json, _write_json
from cytoflow_qc.compensate import get_spillover, apply_compensation
from cytoflow_qc.qc import add_qc_flags, qc_summary
from cytoflow_qc.gate import auto_gate
from cytoflow_qc.drift import compute_batch_drift, extract_sample_features
from cytoflow_qc.stats import effect_sizes
from cytoflow_qc.viz import (
    plot_gating_scatter, 
    plot_qc_summary, 
    plot_batch_drift_pca, 
    plot_batch_drift_umap, 
    plot_effect_sizes
)
from .interactive_report import create_and_save_report

logger = logging.getLogger(__name__)
CACHE_DIR = Path(".cytoflow_cache")
CHUNK_SIZE = 100_000

class PipelineStage(ABC):
    """Abstract base class for a pipeline stage."""

    @property
    @abstractmethod
    def stage_name(self) -> str:
        raise NotImplementedError

    def run(self, input_result: StageResult | None = None) -> StageResult | None:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_key = self._get_cache_key(input_result)
        if cache_key:
            cache_file = CACHE_DIR / f"{self.stage_name}_{cache_key}.json"
            if cache_file.exists():
                logger.info(f"Cache hit for stage '{self.stage_name}'. Loading from cache.")
                return StageResult.parse_file(cache_file)
        logger.info(f"Cache miss for stage '{self.stage_name}'. Executing stage.")
        output_result = self._run_logic(input_result)
        if cache_key and output_result:
            with open(cache_file, 'w') as f:
                f.write(output_result.json(indent=2))
        return output_result

    @abstractmethod
    def _run_logic(self, input_result: StageResult | None = None) -> StageResult | None:
        raise NotImplementedError
        
    def _get_cache_key(self, input_result: StageResult | None) -> str | None:
        return None

    def _hash_inputs(self, *args) -> str:
        hasher = hashlib.md5()
        for arg in args:
            if arg is None: continue
            if isinstance(arg, (str, Path)): hasher.update(str(arg).encode())
            elif isinstance(arg, BaseModel): hasher.update(arg.json(sort_keys=True).encode())
            else: hasher.update(str(arg).encode())
        try: source = inspect.getsource(self.process_sample)
        except AttributeError: source = inspect.getsource(self._run_logic)
        hasher.update(source.encode())
        return hasher.hexdigest()

class IngestionStage(PipelineStage):
    stage_name = "ingest"
    def __init__(self, outdir: Path, samplesheet_path: Path, config: AppConfig):
        self.outdir = outdir
        self.samplesheet_path = samplesheet_path
        self.config = config
        self.events_dir = ensure_dir(self.outdir / "events")
        self.meta_dir = ensure_dir(self.outdir / "metadata")

    def _run_logic(self, input_result: StageResult | None = None) -> StageResult:
        sheet = load_samplesheet(str(self.samplesheet_path))
        channel_map = self.config.channels.dict()
        samples = []
        for row in sheet.to_dict(orient="records"):
            if row.get("missing_file"): continue
            metadata = get_fcs_metadata(row["file_path"])
            sample_id = row["sample_id"]
            events_file = self.events_dir / f"{sample_id}.parquet"
            metadata_file = self.meta_dir / f"{sample_id}.json"
            
            writer = None
            first_chunk = True
            for chunk in stream_events(Path(row["file_path"]), chunk_size=CHUNK_SIZE):
                if channel_map:
                    chunk = standardize_channels(chunk, metadata, channel_map)
                if first_chunk:
                    writer = pq.ParquetWriter(events_file, chunk.columns.to_arrow())
                    first_chunk = False
                writer.write_table(pa.Table.from_pandas(chunk))
            if writer: writer.close()
            
            _write_json(metadata_file, metadata)
            sample = Sample(
                sample_id=sample_id, file_path=Path(row["file_path"]), events_file=events_file,
                metadata_file=metadata_file,
                extra_metadata={k: v for k, v in row.items() if k not in ["sample_id", "file_path"]},
            )
            samples.append(sample)
        return StageResult(stage_name=self.stage_name, samples=samples)

    def _get_cache_key(self, input_result: StageResult | None) -> str | None:
        with open(self.samplesheet_path, 'rb') as f:
            samplesheet_hash = hashlib.md5(f.read()).hexdigest()
        return self._hash_inputs(samplesheet_hash, self.config.channels)

class ParallelPipelineStage(PipelineStage):
    def __init__(self, outdir: Path, workers: int):
        self.outdir = outdir
        self.workers = workers
        self.events_dir = ensure_dir(self.outdir / "events")
        self.meta_dir = ensure_dir(self.outdir / "metadata")

    def _run_logic(self, input_result: StageResult) -> StageResult:
        results = []
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.process_sample, sample): sample for sample in input_result.samples}
            for future in as_completed(futures):
                results.append(future.result())
        updated_samples, optional_data = zip(*results) if results else ([], [])
        self.finalize_stage(list(optional_data))
        return StageResult(stage_name=self.stage_name, samples=list(updated_samples))

    @abstractmethod
    def process_sample(self, sample: Sample) -> Tuple[Sample, Any]:
        raise NotImplementedError

    def finalize_stage(self, optional_data: List[Any]) -> None:
        pass

class CompensationStage(ParallelPipelineStage):
    stage_name = "compensate"
    def __init__(self, outdir: Path, workers: int, spill: Path | None):
        super().__init__(outdir, workers)
        self.spill = spill

    def process_sample(self, sample: Sample) -> Tuple[Sample, None]:
        metadata = _read_json(sample.metadata_file)
        matrix, channels = get_spillover(metadata, str(self.spill) if self.spill else None)
        events_file = self.events_dir / f"{sample.sample_id}.parquet"
        
        writer = None
        first_chunk = True
        for chunk in stream_events(sample.events_file, chunk_size=CHUNK_SIZE):
            if matrix is not None and channels is not None:
                chunk = apply_compensation(chunk, matrix, channels)
            if first_chunk:
                writer = pq.ParquetWriter(events_file, chunk.columns.to_arrow())
                first_chunk = False
            writer.write_table(pa.Table.from_pandas(chunk))
        if writer: writer.close()
        
        metadata["compensated"] = matrix is not None
        metadata_file = self.meta_dir / f"{sample.sample_id}.json"
        _write_json(metadata_file, metadata)
        
        sample.events_file = events_file
        sample.metadata_file = metadata_file
        return sample, None

class QCStage(ParallelPipelineStage):
    stage_name = "qc"
    def __init__(self, outdir: Path, workers: int, qc_config: QCConfig):
        super().__init__(outdir, workers)
        self.qc_config = qc_config

    def process_sample(self, sample: Sample) -> Tuple[Sample, pd.DataFrame]:
        # Pre-pass for percentile calculation
        fsc_ssc_cols = [self.qc_config.debris.fsc_a_pct, self.qc_config.debris.ssc_a_pct]
        fsc_ssc_data = pd.concat(list(stream_events(sample.events_file, columns=fsc_ssc_cols)))
        fsc_low = np.percentile(fsc_ssc_data[self.qc_config.debris.fsc_a_pct], 2)
        ssc_low = np.percentile(fsc_ssc_data[self.qc_config.debris.ssc_a_pct], 2)
        
        events_file = self.events_dir / f"{sample.sample_id}.parquet"
        writer = None
        first_chunk = True
        all_qc_dfs = []
        for chunk in stream_events(sample.events_file, chunk_size=CHUNK_SIZE):
            qc_chunk = add_qc_flags(chunk, self.qc_config.dict(), precomputed_percentiles={'fsc_low': fsc_low, 'ssc_low': ssc_low})
            all_qc_dfs.append(qc_chunk)
            if first_chunk:
                writer = pq.ParquetWriter(events_file, qc_chunk.columns.to_arrow())
                first_chunk = False
            writer.write_table(pa.Table.from_pandas(qc_chunk))
        if writer: writer.close()
        
        full_qc_df = pd.concat(all_qc_dfs)
        sample.events_file = events_file
        return sample, (sample.sample_id, full_qc_df)

class GatingStage(ParallelPipelineStage):
    stage_name = "gate"
    def __init__(self, outdir: Path, workers: int, strategy: str, config: AppConfig):
        super().__init__(outdir, workers)
        self.strategy = strategy
        self.config = config
        self.params_dir = ensure_dir(self.outdir / "params")
        self.figures_dir = ensure_dir(self.outdir / "figures")

    def process_sample(self, sample: Sample) -> Tuple[Sample, Dict[str, Any]]:
        # Pre-pass to build KDE on a sample of the data
        gate_config = self.config.gating.dict()
        gate_config["channels"] = self.config.channels.dict()
        fsc_col = gate_config["channels"].get("fsc_a", "FSC-A")
        ssc_col = gate_config["channels"].get("ssc_a", "SSC-A")
        
        initial_chunks = []
        total_events = 0
        for chunk in stream_events(sample.events_file, chunk_size=CHUNK_SIZE, columns=[fsc_col, ssc_col]):
            initial_chunks.append(chunk)
            total_events += len(chunk)
            if total_events >= 50000: break
        initial_df = pd.concat(initial_chunks)

        # Build gate model
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(initial_df.T)
        percentile = gate_config.get("lymphocytes", {}).get("percentile", 90)
        density_threshold = np.percentile(kde(initial_df.T), 100 - percentile)
        
        events_file = self.events_dir / f"{sample.sample_id}.parquet"
        params_file = self.params_dir / f"{sample.sample_id}.json"
        
        writer = None
        first_chunk = True
        gated_event_count = 0
        total_event_count = 0

        for chunk in stream_events(sample.events_file, chunk_size=CHUNK_SIZE):
            total_event_count += len(chunk)
            density = kde(chunk[[fsc_col, ssc_col]].T)
            gated_chunk = chunk[density > density_threshold].copy()
            gated_event_count += len(gated_chunk)
            
            if first_chunk and not gated_chunk.empty:
                writer = pq.ParquetWriter(events_file, pa.Table.from_pandas(gated_chunk).schema)
                first_chunk = False
            if writer and not gated_chunk.empty:
                writer.write_table(pa.Table.from_pandas(gated_chunk))
        if writer: writer.close()

        params = {"lymphocyte_density_threshold": density_threshold}
        _write_json(params_file, params)
        sample.events_file = events_file
        sample.params_file = params_file
        
        summary_row = {"sample_id": sample.sample_id, "input_events": total_event_count, "gated_events": gated_event_count}
        
        # Plotting is tricky with streaming. For now, we omit it in the streaming version of gate.
        
        return sample, summary_row

    def finalize_stage(self, optional_data: List[Dict[str, Any]]) -> None:
        if not optional_data: return
        pd.DataFrame(optional_data).to_csv(self.outdir / "summary.csv", index=False)
        
    def _get_cache_key(self, input_result: StageResult | None) -> str | None:
        return self._hash_inputs(input_result, self.strategy, self.config.gating, self.config.channels)

class DriftStage(PipelineStage):
    stage_name = "drift"
    def __init__(self, outdir: Path, batch_col: str, config: AppConfig):
        self.outdir = outdir
        self.batch_col = batch_col
        self.config = config

    def _run_logic(self, input_result: StageResult) -> StageResult:
        ensure_dir(self.outdir)
        figures_dir = ensure_dir(self.outdir / "figures")
        sample_events = {s.sample_id: load_dataframe(s.events_file) for s in input_result.samples}
        meta_cols = ["sample_id", self.batch_col]
        if input_result.samples and "condition" in input_result.samples[0].extra_metadata:
            meta_cols.append("condition")
        metadata = pd.DataFrame([{**s.extra_metadata, "sample_id": s.sample_id} for s in input_result.samples])
        metadata = metadata[meta_cols].drop_duplicates()
        markers = self.config.channels.markers
        features = extract_sample_features(sample_events, metadata, marker_channels=markers)
        features.to_csv(self.outdir / "features.csv", index=False)
        drift_res = compute_batch_drift(features, by=self.batch_col)
        drift_res["tests"].to_csv(self.outdir / "tests.csv", index=False)
        drift_res["pca"].to_csv(self.outdir / "pca.csv", index=False)
        if drift_res.get("umap") is not None:
            drift_res["umap"].to_csv(self.outdir / "umap.csv", index=False)
        fig1 = plot_batch_drift_pca(drift_res["pca"], str(figures_dir / "pca.png"), self.batch_col)
        fig2 = plot_batch_drift_umap(drift_res.get("umap"), str(figures_dir / "umap.png"), self.batch_col)
        plt.close(fig1)
        plt.close(fig2)
        return input_result
        
    def _get_cache_key(self, input_result: StageResult | None) -> str | None:
        return self._hash_inputs(input_result, self.batch_col, self.config.channels)

class StatsStage(PipelineStage):
    stage_name = "stats"
    def __init__(self, outdir: Path, group_col: str, value_cols: List[str]):
        self.outdir = outdir
        self.group_col = group_col
        self.value_cols = value_cols

    def _run_logic(self, input_result: StageResult) -> StageResult:
        ensure_dir(self.outdir)
        records = []
        for sample in input_result.samples:
            df = load_dataframe(sample.events_file)
            summary = df[self.value_cols].mean().to_dict()
            summary[self.group_col] = sample.extra_metadata.get(self.group_col)
            summary["sample_id"] = sample.sample_id
            records.append(summary)
        aggregated = pd.DataFrame(records)
        aggregated.to_csv(self.outdir / "per_sample_summary.csv", index=False)
        effects = effect_sizes(aggregated, self.group_col, self.value_cols)
        effects.to_csv(self.outdir / "effect_sizes.csv", index=False)
        fig = plot_effect_sizes(effects, str(self.outdir / "figures" / "effect_sizes.png"))
        plt.close(fig)
        return input_result
        
    def _get_cache_key(self, input_result: StageResult | None) -> str | None:
        return self._hash_inputs(input_result, self.group_col, "".join(sorted(self.value_cols)))

class ReportStage(PipelineStage):
    """Generates the final interactive HTML report."""
    stage_name = "report"

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def _run_logic(self, input_result: StageResult | None = None) -> StageResult | None:
        report_path = self.root_dir / "report.html"
        create_and_save_report(self.root_dir, report_path)
        logger.info(f"Interactive report saved to {report_path}")
        return None  # This is a terminal stage

    def _get_cache_key(self, input_result: StageResult | None) -> str | None:
        # Don't cache report generation
        return None
