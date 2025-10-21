"""Build HTML reports from generated artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from cytoflow_qc.experiment_design import ExperimentManager, CohortManager


def build_report(artifacts_dir: str, template: str, out: str, experiment_dir: str | Path | None = None) -> None:
    """Render the HTML report from pipeline outputs."""

    base = Path(artifacts_dir)
    context = _gather_context(base, experiment_dir)
    tpl_path = Path(template)
    env = Environment(
        loader=FileSystemLoader(str(tpl_path.parent)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    html = env.get_template(tpl_path.name).render(**context)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html, encoding="utf-8")


def _gather_context(base: Path, experiment_dir: str | Path | None) -> Dict[str, object]:
    qc_summary = _safe_read_csv(base / "qc" / "summary.csv")
    gate_summary = _safe_read_csv(base / "gate" / "summary.csv")
    drift_tests = _safe_read_csv(base / "drift" / "tests.csv")
    effects = _safe_read_csv(base / "stats" / "effect_sizes.csv")

    # Initialize ExperimentManager and CohortManager if experiment_dir is provided
    experiment_info: Dict[str, Any] = {}
    cohort_info: Dict[str, Any] = {}

    if experiment_dir:
        try:
            exp_manager = ExperimentManager(experiment_dir)
            experiment_info = exp_manager.get_experiment_info()

            cohort_manager = CohortManager(exp_manager)
            cohort_info = {
                "cohort_names": cohort_manager.list_cohorts(),
                "cohort_details": {name: cohort_manager.get_cohort_details(name) for name in cohort_manager.list_cohorts()}
            }
        except Exception as e:
            print(f"Warning: Could not load experiment design from {experiment_dir}: {e}")

    total_samples = int(qc_summary["sample_id"].nunique()) if "sample_id" in qc_summary else 0
    total_events = int((gate_summary.get("gated_events") if not gate_summary.empty else pd.Series()).sum())
    if total_events == 0 and "total_events" in qc_summary:
        total_events = int(qc_summary["total_events"].sum())

    summary = {
        "total_samples": total_samples,
        "total_events": total_events,
        "qc_pass_rate": float(qc_summary.get("qc_pass_fraction", pd.Series([0.0])).mean() if not qc_summary.empty else 0.0),
        "analysis_date": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
    }

    quality_issues: List[str] = []
    if not qc_summary.empty:
        high_debris = qc_summary.loc[qc_summary.get("debris_fraction", pd.Series([0.0])) > 0.2, "sample_id"].tolist()
        if high_debris:
            quality_issues.append(f"High debris fraction in: {', '.join(high_debris)}")

    figures = []
    for path in sorted((base / "drift" / "figures").glob("*.png")):
        figures.append({"path": path.relative_to(base), "caption": path.stem.replace("_", " ")})
    for path in sorted((base / "stats" / "figures").glob("*.png")):
        figures.append({"path": path.relative_to(base), "caption": path.stem.replace("_", " ")})

    return {
        "summary": summary,
        "qc_summary": qc_summary.to_dict(orient="records") if not qc_summary.empty else [],
        "gating_summary": gate_summary.to_dict(orient="records") if not gate_summary.empty else [],
        "drift_tests": drift_tests.to_dict(orient="records") if not drift_tests.empty else [],
        "effect_sizes": effects.to_dict(orient="records") if not effects.empty else [],
        "quality_issues": quality_issues,
        "figures": figures,
        "experiment_info": experiment_info, # Add experiment info to context
        "cohort_info": cohort_info,         # Add cohort info to context
    }


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
