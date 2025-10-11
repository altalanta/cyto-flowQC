"""Build HTML reports from generated artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape


def build_report(artifacts_dir: str, template: str, out: str) -> None:
    """Render the HTML report from pipeline outputs."""

    base = Path(artifacts_dir)
    context = _gather_context(base)
    tpl_path = Path(template)
    env = Environment(
        loader=FileSystemLoader(str(tpl_path.parent)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    html = env.get_template(tpl_path.name).render(**context)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html, encoding="utf-8")


def _gather_context(base: Path) -> Dict[str, object]:
    qc_summary = _safe_read_csv(base / "qc" / "summary.csv")
    gate_summary = _safe_read_csv(base / "gate" / "summary.csv")
    drift_tests = _safe_read_csv(base / "drift" / "tests.csv")
    effects = _safe_read_csv(base / "stats" / "effect_sizes.csv")

    total_samples = int(qc_summary["sample_id"].nunique()) if "sample_id" in qc_summary else 0
    total_events = int(
        (gate_summary.get("gated_events") if not gate_summary.empty else pd.Series()).sum()
    )
    if total_events == 0 and "total_events" in qc_summary:
        total_events = int(qc_summary["total_events"].sum())

    summary = {
        "total_samples": total_samples,
        "total_events": total_events,
        "qc_pass_rate": float(
            qc_summary.get("qc_pass_fraction", pd.Series([0.0])).mean()
            if not qc_summary.empty
            else 0.0
        ),
        "analysis_date": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
    }

    quality_issues: list[str] = []
    if not qc_summary.empty:
        DEBRIS_THRESHOLD = 0.2
        high_debris = qc_summary.loc[
            qc_summary.get("debris_fraction", 0) > DEBRIS_THRESHOLD, "sample_id"
        ].tolist()
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
    }


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
