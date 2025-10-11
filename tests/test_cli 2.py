from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cytoflow_qc.cli import app


def test_cli_run_smoke(tmp_path: Path, samplesheet_path: Path, config_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--samplesheet",
            str(samplesheet_path),
            "--config",
            str(config_path),
            "--out",
            str(tmp_path / "results"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    report_path = tmp_path / "results" / "report.html"
    assert report_path.exists()
