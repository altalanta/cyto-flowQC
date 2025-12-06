from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cytoflow_qc.cli import app


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Typer CLI test runner."""
    return CliRunner()


def test_cli_run_smoke(tmp_path: Path, samplesheet_path: Path, config_path: Path, cli_runner: CliRunner) -> None:
    """Smoke test for the full pipeline run command."""
    result = cli_runner.invoke(
        app,
        [
            "run",
            "--samplesheet",
            str(samplesheet_path),
            "--config",
            str(config_path),
            "--out",
            str(tmp_path / "results"),
            "--batch",
            "batch",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    report_path = tmp_path / "results" / "report.html"
    assert report_path.exists()


def test_cli_version(cli_runner: CliRunner) -> None:
    """Test that --version outputs the version string."""
    result = cli_runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    # Version should be in the output (either a real version or "unknown")
    assert result.output.strip()


def test_cli_validate_missing_files(tmp_path: Path, config_path: Path, cli_runner: CliRunner) -> None:
    """Test validation catches missing FCS files."""
    # Create a samplesheet with a non-existent file
    bad_samplesheet = tmp_path / "bad_samplesheet.csv"
    bad_samplesheet.write_text("sample_id,file_path\nsample_001,/nonexistent/file.fcs\n")
    
    result = cli_runner.invoke(
        app,
        [
            "validate",
            "--samplesheet",
            str(bad_samplesheet),
            "--config",
            str(config_path),
        ],
    )
    # Should fail because file doesn't exist
    assert result.exit_code == 1
