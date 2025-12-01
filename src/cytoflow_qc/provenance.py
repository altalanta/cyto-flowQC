"""Module for capturing and recording data provenance for a pipeline run."""

import logging
import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from importlib.metadata import distributions
import platform

from cytoflow_qc.config import AppConfig

logger = logging.getLogger(__name__)


def _run_command(command: list[str]) -> Optional[str]:
    """Helper to run a shell command and return its output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        logger.debug(f"Command {command} failed: {e}")
        return None


def get_git_info() -> Dict[str, Any]:
    """Retrieves Git repository information."""
    try:
        commit = _run_command(["git", "rev-parse", "HEAD"])
        branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        status = _run_command(["git", "status", "--porcelain"])
        return {
            "commit": commit,
            "branch": branch,
            "is_dirty": bool(status),
            "status": status,
        }
    except Exception as e:
        logger.warning(f"Failed to retrieve Git info: {e}")
        return {"error": str(e), "status": "Git information unavailable."}


def get_dvc_info(dvc_repo_path: Path) -> Dict[str, Any]:
    """Retrieves DVC information for the repository."""
    try:
        dvc_status_output = _run_command(["dvc", "status", "--json"])
        if dvc_status_output:
            try:
                return json.loads(dvc_status_output)
            except json.JSONDecodeError:
                return {"status": "DVC output was not valid JSON.", "raw_output": dvc_status_output}
        return {"status": "DVC not found or no files tracked."}
    except Exception as e:
        logger.warning(f"Failed to retrieve DVC info: {e}")
        return {"error": str(e), "status": "DVC information unavailable."}


def get_dependencies() -> Dict[str, str]:
    """Lists all installed packages and their versions using importlib.metadata."""
    try:
        deps = {}
        for dist in distributions():
            name = dist.metadata.get("Name")
            version = dist.version
            if name:
                deps[name] = version
        return deps
    except Exception as e:
        logger.warning(f"Failed to retrieve dependencies: {e}")
        return {"error": str(e)}


def get_platform_info() -> Dict[str, str]:
    """Gets information about the execution platform."""
    try:
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
        }
    except Exception as e:
        logger.warning(f"Failed to retrieve platform info: {e}")
        return {"error": str(e)}

def generate_provenance_report(
    config: AppConfig,
    samplesheet_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Generates a complete provenance report for a pipeline run.
    
    This function is designed to be resilient - it will capture as much
    information as possible and log warnings for any failures, but will
    not raise exceptions that could interrupt the main pipeline.
    
    Args:
        config: The AppConfig object used for the run.
        samplesheet_path: Path to the samplesheet CSV.
        output_dir: The main output directory for the run.
        
    Returns:
        A dictionary containing the full provenance report.
        Returns an error report if critical failures occur.
    """
    report: Dict[str, Any] = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_version": "1.0",
    }
    
    # Capture platform info (should rarely fail)
    report["platform"] = get_platform_info()
    
    # Capture Git info (may fail if not in a git repo)
    report["git"] = get_git_info()
    
    # Capture DVC info (may fail if DVC not installed)
    report["dvc"] = get_dvc_info(Path("."))
    
    # Read samplesheet content
    try:
        with open(samplesheet_path, "r", encoding="utf-8") as f:
            samplesheet_content = f.read()
        report["inputs"] = {
            "samplesheet": {
                "filename": samplesheet_path.name,
                "content": samplesheet_content,
            },
        }
    except (OSError, IOError) as e:
        logger.warning(f"Failed to read samplesheet for provenance: {e}")
        report["inputs"] = {
            "samplesheet": {
                "filename": samplesheet_path.name,
                "error": str(e),
            },
        }
    
    # Capture configuration
    try:
        report["inputs"]["configuration"] = config.model_dump()
    except Exception as e:
        logger.warning(f"Failed to serialize config for provenance: {e}")
        report["inputs"]["configuration"] = {"error": str(e)}
    
    # Capture dependencies
    report["dependencies"] = get_dependencies()
    
    # Save the report
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        provenance_file = output_dir / "provenance.json"
        with open(provenance_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Provenance report saved to {provenance_file}")
    except (OSError, IOError, TypeError) as e:
        logger.error(f"Failed to save provenance report: {e}")
        # Don't raise - provenance failure shouldn't stop the pipeline
        
    return report


