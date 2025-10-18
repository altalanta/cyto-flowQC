#!/usr/bin/env python3
"""Generate API documentation for cytoflow-qc using pdoc."""

import subprocess
import sys
from pathlib import Path


def generate_api_docs() -> None:
    """Generate API documentation using pdoc."""
    # Check if pdoc is available
    try:
        import pdoc
    except ImportError:
        print("pdoc not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pdoc"], check=True)

    # Generate API docs
    src_path = Path(__file__).parent.parent / "src"
    output_dir = Path(__file__).parent / "api"

    print(f"Generating API documentation from {src_path} to {output_dir}")

    # Use pdoc to generate HTML docs
    cmd = [
        sys.executable, "-m", "pdoc",
        "--output-directory", str(output_dir),
        "--docformat", "google",
        "cytoflow_qc"
    ]

    subprocess.run(cmd, cwd=src_path, check=True)

    print(f"API documentation generated in {output_dir}")


if __name__ == "__main__":
    generate_api_docs()







