#!/bin/bash
# Development setup script for CytoFlow-QC

set -e

echo "Setting up CytoFlow-QC development environment..."

# Create conda environment if it doesn't exist
if ! conda info --envs | grep -q cytoflow-qc; then
    echo "Creating conda environment..."
    conda env create -f env/environment.yml
fi

# Activate environment
echo "Activating environment..."
conda activate cytoflow-qc

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
pip install -e .[dev]

# Run linting to check for issues
echo "Running linting checks..."
make lint

echo "Development environment setup complete!"
echo "You can now run:"
echo "  make test    - Run tests"
echo "  make lint    - Check code style"
echo "  make format  - Format code"

