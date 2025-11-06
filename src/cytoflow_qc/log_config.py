"""Logging configuration for CytoFlow-QC."""
import logging
import sys
from pathlib import Path

def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """Set up logging to console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "cytoflow_qc.log"

    log_level_upper = log_level.upper()
    numeric_level = getattr(logging, log_level_upper, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Root logger configuration
    logger = logging.getLogger("cytoflow_qc")
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Redirect uncaught exceptions to the logger
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception




