"""Custom exception classes for CytoFlow-QC."""


class CytoflowQCError(Exception):
    """Base exception for all cytoflow-qc errors."""
    pass


class ConfigurationError(CytoflowQCError):
    """Raised for configuration-related errors."""
    pass


class ValidationError(CytoflowQCError):
    """Raised when input validation fails (samplesheet, config, channels, etc.)."""
    pass


class DataProcessingError(CytoflowQCError):
    """Raised for errors during data processing stages."""
    pass


class FileOperationError(CytoflowQCError):
    """Raised for file reading or writing errors."""
    pass


class GatingError(DataProcessingError):
    """Raised for errors during the gating stage."""
    pass


class QCError(DataProcessingError):
    """Raised for errors during the QC stage."""
    pass


class CompensationError(DataProcessingError):
    """Raised for errors during compensation stage."""
    pass


class DriftAnalysisError(DataProcessingError):
    """Raised for errors during batch drift analysis."""
    pass


class PluginError(CytoflowQCError):
    """Raised when a plugin fails to load or execute."""
    pass






