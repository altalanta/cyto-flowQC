"""CytoFlow-QC: Automated QC and gating pipeline for flow cytometry data."""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__author__ = "CytoFlow-QC Team"
__email__ = "cytoflow-qc@example.com"

# Import main modules for easier access
from . import cli, compensate, drift, gate, io, qc, report, stats, utils
from . import interactive_viz, viz_3d
from . import plugins, ml_plugins
from . import cloud, realtime, high_performance
from . import security
from . import experiment_design
from . import data_connectors
