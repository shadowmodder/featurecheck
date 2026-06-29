"""featurecheck: drift + schema validation for ML feature pipelines."""
from .core import psi, ks_statistic, check_schema, drift_report

__all__ = ["psi", "ks_statistic", "check_schema", "drift_report"]
__version__ = "0.1.0"
