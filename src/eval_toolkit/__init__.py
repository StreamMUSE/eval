"""StreamMUSE evaluation toolkit.

Provides modular, dependency-free utilities for processing, analyzing,
and exporting evaluation metrics from StreamMUSE experiments.

Sub-modules:
    path_utils   - metric type registry and path resolution helpers
    json_parser  - extract values from RESULT / NLL / EXP_RAW JSON files
    stats        - descriptive statistics (stdlib only, no numpy)
    csv_exporter - CLI tool to aggregate and export results to CSV
"""

from .path_utils import get_path, get_keys_from_dir, Type, TypeFromNll, TypeFromResult, TypeFromExpRaw
from .json_parser import parse_by_type, parse_nll_file, parse_result_file
from .stats import compute_stats
from .csv_exporter import main as export_csv

__all__ = [
    "get_path",
    "get_keys_from_dir",
    "Type",
    "TypeFromNll",
    "TypeFromResult",
    "TypeFromExpRaw",
    "parse_by_type",
    "parse_nll_file",
    "parse_result_file",
    "compute_stats",
    "export_csv",
]
