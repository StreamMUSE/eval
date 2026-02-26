"""Benchmark analysis for StreamMUSE real-time latency experiments.

Standard real-time settings used in the paper (exactly three):
- remote:   Remote server, PC client (e.g. SSH port-forward)
- local:    PC server, PC client (same machine)
- local_server: PC server, Mac client (e.g. SSH port-forward)
"""

from src.benchmarks.constants import REAL_TIME_SETTINGS

__all__ = ["REAL_TIME_SETTINGS"]
