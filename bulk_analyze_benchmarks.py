#!/usr/bin/env python3
"""
Bulk-analyze StreamMUSE benchmark experiments from the eval repo.

This script wraps `src.benchmarks.bulk_analysis.BulkGenerationLengthAnalyzer`.

Typical usage (from eval/):

  # Analyze multiple experiment directories (manual or bulk_benchmark) and
  # write combined CSVs + plots under results/benchmarks/bulk_analysis
  uv run bulk_analyze_benchmarks.py \\
    ../StreamMUSE/experiments/exp1 \\
    ../StreamMUSE/experiments/exp2 \\
    --output_dir results/benchmarks/bulk_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure eval repo root is on path for src.benchmarks
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarks.bulk_analysis import run_bulk_analysis  # noqa: E402


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Bulk analysis for multiple StreamMUSE benchmark experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "data_sources",
        nargs="+",
        type=Path,
        help="One or more experiment directories (manual gen_*.csv dirs or bulk_benchmark outputs).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/benchmarks/bulk_analysis"),
        help="Output directory for combined CSVs and plots.",
    )
    parser.add_argument(
        "--anomaly_filter",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Remove PCT%% from each tail per GL for each experiment (0–50).",
    )
    args = parser.parse_args()

    if args.anomaly_filter < 0 or args.anomaly_filter > 50:
        print("❌ --anomaly_filter must be between 0 and 50")
        return 1

    return run_bulk_analysis(
        data_sources=args.data_sources,
        output_dir=args.output_dir,
        anomaly_filter_pct=args.anomaly_filter,
    )


if __name__ == "__main__":
    raise SystemExit(_main())

