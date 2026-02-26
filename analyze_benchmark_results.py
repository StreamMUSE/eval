#!/usr/bin/env python3
"""
Analyze StreamMUSE benchmark results from the eval repo.

Standard real-time settings (paper): remote, local, local_server.
Reads from StreamMUSE bench_results/<setting>/ (or bench_results/manual_<setting>/)
and writes summary tables + plots under results/benchmarks/<setting>/.

Usage (from eval/):
  uv run analyze_benchmark_results.py --bench_results_dir ../StreamMUSE/bench_results --setting remote
  uv run analyze_benchmark_results.py --bench_results_dir ../StreamMUSE/bench_results --all
  uv run analyze_benchmark_results.py ../StreamMUSE/bench_results/gen_length_sweep/raw_data --output_dir results/benchmarks/sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure eval repo root is on path for src.benchmarks
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.benchmarks.analyze_generation_length import run_analysis
from src.benchmarks.constants import REAL_TIME_SETTINGS


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze StreamMUSE benchmark results (standard settings: remote, local, local_server).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "data_source",
        type=Path,
        nargs="?",
        default=None,
        help="Path to benchmark data dir (or use --bench_results_dir + --setting)",
    )
    parser.add_argument(
        "--bench_results_dir",
        type=Path,
        default=None,
        help="StreamMUSE bench_results directory (e.g. ../StreamMUSE/bench_results). Use with --setting or --all.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=REAL_TIME_SETTINGS,
        default=None,
        help=f"Real-time setting: one of {REAL_TIME_SETTINGS}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Run analysis for all settings: {REAL_TIME_SETTINGS}",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/benchmarks"),
        help="Base output directory; per-setting outputs go under <output_dir>/<setting>/",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--anomaly_filter",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Remove PCT%% from each tail per generation length (0-50)",
    )
    args = parser.parse_args()

    if args.anomaly_filter < 0 or args.anomaly_filter > 50:
        print("❌ --anomaly_filter must be between 0 and 50")
        return 1

    # Resolve runs: (data_source,) or (bench_results_dir, setting) or (bench_results_dir, all)
    if args.data_source is not None:
        if args.bench_results_dir is not None or args.setting is not None or args.all:
            print("❌ When providing data_source, do not use --bench_results_dir / --setting / --all")
            return 1
        return run_analysis(
            args.data_source,
            args.output_dir,
            anomaly_filter_pct=args.anomaly_filter,
            no_plots=args.no_plots,
        )

    if args.bench_results_dir is None:
        print("❌ Provide either data_source or --bench_results_dir with --setting or --all")
        return 1
    bench_root = Path(args.bench_results_dir)
    if not bench_root.is_dir():
        print(f"❌ Not a directory: {bench_root}")
        return 1

    if args.all:
        settings = REAL_TIME_SETTINGS
    elif args.setting:
        settings = (args.setting,)
    else:
        print("❌ Use --setting <name> or --all with --bench_results_dir")
        return 1

    # Standard subdir names; accept legacy manual_<setting> if standard missing
    ok = 0
    for s in settings:
        data_dir = bench_root / s
        if not data_dir.is_dir():
            data_dir = bench_root / f"manual_{s}"
        if not data_dir.is_dir():
            print(f"⚠️ Skipping {s}: no dir at {bench_root / s} or {bench_root / f'manual_{s}'}")
            continue
        out_dir = args.output_dir / s
        print(f"\n=== {s} ===")
        code = run_analysis(
            data_dir,
            out_dir,
            anomaly_filter_pct=args.anomaly_filter,
            no_plots=args.no_plots,
        )
        if code == 0:
            ok += 1
    if ok == 0:
        print("❌ No settings processed successfully")
        return 1
    print(f"\n✅ Done: {ok}/{len(settings)} settings")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
