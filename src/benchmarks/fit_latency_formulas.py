#!/usr/bin/env python3
"""
Fit quadratic latency formulas for StreamMUSE real-time benchmarks.

This helper computes per-setting quadratic fits of

    round_trip_time_ms ≈ a * GL^2 + b * GL + c

where GL is the generation length in frames and round_trip_time_ms is the
mean round-trip latency in milliseconds at each GL.

Usage (from AE root):

    cd eval
    uv run -m src.benchmarks.fit_latency_formulas configs/analysis_config_fig3_fig4.yaml

It will:
  - Load the `experiments[*].path` entries from the given YAML config.
  - Use `BulkGenerationLengthAnalyzer` to aggregate benchmark CSVs.
  - Fit a quadratic curve for each experiment using mean RTT vs GL.
  - Print YAML snippets with `a`, `b`, `c` (and R²) that you can copy/paste
    back into `formula:` blocks in the config.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from .bulk_analysis import BulkGenerationLengthAnalyzer


def _fit_quadratic(gl_values: np.ndarray, rtt_ms: np.ndarray) -> Tuple[float, float, float, float]:
    """Fit y ≈ a * x^2 + b * x + c and return a, b, c, r2."""
    if len(gl_values) < 3:
        raise ValueError("Need at least 3 distinct generation lengths to fit a quadratic model.")

    coeffs = np.polyfit(gl_values, rtt_ms, 2)
    a, b, c = coeffs

    y_pred = np.polyval(coeffs, gl_values)
    ss_res = float(np.sum((rtt_ms - y_pred) ** 2))
    ss_tot = float(np.sum((rtt_ms - np.mean(rtt_ms)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), float(c), float(r2)


def _load_summary_for_path(data_path: Path, anomaly_filter_pct: float) -> pd.DataFrame:
    """Use BulkGenerationLengthAnalyzer to build summary stats for a single experiment directory."""
    analyzer = BulkGenerationLengthAnalyzer(
        data_sources=[data_path],
        output_dir=Path("results/benchmarks/formula_fit"),
        anomaly_filter_pct=anomaly_filter_pct,
    )
    if not analyzer.load_all_experiments():
        raise ValueError(f"Failed to load any benchmark data from {data_path}")
    analyzer.combine_all_data()
    if analyzer.summary_data is None:
        raise ValueError(f"No summary data available after combining experiments for {data_path}")
    return analyzer.summary_data


def _fit_for_experiment(
    exp_cfg: Dict[str, Any],
    anomaly_filter_pct: float,
    base_dir: Path,
) -> Dict[str, Any]:
    """Fit quadratic coefficients for a single experiment configuration."""
    rel_path = Path(exp_cfg["path"])
    data_path = (base_dir / rel_path).resolve() if not rel_path.is_absolute() else rel_path
    if not data_path.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {data_path}")

    summary = _load_summary_for_path(data_path, anomaly_filter_pct)

    # BulkGenerationLengthAnalyzer names the experiment after the directory name.
    exp_name = data_path.name
    df = summary[summary["experiment_name"] == exp_name].copy()
    if df.empty:
        # Fallback: if experiment_name does not match, just use all rows
        df = summary.copy()

    if "round_trip_time_mean" not in df.columns:
        raise ValueError(f"No round_trip_time_mean column found for {data_path}")

    df = df.sort_values("generation_length")
    gl = df["generation_length"].to_numpy()
    rtt_ms = (df["round_trip_time_mean"] * 1000.0).to_numpy()

    a, b, c, r2 = _fit_quadratic(gl, rtt_ms)
    return {
        "formal_name": exp_cfg.get("formal_name", exp_name),
        "path": str(rel_path),
        "a": a,
        "b": b,
        "c": c,
        "r2": r2,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit quadratic latency formulas for StreamMUSE benchmarks.",
        epilog=(
            "Run this before yaml_bulk_analysis to compute a, b, c for each setting.\n"
            "Then copy the printed formula blocks into your YAML config."
        ),
    )
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to analysis_config_fig3_fig4.yaml (or a similar config).",
    )
    parser.add_argument(
        "--anomaly_filter",
        type=float,
        default=None,
        metavar="PCT",
        help="Override anomaly filter percentage per tail (defaults to value in config, or 0).",
    )
    args = parser.parse_args(argv)

    if not args.config_file.exists():
        raise SystemExit(f"Config file not found: {args.config_file}")

    with args.config_file.open("r") as f:
        cfg = yaml.safe_load(f)

    experiments = cfg.get("experiments", [])
    if not experiments:
        raise SystemExit("No 'experiments' section found in config.")

    analysis_settings = cfg.get("analysis_settings", {})
    anomaly_filter_pct = (
        args.anomaly_filter
        if args.anomaly_filter is not None
        else float(analysis_settings.get("anomaly_filter_percentage", 0.0))
    )

    base_dir = Path.cwd()

    print("📐 Fitting quadratic latency formulas")
    print("====================================")
    print(f"Config file: {args.config_file}")
    print(f"Anomaly filter per tail: {anomaly_filter_pct}%")
    print("")

    fits: List[Dict[str, Any]] = []
    for exp_cfg in experiments:
        try:
            result = _fit_for_experiment(exp_cfg, anomaly_filter_pct, base_dir)
        except Exception as e:  # pragma: no cover - CLI error path
            print(f"❌ Failed to fit experiment at path {exp_cfg.get('path')}: {e}")
            continue
        fits.append(result)

    if not fits:
        print("❌ No fits were produced. Please check that your experiment folders and CSVs exist.")
        return 1

    print("✅ Fit results:")
    for res in fits:
        print(f"- {res['formal_name']} ({res['path']}):")
        print(f"    a = {res['a']:.4f}, b = {res['b']:.4f}, c = {res['c']:.4f}, R² = {res['r2']:.4f}")
    print("")

    print("YAML snippets to copy into your config (replace existing formula blocks):")
    print("-----------------------------------------------------------------------")
    for res in fits:
        print(f"# {res['formal_name']} ({res['path']})")
        print("formula:")
        print('  type: "quadratic"')
        print(f"  a: {res['a']:.4f}")
        print(f"  b: {res['b']:.4f}")
        print(f"  c: {res['c']:.4f}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

