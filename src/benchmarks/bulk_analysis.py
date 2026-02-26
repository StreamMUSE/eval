#!/usr/bin/env python3
"""
Bulk analysis for StreamMUSE generation-length benchmarks (eval repo).

This module is a port of the `BulkGenerationLengthAnalyzer` logic from the
`bench_results` branch. It combines multiple benchmark experiments
(manual CSVs or bulk_benchmark combined_results.csv) and produces:

- Combined detailed CSV:  processed_data/all_experiments_combined.csv
- Per-experiment summary: processed_data/all_experiments_summary.csv
- Comparative plots under plots/:
  - experiment_comparison.png
  - generation_length_trends.png
  - performance_comparison.png
  - variability_comparison.png
  - constraint_analysis.png
  - detailed_constraint_analysis.png
  - constraint_heatmaps.png
  - parameter_constraint_analysis.png

The implementation here focuses on matching the *shape* of the outputs
from bench_results (filenames, metrics, basic styling), while staying
within the eval repo’s packaging.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class ExperimentData:
    """Container for one experiment's benchmark data."""

    type: str  # "bulk_benchmark" or "manual_benchmark"
    name: str
    data: pd.DataFrame
    source_path: Path
    metadata: Dict[str, Any]
    generation_lengths: List[int]
    total_requests: int


class BulkGenerationLengthAnalyzer:
    """
    Analyze generation length effects across multiple experiment directories.

    Inputs:
    - List of experiment directories. Each directory is either:
      - bulk_benchmark output (contains combined_results.csv), or
      - manual benchmark dir with gen_*.csv / gen_length_*.csv, etc.

    Outputs:
    - processed_data/all_experiments_combined.csv
    - processed_data/all_experiments_summary.csv
    - processed_data/constraint_analysis.csv (simple constraint stats)
    - plots/*.png (comparative plots)
    """

    def __init__(
        self,
        data_sources: List[Path],
        output_dir: Path = Path("results/benchmarks/bulk_analysis"),
        anomaly_filter_pct: float = 0.0,
    ):
        self.data_sources = [Path(source) for source in data_sources]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = self.output_dir / "processed_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.experiments: Dict[str, ExperimentData] = {}
        self.combined_data: Optional[pd.DataFrame] = None
        self.summary_data: Optional[pd.DataFrame] = None
        self.anomaly_filter_pct = anomaly_filter_pct

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def load_all_experiments(self) -> bool:
        """Load data from all experiment directories."""
        print(f"🔍 Loading data from {len(self.data_sources)} experiment directories...")
        success_count = 0

        for idx, data_source in enumerate(self.data_sources):
            if not data_source.exists():
                print(f"❌ Directory not found: {data_source}")
                continue

            experiment_name = self._generate_experiment_name(data_source, idx)
            print(f"\n📂 Loading experiment: {experiment_name}")
            print(f"   Source: {data_source}")

            exp = self._load_single_experiment(data_source, experiment_name)
            if exp is None:
                print(f"❌ Failed to load {experiment_name}")
                continue

            self.experiments[experiment_name] = exp
            success_count += 1
            print(f"✅ Successfully loaded {experiment_name}")

        if success_count == 0:
            print("❌ No experiments could be loaded")
            return False

        print(f"\n✅ Successfully loaded {success_count} out of {len(self.data_sources)} experiments")
        return True

    def _generate_experiment_name(self, data_source: Path, index: int) -> str:
        """Derive a readable experiment name from the directory path."""
        base_name = data_source.name or f"experiment_{index + 1}"
        return base_name

    def _load_single_experiment(self, data_source: Path, experiment_name: str) -> Optional[ExperimentData]:
        """Dispatch to bulk_benchmark or manual CSV loader."""
        if self._is_bulk_benchmark_output(data_source):
            return self._load_bulk_benchmark_data(data_source, experiment_name)
        return self._load_manual_benchmark_data(data_source, experiment_name)

    def _is_bulk_benchmark_output(self, data_source: Path) -> bool:
        """Detect whether a directory looks like bulk_benchmark output."""
        indicators = [
            data_source / "combined_results.csv",
            data_source / "config.yaml",
            data_source / "experiment_summary.json",
        ]
        return any(p.exists() for p in indicators)

    # ---- anomaly filtering ---------------------------------------------------

    def _apply_anomaly_filter(self, df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
        """Filter RTT anomalies per generation_length for one experiment."""
        if self.anomaly_filter_pct <= 0 or "round_trip_time" not in df.columns:
            return df
        if "generation_length" not in df.columns:
            print(f"   ⚠️ {experiment_name}: no generation_length column; skipping anomaly filtering")
            return df

        original_count = len(df)
        filtered_parts: List[pd.DataFrame] = []
        print(f"   🔍 Anomaly filtering for {experiment_name} ({self.anomaly_filter_pct}% each tail):")

        for gl in sorted(df["generation_length"].unique()):
            sub = df[df["generation_length"] == gl]
            n = len(sub)
            if n < 10:
                print(f"      GL {gl}: {n} samples (too few, no filtering)")
                filtered_parts.append(sub)
                continue
            lo = sub["round_trip_time"].quantile(self.anomaly_filter_pct / 100)
            hi = sub["round_trip_time"].quantile(1 - self.anomaly_filter_pct / 100)
            kept = sub[(sub["round_trip_time"] >= lo) & (sub["round_trip_time"] <= hi)].copy()
            removed = n - len(kept)
            print(
                f"      GL {gl}: {n} → {len(kept)} samples "
                f"(removed {removed}, {removed/n*100:.1f}%), RTT range {lo*1000:.1f}–{hi*1000:.1f}ms"
            )
            filtered_parts.append(kept)

        if not filtered_parts:
            return df

        filtered_df = pd.concat(filtered_parts, ignore_index=True)
        removed_total = original_count - len(filtered_df)
        print(
            f"      Total for {experiment_name}: {original_count} → {len(filtered_df)} "
            f"(removed {removed_total}, {removed_total/original_count*100:.1f}%)"
        )
        return filtered_df

    # ---- bulk_benchmark loader ----------------------------------------------

    def _load_bulk_benchmark_data(self, data_source: Path, experiment_name: str) -> Optional[ExperimentData]:
        """Load data from bulk_benchmark combined_results.csv."""
        combined_file = data_source / "combined_results.csv"
        if not combined_file.exists():
            print(f"   ❌ No combined_results.csv in {data_source}")
            return None

        try:
            df = pd.read_csv(combined_file)
        except Exception as e:
            print(f"   ❌ Error reading {combined_file}: {e}")
            return None

        gen_col = None
        for col in ["generation_length", "generation_length_frames"]:
            if col in df.columns:
                gen_col = col
                break
        if gen_col is None:
            print(f"   ❌ No generation length column in {combined_file}")
            return None

        if gen_col != "generation_length":
            df["generation_length"] = df[gen_col]

        df = self._apply_anomaly_filter(df, experiment_name)
        gen_lengths = sorted(df["generation_length"].unique())

        # Minimal metadata from summary file if present
        metadata: Dict[str, Any] = {}
        summary_file = data_source / "experiment_summary.json"
        if summary_file.exists():
            try:
                metadata["summary"] = json.loads(summary_file.read_text())
            except Exception:
                pass

        return ExperimentData(
            type="bulk_benchmark",
            name=experiment_name,
            data=df,
            source_path=data_source,
            metadata=metadata,
            generation_lengths=gen_lengths,
            total_requests=len(df),
        )

    # ---- manual benchmark loader --------------------------------------------

    def _load_manual_benchmark_data(self, data_source: Path, experiment_name: str) -> Optional[ExperimentData]:
        """Load data from manual gen_*.csv / gen_length_*.csv files."""
        csv_files = list(data_source.glob("*.csv"))
        csv_files.extend(list(data_source.glob("**/*.csv")))
        if not csv_files:
            print(f"   ❌ No CSV files in {data_source}")
            return None

        dfs: List[pd.DataFrame] = []
        gls: List[int] = []

        for p in csv_files:
            gl = self._extract_generation_length_from_filename(p.name)
            if gl is None:
                print(f"   ⚠️ Skipping {p.name}: cannot extract generation length")
                continue
            try:
                df = pd.read_csv(p)
            except Exception as e:
                print(f"   ⚠️ Error reading {p}: {e}")
                continue
            df["generation_length"] = gl
            df["source_file"] = p.name
            dfs.append(df)
            gls.append(gl)

        if not dfs:
            print(f"   ❌ No valid manual CSV files in {data_source}")
            return None

        combined = pd.concat(dfs, ignore_index=True)
        combined = self._apply_anomaly_filter(combined, experiment_name)

        return ExperimentData(
            type="manual_benchmark",
            name=experiment_name,
            data=combined,
            source_path=data_source,
            metadata={},
            generation_lengths=sorted(set(gls)),
            total_requests=len(combined),
        )

    @staticmethod
    def _extract_generation_length_from_filename(filename: str) -> Optional[int]:
        """Best-effort GL extraction from filenames."""
        patterns = [
            r"gen_length_(\d+)",
            r"gen_(\d+)",
            r"generation_(\d+)",
            r"gl_(\d+)",
            r"GL(\d+)",
            r"(\d+)_frames",
            r"frames_(\d+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, filename, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    # -------------------------------------------------------------------------
    # Combination + summary
    # -------------------------------------------------------------------------

    def combine_all_data(self) -> None:
        """Combine detailed data and build per-experiment summaries."""
        print("\n🔗 Combining data from all experiments...")
        if not self.experiments:
            print("❌ No experiments loaded")
            return

        all_detailed: List[pd.DataFrame] = []
        summary_rows: List[Dict[str, Any]] = []

        for exp_name, exp in self.experiments.items():
            df = exp.data.copy()
            df["experiment_source"] = exp_name
            df["experiment_type"] = exp.type
            all_detailed.append(df)

            for row in self._calculate_experiment_summary(exp):
                summary_rows.append(row)

        self.combined_data = pd.concat(all_detailed, ignore_index=True)
        self.summary_data = pd.DataFrame(summary_rows)

        combined_file = self.data_dir / "all_experiments_combined.csv"
        self.combined_data.to_csv(combined_file, index=False)

        summary_file = self.data_dir / "all_experiments_summary.csv"
        self.summary_data.to_csv(summary_file, index=False)

        # Simple constraint analysis CSV (for parity)
        constraint_df = self._calculate_constraint_satisfaction_rates()
        if constraint_df is not None:
            constraint_file = self.data_dir / "constraint_analysis.csv"
            constraint_df.to_csv(constraint_file, index=False)
            print(f"   Constraint analysis saved to: {constraint_file}")

        print(f"✅ Combined {len(self.experiments)} experiments")
        print(f"   Total requests: {len(self.combined_data)}")
        print(
            f"   Generation lengths: {sorted(self.combined_data['generation_length'].unique())}"
        )
        print(f"   Combined data saved to: {combined_file}")
        print(f"   Summary data saved to: {summary_file}")

    def _calculate_experiment_summary(self, exp: ExperimentData) -> List[Dict[str, Any]]:
        """Per-experiment, per-generation_length summary stats."""
        df = exp.data
        out: List[Dict[str, Any]] = []

        for gl in sorted(df["generation_length"].unique()):
            sub = df[df["generation_length"] == gl]
            row: Dict[str, Any] = {
                "experiment_name": exp.name,
                "experiment_type": exp.type,
                "generation_length": gl,
                "num_requests": len(sub),
                "source_path": str(exp.source_path),
            }
            metrics = [
                "round_trip_time",
                "server_processing_duration",
                "inference_duration",
                "preprocess_duration",
                "postprocess_duration",
                "total_network_latency",
            ]
            for metric in metrics:
                if metric in sub.columns:
                    s = sub[metric]
                    row.update(
                        {
                            f"{metric}_mean": s.mean(),
                            f"{metric}_std": s.std(),
                            f"{metric}_min": s.min(),
                            f"{metric}_max": s.max(),
                            f"{metric}_median": s.median(),
                            f"{metric}_p95": s.quantile(0.95),
                            f"{metric}_p99": s.quantile(0.99),
                        }
                    )
            if "num_generated_notes" in sub.columns:
                notes = sub["num_generated_notes"]
                row.update(
                    {
                        "num_generated_notes_mean": notes.mean(),
                        "num_generated_notes_std": notes.std(),
                    }
                )
            out.append(row)
        return out

    def _calculate_constraint_satisfaction_rates(self) -> Optional[pd.DataFrame]:
        """
        Simple constraint analysis: whether mean RTT fits within musical time for
        each experiment / GL, similar in spirit to bench_results.
        """
        if self.summary_data is None:
            return None

        df = self.summary_data.copy()
        if "round_trip_time_mean" not in df.columns:
            return None

        # Only consider odd generation lengths (accompaniment frames) and convert to ticks.
        odd = df[df["generation_length"] % 2 == 1].copy()
        if odd.empty:
            return None
        odd["generation_ticks"] = (odd["generation_length"] + 1) / 2

        rows: List[Dict[str, Any]] = []
        for _, row in odd.iterrows():
            gl_ticks = row["generation_ticks"]
            rtt_ms = row["round_trip_time_mean"] * 1000
            musical_deadline_ms = gl_ticks * 125
            # Simple one-tick safety margin check
            both_satisfied = rtt_ms <= (musical_deadline_ms - 125)
            rows.append(
                {
                    "experiment_name": row["experiment_name"],
                    "generation_ticks": gl_ticks,
                    "generation_length": row["generation_length"],
                    "round_trip_time_mean_ms": rtt_ms,
                    "musical_deadline_ms": musical_deadline_ms,
                    "both_satisfied": both_satisfied,
                }
            )
        if not rows:
            return None
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Comparative plotting
    # -------------------------------------------------------------------------

    def generate_comparative_analysis(self) -> None:
        """Generate comparative visualizations across all experiments."""
        print("\n📊 Generating comparative analysis...")
        if self.summary_data is None:
            print("❌ No summary data available")
            return

        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("husl")

        self._plot_experiment_comparison()
        self._plot_generation_length_trends()
        self._plot_performance_comparison()
        self._plot_variability_comparison()
        self._plot_constraint_analysis()

        print(f"✅ Comparative analysis plots saved to {self.plots_dir}")

    def _plot_experiment_comparison(self) -> None:
        """Compare latency across experiments for each generation length."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        experiments = self.summary_data["experiment_name"].unique()

        for exp in experiments:
            exp_data = self.summary_data[self.summary_data["experiment_name"] == exp]
            if "round_trip_time_mean" in exp_data.columns:
                ax1.plot(
                    exp_data["generation_length"],
                    exp_data["round_trip_time_mean"] * 1000,
                    marker="o",
                    linewidth=2.5,
                    markersize=8,
                    label=exp,
                )
        ax1.set_xlabel("Generation Length (Frames)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Round Trip Time (ms)", fontsize=14, fontweight="bold")
        ax1.set_title("Round Trip Time Comparison Across Experiments", fontsize=16, fontweight="bold")
        ax1.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        for exp in experiments:
            exp_data = self.summary_data[self.summary_data["experiment_name"] == exp]
            if "round_trip_time_p95" in exp_data.columns:
                ax2.plot(
                    exp_data["generation_length"],
                    exp_data["round_trip_time_p95"] * 1000,
                    marker="s",
                    linewidth=2.5,
                    markersize=8,
                    label=exp,
                    linestyle="--",
                )
        ax2.set_xlabel("Generation Length (Frames)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("95th Percentile RTT (ms)", fontsize=14, fontweight="bold")
        ax2.set_title("95th Percentile Performance Comparison", fontsize=16, fontweight="bold")
        ax2.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "experiment_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_generation_length_trends(self) -> None:
        """Show how different experiments scale with generation length."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        metrics = [
            ("round_trip_time_mean", "Mean Round Trip Time (ms)", 1000),
            ("inference_duration_mean", "Mean Inference Duration (ms)", 1000),
            ("server_processing_duration_mean", "Mean Server Processing (ms)", 1000),
            ("total_network_latency_mean", "Mean Network Latency (ms)", 1000),
        ]

        experiments = self.summary_data["experiment_name"].unique()
        colors = sns.color_palette("husl", len(experiments))

        for i, (metric, title, scale) in enumerate(metrics):
            if metric not in self.summary_data.columns:
                axes[i].text(
                    0.5,
                    0.5,
                    f"No {metric} data available",
                    transform=axes[i].transAxes,
                    ha="center",
                    va="center",
                )
                axes[i].set_title(title, fontsize=14, fontweight="bold")
                continue

            for j, exp in enumerate(experiments):
                exp_data = self.summary_data[self.summary_data["experiment_name"] == exp]
                if exp_data.empty:
                    continue
                axes[i].plot(
                    exp_data["generation_length"],
                    exp_data[metric] * scale,
                    marker="o",
                    linewidth=2.5,
                    markersize=8,
                    label=exp,
                    color=colors[j],
                )
                if len(exp_data) > 1:
                    x = exp_data["generation_length"]
                    y = exp_data[metric] * scale
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    axes[i].plot(x, p(x), "--", alpha=0.7, color=colors[j], linewidth=1.5)

            axes[i].set_xlabel("Generation Length (Frames)", fontsize=12, fontweight="bold")
            axes[i].set_ylabel(title.split("(")[0].strip(), fontsize=12, fontweight="bold")
            axes[i].set_title(title, fontsize=14, fontweight="bold")
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "generation_length_trends.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_performance_comparison(self) -> None:
        """Bar-chart comparison of key metrics across experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        key_gen_lengths = [1, 3, 5, 7, 9]
        available = sorted(self.summary_data["generation_length"].unique())
        comparison_lengths = [gl for gl in key_gen_lengths if gl in available] or available[:5]

        metrics = [
            ("round_trip_time_mean", "Mean Round Trip Time (ms)", 1000),
            ("inference_duration_mean", "Mean Inference Duration (ms)", 1000),
            ("round_trip_time_std", "Round Trip Time Std Dev (ms)", 1000),
            ("round_trip_time_p95", "95th Percentile RTT (ms)", 1000),
        ]

        experiments = self.summary_data["experiment_name"].unique()
        x_pos = np.arange(len(comparison_lengths))
        width = 0.8 / max(1, len(experiments))

        for i, (metric, title, scale) in enumerate(metrics):
            if metric not in self.summary_data.columns:
                axes[i].text(
                    0.5,
                    0.5,
                    f"No {metric} data available",
                    transform=axes[i].transAxes,
                    ha="center",
                    va="center",
                )
                axes[i].set_title(title, fontsize=14, fontweight="bold")
                continue

            for j, exp in enumerate(experiments):
                values = []
                for gl in comparison_lengths:
                    sub = self.summary_data[
                        (self.summary_data["experiment_name"] == exp)
                        & (self.summary_data["generation_length"] == gl)
                    ]
                    if not sub.empty:
                        values.append(sub[metric].iloc[0] * scale)
                    else:
                        values.append(0)
                axes[i].bar(x_pos + j * width, values, width, label=exp, alpha=0.8)

            axes[i].set_xlabel("Generation Length (Frames)", fontsize=12, fontweight="bold")
            axes[i].set_ylabel(title.split("(")[0].strip(), fontsize=12, fontweight="bold")
            axes[i].set_title(title, fontsize=14, fontweight="bold")
            axes[i].set_xticks(x_pos + width * (len(experiments) - 1) / 2)
            axes[i].set_xticklabels(comparison_lengths)
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_variability_comparison(self) -> None:
        """Compare RTT variability across experiments."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        experiments = self.summary_data["experiment_name"].unique()

        for exp in experiments:
            exp_data = self.summary_data[self.summary_data["experiment_name"] == exp]
            if "round_trip_time_std" in exp_data.columns:
                ax1.plot(
                    exp_data["generation_length"],
                    exp_data["round_trip_time_std"] * 1000,
                    marker="o",
                    linewidth=2.5,
                    markersize=8,
                    label=exp,
                )
        ax1.set_xlabel("Generation Length (Frames)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Round Trip Time Std Dev (ms)", fontsize=14, fontweight="bold")
        ax1.set_title("Latency Variability Comparison", fontsize=16, fontweight="bold")
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        for exp in experiments:
            exp_data = self.summary_data[self.summary_data["experiment_name"] == exp]
            if "round_trip_time_mean" in exp_data.columns and "round_trip_time_std" in exp_data.columns:
                cv = (exp_data["round_trip_time_std"] / exp_data["round_trip_time_mean"]) * 100
                ax2.plot(
                    exp_data["generation_length"],
                    cv,
                    marker="s",
                    linewidth=2.5,
                    markersize=8,
                    label=exp,
                )
        ax2.set_xlabel("Generation Length (Frames)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Coefficient of Variation (%)", fontsize=14, fontweight="bold")
        ax2.set_title("Relative Variability Comparison", fontsize=16, fontweight="bold")
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "variability_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_constraint_analysis(self) -> None:
        """Overlay mean/95th RTT vs musical deadline in tick space."""
        if self.summary_data is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        odd = self.summary_data[self.summary_data["generation_length"] % 2 == 1].copy()
        if odd.empty:
            print("⚠️ No odd generation lengths found for constraint analysis")
            return
        odd["generation_ticks"] = (odd["generation_length"] + 1) / 2
        experiments = odd["experiment_name"].unique()

        for exp in experiments:
            exp_data = odd[odd["experiment_name"] == exp]
            if "round_trip_time_mean" in exp_data.columns:
                x = exp_data["generation_ticks"]
                y = exp_data["round_trip_time_mean"] * 1000
                ax1.plot(x, y, "o-", linewidth=2.5, markersize=8, label=exp)

        tick_values = sorted(odd["generation_ticks"].unique())
        if tick_values:
            musical_y = [t * 125 for t in tick_values]
            buffer_minus1_y = [max(0, t * 125 - 125) for t in tick_values]
            ax1.plot(tick_values, musical_y, "r-", linewidth=3, alpha=0.8, label="Musical Time Deadline", zorder=5)
            ax1.plot(tick_values, buffer_minus1_y, "purple", linewidth=3, alpha=0.8, label="1 Useable Tick", zorder=5)

        ax1.set_xlabel("Generation Length (Ticks)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Round Trip Time (ms)", fontsize=14, fontweight="bold")
        ax1.set_title("Real-Time Constraint Analysis (Mean)", fontsize=16, fontweight="bold")
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        for exp in experiments:
            exp_data = odd[odd["experiment_name"] == exp]
            if "round_trip_time_p95" in exp_data.columns:
                x = exp_data["generation_ticks"]
                y = exp_data["round_trip_time_p95"] * 1000
                ax2.plot(x, y, "s--", linewidth=2.5, markersize=8, label=exp)

        if tick_values:
            musical_y = [t * 125 for t in tick_values]
            buffer_minus1_y = [max(0, t * 125 - 125) for t in tick_values]
            ax2.plot(tick_values, musical_y, "r-", linewidth=3, alpha=0.8, label="Musical Time Deadline", zorder=5)
            ax2.plot(tick_values, buffer_minus1_y, "purple", linewidth=3, alpha=0.8, label="1 Useable Tick", zorder=5)

        ax2.set_xlabel("Generation Length (Ticks)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Round Trip Time (ms)", fontsize=14, fontweight="bold")
        ax2.set_title("Real-Time Constraint Analysis (95th Percentile)", fontsize=16, fontweight="bold")
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "constraint_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()


def run_bulk_analysis(
    data_sources: List[Path],
    output_dir: Path,
    anomaly_filter_pct: float = 0.0,
) -> int:
    """Top-level entry to run bulk benchmark analysis."""
    analyzer = BulkGenerationLengthAnalyzer(
        data_sources=data_sources,
        output_dir=output_dir,
        anomaly_filter_pct=anomaly_filter_pct,
    )
    if not analyzer.load_all_experiments():
        return 1
    analyzer.combine_all_data()
    analyzer.generate_comparative_analysis()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bulk analysis for StreamMUSE generation-length benchmarks (eval).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze multiple experiment directories and write to results/benchmarks/bulk_analysis
  %(prog)s experiments/exp1 experiments/exp2 --output_dir results/benchmarks/bulk_analysis
""",
    )
    parser.add_argument(
        "data_sources",
        nargs="+",
        type=Path,
        help="One or more experiment directories (manual benchmark dirs or bulk_benchmark outputs).",
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
    return run_bulk_analysis(args.data_sources, args.output_dir, anomaly_filter_pct=args.anomaly_filter)


if __name__ == "__main__":
    raise SystemExit(main())

