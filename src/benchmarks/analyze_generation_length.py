#!/usr/bin/env python3
"""
Analysis and visualization script for generation length benchmark results.

Ported from the `bench_results` branch to ensure plot styling and filenames match.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


class GenerationLengthAnalyzer:
    """
    Analyzes benchmark results to understand generation length effects on latency.
    """

    def __init__(
        self,
        data_source: str,
        output_dir: str = "analysis_results",
        anomaly_filter_pct: float = 0.0,
    ):
        self.data_source = Path(data_source)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.detailed_data: Optional[pd.DataFrame] = None
        self.summary_data: Optional[pd.DataFrame] = None
        self.anomaly_filter_pct = anomaly_filter_pct

    def load_data(self) -> bool:
        """Load data from various possible sources."""

        if self.data_source.is_file() and self.data_source.suffix == ".csv":
            # Single CSV file
            return self._load_single_csv()
        if self.data_source.is_dir():
            # Directory with multiple files
            return self._load_directory()
        print(f"❌ Invalid data source: {self.data_source}")
        return False

    def _load_single_csv(self) -> bool:
        """Load data from a single CSV file with generation_length column."""
        try:
            df = pd.read_csv(self.data_source)
            if "generation_length" not in df.columns:
                print("❌ CSV file must contain 'generation_length' column")
                return False

            # Apply anomaly filtering
            self.detailed_data = self._apply_anomaly_filter(df)
            self._calculate_summary_from_detailed()
            print(f"✅ Loaded data from {self.data_source}")
            return True

        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False

    def _load_directory(self) -> bool:
        """Load data from directory structure (manual or automated results)."""

        # Try to load from generation_length_benchmark.py output structure
        analysis_dir = self.data_source / "analysis"
        if analysis_dir.exists():
            return self._load_automated_results()

        # Try to load from manual benchmark results
        return self._load_manual_results()

    def _load_automated_results(self) -> bool:
        """Load from automated benchmark output."""
        analysis_dir = self.data_source / "analysis"

        # Load detailed results
        detailed_file = analysis_dir / "detailed_results_all_generation_lengths.csv"
        if detailed_file.exists():
            raw_data = pd.read_csv(detailed_file)
            self.detailed_data = self._apply_anomaly_filter(raw_data)

        # Load summary statistics
        summary_file = analysis_dir / "summary_statistics.csv"
        if summary_file.exists():
            self.summary_data = pd.read_csv(summary_file)

        if self.detailed_data is not None or self.summary_data is not None:
            print(f"✅ Loaded automated benchmark results from {self.data_source}")
            return True

        print(f"❌ No automated results found in {self.data_source}")
        return False

    def _load_manual_results(self) -> bool:
        """Load from manual benchmark CSV files."""

        # Look for CSV files matching patterns like gen_length_XX.csv
        csv_files = list(self.data_source.glob("*.csv"))
        csv_files.extend(list(self.data_source.glob("**/*.csv")))

        if not csv_files:
            print(f"❌ No CSV files found in {self.data_source}")
            return False

        all_data = []

        for csv_file in csv_files:
            # Try to extract generation length from filename
            gen_length = self._extract_generation_length_from_filename(csv_file.name)

            if gen_length is None:
                print(f"⚠️  Skipping {csv_file.name} - cannot extract generation length")
                continue

            try:
                df = pd.read_csv(csv_file)
                df["generation_length"] = gen_length
                df["source_file"] = csv_file.name
                all_data.append(df)
                print(f"✅ Loaded {csv_file.name} (generation length: {gen_length})")

            except Exception as e:
                print(f"⚠️  Error loading {csv_file.name}: {e}")
                continue

        if not all_data:
            print("❌ No valid data files found")
            return False

        combined_data = pd.concat(all_data, ignore_index=True)

        # Apply anomaly filtering
        self.detailed_data = self._apply_anomaly_filter(combined_data)
        self._calculate_summary_from_detailed()

        print(f"✅ Loaded {len(all_data)} files with {len(self.detailed_data)} total requests")
        return True

    def _extract_generation_length_from_filename(self, filename: str) -> Optional[int]:
        """Extract generation length from various filename patterns."""

        patterns = [
            r"gen_length_(\d+)",
            r"gen_(\d+)",
            r"generation_(\d+)",
            r"gl_(\d+)",
            r"(\d+)_frames",
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _apply_anomaly_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out anomalies based on round trip time percentiles per generation length.
        """
        if self.anomaly_filter_pct <= 0 or "round_trip_time" not in df.columns:
            return df

        if "generation_length" not in df.columns:
            print("⚠️ No generation_length column found, skipping anomaly filtering")
            return df

        original_count = len(df)
        filtered_data = []

        print(
            f"🔍 Anomaly filtering ({self.anomaly_filter_pct}% each tail per generation length):"
        )

        for gen_length in sorted(df["generation_length"].unique()):
            gl_data = df[df["generation_length"] == gen_length]
            gl_original_count = len(gl_data)

            if gl_original_count < 10:  # Skip filtering if too few samples
                print(f"   GL {gen_length}: {gl_original_count} samples (too few, no filtering)")
                filtered_data.append(gl_data)
                continue

            lower_threshold = gl_data["round_trip_time"].quantile(
                self.anomaly_filter_pct / 100
            )
            upper_threshold = gl_data["round_trip_time"].quantile(
                1 - self.anomaly_filter_pct / 100
            )

            gl_filtered = gl_data[
                (gl_data["round_trip_time"] >= lower_threshold)
                & (gl_data["round_trip_time"] <= upper_threshold)
            ].copy()

            gl_filtered_count = len(gl_filtered)
            gl_removed_count = gl_original_count - gl_filtered_count

            print(
                f"   GL {gen_length}: {gl_original_count} → {gl_filtered_count} samples "
                f"(removed {gl_removed_count}, {gl_removed_count/gl_original_count*100:.1f}%)"
            )
            print(
                f"      RTT range: {lower_threshold*1000:.1f}ms - {upper_threshold*1000:.1f}ms"
            )

            filtered_data.append(gl_filtered)

        if not filtered_data:
            return df

        filtered_df = pd.concat(filtered_data, ignore_index=True)
        filtered_count = len(filtered_df)
        removed_count = original_count - filtered_count

        print(
            f"   Total: {original_count} → {filtered_count} samples "
            f"(removed {removed_count}, {removed_count/original_count*100:.1f}%)"
        )

        return filtered_df

    def _calculate_summary_from_detailed(self):
        """Calculate summary statistics from detailed data."""
        if self.detailed_data is None:
            return

        summary_list = []

        for gen_length in sorted(self.detailed_data["generation_length"].unique()):
            df_subset = self.detailed_data[
                self.detailed_data["generation_length"] == gen_length
            ]

            def safe_stats(series):
                if len(series) == 0:
                    return {
                        "mean": 0,
                        "std": 0,
                        "min": 0,
                        "max": 0,
                        "median": 0,
                        "p95": 0,
                        "p99": 0,
                        "upper_bound_60": 0,
                        "upper_bound_70": 0,
                        "upper_bound_80": 0,
                        "upper_bound_85": 0,
                        "upper_bound_90": 0,
                        "upper_bound_95": 0,
                        "upper_bound_98": 0,
                        "upper_bound_99": 0,
                        "upper_bound_99_5": 0,
                        "upper_bound_99_9": 0,
                    }

                upper_bounds = {}
                if len(series) >= 5:
                    try:
                        percentiles = [60, 70, 80, 85, 90, 95, 98, 99, 99.5, 99.9, 100]
                        for percentile in percentiles:
                            if percentile == 100:
                                value = series.max()
                                key_name = "upper_bound_100"
                            else:
                                value = series.quantile(percentile / 100.0)
                                key_name = (
                                    f"upper_bound_{int(percentile)}"
                                    if percentile == int(percentile)
                                    else f"upper_bound_{percentile}".replace(".", "_")
                                )
                            upper_bounds[key_name] = value
                    except Exception as e:
                        print(f"Warning: Error calculating percentiles: {e}")
                        for p in [60, 70, 80, 85, 90, 95, 98, 99, 99.5, 99.9, 100]:
                            key_name = (
                                f"upper_bound_{int(p)}"
                                if p == int(p)
                                else f"upper_bound_{p}".replace(".", "_")
                            )
                            upper_bounds[key_name] = 0
                else:
                    for p in [60, 70, 80, 85, 90, 95, 98, 99, 99.5, 99.9, 100]:
                        key_name = (
                            f"upper_bound_{int(p)}"
                            if p == int(p)
                            else f"upper_bound_{p}".replace(".", "_")
                        )
                        upper_bounds[key_name] = 0

                base_stats = {
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                    "median": series.median(),
                    "p95": series.quantile(0.95),
                    "p99": series.quantile(0.99),
                }
                base_stats.update(upper_bounds)
                return base_stats

            summary = {"generation_length": gen_length, "num_requests": len(df_subset)}

            metrics = [
                "round_trip_time",
                "server_processing_duration",
                "inference_duration",
                "preprocess_duration",
                "postprocess_duration",
                "total_network_latency",
            ]

            for metric in metrics:
                if metric in df_subset.columns:
                    s = safe_stats(df_subset[metric])
                    for stat_name, value in s.items():
                        summary[f"{metric}_{stat_name}"] = value

            if "num_generated_notes" in df_subset.columns:
                notes_stats = safe_stats(df_subset["num_generated_notes"])
                for stat_name, value in notes_stats.items():
                    summary[f"num_generated_notes_{stat_name}"] = value

            summary_list.append(summary)

        self.summary_data = pd.DataFrame(summary_list)

    def _set_fine_x_ticks(self, ax):
        """Set finer x-axis ticks for better readability."""
        if self.summary_data is None:
            return

        gen_lengths = sorted(self.summary_data["generation_length"].unique())
        if len(gen_lengths) > 1:
            min_gap = min(
                gen_lengths[i + 1] - gen_lengths[i] for i in range(len(gen_lengths) - 1)
            )
            tick_step = max(1, min_gap // 2)
            x_min, x_max = min(gen_lengths), max(gen_lengths)
            x_ticks = np.arange(x_min, x_max + tick_step, tick_step)
            ax.set_xticks(x_ticks)
            ax.set_xlim(x_min - tick_step, x_max + tick_step)

    def _prepare_tick_based_data(self):
        """Filter data to odd generation lengths and convert to tick-based X-axis."""
        if self.summary_data is None:
            return None

        odd_data = self.summary_data[
            self.summary_data["generation_length"] % 2 == 1
        ].copy()
        if len(odd_data) == 0:
            print("⚠️ No odd generation lengths found in data")
            return None
        odd_data["generation_ticks"] = (odd_data["generation_length"] + 1) / 2
        return odd_data

    def generate_all_visualizations(self):
        """Generate comprehensive visualization suite."""
        print("📊 Generating visualizations...")

        if self.summary_data is None:
            print("❌ No summary data available for visualization")
            return

        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("husl")

        self._plot_latency_vs_generation_length()
        self._plot_variability_analysis()
        self._plot_distribution_comparison()
        self._plot_component_breakdown()
        self._plot_performance_metrics()
        self._plot_confidence_intervals()

        if self.detailed_data is not None:
            self._plot_detailed_distributions()
            self._plot_stacked_distributions()
            self._plot_correlation_analysis()
            self._plot_parameter_constraint_analysis()

        print(f"✅ Visualizations saved to {self.plots_dir}")

    # ---- Plot functions ported from bench_results ----
    def _plot_latency_vs_generation_length(self):
        """Primary relationship plot: latency vs generation ticks (accompaniment focus)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        tick_data = self._prepare_tick_based_data()
        if tick_data is None:
            ax.text(
                0.5,
                0.5,
                "No odd generation lengths available for tick-based analysis",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title("Server Latency vs Generation Ticks", fontsize=16, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "latency_vs_generation_length.png", dpi=300, bbox_inches="tight")
            plt.close()
            return

        metrics = [
            ("round_trip_time", "Round Trip Time", "o-", "#1f77b4"),
            ("server_processing_duration", "Server Processing", "s--", "#ff7f0e"),
            ("inference_duration", "Inference Time", "^:", "#2ca02c"),
            ("total_network_latency", "Network Latency", "d-.", "#d62728"),
        ]
        for metric, label, style, color in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in tick_data.columns:
                ax.errorbar(
                    tick_data["generation_ticks"],
                    tick_data[mean_col] * 1000,
                    yerr=tick_data[std_col] * 1000 if std_col in tick_data.columns else None,
                    label=label,
                    marker=style[0],
                    linestyle=style[1:],
                    color=color,
                    capsize=5,
                    capthick=2,
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8,
                )

        ax.set_xlabel("Generation Length (Ticks)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Latency (milliseconds)", fontsize=14, fontweight="bold")
        ax.set_title(
            "Server Latency vs Generation Ticks\n(Accompaniment Frame Focus)",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        tick_values = sorted(tick_data["generation_ticks"].unique())
        ax.set_xticks(tick_values)
        ax.set_xlim(min(tick_values) - 0.5, max(tick_values) + 0.5)

        if tick_values:
            frontier_x = tick_values
            musical_time_y = [tick * 125 for tick in frontier_x]
            buffer_minus1_y = [max(0, tick * 125 - 125) for tick in frontier_x]
            buffer_minus2_y = [max(0, tick * 125 - 250) for tick in frontier_x]
            ax.plot(
                frontier_x,
                musical_time_y,
                color="red",
                linewidth=2,
                linestyle="-",
                alpha=0.8,
                label="Musical Time Deadline",
                zorder=5,
            )
            ax.plot(
                frontier_x,
                buffer_minus1_y,
                color="purple",
                linewidth=2,
                linestyle="-",
                alpha=0.8,
                label="1 Useable Tick",
                zorder=5,
            )
            ax.plot(
                frontier_x,
                buffer_minus2_y,
                color="darkblue",
                linewidth=2,
                linestyle="-",
                alpha=0.8,
                label="2 Useable Ticks",
                zorder=5,
            )

        if "round_trip_time_mean" in tick_data.columns:
            x_ticks = tick_data["generation_ticks"]
            y = tick_data["round_trip_time_mean"] * 1000
            z = np.polyfit(x_ticks, y, 1)
            p = np.poly1d(z)
            ax.plot(
                x_ticks,
                p(x_ticks),
                "gray",
                linestyle=":",
                alpha=0.7,
                linewidth=1,
                label=f"Trend: {z[0]:.2f}ms/tick",
            )

        ax.legend(fontsize=11, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "latency_vs_generation_length.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_variability_analysis(self):
        """Plot showing how latency variability changes with generation length."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        metrics = [
            ("round_trip_time", "Round Trip Time", "#1f77b4"),
            ("server_processing_duration", "Server Processing", "#ff7f0e"),
            ("inference_duration", "Inference Time", "#2ca02c"),
        ]
        for metric, label, color in metrics:
            std_col = f"{metric}_std"
            if self.summary_data is not None and std_col in self.summary_data.columns:
                ax1.plot(
                    self.summary_data["generation_length"],
                    self.summary_data[std_col] * 1000,
                    marker="o",
                    linewidth=2.5,
                    markersize=8,
                    label=label,
                    color=color,
                )
        ax1.set_xlabel("Generation Length (Frames)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Standard Deviation (ms)", fontsize=12, fontweight="bold")
        ax1.set_title("Latency Variability vs Generation Length", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        self._set_fine_x_ticks(ax1)

        for metric, label, color in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if (
                self.summary_data is not None
                and mean_col in self.summary_data.columns
                and std_col in self.summary_data.columns
            ):
                cv = self.summary_data[std_col] / self.summary_data[mean_col] * 100
                ax2.plot(
                    self.summary_data["generation_length"],
                    cv,
                    marker="s",
                    linewidth=2.5,
                    markersize=8,
                    label=label,
                    color=color,
                )
        ax2.set_xlabel("Generation Length (Frames)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Coefficient of Variation (%)", fontsize=12, fontweight="bold")
        ax2.set_title("Relative Variability vs Generation Length", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        self._set_fine_x_ticks(ax2)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "variability_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_distribution_comparison(self):
        """Box plots comparing distributions across generation lengths."""
        if self.detailed_data is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        metrics = [
            ("round_trip_time", "Round Trip Time (ms)", 1000),
            ("inference_duration", "Inference Duration (ms)", 1000),
            ("server_processing_duration", "Server Processing (ms)", 1000),
            ("total_network_latency", "Network Latency (ms)", 1000),
        ]
        for i, (metric, title, scale) in enumerate(metrics):
            if metric in self.detailed_data.columns:
                data_for_plot = []
                labels = []
                for gen_length in sorted(self.detailed_data["generation_length"].unique()):
                    subset = self.detailed_data[self.detailed_data["generation_length"] == gen_length]
                    data_for_plot.append(subset[metric] * scale)
                    labels.append(f"{gen_length}")
                bp = axes[i].boxplot(data_for_plot, labels=labels, patch_artist=True)
                colors = sns.color_palette("husl", len(data_for_plot))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                axes[i].set_title(title, fontsize=12, fontweight="bold")
                axes[i].set_xlabel("Generation Length (Frames)", fontsize=10)
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "distribution_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_component_breakdown(self):
        """Stacked bar chart showing latency component breakdown."""
        fig, ax = plt.subplots(figsize=(14, 8))
        if self.summary_data is None:
            return
        components = [
            ("preprocess_duration", "Preprocessing", "#ff9999"),
            ("inference_duration", "Inference", "#66b3ff"),
            ("postprocess_duration", "Postprocessing", "#99ff99"),
            ("total_network_latency", "Network", "#ffcc99"),
        ]
        x = self.summary_data["generation_length"]
        bottom = np.zeros(len(x))
        for component, label, color in components:
            mean_col = f"{component}_mean"
            if mean_col in self.summary_data.columns:
                values = self.summary_data[mean_col] * 1000
                ax.bar(x, values, bottom=bottom, label=label, color=color, alpha=0.8)
                bottom += values
        ax.set_xlabel("Generation Length (Frames)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Latency (milliseconds)", fontsize=14, fontweight="bold")
        ax.set_title(
            "Latency Component Breakdown by Generation Length",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "component_breakdown.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_performance_metrics(self):
        """Performance efficiency and scaling metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        if self.summary_data is None:
            return
        if "round_trip_time_mean" in self.summary_data.columns:
            throughput = 1.0 / self.summary_data["round_trip_time_mean"]
            ax1.plot(
                self.summary_data["generation_length"],
                throughput,
                "o-",
                linewidth=3,
                markersize=10,
                color="#1f77b4",
            )
            ax1.set_xlabel("Generation Length (Frames)", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Estimated Throughput (req/sec)", fontsize=12, fontweight="bold")
            ax1.set_title("Request Throughput vs Generation Length", fontsize=14, fontweight="bold")
            ax1.grid(True, alpha=0.3)
            self._set_fine_x_ticks(ax1)

        if "num_generated_notes_mean" in self.summary_data.columns and "inference_duration_mean" in self.summary_data.columns:
            efficiency = self.summary_data["num_generated_notes_mean"] / self.summary_data["inference_duration_mean"]
            ax2.plot(
                self.summary_data["generation_length"],
                efficiency,
                "s-",
                linewidth=3,
                markersize=10,
                color="#ff7f0e",
            )
            ax2.set_xlabel("Generation Length (Frames)", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Notes Generated per Second", fontsize=12, fontweight="bold")
            ax2.set_title("Generation Efficiency vs Generation Length", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            self._set_fine_x_ticks(ax2)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_confidence_intervals(self):
        """Plot upper bound percentiles for round trip time (tick-based)."""
        fig, ax = plt.subplots(figsize=(14, 8))
        tick_data = self._prepare_tick_based_data()
        if tick_data is None:
            ax.text(
                0.5,
                0.5,
                "No odd generation lengths available for tick-based analysis",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title("Round Trip Time Upper Bounds", fontsize=16, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "confidence_intervals.png", dpi=300, bbox_inches="tight")
            plt.close()
            return

        metric = "round_trip_time"
        scale = 1000
        mean_col = f"{metric}_mean"
        if mean_col not in tick_data.columns:
            ax.text(0.5, 0.5, "No round trip time data available", transform=ax.transAxes, ha="center", va="center")
            ax.set_title("Round Trip Time Upper Bounds", fontsize=16, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "confidence_intervals.png", dpi=300, bbox_inches="tight")
            plt.close()
            return

        x = tick_data["generation_ticks"]
        y_mean = tick_data[mean_col] * scale
        ax.plot(x, y_mean, "ko-", linewidth=3, markersize=8, label="Mean", zorder=10)

        percentiles = ["60", "70", "80", "85", "90", "95", "98", "99", "99_5", "99_9", "100"]
        colors = ["#f0f8ff", "#e0f0ff", "#d0e8ff", "#c0e0ff", "#b0d8ff", "#90c8ff", "#70b8ff", "#50a8ff", "#3098ff", "#1088ff", "#000080"]
        for percentile, color in zip(percentiles, colors):
            upper_col = f"{metric}_upper_bound_{percentile}"
            if upper_col in tick_data.columns:
                y_upper = tick_data[upper_col] * scale
                if y_upper.sum() > 0:
                    display_level = percentile.replace("_", ".")
                    ax.plot(
                        x,
                        y_upper,
                        "--",
                        color=color,
                        linewidth=2,
                        markersize=6,
                        marker="o",
                        label=f"{display_level}% upper bound",
                        alpha=0.8,
                    )

        ax.set_xlabel("Generation Length (Ticks)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Round Trip Time (milliseconds)", fontsize=14, fontweight="bold")
        ax.set_title(
            "Round Trip Time Upper Bounds\n(X% of requests complete within this time - Accompaniment Focus)",
            fontsize=16,
            fontweight="bold",
        )

        tick_values = sorted(tick_data["generation_ticks"].unique())
        musical_time_y = [tick * 125 for tick in tick_values]
        buffer_minus1_y = [max(0, tick * 125 - 125) for tick in tick_values]
        buffer_minus2_y = [max(0, tick * 125 - 250) for tick in tick_values]
        ax.plot(tick_values, musical_time_y, color="red", linewidth=3, linestyle="-", alpha=0.9, label="Musical Time Deadline", zorder=5)
        ax.plot(tick_values, buffer_minus1_y, color="purple", linewidth=3, linestyle="-", alpha=0.9, label="1 Useable Tick", zorder=5)
        ax.plot(tick_values, buffer_minus2_y, color="darkblue", linewidth=3, linestyle="-", alpha=0.9, label="2 Useable Ticks", zorder=5)

        handles, labels = ax.get_legend_handles_labels()
        percentile_handles = []
        percentile_labels = []
        frontier_handles = []
        frontier_labels = []
        for handle, label in zip(handles, labels):
            if any(frontier_name in label for frontier_name in ["Musical Time Deadline", "Useable Tick"]):
                frontier_handles.append(handle)
                frontier_labels.append(label)
            else:
                percentile_handles.append(handle)
                percentile_labels.append(label)

        if len(percentile_handles) > 6:
            legend1 = ax.legend(percentile_handles[:6], percentile_labels[:6], loc="upper left", fontsize=10, framealpha=0.9, title="Lower Percentiles")
            ax.add_artist(legend1)
            legend2 = ax.legend(percentile_handles[6:], percentile_labels[6:], loc="upper right", fontsize=10, framealpha=0.9, title="Higher Percentiles")
            ax.add_artist(legend2)
        else:
            if percentile_handles:
                legend1 = ax.legend(percentile_handles, percentile_labels, loc="upper left", fontsize=10, framealpha=0.9, title="Percentiles")
                ax.add_artist(legend1)
        if frontier_handles:
            ax.legend(frontier_handles, frontier_labels, loc="lower right", fontsize=11, framealpha=0.9, title="Musical Time Constraints")

        ax.grid(True, alpha=0.3)
        self._set_fine_x_ticks(ax)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "confidence_intervals.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_detailed_distributions(self):
        """Detailed histogram grid for each generation length."""
        if self.detailed_data is None:
            return
        generation_lengths = sorted(self.detailed_data["generation_length"].unique())
        n_lengths = len(generation_lengths)
        cols = min(4, n_lengths)
        rows = (n_lengths + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, gen_length in enumerate(generation_lengths):
            if i >= len(axes):
                break
            ax = axes[i]
            data = self.detailed_data[self.detailed_data["generation_length"] == gen_length]
            rtt_data = data["round_trip_time"] * 1000
            bin_width = 5
            bins = np.arange(rtt_data.min(), rtt_data.max() + bin_width, bin_width)
            ax.hist(
                rtt_data,
                bins=bins,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                color=sns.color_palette("husl", n_lengths)[i],
            )
            ax.set_title(f"Generation Length: {gen_length} frames", fontsize=12, fontweight="bold")
            ax.set_xlabel("Round Trip Time (ms)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.grid(True, alpha=0.3)

            mean_rtt = data["round_trip_time"].mean() * 1000
            std_rtt = data["round_trip_time"].std() * 1000
            median_rtt = data["round_trip_time"].median() * 1000
            stats_text = f"μ = {mean_rtt:.1f}ms\nσ = {std_rtt:.1f}ms\nmedian = {median_rtt:.1f}ms"
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=9,
            )

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "detailed_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_stacked_distributions(self):
        """Plot all distributions as lines on a single plot."""
        if self.detailed_data is None:
            return
        generation_lengths = sorted(self.detailed_data["generation_length"].unique())
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = sns.color_palette("husl", len(generation_lengths))
        for i, gen_length in enumerate(generation_lengths):
            data = self.detailed_data[self.detailed_data["generation_length"] == gen_length]
            rtt_data = data["round_trip_time"] * 1000
            bin_width = 5
            bins = np.arange(rtt_data.min(), rtt_data.max() + bin_width, bin_width)
            counts, bins = np.histogram(rtt_data, bins=bins, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.plot(bin_centers, counts, label=f"{gen_length} frames", color=colors[i], linewidth=2.5, alpha=0.8)
            mean_rtt = rtt_data.mean()
            ax.axvline(mean_rtt, color=colors[i], linestyle="--", alpha=0.6, linewidth=1.5)

        ax.set_xlabel("Round Trip Time (ms)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Density", fontsize=14, fontweight="bold")
        ax.set_title(
            "Round Trip Time Distributions by Generation Length\n(Solid lines: distributions, Dashed lines: means)",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "stacked_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_correlation_analysis(self):
        """Correlation matrix and scatter plots."""
        if self.detailed_data is None:
            return

        numeric_cols = [
            "generation_length",
            "round_trip_time",
            "server_processing_duration",
            "inference_duration",
            "preprocess_duration",
            "postprocess_duration",
            "total_network_latency",
        ]
        numeric_cols = [col for col in numeric_cols if col in self.detailed_data.columns]
        if len(numeric_cols) < 3:
            return
        correlation_data = self.detailed_data[numeric_cols]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        corr_matrix = correlation_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, square=True, ax=ax1, cbar_kws={"shrink": 0.8})
        ax1.set_title("Correlation Matrix", fontsize=14, fontweight="bold")

        if "round_trip_time" in correlation_data.columns:
            scatter = ax2.scatter(
                correlation_data["generation_length"],
                correlation_data["round_trip_time"] * 1000,
                alpha=0.6,
                s=30,
                c=correlation_data["generation_length"],
                cmap="viridis",
            )
            x = correlation_data["generation_length"]
            y = correlation_data["round_trip_time"] * 1000
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax2.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
            ax2.set_xlabel("Generation Length (Frames)", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Round Trip Time (ms)", fontsize=12, fontweight="bold")
            ax2.set_title("Generation Length vs Round Trip Time", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax2, label="Generation Length")

        plt.tight_layout()
        plt.savefig(self.plots_dir / "correlation_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _calculate_percentiles_by_generation_length(self, percentiles=[50, 60, 70, 80, 90, 99.5]):
        """Calculate round trip time percentiles for each generation length separately."""
        if self.detailed_data is None:
            return {}
        percentile_data = {}
        for gen_length in sorted(self.detailed_data["generation_length"].unique()):
            gen_data = self.detailed_data[self.detailed_data["generation_length"] == gen_length]
            rtt_ms = gen_data["round_trip_time"] * 1000
            for p in percentiles:
                if len(rtt_ms) > 0:
                    percentile_value = np.percentile(rtt_ms, p)
                    percentile_data[(gen_length, p)] = percentile_value
                else:
                    percentile_data[(gen_length, p)] = np.inf
        return percentile_data

    def _evaluate_constraints_detailed(self, inference_interval, generation_length, round_trip_time_ms):
        """Evaluate parameter constraints with detailed breakdown."""
        TICK_DURATION_MS = 125
        constraint1_satisfied = inference_interval * TICK_DURATION_MS >= round_trip_time_ms
        musical_buffer_ms = (generation_length - inference_interval) * TICK_DURATION_MS
        constraint2_satisfied = round_trip_time_ms < musical_buffer_ms
        if constraint1_satisfied and constraint2_satisfied:
            constraint_status = 0
        elif not constraint1_satisfied and constraint2_satisfied:
            constraint_status = 1
        elif constraint1_satisfied and not constraint2_satisfied:
            constraint_status = 2
        else:
            constraint_status = 3
        return constraint1_satisfied, constraint2_satisfied, constraint_status

    def _create_constraint_matrix_detailed(self, percentile_data, percentile, inference_range, generation_range):
        """Create matrix showing detailed constraint status for parameter combinations."""
        matrix = np.full((len(inference_range), len(generation_range)), -1, dtype=int)
        for i, inference_interval in enumerate(inference_range):
            for j, generation_length_ticks in enumerate(generation_range):
                generation_length_frames = (generation_length_ticks * 2) - 1
                if (generation_length_frames, percentile) in percentile_data:
                    rtt_ms = percentile_data[(generation_length_frames, percentile)]
                    _, _, constraint_status = self._evaluate_constraints_detailed(inference_interval, generation_length_ticks, rtt_ms)
                    matrix[i, j] = constraint_status
                else:
                    matrix[i, j] = -1
        return matrix

    def _export_parameter_constraint_data(self, percentile_data, generation_range, inference_range):
        """Export parameter constraint analysis data to CSV."""
        constraint_data = []
        percentiles = [50, 60, 70, 80, 90, 99.5]
        for percentile in percentiles:
            for inference_interval in inference_range:
                for generation_ticks_val in generation_range:
                    gen_length_frames = (generation_ticks_val * 2) - 1
                    rtt_value = percentile_data.get((gen_length_frames, percentile), None)
                    if rtt_value is None:
                        status = -1
                    else:
                        _, _, status = self._evaluate_constraints_detailed(inference_interval, generation_ticks_val, rtt_value)
                    status_labels = {-1: "No Data", 0: "Valid", 1: "C1 Violated", 2: "C2 Violated", 3: "Both Violated"}
                    constraint_data.append(
                        {
                            "percentile": percentile,
                            "inference_interval_ticks": inference_interval,
                            "generation_length_ticks": generation_ticks_val,
                            "generation_length_frames": gen_length_frames,
                            "constraint_status_code": status,
                            "constraint_status": status_labels.get(status, "Unknown"),
                            "rtt_percentile_ms": rtt_value,
                            "constraint1_satisfied": status in [0, 2],
                            "constraint2_satisfied": status in [0, 1],
                            "both_constraints_satisfied": status == 0,
                        }
                    )
        if constraint_data:
            constraint_df = pd.DataFrame(constraint_data)
            constraint_file = self.output_dir / "parameter_constraint_analysis.csv"
            constraint_df.to_csv(constraint_file, index=False)
            print(f"📊 Parameter constraint analysis saved to: {constraint_file}")

    def _plot_parameter_constraint_analysis(self):
        """Generate parameter constraint analysis plots (I vs GL validity)."""
        if self.detailed_data is None:
            return
        percentile_data = self._calculate_percentiles_by_generation_length()
        if not percentile_data:
            return
        inference_range = np.arange(1, 5)
        generation_range = np.arange(1, 9)
        percentiles = [50, 60, 70, 80, 90, 99.5]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Parameter Constraint Analysis: Inference Interval vs Generation Length\n"
            + "Green=Valid, Orange=C1 Violated, Yellow=C2 Violated, Red=Both Violated",
            fontsize=20,
            fontweight="bold",
            y=0.95,
        )
        axes_flat = axes.flatten()
        constraint_colors = ["#d3d3d3", "#90ee90", "#ffa500", "#ffff80", "#ff6b6b"]
        from matplotlib.colors import ListedColormap

        constraint_cmap = ListedColormap(constraint_colors)
        for idx, percentile in enumerate(percentiles):
            ax = axes_flat[idx]
            detailed_matrix = self._create_constraint_matrix_detailed(percentile_data, percentile, inference_range, generation_range)
            sns.heatmap(
                detailed_matrix,
                ax=ax,
                xticklabels=generation_range,
                yticklabels=inference_range,
                cmap=constraint_cmap,
                vmin=-1,
                vmax=3,
                square=True,
                cbar=False,
                annot=False,
                linewidths=1,
                linecolor="white",
            )
            for i in range(len(inference_range)):
                for j in range(len(generation_range)):
                    status = detailed_matrix[i, j]
                    if status == -1:
                        text, color = "N/A", "black"
                    elif status == 0:
                        text, color = "✓", "darkgreen"
                    elif status == 1:
                        text, color = "C1", "darkorange"
                    elif status == 2:
                        text, color = "C2", "goldenrod"
                    elif status == 3:
                        text, color = "✗", "darkred"
                    else:
                        text, color = "?", "black"
                    ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", color=color, fontsize=12, fontweight="bold")

            ax.set_xlabel("Generation Length (ticks)", fontsize=14, fontweight="bold")
            if idx in [0, 3]:
                ax.set_ylabel("Inference Interval (ticks)", fontsize=14, fontweight="bold")
            ax.set_title(
                f"{percentile}th Percentile\n(Generation-Length-Specific RTT)",
                fontsize=16,
                fontweight="bold",
            )
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=constraint_colors[1], label="Valid (Both Constraints Satisfied)"),
            Patch(facecolor=constraint_colors[2], label="Constraint 1 Violated (Interval Too Short)"),
            Patch(facecolor=constraint_colors[3], label="Constraint 2 Violated (Buffer Too Small)"),
            Patch(facecolor=constraint_colors[4], label="Both Constraints Violated"),
            Patch(facecolor=constraint_colors[0], label="No Benchmark Data"),
        ]
        fig.legend(handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(top=0.82, bottom=0.18, hspace=0.3, wspace=0.3)
        plt.savefig(self.plots_dir / "parameter_constraint_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
        self._export_parameter_constraint_data(percentile_data, generation_range, inference_range)

    def generate_summary_report(self) -> str:
        """Generate text summary of findings."""
        if self.summary_data is None:
            return "No data available for analysis."
        report = []
        report.append("# Generation Length Analysis Summary\n")
        report.append(
            f"**Generation Lengths Tested:** {sorted(self.summary_data['generation_length'].tolist())}"
        )
        report.append(f"**Total Requests:** {self.summary_data['num_requests'].sum()}")
        if "round_trip_time_mean" in self.summary_data.columns:
            min_latency_idx = self.summary_data["round_trip_time_mean"].idxmin()
            optimal_gen_length = self.summary_data.iloc[min_latency_idx]["generation_length"]
            min_latency = self.summary_data.iloc[min_latency_idx]["round_trip_time_mean"] * 1000
            report.append(
                f"**Optimal Generation Length (Latency):** {optimal_gen_length} frames ({min_latency:.1f}ms)"
            )
            min_rtt = self.summary_data["round_trip_time_mean"].min() * 1000
            max_rtt = self.summary_data["round_trip_time_mean"].max() * 1000
            report.append(f"**Latency Range:** {min_rtt:.1f}ms - {max_rtt:.1f}ms")
            gen_lengths = self.summary_data["generation_length"].values
            latencies = self.summary_data["round_trip_time_mean"].values * 1000
            if len(gen_lengths) > 1:
                slope = (latencies[-1] - latencies[0]) / (gen_lengths[-1] - gen_lengths[0])
                report.append(f"**Latency Scaling:** {slope:.2f}ms per frame increase")
        if "round_trip_time_std" in self.summary_data.columns:
            min_var_idx = self.summary_data["round_trip_time_std"].idxmin()
            most_consistent = self.summary_data.iloc[min_var_idx]["generation_length"]
            min_std = self.summary_data.iloc[min_var_idx]["round_trip_time_std"] * 1000
            report.append(
                f"**Most Consistent Performance:** {most_consistent} frames ({min_std:.1f}ms std dev)"
            )
        return "\n".join(report)

    def export_summary_table(self):
        """Export a clean summary table."""
        if self.summary_data is None:
            return
        export_data = []
        for _, row in self.summary_data.iterrows():
            entry = {"Generation Length": int(row["generation_length"]), "Requests": row["num_requests"]}
            if "round_trip_time_mean" in row:
                entry["Mean RTT (ms)"] = f"{row['round_trip_time_mean'] * 1000:.1f}"
                entry["Std RTT (ms)"] = f"{row['round_trip_time_std'] * 1000:.1f}"
                if "round_trip_time_upper_bound_95" in row:
                    entry["95% Upper Bound RTT (ms)"] = f"{row['round_trip_time_upper_bound_95'] * 1000:.1f}"
                if "round_trip_time_upper_bound_99" in row:
                    entry["99% Upper Bound RTT (ms)"] = f"{row['round_trip_time_upper_bound_99'] * 1000:.1f}"
            if "inference_duration_mean" in row:
                entry["Mean Inference (ms)"] = f"{row['inference_duration_mean'] * 1000:.1f}"
            if "num_generated_notes_mean" in row:
                entry["Notes Generated"] = f"{row['num_generated_notes_mean']:.1f}"
            export_data.append(entry)
        export_df = pd.DataFrame(export_data)
        export_file = self.output_dir / "summary_table.csv"
        export_df.to_csv(export_file, index=False)
        self._export_confidence_intervals_table()
        print(f"📊 Summary table saved to: {export_file}")
        print("\nSummary Table:")
        print(export_df.to_string(index=False))

    def _export_confidence_intervals_table(self):
        """Export a detailed upper bounds table for round trip time only."""
        if self.summary_data is None:
            return
        ci_data = []
        for _, row in self.summary_data.iterrows():
            gen_length = int(row["generation_length"])
            if "round_trip_time_mean" in row:
                base_entry = {
                    "Generation Length": gen_length,
                    "Mean (ms)": f"{row['round_trip_time_mean'] * 1000:.2f}",
                    "Std (ms)": f"{row['round_trip_time_std'] * 1000:.2f}",
                    "Sample Size": int(row["num_requests"]),
                }
                percentiles = [60, 70, 80, 85, 90, 95, 98, 99, 99.5, 99.9, 100]
                for percentile in percentiles:
                    if percentile == int(percentile):
                        col_name = f"round_trip_time_upper_bound_{int(percentile)}"
                        display_name = f"{int(percentile)}%"
                    else:
                        col_name = f"round_trip_time_upper_bound_{str(percentile).replace('.', '_')}"
                        display_name = f"{percentile}%"
                    if col_name in row:
                        upper_bound = row[col_name] * 1000
                        base_entry[f"{display_name} Upper Bound (ms)"] = f"{upper_bound:.2f}"
                ci_data.append(base_entry)
        if ci_data:
            ci_df = pd.DataFrame(ci_data)
            ci_file = self.output_dir / "round_trip_time_upper_bounds.csv"
            ci_df.to_csv(ci_file, index=False)
            print(f"📊 Round trip time upper bounds saved to: {ci_file}")


def run_analysis(
    data_source: Path,
    output_dir: Path,
    anomaly_filter_pct: float = 0.0,
    no_plots: bool = False,
) -> int:
    """Entry used by eval/analyze_benchmark_results.py. Returns 0 on success."""
    if anomaly_filter_pct < 0 or anomaly_filter_pct > 50:
        print("❌ --anomaly_filter must be between 0 and 50")
        return 1
    analyzer = GenerationLengthAnalyzer(str(data_source), str(output_dir), anomaly_filter_pct)
    if not analyzer.load_data():
        print("❌ Failed to load data")
        return 1
    summary = analyzer.generate_summary_report()
    print(f"\n{summary}")
    analyzer.export_summary_table()
    if not no_plots:
        analyzer.generate_all_visualizations()
    print(f"\n✅ Analysis complete! Results saved to: {Path(output_dir).absolute()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze generation length benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze manual CSV files in directory
  %(prog)s bench_results/manual_remote --output_dir bench_analysis/remote/
""",
    )
    parser.add_argument("data_source", help="Path to data source (CSV file or directory)")
    parser.add_argument("--output_dir", default="analysis_results", help="Output directory for analysis results")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--anomaly_filter", type=float, default=0.0, help="Filter out anomalies: percentage of data to remove from each tail (0-50)")
    args = parser.parse_args()
    return run_analysis(Path(args.data_source), Path(args.output_dir), anomaly_filter_pct=args.anomaly_filter, no_plots=args.no_plots)


if __name__ == "__main__":
    raise SystemExit(main())
