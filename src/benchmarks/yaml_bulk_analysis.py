#!/usr/bin/env python3
"""
YAML-based StreamMUSE Bulk Analysis Tool (eval repo)

This is a port of the `app/analysis/yaml_bulk_analysis.py` script from the
`bench_results` branch of the StreamMUSE repository. It is responsible for
producing the two key latency figures used in the paper:

- Fig. 3: experiment_fitting_analysis.png
- Fig. 4: parameter_constraint_analysis_bpm.png

Compared to the original implementation, this version:
- Uses the eval repo's `BulkGenerationLengthAnalyzer` for data loading.
- Avoids unused dependencies (`scipy.optimize.curve_fit`, `sklearn.metrics.r2_score`).
- Is packaged under `src.benchmarks` for reuse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402

from .bulk_analysis import BulkGenerationLengthAnalyzer  # noqa: E402


class YAMLBulkAnalysisEngine:
    """YAML-configured bulk analysis engine with formula-based fitting."""

    def __init__(self, config_path: str):
        """Initialize with YAML configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.experiments_data: Dict[str, Dict[str, Any]] = {}
        self.combined_data: pd.DataFrame | None = None

        # Set up output directory
        self.output_dir = Path(self.config["analysis_settings"]["output_directory"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:  # pragma: no cover - defensive
            raise ValueError(f"Failed to load configuration from {self.config_path}: {e}") from e

    def load_experiments(self) -> None:
        """Load all experiments specified in the YAML configuration."""
        print("📊 YAML-Based StreamMUSE Bulk Analysis")
        print("=" * 50)

        anomaly_filter_pct = self.config["analysis_settings"].get("anomaly_filter_percentage", 0)
        if anomaly_filter_pct > 0:
            print(f"🔍 Anomaly filtering enabled: removing {anomaly_filter_pct}% from each tail")

        print(f"🔍 Loading {len(self.config['experiments'])} experiments from YAML configuration...")

        # Use the existing BulkGenerationLengthAnalyzer for data loading
        experiment_paths = [Path(exp["path"]) for exp in self.config["experiments"]]
        bulk_engine = BulkGenerationLengthAnalyzer(
            experiment_paths,
            self.output_dir,
            anomaly_filter_pct,
        )
        if not bulk_engine.load_all_experiments():
            raise ValueError("No experiment data could be loaded via BulkGenerationLengthAnalyzer")
        bulk_engine.combine_all_data()

        # Extract the loaded data and apply additional processing
        if bulk_engine.combined_data is not None:
            for i, exp_config in enumerate(self.config["experiments"]):
                exp_name = Path(exp_config["path"]).name
                formal_name = exp_config["formal_name"]

                # Get the data for this experiment
                exp_data = bulk_engine.combined_data[
                    bulk_engine.combined_data["experiment_source"] == exp_name
                ].copy()

                if len(exp_data) > 0:
                    # Store with formal name mapping
                    exp_data["formal_name"] = formal_name
                    exp_data["color"] = exp_config["color"]
                    self.experiments_data[formal_name] = {
                        "data": exp_data,
                        "config": exp_config,
                        "original_name": exp_name,
                    }
                    print(f"✅ Successfully loaded {formal_name}: {len(exp_data)} samples")
                else:
                    print(f"❌ No data found for {formal_name} (experiment_source: {exp_name})")
        else:
            print("❌ No combined data available from bulk engine")

        if self.experiments_data:
            # Combine all data for analysis
            all_data: List[pd.DataFrame] = []
            for _, exp_info in self.experiments_data.items():
                all_data.append(exp_info["data"])
            self.combined_data = pd.concat(all_data, ignore_index=True)
            print(
                f"✅ Combined {len(self.experiments_data)} experiments: "
                f"{len(self.combined_data)} total samples"
            )
        else:
            raise ValueError("No experiment data loaded successfully")

    # -------------------------------------------------------------------------
    # Experiment fitting (Fig. 3)
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_formula_values(gl_values: np.ndarray, formula_config: Dict[str, Any]) -> np.ndarray:
        """Calculate formula values for given generation lengths."""
        a = formula_config.get("a", 0)
        b = formula_config.get("b", 0)
        c = formula_config.get("c", 0)

        if formula_config.get("type") == "quadratic":
            return a * gl_values**2 + b * gl_values + c
        return b * gl_values + c

    def plot_experiment_fitting(self) -> None:
        """Create the main box plot + line fitting visualization (Fig. 3)."""
        if self.combined_data is None:
            raise ValueError("No combined data available. Did you call load_experiments()?")

        print("📊 Generating experiment fitting visualization...")

        plot_config = self.config["plot_settings"]

        # Use a more professional style
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=plot_config["figure_size"])

        # Get all unique generation lengths
        all_gl = sorted(self.combined_data["generation_length"].unique())

        # Create separate box plots for each experiment with offset positions
        num_experiments = len(self.experiments_data)
        box_width = 0.8 / max(num_experiments, 1)
        colors: List[str] = []

        for i, (formal_name, exp_info) in enumerate(self.experiments_data.items()):
            exp_data = exp_info["data"]
            color = exp_info["config"]["color"]
            colors.append(color)

            # Collect box plot data for this experiment
            box_data_exp: List[np.ndarray] = []
            box_positions_exp: List[float] = []

            for j, gl in enumerate(all_gl):
                gl_data = exp_data[exp_data["generation_length"] == gl]
                if len(gl_data) > 0:
                    # Convert to milliseconds for display
                    rtt_ms = gl_data["round_trip_time"] * 1000
                    box_data_exp.append(rtt_ms.values)
                    # Use actual GL values for box plot positions with offsets
                    position = gl + (i - (num_experiments - 1) / 2) * box_width * 2.0
                    box_positions_exp.append(position)

            # Create box plot for this experiment
            if box_data_exp:
                bp = ax.boxplot(
                    box_data_exp,
                    positions=box_positions_exp,
                    patch_artist=True,
                    widths=box_width,
                    showfliers=False,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(color=color, linewidth=2.5),
                    capprops=dict(color=color, linewidth=2.5),
                )

                # Color the box plots with enhanced visibility
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.9)
                    patch.set_edgecolor(color)
                    patch.set_linewidth(3)

        # Add fitted lines with confidence bands - extend from 0 to 17
        gl_range = np.linspace(0, 17, 200)

        for formal_name, exp_info in self.experiments_data.items():
            formula_config = exp_info["config"]["formula"]
            color = exp_info["config"]["color"]

            # Calculate fitted line
            fitted_values = self.calculate_formula_values(gl_range, formula_config)

            # Calculate ±5% confidence band
            upper_band = fitted_values * 1.05
            lower_band = fitted_values * 0.95

            # Plot fitted line using actual GL values
            line_label = f"{formal_name} ({formula_config['type'].title()} Fit)"
            ax.plot(
                gl_range,
                fitted_values,
                color=color,
                linewidth=plot_config["line_width"],
                label=line_label,
                alpha=0.9,
                linestyle="-",
            )

            # Add shaded confidence band
            ax.fill_between(
                gl_range,
                lower_band,
                upper_band,
                color=color,
                alpha=plot_config["shaded_area_alpha"],
                linestyle="None",
                label="±5% Band",
            )

        # Customize plot appearance
        ax.set_xlabel("Generation Length (Frames)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Round Trip Time (ms)", fontsize=16, fontweight="bold")
        ax.set_title(
            "StreamMUSE Performance Analysis\nEmpirical Data vs. Formula-Based Models",
            fontsize=18,
            fontweight="bold",
            pad=25,
        )

        ax.set_xlim(0, 17)
        ax.set_xticks(all_gl)
        ax.set_xticklabels(all_gl, fontsize=12)
        ax.tick_params(axis="y", labelsize=12)

        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.8)
        ax.set_facecolor("white")

        # Create a clean legend with only line fits (not confidence bands)
        handles, labels = ax.get_legend_handles_labels()
        fit_handles = []
        fit_labels = []
        for handle, label in zip(handles, labels):
            if "Fit)" in label and "±5%" not in label:
                fit_handles.append(handle)
                fit_labels.append(label)

        legend = ax.legend(
            fit_handles,
            fit_labels,
            loc="upper left",
            fontsize=13,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")

        # Improve y-axis limits for better visualization
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(max(0, y_min * 0.95), y_max * 1.05)

        # Add subtle formula information as text box
        formula_text: List[str] = []
        for formal_name, exp_info in self.experiments_data.items():
            formula_config = exp_info["config"]["formula"]
            a = formula_config.get("a", 0.0)
            b = formula_config.get("b", 0.0)
            c = formula_config.get("c", 0.0)

            if formula_config["type"] == "quadratic":
                if abs(a) < 0.01:
                    formula_text.append(f"{formal_name}: RT = {b:.1f}×GL + {c:.0f}")
                else:
                    formula_text.append(
                        f"{formal_name}: RT = {a:.3f}×GL² + {b:.1f}×GL + {c:.0f}"
                    )
            else:
                formula_text.append(f"{formal_name}: RT = {b:.1f}×GL + {c:.0f}")

        textstr = "\n".join(formula_text)
        props = dict(
            boxstyle="round,pad=0.5",
            facecolor="lightgray",
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )
        ax.text(
            0.98,
            0.02,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        plt.tight_layout()

        output_path = self.plots_dir / "experiment_fitting_analysis.png"
        plt.savefig(
            output_path,
            dpi=plot_config["dpi"],
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"✅ Experiment fitting plot saved to: {output_path}")

        # Print fitting statistics
        print("\n📈 Formula Fitting Summary:")
        for formal_name, exp_info in self.experiments_data.items():
            formula_config = exp_info["config"]["formula"]
            exp_data = exp_info["data"]

            gl_values = exp_data["generation_length"].values
            actual_rtt = exp_data["round_trip_time"].values * 1000
            predicted_rtt = self.calculate_formula_values(gl_values, formula_config)

            ss_res = np.sum((actual_rtt - predicted_rtt) ** 2)
            ss_tot = np.sum((actual_rtt - np.mean(actual_rtt)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            print(f"  {formal_name}:")
            print(f"    Formula: {formula_config['type']}")
            print(f"    R² = {r_squared:.3f}")
            print(f"    Samples: {len(exp_data)}")

    def generate_summary_report(self) -> None:
        """Generate a summary report of the YAML-based analysis."""
        report_path = self.output_dir / "yaml_analysis_summary.md"

        with open(report_path, "w") as f:
            f.write("# YAML-Based StreamMUSE Analysis Report\n\n")
            f.write(f"Generated from configuration: `{self.config_path.name}`\n\n")

            f.write("## Experiment Configuration\n\n")
            for formal_name, exp_info in self.experiments_data.items():
                config = exp_info["config"]
                f.write(f"### {formal_name}\n")
                f.write(f"- **Source**: {config['path']}\n")
                f.write(f"- **Description**: {config['description']}\n")
                f.write(f"- **Formula Type**: {config['formula']['type']}\n")
                if config["formula"]["type"] == "quadratic":
                    f.write(
                        "- **Formula**: RT = "
                        f"{config['formula']['a']:.2f}×GL² + "
                        f"{config['formula']['b']:.2f}×GL + "
                        f"{config['formula']['c']:.1f}\n"
                    )
                else:
                    f.write(
                        "- **Formula**: RT = "
                        f"{config['formula']['b']:.2f}×GL + "
                        f"{config['formula']['c']:.1f}\n"
                    )
                f.write(f"- **Color**: {config['color']}\n")
                f.write(f"- **Samples**: {len(exp_info['data'])}\n\n")

            f.write("## Analysis Settings\n\n")
            f.write(
                "- **Anomaly Filtering**: "
                f"{self.config['analysis_settings']['anomaly_filter_percentage']}% per tail\n"
            )
            f.write(
                f"- **Output Directory**: "
                f"{self.config['analysis_settings']['output_directory']}\n"
            )
            f.write(
                "- **Individual Analyses**: "
                f"{self.config['analysis_settings']['individual_analyses']}\n\n"
            )

            f.write("## Visualizations Generated\n\n")
            f.write(
                "- `experiment_fitting_analysis.png`: Box plots with formula-based "
                "fitted lines and ±5% confidence bands\n"
            )
            f.write(
                "- `parameter_constraint_analysis_bpm.png`: BPM-based constraint "
                "heatmaps across generation lengths and inference intervals\n"
            )

        print(f"📄 Summary report saved to: {report_path}")

    # -------------------------------------------------------------------------
    # Parameter constraint analysis (Fig. 4)
    # -------------------------------------------------------------------------

    def _calculate_experiment_percentiles(
        self,
        exp_data: pd.DataFrame,
        percentile: float,
        generation_frames: List[int],
    ) -> Dict[int, float]:
        """Calculate percentile data for an experiment at specific generation frames."""
        percentile_data: Dict[int, float] = {}

        for gl in generation_frames:
            gl_data = exp_data[exp_data["generation_length"] == gl]
            if len(gl_data) > 0:
                rtt_percentile = gl_data["round_trip_time"].quantile(percentile / 100) * 1000
                percentile_data[gl] = rtt_percentile

        return percentile_data

    @staticmethod
    def _create_constraint_matrix_bpm(
        percentile_data: Dict[int, float],
        tau_tick: float,
        inference_intervals: List[int],
        generation_frames: List[int],
    ) -> np.ndarray:
        """Create constraint matrix using BPM-based constraints."""
        matrix = np.full((len(inference_intervals), len(generation_frames)), -1, dtype=int)

        for i, inference_interval in enumerate(inference_intervals):
            for j, gl in enumerate(generation_frames):
                if gl in percentile_data:
                    rt_ms = percentile_data[gl]

                    # Constraint 1: ceil(RT / tau_tick) <= inference_interval
                    required_ticks = np.ceil(rt_ms / tau_tick)
                    constraint1_satisfied = required_ticks <= inference_interval

                    # Constraint 2: GL >= ceil(RT / tau_tick) + I
                    constraint2_satisfied = gl >= required_ticks + inference_interval

                    if constraint1_satisfied and constraint2_satisfied:
                        matrix[i, j] = 0  # Valid (both satisfied)
                    elif not constraint1_satisfied and constraint2_satisfied:
                        matrix[i, j] = 1  # Only constraint 1 violated
                    elif constraint1_satisfied and not constraint2_satisfied:
                        matrix[i, j] = 2  # Only constraint 2 violated
                    else:
                        matrix[i, j] = 3  # Both violated

        return matrix

    @staticmethod
    def _get_constraint_colormap():
        """Create custom colormap for constraint status."""
        from matplotlib.colors import ListedColormap

        colors = ["#2E7D32", "#FF9800", "#F44336", "#8B0000"]  # Green, Orange, Red, Dark Red
        return ListedColormap(colors)

    def _plot_constraint_boundaries(
        self,
        ax: matplotlib.axes.Axes,
        exp_config: Dict[str, Any],
        tau_tick: float,
        inference_intervals: List[int],
    ) -> None:
        """Plot constraint boundary traces on the heatmap."""
        formula_config = exp_config["formula"]
        a = formula_config.get("a", 0.0)
        b = formula_config.get("b", 0.0)
        c = formula_config.get("c", 0.0)
        t = tau_tick

        # Create continuous traces that extend beyond edges for clipping
        I_range = np.linspace(0, 10, 200)

        x_coords_1: List[float] = []
        y_coords_1: List[float] = []
        x_coords_2: List[float] = []
        y_coords_2: List[float] = []

        for I in I_range:
            y_grid = I - 1

            # Formula 1: GL = (-b + sqrt(b² - 4(a)(c - I*tau_tick))) / (2a)
            try:
                discriminant1 = b**2 - 4 * a * (c - I * t)
                if discriminant1 >= 0 and a != 0:
                    GL1 = (-b + np.sqrt(discriminant1)) / (2 * a)
                    x_grid1 = (GL1 - 1) / 2
                    if -2 <= y_grid <= 10:
                        x_coords_1.append(float(x_grid1))
                        y_coords_1.append(float(y_grid))
            except (ValueError, ZeroDivisionError):
                pass

            # Formula 2: GL = (-(b-t) - sqrt((b-t)² - 4(a)(c + I*tau_tick))) / (2a)
            try:
                b_minus_t = b - t
                discriminant2 = b_minus_t**2 - 4 * a * (c + I * t)
                if discriminant2 >= 0 and a != 0:
                    GL2 = (-b_minus_t - np.sqrt(discriminant2)) / (2 * a)
                    x_grid2 = (GL2 - 1) / 2
                    if -2 <= y_grid <= 10:
                        x_coords_2.append(float(x_grid2))
                        y_coords_2.append(float(y_grid))
            except (ValueError, ZeroDivisionError):
                pass

        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.5, 7.5)

        if x_coords_1 and y_coords_1:
            ax.plot(x_coords_1, y_coords_1, "k-", linewidth=2, alpha=0.8, label="C1 Boundary")

        if x_coords_2 and y_coords_2:
            ax.plot(x_coords_2, y_coords_2, "k--", linewidth=2, alpha=0.8, label="C2 Boundary")

    def _plot_individual_experiment_constraints(
        self,
        bpm_list: List[int],
        percentile: float,
        generation_frames: List[int],
        inference_intervals: List[int],
    ) -> None:
        """Create individual constraint plots for each experiment."""
        for formal_name, exp_info in self.experiments_data.items():
            exp_data = exp_info["data"]

            percentile_data = self._calculate_experiment_percentiles(
                exp_data,
                percentile,
                generation_frames,
            )

            fig, axes = plt.subplots(1, len(bpm_list), figsize=(4 * len(bpm_list), 4))
            if len(bpm_list) == 1:
                axes = [axes]

            for bpm_idx, bpm in enumerate(bpm_list):
                ax = axes[bpm_idx]

                tau_tick = 15000 / bpm

                matrix = self._create_constraint_matrix_bpm(
                    percentile_data,
                    tau_tick,
                    inference_intervals,
                    generation_frames,
                )

                im = ax.imshow(
                    matrix,
                    cmap=self._get_constraint_colormap(),
                    aspect="equal",
                    origin="lower",
                    vmin=0,
                    vmax=3,
                )

                ax.set_xlabel("Generation Length (Frames)", fontsize=12)
                if bpm_idx == 0:
                    ax.set_ylabel("Inference Interval (Ticks)", fontsize=12)
                ax.set_title(f"BPM: {bpm}\nτ = {tau_tick:.1f} ms/tick", fontsize=12)

                x_positions = range(len(generation_frames))
                y_positions = range(len(inference_intervals))
                ax.set_xticks(list(x_positions))
                ax.set_xticklabels([str(gl) for gl in generation_frames], fontsize=10)
                ax.set_yticks(list(y_positions))
                ax.set_yticklabels([str(ii) for ii in inference_intervals], fontsize=10)

                ax.set_xticks(np.arange(-0.5, len(generation_frames), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, len(inference_intervals), 1), minor=True)
                ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
                ax.tick_params(which="minor", size=0, width=0)
                ax.tick_params(which="major", size=0, width=0)

                for i in range(len(inference_intervals)):
                    for j in range(len(generation_frames)):
                        if matrix[i, j] == 1:
                            ax.text(
                                j,
                                i,
                                "C1",
                                ha="center",
                                va="center",
                                color="white",
                                fontsize=8,
                                fontweight="bold",
                            )
                        elif matrix[i, j] == 2:
                            ax.text(
                                j,
                                i,
                                "C2",
                                ha="center",
                                va="center",
                                color="white",
                                fontsize=8,
                                fontweight="bold",
                            )
                        elif matrix[i, j] == 3:
                            ax.text(
                                j,
                                i,
                                "C1+C2",
                                ha="center",
                                va="center",
                                color="white",
                                fontsize=7,
                                fontweight="bold",
                            )

                self._plot_constraint_boundaries(
                    ax,
                    exp_info["config"],
                    tau_tick,
                    inference_intervals,
                )

            plt.suptitle(f"Parameter Constraint Analysis: {formal_name}", fontsize=16, fontweight="bold")

            plt.subplots_adjust(right=0.85)

            cbar = plt.colorbar(im, ax=axes, shrink=0.7, aspect=15)
            cbar.set_ticks([0, 1, 2, 3])
            cbar.set_ticklabels(
                ["Valid", "C1 Violated", "C2 Violated", "Both Violated"],
            )
            cbar.set_label("Constraint Status", fontsize=12)

            safe_name = (
                formal_name.replace("/", "_")
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )
            output_path = self.plots_dir / f"constraint_analysis_{safe_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

            print(f"✅ Individual constraint analysis for {formal_name} saved to: {output_path}")

    def plot_parameter_constraint_analysis(self) -> None:
        """Generate parameter constraint analysis with BPM-based constraints (Fig. 4)."""
        if self.combined_data is None:
            raise ValueError("No combined data available. Did you call load_experiments()?")

        print("📊 Generating parameter constraint analysis...")

        constraint_config = self.config.get("constraint_analysis", {})
        bpm_list = constraint_config.get("bpm_list", [60, 80, 100, 120, 140, 160])
        percentile = constraint_config.get("percentile", 95.0)
        generation_frames = constraint_config.get(
            "generation_frames",
            [1, 3, 5, 7, 9, 11, 13, 15],
        )
        inference_intervals = constraint_config.get("inference_intervals", [1, 2, 3, 4, 5, 6, 7, 8])

        n_experiments = len(self.experiments_data)
        n_bpm = len(bpm_list)

        fig, axes = plt.subplots(n_experiments, n_bpm, figsize=(4 * n_bpm, 4 * n_experiments))
        if n_experiments == 1:
            axes = axes.reshape(1, -1)
        if n_bpm == 1:
            axes = axes.reshape(-1, 1)

        last_im = None

        for exp_idx, (formal_name, exp_info) in enumerate(self.experiments_data.items()):
            exp_data = exp_info["data"]

            percentile_data = self._calculate_experiment_percentiles(
                exp_data,
                percentile,
                generation_frames,
            )

            for bpm_idx, bpm in enumerate(bpm_list):
                ax = axes[exp_idx, bpm_idx]

                tau_tick = 15000 / bpm

                matrix = self._create_constraint_matrix_bpm(
                    percentile_data,
                    tau_tick,
                    inference_intervals,
                    generation_frames,
                )

                im = ax.imshow(
                    matrix,
                    cmap=self._get_constraint_colormap(),
                    aspect="equal",
                    origin="lower",
                    vmin=0,
                    vmax=3,
                )
                last_im = im

                ax.set_xlabel("Generation Length (Frames)", fontsize=12)
                ax.set_ylabel("Inference Interval (Ticks)", fontsize=12)
                ax.set_title(f"{formal_name}\nBPM: {bpm} (τ={tau_tick:.1f}ms)", fontsize=11)

                x_positions = range(len(generation_frames))
                y_positions = range(len(inference_intervals))
                ax.set_xticks(list(x_positions))
                ax.set_xticklabels([str(gl) for gl in generation_frames], fontsize=10)
                ax.set_yticks(list(y_positions))
                ax.set_yticklabels([str(ii) for ii in inference_intervals], fontsize=10)

                ax.set_xticks(np.arange(-0.5, len(generation_frames), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, len(inference_intervals), 1), minor=True)
                ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
                ax.tick_params(which="minor", size=0, width=0)
                ax.tick_params(which="major", size=0, width=0)

                for i in range(len(inference_intervals)):
                    for j in range(len(generation_frames)):
                        if matrix[i, j] == 1:
                            ax.text(
                                j,
                                i,
                                "C1",
                                ha="center",
                                va="center",
                                color="white",
                                fontsize=8,
                                fontweight="bold",
                            )
                        elif matrix[i, j] == 2:
                            ax.text(
                                j,
                                i,
                                "C2",
                                ha="center",
                                va="center",
                                color="white",
                                fontsize=8,
                                fontweight="bold",
                            )
                        elif matrix[i, j] == 3:
                            ax.text(
                                j,
                                i,
                                "C1+C2",
                                ha="center",
                                va="center",
                                color="white",
                                fontsize=7,
                                fontweight="bold",
                            )

                self._plot_constraint_boundaries(
                    ax,
                    exp_info["config"],
                    tau_tick,
                    inference_intervals,
                )

        plt.subplots_adjust(right=0.8)

        if last_im is not None:
            cbar = plt.colorbar(last_im, ax=axes, shrink=0.6, aspect=20)
            cbar.set_ticks([0, 1, 2, 3])
            cbar.set_ticklabels(
                ["Valid", "C1 Violated", "C2 Violated", "Both Violated"],
            )
            cbar.set_label("Constraint Status", fontsize=12)

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="black", linewidth=2, linestyle="-", label="C1 Boundary"),
            Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="C2 Boundary"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            fontsize=12,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor="black",
        )

        output_path = self.plots_dir / "parameter_constraint_analysis_bpm.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"✅ Parameter constraint analysis saved to: {output_path}")

        self._plot_individual_experiment_constraints(
            bpm_list,
            percentile,
            generation_frames,
            inference_intervals,
        )


def main(argv: List[str] | None = None) -> int:
    """CLI entry point mirroring the original bench_results script."""
    parser = argparse.ArgumentParser(description="YAML-based StreamMUSE Bulk Analysis (eval)")
    parser.add_argument("config_file", help="Path to YAML configuration file")
    parser.add_argument(
        "--output_dir",
        help="Override output directory from config (relative to eval repo root)",
    )

    args = parser.parse_args(argv)

    try:
        engine = YAMLBulkAnalysisEngine(args.config_file)

        if args.output_dir:
            engine.output_dir = Path(args.output_dir)
            engine.output_dir.mkdir(parents=True, exist_ok=True)
            engine.plots_dir = engine.output_dir / "plots"
            engine.plots_dir.mkdir(exist_ok=True)

        engine.load_experiments()
        engine.plot_experiment_fitting()
        engine.plot_parameter_constraint_analysis()
        engine.generate_summary_report()

        print(f"\n✅ YAML-based analysis complete! Results saved to: {engine.output_dir}")
        return 0
    except Exception as e:  # pragma: no cover - CLI error path
        print(f"❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

