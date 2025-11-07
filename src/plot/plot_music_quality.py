#!/usr/bin/env python3
"""Generate JSD box plots and FMD bar charts from aggregated CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import math

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

# Shared palette (matches earlier plots)
MODEL_ORDER = ['aria025b', 'dropout', 'baseline012b']
MODEL_LABELS = {
    'aria025b': 'Aria 0.25B',
    'dropout': 'Dropout',
    'baseline012b': 'Baseline 0.12B',
}
MODEL_COLORS = ['#4C72B0', '#55A868', '#DD8452']
PROMPT_ALPHA = 0.65


def _parse_jsd_summary(text: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    if not isinstance(text, str):
        return metrics
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith('JSD:'):
            metric_name = stripped.split()[0]
            stats: dict[str, float] = {}
            j = idx + 1
            while j < len(lines) and lines[j].startswith('    '):
                parts = lines[j].strip().split(':', 1)
                if len(parts) == 2:
                    key, value = parts
                    key = key.lower().strip()
                    try:
                        stats[key] = float(value.strip())
                    except ValueError:
                        pass
                j += 1
            metrics[metric_name] = stats
    return metrics


def _parse_fmd_value(text: str) -> Optional[float]:
    if not isinstance(text, str):
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('Frechet Music Distance:'):
            try:
                return float(line.split(':', 1)[1].strip())
            except ValueError:
                return None
    return None


def _collect_jsd_records(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str | int]] = []
    for _, row in df.iterrows():
        prompt_len = row.iloc[0]
        for model in MODEL_ORDER:
            summary = _parse_jsd_summary(row[model])
            for metric in ('Pitch', 'Onset', 'Duration'):
                stats = summary.get(metric, {})
                if not {'min', 'q1', 'median', 'q3', 'max'}.issubset(stats):
                    continue
                records.append(
                    {
                        'prompt_len': prompt_len,
                        'model': model,
                        'metric': metric,
                        'min': stats['min'],
                        'q1': stats['q1'],
                        'median': stats['median'],
                        'q3': stats['q3'],
                        'max': stats['max'],
                        'mean': stats.get('mean'),
                    }
                )
    return pd.DataFrame(records)


def _collect_fmd_records(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for _, row in df.iterrows():
        prompt_len = row.iloc[0]
        for model in MODEL_ORDER:
            fmd = _parse_fmd_value(row[model])
            if fmd is not None:
                records.append(
                    {
                        'Prompt Length': f'Prompt {prompt_len}',
                        'Model': MODEL_LABELS.get(model, model),
                        'FMD': fmd,
                    }
                )
    return pd.DataFrame(records)


def _parse_polydis_texture(text: str) -> dict[str, float]:
    if not isinstance(text, str):
        return {}

    metrics: dict[str, float] = {}
    current_section: str | None = None
    lines = text.splitlines()

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('PolyDis (') and stripped.endswith('):'):
            current_section = stripped[len('PolyDis ('):-2]
            continue

        if current_section and stripped.startswith('txt mean distance:'):
            mean_value: float | None = None
            j = idx + 1
            while j < len(lines) and lines[j].startswith('      '):
                sub = lines[j].strip()
                if sub.startswith('mean:'):
                    try:
                        mean_value = float(sub.split(':', 1)[1].strip())
                    except ValueError:
                        mean_value = None
                    break
                j += 1

            if mean_value is not None:
                metrics[current_section] = mean_value

    return metrics


def _collect_polydis_records(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for _, row in df.iterrows():
        prompt_len = row.iloc[0]
        for model in MODEL_ORDER:
            metrics = _parse_polydis_texture(row.get(model))
            for section_name, value in metrics.items():
                canonical = {
                    'prompt': 'prompt',
                    'prompt vs continuation': 'prompt vs ground truth continuation',
                    'prompt vs generated continuation': 'prompt vs generated continuation',
                    'prompt vs ground truth continuation': 'prompt vs ground truth continuation',
                }.get(section_name, section_name)
                if canonical not in {
                    'prompt',
                    'prompt vs generated continuation',
                    'prompt vs ground truth continuation',
                }:
                    continue
                records.append(
                    {
                        'Prompt Length': f'Prompt {prompt_len}',
                        'Model': MODEL_LABELS.get(model, model),
                        'Type': canonical,
                        'Value': value,
                    }
                )
    return pd.DataFrame(records)


def _parse_harmonicity(text: str) -> dict[str, float]:
    if not isinstance(text, str):
        return {}

    metrics: dict[str, float] = {}
    in_harmonicity = False
    current_metric: str | None = None

    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()

        if stripped.endswith('Harmonicity:'):
            in_harmonicity = True
            current_metric = None
            continue

        if not in_harmonicity:
            continue

        if stripped == '':
            continue

        if not line.startswith('    '):
            # Section ended
            if not line.startswith(' '):
                break
            continue

        if stripped.endswith('ratio:'):
            current_metric = stripped[:-1]
            continue

        if current_metric and stripped.startswith('mean:'):
            try:
                metrics[current_metric] = float(stripped.split(':', 1)[1].strip())
            except ValueError:
                pass
            continue

    return metrics


def _collect_harmonicity_records(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for _, row in df.iterrows():
        prompt_len = row.iloc[0]
        for model in MODEL_ORDER:
            metrics = _parse_harmonicity(row.get(model))
            for metric_name, value in metrics.items():
                records.append(
                    {
                        'Prompt Length': f'Prompt {prompt_len}',
                        'Model': MODEL_LABELS.get(model, model),
                        'Metric': metric_name,
                        'Value': value,
                    }
                )
    return pd.DataFrame(records)


def plot_jsd_boxplots(df: pd.DataFrame, output_path: Path) -> None:
    parsed = _collect_jsd_records(df)
    if parsed.empty:
        raise RuntimeError('No JSD statistics found in the input CSV.')

    prompt_lengths = sorted(parsed['prompt_len'].unique())
    color_map = {pl: MODEL_COLORS[idx % len(MODEL_COLORS)] for idx, pl in enumerate(prompt_lengths)}

    if len(prompt_lengths) == 1:
        offsets = {prompt_lengths[0]: 0.0}
    else:
        span = 0.4
        step = span / max(1, len(prompt_lengths) - 1)
        start = -span / 2
        offsets = {pl: start + idx * step for idx, pl in enumerate(prompt_lengths)}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for ax, metric in zip(axes, ('Pitch', 'Onset', 'Duration')):
        metric_data = parsed[parsed['metric'] == metric]
        base_cap = 0.4
        if metric_data.empty:
            y_focus = base_cap
            iqr_max = 0.0
        else:
            metric_numeric = metric_data.assign(
                q1=pd.to_numeric(metric_data['q1'], errors='coerce'),
                q3=pd.to_numeric(metric_data['q3'], errors='coerce'),
            )
            box_top = metric_numeric['q3'].max(skipna=True)
            iqr_max = (metric_numeric['q3'] - metric_numeric['q1']).max(skipna=True)
            if not np.isfinite(box_top):
                box_top = base_cap
            if not np.isfinite(iqr_max):
                iqr_max = 0.0
            margin = max(0.01, 0.3 * iqr_max)
            y_focus = box_top + margin
        y_max = y_focus if np.isfinite(y_focus) else base_cap

        # Pick tighter tick spacing when the distribution is concentrated near zero.
        thresholds = [
            (0.05, 0.005),
            (0.10, 0.01),
            (0.20, 0.02),
            (0.30, 0.05),
        ]
        major_step = 0.1
        for limit, step in thresholds:
            if y_max <= limit:
                major_step = step
                break

        cap_target = max(y_max + major_step * 0.25, major_step)
        cap = max(major_step, math.ceil(cap_target / major_step) * major_step)
        if cap < base_cap and y_max > base_cap:
            cap = base_cap

        minor_step = major_step / 2
        formatter = '{:.3f}' if major_step < 0.01 else '{:.2f}'
        for _, rec in metric_data.iterrows():
            model_idx = MODEL_ORDER.index(rec['model'])
            prompt_len = rec['prompt_len']
            pos = model_idx + offsets[prompt_len]
            stats = {
                'med': float(rec['median']),
                'q1': float(rec['q1']),
                'q3': float(rec['q3']),
                'whislo': float(rec['min']),
                'whishi': float(rec['max']),
                'fliers': [],
            }
            if pd.notna(rec['mean']):
                stats['mean'] = float(rec['mean'])

            artists = ax.bxp([stats], positions=[pos], widths=0.18, showmeans=True, patch_artist=True)
            color = color_map[prompt_len]
            box = artists['boxes'][0]
            box.set(facecolor=color, edgecolor=color, alpha=PROMPT_ALPHA)
            for element_name, element_list in artists.items():
                for artist in element_list:
                    if element_name == 'means':
                        artist.set_marker('o')
                        artist.set_markerfacecolor(color)
                        artist.set_markeredgecolor('black')
                        artist.set_markersize(4)
                    else:
                        artist.set_color(color)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in MODEL_ORDER])
        ax.set_title(f'{metric} JSD')
        ax.set_ylabel('Jensen-Shannon Divergence')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        major_ticks = np.arange(0.0, cap + major_step / 2, major_step)
        ax.set_yticks(major_ticks)
        tick_labels = [formatter.format(t) for t in major_ticks]
        ax.set_yticklabels(tick_labels)
        if major_step <= 0.05:
            minor_ticks = np.arange(0.0, cap + minor_step / 2, minor_step)
            ax.set_yticks(minor_ticks, minor=True)
        else:
            ax.set_yticks([], minor=True)
        ax.set_ylim(0.0, cap)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker='s',
            color='none',
            markerfacecolor=color_map[pl],
            markeredgecolor=color_map[pl],
            alpha=PROMPT_ALPHA,
            markersize=8,
            label=f'Prompt {pl}',
        )
        for pl in prompt_lengths
    ]
    axes[0].legend(handles=legend_handles, title='Prompt Length', loc='upper right')

    fig.suptitle('Accompaniment vs Ground Truth: JSD Distributions by Model', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_fmd_bar_chart(df: pd.DataFrame, output_path: Path) -> None:
    parsed = _collect_fmd_records(df)
    if parsed.empty:
        raise RuntimeError('No Frechet Music Distance values found in the input CSV.')

    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bar_width = 0.23
    prompt_lengths = sorted(parsed['Prompt Length'].unique())
    model_colors = {MODEL_LABELS[m]: MODEL_COLORS[idx] for idx, m in enumerate(MODEL_ORDER)}
    offsets = {
        MODEL_LABELS[m]: idx - (len(MODEL_ORDER) - 1) / 2 for idx, m in enumerate(MODEL_ORDER)
    }

    x_positions = list(range(len(prompt_lengths)))

    for _, rec in parsed.iterrows():
        prompt_idx = prompt_lengths.index(rec['Prompt Length'])
        model = rec['Model']
        x = x_positions[prompt_idx] + offsets[model] * bar_width
        ax.bar(
            x,
            rec['FMD'],
            width=bar_width * 0.9,
            color=model_colors[model],
            edgecolor=model_colors[model],
            linewidth=0.8,
            alpha=PROMPT_ALPHA,
            label=model,
        )

    handles = []
    seen: set[str] = set()
    for model, color in model_colors.items():
        if model not in seen:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker='s',
                    color='none',
                    markerfacecolor=color,
                    markeredgecolor=color,
                    alpha=PROMPT_ALPHA,
                    linewidth=0,
                    markersize=8,
                    label=model,
                )
            )
            seen.add(model)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(prompt_lengths)
    ax.set_ylabel('Frechet Music Distance')
    ax.set_title('Frechet Music Distance Across Models and Prompt Lengths')
    ax.legend(handles=handles, title='Model', frameon=True)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)

    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(1.0, height * 0.01),
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_polydis_texture(df: pd.DataFrame, output_path: Path) -> None:
    parsed = _collect_polydis_records(df)
    if parsed.empty:
        raise RuntimeError('No PolyDis texture statistics found in the input CSV.')

    plt.style.use('seaborn-v0_8-whitegrid')

    type_labels = {
        'prompt': 'Prompt vs Prompt',
        'prompt vs generated continuation': 'Prompt vs Generated Continuation',
        'prompt vs ground truth continuation': 'Prompt vs Ground Truth Continuation',
    }

    available_types = [t for t in parsed['Type'].unique() if t in type_labels]
    if not available_types:
        raise RuntimeError('PolyDis texture data missing expected sections.')

    poly_types = [t for t in type_labels if t in available_types]
    prompt_lengths = sorted(parsed['Prompt Length'].unique())
    model_colors = {MODEL_LABELS[m]: MODEL_COLORS[idx] for idx, m in enumerate(MODEL_ORDER)}
    offsets = {
        MODEL_LABELS[m]: idx - (len(MODEL_ORDER) - 1) / 2 for idx, m in enumerate(MODEL_ORDER)
    }

    fig, axes = plt.subplots(1, len(poly_types), figsize=(7 * len(poly_types), 4.5), sharey=True)
    if len(poly_types) == 1:
        axes = [axes]

    bar_width = 0.18
    x_positions = list(range(len(prompt_lengths)))

    for ax, poly_type in zip(axes, poly_types):
        subset = parsed[parsed['Type'] == poly_type]
        for _, rec in subset.iterrows():
            prompt_idx = prompt_lengths.index(rec['Prompt Length'])
            model = rec['Model']
            x = x_positions[prompt_idx] + offsets[model] * bar_width
            ax.bar(
                x,
                rec['Value'],
                width=bar_width * 0.9,
                color=model_colors[model],
                edgecolor=model_colors[model],
                linewidth=0.8,
                alpha=PROMPT_ALPHA,
                label=model,
            )

        handles = []
        seen: set[str] = set()
        for model, color in model_colors.items():
            if model not in seen:
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker='s',
                        color='none',
                        markerfacecolor=color,
                        markeredgecolor=color,
                        alpha=PROMPT_ALPHA,
                        linewidth=0,
                        markersize=8,
                        label=model,
                    )
                )
                seen.add(model)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(prompt_lengths)
        ax.set_ylabel('PolyDis Texture Distance (mean)')
        ax.set_title(type_labels.get(poly_type, poly_type))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.xaxis.grid(False)

        for bar in ax.patches:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(0.05, height * 0.01),
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )

        ax.legend(handles=handles, title='Model', frameon=True)

    fig.suptitle('PolyDis Texture Distance Comparison', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_harmonicity(df: pd.DataFrame, output_path: Path) -> None:
    parsed = _collect_harmonicity_records(df)
    if parsed.empty:
        raise RuntimeError('No harmonicity statistics found in the input CSV.')

    plt.style.use('seaborn-v0_8-whitegrid')

    metrics_order = ['consonant ratio', 'dissonant ratio', 'unsupported ratio']
    prompt_lengths = sorted(parsed['Prompt Length'].unique())
    model_colors = {MODEL_LABELS[m]: MODEL_COLORS[idx] for idx, m in enumerate(MODEL_ORDER)}
    offsets = {
        MODEL_LABELS[m]: idx - (len(MODEL_ORDER) - 1) / 2 for idx, m in enumerate(MODEL_ORDER)
    }

    fig, axes = plt.subplots(1, len(metrics_order), figsize=(6.5 * len(metrics_order), 4.5), sharey=False)

    bar_width = 0.18
    x_positions = list(range(len(prompt_lengths)))

    for ax, metric in zip(axes, metrics_order):
        subset = parsed[parsed['Metric'] == metric]
        if subset.empty:
            continue
        for _, rec in subset.iterrows():
            prompt_idx = prompt_lengths.index(rec['Prompt Length'])
            model = rec['Model']
            x = x_positions[prompt_idx] + offsets[model] * bar_width
            ax.bar(
                x,
                rec['Value'],
                width=bar_width * 0.9,
                color=model_colors[model],
                edgecolor=model_colors[model],
                linewidth=0.8,
                alpha=PROMPT_ALPHA,
                label=model,
            )

        handles = []
        seen: set[str] = set()
        for model, color in model_colors.items():
            if model not in seen:
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker='s',
                        color='none',
                        markerfacecolor=color,
                        markeredgecolor=color,
                        alpha=PROMPT_ALPHA,
                        linewidth=0,
                        markersize=8,
                        label=model,
                    )
                )
                seen.add(model)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(prompt_lengths)
        ax.set_ylabel('Mean Ratio')
        ax.set_title(metric.title())
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.xaxis.grid(False)

        for bar in ax.patches:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(0.01, height * 0.01),
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )

        ax.legend(handles=handles, title='Model', frameon=True)

    fig.suptitle('Harmonicity Ratios by Model', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot music quality metrics from aggregated CSV.')
    parser.add_argument('csv', type=Path, help='Path to the aggregated results CSV.')
    parser.add_argument('--jsd-output', type=Path, default=Path('music_quality_jsd_boxplots.png'))
    parser.add_argument('--fmd-output', type=Path, default=Path('music_quality_fmd.png'))
    parser.add_argument('--polydis-output', type=Path, default=Path('music_quality_polydis_texture.png'))
    parser.add_argument('--harmonicity-output', type=Path, default=Path('music_quality_harmonicity.png'))
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f'CSV file not found: {args.csv}')

    df = pd.read_csv(args.csv)
    if df.empty:
        parser.error('CSV file is empty.')

    plot_jsd_boxplots(df, args.jsd_output)
    plot_fmd_bar_chart(df, args.fmd_output)
    plot_polydis_texture(df, args.polydis_output)
    plot_harmonicity(df, args.harmonicity_output)
    print(f'Saved JSD plot to {args.jsd_output.resolve()}')
    print(f'Saved FMD plot to {args.fmd_output.resolve()}')
    print(f'Saved PolyDis plot to {args.polydis_output.resolve()}')
    print(f'Saved Harmonicity plot to {args.harmonicity_output.resolve()}')


if __name__ == '__main__':
    main()
