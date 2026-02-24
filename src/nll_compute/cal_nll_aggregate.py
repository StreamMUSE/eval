"""Aggregate per-file NLL JSONs into dataset-level statistics.

Input JSON format (as produced by `make_nll_dir`) should be a mapping:
  "file.mid": {"total_nll": float, "avg_nll": float, "total_tokens": int},
  "other.mid": {"error": "..."}

This script computes:
- files_count (processed files)
- files_successful (files with numeric results)
- total_tokens (sum)
- total_nll_sum (sum)
- weighted_avg_nll = total_nll_sum / total_tokens
- mean_avg_nll_unweighted, std, var, median, min, max (over per-file avg_nll)
- also stats for total_nll (mean/std/var/median/min/max)

Usage:
  uv run python cal_nll_aggregate.py --input records/nll_results.json --output records/nll_summary.json

"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional
import math
import statistics


def aggregate_nll(input_json: str) -> Dict[str, Any]:
    """Read input_json and compute summary statistics.

    Returns a dict suitable for JSON serialization.
    """
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    files_count = len(data)
    total_tokens = 0
    total_nll_sum = 0.0

    avg_nll_list = []
    total_nll_list = []
    per_file = {}

    for fname, val in data.items():
        if not isinstance(val, dict):
            continue
        if 'error' in val:
            per_file[fname] = {'error': val['error']}
            continue
        try:
            total_nll = float(val['total_nll'])
            avg_nll = float(val['avg_nll'])
            tokens = int(val['total_tokens'])
        except Exception as e:
            per_file[fname] = {'error': f'parse_error: {e}'}
            continue

        total_tokens += tokens
        total_nll_sum += total_nll
        avg_nll_list.append(avg_nll)
        total_nll_list.append(total_nll)
        per_file[fname] = {'total_nll': total_nll, 'avg_nll': avg_nll, 'total_tokens': tokens}

    successful = len(avg_nll_list)

    def safe_stats(xs):
        if not xs:
            return {'count': 0, 'mean': None, 'std': None, 'var': None, 'median': None, 'min': None, 'max': None}
        mean = statistics.mean(xs)
        var = statistics.pvariance(xs) if len(xs) > 0 else 0.0
        std = math.sqrt(var)
        med = statistics.median(xs)
        return {'count': len(xs), 'mean': mean, 'std': std, 'var': var, 'median': med, 'min': min(xs), 'max': max(xs)}

    avg_stats = safe_stats(avg_nll_list)
    total_stats = safe_stats(total_nll_list)

    weighted_avg_nll: Optional[float] = None
    if total_tokens > 0:
        weighted_avg_nll = total_nll_sum / float(total_tokens)

    summary = {
        'files_count': files_count,
        'files_successful': successful,
        'total_tokens': total_tokens,
        'total_nll_sum': total_nll_sum,
        'weighted_avg_nll': weighted_avg_nll,
        'per_file_stats': {
            'avg_nll': avg_stats,
            'total_nll': total_stats,
        },
    }

    return {'summary': summary, 'per_file': per_file}


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Aggregate NLL JSON into summary statistics')
    parser.add_argument('--input', type=str, required=True, help='Input JSON path produced by make_nll_dir')
    parser.add_argument('--output', type=str, required=False, help='Optional output JSON path for summary')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print the JSON summary to stdout')
    args = parser.parse_args()

    summary = aggregate_nll(os.path.expanduser(args.input))

    if args.pretty:
        print(json.dumps(summary['summary'], indent=2, ensure_ascii=False))

    if args.output:
        outp = os.path.expanduser(args.output)
        os.makedirs(os.path.dirname(outp) or '.', exist_ok=True)
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f'Wrote summary to {outp}')


if __name__ == '__main__':
    main()
