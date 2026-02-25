"""Numeric and statistical helpers for evaluation metrics.

Provides a single entrypoint `compute_stats(values)` which returns a dict
with count, sum, min, max, mean, variance (pop/sample), stdev (pop/sample),
and percentiles (p25,p50,p75) plus iqr. Designed to be dependency-free
(only stdlib `statistics`), and robust against missing values.
"""

from __future__ import annotations

import math
import statistics
from typing import Iterable, List, Dict, Union


def _percentile_sorted(sorted_vals: List[float], p: float) -> float:
    """Compute percentile p (0-100) on already-sorted list using linear interpolation.

    If list is empty returns NaN. If single item, returns that item.
    """
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_vals[0])
    # Convert p to fractional rank between 0 and n-1
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def compute_stats(values: Iterable[Union[float, int, str, None]]) -> Dict[str, float]:
    """Return a dictionary of common descriptive statistics for `values`.

    Keys returned include:
      - count, sum, min, max
      - mean
      - variance_pop, variance_samp
      - stdev_pop, stdev_samp
      - p25, p50, p75, iqr

    All numeric outputs are floats (or NaN when not applicable).
    """
    # Coerce values to float where possible. Skip None or non-convertible items.
    vals: List[float] = []
    for x in values:
        if x is None:
            continue
        try:
            val = float(x)
            if not math.isnan(val):
                vals.append(val)
        except (TypeError, ValueError):
            # skip entries that cannot be converted to float
            continue
    out: Dict[str, float] = {}
    if not vals:
        # consistent shape when no data: count=0, sum=0.0, other stats -> NaN
        nan = float("nan")
        out.update(
            {
                "count": 0.0,
                "sum": nan,
                "min": nan,
                "max": nan,
                "mean": nan,
                "variance_pop": nan,
                "variance_samp": nan,
                "stdev_pop": nan,
                "stdev_samp": nan,
                "std": nan,
                "p25": nan,
                "p50": nan,
                "p75": nan,
                "iqr": nan,
            }
        )
        return out

    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    s = sum(vals_sorted)
    mn = float(vals_sorted[0])
    mx = float(vals_sorted[-1])
    mean_v = float(statistics.mean(vals_sorted))
    # population (p) and sample (s) variance
    try:
        var_pop = float(statistics.pvariance(vals_sorted))
    except Exception:
        var_pop = 0.0
    try:
        var_samp = float(statistics.variance(vals_sorted)) if n > 1 else 0.0
    except Exception:
        var_samp = 0.0
    try:
        stdev_pop = float(statistics.pstdev(vals_sorted))
    except Exception:
        stdev_pop = 0.0
    try:
        stdev_samp = float(statistics.stdev(vals_sorted)) if n > 1 else 0.0
    except Exception:
        stdev_samp = 0.0

    p25 = _percentile_sorted(vals_sorted, 25.0)
    p50 = _percentile_sorted(vals_sorted, 50.0)
    p75 = _percentile_sorted(vals_sorted, 75.0)
    iqr = float(p75 - p25)

    out.update(
        {
            "count": float(n),
            "sum": float(s),
            "min": mn,
            "max": mx,
            "mean": mean_v,
            "variance_pop": var_pop,
            "variance_samp": var_samp,
            "stdev_pop": stdev_pop,
            "stdev_samp": stdev_samp,
            "std": stdev_samp,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "iqr": iqr,
        }
    )
    return out
