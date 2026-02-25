#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from nll_compute.plot_nll_heatmap import build_pivot_table, plot_heatmap


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Plot heatmap from per-dir .nll.json files")
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default="records/nll_runs/experiments1",
        help="Directory containing .nll.json files",
    )
    parser.add_argument("--out", "-o", type=str, default="records/nll_heatmap_experiments1.png", help="Output PNG path")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional CSV output for the pivot matrix")
    parser.add_argument(
        "--value-mode",
        type=str,
        choices=["avg_mean", "weighted_avg", "weighted_total"],
        default="avg_mean",
        help="Which scalar to compute per .nll.json",
    )
    parser.add_argument("--annotate", action="store_true", help="Annotate cells with numeric values")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    args = parser.parse_args(argv)

    inp = Path(os.path.expanduser(args.input_dir))
    if not inp.exists() or not inp.is_dir():
        print(f"Input directory not found: {inp}")
        return 2

    intervals, gen_frames, mat = build_pivot_table(inp, mode=args.value_mode)
    if intervals is None:
        print("No data found to plot.")
        return 3

    outpath = Path(os.path.expanduser(args.out))
    plot_heatmap(intervals, gen_frames, mat, outpath, cmap=args.cmap, annotate=args.annotate)
    print(f"Wrote heatmap to {outpath}")

    if args.csv_out:
        import csv
        import math

        csvp = Path(os.path.expanduser(args.csv_out))
        csvp.parent.mkdir(parents=True, exist_ok=True)
        with csvp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["interval/gen_frame"] + [str(g) for g in gen_frames]
            writer.writerow(header)
            for i, itv in enumerate(intervals):
                row = [str(itv)] + ["" if math.isnan(mat[i, j]) else f"{mat[i, j]:.6f}" for j in range(mat.shape[1])]
                writer.writerow(row)
        print(f"Wrote pivot CSV to {csvp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
