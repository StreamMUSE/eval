#!/usr/bin/env python3
import os
import json
import argparse
from src.nll_compute.aggregate import aggregate_nll


def main():
    parser = argparse.ArgumentParser(description="Aggregate NLL JSON into summary statistics")
    parser.add_argument("--input", type=str, required=True, help="Input JSON path produced by make_nll_dir")
    parser.add_argument("--output", type=str, required=False, help="Optional output JSON path for summary")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON summary to stdout")
    args = parser.parse_args()

    summary = aggregate_nll(os.path.expanduser(args.input))

    if args.pretty:
        print(json.dumps(summary["summary"], indent=2, ensure_ascii=False))

    if args.output:
        outp = os.path.expanduser(args.output)
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote summary to {outp}")


if __name__ == "__main__":
    main()
