#!/usr/bin/env python3
"""
Orchestrator to run cal_nll over directories listed in a manifest (midi_dirs.json) and
aggregate results.

Behavior:
- Load manifest JSON produced by `make_midi_dirs_json.py`.
- Load a model checkpoint once (RoFormerSymbolicTransformer) and keep it on the chosen device.
- For each manifest root, process each matching directory (the keys of `dirs`) by calling
  `make_nll_dir(model, dir_path, window, offset, save_path)` which writes per-dir JSONs.
- Merge per-dir JSON outputs into a combined JSON and run the aggregator (`aggregate_nll`)
  to produce a summary JSON per root.

This script does NOT attempt to run expensive GPU workloads in this environment; it will
attempt to load the checkpoint you provide. If you want to run on a cluster or local machine
with the checkpoint available, pass `--ckpt`.

Example:
  python run_nll_from_manifest.py --manifest records/midi_dirs.json --ckpt /path/to/model.ckpt \
    --out records/nll_runs --window 384 --offset 128

"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import torch

# Import the cal_nll helpers and aggregator from the repo
from cal_nll import make_nll_dir
from cal_nll_aggregate import aggregate_nll

try:
    # Import model class lazily; if unavailable this will raise when loading model
    from m2a_transformer import RoFormerSymbolicTransformer
except Exception:
    RoFormerSymbolicTransformer = None  # type: ignore


def abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def merge_json_dicts(json_paths: list[Path]) -> Dict[str, Any]:
    """Merge many per-dir JSON files (each mapping filename->result) into one dict."""
    merged: Dict[str, Any] = {}
    for p in json_paths:
        try:
            with p.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                merged.update(data)
        except Exception as e:
            print(f'Warning: failed to load {p}: {e}')
    return merged


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Run cal_nll over manifest and aggregate results')
    parser.add_argument('--manifest', '-m', type=str, default='records/midi_dirs.json', help='Path to manifest JSON')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--out', '-o', type=str, default='records/nll_runs', help='Output base directory')
    parser.add_argument('--window', type=int, default=384, help='Window length in frames')
    parser.add_argument('--offset', type=int, default=128, help='Window offset/stride')
    parser.add_argument('--device', type=str, default=None, help="Device string like 'cuda:0' or 'cpu'. If omitted, auto-selects.")
    parser.add_argument('--model_size', type=str, default='0.12B', help="Model size string passed to load_from_checkpoint (e.g. '0.12B')")
    parser.add_argument('--process-root-as-dir', action='store_true', help='If set, call make_nll_dir on the manifest root path itself instead of per-subdirectory')
    args = parser.parse_args(argv)

    manifest_path = Path(abs_path(args.manifest))
    if not manifest_path.exists():
        print(f'Manifest not found: {manifest_path}')
        return 2

    with manifest_path.open('r', encoding='utf-8') as f:
        manifest = json.load(f)

    out_base = Path(abs_path(args.out))
    ensure_dir(out_base)

    device = args.device
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load model once
    if RoFormerSymbolicTransformer is None:
        print('Model class RoFormerSymbolicTransformer not importable from m2a_transformer. Aborting model load.')
        return 3

    ckpt_path = abs_path(args.ckpt)
    if not os.path.exists(ckpt_path):
        print(f'Checkpoint not found: {ckpt_path}')
        return 4

    print(f'Loading model from {ckpt_path} to device {device} (model_size={args.model_size}) ...')
    model = RoFormerSymbolicTransformer.load_from_checkpoint(ckpt_path, model_size=args.model_size, map_location=device)
    model.to(device)
    model.eval()

    # Iterate manifest roots
    for root_key, info in manifest.items():
        print(f'Processing manifest root: {root_key}')
        if not isinstance(info, dict):
            print(f'  Skipping {root_key}: not a dict in manifest')
            continue
        if 'error' in info:
            print(f'  Skipping {root_key}: manifest error: {info.get("error")}')
            continue

        root_out_dir = out_base / root_key.replace(os.sep, '_')
        ensure_dir(root_out_dir)

        per_dir_json_paths = []

        if args.process_root_as_dir:
            # call make_nll_dir on the absolute root path recorded in manifest
            root_abs = info.get('root') or ''
            if not root_abs:
                print(f'  No root path recorded for {root_key}, skipping')
                continue
            save_path = root_out_dir / 'combined_nll.json'
            try:
                make_nll_dir(model, root_abs, args.window, args.offset, str(save_path))
                per_dir_json_paths.append(save_path)
            except Exception as e:
                print(f'  Error processing root dir {root_abs}: {e}')
                continue
            # If the combined JSON is empty (no files), fall back to per-subdirectory processing
            try:
                with save_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and len(data) == 0:
                    print(f'  Combined JSON at {save_path} contains 0 files — falling back to per-subdirectory processing')
                    per_dir_json_paths.clear()
                    dirs_map = info.get('dirs', {})
                    for rel_dir in sorted(dirs_map.keys()):
                        dir_abs = abs_path(rel_dir)
                        if not os.path.isdir(dir_abs):
                            print(f'    Skipping non-dir {dir_abs}')
                            continue
                        safe_name = rel_dir.replace(os.sep, '_').replace('..', '')
                        out_json = root_out_dir / (safe_name + '.nll.json')
                        try:
                            print(f'    Running make_nll_dir on {dir_abs} -> {out_json}')
                            make_nll_dir(model, dir_abs, args.window, args.offset, str(out_json))
                            per_dir_json_paths.append(out_json)
                        except Exception as e:
                            print(f'      Error running make_nll_dir on {dir_abs}: {e}')
                            continue
            except Exception:
                # If reading the save_path failed, ignore and continue; downstream logic will handle emptiness
                pass
        else:
            dirs_map = info.get('dirs', {})
            if not dirs_map:
                print(f'  No matching subdirectories for {root_key}, skipping')
                continue
            for rel_dir in sorted(dirs_map.keys()):
                # rel_dir in manifest was stored relative to cwd. Convert to absolute.
                dir_abs = abs_path(rel_dir)
                if not os.path.isdir(dir_abs):
                    print(f'  Skipping non-dir {dir_abs}')
                    continue
                # create per-dir output path
                safe_name = rel_dir.replace(os.sep, '_').replace('..', '')
                out_json = root_out_dir / (safe_name + '.nll.json')
                try:
                    print(f'  Running make_nll_dir on {dir_abs} -> {out_json}')
                    make_nll_dir(model, dir_abs, args.window, args.offset, str(out_json))
                    per_dir_json_paths.append(out_json)
                except Exception as e:
                    print(f'    Error running make_nll_dir on {dir_abs}: {e}')
                    continue

        # Merge per-dir JSONs into one combined JSON for the root
        if per_dir_json_paths:
            combined = merge_json_dicts(per_dir_json_paths)
            combined_path = root_out_dir / 'combined_nll.json'
            with combined_path.open('w', encoding='utf-8') as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
            print(f'  Wrote combined JSON: {combined_path} ({len(combined)} files)')

            # Run aggregation
            try:
                summary = aggregate_nll(str(combined_path))
                summary_path = root_out_dir / 'summary_nll.json'
                with summary_path.open('w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f'  Wrote summary JSON: {summary_path}')
            except Exception as e:
                print(f'  Error aggregating {combined_path}: {e}')
        else:
            print(f'  No per-dir JSON outputs for {root_key}; nothing to aggregate')

    print('Done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# run_nll_from_manifest.py \ \
#   --manifest records/midi_dirs.json \
#   --ckpt '/home/ubuntu/ugrip/shared_models/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt' \
#   --out records/nll_runs \
#   --window 128 \
#   --offset 32 \
#   --device 'cuda:0' \
#   --model_size '0.12B'
#   --process-root-as-dir