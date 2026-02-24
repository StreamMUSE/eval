"""Batch processing utilities for NLL compute over directories."""

import os
import json
import torch
from typing import Dict, Any

from move_to_eval.nll_compute.core import cal_nll
from m2a_transformer import RoFormerSymbolicTransformer


def cal_nll_dir(model: object, midi_dir: str, window: int, offset: int) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute NLLs for all MIDI/.pt files in a directory."""
    if not os.path.exists(midi_dir):
        raise ValueError(f"Directory not found: {midi_dir}")

    results: Dict[str, Dict[str, torch.Tensor]] = {}
    for fname in sorted(os.listdir(midi_dir)):
        path = os.path.join(midi_dir, fname)
        if os.path.isdir(path):
            continue
        if not (fname.lower().endswith(".mid") or fname.lower().endswith(".pt")):
            continue
        try:
            res = cal_nll(model, path, window, offset)
            results[fname] = res
        except Exception as e:
            results[fname] = {"error": str(e)}
    return results


def make_nll_dir(model: object, midi_dir: str, window: int, offset: int, save_path: str) -> None:
    """Compute NLLs for a directory and save results to JSON at save_path."""
    results = cal_nll_dir(model, midi_dir, window, offset)

    serializable: Dict[str, Any] = {}
    for fname, val in results.items():
        if isinstance(val, dict) and "error" in val:
            serializable[fname] = {"error": val["error"]}
            continue
        try:
            serializable[fname] = {
                "total_nll": float(val["total_nll"].cpu().item()),
                "avg_nll": float(val["avg_nll"].cpu().item()),
                "total_tokens": int(val["total_tokens"].cpu().item()),
            }
        except Exception:
            serializable[fname] = {k: str(v) for k, v in val.items()}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def start_call_nll_dir(
    ckpt_path: str,
    midi_dir: str,
    window: int,
    offset: int,
    save_path: str,
    model_size: str = "0.12B",
    device: str | None = None,
) -> None:
    """Load a model from a checkpoint and run make_nll_dir."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = RoFormerSymbolicTransformer.load_from_checkpoint(ckpt_path, model_size=model_size, map_location=device)
    model.save_name = os.path.basename(ckpt_path)
    model.to(device)
    model.eval()

    make_nll_dir(model, midi_dir, window, offset, save_path)
