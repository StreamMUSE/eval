#!/usr/bin/env python3
import os
import argparse
import torch
from move_to_eval.nll_compute.batch import start_call_nll_dir


def main():
    parser = argparse.ArgumentParser(
        description="Compute NLLs for a directory of MID/PT files using a model checkpoint"
    )
    parser.add_argument(
        "--midi_dir",
        type=str,
        default="~/stanleyz/StreamMUSE/experiments/realtime/prompt_75_gen_384/generated_without_prompt",
        help="Directory containing .mid or .pt files",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/ubuntu/stanleyz/shared_models/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--save_json_path", type=str, default="records/nll_results.json", help="Output JSON path")
    parser.add_argument("--window", type=int, default=384, help="Sliding window length in frames")
    parser.add_argument("--offset", type=int, default=128, help="Sliding window offset/stride")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device string, e.g. 'cuda:0' or 'cpu'",
    )
    parser.add_argument("--model_size", type=str, default="0.12B", help="Model size string passed to loader")

    args = parser.parse_args()

    midi_dir = os.path.expanduser(args.midi_dir)
    ckpt_path = os.path.expanduser(args.ckpt_path)
    save_json_path = os.path.expanduser(args.save_json_path)

    print(f"Loading checkpoint: {ckpt_path} on device {args.device}")
    print(f"Processing directory: {midi_dir} -> saving to {save_json_path}")

    start_call_nll_dir(
        ckpt_path=ckpt_path,
        midi_dir=midi_dir,
        window=args.window,
        offset=args.offset,
        save_path=save_json_path,
        model_size=args.model_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
