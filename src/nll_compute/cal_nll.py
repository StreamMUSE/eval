# 1===
# cal_nll(model:any, midi_file:str,window:int,offset:int) -> tensor.Float:
# Calculate the negative log-likelihood of a MIDI file given a model. by using a sliding window approach.
# using cross-entropy loss.

# midi file 是以 mid 文件存储， 没有 .pt 文件，没有区分 mel或是acc，所以请参考 inference 中转换的代码
# mel 所在的位置是[1,3...] acc所在的位置[0,2...]


"""Utilities to compute negative log-likelihood (NLL) of MIDI tensors using a trained model.

Implemented functions:
- cal_nll(model, midi_file, window, offset) -> torch.FloatTensor

Assumptions made:
- `midi_file` points to an accompaniment tensor file (e.g. '..._acc.pt'). The corresponding
  melody file is found by replacing 'acc.pt' with 'mel.pt'. If that pattern is not present,
  the function will try a few reasonable alternatives and raise if the melody file cannot be found.
- The .pt files contain tensors indexed by frame (shape [num_frames, subseq_len]). The function
  slices windows of frames and forms a batch of size 1 for evaluation.
- Returned value: total negative log-likelihood (sum of per-token negative log-probabilities)
  across all processed windows, as a torch.FloatTensor. Also stable when model is on GPU.

Notes:
- Uses model.preprocess(...) to compute which target positions are PAD and to count the
  number of non-ignored tokens for proper weighting of the per-window mean loss returned
  by model.loss(...).
"""

from typing import Optional
import os
import json
import torch
from typing import Dict, Any
from preprocess.preprocess_midi2pt_dataset import preprocess_midi
from m2a_transformer import RoFormerSymbolicTransformer


def _find_mel_path(acc_path: str) -> Optional[str]:
    """Try to derive melody (.pt) path from accompaniment path.

    Returns the melody path string if found on disk, otherwise None.
    """
    # common pattern: replace 'acc.pt' with 'mel.pt'
    candidates = []
    if acc_path.endswith("acc.pt"):
        candidates.append(acc_path.replace("acc.pt", "mel.pt"))
    # replace suffix
    candidates.append(acc_path.replace(".pt", "_mel.pt"))
    candidates.append(acc_path.replace(".pt", "-mel.pt"))
    candidates.append(acc_path.replace(".pt", ".mel.pt"))
    # same file name (in case the file contains both) — allow it but check shape later
    candidates.append(acc_path)

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def cal_nll(model: object, midi_file: str, window: int, offset: int) -> torch.FloatTensor:
    """Calculate the negative log-likelihood (NLL) of a MIDI file using a sliding window.

    Parameters:
    - model: a RoFormerSymbolicTransformer-like model with `preprocess` and `loss` methods.
    - midi_file: path to accompaniment .pt file. The function will look for the corresponding
                             melody file by replacing 'acc.pt' with 'mel.pt' (and a few other patterns).
    - window: number of frames per window (target_length in dataset terms).
    - offset: sliding step between windows (stride).

    Returns:
    - dict: contains the following keys (all torch.Tensors on CPU):
        - 'total_nll': total negative log-likelihood (sum over non-ignored tokens)
        - 'avg_nll': average NLL (total_nll / total_tokens)
        - 'total_tokens': number of non-ignored tokens (int tensor)

    Raises:
    - ValueError if the files cannot be loaded or are shorter than window.
    """

    device = None
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    # Support raw MIDI (.mid) input: preprocess and split into acc/mel slots.
    if not os.path.exists(midi_file):
        raise ValueError(f"MIDI file not found: {midi_file}")

    acc = None
    mel = None

    if midi_file.lower().endswith('.mid'):
        # Preprocess the single .mid file and split instrument slots into accompaniment and melody
        # according to slot parity: acc -> slots 0,2,4...; mel -> slots 1,3,5...
        pre = preprocess_midi(midi_file, max_polyphony=4)
        if pre is None:
            raise ValueError(f"preprocess_midi returned None for {midi_file}")
        data, _shift = pre
        # data: Tensor [T, 3 * num_slots]
        T, C = data.shape
        num_slots = C // 3
        acc_slots = [i for i in range(num_slots) if i % 2 == 0]
        mel_slots = [i for i in range(num_slots) if i % 2 == 1]
        if len(acc_slots) == 0 or len(mel_slots) == 0:
            raise ValueError(f"Unable to split slots into acc/mel for {midi_file}; num_slots={num_slots}")

        acc_cols = []
        mel_cols = []
        for s in acc_slots:
            acc_cols += [s * 3 + 0, s * 3 + 1, s * 3 + 2]
        for s in mel_slots:
            mel_cols += [s * 3 + 0, s * 3 + 1, s * 3 + 2]

        acc = data[:, acc_cols]
        mel = data[:, mel_cols]
    else:
        # Fallback: assume accompaniment and melody are stored as .pt files (acc + mel)
        if not os.path.exists(midi_file):
            raise ValueError(f"Accompaniment file not found: {midi_file}")

        mel_path = _find_mel_path(midi_file)
        if mel_path is None:
            raise ValueError(f"Couldn't locate melody file corresponding to {midi_file}")

        # Load tensors (use mmap when available to save memory)
        acc = torch.load(midi_file, mmap_mode=True)
        mel = torch.load(mel_path, mmap_mode=True)

    # Basic sanity checks
    if acc.shape[0] != mel.shape[0]:
        # Some datasets may be off-by-one; require at least min length
        min_len = min(acc.shape[0], mel.shape[0])
        acc = acc[:min_len]
        mel = mel[:min_len]

    total_frames = acc.shape[0]
    if total_frames < window:
        raise ValueError(f"MIDI file shorter than window: frames={total_frames}, window={window}")

    model_device = device
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    # Slide windows with stride `offset` starting at 0
    for start in range(0, total_frames - window + 1, offset):
        end = start + window
        acc_seg = acc[start:end]
        mel_seg = mel[start:end]

        # Add batch dim
        acc_b = acc_seg.unsqueeze(0).to(model_device)
        mel_b = mel_seg.unsqueeze(0).to(model_device)

        # For pitch_shift we use zero (no shift) by default
        pitch_shift = torch.zeros((1,), dtype=torch.long, device=model_device)

        with torch.no_grad():
            # Use model.preprocess to determine how many non-pad target tokens exist
            try:
                mel_proc, acc_proc = model.preprocess(mel_b, pitch_shift, y=acc_b)
            except TypeError:
                # Some models might have different preprocess signature; try without y kw
                mel_proc, acc_proc = model.preprocess(mel_b, pitch_shift, acc_b)

            # Build x_target the same way as model.loss to count non-PAD tokens
            batch_size, seq_len, subseq_len = mel_proc.shape
            stacked = torch.stack([acc_proc, mel_proc], dim=2)
            x = stacked.view(batch_size, seq_len * 2, subseq_len)

            idx = torch.arange(seq_len * 2, device=x.device)
            mel_mask = (idx % 2 == 1).unsqueeze(0).unsqueeze(-1)
            mel_mask = mel_mask.expand(batch_size, seq_len * 2, subseq_len)
            x_target = x.clone()
            x_target[mel_mask] = (
                model.__class__.__dict__.get("PAD_TOKEN", None) or getattr(model, "PAD_TOKEN", None) or 3202
            )

            # Count non-ignored tokens
            nonpad = (x_target != (getattr(model, "PAD_TOKEN", 3202))).sum().item()

            if nonpad == 0:
                continue

            loss_mean = model.loss(mel_b, acc_b, pitch_shift)

            # model.loss returns mean per non-ignored token; multiply to get sum NLL for window
            total_nll += float(loss_mean.item()) * nonpad
            total_tokens += nonpad

    if total_tokens == 0:
        return {
            'total_nll': torch.tensor(float('nan'), dtype=torch.float32),
            'avg_nll': torch.tensor(float('nan'), dtype=torch.float32),
            'total_tokens': torch.tensor(0, dtype=torch.long),
        }

    # Return total and average negative log-likelihood as CPU tensors
    total_nll_t = torch.tensor(total_nll, dtype=torch.float32)
    avg_nll_t = torch.tensor(total_nll / float(total_tokens), dtype=torch.float32)
    tokens_t = torch.tensor(total_tokens, dtype=torch.long)
    return {
        'total_nll': total_nll_t,
        'avg_nll': avg_nll_t,
        'total_tokens': tokens_t,
    }


# 2===
# cal_nll_dir(model:any, midi_dir:str,window:int,offset:int) -> dict[str,tensor.Float]:
# Calculate the negative log-likelihoods of all MIDI files in a directory given a model. using a sliding window approach.
# using cross-entropy loss.

# make_nll_dir(model:any, midi_dir:str, window:int, offset:int, save_path:str) -> None:
# Calculate the negative log-likelihoods of all MIDI files in a directory given a model and save the results to a file.
# using a sliding window approach and cross-entropy loss.
# The results are saved as a JSON file.

def cal_nll_dir(model: object, midi_dir: str, window: int, offset: int) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute NLLs for all MIDI/.pt files in a directory.

    Returns a dict mapping filename -> result-dict (as returned by `cal_nll`).
    Files considered: .mid/.MID, .pt. Non-matching files are skipped.
    """
    if not os.path.exists(midi_dir):
        raise ValueError(f"Directory not found: {midi_dir}")

    results: Dict[str, Dict[str, torch.Tensor]] = {}
    for fname in sorted(os.listdir(midi_dir)):
        path = os.path.join(midi_dir, fname)
        if os.path.isdir(path):
            continue
        if not (fname.lower().endswith('.mid') or fname.lower().endswith('.pt')):
            continue
        try:
            res = cal_nll(model, path, window, offset)
            results[fname] = res
        except Exception as e:
            # store error string instead of tensor results for traceability
            results[fname] = {'error': str(e)}
    return results


def make_nll_dir(model: object, midi_dir: str, window: int, offset: int, save_path: str) -> None:
    """Compute NLLs for a directory and save results to JSON at save_path.

    The JSON will map filenames to a dict with keys 'total_nll', 'avg_nll', 'total_tokens',
    or to an 'error' string when processing failed.
    """
    results = cal_nll_dir(model, midi_dir, window, offset)

    # Convert tensors to native types for JSON serialization
    serializable: Dict[str, Any] = {}
    for fname, val in results.items():
        if isinstance(val, dict) and 'error' in val:
            serializable[fname] = {'error': val['error']}
            continue
        try:
            serializable[fname] = {
                'total_nll': float(val['total_nll'].cpu().item()),
                'avg_nll': float(val['avg_nll'].cpu().item()),
                'total_tokens': int(val['total_tokens'].cpu().item()),
            }
        except Exception:
            # fallback: try to stringify
            serializable[fname] = {k: str(v) for k, v in val.items()}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

# 3 ===
# start_call_nll_dir(ckpt_path:str, midi_dir:str, window:int, offset:int, save_path:str) -> None:
# Load model from checkpoint and compute NLLs for all MIDI files in a directory, saving results to a JSON file.


def start_call_nll_dir(
    ckpt_path: str,
    midi_dir: str,
    window: int,
    offset: int,
    save_path: str,
    model_size: str = '0.12B',
    device: str | None = None,
) -> None:
    """Load a model from a checkpoint and run make_nll_dir.

    Parameters:
    - ckpt_path: path to a PyTorch Lightning checkpoint compatible with RoFormerSymbolicTransformer
    - midi_dir: directory containing .mid or .pt files to process
    - window, offset: sliding window params passed to cal_nll/make_nll_dir
    - save_path: path to write JSON results
    - model_size: model size string passed to the model loader
    - device: device string (e.g. 'cuda:0' or 'cpu'); if None, auto-selects CUDA if available
    """

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load model via LightningModule classmethod
    model = RoFormerSymbolicTransformer.load_from_checkpoint(ckpt_path, model_size=model_size, map_location=device)
    model.save_name = os.path.basename(ckpt_path)
    model.to(device)
    model.eval()

    # Compute and save NLLs
    make_nll_dir(model, midi_dir, window, offset, save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute NLLs for a directory of MID/PT files using a model checkpoint')
    parser.add_argument('--midi_dir', type=str, default='~/stanleyz/StreamMUSE/experiments/realtime/prompt_75_gen_384/generated_without_prompt', help='Directory containing .mid or .pt files')
    parser.add_argument('--ckpt_path', type=str, default='/home/ubuntu/stanleyz/shared_models/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt', help='Path to model checkpoint')
    parser.add_argument('--save_json_path', type=str, default='records/nll_results.json', help='Output JSON path')
    parser.add_argument('--window', type=int, default=384, help='Sliding window length in frames')
    parser.add_argument('--offset', type=int, default=128, help='Sliding window offset/stride')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help="Device string, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument('--model_size', type=str, default='0.12B', help='Model size string passed to loader')

    args = parser.parse_args()

    midi_dir = os.path.expanduser(args.midi_dir)
    ckpt_path = os.path.expanduser(args.ckpt_path)
    save_json_path = os.path.expanduser(args.save_json_path)

    print(f"Loading checkpoint: {ckpt_path} on device {args.device}")
    print(f"Processing directory: {midi_dir} -> saving to {save_json_path}")

    start_call_nll_dir(ckpt_path=ckpt_path, midi_dir=midi_dir, window=args.window, offset=args.offset, save_path=save_json_path, model_size=args.model_size, device=args.device)

# ckpt_path /home/ubuntu/stanleyz/shared_models/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch\=00.val_loss\=0.90296.ckpt
# save_json_path records/nll_results.json
# window 384
# offset 128
# device 'cuda:0'


# python3 cal_nll.py \
#   --midi_dir ~/stanleyz/StreamMUSE/experiments/realtime/prompt_75_gen_384/generated_without_prompt \
#   --ckpt_path '/home/ubuntu/ugrip/shared_models/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt' \
#   --save_json_path records/nll_results.json \
#   --window 128 \
#   --offset 32 \
#   --device 'cuda:0' \
#   --model_size '0.12B'