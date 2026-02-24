"""Core module for cal_nll computing negative log-likelihood."""

from typing import Optional
import os
import torch
from preprocess.preprocess_midi2pt_dataset import preprocess_midi
from m2a_transformer import RoFormerSymbolicTransformer


def _find_mel_path(acc_path: str) -> Optional[str]:
    """Try to derive melody (.pt) path from accompaniment path.

    Returns the melody path string if found on disk, otherwise None.
    """
    candidates = []
    if acc_path.endswith("acc.pt"):
        candidates.append(acc_path.replace("acc.pt", "mel.pt"))
    candidates.append(acc_path.replace(".pt", "_mel.pt"))
    candidates.append(acc_path.replace(".pt", "-mel.pt"))
    candidates.append(acc_path.replace(".pt", ".mel.pt"))
    candidates.append(acc_path)

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def cal_nll(model: object, midi_file: str, window: int, offset: int) -> torch.FloatTensor:
    """Calculate the negative log-likelihood (NLL) of a MIDI file using a sliding window.

    Returns:
    - dict: contains the following keys (all torch.Tensors on CPU):
        - 'total_nll': total negative log-likelihood (sum over non-ignored tokens)
        - 'avg_nll': average NLL (total_nll / total_tokens)
        - 'total_tokens': number of non-ignored tokens (int tensor)
    """

    device = None
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    if not os.path.exists(midi_file):
        raise ValueError(f"MIDI file not found: {midi_file}")

    acc = None
    mel = None

    if midi_file.lower().endswith(".mid"):
        pre = preprocess_midi(midi_file, max_polyphony=4)
        if pre is None:
            raise ValueError(f"preprocess_midi returned None for {midi_file}")
        data, _shift = pre
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
        if not os.path.exists(midi_file):
            raise ValueError(f"Accompaniment file not found: {midi_file}")

        mel_path = _find_mel_path(midi_file)
        if mel_path is None:
            raise ValueError(f"Couldn't locate melody file corresponding to {midi_file}")

        acc = torch.load(midi_file, mmap_mode=True)
        mel = torch.load(mel_path, mmap_mode=True)

    if acc.shape[0] != mel.shape[0]:
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

    for start in range(0, total_frames - window + 1, offset):
        end = start + window
        acc_seg = acc[start:end]
        mel_seg = mel[start:end]

        acc_b = acc_seg.unsqueeze(0).to(model_device)
        mel_b = mel_seg.unsqueeze(0).to(model_device)

        pitch_shift = torch.zeros((1,), dtype=torch.long, device=model_device)

        with torch.no_grad():
            try:
                mel_proc, acc_proc = model.preprocess(mel_b, pitch_shift, y=acc_b)
            except TypeError:
                mel_proc, acc_proc = model.preprocess(mel_b, pitch_shift, acc_b)

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

            nonpad = (x_target != (getattr(model, "PAD_TOKEN", 3202))).sum().item()

            if nonpad == 0:
                continue

            loss_mean = model.loss(mel_b, acc_b, pitch_shift)

            total_nll += float(loss_mean.item()) * nonpad
            total_tokens += nonpad

    if total_tokens == 0:
        return {
            "total_nll": torch.tensor(float("nan"), dtype=torch.float32),
            "avg_nll": torch.tensor(float("nan"), dtype=torch.float32),
            "total_tokens": torch.tensor(0, dtype=torch.long),
        }

    total_nll_t = torch.tensor(total_nll, dtype=torch.float32)
    avg_nll_t = torch.tensor(total_nll / float(total_tokens), dtype=torch.float32)
    tokens_t = torch.tensor(total_tokens, dtype=torch.long)
    return {
        "total_nll": total_nll_t,
        "avg_nll": avg_nll_t,
        "total_tokens": tokens_t,
    }
