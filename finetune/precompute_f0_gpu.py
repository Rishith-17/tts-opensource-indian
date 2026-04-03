"""
GPU-accelerated F0 precompute using torchcrepe (replaces slow CPU pyin).
Runs on RTX 4050 — ~10-20x faster than librosa pyin.

Usage:
    python finetune/precompute_f0_gpu.py --lang kn
"""
import os, sys, argparse
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 22050

def compute_f0_gpu(wav_path, fmin=65.0, fmax=640.0, hop_length=256):
    """Compute F0 using torchcrepe on GPU."""
    import torchcrepe

    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    # Resample if needed
    if sr != TARGET_SR:
        from scipy import signal as sig
        wav = sig.resample(wav, int(len(wav) * TARGET_SR / sr)).astype(np.float32)

    audio = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)  # [1, T]

    # torchcrepe expects 16kHz — resample internally
    f0, periodicity = torchcrepe.predict(
        audio,
        TARGET_SR,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        model="tiny",       # fast model
        batch_size=512,
        device=DEVICE,
        return_periodicity=True,
    )

    # Silence unvoiced frames
    f0 = torchcrepe.filter.median(f0, 3)
    f0 = torchcrepe.threshold.At(0.21)(f0, periodicity)
    f0 = f0.squeeze().cpu().numpy()
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.astype(np.float32)

def main(lang):
    data_dir  = os.path.join(ROOT, "finetune", "data", lang)
    cache_dir = os.path.join(ROOT, "finetune", "cache", lang, "f0_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Read metadata
    meta_path = os.path.join(data_dir, "metadata.csv")
    items = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 1:
                name = parts[0]
                wav  = os.path.join(data_dir, "wavs", name + ".wav")
                if os.path.exists(wav):
                    items.append((name, wav))

    print(f"Device: {DEVICE}")
    print(f"Total samples: {len(items)}")

    # Count already done
    done = {f[:-4] for f in os.listdir(cache_dir) if f.endswith(".npy")}
    todo = [(n, w) for n, w in items if n not in done]
    print(f"Already cached: {len(done)}  |  Remaining: {len(todo)}")

    if not todo:
        print("✓ All F0s already computed!")
        _save_stats(cache_dir, items, data_dir)
        return

    errors = 0
    for name, wav_path in tqdm(todo, desc="F0 GPU"):
        try:
            f0 = compute_f0_gpu(wav_path)
            np.save(os.path.join(cache_dir, name + ".npy"), f0)
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"\n  Error on {name}: {e}")

    print(f"\n✓ Done. Errors: {errors}")
    _save_stats(cache_dir, items, data_dir)

def _save_stats(cache_dir, items, data_dir):
    """Compute and save pitch_stats.npy (mean/std) — required by TTS trainer."""
    print("Computing pitch statistics...")
    all_f0 = []
    for name, _ in items:
        p = os.path.join(cache_dir, name + ".npy")
        if os.path.exists(p):
            f0 = np.load(p)
            voiced = f0[f0 > 0]
            if len(voiced) > 0:
                all_f0.extend(voiced.tolist())

    if all_f0:
        mean = float(np.mean(all_f0))
        std  = float(np.std(all_f0))
        stats = {"mean": np.array(mean, dtype=np.float32),
                 "std":  np.array(std,  dtype=np.float32)}
        np.save(os.path.join(cache_dir, "pitch_stats.npy"), stats)
        print(f"✓ pitch_stats.npy saved  (mean={mean:.1f} Hz, std={std:.1f})")
    else:
        print("✗ No voiced frames found — check audio files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="kn")
    args = parser.parse_args()
    main(args.lang)
