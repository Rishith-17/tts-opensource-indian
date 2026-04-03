"""
Download and prepare IndicTTS datasets for finetuning FastPitch.
Source: SPRINGLab collection on HuggingFace (IIT Madras IndicTTS corpus)

Usage:
    python finetune/prepare_dataset.py --lang kn
    python finetune/prepare_dataset.py --lang all
"""
import os, sys, csv, argparse
import numpy as np
import soundfile as sf
from scipy import signal

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_SR  = 22050

# Correct HuggingFace repo names (SPRINGLab collection)
REPOS = {
    "kn": "SPRINGLab/IndicTTS_Kannada",
    "hi": "SPRINGLab/IndicTTS-Hindi",
    "ta": "SPRINGLab/IndicTTS_Tamil",
    "te": "SPRINGLab/IndicTTS_Telugu",
    "en": "SPRINGLab/IndicTTS-English",
}

def resample(audio, orig_sr):
    if orig_sr == TARGET_SR:
        return audio.astype(np.float32)
    n = int(len(audio) * TARGET_SR / orig_sr)
    return signal.resample(audio, n).astype(np.float32)

def prepare(lang):
    from datasets import load_dataset, Audio

    repo = REPOS[lang]
    out  = os.path.join(ROOT, "finetune", "data", lang)
    wavs = os.path.join(out, "wavs")
    os.makedirs(wavs, exist_ok=True)

    # Use HF default cache — resumes automatically, never re-downloads completed shards
    print(f"\n[{lang.upper()}] Downloading {repo} (auto-resumes from HF cache)...")
    # Load WITHOUT audio decoding first, then decode manually with soundfile
    ds = load_dataset(repo, split="train", cache_dir=None)
    # Cast audio column to use soundfile decoder (avoids torchcodec dependency)
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"  {len(ds)} samples")

    # Detect text field name
    text_key = None
    for k in ("text", "sentence", "transcript", "transcription"):
        if k in ds.features:
            text_key = k
            break
    if text_key is None:
        # fallback: first string field that isn't 'id'
        for k, v in ds.features.items():
            if str(v) == "Value(dtype='string', id=None)" and k != "id":
                text_key = k
                break
    print(f"  Text field: '{text_key}'")

    meta = os.path.join(out, "metadata.csv")
    count = 0
    with open(meta, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for i, sample in enumerate(ds):
            try:
                text = sample[text_key].strip()
                if not text or len(text) < 3:
                    continue

                # Audio is raw bytes (decode=False) — decode with soundfile
                audio_info = sample["audio"]
                raw_bytes = audio_info.get("bytes") or open(audio_info["path"], "rb").read()
                import io
                wav, orig_sr = sf.read(io.BytesIO(raw_bytes))
                wav = wav.astype(np.float32)
                if wav.ndim > 1:          # stereo → mono
                    wav = wav.mean(axis=1)
                wav = resample(wav, orig_sr)
                # Skip clips shorter than 0.3s or longer than 15s
                dur = len(wav) / TARGET_SR
                if dur < 0.3 or dur > 15:
                    continue
                name = f"{lang}_{i:05d}"
                sf.write(os.path.join(wavs, f"{name}.wav"), wav, TARGET_SR)
                writer.writerow([name, text, text])
                count += 1
                if count % 200 == 0:
                    print(f"  {count} samples saved...")
            except Exception:
                continue

    print(f"✓ [{lang.upper()}] {count} samples → {out}")
    return meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="kn",
                        help="kn | hi | ta | te | en | all")
    args = parser.parse_args()

    langs = list(REPOS.keys()) if args.lang == "all" else [args.lang]
    for lang in langs:
        prepare(lang)

    print("\n✓ Dataset ready. Next: python finetune/train.py --lang kn")
