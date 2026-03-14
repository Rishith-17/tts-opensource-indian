"""
Download all required TTS models.
Run once after cloning: python setup_models.py

Models downloaded:
  - IndicTTS (Kannada, Tamil, Telugu, English) from AI4Bharat
  - MMS TTS (Hindi) from Facebook via HuggingFace — auto-downloaded on first use
  - Kokoro-82M (English) from HuggingFace — auto-downloaded on first use
  - fastText language detection model
"""
import os
import urllib.request

# ── fastText language detection model ────────────────────────────────────────
def download_fasttext():
    path = "lid.176.bin"
    if os.path.exists(path):
        print("✓ fastText model already exists")
        return
    print("Downloading fastText language detection model (~130MB)...")
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    urllib.request.urlretrieve(url, path, reporthook=_progress)
    print(f"\n✓ Saved to {path}")

# ── IndicTTS models ───────────────────────────────────────────────────────────
def download_indic_tts():
    """Download IndicTTS FastPitch + HiFiGAN models from AI4Bharat."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub -q")
        from huggingface_hub import snapshot_download

    languages = {
        "kn": "ai4bharat/indic-tts-coqui-kn",
        "ta": "ai4bharat/indic-tts-coqui-ta",
        "te": "ai4bharat/indic-tts-coqui-te",
        "en": "ai4bharat/indic-tts-coqui-en",
    }

    for lang, repo in languages.items():
        dest = os.path.join("models", lang)
        if os.path.exists(os.path.join(dest, "fastpitch", "best_model.pth")):
            print(f"✓ {lang} model already exists")
            continue
        print(f"Downloading {lang} model from {repo}...")
        try:
            snapshot_download(repo_id=repo, local_dir=dest)
            print(f"✓ {lang} model saved to {dest}")
        except Exception as e:
            print(f"✗ Failed to download {lang}: {e}")
            print(f"  Manual download: https://huggingface.co/{repo}")

def _progress(count, block_size, total_size):
    pct = int(count * block_size * 100 / total_size)
    print(f"\r  {pct}%", end="", flush=True)

if __name__ == "__main__":
    print("=" * 50)
    print("  Multilingual TTS — Model Setup")
    print("=" * 50)
    print()
    download_fasttext()
    print()
    download_indic_tts()
    print()
    print("MMS TTS (Hindi) and Kokoro (English) download automatically on first use.")
    print()
    print("✓ Setup complete! Run: start.bat")
