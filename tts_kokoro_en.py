"""
English TTS using Kokoro-82M — #1 ranked open-source TTS model.
82M params, Apache licensed, runs on RTX 4050 (< 2GB VRAM), 24kHz output.

Install:
    pip install kokoro soundfile

Voices available:
    female      → af_sarah   (American Female, clear & natural)
    male        → am_michael (American Male, deep & clear)
    female_slow → af_bella   (American Female, slightly slower)
    male_slow   → bm_george  (British Male, slower & articulate)
"""
import warnings
warnings.filterwarnings("ignore", message="dropout option adds dropout")
warnings.filterwarnings("ignore", message=".*weight_norm.*is deprecated")

import numpy as np
import soundfile as sf
from scipy import signal

SAMPLE_RATE = 24000  # Kokoro native sample rate

# Voice mapping: our standard speaker names → Kokoro voice IDs
VOICE_MAP = {
    "female":      "af_sarah",
    "male":        "am_michael",   # deep natural American male
    "female_slow": "af_bella",
    "male_slow":   "bm_george",    # British male, slower and articulate
}


class KokoroEnglishTTS:
    """High-quality English TTS using Kokoro-82M."""

    def __init__(self):
        print("Loading Kokoro-82M English TTS...")
        try:
            from kokoro import KPipeline
            # lang_code "a" = American English
            self.pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
            self.available = True
            print("✓ Kokoro-82M loaded (English, 24kHz, 54 voices)")
        except ImportError:
            print("✗ kokoro not installed. Run: pip install kokoro")
            self.pipe = None
            self.available = False
        except Exception as e:
            print(f"✗ Kokoro load failed: {e}")
            self.pipe = None
            self.available = False

    def generate_speech(self, text: str, speaker: str = "female") -> np.ndarray:
        """
        Generate English speech.

        Args:
            text:    Input English text
            speaker: female | male | female_slow | male_slow

        Returns:
            numpy float32 waveform at 24kHz
        """
        if not self.available:
            raise RuntimeError("Kokoro not available. Run: pip install kokoro")

        voice = VOICE_MAP.get(speaker, "af_sarah")
        print(f"Kokoro TTS | voice={voice} | text={text[:60]}...")

        chunks = []
        for _, _, chunk in self.pipe(text, voice=voice):
            chunks.append(chunk)

        if not chunks:
            raise RuntimeError("Kokoro returned empty audio")

        return np.concatenate(chunks).astype(np.float32)

    def save_audio(self, wav: np.ndarray, output_path: str = "output.wav"):
        """Save with light post-processing (DC removal, normalize, fade)."""
        wav = np.array(wav, dtype=np.float32)

        # Remove DC offset
        wav = wav - np.mean(wav)

        # Normalize to 85%
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * 0.85

        # Fade in/out (10ms) to avoid clicks
        fade = int(SAMPLE_RATE * 0.01)
        if len(wav) > fade * 2:
            wav[:fade]  *= np.linspace(0, 1, fade)
            wav[-fade:] *= np.linspace(1, 0, fade)

        sf.write(output_path, wav, SAMPLE_RATE, subtype="PCM_16")
        print(f"Audio saved: {output_path}")
