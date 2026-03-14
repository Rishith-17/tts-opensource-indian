"""
Multilingual TTS wrapper.
- Kokoro-82M:  English only  (best quality, #1 open-source TTS)
- MMS TTS:     Hindi only    (fast, accurate for Hindi)
- Indic TTS:   Kannada, Tamil, Telugu
"""
import sys
import os
import importlib.util

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

# ── Indic TTS (kn, ta, te) ────────────────────────────────────────────────────
INDIC_TTS_AVAILABLE = False
tts_file = os.path.join(parent_dir, 'tts.py')
if os.path.exists(tts_file):
    try:
        spec = importlib.util.spec_from_file_location("tts_module", tts_file)
        tts_module = importlib.util.module_from_spec(spec)
        sys.modules['tts_module'] = tts_module
        spec.loader.exec_module(tts_module)
        IndicTTS = tts_module.MultilingualTTS
        INDIC_TTS_AVAILABLE = True
        print("✓ Indic TTS loaded (Kannada, Tamil, Telugu)")
    except Exception as e:
        print(f"✗ Indic TTS load failed: {e}")

# ── MMS TTS (hi) ──────────────────────────────────────────────────────────────
MMS_TTS_AVAILABLE = False
tts_mms_file = os.path.join(parent_dir, 'tts_mms.py')
if os.path.exists(tts_mms_file):
    try:
        spec = importlib.util.spec_from_file_location("tts_mms_module", tts_mms_file)
        tts_mms_module = importlib.util.module_from_spec(spec)
        sys.modules['tts_mms_module'] = tts_mms_module
        spec.loader.exec_module(tts_mms_module)
        MMSTTS = tts_mms_module.MMSTTS
        MMS_TTS_AVAILABLE = True
        print("✓ MMS TTS loaded (Hindi only)")
    except Exception as e:
        print(f"⚠ MMS TTS load failed: {e}")

# ── Kokoro-82M (en) ───────────────────────────────────────────────────────────
KOKORO_TTS_AVAILABLE = False
kokoro_file = os.path.join(parent_dir, 'tts_kokoro_en.py')
if os.path.exists(kokoro_file):
    try:
        spec = importlib.util.spec_from_file_location("tts_kokoro_module", kokoro_file)
        kokoro_module = importlib.util.module_from_spec(spec)
        sys.modules['tts_kokoro_module'] = kokoro_module
        spec.loader.exec_module(kokoro_module)
        KokoroTTS = kokoro_module.KokoroEnglishTTS
        KOKORO_TTS_AVAILABLE = True
        print("✓ Kokoro-82M loaded (English only)")
    except Exception as e:
        print(f"⚠ Kokoro TTS load failed (English will fall back to Indic TTS): {e}")


class MultilingualTTS:
    """
    Routing:
      en → Kokoro-82M   (highest quality English)
      hi → MMS TTS      (accurate Hindi)
      kn/ta/te → Indic TTS
    """

    def __init__(self):
        self.indic_tts  = IndicTTS()  if INDIC_TTS_AVAILABLE  else None
        self.mms_tts    = MMSTTS()    if MMS_TTS_AVAILABLE    else None
        self.kokoro_tts = KokoroTTS() if KOKORO_TTS_AVAILABLE else None

        print("TTS routing: en→Kokoro | hi→MMS | kn/ta/te→Indic")

    def synthesize(self, text, language_code, speaker="female", output_path="response.wav"):
        """
        Synthesize speech. Returns output_path on success, None on failure.
        """
        try:
            # English → Kokoro-82M
            if language_code == 'en':
                if self.kokoro_tts is not None and self.kokoro_tts.available:
                    print("Using Kokoro-82M for English")
                    wav = self.kokoro_tts.generate_speech(text, speaker)
                    self.kokoro_tts.save_audio(wav, output_path)
                    return output_path
                # fallback to Indic TTS if Kokoro not installed
                print("⚠ Kokoro unavailable, falling back to Indic TTS for English")

            # Hindi → MMS TTS
            if language_code == 'hi' and self.mms_tts is not None:
                print("Using MMS TTS for Hindi")
                wav = self.mms_tts.generate_speech(text, 'hi', speaker)
                self.mms_tts.save_audio(wav, output_path)
                return output_path

            # Kannada / Tamil / Telugu → Indic TTS
            if self.indic_tts is not None:
                print(f"Using Indic TTS for {language_code}")
                wav = self.indic_tts.generate_speech(text, language_code, speaker)
                self.indic_tts.save_audio(wav, output_path)
                return output_path

            print("✗ No TTS engine available")
            return None

        except Exception as e:
            print(f"TTS error: {e}")
            import traceback
            traceback.print_exc()
            return None
