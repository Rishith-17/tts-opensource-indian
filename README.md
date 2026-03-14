# 🎙️ Multilingual TTS — AI Voice Studio

A high-quality, local Text-to-Speech system for Indian languages with a modern React UI and FastAPI backend.

## Supported Languages

| Language | Engine | Quality |
|----------|--------|---------|
| Kannada (ಕನ್ನಡ) | IndicTTS (FastPitch + HiFiGAN) | High |
| Hindi (हिंदी) | Facebook MMS TTS | High |
| Tamil (தமிழ்) | IndicTTS (FastPitch + HiFiGAN) | High |
| Telugu (తెలుగు) | IndicTTS (FastPitch + HiFiGAN) | High |
| English | Kokoro-82M (#1 open-source TTS) | Very High |

## Requirements

- Python 3.10 – 3.12
- Node.js 18+
- GPU recommended (NVIDIA, 4GB+ VRAM) — CPU works but slower

## Quick Install

```
git clone https://github.com/YOUR_USERNAME/multilingual-tts.git
cd multilingual-tts
install.bat
```

That's it. `install.bat` will:
1. Install all Python dependencies
2. Install React frontend dependencies
3. Download TTS models (~2GB total)

## Start the App

```
start.bat
```

Opens two windows (API server + React UI) and launches `http://localhost:5173` in your browser.

## Manual Start (if needed)

**Terminal 1 — API server:**
```
python api_server.py
```

**Terminal 2 — React UI:**
```
cd assistant_project\frontend\tts-ui
npm run dev
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Returns WAV audio file |
| POST | `/tts/base64` | Returns base64-encoded WAV in JSON |
| GET | `/languages` | List supported languages and voices |
| GET | `/health` | API health check |
| GET | `/docs` | Swagger UI |

**Example request:**
```bash
curl -X POST http://localhost:8000/tts/base64 \
  -H "Content-Type: application/json" \
  -d '{"text": "ನಮಸ್ಕಾರ", "language": "kn", "speaker": "female"}'
```

**Voice options:** `female` · `male` · `female_slow` · `male_slow`

## Project Structure

```
multilingual-tts/
├── api_server.py              # FastAPI backend
├── tts.py                     # IndicTTS engine (kn, ta, te, en)
├── tts_mms.py                 # MMS TTS engine (hi)
├── tts_kokoro_en.py           # Kokoro-82M engine (en)
├── language_router.py         # Language detection + model routing
├── setup_models.py            # One-time model downloader
├── install.bat                # One-command installer (Windows)
├── start.bat                  # Launch everything
├── requirements.txt
├── models/                    # Downloaded TTS models (gitignored)
│   ├── kn/  hi/  en/  ta/  te/
└── assistant_project/
    ├── frontend/
    │   ├── app.py             # Legacy Streamlit UI (optional)
    │   └── tts-ui/            # React + Vite frontend
    ├── llm/sarvam_client.py   # Sarvam AI integration
    ├── speech/whisper_stt.py  # Whisper STT
    ├── language/              # Language detection
    ├── tts/multilingual_tts.py
    └── translation/           # IndicTrans2 translation
```

## Notes

- Models are **not included** in the repo (too large). `install.bat` downloads them automatically.
- `lid.176.bin` (fastText model) is also downloaded automatically.
- First run of MMS (Hindi) and Kokoro (English) will download their models from HuggingFace automatically.
- API key for Sarvam AI is set in `assistant_project/llm/sarvam_client.py` — replace with your own from [sarvam.ai](https://sarvam.ai).
