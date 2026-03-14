"""
FastAPI server for Multilingual TTS.
Endpoints:
  POST /tts          → returns WAV audio file
  POST /tts/base64   → returns base64-encoded WAV
  GET  /languages    → list supported languages
  GET  /health       → health check
  GET  /docs         → Swagger UI (auto-generated)

Run:
  pip install fastapi uvicorn
  python api_server.py
"""
import os
import sys
import base64
import tempfile

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

# ── Path setup ────────────────────────────────────────────────────────────────
root = os.path.abspath(os.path.dirname(__file__))
assistant_dir = os.path.join(root, "assistant_project")
sys.path.insert(0, root)
sys.path.insert(0, assistant_dir)

# Load MultilingualTTS directly from file to avoid conflict with root tts.py
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "multilingual_tts",
    os.path.join(assistant_dir, "tts", "multilingual_tts.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MultilingualTTS = _mod.MultilingualTTS

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multilingual TTS API",
    description="Text-to-Speech for Kannada, Hindi, Tamil, Telugu, English",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models (loaded once at startup) ───────────────────────────────────────────
tts: MultilingualTTS = None

@app.on_event("startup")
async def startup():
    global tts
    print("Loading TTS models...")
    tts = MultilingualTTS()
    print("✓ TTS ready")

# ── Schemas ───────────────────────────────────────────────────────────────────
SUPPORTED_LANGS = ["kn", "hi", "en", "ta", "te"]
SUPPORTED_SPEAKERS = ["female", "male", "female_slow", "male_slow"]

class TTSRequest(BaseModel):
    text: str
    language: str = "kn"
    speaker: str = "female"

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "ನಮಸ್ಕಾರ, ನಾನು ನಿಮ್ಮ ಸಹಾಯಕ",
                "language": "kn",
                "speaker": "female"
            }
        }
    }

# ── Helpers ───────────────────────────────────────────────────────────────────
def validate_request(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    if req.language not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"language '{req.language}' not supported. Use one of: {SUPPORTED_LANGS}"
        )
    if req.speaker not in SUPPORTED_SPEAKERS:
        raise HTTPException(
            status_code=400,
            detail=f"speaker '{req.speaker}' not supported. Use one of: {SUPPORTED_SPEAKERS}"
        )

def synthesize(req: TTSRequest) -> str:
    """Run TTS and return path to WAV file."""
    if tts is None:
        raise HTTPException(status_code=503, detail="TTS models not loaded yet")
    validate_request(req)

    # Write to a temp file so concurrent requests don't collide
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    path = tts.synthesize(req.text, req.language, speaker=req.speaker, output_path=tmp.name)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=500, detail="TTS generation failed")
    return path

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "tts_loaded": tts is not None}

@app.get("/languages")
def languages():
    return {
        "languages": [
            {"code": "kn", "name": "Kannada",  "engine": "Indic TTS"},
            {"code": "hi", "name": "Hindi",    "engine": "MMS TTS"},
            {"code": "en", "name": "English",  "engine": "Indic TTS"},
            {"code": "ta", "name": "Tamil",    "engine": "Indic TTS"},
            {"code": "te", "name": "Telugu",   "engine": "Indic TTS"},
        ],
        "speakers": SUPPORTED_SPEAKERS,
    }

@app.post("/tts", summary="Text to Speech — returns WAV file")
def text_to_speech(req: TTSRequest):
    """
    Convert text to speech. Returns a WAV audio file.

    - **text**: Input text in the target language
    - **language**: `kn` Kannada · `hi` Hindi · `en` English · `ta` Tamil · `te` Telugu
    - **speaker**: `female` · `male` · `female_slow` · `male_slow`
    """
    path = synthesize(req)
    return FileResponse(path, media_type="audio/wav", filename=f"tts_{req.language}.wav")

@app.post("/tts/base64", summary="Text to Speech — returns base64 JSON")
def text_to_speech_base64(req: TTSRequest):
    """
    Convert text to speech. Returns base64-encoded WAV in JSON.
    Useful for web/mobile clients that can't handle binary responses.
    """
    path = synthesize(req)
    with open(path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    os.unlink(path)  # clean up temp file
    return {
        "text": req.text,
        "language": req.language,
        "speaker": req.speaker,
        "format": "wav",
        "audio_base64": audio_b64,
    }

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Starting TTS API server...")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
