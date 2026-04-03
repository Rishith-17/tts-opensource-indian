"""
Customer Care AI Agent — FastAPI backend.
Handles text/voice queries, responds in user's language with TTS.
"""
import os, sys, base64, tempfile, importlib.util

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "assistant_project"))

# ── Load Sarvam client ────────────────────────────────────────────────────────
from llm.sarvam_client import SarvamClient

# ── Load TTS ──────────────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("multilingual_tts",
    os.path.join(ROOT, "assistant_project", "tts", "multilingual_tts.py"))
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
MultilingualTTS = _mod.MultilingualTTS

# ── Load Whisper STT ──────────────────────────────────────────────────────────
from speech.whisper_stt import WhisperSTT

# ── Load language detector ────────────────────────────────────────────────────
from language.detect_language import LanguageDetector

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Customer Care AI Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve React frontend
ui_dist = os.path.join(os.path.dirname(__file__), "ui", "dist")
if os.path.exists(ui_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(ui_dist, "assets")), name="assets")

# ── Customer care system prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are Arjun, a professional and warm customer care executive. You speak fluently in all Indian languages with perfect grammar.

🌐 LANGUAGE RULE — ABSOLUTE PRIORITY:
Respond ONLY in the exact script of the customer's message. Never mix languages.

📚 KANNADA GRAMMAR GUIDE (ಕನ್ನಡ):
Use proper Kannada grammar with correct verb endings and honorifics:
- Address customer as "ನೀವು" (formal you), never "ನೀನು"
- Use "ಅವರೇ" after names: "ರಾಜೇಶ್ ಅವರೇ"
- Correct verb forms: "ಮಾಡುತ್ತೇನೆ" (I will do), "ಮಾಡಬಹುದು" (can do), "ಮಾಡಿ" (please do)
- Natural phrases: "ಖಂಡಿತವಾಗಿ" (certainly), "ಅರ್ಥವಾಯಿತು" (understood), "ಚಿಂತಿಸಬೇಡಿ" (don't worry)
- Sentence structure: Subject + Object + Verb (SOV order)

Good Kannada examples:
✓ "ಖಂಡಿತವಾಗಿ ಸಹಾಯ ಮಾಡುತ್ತೇನೆ. ನಿಮ್ಮ ಸಮಸ್ಯೆ ಏನೆಂದು ವಿವರಿಸುತ್ತೀರಾ?"
✓ "ಅರ್ಥವಾಯಿತು ರಾಜೇಶ್ ಅವರೇ. ನಾನು ಈಗಲೇ ಪರಿಶೀಲಿಸುತ್ತೇನೆ."
✓ "ಚಿಂತಿಸಬೇಡಿ, ನಾವು ಇದನ್ನು ಸರಿಪಡಿಸುತ್ತೇವೆ."

📚 HINDI GRAMMAR GUIDE (हिंदी):
- Address as "आप" (formal), use "जी" after names: "राहुल जी"
- Natural phrases: "बिल्कुल" (absolutely), "समझ गया" (understood), "चिंता मत करिए" (don't worry)
- Correct verb forms: "करूँगा" (I will do), "कर सकते हैं" (can do)

Good Hindi examples:
✓ "बिल्कुल राहुल जी, मैं आपकी मदद करूँगा। आपकी समस्या क्या है?"
✓ "समझ गया। मैं अभी आपके ऑर्डर की जानकारी देखता हूँ।"
✓ "चिंता मत करिए, हम इसे ठीक कर देंगे।"

📚 TAMIL GRAMMAR GUIDE (தமிழ்):
- Address as "நீங்கள்" (formal), use "அவர்கள்" respectfully
- Natural phrases: "நிச்சயமாக" (certainly), "புரிந்தது" (understood), "கவலைப்படாதீர்கள்" (don't worry)

📚 TELUGU GRAMMAR GUIDE (తెలుగు):
- Address as "మీరు" (formal), use "గారు" after names
- Natural phrases: "తప్పకుండా" (certainly), "అర్థమైంది" (understood), "చింతించకండి" (don't worry)

🎯 RESPONSE RULES:
- Maximum 2-3 sentences — spoken aloud via voice
- Acknowledge → Help → Next step
- Use customer's name naturally (not every sentence)
- No markdown, no lists, no emojis in response text
- Sound like a real human agent, warm and professional

🚫 NEVER:
- Mix scripts or languages
- Use "As per our policy" or robotic phrases
- Give long paragraph answers
"""

# ── Global models ─────────────────────────────────────────────────────────────
models = {}

@app.on_event("startup")
async def startup():
    print("Loading Customer Care Agent models...")
    models["llm"]      = SarvamClient(model="sarvam-105b")
    models["llm"].conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    models["tts"]      = MultilingualTTS()
    models["stt"]      = WhisperSTT(model_size="small")
    models["detector"] = LanguageDetector()
    print("✓ All models ready")

# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None   # auto-detect if None
    voice: str = "female"
    session_id: str = "default"

class ChatResponse(BaseModel):
    user_message: str
    agent_response: str
    language: str
    audio_base64: Optional[str] = None

# Per-session conversation history
sessions: dict = {}

def get_client(session_id: str) -> SarvamClient:
    if session_id not in sessions:
        client = SarvamClient(model="sarvam-105b")
        client.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        sessions[session_id] = client
    return sessions[session_id]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    if os.path.exists(ui_dist):
        return FileResponse(os.path.join(ui_dist, "index.html"))
    return {"status": "Customer Care Agent running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "models": list(models.keys())}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    lang = req.language or models["detector"].detect(req.message)

    # Special greeting trigger
    if req.message == '__greeting__':
        greetings = {
            'kn': 'ನಮಸ್ಕಾರ! ನಾನು ಅರ್ಜುನ್, ನಿಮ್ಮ ಗ್ರಾಹಕ ಸೇವಾ ಪ್ರತಿನಿಧಿ. ನಿಮ್ಮ ಹೆಸರು ತಿಳಿಸುತ್ತೀರಾ?',
            'hi': 'नमस्ते! मैं अर्जुन हूँ, आपका कस्टमर केयर एग्जीक्यूटिव। क्या मैं आपका नाम जान सकता हूँ?',
            'ta': 'வணக்கம்! நான் அர்ஜுன், உங்கள் வாடிக்கையாளர் சேவை பிரதிநிதி. உங்கள் பெயர் சொல்ல முடியுமா?',
            'te': 'నమస్కారం! నేను అర్జున్, మీ కస్టమర్ కేర్ ఎగ్జిక్యూటివ్. మీ పేరు చెప్పగలరా?',
            'en': "Hi there! I'm Arjun, your customer care executive. Before we get started, could I get your name please?",
        }
        lang = req.language or 'en'
        response = greetings.get(lang, greetings['en'])
    else:
        print(f"Detected language: {lang} for: {req.message[:50]}")
        client = get_client(req.session_id)
        prev_lang = getattr(client, '_last_lang', None)
        if prev_lang and prev_lang != lang:
            print(f"Language switched {prev_lang} → {lang}, resetting history")
            client.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        client._last_lang = lang

        # First message — return hardcoded intro, don't call Sarvam
        is_first = not getattr(client, '_greeted', False)
        if is_first:
            client._greeted = True
            intros = {
                'kn': 'ನಮಸ್ಕಾರ! ನಾನು ಅರ್ಜುನ್, ನಿಮ್ಮ ಗ್ರಾಹಕ ಸೇವಾ ಪ್ರತಿನಿಧಿ. ನಿಮ್ಮ ಹೆಸರು ತಿಳಿಸುತ್ತೀರಾ?',
                'hi': 'नमस्ते! मैं अर्जुन हूँ, आपका कस्टमर केयर एग्जीक्यूटिव। क्या मैं आपका नाम जान सकता हूँ?',
                'ta': 'வணக்கம்! நான் அர்ஜுன், உங்கள் வாடிக்கையாளர் சேவை பிரதிநிதி. உங்கள் பெயர் சொல்ல முடியுமா?',
                'te': 'నమస్కారం! నేను అర్జున్, మీ కస్టమర్ కేర్ ఎగ్జిక్యూటివ్. మీ పేరు చెప్పగలరా?',
                'en': "Hi there! I'm Arjun, your customer care executive. Before we get started, could I get your name please?",
            }
            response = intros.get(lang, intros['en'])
        else:
            response = client.generate(req.message, language=lang)

    # Generate TTS audio
    audio_b64 = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        path = models["tts"].synthesize(response, lang, speaker="male", output_path=tmp.name)
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            os.unlink(path)
    except Exception as e:
        print(f"TTS error: {e}")

    return ChatResponse(
        user_message=req.message,
        agent_response=response,
        language=lang,
        audio_base64=audio_b64,
    )

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe voice input to text using soundfile (no ffmpeg needed)."""
    import numpy as np
    import soundfile as sf
    import io

    data = await file.read()

    # Decode audio with soundfile (no ffmpeg dependency)
    try:
        audio_np, sr = sf.read(io.BytesIO(data))
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        audio_np = audio_np.astype(np.float32)
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sr != 16000:
            from scipy import signal as sig
            audio_np = sig.resample(audio_np, int(len(audio_np) * 16000 / sr)).astype(np.float32)
    except Exception:
        # Fallback: try writing to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(data)
        tmp.close()
        audio_np = None

    # Transcribe with Whisper
    import whisper as _whisper
    wmodel = models.get("whisper")
    if wmodel is None:
        models["whisper"] = _whisper.load_model("small")
        wmodel = models["whisper"]

    if audio_np is not None:
        result = wmodel.transcribe(audio_np, fp16=False)
    else:
        result = wmodel.transcribe(tmp.name, fp16=False)
        os.unlink(tmp.name)

    text = result["text"].strip()
    lang = models["detector"].detect(text) if text else "en"
    return {"text": text, "language": lang}

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Reset conversation history for a session."""
    if session_id in sessions:
        del sessions[session_id]
    return {"cleared": session_id}

@app.post("/session/{session_id}/switch")
def switch_language(session_id: str):
    """Clear memory on language switch — fresh conversation start."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "memory cleared"}
if __name__ == "__main__":
    import uvicorn
    print("Starting Customer Care Agent on http://localhost:8001")
    print("UI: http://localhost:5174")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
