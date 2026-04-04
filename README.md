# 🎙️ Multilingual AI Voice Assistant

A multilingual AI system with two components:
- **TTS API** — Text-to-Speech for Kannada, Hindi, Tamil, Telugu, English
- **Customer Care Agent** — Voice-enabled AI customer care in all 5 languages

## Quick Install

```bash
git clone https://github.com/Rishith-17/tts-opensource-indian.git
cd tts-opensource-indian
install.bat
```

> Requires Python 3.10–3.12 and Node.js 18+

## Setup API Key

Copy `.env.example` to `.env` and add your key:
```
SARVAM_API_KEY=your_key_here
```
Get your free key at [sarvam.ai](https://sarvam.ai)

## Start

```bash
# TTS API only
python api_server.py          # http://localhost:8000/docs

# Customer Care Agent
customer_care\start.bat       # http://localhost:5174
```

## Languages

| Language | TTS Engine |
|----------|-----------|
| Kannada (ಕನ್ನಡ) | IndicTTS |
| Hindi (हिंदी) | MMS TTS |
| Tamil (தமிழ்) | IndicTTS |
| Telugu (తెలుగు) | IndicTTS |
| English | Kokoro-82M |

## Requirements

- Python 3.10–3.12
- Node.js 18+
- GPU recommended (4GB+ VRAM)

## Finetune (optional)

```bash
python finetune/prepare_dataset.py --lang kn
python finetune/train.py --lang kn --batch_size 8 --evaluate
```
