@echo off
echo ============================================
echo   Multilingual TTS - Install
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Install Python 3.10-3.12 from https://python.org
    pause & exit /b 1
)

:: Check Node
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found.
    echo Install Node.js 18+ from https://nodejs.org
    pause & exit /b 1
)

echo [1/4] Installing PyTorch...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
if errorlevel 1 (
    echo PyTorch GPU install failed, trying CPU version...
    pip install torch torchaudio --quiet
)

echo.
echo [2/4] Installing Python dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo Retrying with --no-deps fallback...
    pip install TTS>=0.22.0 kokoro>=0.9.4 transformers soundfile scipy openai-whisper fasttext-wheel sarvamai fastapi "uvicorn[standard]" python-multipart --quiet
)

echo.
echo [3/4] Installing React frontend...
cd customer_care\ui
npm install --silent
if errorlevel 1 ( echo [ERROR] npm install failed & cd ..\.. & pause & exit /b 1 )
npm run build --silent
cd ..\..

echo.
echo [4/4] Setting up environment...
if not exist .env (
    copy .env.example .env >nul
    echo.
    echo [ACTION REQUIRED] Open .env and add your Sarvam API key:
    echo   SARVAM_API_KEY=your_key_here
    echo   Get your key at: https://sarvam.ai
    echo.
)

echo.
echo [5/5] Downloading TTS models...
python setup_models.py

echo.
echo ============================================
echo   Done! Next steps:
echo   1. Add your Sarvam API key to .env
echo   2. Run: start.bat
echo ============================================
pause
