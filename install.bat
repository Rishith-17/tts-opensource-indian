@echo off
echo ============================================
echo   Multilingual TTS - Install
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause & exit /b 1
)

:: Check Node
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org
    pause & exit /b 1
)

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed & pause & exit /b 1 )

echo.
echo [2/3] Installing React frontend dependencies...
cd assistant_project\frontend\tts-ui
npm install
if errorlevel 1 ( echo [ERROR] npm install failed & pause & exit /b 1 )
cd ..\..\..

echo.
echo [3/3] Setting up environment...
if not exist .env (
    copy .env.example .env
    echo.
    echo [ACTION REQUIRED] Open .env and add your Sarvam API key:
    echo   SARVAM_API_KEY=your_key_here
    echo   Get your key at: https://sarvam.ai
    echo.
)

echo [4/4] Downloading TTS models...
python setup_models.py
if errorlevel 1 ( echo [WARNING] Model download had issues - check setup_models.py manually )

echo.
echo ============================================
echo   Installation complete!
echo   1. Add your Sarvam API key to .env
echo   2. Run start.bat to launch the app.
echo ============================================
pause
