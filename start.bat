@echo off
echo ============================================
echo   Multilingual TTS - Starting
echo ============================================
echo.
echo Starting API server on http://localhost:8000
echo Starting React UI  on http://localhost:5173
echo.
echo Press Ctrl+C in each window to stop.
echo.

:: Start API server in a new window
start "TTS API Server" cmd /k "python api_server.py"

:: Wait 3 seconds for API to boot
timeout /t 3 /nobreak >nul

:: Start React dev server in a new window
start "TTS React UI" cmd /k "cd assistant_project\frontend\tts-ui && npm run dev"

:: Open browser
timeout /t 4 /nobreak >nul
start http://localhost:5173
