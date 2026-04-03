@echo off
echo Starting Customer Care AI Agent...
echo.
echo API:  http://localhost:8001
echo UI:   http://localhost:5174
echo Docs: http://localhost:8001/docs
echo.

start "Customer Care API" cmd /k "cd /d %~dp0.. && python customer_care/agent.py"
timeout /t 4 /nobreak >nul
start "Customer Care UI" cmd /k "cd /d %~dp0ui && npm run dev -- --port 5174"
timeout /t 3 /nobreak >nul
start http://localhost:5174
