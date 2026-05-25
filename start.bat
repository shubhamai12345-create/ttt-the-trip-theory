@echo off
title TTT-V0 — The Trip Theory
color 0A
echo.
echo  ==========================================
echo   TTT-V0 — The Trip Theory
echo  ==========================================
echo.
cd /d "%~dp0"
echo Installing dependencies...
pip install -r requirements.txt -q
echo Starting server...
echo.
echo  Open browser: http://localhost:8000
echo  Press Ctrl+C to stop
echo.
python app.py
pause
