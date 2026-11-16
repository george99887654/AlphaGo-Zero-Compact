@echo off
rem Serve the best-rated bot so you can play it at http://127.0.0.1:5000/
rem Double-click this file. Close the window or press Ctrl+C to stop.
cd /d "%~dp0"
set TF_CPP_MIN_LOG_LEVEL=2
title RLGo demo - http://127.0.0.1:5000/ (close this window to stop)
rem Open the browser once the bot has loaded (about 15 s).
start "" /min cmd /c "timeout /t 15 >nul & start http://127.0.0.1:5000/"
"%~dp0..\.venv\Scripts\python.exe" web_demo.py %*
pause
