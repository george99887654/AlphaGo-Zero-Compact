@echo off
rem Resume the RLGo training run in work\scratch2 in its own window.
rem Double-click this file (or run it from a terminal). Close the window or
rem press Ctrl+C to stop; start it again later and it continues where it
rem stopped. The Elo chart is updated in ..\schematics after every iteration.
cd /d "%~dp0"
set TF_CPP_MIN_LOG_LEVEL=2
set PYTHONUNBUFFERED=1
title RLGo training (close this window to stop)
"%~dp0..\.venv\Scripts\python.exe" play_train_eval_zero.py --resume --report-dir ..\schematics ^
  -d work\scratch2 -l work\scratch2\log ^
  -g 256 --rounds-per-eval 1 --eval-games 64 --elo-games 40 --elo-opponents 1 ^
  -w 2 -p 128 --inference-server --league-opponents 2 --league-games 32 ^
  --lr 0.001 --bs 256
echo.
echo Training stopped. Run this file again to resume.
pause
