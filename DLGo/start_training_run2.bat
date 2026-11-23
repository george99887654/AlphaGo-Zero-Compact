@echo off
rem Run 2: a training run from scratch in work\scratch3 with komi 7.5 and the
rem same settings as run 1 now uses. Its Elo chart is
rem ..\schematics\rlgo_elo_report_run2.html (run 1 keeps rlgo_elo_report.html).
rem The first start creates a random network (work\scratch3\init.hdf5); later
rem starts resume where it stopped. Close the window or press Ctrl+C to stop.
rem Don't run it at the same time as start_training.bat: both need the GPU.
cd /d "%~dp0"
set TF_CPP_MIN_LOG_LEVEL=2
set PYTHONUNBUFFERED=1
title RLGo training run 2 (close this window to stop)
set PY="%~dp0..\.venv\Scripts\python.exe"
set RUN=work\scratch3
set OPTS=--report-dir ..\schematics --report-name run2 -d %RUN% -l %RUN%\log ^
  -g 256 --rounds-per-eval 1 --eval-games 128 --gate-threshold 0.60 --elo-games 40 --elo-opponents 1 ^
  -w 2 -p 128 --inference-server --league-opponents 2 --league-games 32 --league-checkpoints curated ^
  --lr 0.0005 --bs 256 --komi 7.5
if exist %RUN%\zero_agent_train.hdf5 (
  %PY% play_train_eval_zero.py --resume %OPTS%
) else (
  if not exist %RUN% mkdir %RUN%
  if not exist %RUN%\init.hdf5 %PY% init_zero_agent_v2.py %RUN%\init.hdf5 > %RUN%\init.log
  %PY% play_train_eval_zero.py --lagent %RUN%\init.hdf5 --ragent %RUN%\init.hdf5 %OPTS%
)
echo.
echo Training stopped. Run this file again to resume.
pause
