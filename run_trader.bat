@echo off
cd C:\Users\mteev\Documents\Claude-Investing\paper_trader\src
call ..\venv\Scripts\activate
python main.py >> ..\models\daily_log.txt 2>&1

rem Sunday only: full dashboard rerun + weekly performance summary.
rem DayOfWeek as int is locale-safe: Sunday=0, Monday=1, ..., Saturday=6.
for /f %%i in ('powershell -NoProfile -Command "[int](Get-Date).DayOfWeek"') do set DOW=%%i
if "%DOW%"=="0" (
    python morning_summary.py >> ..\models\weekly_log.txt 2>&1
)