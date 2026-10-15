@echo off
rem Starts the studio and opens it in the browser.
cd /d "%~dp0"
uv sync --quiet || goto :error
uv run python -m server
goto :eof
:error
echo Instale o uv (https://docs.astral.sh/uv/) e tente de novo.
pause
