@echo off
title RRRIE-CDSS Setup
echo ============================================================
echo  RRRIE-CDSS  ^|  First-Time Setup
echo ============================================================
echo.
echo This will:
echo   - Create .venv and install dependencies
echo   - Copy .env.example to .env
echo   - Download GGUF models into models\
echo   - Run installation verification
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0setup\install.ps1"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================
    echo  Setup FAILED. See output above for details.
    echo ============================================================
    pause
    exit /b 1
)
echo.
echo ============================================================
echo  Setup complete!
echo  Double-click run.bat to start RRRIE-CDSS.
echo ============================================================
pause
