@echo off
:: RRRIE-CDSS One-Click Launcher
:: No PowerShell execution policy needed — just double-click this file.

title RRRIE-CDSS Launcher
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  ERROR: Virtual environment not found.
    echo  Run install first: powershell -ExecutionPolicy Bypass -File .\setup\install.ps1
    echo.
    pause
    exit /b 1
)

echo.
echo  ============================================
echo   RRRIE-CDSS Clinical Decision Support System
echo  ============================================
echo.

:: Activate venv and run
".venv\Scripts\python.exe" run.py %*

if errorlevel 1 (
    echo.
    echo  RRRIE-CDSS exited with an error.
    echo  Check output\logs\ for details.
    echo.
    pause
)
