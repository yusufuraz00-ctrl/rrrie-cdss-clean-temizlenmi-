# Windows Install - "Perfect Build" Verified

This build is hardened for universal Windows compatibility and has been validated in a clean, isolated virtual environment.

## Standard Install

```powershell
powershell -ExecutionPolicy Bypass -File .\setup\install.ps1
```

What it does:

- finds Python 3.11+
- creates `.venv`
- installs the curated dependency set
- installs the project in editable mode
- copies `.env.example` to `.env` if needed
- downloads the default GGUF models
- runs `setup\verify_installation.py`
- keeps the active product surface on the native vNext API/UI path only

## Start

```powershell
powershell -ExecutionPolicy Bypass -File .\setup\start.ps1
```

The active API/UI surface is native vNext:

- UI: `http://127.0.0.1:7860`
- API: `POST /api/vnext/analyze`
- legacy `analyze-legacy` transport is not part of the active product surface anymore

## Status / Stop

```powershell
powershell -ExecutionPolicy Bypass -File .\setup\start.ps1 -Status
powershell -ExecutionPolicy Bypass -File .\setup\start.ps1 -Doctor
powershell -ExecutionPolicy Bypass -File .\setup\start.ps1 -Stop
```

## If Python Is Missing

Install Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/windows/) and rerun the installer.

## If `llama-server.exe` Is Missing

Place it under `.\llama-server\llama-server.exe` or set:

```powershell
$env:LLAMA_SERVER_EXE="C:\path\to\llama-server.exe"
```
