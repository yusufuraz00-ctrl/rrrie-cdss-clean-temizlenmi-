param(
    [switch]$Doctor,
    [switch]$Status,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. (Join-Path $PSScriptRoot "common.ps1")

$projectRoot = Get-ProjectRoot

function Invoke-InstallRepair {
    $installScript = Join-Path $PSScriptRoot "install.ps1"
    Write-Warn "Project .venv is missing or broken. Attempting automatic repair..."
    & powershell.exe -ExecutionPolicy Bypass -File $installScript -Repair
    return $LASTEXITCODE
}

$venvPython = $null
try {
    $venvPython = Get-HealthyProjectVenvPythonCommand
}
catch {
    Write-Warn $_.Exception.Message
    $bootstrapPython = Resolve-PythonCommand
    if (-not $bootstrapPython) {
        Write-Fail "No compatible Python bootstrap interpreter was found. Install Python 3.11 or 3.12, then rerun .\\setup\\install.ps1."
        exit 1
    }

    $repairExitCode = Invoke-InstallRepair
    if ($repairExitCode -ne 0) {
        Write-Fail "Automatic environment repair failed. Run .\\setup\\install.ps1 -Repair manually after fixing Python/network issues."
        exit $repairExitCode
    }

    try {
        $venvPython = Get-HealthyProjectVenvPythonCommand
    }
    catch {
        Write-Fail $_.Exception.Message
        exit 1
    }
}

if ($Status) {
    $statusExitCode = Invoke-PythonCommand -PythonCommand $venvPython -Arguments @((Join-Path $projectRoot "run.py"), "--status")
    exit $statusExitCode
}

if ($Stop) {
    $stopExitCode = Invoke-PythonCommand -PythonCommand $venvPython -Arguments @((Join-Path $projectRoot "run.py"), "--stop")
    exit $stopExitCode
}

if ($Doctor) {
    $doctorExitCode = Invoke-PythonCommand -PythonCommand $venvPython -Arguments @((Join-Path $projectRoot "run.py"), "--doctor")
    exit $doctorExitCode
}

$doctorExitCode = Invoke-PythonCommand -PythonCommand $venvPython -Arguments @((Join-Path $projectRoot "run.py"), "--doctor")
if ($doctorExitCode -ne 0) {
    Write-Fail "Runtime doctor reported fatal issues. Fix them before starting the launcher."
    exit $doctorExitCode
}

Write-Info "Starting RRRIE-CDSS..."
Write-Host "UI will be available at http://127.0.0.1:7860"
$runExitCode = Invoke-PythonCommand -PythonCommand $venvPython -Arguments @((Join-Path $projectRoot "run.py"))
exit $runExitCode
