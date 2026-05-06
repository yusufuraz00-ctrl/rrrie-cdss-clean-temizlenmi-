Set-StrictMode -Version Latest

$script:ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Get-ProjectRoot {
    return $script:ProjectRoot
}

function Get-ProjectPath {
    param([string]$RelativePath = "")

    if ([string]::IsNullOrWhiteSpace($RelativePath)) {
        return $script:ProjectRoot
    }
    return Join-Path $script:ProjectRoot $RelativePath
}

function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host "ERROR: $Message" -ForegroundColor Red
}

function Test-PathLocked {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return $false
    }

    $stream = $null
    try {
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
        return $false
    }
    catch {
        return $true
    }
    finally {
        if ($stream) {
            $stream.Close()
            $stream.Dispose()
        }
    }
}

function Test-PythonCandidate {
    param([string]$Executable)

    if (-not $Executable -or -not (Test-Path $Executable)) {
        return $false
    }

    try {
        $output = & $Executable -c "import sys; print(sys.executable)" 2>$null
        return $LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace(($output | Select-Object -First 1))
    }
    catch {
        return $false
    }
}

function Resolve-PythonCommand {
    $projectPython = Get-ProjectPath "runtime\python\python.exe"
    if (Test-PythonCandidate $projectPython) {
        return @{
            Executable = $projectPython
            PrefixArgs = @()
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and (Test-PythonCandidate $pythonCommand.Source)) {
        return @{
            Executable = $pythonCommand.Source
            PrefixArgs = @()
        }
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($versionArg in @("-3.12", "-3.11")) {
            try {
                $output = & $pyLauncher.Source $versionArg -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace(($output | Select-Object -First 1))) {
                    return @{
                        Executable = $pyLauncher.Source
                        PrefixArgs = @($versionArg)
                    }
                }
            }
            catch {
            }
        }
    }

    foreach ($candidate in @(
        "$env:LocalAppData\Programs\Python\Python313\python.exe",
        "$env:LocalAppData\Programs\Python\Python312\python.exe",
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python313\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "$env:ProgramFiles\Python311\python.exe"
    )) {
        if (Test-PythonCandidate $candidate) {
            return @{
                Executable = $candidate
                PrefixArgs = @()
            }
        }
    }

    return $null
}

function Invoke-PythonCommand {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$PythonCommand,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [string]$WorkingDirectory = (Get-ProjectRoot),

        [switch]$CaptureOutput
    )

    Push-Location $WorkingDirectory
    try {
        if ($CaptureOutput) {
            return (& $PythonCommand.Executable @($PythonCommand.PrefixArgs + $Arguments))
        }

        & $PythonCommand.Executable @($PythonCommand.PrefixArgs + $Arguments) | Out-Host
        return $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

function Get-VenvPythonCommand {
    $venvPython = Get-ProjectPath ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        return $null
    }

    if (-not (Test-PythonCandidate $venvPython)) {
        return $null
    }

    return @{
        Executable = $venvPython
        PrefixArgs = @()
    }
}

function Get-VenvPythonStatus {
    $venvPython = Get-ProjectPath ".venv\Scripts\python.exe"
    $status = @{
        Exists = $false
        Healthy = $false
        Path = $venvPython
        Issue = "missing"
    }

    if (-not (Test-Path $venvPython)) {
        return $status
    }

    $status.Exists = $true
    if (Test-PythonCandidate $venvPython) {
        $status.Healthy = $true
        $status.Issue = ""
        return $status
    }

    $status.Issue = "broken"
    return $status
}

function Get-HealthyProjectVenvPythonCommand {
    $venvStatus = Get-VenvPythonStatus
    $venvPython = Get-VenvPythonCommand
    if ($venvPython) {
        return $venvPython
    }

    if ($venvStatus.Exists -and $venvStatus.Issue -eq "broken") {
        throw ".venv exists but the Python executable is unusable or points to a missing base interpreter. Run: powershell -ExecutionPolicy Bypass -File .\setup\install.ps1 -Repair"
    }

    throw ".venv not found. Run .\setup\install.ps1 first."
}

function Resolve-LlamaServerPath {
    $envPath = $env:LLAMA_SERVER_EXE
    if ($envPath -and (Test-Path $envPath)) {
        return (Resolve-Path $envPath).Path
    }

    $projectBinary = Get-ProjectPath "llama-server\llama-server.exe"
    if (Test-Path $projectBinary) {
        return $projectBinary
    }

    foreach ($commandName in @("llama-server.exe", "llama-server")) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    return $null
}

function Resolve-LlamaServerLegacyPath {
    $envPath = $env:LLAMA_SERVER_EXE_LEGACY
    if ($envPath -and (Test-Path $envPath)) {
        return (Resolve-Path $envPath).Path
    }

    foreach ($candidate in @(
        (Get-ProjectPath "llama-server\llama-server-legacy.exe"),
        (Get-ProjectPath "llama-server\llama-server-sm75.exe"),
        (Get-ProjectPath "llama-server\llama-server-cc75.exe")
    )) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}
