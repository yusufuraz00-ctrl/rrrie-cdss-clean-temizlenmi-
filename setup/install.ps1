param(
    [string]$PythonMinVersion = "3.11",
    [switch]$Repair
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$script:ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script:RequirementsFile = (Join-Path $script:ProjectRoot "requirements.txt")
$script:EnvFile = (Join-Path $script:ProjectRoot ".env")
$script:EnvExampleFile = (Join-Path $script:ProjectRoot ".env.example")
$script:VenvPath = (Join-Path $script:ProjectRoot ".venv")
$script:ModelsDir = (Join-Path $script:ProjectRoot "models")
$script:PortablePythonDir = (Join-Path $script:ProjectRoot "runtime\python")
$script:LlamaInstallDir = (Join-Path $script:ProjectRoot "llama-server")
$script:LlamaServerDefault = (Join-Path $script:ProjectRoot "llama-server\llama-server.exe")
$script:LlamaReleaseApi = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
$script:ModelManifestPath = (Join-Path $script:ProjectRoot "setup\models_urls.txt")
$script:RuntimeCompactThresholdGb = 4.75
$script:DefaultDownloadRetries = 5
$script:MinDiskSpaceGb = 5.0
$script:RequiredModules = @(
    "fastapi",
    "uvicorn",
    "httpx",
    "requests",
    "pydantic",
    "pydantic_settings",
    "structlog",
    "google.genai"
)
$script:DefaultModelDefinitions = @(
    [pscustomobject]@{
        Role = "main"
        FileName = "Qwen3.5-4B-Q4_K_M.gguf"
        Urls = @("https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf")
        MinBytes = [int64]2000000000
    },
    [pscustomobject]@{
        Role = "dllm"
        FileName = "Qwen3.5-0.8B-Q4_K_M.gguf"
        Urls = @("https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf")
        MinBytes = [int64]400000000
    }
)
$script:ModelDefinitions = @()

function Test-DiskSpace {
    param([string]$Path, [double]$RequiredGb)
    
    $drive = (Get-Item $Path).PSDrive
    if ($null -eq $drive) {
        $driveLetter = [System.IO.Path]::GetPathRoot($Path).Replace("\", "")
        $drive = Get-PSDrive -Name $driveLetter.Replace(":", "")
    }
    
    $freeGb = $drive.Free / 1GB
    if ($freeGb -lt $RequiredGb) {
        throw "Insufficient disk space on $($drive.Name). Required: $RequiredGb GB, Available: $($freeGb.ToString('F2')) GB."
    }
    Write-Okay "Disk space check passed: $($freeGb.ToString('F2')) GB available."
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

function Write-Okay {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Get-ModelDefinitions {
    $definitions = New-Object System.Collections.Generic.List[object]

    if (Test-Path $script:ModelManifestPath) {
        foreach ($line in (Get-Content -Path $script:ModelManifestPath -ErrorAction SilentlyContinue)) {
            $trimmed = $line.Trim()
            if (-not $trimmed -or $trimmed.StartsWith("#")) {
                continue
            }

            $parts = $trimmed.Split("|")
            if ($parts.Length -lt 4) {
                continue
            }

            $role = $parts[0].Trim().ToLowerInvariant()
            $fileName = $parts[1].Trim()
            $minBytesRaw = $parts[2].Trim()
            $urls = @($parts[3..($parts.Length - 1)] | ForEach-Object { $_.Trim() } | Where-Object { $_ })

            $minBytes = [int64]0
            if (-not [int64]::TryParse($minBytesRaw, [ref]$minBytes)) {
                continue
            }
            if (-not $fileName -or $urls.Count -eq 0) {
                continue
            }

            $definitions.Add([pscustomobject]@{
                Role = $role
                FileName = $fileName
                Urls = $urls
                MinBytes = $minBytes
            }) | Out-Null
        }
    }

    if ($definitions.Count -gt 0) {
        return @($definitions.ToArray())
    }

    return @($script:DefaultModelDefinitions)
}

function Get-DownloadMethods {
    param([switch]$RequiresCustomHeaders)

    $methods = New-Object System.Collections.Generic.List[string]

    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        $methods.Add("curl")
    }

    if (-not $RequiresCustomHeaders -and (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue)) {
        $methods.Add("bits")
    }

    $methods.Add("webrequest")
    return @($methods)
}

function Invoke-DownloadViaCurl {
    param(
        [string]$Url,
        [string]$Destination,
        [hashtable]$Headers,
        [switch]$Resume,
        [int]$Attempt
    )

    $arguments = @(
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "2",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--output",
        $Destination
    )

    if ($Resume) {
        $arguments += @("--continue-at", "-")
    }

    foreach ($headerKey in $Headers.Keys) {
        $arguments += @("-H", ("{0}: {1}" -f $headerKey, $Headers[$headerKey]))
    }

    $arguments += $Url
    & curl.exe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "curl.exe failed on attempt $Attempt."
    }
}

function Invoke-DownloadViaBits {
    param(
        [string]$Url,
        [string]$Destination
    )

    Start-BitsTransfer -Source $Url -Destination $Destination -TransferType Download -ErrorAction Stop
}

function Invoke-DownloadViaWebRequest {
    param(
        [string]$Url,
        [string]$Destination,
        [hashtable]$Headers
    )

    Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing -Headers $Headers
}

function Invoke-DownloadWithRetries {
    param(
        [string]$Url,
        [string]$Destination,
        [hashtable]$Headers = @{},
        [switch]$AllowResume,
        [int]$MaxRetries = 0,
        [string]$DisplayName = ""
    )

    if (-not $MaxRetries -or $MaxRetries -lt 1) {
        $MaxRetries = $script:DefaultDownloadRetries
    }

    if (-not $DisplayName) {
        $DisplayName = [System.IO.Path]::GetFileName($Destination)
    }

    $requiresHeaders = ($Headers.Keys.Count -gt 0)
    $methods = Get-DownloadMethods -RequiresCustomHeaders:$requiresHeaders
    $lastError = $null

    foreach ($method in $methods) {
        for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
            try {
                Write-Info ("Downloading {0} via {1} ({2}/{3})..." -f $DisplayName, $method, $attempt, $MaxRetries)

                switch ($method) {
                    "curl" {
                        Invoke-DownloadViaCurl -Url $Url -Destination $Destination -Headers $Headers -Resume:$AllowResume -Attempt $attempt
                        return $method
                    }
                    "bits" {
                        if (Test-Path $Destination) {
                            Remove-Item -LiteralPath $Destination -Force -ErrorAction SilentlyContinue
                        }
                        Invoke-DownloadViaBits -Url $Url -Destination $Destination
                        return $method
                    }
                    "webrequest" {
                        if ((Test-Path $Destination) -and (-not $AllowResume)) {
                            Remove-Item -LiteralPath $Destination -Force -ErrorAction SilentlyContinue
                        }
                        Invoke-DownloadViaWebRequest -Url $Url -Destination $Destination -Headers $Headers
                        return $method
                    }
                }
            }
            catch {
                $lastError = $_
                if ($attempt -lt $MaxRetries) {
                    Start-Sleep -Seconds ([math]::Min(8, 2 * $attempt))
                }
            }
        }

        if ((Test-Path $Destination) -and ($method -ne "curl")) {
            Remove-Item -LiteralPath $Destination -Force -ErrorAction SilentlyContinue
        }
    }

    if ($lastError) {
        throw $lastError
    }

    throw "Download failed for $DisplayName."
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
    $portablePython = Join-Path $script:PortablePythonDir "python.exe"
    if (Test-PythonCandidate $portablePython) {
        return @{
            Executable = $portablePython
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

        [string]$WorkingDirectory = $script:ProjectRoot
    )

    Push-Location $WorkingDirectory
    try {
        & $PythonCommand.Executable @($PythonCommand.PrefixArgs + $Arguments) | Out-Host
        return $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

function Invoke-PythonCapture {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$PythonCommand,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [string]$WorkingDirectory = $script:ProjectRoot
    )

    Push-Location $WorkingDirectory
    try {
        $output = & $PythonCommand.Executable @($PythonCommand.PrefixArgs + $Arguments) 2>&1
        return @{
            ExitCode = $LASTEXITCODE
            Output = @($output)
        }
    }
    finally {
        Pop-Location
    }
}

function Get-VenvPythonCommand {
    $venvPython = Join-Path $script:VenvPath "Scripts\python.exe"
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
    $venvPython = Join-Path $script:VenvPath "Scripts\python.exe"
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

function Resolve-LlamaServerPath {
    if ($env:LLAMA_SERVER_EXE -and (Test-Path $env:LLAMA_SERVER_EXE)) {
        return (Resolve-Path $env:LLAMA_SERVER_EXE).Path
    }

    if (Test-Path $script:LlamaServerDefault) {
        return $script:LlamaServerDefault
    }

    $parentLlama = (Join-Path $script:ProjectRoot "..\llama-server\llama-server.exe")
    if (Test-Path $parentLlama) {
        return (Resolve-Path $parentLlama).Path
    }

    foreach ($commandName in @("llama-server.exe", "llama-server")) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    return $null
}

function Test-NvidiaRuntimeAvailable {
    try {
        $null = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Get-LlamaReleaseInfo {
    try {
        $headers = @{
            "User-Agent" = "RRRIE-CDSS-Installer"
            "Accept" = "application/vnd.github+json"
        }
        return Invoke-RestMethod -Uri $script:LlamaReleaseApi -Headers $headers
    }
    catch {
        throw "Could not reach the official llama.cpp release feed on GitHub. Connect to the internet or include the llama-server folder in the package."
    }
}

function Get-LlamaAssetsForMachine {
    $release = Get-LlamaReleaseInfo
    $assets = @($release.assets)
    if (-not $assets -or $assets.Count -eq 0) {
        throw "Latest llama.cpp release metadata did not include downloadable assets."
    }

    if (Test-NvidiaRuntimeAvailable) {
        $cudaBinary = $assets | Where-Object { $_.name -match '^llama-b.+-bin-win-cuda-12\.4-x64\.zip$' } | Select-Object -First 1
        $cudaDlls = $assets | Where-Object { $_.name -eq 'cudart-llama-bin-win-cuda-12.4-x64.zip' } | Select-Object -First 1
        
        # Also try to grab legacy CUDA 11.7 for older GPUs (GTX 1650 Ti)
        $cudaLegacyDlls = $assets | Where-Object { $_.name -eq 'cudart-llama-bin-win-cuda-11.7-x64.zip' } | Select-Object -First 1
        $cudaLegacyBinary = $assets | Where-Object { $_.name -match '^llama-b.+-bin-win-cuda-11\.7-x64\.zip$' } | Select-Object -First 1
        
        if ($cudaBinary -and $cudaDlls) {
            $selectedAssets = @($cudaBinary, $cudaDlls)
            if ($cudaLegacyBinary -and $cudaLegacyDlls) {
                 $selectedAssets += $cudaLegacyDlls
                 $selectedAssets += $cudaLegacyBinary
            }
            return [pscustomobject]@{
                Backend = "cuda-12.4+legacy"
                Tag = $release.tag_name
                Assets = $selectedAssets
            }
        }

        Write-Warn "CUDA runtime was detected, but the latest CUDA 12.4 llama.cpp assets were not available. Falling back to CPU package."
    }

    $cpuBinary = $assets | Where-Object { $_.name -match '^llama-b.+-bin-win-cpu-x64\.zip$' } | Select-Object -First 1
    if (-not $cpuBinary) {
        throw "Latest llama.cpp release did not include the Windows x64 CPU package."
    }

    return [pscustomobject]@{
        Backend = "cpu"
        Tag = $release.tag_name
        Assets = @($cpuBinary)
    }
}

function Download-File {
    param(
        [string]$Url,
        [string]$Destination
    )

    try {
        $headers = @{
            "User-Agent" = "RRRIE-CDSS-Installer"
            "Accept" = "application/octet-stream"
        }
        $method = Invoke-DownloadWithRetries -Url $Url -Destination $Destination -Headers $headers -AllowResume -DisplayName ([System.IO.Path]::GetFileName($Destination))
        Write-Okay ("Download complete via {0}: {1}" -f $method, ([System.IO.Path]::GetFileName($Destination)))
    }
    catch {
        throw "Could not download $Destination from GitHub. Connect to the internet or include the llama-server folder in the package."
    }
}

function Ensure-LlamaServer {
    $existing = Resolve-LlamaServerPath
    if ($existing) {
        return [pscustomobject]@{
            Path = $existing
            Source = "bundled"
            Backend = "existing"
        }
    }

    Write-Warn "Bundled llama-server was not found. The installer will fetch an official package from ggml-org/llama.cpp."

    $selection = Get-LlamaAssetsForMachine
    $tempRoot = Join-Path $env:TEMP ("rrrie-llama-server-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $tempRoot | Out-Null

    try {
        if (-not (Test-Path $script:LlamaInstallDir)) {
            New-Item -ItemType Directory -Path $script:LlamaInstallDir | Out-Null
        }

        foreach ($asset in $selection.Assets) {
            $zipPath = Join-Path $tempRoot $asset.name
            Write-Info ("Downloading llama.cpp asset: {0}" -f $asset.name)
            Download-File -Url $asset.browser_download_url -Destination $zipPath
            Expand-Archive -Path $zipPath -DestinationPath $script:LlamaInstallDir -Force
        }
    }
    finally {
        if (Test-Path $tempRoot) {
            Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    $installed = Resolve-LlamaServerPath
    if (-not $installed) {
        throw "llama-server download completed, but llama-server.exe was not found after extraction."
    }

    Write-Okay ("llama-server installed from official llama.cpp release ({0}, {1})." -f $selection.Tag, $selection.Backend)
    return [pscustomobject]@{
        Path = $installed
        Source = "downloaded"
        Backend = $selection.Backend
    }
}

function Get-DetectedPythonVersion {
    param([hashtable]$PythonCommand)

    $result = Invoke-PythonCapture -PythonCommand $PythonCommand -Arguments @(
        "-c",
        "import sys; print('.'.join(map(str, sys.version_info[:3])))"
    )

    if ($result.ExitCode -ne 0 -or -not $result.Output) {
        throw "Python version could not be detected."
    }

    return ($result.Output | Select-Object -First 1).ToString().Trim()
}

function Get-DetectedVramGb {
    try {
        $output = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $null
        }

        $firstLine = $output | Where-Object { $_.ToString().Trim() } | Select-Object -First 1
        if (-not $firstLine) {
            return $null
        }

        return [math]::Round(([double]$firstLine.ToString().Trim()) / 1024, 2)
    }
    catch {
        return $null
    }
}

function Resolve-RuntimeProfile {
    $vramGb = Get-DetectedVramGb
    if ($null -eq $vramGb) {
        return [pscustomobject]@{
            Name = "compact_4gb"
            DetectedVramGb = $null
            Reason = "GPU VRAM could not be detected. Falling back to compact profile."
        }
    }

    if ($vramGb -le $script:RuntimeCompactThresholdGb) {
        return [pscustomobject]@{
            Name = "compact_4gb"
            DetectedVramGb = $vramGb
            Reason = "Detected VRAM is $vramGb GB."
        }
    }

    return [pscustomobject]@{
        Name = "standard_6gb"
        DetectedVramGb = $vramGb
        Reason = "Detected VRAM is $vramGb GB."
    }
}

function Ensure-EnvFile {
    param([pscustomobject]$RuntimeProfile)

    if (-not (Test-Path $script:EnvFile)) {
        if (Test-Path $script:EnvExampleFile) {
            Copy-Item $script:EnvExampleFile $script:EnvFile
            Write-Warn ".env created from .env.example."
        }
        else {
            @(
                "LLAMA_SERVER_URL=http://127.0.0.1:8080"
                "DLLM_SERVER_URL=http://127.0.0.1:8081"
                "RUNTIME_PROFILE_DEFAULT=compact_4gb"
            ) | Set-Content -Path $script:EnvFile -Encoding UTF8
            Write-Warn ".env created with fallback defaults."
        }
    }

    Set-EnvValue -Path $script:EnvFile -Key "LLAMA_SERVER_URL" -Value "http://127.0.0.1:8080"
    Set-EnvValue -Path $script:EnvFile -Key "DLLM_SERVER_URL" -Value "http://127.0.0.1:8081"
    Set-EnvValue -Path $script:EnvFile -Key "RUNTIME_PROFILE_DEFAULT" -Value $RuntimeProfile.Name
}

function Set-EnvValue {
    param(
        [string]$Path,
        [string]$Key,
        [string]$Value
    )

    $lines = @()
    if (Test-Path $Path) {
        $lines = @(Get-Content $Path -Encoding UTF8)
    }

    $pattern = '^\s*' + [regex]::Escape($Key) + '\s*='
    $replacement = "$Key=$Value"
    $updated = $false

    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $pattern) {
            $lines[$i] = $replacement
            $updated = $true
            break
        }
    }

    if (-not $updated) {
        $lines += $replacement
    }

    Set-Content -Path $Path -Value $lines -Encoding UTF8
}

function Test-PythonImports {
    param([hashtable]$PythonCommand)

    $pythonCode = @'
import importlib.util
import json
import sys

required = [
    'fastapi',
    'uvicorn',
    'httpx',
    'requests',
    'pydantic',
    'pydantic_settings',
    'structlog',
    'google.genai',
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(json.dumps(missing))
sys.exit(0 if not missing else 1)
'@

    $result = Invoke-PythonCapture -PythonCommand $PythonCommand -Arguments @("-c", $pythonCode)
    $missing = @()
    if ($result.Output) {
        $jsonLine = ($result.Output | Select-Object -Last 1).ToString().Trim()
        if ($jsonLine) {
            try {
                $missing = @((ConvertFrom-Json $jsonLine))
            }
            catch {
                $missing = @()
            }
        }
    }

    return [pscustomobject]@{
        Ready = ($result.ExitCode -eq 0)
        Missing = $missing
    }
}

function Test-PipCheck {
    param([hashtable]$PythonCommand)

    $result = Invoke-PythonCapture -PythonCommand $PythonCommand -Arguments @("-m", "pip", "check")
    return [pscustomobject]@{
        Ready = ($result.ExitCode -eq 0)
        Output = ($result.Output | ForEach-Object { $_.ToString() })
    }
}

function Test-VenvHealth {
    $venvStatus = Get-VenvPythonStatus
    $venvPython = Get-VenvPythonCommand
    if (-not $venvPython) {
        return [pscustomobject]@{
            Ready = $false
            Reason = if ($venvStatus.Exists -and $venvStatus.Issue -eq "broken") {
                ".venv exists but the Python executable is broken or points to a missing base interpreter."
            }
            else {
                ".venv is missing."
            }
            MissingModules = @()
            PipCheckOutput = @()
        }
    }

    $importState = Test-PythonImports -PythonCommand $venvPython
    if (-not $importState.Ready) {
        return [pscustomobject]@{
            Ready = $false
            Reason = "Required Python packages are missing."
            MissingModules = $importState.Missing
            PipCheckOutput = @()
        }
    }

    $pipState = Test-PipCheck -PythonCommand $venvPython
    if (-not $pipState.Ready) {
        return [pscustomobject]@{
            Ready = $false
            Reason = "pip check reported dependency issues."
            MissingModules = @()
            PipCheckOutput = $pipState.Output
        }
    }

    return [pscustomobject]@{
        Ready = $true
        Reason = "Virtual environment is healthy."
        MissingModules = @()
        PipCheckOutput = @()
    }
}

function Ensure-Venv {
    param([hashtable]$PythonCommand)

    if (-not (Test-Path $script:RequirementsFile)) {
        throw "requirements.txt not found at $($script:RequirementsFile)"
    }

    $health = Test-VenvHealth
    if ($health.Ready) {
        Write-Okay ".venv is already healthy. Skipping dependency reinstall."
        return (Get-VenvPythonCommand)
    }

    if (Test-Path $script:VenvPath) {
        Write-Warn "Rebuilding .venv: $($health.Reason)"
        Remove-Item -LiteralPath $script:VenvPath -Recurse -Force
    }
    else {
        Write-Info "Creating virtual environment..."
    }

    $createExitCode = Invoke-PythonCommand -PythonCommand $PythonCommand -Arguments @("-m", "venv", $script:VenvPath)
    if ($createExitCode -ne 0) {
        throw "Virtual environment creation failed."
    }

    $venvPython = Get-VenvPythonCommand
    if (-not $venvPython) {
        throw "Virtual environment creation failed."
    }

    Write-Info "Upgrading pip tooling..."
    $toolingExitCode = Invoke-PythonCommand -PythonCommand $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    if ($toolingExitCode -ne 0) {
        throw "pip tooling upgrade failed."
    }

    Write-Info "Installing project dependencies..."
    $requirementsExitCode = Invoke-PythonCommand -PythonCommand $venvPython -Arguments @("-m", "pip", "install", "-r", $script:RequirementsFile)
    if ($requirementsExitCode -ne 0) {
        throw "Dependency installation failed."
    }

    Write-Info "Installing the project in editable mode..."
    $editableExitCode = Invoke-PythonCommand -PythonCommand $venvPython -Arguments @("-m", "pip", "install", "-e", $script:ProjectRoot)
    if ($editableExitCode -ne 0) {
        throw "Editable project install failed."
    }

    Write-Info "Checking dependency consistency..."
    $pipState = Test-PipCheck -PythonCommand $venvPython
    if (-not $pipState.Ready) {
        $message = if ($pipState.Output) { ($pipState.Output -join " | ") } else { "Unknown pip check error." }
        throw "Dependency consistency check failed: $message"
    }

    $importState = Test-PythonImports -PythonCommand $venvPython
    if (-not $importState.Ready) {
        $missing = if ($importState.Missing) { $importState.Missing -join ", " } else { "unknown modules" }
        throw "Required imports are missing after install: $missing"
    }

    Write-Okay ".venv is ready."
    return $venvPython
}

function Test-ModelReady {
    param(
        [pscustomobject]$Definition
    )

    $path = Join-Path $script:ModelsDir $Definition.FileName
    if (-not (Test-Path $path)) {
        $parentPath = (Join-Path $script:ProjectRoot "..\models\$($Definition.FileName)")
        if (Test-Path $parentPath) {
            Write-Info "Found existing model in parent directory: $($Definition.FileName). Using it."
            $path = $parentPath
        }
        else {
            return [pscustomobject]@{
                Ready = $false
                Path = $path
                Size = [int64]0
            }
        }
    }

    $item = Get-Item $path
    return [pscustomobject]@{
        Ready = ($item.Length -ge $Definition.MinBytes)
        Path = $path
        Size = $item.Length
    }
}

function Get-RequiredFreeBytes {
    $requiredBytes = [int64]2147483648
    foreach ($definition in $script:ModelDefinitions) {
        $state = Test-ModelReady -Definition $definition
        if (-not $state.Ready) {
            $requiredBytes += $definition.MinBytes
        }
    }
    return $requiredBytes
}

function Get-FreeBytesOnProjectDrive {
    $projectDriveRoot = [System.IO.Path]::GetPathRoot($script:ProjectRoot)
    $drive = [System.IO.DriveInfo]::new($projectDriveRoot)
    return $drive.AvailableFreeSpace
}

function Test-ModelDownloadReachability {
    try {
        Invoke-WebRequest -Uri "https://huggingface.co/" -Method Head -UseBasicParsing -TimeoutSec 10 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Download-Model {
    param([pscustomobject]$Definition)

    $destination = Join-Path $script:ModelsDir $Definition.FileName
    $tempPath = "$destination.part"

    if (Test-Path $tempPath) {
        if (Test-PathLocked $tempPath) {
            return [pscustomobject]@{
                Status = "busy"
                Message = "$($Definition.FileName) is already downloading."
            }
        }

        Write-Warn "Removing stale partial file for $($Definition.FileName)."
        Remove-Item -LiteralPath $tempPath -Force
    }

    $headers = @{}
    if ($env:HF_TOKEN) {
        $headers["Authorization"] = "Bearer $env:HF_TOKEN"
    }

    $urls = @($Definition.Urls)
    if ($urls.Count -eq 0 -and $Definition.PSObject.Properties.Match("Url").Count -gt 0 -and $Definition.Url) {
        $urls = @($Definition.Url)
    }

    $lastFailure = ""
    foreach ($url in $urls) {
        try {
            $method = Invoke-DownloadWithRetries -Url $url -Destination $tempPath -Headers $headers -AllowResume -DisplayName $Definition.FileName
            Move-Item -LiteralPath $tempPath -Destination $destination -Force

            $state = Test-ModelReady -Definition $Definition
            if (-not $state.Ready) {
                return [pscustomobject]@{
                    Status = "failed"
                    Message = "$($Definition.FileName) downloaded but is smaller than expected."
                }
            }

            return [pscustomobject]@{
                Status = "ready"
                Message = "$($Definition.FileName) downloaded successfully via $method."
            }
        }
        catch {
            $lastFailure = $_.Exception.Message
            if ((Test-Path $tempPath) -and -not (Test-PathLocked $tempPath)) {
                Remove-Item -LiteralPath $tempPath -Force
            }
        }
    }

    return [pscustomobject]@{
        Status = "failed"
        Message = "Failed to download $($Definition.FileName). Check internet access, HF_TOKEN, manifest URLs in setup\\models_urls.txt, or copy the file into .\\models\\ manually. $lastFailure"
    }
}

function Ensure-Models {
    if (-not (Test-Path $script:ModelsDir)) {
        New-Item -ItemType Directory -Path $script:ModelsDir | Out-Null
    }

    $allReady = $true
    foreach ($definition in $script:ModelDefinitions) {
        $state = Test-ModelReady -Definition $definition
        if ($state.Ready) {
            $sizeGb = [math]::Round($state.Size / 1GB, 2)
            Write-Okay "Model ready: $($definition.FileName) ($sizeGb GB)"
            continue
        }
        $allReady = $false
    }

    if ($allReady) {
        return
    }

    $requiredBytes = Get-RequiredFreeBytes
    $freeBytes = Get-FreeBytesOnProjectDrive
    if ($freeBytes -lt $requiredBytes) {
        $neededGb = [math]::Round($requiredBytes / 1GB, 2)
        $freeGb = [math]::Round($freeBytes / 1GB, 2)
        throw "Not enough free disk space. Need about $neededGb GB free on the project drive, but only $freeGb GB is available."
    }

    if (-not (Test-ModelDownloadReachability)) {
        throw "Cannot reach huggingface.co. Connect to the internet or place the required GGUF files into .\models\ before running install.ps1 again."
    }

    foreach ($definition in $script:ModelDefinitions) {
        $state = Test-ModelReady -Definition $definition
        if ($state.Ready) {
            continue
        }

        $download = Download-Model -Definition $definition
        if ($download.Status -eq "ready") {
            Write-Okay $download.Message
            continue
        }

        if ($download.Status -eq "busy") {
            throw "$($download.Message) Wait for the current install/download process to finish."
        }

        throw $download.Message
    }
}

function Get-PortState {
    param([int]$Port)

    $client = $null
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $iar = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        $connected = $iar.AsyncWaitHandle.WaitOne(1500, $false) -and $client.Connected
        return $connected
    }
    catch {
        return $false
    }
    finally {
        if ($client) {
            $client.Close()
        }
    }
}

function Write-ReadinessSummary {
    param(
        [string]$PythonVersion,
        [hashtable]$PythonCommand,
        [pscustomobject]$LlamaServer,
        [pscustomobject]$RuntimeProfile
    )

    Write-Host ""
    Write-Host ("=" * 64)
    Write-Host "RRRIE-CDSS install summary" -ForegroundColor Green
    Write-Host ("=" * 64)
    Write-Host ("Python        : {0}" -f $PythonVersion)
    Write-Host ("Python path   : {0}" -f $PythonCommand.Executable)
    Write-Host ("Virtualenv    : {0}" -f $script:VenvPath)
    Write-Host ("Dependencies  : verified")
    Write-Host ("llama-server  : {0}" -f $LlamaServer.Path)
    Write-Host ("Llama source  : {0}" -f $LlamaServer.Source)

    foreach ($definition in $script:ModelDefinitions) {
        $state = Test-ModelReady -Definition $definition
        $sizeGb = [math]::Round($state.Size / 1GB, 2)
        Write-Host ("Model ({0})   : {1} ({2} GB)" -f $definition.Role, $definition.FileName, $sizeGb)
    }

    $profileLabel = if ($null -eq $RuntimeProfile.DetectedVramGb) {
        "$($RuntimeProfile.Name) (VRAM not detected)"
    }
    else {
        "$($RuntimeProfile.Name) (VRAM $($RuntimeProfile.DetectedVramGb) GB)"
    }
    Write-Host ("Runtime       : {0}" -f $profileLabel)
    Write-Host ("Environment   : {0}" -f $script:EnvFile)

    $busyPorts = @()
    foreach ($port in @(7860, 8080, 8081)) {
        if (Get-PortState -Port $port) {
            $busyPorts += $port
        }
    }
    if ($busyPorts.Count -gt 0) {
        Write-Warn ("Ports already in use: {0}" -f ($busyPorts -join ", "))
    }

    Write-Host ""
    Write-Host "Next command:" -ForegroundColor Cyan
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\setup\start.ps1"
    Write-Host ("=" * 64)
}

function Invoke-PostInstallDoctor {
    param([hashtable]$VenvPython)

    Write-Info "Running post-install runtime doctor..."
    $exitCode = Invoke-PythonCommand -PythonCommand $VenvPython -Arguments @((Join-Path $script:ProjectRoot "run.py"), "--doctor")
    if ($exitCode -ne 0) {
        throw "Post-install runtime doctor reported fatal issues. Fix the reported GPU/binary/model problems before using the launcher."
    }
}

try {
    Write-Info "Starting RRRIE-CDSS full install..."
    $script:ModelDefinitions = @(Get-ModelDefinitions)

    $pythonCommand = Resolve-PythonCommand
    if (-not $pythonCommand) {
        throw "Python $PythonMinVersion+ was not found. Install Python 3.11 or 3.12, or place a project-local runtime at .\runtime\python\python.exe, then rerun install.ps1."
    }

    $pythonVersion = Get-DetectedPythonVersion -PythonCommand $pythonCommand
    Write-Info "Detected Python version: $pythonVersion"
    if ([version]$pythonVersion -lt [version]"$PythonMinVersion.0") {
        throw "Python $PythonMinVersion or newer is required."
    }

    if ([version]$pythonVersion -ge [version]"3.13.0") {
        Write-Warn "Python 3.13+ detected. Compatibility with some core libraries (pydantic-settings, google-genai) is EXPERIMENTAL. Python 3.11 or 3.12 is recommended for 'Perfection' setup."
    }

    Test-DiskSpace -Path $script:ProjectRoot -RequiredGb $script:MinDiskSpaceGb

    $llamaServer = Ensure-LlamaServer
    Write-Okay "llama-server ready: $($llamaServer.Path)"

    $runtimeProfile = Resolve-RuntimeProfile
    Write-Info "Selected runtime profile: $($runtimeProfile.Name)"
    Write-Warn $runtimeProfile.Reason

    Ensure-EnvFile -RuntimeProfile $runtimeProfile
    $venvPython = Ensure-Venv -PythonCommand $pythonCommand
    Ensure-Models
    Invoke-PostInstallDoctor -VenvPython $venvPython
    Write-ReadinessSummary -PythonVersion $pythonVersion -PythonCommand $pythonCommand -LlamaServer $llamaServer -RuntimeProfile $runtimeProfile
}
catch {
    Write-Fail $_.Exception.Message
    if (-not $Repair) {
        Write-Warn "If the environment was partially created, rerun: powershell -ExecutionPolicy Bypass -File .\setup\install.ps1 -Repair"
    }
    exit 1
}
