param(
    [string]$Destination = "",
    [switch]$IncludeModels,
    [switch]$IncludePythonRuntime
)

$ErrorActionPreference = "Stop"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. (Join-Path $PSScriptRoot "common.ps1")

$projectRoot = Get-ProjectRoot
$mainBinary = Resolve-LlamaServerPath
$legacyBinary = Resolve-LlamaServerLegacyPath
function Test-IsInsideProject {
    param([string]$PathToCheck)

    if (-not $PathToCheck) {
        return $false
    }

    $fullProject = [System.IO.Path]::GetFullPath($projectRoot)
    $fullPath = [System.IO.Path]::GetFullPath($PathToCheck)
    return $fullPath.StartsWith($fullProject, [System.StringComparison]::OrdinalIgnoreCase)
}

if (-not $mainBinary) {
    Write-Fail "Installer-ready mirror requires llama-server.exe inside the project. Put it under .\\llama-server\\ or set LLAMA_SERVER_EXE to a project-local path."
    exit 1
}
if (-not (Test-IsInsideProject $mainBinary)) {
    Write-Fail "Resolved llama-server binary is outside the project root. Move the binary bundle into .\\llama-server\\ before exporting the installer-ready mirror."
    exit 1
}
if ($legacyBinary -and -not (Test-IsInsideProject $legacyBinary)) {
    Write-Fail "Resolved legacy llama-server binary is outside the project root. Move the legacy binary bundle into .\\llama-server\\ before exporting the installer-ready mirror."
    exit 1
}
if (-not $Destination) {
    $Destination = Join-Path $projectRoot "portable\rrrie-cdss-clean"
}

$portableRoot = [System.IO.Path]::GetFullPath($Destination)
if ($portableRoot -eq $projectRoot) {
    Write-Fail "Destination cannot be the project root."
    exit 1
}

if (Test-Path $portableRoot) {
    Remove-Item -Recurse -Force $portableRoot
}

New-Item -ItemType Directory -Path $portableRoot | Out-Null

$excludeDirs = @(
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "output",
    "tmp",
    "portable",
    "clinical-learning-work",
    "_removed_legacy_disabled",
    "tests",
    "docs"
)

if (-not $IncludeModels) {
    $excludeDirs += "models"
}

if (-not $IncludePythonRuntime) {
    $excludeDirs += "runtime"
}

$excludeFiles = @(
    ".env",
    "*.log",
    "*.zip",
    "*.db",
    "*.db-shm",
    "*.db-wal",
    "*.sqlite",
    "*.sqlite-shm",
    "*.sqlite-wal",
    "files_to_clean.txt",
    "git_status*.txt",
    "requirements.locked.txt",
    "swarm_verify.json",
    "test_output*.txt",
    "tmp_charter.txt",
    "tmp_week5.txt"
)

$excludeRelativeDirs = @(
    "data\reports"
)

$excludeRelativeFiles = @(
    "DLLM_IMPLEMENTATION_PLAN.md",
    "R3_IE_OPTIMIZATION_PLAN.md",
    "RRRIE_MASTER_PLAN.md",
    "RRRIE_QWEN4B_MASTER_PLAN.md",
    "test_l1.py",
    "test_regex.py",
    "test_swarm.py",
    "test_swarm_2.py",
    "config\drug_cache.json"
)

function Remove-ExcludedArtifacts {
    param(
        [string]$RootPath,
        [string[]]$DirNames,
        [string[]]$FilePatterns,
        [string[]]$RelativeDirs = @(),
        [string[]]$RelativeFiles = @()
    )

    foreach ($dirName in $DirNames) {
        Get-ChildItem -Path $RootPath -Directory -Force -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ieq $dirName } |
            ForEach-Object {
                Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
            }
    }

    foreach ($pattern in $FilePatterns) {
        Get-ChildItem -Path $RootPath -File -Force -Recurse -Filter $pattern -ErrorAction SilentlyContinue |
            ForEach-Object {
                Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
            }
    }

    foreach ($relativeDir in $RelativeDirs) {
        $fullDir = Join-Path $RootPath $relativeDir
        if (Test-Path $fullDir) {
            Remove-Item -LiteralPath $fullDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    foreach ($relativeFile in $RelativeFiles) {
        $fullFile = Join-Path $RootPath $relativeFile
        if (Test-Path $fullFile) {
            Remove-Item -LiteralPath $fullFile -Force -ErrorAction SilentlyContinue
        }
    }
}

& robocopy $projectRoot $portableRoot /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP `
    /XD $excludeDirs `
    /XF $excludeFiles | Out-Null

if ($LASTEXITCODE -ge 8) {
    Write-Fail "Installer-ready mirror export failed."
    exit $LASTEXITCODE
}

Remove-ExcludedArtifacts `
    -RootPath $portableRoot `
    -DirNames $excludeDirs `
    -FilePatterns $excludeFiles `
    -RelativeDirs $excludeRelativeDirs `
    -RelativeFiles $excludeRelativeFiles

New-Item -ItemType Directory -Path (Join-Path $portableRoot "models") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $portableRoot "output\logs") -Force | Out-Null

if (-not $IncludePythonRuntime) {
    New-Item -ItemType Directory -Path (Join-Path $portableRoot "runtime") -Force | Out-Null
}

$portablePythonPath = Join-Path $portableRoot "runtime\python\python.exe"
if ($IncludePythonRuntime -and -not (Test-Path $portablePythonPath)) {
    Write-Host "Python runtime requested, but .\\runtime\\python\\python.exe was not present in the source project." -ForegroundColor Yellow
}

Write-Host "Installer-ready mirror created at: $portableRoot" -ForegroundColor Green
Write-Host "Included: active source, setup scripts, root docs, project-local llama-server binaries." -ForegroundColor Green
Write-Host "Primary binary: $mainBinary" -ForegroundColor Green
if ($legacyBinary) {
    Write-Host "Legacy binary: $legacyBinary" -ForegroundColor Green
}
else {
    Write-Host "Legacy binary: not bundled; export remains valid for non-legacy targets." -ForegroundColor Yellow
}
if ($IncludeModels) {
    Write-Host "Models: included for offline/runtime bundle use." -ForegroundColor Green
}
else {
    Write-Host "Models: excluded; target machine will download them during install." -ForegroundColor Green
}
if ($IncludePythonRuntime) {
    Write-Host "Python runtime: included from .\\runtime\\python if present." -ForegroundColor Green
}
else {
    Write-Host "Python runtime: excluded; target machine needs Python 3.11/3.12 or a later copied runtime\\python folder." -ForegroundColor Green
}
Write-Host "Excluded: .env, virtualenv, logs, caches, reports, temp files." -ForegroundColor Green
