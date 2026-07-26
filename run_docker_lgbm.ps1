$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Image = "der-optuna-lgbm"
$Platform = "linux/amd64"
$CompetitionDir = "/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems"
$DataDir = Join-Path $RepoRoot "data"
$WorkingDir = Join-Path $RepoRoot "kaggle-working"

if (-not (Test-Path $DataDir)) {
    Write-Error "Missing data directory: $DataDir"
}

New-Item -ItemType Directory -Force -Path $WorkingDir | Out-Null

$ImageId = docker image ls -q $Image
if ([string]::IsNullOrWhiteSpace($ImageId)) {
    Write-Error "Docker image $Image was not found. Build it with: docker build --platform $Platform -f Dockerfile.lgbm -t $Image ."
}

docker run --rm `
    --platform $Platform `
    -v "${RepoRoot}:/workspace" `
    -v "${DataDir}:${CompetitionDir}:ro" `
    -v "${WorkingDir}:/kaggle/working" `
    -w /workspace `
    $Image `
    python -m src.optuna_lgbm_runner @args

