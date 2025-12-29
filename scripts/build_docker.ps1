$ErrorActionPreference = "Stop"

if (-not (Test-Path "artifacts/model/model.joblib")) {
  Write-Host "Model not found. Running pipeline first..."
  & .\.venv\Scripts\Activate.ps1
  python -m heart_disease_mlops.pipeline run --quick
}

docker build -t heart-mlops:latest .
