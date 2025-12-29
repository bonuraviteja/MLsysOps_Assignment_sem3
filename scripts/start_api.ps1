$ErrorActionPreference = "Stop"
& .\.venv\Scripts\Activate.ps1

if (-not (Test-Path "artifacts/model/model.joblib")) {
  Write-Host "Model not found. Running pipeline first..."
  python -m heart_disease_mlops.pipeline run --quick
}

$env:MODEL_PATH = "$(Resolve-Path artifacts/model/model.joblib)"
uvicorn heart_disease_mlops.serving.app:app --host 0.0.0.0 --port 8000
