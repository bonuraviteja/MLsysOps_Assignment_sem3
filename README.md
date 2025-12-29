# Heart Disease UCI — End-to-End MLOps

This repository implements the complete assignment requirements end-to-end:
- Data download + preprocessing + EDA plots
- Train 2+ models (Logistic Regression, Random Forest), cross-validation, metrics
- MLflow experiment tracking (params/metrics/artifacts/plots)
- Reproducible preprocessing pipeline + packaged model
- FastAPI `/predict` service (Docker + Kubernetes manifests)
- Request logging + Prometheus metrics
- CI pipeline (lint, unit tests, train)

## 1) Setup (Windows PowerShell)

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 2) Run the full pipeline locally

```powershell
python -m heart_disease_mlops.pipeline run
```

If your network intercepts TLS (common on corporate networks) and UCI download fails, try:

```powershell
$env:UCI_ALLOW_INSECURE_SSL = "1"
python -m heart_disease_mlops.pipeline run
```

Outputs (generated):
- `data/raw/heart_disease.csv`
- `data/processed/train.csv`, `data/processed/test.csv`
- `artifacts/eda/*` (plots)
- `artifacts/mlruns/*` (MLflow file store)
- `artifacts/model/model.joblib` (packaged model pipeline)

Notebook (optional, for deliverables):
- `notebooks/assignment_walkthrough.ipynb`

## 3) Start MLflow UI (optional)

```powershell
mlflow ui --backend-store-uri "file:$(Resolve-Path artifacts/mlruns)" --port 5000
```

## 4) Run the API locally (no Docker)

```powershell
$env:MODEL_PATH = "$(Resolve-Path artifacts/model/model.joblib)"
uvicorn heart_disease_mlops.serving.app:app --host 0.0.0.0 --port 8000
```

If port `8000` is already in use, run on `8001`:

```powershell
uvicorn heart_disease_mlops.serving.app:app --host 0.0.0.0 --port 8001
```

In a separate PowerShell terminal (leave Uvicorn running):

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz

$body = @{
  age = 63; sex = 1; cp = 3; trestbps = 145; chol = 233; fbs = 1; restecg = 0; thalach = 150;
  exang = 0; oldpeak = 2.3; slope = 0; ca = 0; thal = 1
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType application/json -Body $body
(Invoke-WebRequest http://127.0.0.1:8000/metrics -UseBasicParsing).Content -split "`n" | Select-Object -First 30
```

## 5) Docker

```powershell
docker build -t heart-mlops:latest .
docker run --rm -p 8000:8000 -e MODEL_PATH=/app/artifacts/model/model.joblib heart-mlops:latest
```

## 6) Kubernetes (local Minikube)

```powershell
docker build -t heart-mlops:latest .
minikube image load heart-mlops:latest
kubectl apply -f deploy/k8s/api.yaml
```

For Minikube you may need:

```powershell
minikube tunnel
```

## 7) Monitoring (Prometheus + Grafana)

```powershell
cd deploy/monitoring
docker compose up --build
```

Grafana: `http://localhost:3000` (default login: `admin` / `admin`)

## 8) Tests + Lint

```powershell
ruff check .
pytest
```

## Deliverables placeholders

- Screenshots: `screenshots/`
- Report: generate `reports/final_report.docx` via `python scripts/generate_report_docx.py`
- Video: `video/README.md`
