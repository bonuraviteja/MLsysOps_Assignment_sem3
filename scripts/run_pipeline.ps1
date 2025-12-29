param(
  [switch]$Quick
)

$ErrorActionPreference = "Stop"

& .\.venv\Scripts\Activate.ps1

if ($Quick) {
  python -m heart_disease_mlops.pipeline run --quick
} else {
  python -m heart_disease_mlops.pipeline run
}
