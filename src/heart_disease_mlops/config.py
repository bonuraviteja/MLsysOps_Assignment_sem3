from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"

    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    eda_dir: Path = artifacts_dir / "eda"
    model_dir: Path = artifacts_dir / "model"
    mlruns_dir: Path = artifacts_dir / "mlruns"


PATHS = Paths()

RANDOM_SEED = 42

UCI_PROCESSED_CLEVELAND_URL_HTTPS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)
UCI_PROCESSED_CLEVELAND_URL_HTTP = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)

UCI_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

FEATURE_COLUMNS = [c for c in UCI_COLUMNS if c != "num"]
TARGET_COLUMN = "target"
