from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from heart_disease_mlops.config import FEATURE_COLUMNS
from heart_disease_mlops.serving.app import _load_model


def main() -> None:
    os.environ.setdefault("MODEL_PATH", str(Path("artifacts/model/model.joblib").resolve()))
    model = _load_model()

    sample = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }
    df = pd.DataFrame([sample], columns=FEATURE_COLUMNS)
    print(float(model.predict_proba(df)[:, 1][0]))


if __name__ == "__main__":
    main()
