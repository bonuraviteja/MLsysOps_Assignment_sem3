from __future__ import annotations

import pandas as pd

from src.heart_disease_mlops.training import train_and_select


def test_train_and_select_smoke(tmp_path, monkeypatch) -> None:
    # Tiny dataset to ensure training code path works.
    df = pd.DataFrame(
        {
            "age": [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
            "sex": [1, 0] * 5,
            "cp": [0, 1] * 5,
            "trestbps": [120, 130] * 5,
            "chol": [200, 220] * 5,
            "fbs": [0, 1] * 5,
            "restecg": [0, 1] * 5,
            "thalach": [150, 140] * 5,
            "exang": [0, 1] * 5,
            "oldpeak": [1.0, 2.0] * 5,
            "slope": [1, 2] * 5,
            "ca": [0, 1] * 5,
            "thal": [2, 3] * 5,
            "target": [0, 1] * 5,
        }
    )
    train_df = df.iloc[:8].reset_index(drop=True)
    test_df = df.iloc[8:].reset_index(drop=True)

    result = train_and_select(
        train_df,
        test_df,
        quick=True,
        mlruns_dir=tmp_path / "mlruns",
        model_dir=tmp_path / "model",
    )
    assert result.model_path.exists()
