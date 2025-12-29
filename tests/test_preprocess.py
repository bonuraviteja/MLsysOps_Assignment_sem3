from __future__ import annotations

import pandas as pd

from heart_disease_mlops.data.preprocess import split_train_test


def test_split_train_test_stratified() -> None:
    df = pd.DataFrame(
        {
            "age": [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
            "sex": [1] * 10,
            "cp": [0] * 10,
            "trestbps": [120] * 10,
            "chol": [200] * 10,
            "fbs": [0] * 10,
            "restecg": [0] * 10,
            "thalach": [150] * 10,
            "exang": [0] * 10,
            "oldpeak": [1.0] * 10,
            "slope": [1] * 10,
            "ca": [0] * 10,
            "thal": [2] * 10,
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    train_df, test_df = split_train_test(df, test_size=0.2)
    assert set(train_df["target"].unique()) == {0, 1}
    assert set(test_df["target"].unique()) == {0, 1}
