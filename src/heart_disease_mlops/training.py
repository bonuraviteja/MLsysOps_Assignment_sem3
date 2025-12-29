from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from heart_disease_mlops.config import FEATURE_COLUMNS, PATHS, RANDOM_SEED, TARGET_COLUMN
from heart_disease_mlops.features import build_preprocessor


@dataclass(frozen=True)
class TrainResult:
    best_model_name: str
    best_params: Dict[str, Any]
    test_metrics: Dict[str, float]
    model_path: Path


def _set_mlflow_local_store(mlruns_dir: Path) -> None:
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_dir.as_posix()}")
    mlflow.set_experiment("heart-disease-uci")


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def train_and_select(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    quick: bool = False,
    mlruns_dir: Path | None = None,
    model_dir: Path | None = None,
) -> TrainResult:
    """Train 2+ models with CV, track all runs in MLflow, and persist the best model pipeline."""

    if mlruns_dir is None:
        mlruns_dir = PATHS.mlruns_dir
    if model_dir is None:
        model_dir = PATHS.model_dir

    _set_mlflow_local_store(mlruns_dir)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].to_numpy()
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].to_numpy()

    pre = build_preprocessor()

    models: Dict[str, Tuple[Any, Dict[str, list[Any]]]] = {
        "logreg": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
            {
                "clf__C": [0.1, 1.0] if quick else [0.01, 0.1, 1.0, 10.0],
                "clf__solver": ["liblinear"],
            },
        ),
        "rf": (
            RandomForestClassifier(random_state=RANDOM_SEED),
            {
                "clf__n_estimators": [100] if quick else [200, 500],
                "clf__max_depth": [None, 5] if quick else [None, 5, 10],
                "clf__min_samples_split": [2] if quick else [2, 5],
            },
        ),
    }

    # Ensure CV works even on small datasets (e.g., unit tests)
    _, counts = np.unique(y_train, return_counts=True)
    min_class = int(counts.min()) if len(counts) else 2
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    best_name = ""
    best_auc = -1.0
    best_pipeline: Pipeline | None = None
    best_params: Dict[str, Any] = {}

    for model_name, (estimator, param_grid) in models.items():
        pipeline = Pipeline(steps=[("pre", pre), ("clf", estimator)])
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            refit=True,
        )

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("quick", quick)
            search.fit(X_train, y_train)

            mlflow.log_params({k: v for k, v in search.best_params_.items()})
            mlflow.log_metric("cv_best_roc_auc", float(search.best_score_))

            y_prob = search.predict_proba(X_test)[:, 1]
            m = _metrics(y_test, y_prob)
            mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})

            # Persist model as artifact in MLflow as well (with signature)
            input_example = X_train.head(5)
            try:
                example_pred = search.best_estimator_.predict_proba(input_example)[:, 1]
                signature = infer_signature(input_example, example_pred)
            except Exception:  # noqa: BLE001
                signature = None
            mlflow.sklearn.log_model(
                search.best_estimator_,
                artifact_path="model",
                input_example=input_example,
                signature=signature,
            )

        if float(search.best_score_) > best_auc:
            best_auc = float(search.best_score_)
            best_name = model_name
            best_pipeline = search.best_estimator_
            best_params = dict(search.best_params_)

    assert best_pipeline is not None

    # Final persistence (reusable format) + metadata
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(best_pipeline, model_path)

    y_prob_best = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = _metrics(y_test, y_prob_best)

    meta = {
        "best_model": best_name,
        "best_params": best_params,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "test_metrics": test_metrics,
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return TrainResult(
        best_model_name=best_name,
        best_params=best_params,
        test_metrics=test_metrics,
        model_path=model_path,
    )
