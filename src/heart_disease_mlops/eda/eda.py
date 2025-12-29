from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_disease_mlops.config import FEATURE_COLUMNS, PATHS, TARGET_COLUMN


def run_eda(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = PATHS.eda_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Class balance
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=TARGET_COLUMN, data=df)
    ax.set_title("Class Balance (Target)")
    ax.set_xlabel("Heart Disease Present")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=200)
    plt.close()

    # Histograms for numeric-ish features
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram: {col}")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{col}.png", dpi=200)
        plt.close()

    # Correlation heatmap
    corr_cols = [c for c in FEATURE_COLUMNS if c in numeric_cols] + [TARGET_COLUMN]
    corr_df = df[corr_cols].corr(numeric_only=True)
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (numeric features + target)")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    return output_dir
