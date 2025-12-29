from __future__ import annotations

import argparse
import logging

import pandas as pd

from heart_disease_mlops.data.download import download_uci_dataset
from heart_disease_mlops.data.preprocess import load_raw, split_train_test, write_splits
from heart_disease_mlops.eda.eda import run_eda
from heart_disease_mlops.logging_utils import configure_logging
from heart_disease_mlops.training import train_and_select


def run_all(*, quick: bool) -> None:
    csv_path, sha = download_uci_dataset()
    logging.getLogger(__name__).info("Downloaded dataset: %s (sha256=%s)", csv_path, sha)

    df = load_raw(csv_path)
    train_df, test_df = split_train_test(df)
    split_paths = write_splits(train_df, test_df)
    logging.getLogger(__name__).info(
        "Wrote splits: %s, %s",
        split_paths.train_csv,
        split_paths.test_csv,
    )

    run_eda(pd.concat([train_df, test_df], ignore_index=True))
    logging.getLogger(__name__).info("Wrote EDA plots")

    result = train_and_select(train_df, test_df, quick=quick)
    logging.getLogger(__name__).info(
        "Best model=%s test_metrics=%s saved=%s",
        result.best_model_name,
        result.test_metrics,
        result.model_path,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="heart_disease_mlops")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run download + preprocess + EDA + training")
    run_p.add_argument("--quick", action="store_true", help="Use smaller grids for CI/fast runs")

    args = parser.parse_args(argv)
    configure_logging(logging.INFO)

    if args.cmd == "run":
        run_all(quick=bool(args.quick))


if __name__ == "__main__":
    main()
