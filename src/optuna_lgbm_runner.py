"""Docker-friendly runner for experiments/detecting-anomalies-optuna-lgbm.ipynb.

This keeps the notebook's core flow intact:
load Kaggle CSVs, drop simple unusable columns, ordinal-encode categoricals,
train tuned LightGBM, tune an F2 threshold, and write a submission CSV.
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

LOGGER = logging.getLogger(__name__)

KAGGLE_COMPETITION_DIR = Path("/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems")
DEFAULT_TRAIN_PATH = KAGGLE_COMPETITION_DIR / "train.csv"
DEFAULT_TEST_PATH = KAGGLE_COMPETITION_DIR / "test.csv"
DEFAULT_OUTPUT_PATH = Path("/kaggle/working/submission_LGBM.csv")

HIGH_NULL_DROP_COLS = [
    "DERCtlAC[0].PFWAbs.Ext",
    "DERCtlAC[0].PFWAbsRvrt.Ext",
    "DERMeasureAC[0].ThrotPct",
    "DERMeasureAC[0].ThrotSrc",
]

TUNED_LGBM_PARAMS = {
    "n_estimators": 3401,
    "learning_rate": 0.020009612153555324,
    "num_leaves": 76,
    "max_depth": 10,
    "min_child_samples": 35,
    "subsample": 0.9974353945552556,
    "colsample_bytree": 0.6161939987092331,
    "reg_alpha": 0.1695611207180594,
    "reg_lambda": 0.16021326556236573,
}

DEFAULT_LGBM_PARAMS = {
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "device_type": "cpu",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Optuna-tuned LightGBM anomaly notebook as a batch job.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-rows", type=int, default=0, help="Use only the first N train/test rows for a quick smoke test.")
    parser.add_argument("--n-estimators", type=int, default=None, help="Override tuned n_estimators, useful for smoke tests.")
    return parser.parse_args()


def read_csv(path: Path, limit_rows: int = 0) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pl.read_csv(path)
    if limit_rows > 0:
        df = df.head(limit_rows)
    return df


def present_columns(df: pl.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def build_preprocessor(cat_cols: list[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        [
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            )
        ]
    )
    return ColumnTransformer(
        transformers=[("cat", cat_pipe, cat_cols)],
        remainder="passthrough",
    )


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_score = 0.0
    best_threshold = 0.0
    for threshold in [i / 100 for i in range(10, 90)]:
        preds = (y_prob > threshold).astype(int)
        score = fbeta_score(y_true, preds, beta=2)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def run() -> None:
    args = parse_args()
    LOGGER.info("loading train data from %s", args.train_path)
    train_df = read_csv(args.train_path, args.limit_rows)
    LOGGER.info("train shape before cleanup: %s", train_df.shape)

    unique_dict = {column: train_df[column].n_unique() for column in train_df.columns}
    unique_count_1_cols = [column for column, count in unique_dict.items() if count == 1]
    unique_count_1_cols = [column for column in unique_count_1_cols if column not in {"Id", "Label"}]
    if unique_count_1_cols:
        train_df = train_df.drop(unique_count_1_cols)
    LOGGER.info("dropped %d constant columns", len(unique_count_1_cols))

    high_null_drop_cols = present_columns(train_df, HIGH_NULL_DROP_COLS)
    if high_null_drop_cols:
        train_df = train_df.drop(high_null_drop_cols)
    LOGGER.info("dropped %d fixed high-null columns", len(high_null_drop_cols))

    train_df = train_df.drop_nulls()
    LOGGER.info("train shape after drop_nulls: %s", train_df.shape)

    x_all = train_df.drop(["Id", "Label"])
    y_all = train_df["Label"]
    cat_cols = x_all.select(pl.col(pl.Utf8)).columns
    LOGGER.info("categorical columns: %d", len(cat_cols))

    x_train, x_val, y_train, y_val = train_test_split(
        x_all,
        y_all,
        test_size=args.validation_size,
        stratify=y_all,
        random_state=args.seed,
    )
    del train_df, x_all, y_all
    gc.collect()

    preprocessor = build_preprocessor(cat_cols)
    LOGGER.info("fitting preprocessing")
    x_train_processed = preprocessor.fit_transform(x_train)
    x_val_processed = preprocessor.transform(x_val)
    del x_train, x_val
    gc.collect()

    model_params = {**TUNED_LGBM_PARAMS, **DEFAULT_LGBM_PARAMS}
    if args.n_estimators is not None:
        model_params["n_estimators"] = args.n_estimators
    LOGGER.info("training LightGBM with params=%s", model_params)
    model = LGBMClassifier(**model_params)
    model.fit(x_train_processed, y_train)

    LOGGER.info("tuning F2 threshold")
    y_val_prob = model.predict_proba(x_val_processed)[:, 1]
    best_threshold, best_score = find_best_threshold(np.asarray(y_val), y_val_prob)
    LOGGER.info("best threshold=%.2f validation_f2=%.6f", best_threshold, best_score)
    del x_train_processed, x_val_processed, y_train, y_val, y_val_prob
    gc.collect()

    LOGGER.info("loading test data from %s", args.test_path)
    test_df = read_csv(args.test_path, args.limit_rows)
    LOGGER.info("test shape: %s", test_df.shape)
    cols_to_remove = present_columns(test_df, [*unique_count_1_cols, *high_null_drop_cols, "Id"])
    x_test = test_df.drop(cols_to_remove)
    x_test_processed = preprocessor.transform(x_test)

    y_test_prob = model.predict_proba(x_test_processed)[:, 1]
    y_pred = (y_test_prob > best_threshold).astype(int)
    submission = pl.DataFrame(
        {
            "Id": test_df["Id"],
            "Label": y_pred,
        }
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(args.output_path)
    LOGGER.info("wrote %s with %d rows", args.output_path, submission.height)


def main() -> None:
    configure_logging()
    run()


if __name__ == "__main__":
    main()
