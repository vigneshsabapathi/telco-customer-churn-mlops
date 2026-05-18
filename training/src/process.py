"""Telco churn data processing stage.

Reads the raw IBM Telco-Customer-Churn CSV and produces train/test splits of
features and target. Encoding is done with patsy `dmatrix` so the same formula
can be reused at serving time to guarantee column parity.
"""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from patsy import dmatrix
from sklearn.model_selection import train_test_split


def load_raw(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    return df


def clean(df: pd.DataFrame, drop_cols: list, target: str) -> pd.DataFrame:
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # TotalCharges arrives as object dtype because ~11 new customers have
    # whitespace strings instead of numbers. Coerce and drop those rows.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # SeniorCitizen is 0/1 ints but semantically categorical — cast to string
    # so patsy one-hot-encodes it consistently with the other Yes/No columns.
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    df[target] = df[target].map({"Yes": 1, "No": 0}).astype(int)
    return df


def rename_patsy_columns(X: pd.DataFrame) -> pd.DataFrame:
    """patsy emits `Col[T.value]` for categorical levels — flatten to `Col_value`."""
    X = X.copy()
    X.columns = (
        X.columns.str.replace(r"\[T\.", "_", regex=True)
        .str.replace(r"\[", "_", regex=True)
        .str.replace(r"\]", "", regex=True)
    )
    return X


def encode_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    formula = " + ".join(features) + " - 1"
    X = dmatrix(formula, df, return_type="dataframe")
    X = rename_patsy_columns(X)
    return X


def split_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: DictConfig,
) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        stratify=y if cfg.split.stratify else None,
    )
    Path(cfg.processed.dir).mkdir(parents=True, exist_ok=True)
    X_train.to_csv(cfg.processed.X_train.path, index=False)
    X_test.to_csv(cfg.processed.X_test.path, index=False)
    y_train.to_csv(cfg.processed.y_train.path, index=False)
    y_test.to_csv(cfg.processed.y_test.path, index=False)


def process_data(cfg: DictConfig) -> None:
    raw = load_raw(cfg.raw.path)
    cleaned = clean(raw, list(cfg.process.drop), cfg.process.target)
    X = encode_features(cleaned, list(cfg.process.features))
    y = cleaned[cfg.process.target]
    split_and_save(X, y, cfg)
    print(
        f"process: X_train={X.shape}, "
        f"churn_rate={y.mean():.3f}, "
        f"rows_after_clean={len(cleaned)}"
    )


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig) -> None:
    process_data(cfg)


if __name__ == "__main__":
    main()
