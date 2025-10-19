from pathlib import Path
import pandas as pd
from __future__ import annotations
from typing import Tuple, Optional, Dict
import numpy as np
from sklearn.model_selection import train_test_split
"""
Scania APS preparation utilities:
- normalize missing values ('na' -> NaN)
- numeric coercion
- label encoding (class: 'pos'/'neg' -> 1/0)
- simple imputations (median + missingness flags)
- stratified split for validation
"""

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning for tabular processed data.
    - Strip whitespace from column names
    - Replace 'na' strings with np.nan
    - Forward-fill then back-fill missing values
    """
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    out = out.replace("na", np.nan)
    out = out.fillna(method="ffill").fillna(method="bfill")
    return out



def save_house_processed(df: pd.DataFrame, name: str, out_dir: Path) -> Path:
    """
    Save an in-house artifact to `data/house_processed/` as parquet or csv.

    Parameters
    df : pd.DataFrame
    name : str
        Base filename without extension.
    out_dir : Path
        Typically `paths.house_processed_dir()`.

    Returns
    Path
        Path to the saved artifact.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        csv_path = out_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path



def split_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and labels from training dataframe.

    Returns
    -------
    X : DataFrame (features only)
    y : Series   (1 for 'pos', 0 for 'neg')
    """
    if "class" not in df.columns:
        raise KeyError("Column 'class' not found; did you load the training set?")
    y = df["class"].map({"pos": 1, "neg": 0}).astype("Int64")
    X = df.drop(columns=["class"])
    return X, y


def coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all feature columns to numeric where possible, preserving NaN.
    Keeps non-numeric (if any) as-is (e.g., leftover IDs).
    """
    out = X.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def impute_with_median_and_flags(
    X: pd.DataFrame, precomputed_medians: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Median-impute numeric columns and add *_was_missing indicator columns.

    Returns
    -------
    X_imp : DataFrame (imputed + indicators)
    medians : dict of medians used for numeric columns (to reuse on val/test)
    """
    X = X.copy()
    medians = precomputed_medians or {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            miss = X[c].isna()
            if c not in medians:
                medians[c] = float(X[c].median(skipna=True)) if miss.any() else float(X[c].median())
            if miss.any():
                X[f"{c}__was_missing"] = miss.astype(int)
                X[c] = X[c].fillna(medians[c])
    return X, medians


def train_val_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified split to preserve the rare 'pos' class proportion.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


