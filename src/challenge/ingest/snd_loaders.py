from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd

# Minimal dtypes for keys; feature dtypes will be inferred
DEFAULT_DTYPES: Dict[str, str] = {"vehicle_id": "int64", "time_step": "int32"}

def read_csv_fast(path: Path, usecols: Optional[List[str]] = None,
                  dtypes: Optional[Dict[str, str]] = None,
                  chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    CSV reader with dtype control, 'na' handling, optional chunking.
    """
    dtypes = dtypes or DEFAULT_DTYPES
    if chunksize:
        it = pd.read_csv(path, usecols=usecols, dtype=dtypes, na_values=["na"], chunksize=chunksize)
        return pd.concat(it, ignore_index=True)
    return pd.read_csv(path, usecols=usecols, dtype=dtypes, na_values=["na"])

def _replace_encoded_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Some exports encode NaNs as large ints ~2.13e9; normalize to proper NaN if present."""
    return df.replace([2130706432, 2130706433, 2130706434], np.nan)

def load_operational_readouts(path: Path, chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    Operational readouts (time series): columns include vehicle_id, time_step, 107 columns total
    with 14 variables (8 counters + 6 histogram variables across bins). Missing <~1% per feature. 
    Source: dataset & paper. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}
    """
    df = read_csv_fast(path, chunksize=chunksize)
    return _replace_encoded_nans(df)

def load_specifications(path: Path) -> pd.DataFrame:
    """
    Vehicle-level categorical specs (~8 features), no missing values per split. 
    Source: dataset & paper. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
    """
    df = read_csv_fast(path)
    for c in df.columns:
        if c != "vehicle_id":
            df[c] = df[c].astype("category")
    return df

def load_tte(path: Path) -> pd.DataFrame:
    """
    Train TTE file with columns:
      - vehicle_id
      - length_of_study_time_step
      - in_study_repair (0/1)
    Source: dataset & paper. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
    """
    return read_csv_fast(path)

def load_labels(path: Path) -> pd.DataFrame:
    """
    Validation labels (vehicle_id + class_label ∈ {0..4}) for the randomly selected last readout per vehicle.
    Source: dataset. :contentReference[oaicite:6]{index=6}
    """
    return read_csv_fast(path)
