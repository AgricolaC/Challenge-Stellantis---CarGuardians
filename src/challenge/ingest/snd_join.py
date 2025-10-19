from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd

def merge_op_with_spec(op: pd.DataFrame, spec: pd.DataFrame) -> pd.DataFrame:
    """Left-join static specs to time-series rows on vehicle_id."""
    if "vehicle_id" not in op or "vehicle_id" not in spec:
        raise KeyError("vehicle_id required in both tables")
    return op.merge(spec, on="vehicle_id", how="left")

def snapshot_last_readout(op: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single-row snapshot per vehicle by taking the last available readout (max time_step).
    This mirrors validation/test 'last-readout' concept where a final readout is chosen at random
    (we use the latest present in file as a stable proxy for one-per-vehicle view). :contentReference[oaicite:7]{index=7}
    """
    idx = op.groupby("vehicle_id")["time_step"].idxmax()
    return op.loc[idx].copy()

def attach_vehicle_labels(df_vehicle: pd.DataFrame, labels: pd.DataFrame,
                          label_col: str) -> pd.DataFrame:
    """Attach a vehicle-level label column to a per-vehicle table."""
    if "vehicle_id" not in df_vehicle or "vehicle_id" not in labels:
        raise KeyError("vehicle_id required")
    lab = labels[["vehicle_id", label_col]].drop_duplicates("vehicle_id")
    return df_vehicle.merge(lab, on="vehicle_id", how="left")

def tte_to_class(tte: pd.Series) -> pd.Series:
    """
    Map time-to-event to the 5 classes used in validation/test:
      0: >48, 1: (48,24], 2: (24,12], 3: (12,6], 4: (6,0].
    Source: dataset & evaluation. :contentReference[oaicite:8]{index=8}
    """
    t = pd.to_numeric(tte, errors="coerce")
    cls = pd.Series(index=t.index, dtype="Int64")
    cls[t > 48] = 0
    cls[(t <= 48) & (t > 24)] = 1
    cls[(t <= 24) & (t > 12)] = 2
    cls[(t <= 12) & (t > 6)]  = 3
    cls[(t <= 6)  & (t >= 0)] = 4
    return cls

def make_binary_from_class(class_series: pd.Series, positives: Sequence[int]) -> pd.Series:
    """Map multi-class {0..4} to binary given a set of 'positive' classes (e.g., {3,4})."""
    return class_series.isin(set(positives)).astype("Int64")
