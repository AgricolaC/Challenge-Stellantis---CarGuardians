import numpy as np
import pandas as pd

# Cost matrix (rows = actual, cols = predicted), from the challenge doc. :contentReference[oaicite:9]{index=9}
COST = np.array([
# pred:  0    1    2    3    4
  [   0,   7,   8,   9,  10],  # actual 0
  [ 200,   0,   7,   8,   9],  # actual 1
  [ 300, 200,   0,   7,   8],  # actual 2
  [ 400, 300, 200,   0,   7],  # actual 3
  [ 500, 400, 300, 200,   0],  # actual 4
], dtype=np.float64)

def total_cost(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """
    Sum of COST[true, pred] across instances (multiclass 0..4).
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred shapes must match")
    return float(COST[yt, yp].sum())

def confusion_cost_table(y_true, y_pred) -> pd.DataFrame:
    """Return a DataFrame with counts and per-cell costs."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    counts = pd.crosstab(yt, yp, dropna=False).reindex(index=range(5), columns=range(5), fill_value=0)
    costs  = pd.DataFrame(COST, index=range(5), columns=range(5))
    return pd.concat({"count": counts, "cost": costs}, axis=1)
