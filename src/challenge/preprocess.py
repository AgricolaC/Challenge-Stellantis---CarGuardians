import numpy as np
import pandas as pd

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))

def peak_to_peak(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.max(x) - np.min(x))

# def basic_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    
