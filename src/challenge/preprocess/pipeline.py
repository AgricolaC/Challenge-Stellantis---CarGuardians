from pathlib import Path
import pandas as pd

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning for tabular processed data.

    Suggested steps (to implement):
    - Drop all-NaN columns
    - Forward/backward fill short gaps
    - Type coercion / unit normalization (if documented)
    """
    raise NotImplementedError

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
    raise NotImplementedError
