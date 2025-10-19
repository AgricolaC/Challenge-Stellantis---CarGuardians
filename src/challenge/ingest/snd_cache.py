import pandas as pd
from pathlib import Path

def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        csv = path.with_suffix(".csv")
        df.to_csv(csv, index=False)
        return csv


def cache_exists(path: Path) -> bool:
    return path.exists() or path.with_suffix(".csv").exists()
