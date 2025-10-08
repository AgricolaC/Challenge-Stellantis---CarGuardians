from pathlib import Path
import os

def project_root() -> Path:
    """Return repository root based on this file location."""
    return Path(__file__).resolve().parents[3]

def data_root() -> Path:
    """Return the /data root (override with DATA_ROOT env var)."""
    return Path(os.getenv("DATA_ROOT", project_root() / "data"))

def raw_dir() -> Path:
    """Path to raw data directory."""
    return data_root() / "raw"

def processed_dir() -> Path:
    """Path to downloaded processed data directory."""
    return data_root() / "processed"

def house_processed_dir() -> Path:
    """Path to in-house processed artifacts directory."""
    return data_root() / "house_processed"
