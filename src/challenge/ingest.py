from pathlib import Path
from typing import Dict, Any
import pandas as pd
from nptdms import TdmsFile

def list_tdms_channels(path: Path) -> Dict[str, list]:
    """Return TDMS group->channels mapping (no full data load)."""
    tdms = TdmsFile.read(path)
    return {g.name: [c.name for c in g.channels()] for g in tdms.groups()}

def read_tdms_channel(path: Path, group: str, channel: str):
    """Load a single TDMS channel as a numpy array."""
    tdms = TdmsFile.read(path)
    return tdms[group][channel][:]

def read_excel(path: Path, sheet: str | int = 0) -> pd.DataFrame:
    """Read performance/emissions Excel sheet."""
    return pd.read_excel(path, sheet_name=sheet)
