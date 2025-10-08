from pathlib import Path
from typing import Dict, List, Any

def list_channels(tdms_path: Path) -> Dict[str, List[str]]:
    """
    Return mapping of TDMS group -> channel names.

    Parameters
    tdms_path : Path
        Filepath to a .tdms file.

    Returns
    dict
        {group_name: [channel_name, ...], ...}

 
    raise NotImplementedError
    """
def read_channel(tdms_path: Path, group: str, channel: str) -> "np.ndarray":
    """
    Load a single TDMS channel as a 1D numpy array.

    Returns
    -------
    np.ndarray
        Samples for the requested channel.

    
    raise NotImplementedError
