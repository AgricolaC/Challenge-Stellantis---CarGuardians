from pathlib import Path
import pandas as pd

def read_sheet(xlsx_path: Path, sheet: str | int = 0) -> pd.DataFrame:
    """
    Read a single Excel sheet into a DataFrame.

    Parameters
    xlsx_path : Path
        Path to Excel file.
    sheet : str | int
        Sheet name or index.

    Returns
    pd.DataFrame
    """
    raise NotImplementedError

def list_sheets(xlsx_path: Path) -> list[str]:
    """
    List sheet names available in the Excel file.

    Returns
    list[str]
    """
    raise NotImplementedError
