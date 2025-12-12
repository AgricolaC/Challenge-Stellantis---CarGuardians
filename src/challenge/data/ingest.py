import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional

def load_data(data_path: str, file_name: str):
    """
    Loads a single Scania APS dataset file (train or test).
    
    - Constructs the full file path.
    - Replaces 'na' strings with np.nan.
    - Identifies the 'class' column as the target.
    - Converts target 'neg' to 0 and 'pos' to 1.
    - Separates and returns features (X) and target (y).
    
    Args:
        data_path: Directory containing data
        file_name: Filename
        return_outliers: If True, returns (X, y, outliers_df). Default False -> (X, y)
    """
    # Construct the full file path
    full_path = os.path.join(data_path, file_name)

    # Load the dataset
    try:
        data = pd.read_csv(full_path, na_values='na') 
    except FileNotFoundError:
        print(f"Error: File not found at {full_path}")
        return None, None
    except Exception as e:
        print(f"Error loading {full_path}: {e}")
        return None, None
    
    target_col = 'class'
    
    if target_col not in data.columns:
        print(f"Error: Target column '{target_col}' not found in {file_name}.")
        print(f"Available columns are: {data.columns.tolist()}")
        return None, None
    try:
        y = data[target_col].map({'neg': 0, 'pos': 1})
        if y.isna().any():
            print(f"Warning: Found unexpected values in 'class' column of {file_name}.")
    except Exception as e:
        print(f"Error mapping target column: {e}")
        return None, None

    # Drop the target column to create the X (features) DataFrame
    X = data.drop(columns=[target_col])
    
    print(f"Successfully loaded {file_name}. X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y

