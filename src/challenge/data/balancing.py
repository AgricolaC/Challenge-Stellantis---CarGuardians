from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler

def scale_and_balance(X, y, sampling_strategy='auto'):
    """
    Scales the data using StandardScaler and balances it using SMOTEENN.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        sampling_strategy: Sampling strategy for SMOTEENN. 
                           'auto' resamples all classes but the majority class.
                           float (e.g., 0.5) corresponds to the desired ratio of the 
                           number of samples in the minority class over the number 
                           of samples in the majority class after resampling.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SMOTEENN combines over- and under-sampling. 
    # Note: SMOTEENN's sampling_strategy applies to the SMOTE part.
    smote_enn = SMOTEENN(random_state=42, sampling_strategy=sampling_strategy)
    X_res, y_res = smote_enn.fit_resample(X_scaled, y)
    
    print(f"After balancing: {sum(y_res==0)} negatives, {sum(y_res==1)} positives.")
    return X_res, y_res, scaler

import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from typing import Union

def balance_with_copula(X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray], sampling_strategy: float = 1.0):
    """
    Balances the training data by synthesizing new samples for the minority class
    using the SDV GaussianCopulaSynthesizer.
    
    IMPORTANT: This function must only be called on the TRAINING set to avoid data leakage.
    
    Args:
        X_train: Training features (fully imputed).
        y_train: Training labels.
        sampling_strategy: Desired ratio of minority samples to majority samples after synthesis.
                           e.g., 1.0 = equal number (balanced), 0.5 = 1 minority for every 2 majority.
    """
    # Check for NaNs
    if X_train.isna().any().any():
        raise ValueError("X_train contains NaNs. Please impute missing values before balancing.")

    # Ensure y_train is a pandas Series for clean joining and index alignment.
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    # Combine X and y into one DataFrame
    # Now we are guaranteed both are pandas objects
    X_train_ = X_train.reset_index(drop=True)
    y_train_ = y_train.reset_index(drop=True) 
    
    target_col = 'target' 
    if target_col in X_train_.columns:
        target_col = 'target_label' 
        
    data = X_train_.join(y_train_.rename(target_col))
    
    minority_data = data[data[target_col] == 1]
    majority_data = data[data[target_col] == 0]
    
    n_minority = len(minority_data)
    n_majority = len(majority_data)
    
    # Calculate target number of minority samples based on strategy
    target_n_minority = int(n_majority * sampling_strategy)
    num_to_synthesize = target_n_minority - n_minority
    
    if num_to_synthesize <= 0:
        print(f"Data already meets target ratio ({n_minority}/{n_majority} >= {sampling_strategy}). No synthesis needed.")
        return X_train, y_train
        
    print(f"Balancing: {n_minority} minority, {n_majority} majority.")
    print(f"Target ratio: {sampling_strategy} -> Target minority count: {target_n_minority}")
    print(f"Synthesizing {num_to_synthesize} samples...")

    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=minority_data)
    
    # Fit synthesizer
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(minority_data)
    
    # Generate new samples
    synthetic_samples = synthesizer.sample(num_rows=num_to_synthesize)
    
    # --- Post-Processing: Fix Physics & Logic Violations ---
    print("Post-processing synthetic data to enforce physical constraints...")
    
    # 1. Fix Binary Flags (e.g., _is_missing)
    # The Copula generates floats like 0.42. We must round them.
    binary_cols = [c for c in data.columns if 'is_missing' in c]
    # Also check for other binary columns (0/1 only)
    for col in data.columns:
        if col not in binary_cols and col != target_col:
            unique_vals = data[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                 binary_cols.append(col)
    
    binary_cols = list(set(binary_cols)) # Remove duplicates
    
    for col in binary_cols:
        if col in synthetic_samples.columns:
            synthetic_samples[col] = synthetic_samples[col].round().clip(0, 1)

    # 2. Fix Integer Counts (Histograms)
    # Counts cannot be negative or fractional
    # We assume float columns that are not binary are potentially histograms or continuous vars
    # A safe heuristic for this dataset: if input was integer, output should be integer.
    # However, data frame might have been converted to float during SimpleImputer.
    # We'll rely on the column naming convention for histograms (ends in digit) or just enforce non-negativity for all.
    
    for col in synthetic_samples.columns:
        if col == target_col or col in binary_cols:
            continue
            
        # Enforce non-negativity for physical sensors/counts
        synthetic_samples[col] = synthetic_samples[col].apply(lambda x: max(0, x))
        
        # If it looks like a histogram bin (ends in digit) or was integer-like, we could round.
        # For now, non-negative is the most critical physics fix.
        # Let's also round if it's a histogram bin to keep it as a 'count'
        if col[-1].isdigit(): # ag_000, etc.
             synthetic_samples[col] = synthetic_samples[col].round()

    # Combine original data with synthetic samples
    data_resampled = pd.concat([data, synthetic_samples], ignore_index=True)
    
    # Shuffle
    data_resampled = data_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balancing complete. New dataset size: {len(data_resampled)}")
    
    # Separate back into X and y
    y_resampled = data_resampled[target_col]
    X_resampled = data_resampled.drop(columns=[target_col])
    
    return X_resampled, y_resampled