from imblearn.combine import SMOTEENN
from sklearn.preprocessing import MinMaxScaler

def scale_and_balance(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    smote_enn = SMOTEENN(random_state=42)
    X_res, y_res = smote_enn.fit_resample(X_scaled, y)
    print(f"After balancing: {sum(y_res==0)} negatives, {sum(y_res==1)} positives.")
    return X_res, y_res,scaler

import pandas as pd
import numpy as np  # <-- Make sure numpy is imported
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from typing import Union # <-- Import Union for type hinting

def balance_with_copula(X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray]):
    """
    Balances the training data by synthesizing new samples for the minority class
    using the SDV GaussianCopulaSynthesizer.
    
    Assumes X_train is fully imputed.
    Accepts y_train as a pd.Series or np.ndarray.
    """
    
    # --- START FIX ---
    # Ensure y_train is a pandas Series for clean joining and index alignment.
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    # --- END FIX ---

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
    
    num_to_synthesize = n_majority - n_minority
    
    if num_to_synthesize <= 0:
        print("Data is already balanced or has more positive samples. No synthesis needed.")
        return X_train, y_train
        
    print(f"Balancing: {n_minority} minority, {n_majority} majority. Synthesizing {num_to_synthesize} samples...")

    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=minority_data)
    
    # Fit synthesizer
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(minority_data)
    
    # Generate new samples
    synthetic_samples = synthesizer.sample(num_rows=num_to_synthesize)
    
    # Combine original data with synthetic samples
    data_resampled = pd.concat([data, synthetic_samples], ignore_index=True)
    
    # Shuffle
    data_resampled = data_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balancing complete. New dataset size: {len(data_resampled)}")
    
    # Separate back into X and y
    y_resampled = data_resampled[target_col]
    X_resampled = data_resampled.drop(columns=[target_col])
    
    return X_resampled, y_resampled