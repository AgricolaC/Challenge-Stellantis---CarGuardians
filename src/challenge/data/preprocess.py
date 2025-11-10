# In src/challenge/data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.linear_model import Ridge

class ScaniaPreprocessor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible preprocessor for the Scania APS dataset.
    
    This transformer will:
    1. Identify columns to drop (high NA, constant).
    2. Identify imputation strategies for remaining columns (median, MICE).
    3. Fit imputers on training data.
    4. Transform train and test data.
    
    --- UPDATED with higher max_iter and alpha ---
    """
    def __init__(self, na_drop_thresh=0.7, low_na_thresh=0.15):
        self.na_drop_thresh = na_drop_thresh
        self.low_na_thresh = low_na_thresh
        
        # --- FIX ---
        # Increased alpha for better stability (fixes LinAlgWarning)
        self.mice_estimator = Ridge(alpha=1.0) 
        self.max_iter = 20 # Increased iterations (fixes ConvergenceWarning)
        # --- END FIX ---
        
        self.columns_to_keep_ = None
        self.na_cols_to_drop_ = None
        self.constant_cols_to_drop_ = None
        self.low_na_cols_ = None
        self.high_na_cols_ = None
        self.median_imputer_ = None
        self.mice_imputer_ = None

    def fit(self, X, y=None):
        na_pct = X.isna().mean()
        self.na_cols_to_drop_ = set(na_pct[na_pct > self.na_drop_thresh].index)
        self.constant_cols_to_drop_ = set(X.columns[X.nunique() == 1])
        
        cols_to_drop = self.na_cols_to_drop_.union(self.constant_cols_to_drop_)
        self.columns_to_keep_ = [col for col in X.columns if col not in cols_to_drop]
        
        X_filtered = X[self.columns_to_keep_]
        
        na_pct_filtered = X_filtered.isna().mean()
        self.low_na_cols_ = list(na_pct_filtered[(na_pct_filtered > 0) & (na_pct_filtered <= self.low_na_thresh)].index)
        self.high_na_cols_ = list(na_pct_filtered[na_pct_filtered > self.low_na_thresh].index)
        
        if self.low_na_cols_:
            self.median_imputer_ = SimpleImputer(strategy='median')
            self.median_imputer_.fit(X_filtered[self.low_na_cols_])
            
        if self.high_na_cols_:
            self.mice_imputer_ = IterativeImputer(
                estimator=self.mice_estimator,
                max_iter=self.max_iter, # Use the new class attribute
                random_state=42,
                imputation_order='ascending',
                verbose=0 # Also good to silence the imputer's own chatter
            )
            self.mice_imputer_.fit(X_filtered[self.high_na_cols_])
            
        return self

    def transform(self, X, y=None):
        X_transformed = X[self.columns_to_keep_].copy()
        
        if self.median_imputer_ and self.low_na_cols_:
            X_transformed[self.low_na_cols_] = self.median_imputer_.transform(X_transformed[self.low_na_cols_])
            
        if self.mice_imputer_ and self.high_na_cols_:
            X_transformed[self.high_na_cols_] = self.mice_imputer_.transform(X_transformed[self.high_na_cols_])
            
        if X_transformed.isna().any().any():
            remaining_na_cols = X_transformed.columns[X_transformed.isna().any()].tolist()
            final_imputer = SimpleImputer(strategy='median')
            X_transformed[remaining_na_cols] = final_imputer.fit_transform(X_transformed[remaining_na_cols])
            
        return X_transformed