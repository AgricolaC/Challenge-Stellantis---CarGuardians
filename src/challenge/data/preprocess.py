from sklearn.experimental import enable_iterative_imputer  # noqa
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class ScaniaPreprocessor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible preprocessor for the Scania APS dataset.
    
    This transformer will:
    1. Identify columns to drop (high NA, constant).
    2. Identify imputation strategies for remaining columns (median, MICE).
    3. Fit imputers on training data.
    4. Transform train and test data.
    """
    def __init__(self, na_drop_thresh=0.7, low_na_thresh=0.15, mice_estimator=Ridge(alpha=1)):
        self.na_drop_thresh = na_drop_thresh
        self.low_na_thresh = low_na_thresh
        self.mice_estimator = mice_estimator
        self.max_iter = 20
        self.columns_to_keep_ = None
        self.na_cols_to_drop_ = None
        self.constant_cols_to_drop_ = None
        self.low_na_cols_ = None
        self.high_na_cols_ = None
        self.median_imputer_ = None
        self.mice_imputer_ = None
    def fit(self, X, y=None):
        """
        Learns which columns to drop and fits the imputers.
        """
        # 1. Identify columns to drop
        na_pct = X.isna().mean()
        self.na_cols_to_drop_ = set(na_pct[na_pct > self.na_drop_thresh].index)
        
        # We drop rows with <4% missing in training *before* CV, so we don't do it here.
        # But we must drop constant columns.
        self.constant_cols_to_drop_ = set(X.columns[X.nunique() == 1])
        
        cols_to_drop = self.na_cols_to_drop_.union(self.constant_cols_to_drop_)
        self.columns_to_keep_ = [col for col in X.columns if col not in cols_to_drop]
        
        X_filtered = X[self.columns_to_keep_]
        
        # 2. Identify imputation strategies
        na_pct_filtered = X_filtered.isna().mean()
        self.low_na_cols_ = list(na_pct_filtered[(na_pct_filtered > 0) & (na_pct_filtered <= self.low_na_thresh)].index)
        self.high_na_cols_ = list(na_pct_filtered[na_pct_filtered > self.low_na_thresh].index)
        
        # 3. Fit imputers
        # Median imputer for low-NA columns
        if self.low_na_cols_:
            self.median_imputer_ = SimpleImputer(strategy='median')
            self.median_imputer_.fit(X_filtered[self.low_na_cols_])
            
        # MICE imputer for high-NA columns
        if self.high_na_cols_:
            self.mice_imputer_ = IterativeImputer(
                estimator=self.mice_estimator,
                max_iter=self.max_iter,
                random_state=42,
                imputation_order='ascending',
                verbose=0
            )
            self.mice_imputer_.fit(X_filtered[self.high_na_cols_])
            
        # 4. Fit fallback imputer for all kept columns (to handle stragglers in transform without leakage)
        self.fallback_imputer_ = SimpleImputer(strategy='median')
        self.fallback_imputer_.fit(X_filtered)
            
        return self

    def transform(self, X, y=None):
        """
        Applies the learned transformations to new data.
        """
        # Drop the same columns identified during fit
        X_transformed = X[self.columns_to_keep_].copy()
        # Apply median imputer
        if self.median_imputer_ and self.low_na_cols_:
            X_transformed[self.low_na_cols_] = self.median_imputer_.transform(X_transformed[self.low_na_cols_])
        # Apply MICE imputer
        if self.mice_imputer_ and self.high_na_cols_:
            X_transformed[self.high_na_cols_] = self.mice_imputer_.transform(X_transformed[self.high_na_cols_])
        # Handle any remaining NaNs (e.g., columns that were full in train but have NAs in test)
        # Use the fallback imputer learned on train data
        if X_transformed.isna().any().any():
            medians = pd.Series(self.fallback_imputer_.statistics_, index=self.columns_to_keep_)
            X_transformed = X_transformed.fillna(medians)
        return X_transformed


