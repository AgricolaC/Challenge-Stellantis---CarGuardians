from sklearn.experimental import enable_iterative_imputer  # noqa
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, IterativeImputer
from scipy.cluster.hierarchy import fcluster, linkage
import scipy.spatial.distance as ssd
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
    5. (Optional) Cluster missingness flags to reduce multicollinearity.
    """
    def __init__(self, na_drop_thresh=0.7, low_na_thresh=0.15, missing_flag_thresh=0.04, mice_estimator=Ridge(alpha=10), reduce_missingness=False, cluster_threshold=0.5):
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
        self.missing_flag_cols_ = None
        self.missing_flag_thresh = missing_flag_thresh
        self.reduce_missingness = reduce_missingness
        self.cluster_threshold = cluster_threshold
        self.missingness_cluster_map_ = None

    def fit(self, X, y=None):
        """
        Learns which columns to drop and fits the imputers.
        Also learns missingness clusters if enabled.
        """
        # Identify columns to drop
        na_pct = X.isna().mean()
        self.na_cols_to_drop_ = set(na_pct[na_pct > self.na_drop_thresh].index)
        print(f"Dropping {len(self.na_cols_to_drop_)} columns with > {self.na_drop_thresh:.0%} missing values: {sorted(list(self.na_cols_to_drop_))}")
        
        # We drop rows with <4% missing in training *before* CV, so we don't do it here.
        # But we must drop constant columns.
        self.constant_cols_to_drop_ = set(X.columns[X.nunique() == 1])
        
        cols_to_drop = self.na_cols_to_drop_.union(self.constant_cols_to_drop_)
        self.columns_to_keep_ = [col for col in X.columns if col not in cols_to_drop]
        
        # Log dropped columns
        print(f"Preprocessing Summary:")
        if self.na_cols_to_drop_:
            print(f"  - Dropping {len(self.na_cols_to_drop_)} columns with > {self.na_drop_thresh:.0%} missing values: {sorted(list(self.na_cols_to_drop_))}")
        if self.constant_cols_to_drop_:
            print(f"  - Dropping {len(self.constant_cols_to_drop_)} constant columns: {sorted(list(self.constant_cols_to_drop_))}")
        print(f"  - Retaining {len(self.columns_to_keep_)} columns.")
        
        X_filtered = X[self.columns_to_keep_]


        
        # Identify imputation strategies
        na_pct_filtered = X_filtered.isna().mean()
        # Low NA: 0 < pct <= threshold (Median)
        self.low_na_cols_ = list(na_pct_filtered[(na_pct_filtered > 0) & (na_pct_filtered <= self.low_na_thresh)].index)
        # High NA: > threshold (MICE)
        self.high_na_cols_ = list(na_pct_filtered[na_pct_filtered > self.low_na_thresh].index)



        # Identify columns that need missingness flags (ALL columns with missing values)
        self.missing_flag_cols_ = list(na_pct_filtered[na_pct_filtered > 0].index)
        
        # Fit imputers
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
            
        # Fit fallback imputer for all kept columns (to handle stragglers in transform without leakage)
        self.fallback_imputer_ = SimpleImputer(strategy='median')
        self.fallback_imputer_.fit(X_filtered)
        
        # Learn Missingness Clusters (Fit on Train)
        if self.reduce_missingness and self.missing_flag_cols_:
            # We need to simulate the flags to compute correlation
            # Create temporary DF of flags
            X_flags = X_filtered[self.missing_flag_cols_].isnull().astype(int)
            X_flags.columns = [f'{col}_is_missing' for col in self.missing_flag_cols_]
            
            # --- THE FIX STARTS HERE ---
            # 1. Calculate Correlation Matrix (0 to 1)
            # Use absolute correlation because negative correlation (-1) also implies strong relationship (though rare for missingness)
            # Missingness usually occurs together (+1).
            corr_matrix = X_flags.corr().fillna(0).abs()
            
            # 2. Convert to Distance Matrix (Distance = 1 - Correlation)
            # Perfect correlation (1.0) -> Distance 0.0
            # No correlation (0.0)    -> Distance 1.0
            # We clip to ensure non-negative due to float precision
            dist_matrix = (1 - corr_matrix).clip(lower=0)
            
            # 3. Convert to condensed distance vector (Required for linkage)
            dist_vec = ssd.squareform(dist_matrix)
            
            try:
                # 4. Use 'average' or 'complete' linkage for correlation distance
                # (Ward is strictly for Euclidean, 'average' is best for correlation)
                Z = linkage(dist_vec, method='average')
                
                # 5. Apply Threshold
                # t=0.2 means: "Merge if correlation > 0.8"
                cluster_labels = fcluster(Z, t=self.cluster_threshold, criterion='distance')
                
                self.missingness_cluster_map_ = {}
                num_clusters = max(cluster_labels)
                flag_cols = X_flags.columns.tolist()
                
                print(f"Clustering Missingness: Compressed {len(flag_cols)} _is_missing flags into {num_clusters} modules (Threshold={self.cluster_threshold}).")
                
                for i in range(1, num_clusters + 1):
                    # Get cols in this cluster
                    cluster_cols = [flag_cols[j] for j, label in enumerate(cluster_labels) if label == i]
                    self.missingness_cluster_map_[i] = cluster_cols
                    
            except Exception as e:
                print(f"Warning: Failed to cluster missingness flags. Skipping reduction. Error: {e}")
                self.reduce_missingness = False # Disable if failed
            
        return self

    def transform(self, X, y=None):
        """
        Applies the learned transformations to new data.
        """
        # Drop the same columns identified during fit
        X_transformed = X[self.columns_to_keep_].copy()
        
        # Create missingness flags (BEFORE imputation)
        # Create missingness flags (BEFORE imputation)
        if self.missing_flag_cols_:
            missing_flags = X_transformed[self.missing_flag_cols_].isnull().astype(int)
            missing_flags.columns = [f'{col}_is_missing' for col in self.missing_flag_cols_]
            
            # If reduction enabled and map exists, Apply reduction
            if self.reduce_missingness and self.missingness_cluster_map_:
                new_modules = pd.DataFrame(index=X_transformed.index)
                
                for i, cluster_cols in self.missingness_cluster_map_.items():
                    # Check if these columns exist in the generated flags (sanity check)
                    valid_cols = [c for c in cluster_cols if c in missing_flags.columns]
                    if valid_cols:
                        # Create module feature: 1 if ANY in cluster is missing
                        new_modules[f'Module_{i}_Connectivity'] = missing_flags[valid_cols].max(axis=1)
                
                # Add ONLY the nxew modules, not the raw flags
                X_transformed = pd.concat([X_transformed, new_modules], axis=1)
            else:
                # Add correlation flags as usual
                X_transformed = pd.concat([X_transformed, missing_flags], axis=1)
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


