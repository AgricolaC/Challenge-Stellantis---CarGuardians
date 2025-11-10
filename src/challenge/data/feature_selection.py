from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd
from scipy.stats import ks_2samp
import numpy as np

def split_histogram_features(X):
    """Separate 7×10 histogram families from numerical ones."""
    hist_families = ['ag', 'ay', 'ba', 'cn', 'cs', 'ee', 'cn']
    hist_cols = [c for c in X.columns if any(c.startswith(f) for f in hist_families)]
    num_cols = [c for c in X.columns if c not in hist_cols]
    return X[hist_cols], X[num_cols], hist_cols, num_cols

def select_top_features(X, y, n_features=15):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    selector = RFE(rf, n_features_to_select=n_features, step=0.1)
    selector.fit(X, y)
    selected_cols = X.columns[selector.support_]
    print(f"Selected top {n_features} features: {list(selected_cols)}")
    return X[selected_cols]

def select_features_ks(X: pd.DataFrame, y: pd.Series, p_value_threshold: float = 0.05) -> list[str]:
    """
    Selects features using the 2-sample Kolmogorov-Smirnov (K-S) test.
    """
    X_neg = X[y == 0]
    X_pos = X[y == 1]
    
    selected_features = []
    
    print(f"Starting K-S test on {X.shape[1]} features...")
    
    for col in X.columns:
        try:
            # Explicitly replace inf with NaN and drop NaNs
            data_neg = X_neg[col].replace([np.inf, -np.inf], np.nan).dropna()
            data_pos = X_pos[col].replace([np.inf, -np.inf], np.nan).dropna()

            # Perform the K-S test if both samples have data
            if not data_neg.empty and not data_pos.empty:
                stat, p_value = ks_2samp(data_neg, data_pos)
                
                if p_value <= p_value_threshold:
                    selected_features.append(col)
            
        except ValueError:
            print(f"Skipping feature {col} due to insufficient data.")
            
    print(f"K-S test complete. Selected {len(selected_features)} out of {X.shape[1]} features.")
    return selected_features

def get_feature_pvalues(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Performs the 2-sample K-S test on all features and returns their p-values.
    """
    X_neg = X[y == 0]
    X_pos = X[y == 1]
    
    p_values = {}
    
    for col in X.columns:
        try:
            data_neg = X_neg[col].replace([np.inf, -np.inf], np.nan).dropna()
            data_pos = X_pos[col].replace([np.inf, -np.inf], np.nan).dropna()

            if not data_neg.empty and not data_pos.empty:
                stat, p_value = ks_2samp(data_neg, data_pos)
                p_values[col] = p_value
            else:
                p_values[col] = 1.0  # Assign max p-value if test can't be run
                
        except ValueError:
            p_values[col] = 1.0  # Assign max p-value on error
            
    return pd.Series(p_values, name="p_value").sort_values()