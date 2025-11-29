from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd
from scipy.stats import ks_2samp
import numpy as np
from typing import Tuple, List
from sklearn.feature_selection import mutual_info_classif


def split_histogram_features(X: pd.DataFrame) -> Tuple[List[str], List[str], dict]:
    """
    Separates histogram-based features from standard numerical features.
    
    A histogram family is defined by a 2-letter prefix (e.g., 'ag', 'ay').
    """
    # Find all column prefixes that appear more than once
    all_prefixes = [col.split('_')[0] for col in X.columns]
    prefix_counts = pd.Series(all_prefixes).value_counts()
    hist_families = prefix_counts[prefix_counts > 1].index.tolist()
    
    print(f"Identified {len(hist_families)} histogram families.")
    
    hist_cols = []
    num_cols = []
    hist_groups = {prefix: [] for prefix in hist_families}

    for col in X.columns:
        prefix = col.split('_')[0]
        if prefix in hist_families:
            hist_cols.append(col)
            hist_groups[prefix].append(col)
        else:
            num_cols.append(col)
            
    return hist_cols, num_cols, hist_groups


def select_top_features(X, y, n_features=15):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    selector = RFE(rf, n_features_to_select=n_features, step=0.1)
    selector.fit(X, y)
    selected_cols = X.columns[selector.support_]
    print(f"Selected top {n_features} features: {list(selected_cols)}")
    return X[selected_cols]

def select_features_ks(X: pd.DataFrame, y: pd.Series, p_value_threshold: float = 0.05, top_n_by_stat: int = None) -> list[str]:
    """
    Selects features using the 2-sample Kolmogorov-Smirnov (K-S) test.
    Optionally limits to top_n features with the highest D-statistic (effect size).
    """
    X_neg = X[y == 0]
    X_pos = X[y == 1]
    
    results = []
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
                    results.append({'feature': col, 'p_value': p_value, 'stat': stat})
        except ValueError:
            print(f"Skipping feature {col} due to insufficient data.")
            
    df_results = pd.DataFrame(results)
    if df_results.empty:
        return []
    # Sort by D-statistic (magnitude of difference) descending
    df_results = df_results.sort_values(by='stat', ascending=False)
    if top_n_by_stat:
        return df_results.head(top_n_by_stat)['feature'].tolist()
    
    print(f"K-S test complete. Selected {len(df_results)} out of {X.shape[1]} features.")
    return df_results['feature'].tolist()


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


def select_features_mutual_info(X: pd.DataFrame, y: pd.Series, top_n: int = 50) -> list[str]:
    """
    Selects top_n features based on Mutual Information.
    """
    print("Calculating Mutual Information...")
    # Fill NaNs for MI calculation as it doesn't handle them natively
    X_filled = X.fillna(X.median()) 
    mi_scores = mutual_info_classif(X_filled, y, random_state=42)
    
    mi_series = pd.Series(mi_scores, index=X.columns)
    selected = mi_series.sort_values(ascending=False).head(top_n).index.tolist()
    print(f"Selected top {top_n} features by Mutual Information.")
    return selected

def engineer_histogram_features(X: pd.DataFrame, hist_groups: dict) -> pd.DataFrame:
    """
    Calculates statistical moments for each histogram family and returns them
    as a new DataFrame.
    """
    print(f"Calculating stats for {len(hist_groups)} groups...")
    
    stats_dfs = []
    for prefix, cols in hist_groups.items():
        group_df = X[cols]
        
        # Calculate statistical "shape" features of the histogram bins
        stats = pd.DataFrame(index=X.index)
        stats[f'{prefix}_sum'] = group_df.sum(axis=1)
        stats[f'{prefix}_mean'] = group_df.mean(axis=1)
        stats[f'{prefix}_std'] = group_df.std(axis=1)
        stats[f'{prefix}_skew'] = group_df.skew(axis=1)
        stats[f'{prefix}_kurt'] = group_df.kurt(axis=1)
        
        stats_dfs.append(stats)
            
    # Combine new stats features
    engineered_features = pd.concat(stats_dfs, axis=1)
    return engineered_features

def create_engineered_feature_set(X: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline to *replace* raw histogram bins with engineered statistics.
    """
    # 1. Identify all feature types
    hist_cols, num_cols, hist_groups = split_histogram_features(X)
    
    # 2. Get the base numerical features (the ones we aren't touching)
    X_numerical_base = X[num_cols]
    
    # 3. Create the new engineered features from the histogram bins
    X_engineered_hist = engineer_histogram_features(X, hist_groups)
    
    # 4. Combine the base numerical features and the new engineered features
    # This correctly *replaces* the original hist_cols
    X_final = pd.concat([X_numerical_base, X_engineered_hist], axis=1)
    
    print(f"Replaced {len(hist_cols)} hist bins with {X_engineered_hist.shape[1]} engineered features.")
    print(f"Original feature count: {X.shape[1]} -> New engineered feature count: {X_final.shape[1]}")
    
    return X_final