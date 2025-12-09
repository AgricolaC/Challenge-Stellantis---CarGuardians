from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
import pandas as pd
from scipy.stats import ks_2samp, entropy, skew, kurtosis, wasserstein_distance, kruskal
import numpy as np
from typing import Tuple, List, Dict, Optional

def split_histogram_features(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]]]:
    """
    Separates histogram-based features from standard numerical features.
    
    Identifies histogram families (e.g., 'ag', 'ay') based on shared 2-letter prefixes.
    Also separates columns ending in '_is_missing'.
    
    Returns:
        hist_cols: List of all histogram bin columns.
        num_cols: List of standard numerical columns.
        is_missing_cols: List of missingness flag columns.
        hist_groups: Dictionary mapping prefix -> sorted list of bin columns.
    """
    # 1. Identify potential families (prefixes appearing more than once)
    # Exclude 'is_missing' cols from this logic
    all_prefixes = [col.split('_')[0] for col in X.columns if 'is_missing' not in col and col[-1].isdigit()]
    prefix_counts = pd.Series(all_prefixes).value_counts()
    hist_families = prefix_counts[prefix_counts > 1].index.tolist()
    
    print(f"Identified {len(hist_families)} histogram families: {hist_families}")
    
    hist_cols = []
    num_cols = []
    is_missing_cols = []
    hist_groups = {prefix: [] for prefix in hist_families}

    # 2. Sort columns to ensure bins are ordered (Critical for Physics calculations)
    sorted_cols = sorted(X.columns.tolist())

    for col in sorted_cols:
        prefix = col.split('_')[0]
        
        if 'is_missing' in col:
            is_missing_cols.append(col)
        elif prefix in hist_families:
            hist_cols.append(col)
            hist_groups[prefix].append(col)
        else:
            num_cols.append(col)

    return hist_cols, num_cols, is_missing_cols, hist_groups


def select_top_features_rf(X: pd.DataFrame, y: pd.Series, n_features=15) -> List[str]:
    """
    Selects top features using Recursive Feature Elimination (RFE) with a Random Forest.
    """
    print(f"Starting RFE with RandomForest (n={n_features})...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    selector = RFE(rf, n_features_to_select=n_features, step=0.1)
    selector.fit(X, y)
    selected_cols = X.columns[selector.support_]
    print(f"Selected top {n_features} features: {list(selected_cols)}")
    return list(selected_cols)


def select_features_ks(X: pd.DataFrame, y: pd.Series, p_value_threshold: float = 0.05, top_n_by_stat: int = None) -> List[str]:
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
        selected = df_results.head(top_n_by_stat)['feature'].tolist()
        print(f"K-S test selected top {top_n_by_stat} features by D-statistic.")
        return selected
    
    print(f"K-S test complete. Selected {len(df_results)} features with p < {p_value_threshold}.")
    return df_results['feature'].tolist()


def get_feature_pvalues(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Performs the 2-sample K-S test on all features and returns their p-values.
    Useful for ranking.
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


def select_features_mutual_info(X: pd.DataFrame, y: pd.Series, top_n: int = 50) -> List[str]:
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


def engineer_histogram_features(X: pd.DataFrame, hist_groups: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Engineers physics-informed features from histogram bins by treating them
    as Probability Mass Functions (PMF).
    
    Args:
        X: The input dataframe (raw counts).
        hist_groups: Dictionary {prefix: [col_names]}.
                     NOTE: col_names MUST be sorted (e.g., _000 to _009).
    
    Returns:
        DataFrame with physics-based features (Sum, Mass_Center, Entropy, Low_Ratio, High_Ratio).
    """
    print(f"Engineering physics features for {len(hist_groups)} groups...")
    
    new_features = pd.DataFrame(index=X.index)
    
    for prefix, cols in hist_groups.items():
        # Identify samples with sensors = 0, avoid them in the feature engineering computation
        
        # 1. Extract raw counts
        raw_counts = X[cols]
        # Ensure counts are floats to avoid extensive casting
        raw_counts = raw_counts.astype(float)
        
        # --- Feature A: Total Activity (Context) ---
        # The total number of events. Correlates with usage/age.
        total_events = raw_counts.sum(axis=1)
        new_features[f'{prefix}_sum'] = total_events
        
        # --- Normalize to PMF (Probabilities) ---
        # Avoid division by zero: if sum is 0, probabilities are 0 (handled by fillna)
        # This treats the histogram as a probability distribution
        pmf = raw_counts.div(total_events, axis=0).fillna(0)
        
        # --- Feature B: Center of Mass (Virtual Physical Sensor) ---
        # Weighted mean of indices: Sum(Index_i * p_i)
        # Represents the average "Pressure" or "Temperature" level
        weights = np.arange(len(cols))
        center_of_mass = pmf.dot(weights)
        new_features[f'{prefix}_center_mass'] = center_of_mass
        
        # --- Feature C: Entropy (System Instability) ---
        # High entropy = System is visiting many states (Oscillation/Flutter)
        # Low entropy = System is stable in one state
        new_features[f'{prefix}_entropy'] = pmf.apply(lambda row: entropy(row) if row.sum() > 0 else 0, axis=1)
        
        # --- Feature D: Low-End Ratio (Leakage Proxy) ---
        # Sum of first 2 bins (assuming _000 and _001 are low values)
        if len(cols) >= 2:
            new_features[f'{prefix}_low_ratio'] = pmf.iloc[:, :2].sum(axis=1)
        else:
            new_features[f'{prefix}_low_ratio'] = pmf.iloc[:, 0]
            
        # --- Feature E: High-End Ratio (Overload Proxy) ---
        # Sum of last 2 bins
        if len(cols) >= 2:
            new_features[f'{prefix}_high_ratio'] = pmf.iloc[:, -2:].sum(axis=1)
        else:
            new_features[f'{prefix}_high_ratio'] = pmf.iloc[:, -1]

        # --- Weighted Skewness & Kurtosis (Physics Correct) ---
        # We need to calculate these based on the distribution shape (pmf), not the counts values.
        # X coordinates are the bin indices (weights variable above).
        
        def weighted_stats(row_pmf, mean_val, x_vals):
            # If total mass is 0 (all counts 0), stats are 0 (or undefined, but 0 is safe for ML)
            if row_pmf.sum() == 0:
                return 0.0, 0.0
            
            # Variance: sum( p_i * (x_i - mean)^2 )
            diff = x_vals - mean_val
            variance = np.dot(row_pmf, diff**2)
            std_dev = np.sqrt(variance)
            
            if std_dev == 0:
                return 0.0, 0.0
                
            # Skew: sum( p_i * (x_i - mean)^3 ) / std^3
            moment3 = np.dot(row_pmf, diff**3)
            skew_val = moment3 / (std_dev**3)
            
            # Kurtosis: sum( p_i * (x_i - mean)^4 ) / std^4 - 3
            moment4 = np.dot(row_pmf, diff**4)
            kurt_val = (moment4 / (std_dev**4)) - 3
            
            return skew_val, kurt_val

        # Vectorized implementation of weighted moments is hard, but we can do a modest apply
        # We already have center_of_mass which is the mean
        
        # Precompute x_vals (0, 1, 2...)
        x_indices = np.arange(len(cols))
        
        stats_results = []
        # optimization: iterate over arrays is faster than apply on rows sometimes, 
        # but apply with raw=True is decent. 
        # Let's stick to a loop or apply for clarity as N is manageable? 
        # Actually, let's allow apply for readability.
        
        # Zip pmf and center_of_mass to compute
        skew_list = []
        kurt_list = []
        
        # Convert to numpy for speed
        pmf_values = pmf.values
        means = center_of_mass.values
        
        for i in range(len(pmf)):
            s, k = weighted_stats(pmf_values[i], means[i], x_indices)
            skew_list.append(s)
            kurt_list.append(k)
            
        new_features[f'{prefix}_skew'] = skew_list
        new_features[f'{prefix}_kurt'] = kurt_list

    print(f"Generated {new_features.shape[1]} features (Sum, Mass, Entropy, Skew, Kurt).")
    return new_features


def get_wasserstein_features(X: pd.DataFrame, hist_groups: Dict[str, List[str]], healthy_references: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Calculates Wasserstein (Earth Mover's) Distance for each histogram family
    against a "healthy reference" distribution.
    
    Args:
        X: Input dataframe
        hist_groups: Dictionary of family -> columns
        healthy_references: Dictionary of family -> reference probability distribution (array)
                            Must sum to 1.
    """
    X_dist = pd.DataFrame(index=X.index)
    
    for prefix, cols in hist_groups.items():
        if prefix not in healthy_references:
            continue
            
        ref_dist = healthy_references[prefix]
        bin_indices = np.arange(len(cols))
        
        # We need to compute WD for each row
        # WD(u_weights, v_weights) where u_values and v_values are bin_indices
        
        counts = X[cols].values
        
        # Pre-normalize rows to avoid doing it inside loop if possible, 
        # but pure WD needs normalized weights.
        sums = counts.sum(axis=1)[:, np.newaxis]
        # Avoid div by zero
        sums[sums == 0] = 1
        pmfs = counts / sums
        
        w_dists = []
        for i in range(len(X)):
            d = wasserstein_distance(bin_indices, bin_indices, u_weights=pmfs[i], v_weights=ref_dist)
            w_dists.append(d)
            
        X_dist[f'{prefix}_wasserstein'] = w_dists
        
    return X_dist


def select_features_kruskal(X: pd.DataFrame, y: pd.Series, top_n: int = 50) -> List[str]:
    """
    Selects top_n features using the Kruskal-Wallis H-test.
    Robust to outliers unlike ANOVA F-test.
    """
    p_values = {}
    print(f"Starting Kruskal-Wallis test on {X.shape[1]} features...")
    
    # Pre-group data
    X_neg = X[y == 0]
    X_pos = X[y == 1]
    
    for col in X.columns:
        # Skip constant columns
        if X[col].nunique() <= 1:
            p_values[col] = 1.0
            continue
            
        try:
            stat, p = kruskal(X_neg[col], X_pos[col])
            p_values[col] = p
        except ValueError:
            p_values[col] = 1.0
            
    # Sort by p-value (lowest is most significant)
    sorted_feats = pd.Series(p_values).sort_values()
    selected = sorted_feats.head(top_n).index.tolist()
    print(f"Kruskal-Wallis selected {top_n} features.")
    return selected


def select_features_ks_drift(X_train: pd.DataFrame, X_test: pd.DataFrame, p_value_threshold: float = 0.05) -> pd.DataFrame:
    """
    Detects Covariate Shift (Drift) between Train and Test sets using KS Test.
    Returns a DataFrame of drifting features and their p-values.
    """
    drift_results = []
    
    for col in X_train.columns:
        if col not in X_test.columns:
            continue
            
        try:
            # KS test: null hypothesis is that samples are drawn from same dist.
            stat, p_value = ks_2samp(X_train[col].dropna(), X_test[col].dropna())
            
            # If p is small, distributions are DIFFERENT -> DRIFT
            if p_value <= p_value_threshold:
                drift_results.append({
                    'feature': col,
                    'p_value': p_value,
                    'stat': stat
                })
        except ValueError:
            pass
            
    return pd.DataFrame(drift_results).sort_values(by='stat', ascending=False)


def remove_uptime_redundancy(X: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    """
    Removes redundant 'Sum' features that all essentially measure trip duration.
    Keeps 'ag_sum' as the proxy for 'Total_Events'.
    """
    # 1. Identify all 'Sum' features created by the histogram engine
    sum_cols = [c for c in X.columns if c.endswith('_sum')]
    
    if len(sum_cols) < 2:
        return X
    
    print(f"Detected {len(sum_cols)} histogram sum features. Analyzing redundancy...")
    
    # 2. Keep 'ag_sum' (usually the cleanest), drop the others
    # We rename it to 'Total_Operational_Events' to be explicit
    col_to_keep = 'ag_sum' 
    if col_to_keep not in sum_cols:
        col_to_keep = sum_cols[0] # Fallback
        
    cols_to_drop = [c for c in sum_cols if c != col_to_keep]
    
    X_clean = X.drop(columns=cols_to_drop)
    X_clean = X_clean.rename(columns={col_to_keep: 'Total_Operational_Events'})
    
    print(f"Dropped {len(cols_to_drop)} redundant sum features: {cols_to_drop}")
    print(f"Renamed '{col_to_keep}' to 'Total_Operational_Events'.")
    
    return X_clean
def create_engineered_feature_set(X: pd.DataFrame, healthy_references: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """
    Full pipeline to REPLACE raw histogram bins with engineered statistics.
    
    This function:
    1. Identifies histogram families.
    2. Creates physics-based features (Center of Mass, Entropy, etc.) for each family.
    3. OPTIONAL: If healthy_references is provided, calculates Wasserstein distance.
    4. REMOVES the original raw bins to prevent multicollinearity.
    5. Combines the base features with the new engineered features.
    """
    # 1. Identify all feature types
    # Ensure groups are sorted for correct physics calculations
    hist_cols, num_cols, is_missing_cols, hist_groups = split_histogram_features(X)
    
    # 2. Get the base features (Standard numeric + Missing flags)
    X_numerical_base = X[num_cols]
    X_is_missing = X[is_missing_cols]
    
    # 3. Create the new engineered features from the histogram bins
    X_engineered_hist = engineer_histogram_features(X, hist_groups)
    
    # 4. Create Distance Features (Wasserstein) if references provided
    if healthy_references is not None:
        print("Calculating Wasserstein distances from healthy references...")
        X_distance = get_wasserstein_features(X, hist_groups, healthy_references)
        # Combine everything
        X_final = pd.concat([X_numerical_base, X_is_missing, X_engineered_hist, X_distance], axis=1)
        print(f"Added {X_distance.shape[1]} distance features.")
    else:
        # Combine without distance
        X_final = pd.concat([X_numerical_base, X_is_missing, X_engineered_hist], axis=1)
    
    # NEW: Remove Redundant Uptime/Mileage Features
    X_final = remove_uptime_redundancy(X_final)
    
    print(f"Replaced {len(hist_cols)} raw bins with {X_engineered_hist.shape[1]} physics features.")
    print(f"Original feature count: {X.shape[1]} -> New engineered feature count: {X_final.shape[1]}")
    
    return X_final