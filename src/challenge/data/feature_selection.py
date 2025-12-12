from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
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
    
    Optimized: Uses vectorized NumPy operations for speed.
    Added: Normalized Entropy, Bimodality Coefficient, Peak Count.
    
    Args:
        X: The input dataframe (raw counts).
        hist_groups: Dictionary {prefix: [col_names]}.
                     NOTE: col_names MUST be sorted (e.g., _000 to _009).
    
    Returns:
        DataFrame with physics-based features.
    """
    print(f"Engineering physics features for {len(hist_groups)} groups (Vectorized)...")
    
    new_features = pd.DataFrame(index=X.index)
    
    for prefix, cols in hist_groups.items():
        # Get data matrix (N_samples, N_bins)
        # Ensure float
        counts = X[cols].values.astype(float)
        
        # --- Feature A: Total Activity (Sum) ---
        total_events = counts.sum(axis=1) # (N,)
        new_features[f'{prefix}_sum'] = total_events
        
        # --- Normalize to PMF ---
        # epsilon for stability
        epsilon = 1e-9
        pmf = counts / (total_events[:, np.newaxis] + epsilon) # (N, bins)
        
        # --- Pre-compute X-coordinates (bin indices) ---
        n_bins = len(cols)
        bin_indices = np.arange(n_bins) # (bins,)
        
        # --- Feature B: Center of Mass (Mean) ---
        # E[x] = sum(p_i * x_i)
        # Dot product of each row with bin_indices
        mean = np.dot(pmf, bin_indices)
        new_features[f'{prefix}_center_mass'] = mean
        
        # --- Feature C: Entropy (Stability) ---
        # H = -sum(p * log(p))
        # Add tiny epsilon to pmf for log
        pmf_safe = np.clip(pmf, 1e-12, 1.0)
        ent = -np.sum(pmf * np.log(pmf_safe), axis=1)
        
        # Normalized Entropy: H / log(N_bins)
        # Ranges 0 (Deterministic) to 1 (Uniform Randomness)
        if n_bins > 1:
            ent_norm = ent / np.log(n_bins)
        else:
            ent_norm = np.zeros_like(ent)
            
        new_features[f'{prefix}_entropy'] = ent_norm
        
        # --- (Removed) Feature D: Low/High Ratio ---
        # User requested removal to rely on Moments (Skew/Kurt) instead.

            
        # --- Vectorized Higher Moments (Skew, Kurtosis) ---
        # 1. Variance = E[x^2] - (E[x])^2
        mean_sq = np.dot(pmf, bin_indices**2)
        variance = mean_sq - (mean**2)
        # Clip negative variance due to float precision
        variance = np.maximum(variance, 0)
        std_dev = np.sqrt(variance)
        
        # 2. Central Central Moments
        # We need (x_i - mean)^k.
        # This requires broadcasting: (N, 1) vs (bins,) -> (N, bins)
        diff = bin_indices[np.newaxis, :] - mean[:, np.newaxis] # (N, bins)
        
        # Skewness numerator: E[(x-mu)^3]
        m3 = np.sum(pmf * (diff**3), axis=1)
        
        # Kurtosis numerator: E[(x-mu)^4]
        m4 = np.sum(pmf * (diff**4), axis=1)
        
        # Handle division by zero (std=0) efficiently
        nonzero_std = std_dev > 1e-6
        
        skew_val = np.zeros_like(mean)
        kurt_val = np.zeros_like(mean)
        
        bias_safe_std = np.where(nonzero_std, std_dev, 1.0)
        
        skew_val[nonzero_std] = m3[nonzero_std] / (bias_safe_std[nonzero_std]**3)
        kurt_val[nonzero_std] = (m4[nonzero_std] / (bias_safe_std[nonzero_std]**4)) - 3
        
        new_features[f'{prefix}_skew'] = skew_val
        new_features[f'{prefix}_kurt'] = kurt_val
        
        # --- Feature E: Bimodality Coefficient ---
        # BC = (gamma^2 + 1) / kappa
        # gamma = skew, kappa = kurtosis + 3 (raw kurtosis)
        # Range: 0 to 1. Values > 0.555 suggest bimodality (uniform or multimodal)
        raw_kurtosis = kurt_val + 3
        bimodality = (skew_val**2 + 1) / (raw_kurtosis + epsilon)
        # Clip for sanity
        bimodality = np.clip(bimodality, 0, 1)
        new_features[f'{prefix}_bimodality'] = bimodality
        
        # --- Feature F: Peak Count ---
        # Count local maxima in the PMF row
        # Simple logic: val[i] > val[i-1] AND val[i] > val[i+1]
        # We can implement this vectorially by comparing shifted arrays
        if n_bins >= 3:
            # Shift Left (val[i+1]) and Right (val[i-1])
            # Wepad with -1 so boundaries aren't peaks locally implies strictly inner peaks?
            # Or usually simple: just check inner.
            
            # Logic: P[i] > P[i-1] AND P[i] > P[i+1]
            left_neighbors = np.hstack([np.zeros((len(pmf), 1)) - 1, pmf[:, :-1]])
            right_neighbors = np.hstack([pmf[:, 1:], np.zeros((len(pmf), 1)) - 1])
            
            is_peak = (pmf > left_neighbors) & (pmf > right_neighbors)
            peak_count = is_peak.sum(axis=1)
            new_features[f'{prefix}_peaks'] = peak_count
        else:
            new_features[f'{prefix}_peaks'] = 1 # Single bin is always 1 peak


    print(f"Generated {new_features.shape[1]} features (Sum, Mass, Entropy, Skew, Kurt, Bimodality, Peak_Count).")
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


def select_features_consensus(X: pd.DataFrame, y: pd.Series, n_features: int = 80) -> List[str]:
    """
    Selects features using a consensus of multiple methods:
    1. K-S Test (Distribution Shift)
    2. Mutual Information (Non-linear Dependency)
    3. Random Forest Importance (Predictive Power)
    
    Returns the UNION of top features from each method robustly.
    """
    print(f"\n--- Consensus Feature Selection (Target per method: {n_features}) ---")
    
    # 1. K-S Test
    ks_feats = select_features_ks(X, y, top_n_by_stat=n_features, p_value_threshold=0.05)
    print(f"KS Features ({len(ks_feats)}): {ks_feats}")

    # 2. Mutual Information
    mi_feats = select_features_mutual_info(X, y, top_n=n_features)
    print(f"MI Features ({len(mi_feats)}): {mi_feats}")
    
    # 3. Random Forest (Model Based)
    # We use a smaller n_estimators for speed in selection
    rf_feats = select_top_features_rf(X, y, n_features=n_features)
    print(f"RF Features ({len(rf_feats)}): {rf_feats}")
    
    # 4. Consensus
    # We take the UNION to ensure we don't miss important signals found by only one method
    # (e.g. MI might catch what KS misses)
    feature_pool = list(set(ks_feats) | set(mi_feats) | set(rf_feats))
    
    print(f"Consensus Pool Size: {len(feature_pool)}")
    
    # Optional: If pool is too large, we could use intersection or voting, 
    # but for RCA, we prefer Recall (Union) over Precision.
    return feature_pool


def select_features_lasso(X: pd.DataFrame, y: pd.Series, n_features: int = 15, alpha: float = 0.01) -> List[str]:
    """
    Selects features using Lasso (L1 regularization).
    Effective for selecting a sparse set of linear features and handling multicollinearity.
    """
    print(f"Starting Lasso Selection (n={n_features}, alpha={alpha})...")
    
    # Lasso requires scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
    lasso.fit(X_scaled, y)
    
    coefs = pd.Series(np.abs(lasso.coef_), index=X.columns)
    selected = coefs.sort_values(ascending=False).head(n_features).index.tolist()
    
    print(f"Lasso Selected {len(selected)} features.")
    return selected


def select_features_lightgbm(X: pd.DataFrame, y: pd.Series, n_features: int = 15) -> List[str]:
    """
    Selects top features using LightGBM Gain Importance.
    Extremely fast and handles non-linearities well.
    """
    print(f"Starting LightGBM Selection (n={n_features})...")
    # Using small trees for speed/robustness
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    lgbm.fit(X, y)
    
    importances = pd.Series(lgbm.feature_importances_, index=X.columns)
    selected = importances.sort_values(ascending=False).head(n_features).index.tolist()
    
    print(f"LightGBM Selected {len(selected)} features.")
    return selected


def select_features_fast_consensus(X: pd.DataFrame, y: pd.Series, n_features: int = 15) -> List[str]:
    """
    Revised "Tri-Method Consensus" for Speed and Robustness.
    Pools features from:
    1. KS Test (Distribution Shift / Physics)
    2. LightGBM (Non-linear Predictive Power)
    3. Lasso (Linear Sparsity / Multicollinearity Handler)
    
    Replaces slow RFE and redundant Mutual Information.
    """
    print(f"\n--- Fast Consensus Feature Selection (Target per method: {n_features}) ---")
    
    # 1. K-S Test (Symptom Detector)
    ks_feats = select_features_ks(X, y, top_n_by_stat=n_features, p_value_threshold=0.05)
    print(f"KS Features ({len(ks_feats)}): {ks_feats}")

    # 2. LightGBM (Non-linear Predictor)
    lgbm_feats = select_features_lightgbm(X, y, n_features=n_features)
    print(f"LightGBM Features ({len(lgbm_feats)}): {lgbm_feats}")
    
    # 3. Lasso (Linear / Sparse Driver)
    # Use a slightly aggressive alpha to encourage sparsity
    lasso_feats = select_features_lasso(X, y, n_features=n_features, alpha=0.005)
    print(f"Lasso Features ({len(lasso_feats)}): {lasso_feats}")
    
    # 4. Consensus (Union)
    feature_pool = list(set(ks_feats) | set(lgbm_feats) | set(lasso_feats))
    
    print(f"Fast Consensus Pool Size: {len(feature_pool)}")
    
    return feature_pool

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