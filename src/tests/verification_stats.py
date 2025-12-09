
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../challenge')))

from data.feature_selection import engineer_histogram_features, get_wasserstein_features, select_features_kruskal
from data.balancing import balance_with_copula

def test_histogram_physics():
    print("\n--- Testing Histogram Physics ---")
    # specific case: mirror images
    # data: 3 bins. 
    # row 0: [10, 0, 0] -> Skew should be positive (tail is at right? No, wait.)
    # x = [0, 1, 2]
    # row 0: mass at 0. Mean=0. Skew undefined or 0? std=0.
    # Let's use [10, 1, 0] vs [0, 1, 10]
    
    df = pd.DataFrame({
        'ag_000': [10, 0],
        'ag_001': [1, 1],
        'ag_002': [0, 10]
    })
    
    hist_groups = {'ag': ['ag_000', 'ag_001', 'ag_002']}
    
    engineered = engineer_histogram_features(df, hist_groups)
    
    skew_0 = engineered.iloc[0]['ag_skew']
    skew_1 = engineered.iloc[1]['ag_skew']
    
    print(f"Row 0 (Left heavy): Skew = {skew_0}")
    print(f"Row 1 (Right heavy): Skew = {skew_1}")
    
    # Left heavy [10, 1, 0] at indices [0, 1, 2]. Mean approx 0.1. Tail is to the right (positive skew).
    # Wait, indices are 0, 1, 2. Mass is at 0.
    # Weighted mean: (10*0 + 1*1 + 0*2)/11 = 1/11 = 0.09.
    # The "tail" is the 1 at index 1. It pulls mean slightly right.
    # Skewness should be positive (right skewed).
    
    # Right heavy [0, 1, 10] at indices [0, 1, 2]. Mean approx 1.9.
    # The "tail" is at index 1 (left). Skewness should be negative.
    
    if skew_0 > 0 and skew_1 < 0:
        print("PASS: Skewness signs are correct (Left Heavy -> Positive Skew, Right Heavy -> Negative Skew).")
    else:
        print("FAIL: Skewness signs unexpected.")

def test_kruskal():
    print("\n--- Testing Kruskal-Wallis ---")
    X = pd.DataFrame({
        'good_feat': [1, 2, 3, 10, 11, 12], # Separates well
        'bad_feat': [1, 1, 1, 1, 1, 1]
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])
    
    selected = select_features_kruskal(X, y, top_n=1)
    print(f"Selected: {selected}")
    
    if 'good_feat' in selected:
        print("PASS: Kruskal selected the discriminative feature.")
    else:
        print("FAIL: Kruskal failed to select good feature.")

def test_wasserstein():
    print("\n--- Testing Wasserstein ---")
    X = pd.DataFrame({
        'ag_000': [10, 0],
        'ag_001': [0, 10]
    })
    hist_groups = {'ag': ['ag_000', 'ag_001']}
    
    # Reference is uniform [0.5, 0.5]
    refs = {'ag': np.array([0.5, 0.5])}
    
    w_feats = get_wasserstein_features(X, hist_groups, refs)
    print(w_feats)
    
    # Row 0: [1, 0] vs [0.5, 0.5]. Distance should be > 0.
    # Wasserstein on 1D is integral of absolute difference of CDFs.
    # CDF1: [1, 1], CDF2: [0.5, 1]. Diff: 0.5. Integral = 0.5 * step_size (1) = 0.5.
    
    if w_feats.iloc[0]['ag_wasserstein'] > 0:
        print("PASS: Wasserstein distance computed.")
    else:
        print("FAIL: Wasserstein distance is 0 for different distributions.")

if __name__ == "__main__":
    test_histogram_physics()
    test_kruskal()
    test_wasserstein()
