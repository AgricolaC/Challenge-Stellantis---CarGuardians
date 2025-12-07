
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../challenge')))

from data.preprocess import ScaniaPreprocessor

def test_missingness_clustering():
    print("\n--- Testing Missingness Clustering ---")
    
    # Create synthetic data: 100 rows, 10 cols.
    # Col 0, 1, 2: Group A (Missing together)
    # Col 3, 4: Group B (Missing together)
    # Col 5, 6, 7, 8: Random (Low correlation)
    # Col 9: Full (No missing)
    
    N = 200
    df = pd.DataFrame(np.random.randn(N, 10), columns=[f'col_{i}' for i in range(10)])
    
    # Induce missingness
    mask_A = np.random.rand(N) < 0.2
    df.loc[mask_A, ['col_0', 'col_1', 'col_2']] = np.nan
    
    mask_B = np.random.rand(N) < 0.3
    df.loc[mask_B, ['col_3', 'col_4']] = np.nan
    
    # Random missingness
    for c in ['col_5', 'col_6', 'col_7', 'col_8']:
        df.loc[np.random.rand(N) < 0.1, c] = np.nan
        
    print("Initial Columns:", df.columns.tolist())
    
    # Initialize Preprocessor with reduction
    # threshold 0.15 ~= 0.85 correlation (since distance = 1 - corr)
    preprocessor = ScaniaPreprocessor(reduce_missingness=True, cluster_threshold=0.15, missing_flag_thresh=0.01)
    
    print("\nFitting Preprocessor...")
    X_trans = preprocessor.fit_transform(df)
    
    print("Transformed Columns:", X_trans.columns.tolist())
    
    # Check for Modules
    modules = [c for c in X_trans.columns if c.startswith('Module_')]
    print(f"Created {len(modules)} Modules: {modules}")
    
    # Expected: Group A shoud be 1 module. Group B should be 1 module.
    # Random cols might remain separate or cluster if coincidentally correlated.
    
    # Let's check the map
    if hasattr(preprocessor, 'missingness_cluster_map_'):
        print("\nCluster Map:")
        for k, v in preprocessor.missingness_cluster_map_.items():
            print(f"Module {k}: {v}")
            
        # Verify Group A
        # We look for a cluster containing col_0_is_missing, col_1_is_missing, col_2_is_missing
        found_A = False
        for cols in preprocessor.missingness_cluster_map_.values():
            if set(['col_0_is_missing', 'col_1_is_missing', 'col_2_is_missing']).issubset(set(cols)):
                found_A = True
                break
        
        if found_A:
            print("PASS: Group A clustered together.")
        else:
            print("FAIL: Group A not clustered together.")
            
        # Verify Group B
        found_B = False
        for cols in preprocessor.missingness_cluster_map_.values():
            if set(['col_3_is_missing', 'col_4_is_missing']).issubset(set(cols)):
                found_B = True
                break
        if found_B:
            print("PASS: Group B clustered together.")
        else:
            print("FAIL: Group B not clustered together.")

    else:
        print("FAIL: No cluster map found.")

    # Test Transform consistency on new data
    print("\nTesting Transform on New Data...")
    df_new = df.copy()
    X_new_trans = preprocessor.transform(df_new)
    
    if set(X_new_trans.columns) == set(X_trans.columns):
        print("PASS: Transform produced consistent columns.")
    else:
        print("FAIL: Transform columns mismatch.")
        print("Train Cols:", X_trans.columns)
        print("Test Cols:", X_new_trans.columns)
        
if __name__ == "__main__":
    test_missingness_clustering()
