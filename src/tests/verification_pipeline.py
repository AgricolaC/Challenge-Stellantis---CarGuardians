
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../challenge')))

from modelling.selectors import KruskalSelector, ConsensusSelector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

def test_pipeline_leakage_mechanics():
    print("\n--- Testing Pipeline Leakage Mechanics ---")
    # Simulate a pipeline
    # Data: 100 samples, 10 features.
    # Feature 0 is correlated with y in first 50 samples (Train), but not in next 50.
    # If selector sees everything, it might pick it?
    # This test primarily checks if the selector runs and subsets columns correctly.
    
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f'f_{i}' for i in range(10)])
    y = np.random.randint(0, 2, 100)
    
    # Make f_0 good predictor
    X['f_0'] = y * 5 + np.random.normal(0, 0.1, 100)
    
    sel = KruskalSelector(top_n=2)
    sel.fit(X, y)
    print(f"Kruskal Selected: {sel.selected_columns_}")
    
    if 'f_0' in sel.selected_columns_:
        print("PASS: Kruskal selected strong feature.")
    else:
        print("FAIL: Kruskal missed strong feature.")
        
    X_trans = sel.transform(X)
    if X_trans.shape[1] == 2:
        print("PASS: Transform reduced dimensions correctly.")
    else:
        print(f"FAIL: Expected 2 cols, got {X_trans.shape[1]}")

def test_consensus_selector():
    print("\n--- Testing Consensus Selector ---")
    X = pd.DataFrame(np.random.rand(100, 20), columns=[f'f_{i}' for i in range(20)])
    y = np.random.randint(0, 2, 100)
    
    # Make f_0, f_1, f_2 strong
    X['f_0'] = y * 10
    X['f_1'] = y * 10 + np.sin(np.arange(100)) # Non-linearish?
    X['f_2'] = y 
    
    # Consensus: Top 5, threshold 2
    cons_sel = ConsensusSelector(top_n=5, consensus_thresh=2)
    cons_sel.fit(X, y)
    
    print(f"Consensus Votes:\n{cons_sel.votes_.sort_values(ascending=False).head(5)}")
    print(f"Selected: {cons_sel.selected_columns_}")
    
    # Check if f_0 is there (likely picked by all)
    if 'f_0' in cons_sel.selected_columns_:
        print("PASS: Consensus picked obvious feature f_0.")
    else:
        print("FAIL: Consensus missed f_0.")
        
    # Check transform
    X_new = cons_sel.transform(X)
    print(f"Output shape: {X_new.shape}")
    
    if X_new.shape[1] > 0:
        print("PASS: Consensus transform returned data.")
    else:
        print("FAIL: Consensus returned empty.")

if __name__ == "__main__":
    test_pipeline_leakage_mechanics()
    test_consensus_selector()
