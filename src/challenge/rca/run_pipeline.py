import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from challenge.rca.pipeline import run_rca_pipeline

# ---------------------------------------------------------
# MOCK SETUP FOR DEMONSTRATION
# In a real scenario, these would be loaded from your data pipeline
# ---------------------------------------------------------

def get_mock_data():
    print("Generating mock data...")
    # 1000 samples, 20 features
    X = pd.DataFrame(np.random.randn(1000, 20), columns=[f"feature_{i:02d}" for i in range(20)])
    # Target depends on feature_00 and feature_01
    logits = 2 * X["feature_00"] - 1 * X["feature_01"]
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    return X, y

def get_mock_model():
    # Base estimator (untrained or pre-configured)
    return LGBMClassifier(n_estimators=10, random_state=42, output_type='scikit_learn')

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Load Data (Replace with real loading logic)
    # import joblib
    # X_dim_fixed = pd.read_parquet("...")
    # Y_fixed = ...
    # lgbm_weighted = joblib.load("...")
    
    # Using mock for now to verify pipeline mechanics
    X_dim_fixed, Y_fixed = get_mock_data()
    lgbm_weighted = get_mock_model()
    
    # 2. Run Pipeline
    run_rca_pipeline(
        lgbm_weighted=lgbm_weighted,
        X_dim_fixed=X_dim_fixed,
        Y_fixed=Y_fixed,
        cost_fp=10.0,
        cost_fn=500.0,
        random_state=42
    )
