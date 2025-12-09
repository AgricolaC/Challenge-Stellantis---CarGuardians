import os
import shutil
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from challenge.rca.pipeline import run_rca_pipeline

def verify_fix():
    print("--- Starting Verification ---")
    
    # 1. Setup Dummy Data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f"f{i}" for i in range(n_features)])
    # Make f0 predictive
    y = (X["f0"] > 0.5).astype(int)
    
    # 2. Train Dummy Model
    model = LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X, y)
    
    # 3. Define Output Config
    output_dir = "./test_rca_out"
    file_prefix = "test_run_"
    
    # Clean up previous run
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    # 4. Run RCA Pipeline
    print(f"Running RCA pipeline with output_dir='{output_dir}', file_prefix='{file_prefix}'...")
    try:
        run_rca_pipeline(model, X, y, output_dir=output_dir, file_prefix=file_prefix)
    except Exception as e:
        print(f"FAILED: Pipeline crashed with error: {e}")
        return

    # 5. Check Artifacts
    expected_files = [
        "model_final_lgbm.pkl",
        "threshold.txt",
        "shap_global_importance.csv",
        "shap_summary_bar.png",
        "shap_summary_beeswarm.png",
        "shap_local_top_contributors.csv",
        "pi_cost_based.csv",
        "pi_bar_top15.png",
        "shap_vs_pi_ranks.csv",
        "rank_scatter_shap_vs_pi.png",
        "pdp_ice_grid.png",
        "surrogate_tree.png",
        "surrogate_rules.txt",
        "decision_matrix.csv",
        "decision_matrix_radar_clean.png"
    ]
    
    missing = []
    print("\nChecking for files:")
    for f in expected_files:
        path = os.path.join(output_dir, f"{file_prefix}{f}")
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  {path}: {status}")
        if not exists:
            missing.append(path)
            
    if not missing:
        print("\nSUCCESS: All artifacts found directly in output_dir with correct prefix.")
        # Cleanup
        shutil.rmtree(output_dir)
    else:
        print(f"\nFAILURE: Missing {len(missing)} files.")

if __name__ == "__main__":
    verify_fix()
