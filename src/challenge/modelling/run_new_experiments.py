
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Project Imports
from challenge.data.ingest import load_data
from challenge.data.feature_selection import (
    select_features_ks, 
    select_features_mutual_info
)
from challenge.modelling.experiment import run_experiment_grid
from challenge.modelling.models import get_models, get_custom_lgbm
from challenge.modelling.tuning import tune_lightgbm
from challenge.modelling.train_eval import evaluate_on_test

# Configuration
DATA_PATH = 'dataset/'
TRAIN_FILE = 'aps_failure_training_set.csv'
TEST_FILE = 'aps_failure_test_set.csv'

def main():
    print("--- 1. Data Loading & Cleaning ---")
    X_train_raw, y_train_raw = load_data(DATA_PATH, TRAIN_FILE)
    X_test_raw, y_test_raw = load_data(DATA_PATH, TEST_FILE)
    
    # Cleaning
    na_pct = X_train_raw.isna().mean()
    low_na_cols = na_pct[(na_pct > 0) & (na_pct <= 0.04)].index
    if not low_na_cols.empty:
        rows_to_drop = X_train_raw[low_na_cols].isna().any(axis=1)
        drop_indices = X_train_raw[rows_to_drop].index
        X_train_raw = X_train_raw.drop(index=drop_indices)
        y_train_raw = y_train_raw.drop(index=drop_indices)
        
    outlier_idx_to_drop = 20683
    if outlier_idx_to_drop in X_train_raw.index:
        X_train_raw = X_train_raw.drop(index=outlier_idx_to_drop)
        y_train_raw = y_train_raw.drop(index=outlier_idx_to_drop)
        
    print(f"Cleaned train shape: {X_train_raw.shape}")

    print("\n--- 2. Feature Selection ---")
    # K-S Selection
    print("Computing K-S Feature Set...")
    selected_features_ks = select_features_ks(X_train_raw, y_train_raw, top_n_by_stat=135, p_value_threshold=0.05)
    X_train_ks = X_train_raw[selected_features_ks]
    X_test_ks = X_test_raw[selected_features_ks]
    
    # MI Selection
    print("Computing MI Feature Set...")
    selected_features_mi = select_features_mutual_info(X_train_raw, y_train_raw, top_n=135)
    X_train_mi = X_train_raw[selected_features_mi]
    X_test_mi = X_test_raw[selected_features_mi]
    
    FEATURE_SETS = {
        "K-S Selected": (X_train_ks, X_test_ks),
        "Mutual Information": (X_train_mi, X_test_mi)
    }
    
    print("\n--- 3. Running Experiment Grid with New Models ---")
    models = get_models(random_state=42)
    
    # We only run a subset of samplers/tuning to save time, based on previous bests
    # Previous best was LightGBM | K-S Selected | Gaussian Copula | Tuned Threshold
    SAMPLERS = {
        "Gaussian Copula": "copula",
        # "SMOTE": "smote" # Optional
    }
    SAMPLING_PCTS = [0.25] # Best was 0.25
    TUNING = {"Tuned Threshold": True}
    
    """
    results_df = run_experiment_grid(
        models=models,
        feature_sets=FEATURE_SETS,
        samplers=SAMPLERS,
        tuning_strategies=TUNING,
        sampling_percentages=SAMPLING_PCTS,
        y_train=y_train_raw,
        n_cv_splits=3,
        verbose=True
    )
    
    print("\n--- Experiment Results (Top 10) ---")
    print(results_df.sort_values(by='cost_mean').head(10)[['model', 'feature_set', 'cost_mean', 'auc_mean']])
    
    print("\n--- 4. Custom LightGBM Tuning ---")
    # Tune on the best feature set found (likely K-S or MI)
    # Let's assume K-S for now or pick based on results
    best_fset_name = results_df.sort_values(by='cost_mean').iloc[0]['feature_set']
    """
    # SKIP GRID RUN FOR SPEED (Already ran)
    # Hardcode best feature set for tuning demo
    best_fset_name = "K-S Selected" 
    print(f"Skipping Grid Run. Tuning LightGBM on feature set: {best_fset_name}")
    print(f"Tuning LightGBM on best feature set: {best_fset_name}")
    
    X_train_tune, X_test_tune = FEATURE_SETS[best_fset_name]
    
    best_params = tune_lightgbm(X_train_tune, y_train_raw, n_trials=10) # 10 trials for demo
    
    print("\n--- 5. Evaluating Tuned Custom LightGBM ---")
    custom_model = get_custom_lgbm(random_state=42)
    custom_model.set_params(**best_params)
    
    # Evaluate on Test
    test_results = evaluate_on_test(
        model=custom_model,
        X_train=X_train_tune,
        y_train=y_train_raw,
        X_test=X_test_tune,
        y_test=y_test_raw,
        sampler='copula', # Assuming copula is best
        sampling_strategy=0.25, # Assuming 0.25 is best
        tune_if_none=True,
        verbose=True
    )
    
    print("\n--- Final Test Results (Tuned Custom LightGBM) ---")
    print(test_results['metrics'])

if __name__ == "__main__":
    main()
