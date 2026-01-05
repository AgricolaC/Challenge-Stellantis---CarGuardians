import numpy as np
import pandas as pd

from .comparison import create_decision_matrix, plot_radar_chart
from .explanation_shap import compute_shap_global, compute_shap_local
from .importance import compare_ranks, compute_permutation_importance
from .pdp_ice import generate_pdp_ice
from .reference_model import select_cost_threshold, train_reference_model
from .surrogate import extract_rules, train_surrogate_tree


def run_rca_pipeline(
    lgbm_weighted,
    X_dim_fixed,
    Y_fixed,
    cost_fp=10.0,
    cost_fn=500.0,
    random_state=42,
    output_dir="results/rca_results",
    file_prefix="",
):
    """
    Executes the full RCA pipeline.

    Args:
        lgbm_weighted: Base LightGBM estimator.
        X_dim_fixed (pd.DataFrame): Input features.
        Y_fixed (pd.Series): Target labels.
        cost_fp (float): False Positive Cost.
        cost_fn (float): False Negative Cost.
        random_state (int): Random seed.
        output_dir (str): Directory to save artifacts.
        file_prefix (str): Prefix for artifact filenames.
    """

    # 0. Prerequisites
    import os

    os.makedirs(output_dir, exist_ok=True)
    print("--- 0) Prerequisites ---")
    print(f"Cost FP: {cost_fp}, Cost FN: {cost_fn}")
    print(f"Output Dir: {output_dir}, Prefix: {file_prefix}")

    # 1. Train Reference Model (No Resampling)
    print("\n--- 1) Train Reference Model (No Resampling) ---")
    model_ref, X_va, y_va, auc = train_reference_model(
        lgbm_weighted, X_dim_fixed, Y_fixed, random_state
    )

    # 2. Cost-Based Threshold Selection
    print("\n--- 2) Cost-Based Threshold Selection ---")
    threshold, min_cost = select_cost_threshold(
        model_ref,
        X_va,
        y_va,
        cost_fp,
        cost_fn,
        save_artifacts=True,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )

    # 3. SHAP: Global & Local Explanations
    print("\n--- 3) SHAP: Global & Local Explanations ---")
    shap_values, X_shap, global_imp = compute_shap_global(
        model_ref,
        X_va,
        random_state=random_state,
        save_artifacts=True,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )

    print("\n--- 3b) SHAP Local Explanations (Top 5 High Risk) ---")
    # Using threshold fro Step 2
    local_expl = compute_shap_local(
        model_ref,
        X_shap,
        shap_values,
        threshold,
        top_k=5,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )

    # 4. Permutation Importance (Cost-Based) & Rank Comparison
    print("\n--- 4) Permutation Importance (Cost-Based) ---")
    # Note: PI is expensive. We use X_va (validation set) which is already 20% of data.
    # Adjust n_repeats if too slow.
    pi_df = compute_permutation_importance(
        model_ref,
        X_va,
        y_va,
        threshold,
        cost_fp,
        cost_fn,
        random_state=random_state,
        save_artifacts=True,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )

    print("\n--- 4b) Compare Ranks ---")
    compare_ranks(
        global_imp,
        pi_df,
        save_artifacts=True,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )

    # 5. PDP/ICE for Top Features
    print("\n--- 5) PDP/ICE for Top Features ---")
    # Take top 5 from SHAP
    top_features = global_imp["feature"].head(5).tolist()
    generate_pdp_ice(
        model_ref,
        X_va,
        top_features,
        save_artifacts=True,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )

    # 6. Surrogate Decision Tree
    print("\n--- 6) Surrogate Decision Tree ---")
    # Use validation set for training surrogate to explain model behavior on unseen data?
    # Or train on same data as model?
    # Usually we want to explain the model's decision boundary, so using a representative set (X_va) is fine.
    surrogate_dt = train_surrogate_tree(
        model_ref,
        X_va,
        max_depth=3,
        random_state=random_state,
        save_artifacts=True,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )
    extract_rules(
        surrogate_dt,
        X_va.columns,
        save_artifacts=True,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )

    # 7. Decision Matrix & Radar Visualization
    print("\n--- 7) Decision Matrix & Radar Visualization ---")
    decision_matrix = create_decision_matrix(
        save_artifacts=True, output_dir=output_dir, file_prefix=file_prefix
    )
    plot_radar_chart(decision_matrix, output_dir=output_dir, file_prefix=file_prefix)

    print(
        f"\n--- RCA Pipeline Complete! Artifacts are in {output_dir} with prefix '{file_prefix}' ---"
    )
