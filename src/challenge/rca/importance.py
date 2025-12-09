import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

def cost_score(estimator, X, y, threshold, cost_fp=10.0, cost_fn=500.0):
    """
    Custom scoring function based on total cost.
    Higher score is better (hence we return negative cost).
    """
    proba = estimator.predict_proba(X)[:, 1]
    y_hat = (proba >= threshold).astype(int)
    y_true = np.asarray(y).ravel()
    
    FP = np.sum((y_hat == 1) & (y_true == 0))
    FN = np.sum((y_hat == 0) & (y_true == 1))
    
    total_cost = FP * cost_fp + FN * cost_fn
    return -total_cost

def compute_permutation_importance(model, X, y, threshold, cost_fp=10.0, cost_fn=500.0, random_state=42, n_repeats=10, save_artifacts=False, output_dir=".", file_prefix=""):
    """
    Computes permutation importance using a cost-based scoring function.
    
    Args:
        model: Trained model.
        X (pd.DataFrame): Validation features.
        y: Validation labels.
        threshold (float): Decision threshold for cost calc.
        cost_fp (float): Cost of False Positive.
        cost_fn (float): Cost of False Negative.
        
    Returns:
        pd.DataFrame: Permutation importance dataframe.
    """
    # Define scorer wrapper that pre-binds threshold and costs
    def scorer(est, X_b, y_b):
        return cost_score(est, X_b, y_b, threshold, cost_fp, cost_fn)
    
    print("[PI] Computing Cost-Based Permutation Importance...")
    pi = permutation_importance(
        model, X, y,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    
    pi_df = (
        pd.DataFrame({
            "feature": X.columns,
            "pi_mean": pi.importances_mean,
            "pi_std":  pi.importances_std
        })
        .sort_values("pi_mean", ascending=False)
        .reset_index(drop=True)
    )
    
    if save_artifacts:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, f"{file_prefix}pi_cost_based.csv")
        pi_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        
        # Visualization
        top_pi = pi_df.head(15)[::-1]
        plt.figure(figsize=(8,6))
        plt.barh(top_pi["feature"], top_pi["pi_mean"])
        plt.xlabel("Permutation Importance (Negative Cost Impact)")
        plt.title("Top 15 Features by Cost Reduction")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{file_prefix}pi_bar_top15.png")
        plt.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path}")
        
    return pi_df

def compare_ranks(shap_df, pi_df, save_artifacts=False, output_dir=".", file_prefix=""):
    """
    Compares rankings between SHAP and Permutation Importance.
    
    Args:
        shap_df (pd.DataFrame): Must contain 'feature' and 'mean_abs_shap'.
        pi_df (pd.DataFrame): Must contain 'feature' and 'pi_mean'.
        
    Returns:
        tuple: (merged_df, spearman_rho)
    """
    shap_rank = shap_df[["feature", "mean_abs_shap"]].copy()
    shap_rank["shap_rank"] = shap_rank["mean_abs_shap"].rank(ascending=False, method="dense")
    
    pi_rank = pi_df[["feature", "pi_mean"]].copy()
    pi_rank["pi_rank"] = pi_rank["pi_mean"].rank(ascending=False, method="dense")
    
    merged = shap_rank.merge(pi_rank, on="feature", how="inner")
    
    rho, pval = spearmanr(merged["shap_rank"], merged["pi_rank"])
    print(f"Spearman(rank SHAP vs PI) = {rho:.3f}  (p={pval:.4g})")
    
    merged["avg_rank"] = (merged["shap_rank"] + merged["pi_rank"]) / 2.0
    merged_sorted = merged.sort_values("avg_rank").reset_index(drop=True)
    
    if save_artifacts:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, f"{file_prefix}shap_vs_pi_ranks.csv")
        merged_sorted.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        
        plt.figure(figsize=(6,6))
        plt.scatter(merged["shap_rank"], merged["pi_rank"], s=18)
        plt.xlabel("SHAP rank (lower=more important)")
        plt.ylabel("PI rank (lower=more important)")
        plt.title(f"Spearman correlation = {rho:.2f}")
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{file_prefix}rank_scatter_shap_vs_pi.png")
        plt.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path}")
        
    return merged_sorted, rho
