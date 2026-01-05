import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_global(
    model,
    X,
    y=None,
    random_state=42,
    max_rows=5000,
    save_artifacts=False,
    output_dir=".",
    file_prefix="",
):
    """
    Computes global SHAP values using TreeExplainer (optimized for tree models).

    Args:
        model: Trained tree-based model (LGBM, XGB, etc.).
        X (pd.DataFrame): Data to explain (usually validation set).
        y: Optional, used for sampling if provided (not strictly used by TreeExplainer).
        random_state (int): Seed for sampling.
        max_rows (int): Max rows to use for SHAP to strictly control runtime.
        save_artifacts (bool): Whether to save plots/CSVs.
        output_dir (str): Directory to save artifacts.
        file_prefix (str): Prefix for artifact filenames.

    Returns:
        tuple: (shap_values, X_sampled, global_importance_df)
    """
    # Sample data if too large
    if len(X) > max_rows:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X_shap = X.iloc[idx]
    else:
        X_shap = X

    # Ensure DataFrame
    if not isinstance(X_shap, pd.DataFrame):
        X_shap = pd.DataFrame(X_shap, columns=getattr(X, "columns", None))

    print(f"[SHAP] initializing TreeExplainer for model: {type(model)}")
    try:
        # TreeExplainer is preferred for trees
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
    except Exception as e:
        print(
            f"[SHAP] TreeExplainer failed: {e}. Falling back to default TreeExplainer settings."
        )
        # Sometimes feature_perturbation needs to be set explicitly
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_shap)

    # For binary classification, shap_values is often a list [shap_neg, shap_pos].
    # Use index 1 for positive class.
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]
    else:
        # Some newer SHAP versions/models might return single array for binary
        # Check shape: if (N, M) -> likely regression or binary log-odds directly
        shap_pos = shap_values

    print("SHAP computed. Shape:", getattr(shap_pos, "shape", None))

    # Global importance
    abs_means = np.abs(shap_pos).mean(axis=0)
    global_importance = (
        pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": abs_means})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    if save_artifacts:
        import os

        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, f"{file_prefix}shap_global_importance.csv")
        global_importance.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # Summary Plots
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_pos, X_shap, plot_type="bar", show=False)
        plt.tight_layout()
        bar_path = os.path.join(output_dir, f"{file_prefix}shap_summary_bar.png")
        plt.savefig(bar_path, dpi=180, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_pos, X_shap, show=False)
        plt.tight_layout()
        bee_path = os.path.join(output_dir, f"{file_prefix}shap_summary_beeswarm.png")
        plt.savefig(bee_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved: {bar_path}, {bee_path}")

        # Dependence Plots for Top 3
        topk = global_importance["feature"].head(3).tolist()
        for i, f in enumerate(topk, 1):
            shap.dependence_plot(
                f, shap_pos, X_shap, show=False, interaction_index=None
            )
            plt.tight_layout()
            outpath = os.path.join(
                output_dir, f"{file_prefix}shap_dependence_{i}_{f}.png"
            )
            plt.savefig(outpath, dpi=180, bbox_inches="tight")
            plt.close()
            print("Saved:", outpath)

    return shap_pos, X_shap, global_importance


def compute_shap_local(
    model, X_shap, shap_values, threshold, top_k=5, output_dir=".", file_prefix=""
):
    """
    Generates local explanations for top high-risk samples.

    Args:
        model: Trained model (for probability).
        X_shap (pd.DataFrame): Data used for SHAP (must match rows of shap_values).
        shap_values (np.array): SHAP matrix (N, M).
        threshold (float): Decision threshold.
        top_k (int): Number of high-risk samples to explain.

    Returns:
        pd.DataFrame: Local explanation table.
    """
    proba = model.predict_proba(X_shap)[:, 1]

    # Identify high risk samples (proba >= threshold)
    # Filter only those that are predicted positive
    data = pd.DataFrame({"proba": proba}, index=X_shap.index)

    # We want samples with highest probability that exceed threshold
    high_risk_candidates = data[data["proba"] >= threshold].sort_values(
        "proba", ascending=False
    )
    highrisk_idx = high_risk_candidates.index[:top_k].tolist()

    if len(highrisk_idx) == 0:
        print("[SHAP] No samples found above threshold for local explanation.")
        return pd.DataFrame()

    local_tables = []

    for ridx in highrisk_idx:
        # Find integer location for SHAP array
        loc_idx = X_shap.index.get_loc(ridx)

        # Determine top contributing features for this row
        shap_row = shap_values[loc_idx]
        feature_vals = X_shap.loc[ridx]

        # Sort by absolute SHAP impact
        order = np.argsort(-np.abs(shap_row))[:8]  # Top 8 features per row

        row_tbl = pd.DataFrame(
            {
                "feature": feature_vals.index[order],
                "value": feature_vals.values[order],
                "shap": shap_row[order],
            }
        )

        row_tbl.insert(0, "row_id", ridx)
        row_tbl.insert(1, "proba", data.loc[ridx, "proba"])
        local_tables.append(row_tbl)

    local_explanations = pd.concat(local_tables, ignore_index=True)
    import os

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"{file_prefix}shap_local_top_contributors.csv")
    local_explanations.to_csv(outpath, index=False)
    print(f"Saved: {outpath}")

    return local_explanations
