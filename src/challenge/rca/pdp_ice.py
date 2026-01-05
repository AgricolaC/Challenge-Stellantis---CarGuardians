import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay


def generate_pdp_ice(
    model, X, features, output_dir=".", file_prefix="", save_artifacts=False
):
    """
    Generates PDP/ICE plots for specified features.

    Args:
        model: Trained model.
        X (pd.DataFrame): Data.
        features (list): List of feature names to plot (usually top importance).
        output_dir (str): Directory to save plots.
        file_prefix (str): Prefix for artifact filenames.
    """
    if not features:
        print("[PDP] No features provided for PDP/ICE.")
        return

    print(f"[PDP] Generating plots for top {len(features)} features...")

    # We can do one large grip plot or individual ones.
    # The reference guide suggests a grid.

    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Flatten axes if needed
    if n_rows * n_cols > 1:
        ax = ax.flatten()
    else:
        ax = [ax]

    display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features,
        kind="both",  # plots both average (PDP) and individual (ICE) lines
        subsample=2000,  # Subsample for speed
        n_jobs=-1,
        grid_resolution=20,
        random_state=42,
        ax=ax[: len(features)],
    )

    # Hide unused subplots
    for i in range(len(features), len(ax)):
        ax[i].set_visible(False)

    plt.tight_layout()

    if save_artifacts:
        import os

        os.makedirs(output_dir, exist_ok=True)

        outpath = os.path.join(output_dir, f"{file_prefix}pdp_ice_grid.png")
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {outpath}")

    return display
