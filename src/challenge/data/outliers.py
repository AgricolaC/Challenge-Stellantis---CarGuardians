import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler


def visualize_outlier_rescue(
    X, y, contamination=0.01, filename="outlier_rescue_diagram.png"
):
    """
    Generates a 2-panel visualization of the 'Outlier Rescue' mechanism.

    Panel 1: The Mechanism (Scatter of Inliers vs Raw Outliers)
    Panel 2: The Rescue (Filter Logic showing Rescued Failures)
    """
    print(f"Generating Outlier Rescue Visualization ({filename})...")

    # 1. Reproduce Logic
    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit IsoForest
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(X_scaled)

    # Identify Groups
    # Mask: 1 (Inlier), -1 (Outlier)
    # y: 0 (Healthy), 1 (Failure)

    # Boolean Masks
    is_inlier = preds == 1
    is_outlier_raw = preds == -1

    # Further breakdown of raw outliers
    is_true_noise = (is_outlier_raw) & (
        y == 0
    )  # Orange (Healthy but Anomalous -> Noise)
    is_rescued = (is_outlier_raw) & (y == 1)  # Green (Failure and Anomalous -> Rescued)

    # 2. Project to 2D for Visualization (PCA)
    # Use standard scaler for PCA viz (Robust was for detection)
    pca = PCA(n_components=2)
    X_viz = pca.fit_transform(StandardScaler().fit_transform(X))

    # Subsample for clutter-free plotting logic if huge
    if len(X_viz) > 5000:
        idx = np.random.choice(len(X_viz), 5000, replace=False)
        X_viz = X_viz[idx]
        y_viz = y.iloc[idx] if isinstance(y, pd.Series) else y[idx]
        # Re-slice boolean masks
        is_inlier = is_inlier[idx]
        is_true_noise = is_true_noise[idx]
        is_rescued = is_rescued[idx]

    plt.figure(figsize=(20, 8))

    # --- PANEL 1: THE MECHANISM (Raw Isolation) ---
    plt.subplot(1, 2, 1)
    plt.title(
        "The Mechanism: Isolation Forest Detection", fontsize=16, fontweight="bold"
    )

    # Plot Inliers (Blue Cloud)
    plt.scatter(
        X_viz[is_inlier, 0],
        X_viz[is_inlier, 1],
        c="#3498db",
        alpha=0.3,
        s=10,
        label="Healthy Inliers (Cluster)",
    )

    # Plot Raw Outliers (Orange/Red) - Both types mixed initially
    plt.scatter(
        X_viz[is_true_noise, 0],
        X_viz[is_true_noise, 1],
        c="#e67e22",
        marker="x",
        s=40,
        label="Raw Outliers (Noise)",
    )
    plt.scatter(
        X_viz[is_rescued, 0], X_viz[is_rescued, 1], c="#e67e22", marker="x", s=40
    )  # Same color initially

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- PANEL 2: THE RESCUE (Safety Logic) ---
    plt.subplot(1, 2, 2)
    plt.title("The Rescue: Safety Gate Activation", fontsize=16, fontweight="bold")

    # 1. Filter Visual (Concept)
    # We plot the same points but colored by final decision

    # Kept Healthy (Blue)
    plt.scatter(
        X_viz[is_inlier, 0],
        X_viz[is_inlier, 1],
        c="#3498db",
        alpha=0.1,
        s=10,
        label="Kept: Healthy",
    )

    # Dropped Noise (Orange)
    plt.scatter(
        X_viz[is_true_noise, 0],
        X_viz[is_true_noise, 1],
        c="#e67e22",
        marker="x",
        alpha=0.2,
        s=30,
        label="Dropped: Noise (Healthy Outliers)",
    )

    # Rescued Failures (Green) - The Hero
    plt.scatter(
        X_viz[is_rescued, 0],
        X_viz[is_rescued, 1],
        c="#2ecc71",
        marker="*",
        s=150,
        edgecolors="black",
        label="RESCUED: Failure Outliers",
    )

    # Annotation
    n_rescued = sum(is_rescued)
    plt.text(
        0.05,
        0.95,
        f"Safety Gate Logic:\nmask[y == 1] = True\n\nRescued {n_rescued} data points",
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.xlabel("PCA Component 1")
    plt.yticks([])
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved visualization to {filename}")
    plt.close()


def fit_predict_isolation_forest(X, y=None, contamination=0.01, random_state=42):
    """
    Fits an Isolation Forest to detect outliers in the dataset.

    Args:
        X (pd.DataFrame): The input features (must be imputed/no NaNs).
        y (pd.Series, optional): Target labels. If provided, Class 1 (Failures)
                                 will be EXEMPT from removal.
        contamination (float): The proportion of outliers in the data set.
        random_state (int): Seed for reproducibility.

    Returns:
        mask (pd.Series): Boolean mask where True = Keeper, False = Outlier.
    """
    print(
        f"\n--- Isolation Forest Outlier Detection (Contamination={contamination}) ---"
    )

    # 1. Scale Data (RobustScaler is best for handling outliers during detection)
    # We apply this locally just for the IsoForest model
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Fit Isolation Forest
    iso = IsolationForest(
        contamination=contamination, random_state=random_state, n_jobs=-1
    )
    # Predict returns 1 for inlier, -1 for outlier
    preds = iso.fit_predict(X_scaled)

    # Create Mask: True if 1 (Inlier), False if -1 (Outlier)
    mask = preds == 1

    n_outliers = (preds == -1).sum()
    print(
        f"  [IsoForest] Flagged {n_outliers} records as outliers ({n_outliers/len(X):.2%})."
    )

    # 3. Protect Failures (Class 1) if y is provided
    if y is not None:
        n_pos_flagged = ((y == 1) & (mask == False)).sum()
        if n_pos_flagged > 0:
            print(
                f"  [Safety] Rescuing {n_pos_flagged} positive class failures from the outlier set."
            )
            # Force keep all failures (y == 1)
            # We assume ALL failures are valid signals, even if anomalous.
            mask[y == 1] = True

    n_final_drops = (~mask).sum()
    print(
        f"  [Final] Dropping {n_final_drops} outliers (Only Negatives) -> {n_final_drops/len(X):.2%} drop rate."
    )

    return pd.Series(mask, index=X.index)


def analyze_isolation_forest_outliers(X, y_raw, mask):
    """
    Analyzes the dropped outliers to ensure we aren't deleting all failures.
    """
    n_dropped = (~mask).sum()
    if n_dropped == 0:
        print("  No outliers dropped.")
        return

    y_dropped = y_raw[~mask]
    n_failures = y_dropped.sum()

    print(f"  [Analysis] Dropped {n_dropped} rows.")
    print(
        f"  [Analysis] Failures inside dropped set: {n_failures} (Failure Rate: {n_failures/n_dropped:.2%})"
    )
    print(f"  [Analysis] Base Failure Rate of Dataset: {y_raw.mean():.2%}")

    if (n_failures / n_dropped) > (y_raw.mean() * 5):
        print(
            "  WARNING: Outlier set has highly disproportionate number of failures. Review contamination."
        )
