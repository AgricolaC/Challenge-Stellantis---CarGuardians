import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_hardware_fingerprint(preprocessor, save_path=None):
    """
    WOW FACTOR: Maps the 'Invisible' hardware configurations.
    Visualizes the clusters of missing values as 'Hardware Modules'.
    """
    # Check if the preprocessor has the cluster map
    clusters = getattr(preprocessor, "missingness_cluster_map_", {})
    if not clusters:
        print("No missingness clusters found in preprocessor.")
        return

    # Prepare data for Heatmap
    data = []
    for mod_id, feats in clusters.items():
        for f in feats:
            # We create a record for each feature belonging to a module
            data.append({"Module": f"Module {mod_id}", "Feature": f, "Present": 1})

    df_plot = pd.DataFrame(data)

    if df_plot.empty:
        print("No cluster data to plot.")
        return

    # Pivot to Matrix (Features x Modules)
    mat = df_plot.pivot(index="Feature", columns="Module", values="Present").fillna(0)

    plt.figure(figsize=(10, 12))
    sns.heatmap(mat, cmap="Blues", cbar=False, linewidths=0.5, linecolor="lightgray")
    plt.title(
        "Forensic Map: Truck Hardware Configurations", fontsize=16, fontweight="bold"
    )
    plt.ylabel("Sensor Feature")
    plt.xlabel("Inferred Hardware Cluster")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_fan_of_death(X, y, feature_col="ag_DCI", save_path=None):
    """
    WOW FACTOR: The 'True' Fan of Death.
    Visualizes Intensity (DCI) vs Mileage (aa_000).
    Separates 'City Haulers' (High Intensity) from 'Highway Cruisers' (Low Intensity).

    Args:
        X: DataFrame containing 'aa_000' and the specified feature_col.
        y: The target labels.
        feature_col: The column name for Intensity (default 'ag_DCI').
    """
    if "aa_000" not in X.columns or feature_col not in X.columns:
        print(f"Missing 'aa_000' or '{feature_col}' for Fan of Death plot.")
        return

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=X["aa_000"],
        y=X[feature_col],
        hue=y,
        palette={0: "#3498db", 1: "#e74c3c"},
        alpha=0.6,
        s=40,
        edgecolor="w",
    )

    plt.title(
        f"The 'True' Fan of Death: Mileage vs. {feature_col}",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Odometer (aa_000)", fontsize=12)
    plt.ylabel(f"Intensity ({feature_col})", fontsize=12)

    # Annotate Zones
    plt.axhline(
        y=X[feature_col].mean(),
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Avg Intensity",
    )

    # High Intensity Zone
    plt.text(
        X["aa_000"].max() * 0.1,
        X[feature_col].max() * 0.9,
        "City Haulers\n(High Intensity)",
        color="red",
        fontsize=12,
        fontweight="bold",
    )

    # Low Intensity Zone
    plt.text(
        X["aa_000"].max() * 0.8,
        X[feature_col].min(),
        "Highway Cruisers\n(Low Intensity)",
        color="blue",
        fontsize=12,
        fontweight="bold",
    )

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_killer_bins(X_raw, y, family="ag", save_path=None):
    """
    WOW FACTOR: The 'Killer Bin' Analysis.
    Shows which histogram bin correlates most with failure.
    Hypothesis: Higher index bins (Redline) should have higher correlation.
    """
    # 1. Identify bins
    cols = [c for c in X_raw.columns if c.startswith(f"{family}_") and c[-1].isdigit()]
    if not cols:
        print(f"No bins found for family {family}")
        return

    cols = sorted(cols)

    # 2. Calculate Correlation with Target
    corrs = []

    # Need to handle NaNs if X_raw hasn't been imputed
    # (Spearman handles monotonic rank, robust to non-linearity)
    for c in cols:
        # Fill NaN with 0 for correlation check (assuming missing = 0 events)
        res = getattr(X_raw[c].fillna(0), "corr")(y, method="spearman")
        corrs.append(res)

    df_corr = pd.DataFrame({"Bin": cols, "Correlation": corrs})

    # Plot
    plt.figure(figsize=(10, 6))
    colors = ["red" if x == max(corrs) else "skyblue" for x in corrs]

    sns.barplot(x="Bin", y="Correlation", data=df_corr, palette=colors)
    plt.title(
        f"Killer Bin Analysis: {family.upper()} Family (Spearman)",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Correlation with Failure (Target)")
    plt.xlabel("Histogram Bin (000=Low, 009=High)")
    plt.ylim(min(corrs) * 1.1, max(corrs) * 1.1)

    # Annotate max
    max_corr = df_corr["Correlation"].max()
    max_bin = df_corr.loc[df_corr["Correlation"] == max_corr, "Bin"].values[0]

    plt.text(
        cols.index(max_bin),
        max_corr,
        f"KILLER: {max_bin}",
        ha="center",
        va="bottom",
        fontweight="bold",
        color="darkred",
    )

    plt.grid(True, axis="y", alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_law_of_physics(cv_results, save_path=None):
    """
    WOW FACTOR: "The Law of Physics".
    Visualizes which features are selected 100% of the time.
    Proves robustness of the physical drivers (like DCI and Mileage).

    Args:
        cv_results: The dictionary returned by cv_cost(). Must contain 'selected_features'.
    """
    if "selected_features" not in cv_results:
        print(
            "cv_results dictionary does not contain 'selected_features'. Cannot plot stability."
        )
        return

    # Flatten the list of lists
    all_feats = [
        f for fold_feats in cv_results["selected_features"] for f in fold_feats
    ]

    if not all_feats:
        print("No features selected across folds.")
        return

    counts = pd.Series(all_feats).value_counts()

    # Plot top 20 stable features
    n_plot = min(20, len(counts))

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=counts.head(n_plot).values,
        y=counts.head(n_plot).index,
        hue=counts.head(n_plot).index,
        palette="viridis",
        legend=False,
    )

    plt.title(
        f"The Law of Physics: Top Robust Features ({len(cv_results['selected_features'])} Folds)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel(f"Selection Count (Max = {len(cv_results['selected_features'])})")
    plt.ylabel("Feature Name")

    # Add vertical line for 100% stability
    plt.axvline(
        len(cv_results["selected_features"]),
        color="red",
        linestyle="--",
        label="100% Stable",
    )
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_interaction_mileage_skew(
    X, y, feature_col, mileage_col="aa_000", save_path=None
):
    """
    Forensic Audit: Interaction between Mileage and Skewness.
    Questions: Do high-mileage trucks develop 'lopsided' pressure distributions?
    """
    if feature_col not in X.columns or mileage_col not in X.columns:
        print(f"Skipping plot: {feature_col} or {mileage_col} missing.")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=X[mileage_col],
        y=X[feature_col],
        hue=y,
        alpha=0.5,
        palette={0: "blue", 1: "red"},
    )
    sns.regplot(
        x=X[mileage_col],
        y=X[feature_col],
        scatter=False,
        color="black",
        line_kws={"linestyle": "--"},
    )
    plt.title(f"Mileage vs {feature_col}: Skewness Drift", fontsize=14)
    plt.xlabel("Mileage (aa_000)")
    plt.ylabel(f"Skewness ({feature_col})")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()

    plt.show()


def plot_shift_spectrum(X, y, feature_col=None, save_path=None):
    """
    WOW FACTOR: "The Shift Spectrum".
    Visualizes the Operating Point (Center of Mass).
    Proves that broken trucks operating at higher RPMs (shifted right) than healthy ones.
    """
    if feature_col is None or feature_col not in X.columns:
        print(f"Center of Mass column '{feature_col}' not found.")
        return

    plt.figure(figsize=(12, 6))

    # KDE Plot for smoothness
    sns.kdeplot(
        data=X.loc[y == 0, feature_col],
        color="#3498db",
        fill=True,
        alpha=0.3,
        label="Healthy (0)",
    )
    sns.kdeplot(
        data=X.loc[y == 1, feature_col],
        color="#e74c3c",
        fill=True,
        alpha=0.3,
        label="Failing (1)",
    )

    plt.title(
        f"The Shift Spectrum: Operating Point Shift ({feature_col})",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Center of Mass (Histogram Index)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_chaos_cloud(X, y, save_path=None):
    """
    WOW FACTOR: "The Chaos Cloud".
    Visualizes Entropy vs Mileage.
    Shows if failures correlate with 'unpredictable' driving (High Entropy).
    """
    entropy_cols = [c for c in X.columns if c.endswith("_entropy")]
    if not entropy_cols or "aa_000" not in X.columns:
        print("Missing entropy columns or aa_000 for Chaos Cloud.")
        return

    # Use mean entropy as "System Chaos" if multiple exist
    # Or just 'ag_entropy'
    target_ent = next((c for c in entropy_cols if "ag_" in c), entropy_cols[0])

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=X["aa_000"],
        y=X[target_ent],
        hue=y,
        palette={0: "#3498db", 1: "#e74c3c"},
        alpha=0.6,
        s=40,
        edgecolor="w",
    )

    plt.title(
        f"The Chaos Cloud: System Entropy vs Mileage ({target_ent})",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Mileage (aa_000)", fontsize=12)
    plt.ylabel("System Entropy (Chaos)", fontsize=12)
    plt.legend(title="Class")
    plt.grid(True, linestyle="--", alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_overload_tail(X, y, skew_cols=None, save_path=None):
    """
    WOW FACTOR: "The Overload Tail".
    Visualizes Skewness distributions.
    Shows if failures are driven by "spikes" (Positive Skew).
    """
    if skew_cols is None:
        skew_cols = [c for c in X.columns if c.endswith("_skew")]

    if not skew_cols:
        print("No skewness columns found.")
        return

    # Filter to top 5 families to keep plot clean
    top_cols = skew_cols[:5]

    X_plot = X[top_cols].copy()
    X_plot["Target"] = y.values
    df_melt = X_plot.melt(
        id_vars="Target", var_name="Sensor Family", value_name="Skewness"
    )

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        x="Sensor Family",
        y="Skewness",
        hue="Target",
        data=df_melt,
        split=True,
        palette={0: "lightblue", 1: "salmon"},
        inner="quartile",
    )

    plt.title(
        "The Overload Tail: Skewness Distribution", fontsize=16, fontweight="bold"
    )
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_tail_risk(X, y, kurtosis_cols=None, save_path=None):
    """
    Forensic Audit: Kurtosis (Tail Risk).
    Question: Do failing systems show 'Fat Tails' (extreme events)?
    """
    if kurtosis_cols is None:
        kurtosis_cols = [c for c in X.columns if c.endswith("_kurt")]

    if not kurtosis_cols:
        print("No kurtosis columns found.")
        return

    X_plot = X[kurtosis_cols].copy()
    X_plot["Target"] = y.values
    df_melt = X_plot.melt(
        id_vars="Target", var_name="Sensor Family", value_name="Kurtosis"
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="Sensor Family",
        y="Kurtosis",
        hue="Target",
        data=df_melt,
        palette={0: "lightgreen", 1: "tomato"},
    )
    plt.title("Tail Risk: Kurtosis Comparison", fontsize=14)
    plt.yscale("symlog")  # Kurtosis can be huge
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_pressure_regimes(X, y, peak_cols=None, save_path=None):
    """
    Forensic Audit: Bimodality / Regimes.
    Question: Do failures correlate with mode shifting (1 vs 2 peaks)?
    """
    if peak_cols is None:
        peak_cols = [c for c in X.columns if c.endswith("_peaks")]

    if not peak_cols:
        print("No peak columns found.")
        return

    # Plot top 3 families with most variance in peaks
    top_cols = X[peak_cols].std().sort_values(ascending=False).head(3).index.tolist()

    for i, col in enumerate(top_cols):
        # Calculate normative percentages to handle class imbalance (59:1)
        # 1. Melt/Group
        df_local = pd.DataFrame({"Peaks": X[col], "Target": y})

        # 2. Count frequencies
        counts = df_local.groupby(["Target", "Peaks"]).size().reset_index(name="Count")

        # 3. Normalize by Class Total
        # Sum counts per target class
        class_totals = counts.groupby("Target")["Count"].transform("sum")
        counts["Percentage"] = counts["Count"] / class_totals * 100

        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=counts,
            x="Peaks",
            y="Percentage",
            hue="Target",
            palette={0: "grey", 1: "red"},
            edgecolor="black",
        )

        plt.title(f"Pressure Regimes: {col} (Normalized)", fontsize=12)
        plt.ylabel("Percentage of Trucks (%)")
        plt.xlabel("Number of Peaks Detected")
        plt.legend(title="Class", labels=["Healthy", "Failing"])

        if save_path:
            # Handle multiple plots by appending index if needed, or user handles it
            # We'll just append index to filename
            base, ext = save_path.rsplit(".", 1)
            p = f"{base}_{i}.{ext}"
            plt.savefig(p)
            print(f"Saved plot to {p}")
        plt.show()


def plot_distribution_drift(X, wasserstein_cols=None, save_path=None):
    """
    Forensic Audit: Wasserstein Drift.
    Question: How far have the trucks drifted from the 'Healthy Reference'?
    """
    if wasserstein_cols is None:
        wasserstein_cols = [c for c in X.columns if c.endswith("_wasserstein")]

    if not wasserstein_cols:
        print("No Wasserstein features found.")
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=X[wasserstein_cols], palette="coolwarm")
    plt.title(
        "Distribution Drift: Wasserstein Distance from Healthy Baseline", fontsize=14
    )
    plt.ylabel("Distance (Earth Mover's)")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()
