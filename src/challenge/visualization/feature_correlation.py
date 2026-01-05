import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns


def plot_correlation_heatmap(df, threshold=0.8, only_missing=False):
    """
    Plots a correlation heatmap for the DataFrame.

    Args:
        df: Input DataFrame.
        threshold: Threshold for returning high correlations.
        only_missing: If True, only plots correlation for columns ending in '_is_missing'.
    """
    if only_missing:
        missing_cols = [col for col in df.columns if str(col).endswith("_is_missing")]
        if not missing_cols:
            print("No columns ending with '_is_missing' found.")
            return pd.DataFrame()  # Return empty DF
        df_plot = df[missing_cols]
        title_suffix = " (Missing Flags)"
    else:
        df_plot = df
        title_suffix = ""

    corr = df_plot.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(f"Feature Correlation Heatmap{title_suffix}")
    plt.show()

    high_corr = corr[(corr.abs() > threshold) & (corr.abs() < 1)]
    return (
        high_corr.stack()
        .reset_index()
        .rename(
            columns={"level_0": "Feature1", "level_1": "Feature2", 0: "Correlation"}
        )
    )


def plot_missingness_dendrogram(df):
    """
    Plots a dendrogram of the missingness flags to visualize the physical sensor architecture.
    """
    missing_cols = [c for c in df.columns if str(c).endswith("_is_missing")]
    if not missing_cols:
        print("No columns ending with '_is_missing' found.")
        return

    X_missing = df[missing_cols]

    # Calculate Correlation Matrix
    corr_matrix = X_missing.corr()

    # Handle NaNs in correlation (e.g., constant columns)
    corr_matrix = corr_matrix.fillna(0)

    # Plot Dendrogram
    plt.figure(figsize=(20, 10))
    plt.title("The Physical Architecture: Sensor Missingness Clusters")

    # Linkage matrix
    # method='ward' minimizes variance within clusters
    Z = sch.linkage(corr_matrix, method="ward")

    dendrogram = sch.dendrogram(Z, labels=missing_cols)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
