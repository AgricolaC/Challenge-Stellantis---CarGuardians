import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns

def pca_inspect(X, n_components=2):
    """Perform PCA and visualize explained variance and scatter."""
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=list(range(1, n_components + 1)), y=pca.explained_variance_ratio_, ax=ax[0])
    ax[0].set_title("PCA Explained Variance Ratio")
    
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], s=20, color="teal", alpha=0.6, ax=ax[1])
    ax[1].set_title("PCA Projection (First 2 Components)")
    plt.show()
    
    return pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])


def inspect_pca_loadings(X, component_idx=0, top_n=15):
    """
    Inspects the feature loadings for a specific PCA component.
    Useful for determining if a component is driven by a few outliers or many features.
    """
    # Fit PCA just enough to get the requested component
    pca = PCA(n_components=component_idx + 1)
    pca.fit(X)
    
    loadings = pd.Series(pca.components_[component_idx], index=X.columns)
    
    # Sort by absolute magnitude to find most influential features
    sorted_abs_loadings = loadings.abs().sort_values(ascending=False).head(top_n)
    
    # Get the actual signed values for the top absolute ones
    top_loadings = loadings[sorted_abs_loadings.index]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_loadings.values, y=top_loadings.index, orient='h')
    plt.title(f"Top {top_n} Feature Loadings for PC{component_idx+1}")
    plt.xlabel("Loading Weight")
    plt.axvline(0, color="k", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.show()
    
    return top_loadings
