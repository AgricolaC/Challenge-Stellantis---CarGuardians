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
