from sklearn.cluster import AgglomerativeClustering, KMeans
import seaborn as sns
import matplotlib.pyplot as plt

def cluster_visualization(X, method="kmeans", n_clusters=2):
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
    
    labels = model.fit_predict(X)
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=labels, palette="viridis")
    plt.title(f"{method.upper()} Clustering (n={n_clusters})")
    plt.show()
    return labels
