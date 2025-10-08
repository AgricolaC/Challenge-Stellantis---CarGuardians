import numpy as np

class PCAIndex:
    """
    PCA-based novelty index.

    Usage pattern (to implement):
    - fit(X_ref): fit PCA on reference feature matrix
    - score(X):  compute distance of samples to reference in PCA space

    Implementation notes:
    - Use sklearn.decomposition.PCA
    - Define how to measure distance (e.g., L2 to ref mean in PCA space)
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self._pca = None
        self._ref_mean = None

    def fit(self, X_ref: np.ndarray) -> "PCAIndex":
        """Fit PCA on reference data and store reference centroid in PCA space."""
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return a 1D novelty score for each row of X."""
        raise NotImplementedError
