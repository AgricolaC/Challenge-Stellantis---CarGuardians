import numpy as np

class MahalanobisIndex:
    """
    Mahalanobis distance-based novelty index.

    Usage pattern (to implement):
    - fit(X_ref): estimate mean and covariance (regularized)
    - score(X):  compute Mahalanobis distance to reference distribution
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self._mu = None
        self._cov_inv = None

    def fit(self, X_ref: np.ndarray) -> "MahalanobisIndex":
        """Compute reference mean and inverse covariance (with small ridge)."""
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return Mahalanobis distance per row."""
        raise NotImplementedError
