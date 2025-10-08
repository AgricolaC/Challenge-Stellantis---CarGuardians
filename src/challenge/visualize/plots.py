import numpy as np

def plot_signal(x: np.ndarray, title: str = "Signal"):
    """
    Create a simple line plot of a 1D signal.

    Notes
    -----
    Implementation should use matplotlib.pyplot.
    Return the axis or figure to allow saving in notebooks.
    """
    raise NotImplementedError

def plot_feature_scatter(x: np.ndarray, y: np.ndarray, labels=None, title: str = "Features"):
    """
    Scatter plot for 2D feature visualization.

    Parameters
    ----------
    x, y : np.ndarray
        Feature coordinates.
    labels : array-like | None
        Optional labels to color points.
    """
    raise NotImplementedError
