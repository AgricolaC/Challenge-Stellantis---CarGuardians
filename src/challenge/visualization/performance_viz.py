import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
# Confusion Matrix Plot (with optional thresholding)
def plot_confusion(y_true, y_prob, threshold=0.5, normalize=False):
    """
    Plot confusion matrix given probabilities and threshold.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt=".2f" if normalize else "d")
    plt.title(f"Confusion Matrix (thr={threshold:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Cost vs Threshold Curve (single model)
def plot_cost_threshold_curve(y_true, y_prob, cost_fp=10, cost_fn=500, label=None, ax=None):
    """
    Plot cost vs threshold for a given model’s predictions.
    Can plot on an existing axes (ax) for comparison.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    
    # Handle edge case where roc_curve might not include 0 or 1
    thr = thr[1:]
    fpr = fpr[1:]
    tpr = tpr[1:]
    
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    FP, FN = fpr * N, (1 - tpr) * P
    costs = cost_fp * FP + cost_fn * FN
    
    min_cost = np.min(costs)
    best_thresh = thr[np.argmin(costs)]

    # If no axes provided, create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Cost vs Threshold")
    
    ax.plot(thr, costs, label=f"{label} (Min Cost: {min_cost:.0f} at thr={best_thresh:.3f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total Cost")
    ax.legend()
    ax.grid(True)
    
    return ax, (best_thresh, min_cost)