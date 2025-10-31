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
def plot_cost_threshold_curve(y_true, y_prob, cost_fp=10, cost_fn=500, label=None):
    """
    Plot cost vs threshold for a given model’s predictions.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    FP, FN = fpr * N, (1 - tpr) * P
    costs = cost_fp * FP + cost_fn * FN

    plt.figure(figsize=(8, 4))
    plt.plot(thr, costs, label=label or "Model")
    plt.xlabel("Threshold")
    plt.ylabel("Total Cost")
    plt.title("Cost vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

