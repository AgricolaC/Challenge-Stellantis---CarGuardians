import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_curve


def tune_threshold(model, X_val, y_val, cost_fp=10, cost_fn=500):
    prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thr = roc_curve(y_val, prob)
    P = sum(y_val == 1)
    N = sum(y_val == 0)
    FP, FN = fpr * N, (1 - tpr) * P
    costs = cost_fp * FP + cost_fn * FN

    best_idx = np.argmin(costs)
    best_thr = thr[best_idx]
    y_pred = (prob >= best_thr).astype(int)

    f1 = f1_score(y_val, y_pred, average="macro")
    recall = recall_score(y_val, y_pred)

    print(
        f"Best Threshold = {best_thr:.3f} | Expected Cost = {costs[best_idx]:.1f} | F1 = {f1:.3f} | Recall = {recall:.3f}"
    )
    return best_thr, {"cost": costs[best_idx], "f1": f1, "recall": recall}
