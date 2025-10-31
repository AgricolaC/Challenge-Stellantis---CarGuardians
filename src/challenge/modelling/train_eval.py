from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Union

# Optional type reference to pandas without hard dependency at import time
ArrayLike = Union[np.ndarray, "pd.Series", "pd.DataFrame"]  # noqa

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, confusion_matrix, recall_score, precision_score
)

import time
from contextlib import contextmanager
from typing import Dict, Any, Optional, Iterable
from tqdm.auto import tqdm

@contextmanager
def timer() -> Iterable[float]:
    """Context manager returning elapsed seconds via yield."""
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"⏱️ Elapsed: {end - start:.2f}s")

def fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.2f}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m {sec:.1f}s"

def _to_array(y) -> np.ndarray:
    """Coerce labels to a flat numpy array."""
    return np.asarray(y).ravel()

def _slice_xy(X, idx):
    """Index DataFrame with .iloc or ndarray with [] transparently."""
    try:
        return X.iloc[idx]
    except AttributeError:
        return X[idx]

def _scores(model, X) -> np.ndarray:
    """
    Return scores on [0,1] for thresholding.
    - prefer predict_proba[:,1]
    - fall back to decision_function mapped monotonically to [0,1]
    - last resort: predict() as {0,1} floats
    """
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(X)
        # Some classifiers can return shape (n_samples,) for binary; standardize to [:,1]
        if s.ndim == 2 and s.shape[1] >= 2:
            return s[:, 1].astype(float)
        return np.asarray(s, dtype=float).ravel()
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float).ravel()
        s_min, s_max = np.min(s), np.max(s)
        if s_max == s_min:
            return np.full_like(s, 0.5, dtype=float)
        return (s - s_min) / (s_max - s_min)
    # last resort
    return np.asarray(model.predict(X), dtype=float).ravel()

def _best_threshold_by_cost(
    y_true: np.ndarray, prob: np.ndarray, cost_fp=10, cost_fn=500
) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(y_true, prob)
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    FP = fpr * N
    FN = (1 - tpr) * P
    costs = cost_fp * FP + cost_fn * FN
    j = int(np.argmin(costs))
    return float(thr[j]), float(costs[j])

def _metrics_at_threshold(
    y_true: np.ndarray, prob: np.ndarray, thr: float, cost_fp=10, cost_fn=500
) -> Dict[str, float]:
    y_pred = (prob >= thr).astype(int)
    # Ensure 2x2 shape even if one class is missing in preds
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(y_true, prob)),
        "MacroF1": float(f1_score(y_true, y_pred, average="macro")),
        "Recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Precision_pos": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "FP": float(fp),
        "FN": float(fn),
        "Cost": float(cost_fp * fp + cost_fn * fn),
    }


# --------------------------
# Cross-validated cost + threshold selection
# --------------------------
def cv_cost(
    model,
    X, y,
    *,
    cost_fp: int = 10,
    cost_fn: int = 500,
    folds: int = 5,
    sampler=None,
    verbose: bool = True,
    return_threshold: bool = False,
    random_state: int = 42,
    show_progress: bool = True,        
) -> Dict[str, Any]:
    """
    Cross-validated AUC/Cost/Macro-F1 with cost-optimized thresholds.
    Returns fold-level timings when show_progress=True.
    """
    y = np.asarray(y).ravel()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    aucs, best_costs, best_thresholds, f1s = [], [], [], []
    fold_times_fit, fold_times_pred = [], []

    fold_iter = skf.split(X, y)
    if show_progress:
        fold_iter = tqdm(fold_iter, total=folds, desc="CV folds", leave=False)

    for tr_idx, va_idx in fold_iter:
        X_tr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
        X_va = X.iloc[va_idx] if hasattr(X, "iloc") else X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        if sampler is not None:
            X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_va)[:, 1]
        elif hasattr(model, "decision_function"):
            s = model.decision_function(X_va).astype(float)
            s_min, s_max = np.min(s), np.max(s)
            prob = np.full_like(s, 0.5, dtype=float) if s_max == s_min else (s - s_min) / (s_max - s_min)
        else:
            prob = model.predict(X_va).astype(float)
        pred_t = time.perf_counter() - t1

        fold_times_fit.append(fit_t)
        fold_times_pred.append(pred_t)

        auc = roc_auc_score(y_va, prob)
        fpr, tpr, thr = roc_curve(y_va, prob)
        P, N = (y_va == 1).sum(), (y_va == 0).sum()
        FP, FN = fpr * N, (1 - tpr) * P
        costs = cost_fp * FP + cost_fn * FN
        j = int(np.argmin(costs))
        best_thr = float(thr[j])

        y_pred = (prob >= best_thr).astype(int)
        f1 = f1_score(y_va, y_pred, average="macro")

        aucs.append(auc)
        best_costs.append(float(costs[j]))
        best_thresholds.append(best_thr)
        f1s.append(f1)

        if show_progress:
            fold_iter.set_postfix({
                "AUC": f"{auc:.3f}",
                "F1": f"{f1:.3f}",
                "Cost": f"{costs[j]:.0f}",
                "thr": f"{best_thr:.3f}",
                "fit": fmt_secs(fit_t),
                "pred": fmt_secs(pred_t),
            })

    out = {
        "AUC_mean": float(np.mean(aucs)),
        "AUC_std": float(np.std(aucs)),
        "Cost_mean": float(np.mean(best_costs)),
        "Cost_std": float(np.std(best_costs)),
        "F1_mean": float(np.mean(f1s)),
        "best_thresholds": [float(t) for t in best_thresholds],
        "fit_time_mean": float(np.mean(fold_times_fit)),
        "pred_time_mean": float(np.mean(fold_times_pred)),
        "fit_time_folds": fold_times_fit,
        "pred_time_folds": fold_times_pred,
    }
    if return_threshold:
        out["recommended_threshold"] = float(np.median(best_thresholds))

    if verbose:
        print(
            f"AUC {out['AUC_mean']:.3f}±{out['AUC_std']:.3f} | "
            f"Macro-F1 {out['F1_mean']:.3f} | "
            f"Cost {out['Cost_mean']:.0f}±{out['Cost_std']:.0f} | "
            f"thr(median) {out.get('recommended_threshold','-')} | "
            f"fit {fmt_secs(out['fit_time_mean'])} | pred {fmt_secs(out['pred_time_mean'])}"
        )
    return out


# --------------------------
# Threshold tuning on train then test evaluation
# --------------------------
def tune_threshold_on_train(
    model,
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    cost_fp: int = 10,
    cost_fn: int = 500,
    sampler=None,
    val_size: float = 0.2,
    random_state: int = 42,
) -> float:
    """
    Split train into internal train/val, find cost-optimal threshold on val,
    then refit on full train outside this function.
    """
    y_train = _to_array(y_train)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    (tr_idx, va_idx), = splitter.split(X_train, y_train)
    X_tr, X_va = _slice_xy(X_train, tr_idx), _slice_xy(X_train, va_idx)
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    if sampler is not None:
        X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

    model.fit(X_tr, y_tr)
    prob_val = _scores(model, X_va)
    thr, _ = _best_threshold_by_cost(y_va, prob_val, cost_fp, cost_fn)
    return float(thr)


def evaluate_on_test(
    model,
    X_train, y_train,
    X_test, y_test,
    *,
    threshold: Optional[float] = None,
    cost_fp: int = 10,
    cost_fn: int = 500,
    sampler=None,
    tune_if_none: bool = True,
    random_state: int = 42,
    verbose: bool = True,           # NEW
) -> Dict[str, float]:
    y_train = np.asarray(y_train).ravel()
    y_test  = np.asarray(y_test).ravel()

    # threshold selection (if needed)
    if threshold is None and tune_if_none:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        (tr_idx, va_idx), = sss.split(X_train, y_train)
        X_tr = X_train.iloc[tr_idx] if hasattr(X_train, "iloc") else X_train[tr_idx]
        X_va = X_train.iloc[va_idx] if hasattr(X_train, "iloc") else X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        if sampler is not None:
            X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

        t_fit0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_sel = time.perf_counter() - t_fit0

        prob_val = model.predict_proba(X_va)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_va)
        fpr, tpr, thr = roc_curve(y_va, prob_val)
        P, N = (y_va == 1).sum(), (y_va == 0).sum()
        FP, FN = fpr * N, (1 - tpr) * P
        costs = cost_fp * FP + cost_fn * FN
        threshold = float(thr[int(np.argmin(costs))])
        if verbose:
            print(f"Threshold tuned on train-val in {fmt_secs(fit_sel)} → thr={threshold:.3f}")

    # Refit on full training
    X_fit, y_fit = X_train, y_train
    if sampler is not None:
        X_fit, y_fit = sampler.fit_resample(X_fit, y_fit)

    t_fit = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_time = time.perf_counter() - t_fit

    t_pred = time.perf_counter()
    if hasattr(model, "predict_proba"):
        prob_test = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X_test).astype(float)
        s_min, s_max = np.min(s), np.max(s)
        prob_test = np.full_like(s, 0.5, dtype=float) if s_max == s_min else (s - s_min) / (s_max - s_min)
    else:
        prob_test = model.predict(X_test).astype(float)
    pred_time = time.perf_counter() - t_pred

    thr = 0.5 if threshold is None else float(threshold)
    y_pred = (prob_test >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    auc = roc_auc_score(y_test, prob_test)
    f1  = f1_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, pos_label=1)
    prec= precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    cost = float(cost_fp * fp + cost_fn * fn)

    if verbose:
        print(
            f"Test → AUC={auc:.3f} | F1={f1:.3f} | Cost={cost:.0f} | "
            f"Recall={rec:.3f} | Precision={prec:.3f} | "
            f"fit={fmt_secs(fit_time)} | pred={fmt_secs(pred_time)} | thr={thr:.3f}"
        )

    return {
        "AUC": float(auc),
        "MacroF1": float(f1),
        "Recall_pos": float(rec),
        "Precision_pos": float(prec),
        "FP": int(fp),
        "FN": int(fn),
        "Cost": cost,
        "Threshold": thr,
        "fit_time": fit_time,
        "pred_time": pred_time,
    }
