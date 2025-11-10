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
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # We'll use this to give an 'smote' option
from copy import deepcopy
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional, Iterable
from tqdm.auto import tqdm
from sklearn.experimental import enable_iterative_imputer


# --- NEW IMPORTS ---
from challenge.data.preprocess import ScaniaPreprocessor
from challenge.data.balancing import balance_with_copula
# --- END NEW IMPORTS ---

#
# --- YOUR HELPER FUNCTIONS (UNCHANGED) ---
# All your great helpers are preserved
#

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
        if s.ndim == 2 and s.shape[1] >= 2:
            return s[:, 1].astype(float)
        return np.asarray(s, dtype=float).ravel()
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float).ravel()
        s_min, s_max = np.min(s), np.max(s)
        if s_max == s_min:
            return np.full_like(s, 0.5, dtype=float)
        return (s - s_min) / (s_max - s_min)
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

#
# --- MODIFIED 'cv_cost' FUNCTION ---
# This is now rebuilt to use the "golden pipeline"
#

def cv_cost(
    model,
    X, y,
    *,
    cost_fp: int = 10,
    cost_fn: int = 500,
    folds: int = 5,
    sampler: Optional[str] = None,  # 'copula', 'smote', or None
    verbose: bool = True,
    return_threshold: bool = False,
    random_state: int = 42,
    show_progress: bool = True,        
) -> Dict[str, Any]:
    """
    Cross-validates the full "Impute -> Balance -> Scale -> Train" pipeline.
    
    X and y are expected to be the *raw* (or K-S selected) data, *not*
    pre-imputed or pre-scaled.
    """
    y = _to_array(y)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    aucs, best_costs, best_thresholds, f1s = [], [], [], []
    fold_times_fit, fold_times_pred = [], []
    
    # Define samplers
    samplers = {
        'smote': SMOTE(random_state=random_state, k_neighbors=5), # k_neighbors=5 is low, good for small minority class
        'copula': balance_with_copula # This is our function
    }

    fold_iter = skf.split(X, y)
    if show_progress:
        fold_iter = tqdm(fold_iter, total=folds, desc="CV folds", leave=False)

    for fold, (tr_idx, va_idx) in enumerate(fold_iter, 1):
        
        X_tr_raw = _slice_xy(X, tr_idx)
        X_va_raw = _slice_xy(X, va_idx)
        y_tr, y_va = y[tr_idx], y[va_idx]

        t0 = time.perf_counter()
        
        # --- 1. Imputation (Golden Pipeline Step 1) ---
        preprocessor = ScaniaPreprocessor()
        X_tr_imputed = preprocessor.fit_transform(X_tr_raw)
        X_va_imputed = preprocessor.transform(X_va_raw)
        
        # --- 2. Balancing (Golden Pipeline Step 2) ---
        X_tr_balanced, y_tr_balanced = X_tr_imputed, y_tr
        if sampler is not None:
            if sampler not in samplers:
                raise ValueError(f"Unknown sampler: {sampler}. Use 'copula' or 'smote'.")
            
            print(f"\nFold {fold}: Balancing with {sampler}...")
            if sampler == 'copula':
                X_tr_balanced, y_tr_balanced = balance_with_copula(X_tr_imputed, y_tr)
            else:
                X_tr_balanced, y_tr_balanced = samplers[sampler].fit_resample(X_tr_imputed, y_tr)
        
        # --- 3. Scaling (Golden Pipeline Step 3) ---
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_balanced)
        X_va_scaled = scaler.transform(X_va_imputed) # Scale val set with train stats
        
        # --- 4. Training ---
        model_fold = deepcopy(model)
        model_fold.fit(X_tr_scaled, y_tr_balanced)
        fit_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        # Use your robust _scores function
        prob = _scores(model_fold, X_va_scaled)
        pred_t = time.perf_counter() - t1

        fold_times_fit.append(fit_t)
        fold_times_pred.append(pred_t)

        # Use your robust _best_threshold function
        best_thr = 0.5  # Hard-code the threshold
        
        auc = roc_auc_score(y_va, prob)
        y_pred = (prob >= best_thr).astype(int)
        
        # Manually calculate metrics at 0.5
        tn, fp, fn, tp = confusion_matrix(y_va, y_pred, labels=[0, 1]).ravel()
        best_cost = float(cost_fp * fp + cost_fn * fn)
        f1 = f1_score(y_va, y_pred, average="macro")
        # --- END MODIFICATION ---

        aucs.append(auc)
        best_costs.append(best_cost) # Append the cost at 0.5
        best_thresholds.append(best_thr) # This will just be a list of 0.5s
        f1s.append(f1)

        if show_progress:
            fold_iter.set_postfix({
                "AUC": f"{auc:.3f}",
                "F1": f"{f1:.3f}",
                "Cost": f"{best_cost:.0f}",
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


#
# --- MODIFIED 'evaluate_on_test' FUNCTION ---
# This is also rebuilt to use the "golden pipeline"
#

def evaluate_on_test(
    model,
    X_train, y_train,
    X_test, y_test,
    *,
    threshold: Optional[float] = None,
    cost_fp: int = 10,
    cost_fn: int = 500,
    sampler: Optional[str] = None, # 'copula', 'smote', or None
    tune_if_none: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates the full pipeline on the test set.
    
    1. Splits train into train/val for threshold tuning.
    2. Runs full (Impute->Balance->Scale->Train) pipeline on train/val.
    3. Finds best threshold on val.
    4. Re-runs full (Impute->Balance->Scale->Train) pipeline on *all* train data.
    5. Evaluates on test data using the tuned threshold.
    """
    y_train = _to_array(y_train)
    y_test  = _to_array(y_test)
    
    samplers = {
        'smote': SMOTE(random_state=random_state, k_neighbors=5, n_jobs=-1),
        'copula': balance_with_copula
    }
    # --- Final Training on Full Data ---
    if verbose:
        print(f"--- Refitting on Full Train Data & Evaluating Test Set ---")
    
    t_fit = time.perf_counter()
    
    # Run pipeline on (full_train, test)
    preprocessor_final = ScaniaPreprocessor()
    X_train_imputed = preprocessor_final.fit_transform(X_train)
    X_test_imputed = preprocessor_final.transform(X_test)

    X_train_balanced, y_train_balanced = X_train_imputed, y_train
    if sampler is not None:
        if sampler not in samplers:
            raise ValueError(f"Unknown sampler: {sampler}.")
        print(f"Final fit: Balancing with {sampler}...")
        if sampler == 'copula':
            X_train_balanced, y_train_balanced = balance_with_copula(X_train_imputed, y_train)
        else:
            X_train_balanced, y_train_balanced = samplers[sampler].fit_resample(X_train_imputed, y_train)

    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train_balanced)
    X_test_scaled = scaler_final.transform(X_test_imputed)
    
    model_final = deepcopy(model)
    model_final.fit(X_train_scaled, y_train_balanced)
    fit_time = time.perf_counter() - t_fit

    # --- Final Evaluation ---
    t_pred = time.perf_counter()
    prob_test = _scores(model_final, X_test_scaled)
    pred_time = time.perf_counter() - t_pred

    thr = float(threshold)
    metrics = _metrics_at_threshold(y_test, prob_test, thr, cost_fp, cost_fn)
    metrics['Threshold'] = thr
    metrics['fit_time'] = fit_time
    metrics['pred_time'] = pred_time
    
    if verbose:
        print(
            f"Test → AUC={metrics['AUC']:.3f} | F1={metrics['MacroF1']:.3f} | Cost={metrics['Cost']:.0f} | "
            f"Recall={metrics['Recall_pos']:.3f} | Precision={metrics['Precision_pos']:.3f} | "
            f"fit={fmt_secs(fit_time)} | pred={fmt_secs(pred_time)} | thr={thr:.3f}"
        )
        print(f"Test CM (thr={thr:.3f}): FP={metrics['FP']:.0f}, FN={metrics['FN']:.0f}")

    # Return full results and artifacts
    return {
        "metrics": metrics,
        "model": model_final,
        "preprocessor": preprocessor_final,
        "scaler": scaler_final,
        "test_probabilities": prob_test
    }