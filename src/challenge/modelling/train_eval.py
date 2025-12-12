from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Union

# Optional type reference to pandas without hard dependency at import time
ArrayLike = Union[np.ndarray, "pd.Series", "pd.DataFrame"]  # noqa

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, confusion_matrix, recall_score, precision_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from copy import deepcopy
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional, Iterable
from tqdm.auto import tqdm
from sklearn.experimental import enable_iterative_imputer
from challenge.data.preprocess import ScaniaPreprocessor
from challenge.data.balancing import balance_with_copula


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
    y_true: np.ndarray, prob: np.ndarray, cost_fp=10, cost_fn=500, max_threshold=0.4
) -> Tuple[float, float]:
    """
    Finds the optimal threshold effectively using a grid scan to handle
    discrete/stepped probability outputs from tree models.
    """
    # 1. Define Search Grid (High precision scan up to max_threshold)
    # 2000 steps ensures precision of ~0.0005 (if max=1.0) or finer
    thresholds = np.linspace(0, max_threshold, 4002)
    
    # 2. Vectorized Cost Calculation
    # Make predicted labels for ALL thresholds at once: (N_samples, N_thresholds)
    # Row i, Col j is True if prob[i] >= thresholds[j]
    preds = prob[:, np.newaxis] >= thresholds[np.newaxis, :]
    
    # Calculate FP and FN for each threshold column
    # FP: Predicted 1 (True) BUT True Label is 0
    fps = np.sum(preds & (y_true[:, np.newaxis] == 0), axis=0)
    
    # FN: Predicted 0 (False) BUT True Label is 1
    fns = np.sum((~preds) & (y_true[:, np.newaxis] == 1), axis=0)
    
    # 3. Compute Total Costs
    costs = (fps * cost_fp) + (fns * cost_fn)
    
    # 4. Select Best
    best_idx = np.argmin(costs)
    return float(thresholds[best_idx]), float(costs[best_idx])

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

from joblib import Parallel, delayed

def _process_fold(
    fold, tr_idx, va_idx, X, y, model, 
    cost_fp, cost_fn, sampler, sampling_strategy, 
    tune_threshold, samplers, max_threshold=0.4
):
    """
    Helper function to process a single CV fold.
    """
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
        
        # print(f"\nFold {fold}: Balancing with {sampler}...") # Avoid printing in parallel
        if sampler == 'copula':
            # Pass sampling_strategy if it's a float, otherwise default to 1.0 if 'auto' (copula needs float)
            strategy = sampling_strategy if isinstance(sampling_strategy, float) else 1.0
            X_tr_balanced, y_tr_balanced = balance_with_copula(X_tr_imputed, y_tr, sampling_strategy=strategy)
        else:
            X_tr_balanced, y_tr_balanced = samplers[sampler].fit_resample(X_tr_imputed, y_tr)
    
    # --- 3. Scaling (Golden Pipeline Step 3) ---
    scaler = StandardScaler()
    # Scikit-learn scaler returns numpy array, striping column names.
    # We reconstruct DataFrame to allow downstream selectors (like KruskalSelector) to use column names.
    _X_tr_scaled_np = scaler.fit_transform(X_tr_balanced)
    _X_va_scaled_np = scaler.transform(X_va_imputed)
    
    # Check if input was DataFrame to preserve columns
    if hasattr(X_tr_balanced, 'columns'):
        X_tr_scaled = pd.DataFrame(_X_tr_scaled_np, columns=X_tr_balanced.columns, index=X_tr_balanced.index)
        X_va_scaled = pd.DataFrame(_X_va_scaled_np, columns=X_tr_balanced.columns, index=X_va_imputed.index)
    else:
        X_tr_scaled = _X_tr_scaled_np
        X_va_scaled = _X_va_scaled_np
    
    # --- 4. Training ---
    model_fold = deepcopy(model)
    model_fold.fit(X_tr_scaled, y_tr_balanced)
    fit_t = time.perf_counter() - t0

    t1 = time.perf_counter()
    prob = _scores(model_fold, X_va_scaled)
    pred_t = time.perf_counter() - t1

    auc = roc_auc_score(y_va, prob)

    if tune_threshold:
        # Find the best threshold to minimize cost
        best_thr, best_cost = _best_threshold_by_cost(y_va, prob, cost_fp, cost_fn)
        y_pred = (prob >= best_thr).astype(int)
    else:
        # Use a fixed 0.5 threshold
        best_thr = 0.5
        y_pred = (prob >= best_thr).astype(int)
        # Calculate the cost at this fixed 0.5 threshold
        tn, fp, fn, tp = confusion_matrix(y_va, y_pred, labels=[0, 1]).ravel()
        best_cost = float(cost_fp * fp + cost_fn * fn)

    f1 = f1_score(y_va, y_pred, average="macro")
    
    return {
        "auc": auc,
        "cost": best_cost,
        "thr": best_thr,
        "f1": f1,
        "fit_time": fit_t,
        "pred_time": pred_t
    }

def cv_cost(
    model,
    X, y,
    *,
    tune_threshold: bool = True,
    cost_fp: int = 10,
    cost_fn: int = 500,
    folds: int = 5,
    sampler: Optional[str] = None,
    sampling_strategy: Union[float, str] = 'auto',
    verbose: bool = True,
    return_threshold: bool = True,
    random_state: int = 42,
    show_progress: bool = True,
    n_jobs: int = -1,
    max_threshold: float = 0.4
) -> Dict[str, Any]:
    """
    Cross-validates the full "Impute -> Balance -> Scale -> Train" pipeline.
    
    The 'tune_threshold' flag controls whether to find the optimal
    cost-based threshold (True) or use a fixed 0.5 threshold (False).
    """
    y = _to_array(y)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    # Define samplers
    samplers = {
        'smote': SMOTE(random_state=random_state, k_neighbors=5, sampling_strategy=sampling_strategy), 
        'copula': balance_with_copula # This is our function
    }

    # Handle "No Sampling" string gracefully
    if sampler == "No Sampling":
        sampler = None

    fold_iter = list(skf.split(X, y))
    
    if verbose:
        print(f"Running {folds}-fold CV with n_jobs={n_jobs}...")

    # Run folds in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_fold)(
            fold, tr_idx, va_idx, X, y, model, 
            cost_fp, cost_fn, sampler, sampling_strategy, 
            tune_threshold, samplers, max_threshold
        ) 
        for fold, (tr_idx, va_idx) in enumerate(fold_iter, 1)
    )

    # Aggregate results
    aucs = [r['auc'] for r in results]
    best_costs = [r['cost'] for r in results]
    best_thresholds = [r['thr'] for r in results]
    f1s = [r['f1'] for r in results]
    fold_times_fit = [r['fit_time'] for r in results]
    fold_times_pred = [r['pred_time'] for r in results]

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
# --- 'evaluate_on_test' FUNCTION (with SMOTE fix) ---
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
    sampling_strategy: Union[float, str] = 'auto', 
    tune_if_none: bool = True,
    random_state: int = 42,
    verbose: bool = True,
    max_threshold: float = 0.4
) -> Dict[str, Any]:
    """
    Evaluates the full pipeline on the test set.
    """
    y_train = _to_array(y_train)
    y_test  = _to_array(y_test)
    
    samplers = {
        'smote': SMOTE(random_state=random_state, k_neighbors=5, sampling_strategy=sampling_strategy), 
        'copula': balance_with_copula
    }

    # Handle "No Sampling" string gracefully
    if sampler == "No Sampling":
        sampler = None

    # --- Threshold tuning step ---
    # This logic is already correct. If tune_if_none=False, it skips
    # this block and threshold remains None.
    if threshold is None and tune_if_none:
        if verbose:
            print("--- Tuning Threshold on Validation Set ---")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        (tr_idx, va_idx), = sss.split(X_train, y_train)

        X_tr_raw = _slice_xy(X_train, tr_idx)
        y_tr = y_train[tr_idx]
        X_va_raw = _slice_xy(X_train, va_idx)
        y_va = y_train[va_idx]
        
        t_fit0 = time.perf_counter()
        
        # Run pipeline on (train_subset, val_subset)
        preprocessor_tune = ScaniaPreprocessor()
        X_tr_imputed = preprocessor_tune.fit_transform(X_tr_raw)
        X_va_imputed = preprocessor_tune.transform(X_va_raw)

        X_tr_balanced, y_tr_balanced = X_tr_imputed, y_tr
        if sampler is not None:
            if sampler not in samplers:
                raise ValueError(f"Unknown sampler: {sampler}.")
            if sampler == 'copula':
                strategy = sampling_strategy if isinstance(sampling_strategy, float) else 1.0
                X_tr_balanced, y_tr_balanced = balance_with_copula(X_tr_imputed, y_tr, sampling_strategy=strategy)
            else:
                X_tr_balanced, y_tr_balanced = samplers[sampler].fit_resample(X_tr_imputed, y_tr)

        scaler_tune = StandardScaler()
        _X_tr_scaled_np = scaler_tune.fit_transform(X_tr_balanced)
        _X_va_scaled_np = scaler_tune.transform(X_va_imputed)
        
        if hasattr(X_tr_balanced, 'columns'):
            X_tr_scaled = pd.DataFrame(_X_tr_scaled_np, columns=X_tr_balanced.columns, index=X_tr_balanced.index)
            X_va_scaled = pd.DataFrame(_X_va_scaled_np, columns=X_tr_balanced.columns, index=X_va_imputed.index)
        else:
            X_tr_scaled = _X_tr_scaled_np
            X_va_scaled = _X_va_scaled_np

        model_tune = deepcopy(model)
        model_tune.fit(X_tr_scaled, y_tr_balanced)
        fit_sel = time.perf_counter() - t_fit0

        prob_val = _scores(model_tune, X_va_scaled)
        threshold, tune_cost = _best_threshold_by_cost(y_va, prob_val, cost_fp, cost_fn, max_threshold)
        if verbose:
            print(f"Threshold tuned in {fmt_secs(fit_sel)} -> thr={threshold:.3f} (Val Cost={tune_cost:.0f})")
    
    elif threshold is None:
        threshold = 0.5 # Default if not tuning
    
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
            strategy = sampling_strategy if isinstance(sampling_strategy, float) else 1.0
            X_train_balanced, y_train_balanced = balance_with_copula(X_train_imputed, y_train, sampling_strategy=strategy)
        else:
            X_train_balanced, y_train_balanced = samplers[sampler].fit_resample(X_train_imputed, y_train)

    scaler_final = StandardScaler()
    _X_train_scaled_np = scaler_final.fit_transform(X_train_balanced)
    _X_test_scaled_np = scaler_final.transform(X_test_imputed)

    # Reconstruct DataFrames if inputs possessed columns
    if hasattr(X_train_balanced, 'columns'):
         X_train_scaled = pd.DataFrame(_X_train_scaled_np, columns=X_train_balanced.columns, index=X_train_balanced.index)
         X_test_scaled = pd.DataFrame(_X_test_scaled_np, columns=X_train_balanced.columns, index=X_test_imputed.index)
    else:
         X_train_scaled = _X_train_scaled_np
         X_test_scaled = _X_test_scaled_np
    
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