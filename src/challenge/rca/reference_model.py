import numpy as np
import joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_reference_model(lgbm_weighted, X, y, random_state=42):
    """
    Clones and trains a reference LightGBM model using a stratified split.
    
    Args:
        lgbm_weighted: The base LightGBM estimator to clone.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.array): Target vector.
        random_state (int): Random seed.
        
    Returns:
        tuple: (trained_model, X_va, y_va, auc_score)
    """
    # Stratified train-validation split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Clone the model to avoid overwriting the original estimator
    lgbm_ref = clone(lgbm_weighted)
    lgbm_ref.fit(X_tr, y_tr)
    
    # Calculate Validation AUC
    # Note: LightGBM predict_proba returns (n_samples, n_classes)
    auc_valid = roc_auc_score(y_va, lgbm_ref.predict_proba(X_va)[:, 1])
    print(f"[REF] AUC(valid) = {auc_valid:.4f}")
    
    return lgbm_ref, X_va, y_va, auc_valid

def select_cost_threshold(model, X_va, y_va, cost_fp=10.0, cost_fn=500.0, save_artifacts=False, output_dir=".", file_prefix=""):
    """
    Selects the best classification threshold based on a cost function.
    
    Args:
        model: Trained classifier.
        X_va (pd.DataFrame): Validation features.
        y_va (pd.Series or np.array): Validation labels.
        cost_fp (float): Cost of a False Positive.
        cost_fn (float): Cost of a False Negative.
        save_artifacts (bool): Whether to save model and threshold to disk.
        output_dir (str): Directory to save artifacts.
        file_prefix (str): Prefix for artifact filenames.
        
    Returns:
        tuple: (best_threshold, min_cost)
    """
    # Generate a dense threshold grid (especially near 0)
    thr_grid = np.concatenate([
        np.linspace(0.00, 0.10, 101),
        np.linspace(0.11, 0.80, 70),
        np.linspace(0.81, 0.99, 19)
    ])
    thr_grid = np.unique(np.clip(thr_grid, 0, 1))

    proba_va = model.predict_proba(X_va)[:, 1]
    y_va_np = np.asarray(y_va).ravel()

    best_thr, best_cost = None, np.inf
    
    # Vectorized cost calculation could be faster, but loop is readable and sufficient for grid size ~280
    for t in thr_grid:
        y_hat = (proba_va >= t).astype(int)
        FP = np.sum((y_hat == 1) & (y_va_np == 0))
        FN = np.sum((y_hat == 0) & (y_va_np == 1))
        cost = FP * cost_fp + FN * cost_fn
        if cost < best_cost:
            best_cost = cost
            best_thr = float(t)

    print(f"[REF] Best threshold = {best_thr:.3f} | Min cost = {best_cost:.1f}")

    if save_artifacts:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f"{file_prefix}model_final_lgbm.pkl")
        thresh_path = os.path.join(output_dir, f"{file_prefix}threshold.txt")
        
        joblib.dump(model, model_path)
        with open(thresh_path, "w") as f:
            f.write(str(best_thr))
        print(f"Artifacts saved: {model_path}, {thresh_path}")

    return best_thr, best_cost
