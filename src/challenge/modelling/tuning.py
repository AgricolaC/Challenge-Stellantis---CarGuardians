import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from challenge.modelling.models import weighted_logistic_loss
from challenge.data.preprocess import ScaniaPreprocessor
from challenge.data.balancing import balance_with_copula

def _best_cost(y_true, probs, cost_fp=10, cost_fn=500):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    # Cost = FP*10 + FN*500
    # FP = fpr * N
    # FN = (1-tpr) * P
    N = (y_true == 0).sum()
    P = (y_true == 1).sum()
    
    costs = cost_fp * (fpr * N) + cost_fn * ((1 - tpr) * P)
    return np.min(costs)

def objective(trial, X, y, cost_fp=10, cost_fn=500):
    # Hyperparameters to tune
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    # Use custom objective
    model = LGBMClassifier(
        random_state=42, 
        verbose=-1,
        objective=weighted_logistic_loss,
        **params
    )
    
    # CV
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    costs = []
    
    for tr_idx, va_idx in skf.split(X, y):
        # Slice data
        if isinstance(X, pd.DataFrame):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        else:
            X_tr, X_va = X[tr_idx], X[va_idx]
            
        if isinstance(y, pd.Series):
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        else:
            y_tr, y_va = y[tr_idx], y[va_idx]
        
        # Pipeline (Simplified for speed in tuning)
        # 1. Impute
        prep = ScaniaPreprocessor()
        X_tr_imp = prep.fit_transform(X_tr)
        X_va_imp = prep.transform(X_va)
        
        # 2. Balance (Using Copula as it was effective)
        # Note: Copula might be slow. If too slow, we might switch to SMOTE or just weights.
        # For tuning, maybe we can skip complex balancing or use a faster one?
        # But we want to tune FOR the pipeline.
        # Let's use Copula but maybe with fewer samples or faster settings if possible.
        # For now, use full Copula.
        try:
            X_tr_bal, y_tr_bal = balance_with_copula(X_tr_imp, y_tr)
        except Exception:
            # Fallback if copula fails
            X_tr_bal, y_tr_bal = X_tr_imp, y_tr
        
        # 3. Scale
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_bal)
        X_va_sc = scaler.transform(X_va_imp)
        
        # 4. Train
        model.fit(X_tr_sc, y_tr_bal)
        
        # 5. Evaluate Cost
        # With custom objective, predict_proba might return raw scores (1D) or (N, 2)
        raw_preds = model.predict_proba(X_va_sc)
        if raw_preds.ndim == 2:
            probs = raw_preds[:, 1]
        else:
            # Assume raw scores (log odds) -> apply sigmoid
            probs = 1.0 / (1.0 + np.exp(-raw_preds))
        
        # Calculate best cost for this fold
        fold_cost = _best_cost(y_va, probs, cost_fp, cost_fn)
        costs.append(fold_cost)
        
    return np.mean(costs)

def tune_lightgbm(X, y, n_trials=20):
    """
    Runs Optuna optimization for LightGBM.
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    print("Best params:", study.best_params)
    print("Best cost:", study.best_value)
    
    return study.best_params
