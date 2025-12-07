from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
import numpy as np

# Try importing XGBoost and CatBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of models to be used in the experiment.
    """
    models = {
        "LightGBM": LGBMClassifier(random_state=random_state, verbose=-1),
        #"CustomLightGBM": get_custom_lgbm(random_state=random_state),
        #"RandomForest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
        #"LogisticRegression": LogisticRegression(random_state=random_state, max_iter=1000, n_jobs=-1),
        #"HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state),
        #"MLP": MLPClassifier(random_state=random_state, max_iter=1000)
    }

    if XGBClassifier:
        models["XGBoost"] = XGBClassifier(random_state=random_state, eval_metric='logloss')
    
    #if CatBoostClassifier:
        #models["CatBoost"] = CatBoostClassifier(random_state=random_state, verbose=0, allow_writing_files=False)

    return models

def weighted_logistic_loss(y_true, y_pred, *args, **kwargs):
    """
    Custom objective function for LightGBM to handle imbalanced costs.
    Gradient and Hessian for weighted binary cross entropy.
    
    Loss = - (w_pos * y * log(p) + w_neg * (1-y) * log(1-p))
    
    Gradient (dLoss/dLogOdds) = p - y  (if weights are 1)
    With weights:
    Grad = p * w_neg - y * w_pos  (if y=1, w_pos; if y=0, w_neg * p)
    Actually:
    Grad_i = (p_i - y_i) * weight_i
    Hess_i = p_i * (1 - p_i) * weight_i
    
    Where weight_i = cost_fn if y_i=1 else cost_fp
    """
    # Hardcoded costs for now as passing them via args is tricky with LGBM API
    cost_fp = 10
    cost_fn = 500
    
    # y_pred comes as log-odds (raw scores) from LightGBM in custom objective
    # We need to convert to probability for calculation
    p = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Define weights
    # If y_true is 1 (positive), we want to penalize False Negatives -> weight = Cost_FN
    # If y_true is 0 (negative), we want to penalize False Positives -> weight = Cost_FP
    
    grad = np.zeros_like(y_pred)
    hess = np.zeros_like(y_pred)
    
    # Vectorized implementation
    # Weights: 500 for positives, 10 for negatives
    weights = np.where(y_true == 1, cost_fn, cost_fp)
    
    # Gradient = (p - y) * w
    grad = (p - y_true) * weights
    
    # Hessian = p * (1-p) * w
    hess = p * (1.0 - p) * weights
    
    return grad, hess

class WeightedLGBMClassifier(LGBMClassifier):
    """
    Subclass of LGBMClassifier that properly handles custom objective
    by converting raw scores to probabilities using sigmoid function.
    This avoids the warning: "Cannot compute class probabilities or labels due to the usage of customized objective function."
    """
    def predict(self, X, raw_score=False, start_iteration=0, num_iteration=None,
                pred_leaf=False, pred_contrib=False, **kwargs):
        if raw_score or pred_leaf or pred_contrib:
            if num_iteration is None:
                num_iteration = self.best_iteration_
            return self.booster_.predict(X, start_iteration=start_iteration, 
                                         num_iteration=num_iteration, raw_score=raw_score, 
                                         pred_leaf=pred_leaf, pred_contrib=pred_contrib, **kwargs)
        
        # Standard prediction via probabilities
        probas = self.predict_proba(X, start_iteration=start_iteration, num_iteration=num_iteration, **kwargs)
        # Return class labels (0 or 1) based on probability
        return np.argmax(probas, axis=1)

    def predict_proba(self, X, start_iteration=0, num_iteration=None, **kwargs):
        if num_iteration is None:
            num_iteration = self.best_iteration_
            
        # Get raw log-odds scores directly from booster to avoid recursion issues with super().predict()
        raw_scores = self.booster_.predict(X, start_iteration=start_iteration, num_iteration=num_iteration, raw_score=True, **kwargs)
        
        # Apply sigmoid function to convert log-odds to probability of positive class
        proba_pos = 1.0 / (1.0 + np.exp(-raw_scores))
        # Return (N, 2) array: [prob_neg, prob_pos]
        return np.vstack([1.0 - proba_pos, proba_pos]).T

def get_custom_lgbm(random_state: int = 42) -> LGBMClassifier:
    """
    Returns a LightGBM classifier with the custom weighted objective.
    """
    # Note: We pass the objective function to the constructor or fit.
    # For sklearn API, we can pass 'objective' as the function.
    return WeightedLGBMClassifier(
        random_state=random_state, 
        verbose=-1, 
        objective=weighted_logistic_loss
    )
