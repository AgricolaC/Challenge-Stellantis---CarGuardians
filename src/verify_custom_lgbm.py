
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from challenge.modelling.models import weighted_logistic_loss

def verify_custom_lgbm():
    print("Generating synthetic data...")
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    print("Training default LightGBM...")
    model_default = LGBMClassifier(random_state=42, verbose=-1)
    model_default.fit(X, y)
    probs_default = model_default.predict_proba(X)[:, 1]
    print(f"Default LogLoss: {log_loss(y, probs_default):.4f}")
    
    print("Training custom LightGBM (Weighted)...")
    # Custom objective
    model_custom = LGBMClassifier(random_state=42, verbose=-1, objective=weighted_logistic_loss)
    model_custom.fit(X, y)
    
    # Check predict_proba
    try:
        probs_custom = model_custom.predict_proba(X)[:, 1]
        print(f"Custom Probabilities (First 5): {probs_custom[:5]}")
        print(f"Custom LogLoss: {log_loss(y, probs_custom):.4f}")
        
        if np.any(probs_custom < 0) or np.any(probs_custom > 1):
            print("WARNING: Probabilities are out of [0, 1] range!")
        else:
            print("Probabilities are within valid range.")
            
    except Exception as e:
        print(f"Error predicting with custom model: {e}")

if __name__ == "__main__":
    verify_custom_lgbm()
