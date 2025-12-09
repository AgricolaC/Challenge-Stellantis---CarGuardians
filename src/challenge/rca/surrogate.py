import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

def train_surrogate_tree(model, X, y=None, max_depth=3, random_state=42, save_artifacts=False):
    """
    Trains a shallow decision tree to mimic the complex model's predictions.
    
    Args:
        model: The complex 'black box' model.
        X (pd.DataFrame): Data.
        y: Not strictly used for training (we use model predictions), but kept for signature consistency.
        max_depth (int): Depth of surrogate tree.
        
    Returns:
        DecisionTreeClassifier: The surrogate tree.
    """
    # 1. Generate predictions from the complex model
    # We mimic the probability or class? Guide suggests decision logic.
    # Usually easier to fit to discrete class labels for readable split rules.
    y_pred_complex = model.predict(X)
    
    # 2. Train surrogate
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X, y_pred_complex)
    
    # 3. Score (how well does it copy?)
    fidelity = dt.score(X, y_pred_complex)
    print(f"[Surrogate] Tree Fidelity (Accuracy vs Complex Model): {fidelity:.4f}")
    
    if save_artifacts:
        # Save visualization
        plt.figure(figsize=(12, 8))
        plot_tree(dt, feature_names=X.columns, filled=True, class_names=["NoFailure", "Failure"], fontsize=10)
        plt.tight_layout()
        plt.savefig("surrogate_tree.png", dpi=180, bbox_inches="tight")
        plt.close()
        print("Saved: surrogate_tree.png")
        
    return dt

def extract_rules(tree, feature_names, save_artifacts=False):
    """
    Extracts text rules from the decision tree.
    """
    r_text = export_text(tree, feature_names=list(feature_names))
    print("[Surrogate] Extracting rules...")
    
    if save_artifacts:
        with open("surrogate_rules.txt", "w") as f:
            f.write(r_text)
        print("Saved: surrogate_rules.txt")
        
    return r_text
