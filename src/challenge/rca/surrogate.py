import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


def train_surrogate_tree(
    model,
    X,
    y=None,
    max_depth=5,
    random_state=42,
    save_artifacts=False,
    output_dir=".",
    file_prefix="",
):
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
        plt.figure(figsize=(24, 16))
        plot_tree(
            dt,
            feature_names=X.columns,
            filled=True,
            class_names=["NoFailure", "Failure"],
            fontsize=10,
        )
        plt.tight_layout()
        import os

        os.makedirs(output_dir, exist_ok=True)

        plot_path = os.path.join(output_dir, f"{file_prefix}surrogate_tree.png")
        plt.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path}")

    return dt


def extract_rules(
    tree, feature_names, save_artifacts=False, output_dir=".", file_prefix=""
):
    """
    Extracts text rules from the decision tree.
    """
    r_text = export_text(tree, feature_names=list(feature_names))
    print("[Surrogate] Extracting rules...")

    if save_artifacts:
        import os

        os.makedirs(output_dir, exist_ok=True)

        txt_path = os.path.join(output_dir, f"{file_prefix}surrogate_rules.txt")
        with open(txt_path, "w") as f:
            f.write(r_text)
        print(f"Saved: {txt_path}")

    return r_text
