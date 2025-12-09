import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def create_decision_matrix(save_artifacts=False, output_dir=".", file_prefix=""):
    """
    Creates and saves a decision matrix comparing RCA techniques.
    Values are hardcoded based on the provided guide or theoretical properties.
    """
    # Define the matrix based on known properties of the methods
    data = {
        "Method": ["SHAP (Tree)", "Permutation (Cost)", "Surrogate Tree", "PDP/ICE"],
        "Local Fidelity": [5, 3, 2, 3],  # SHAP is locally accurate
        "Global Fidelity": [4, 5, 2, 4], # PI is true global importance
        "Actionability": [4, 4, 3, 3],   # Can we act on it?
        "Computational Cost (Inv)": [4, 2, 5, 3], # Surrogate is fast (5), PI is slow (2)
        "Sparsity": [3, 4, 5, 2] # Surrogate is very sparse/simple
    }
    
    df = pd.DataFrame(data)
    
    if save_artifacts:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, f"{file_prefix}decision_matrix.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        
    return df

def plot_radar_chart(matrix_df, output_dir=".", file_prefix=""):
    """
    Generates a radar chart from the decision matrix.
    """
    # Normalize or use raw 1-5 scale? We assume 1-5.
    categories = list(matrix_df.columns[1:])
    N = len(categories)
    
    # Angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=7)
    plt.ylim(0, 5.5)
    
    # Plot each method
    colors = ['b', 'r', 'g', 'y']
    for i, row in matrix_df.iterrows():
        values = row[categories].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=row["Method"], color=colors[i % len(colors)])
        ax.fill(angles, values, colors[i % len(colors)], alpha=0.1)
        
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    import os
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{file_prefix}decision_matrix_radar_clean.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    
