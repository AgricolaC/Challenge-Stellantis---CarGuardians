import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from challenge.data.ingest import load_data

def investigate_feature(feature_name='cn_001'):
    print(f"--- Investigating {feature_name} ---")
    X, y = load_data('dataset/', 'aps_failure_training_set.csv')
    
    # Combine for plotting
    df = X[[feature_name]].copy()
    df['class'] = y
    
    # Drop NaNs and Infs
    df = df.replace([float('inf'), float('-inf')], float('nan')).dropna(subset=[feature_name])
    
    # 1. Statistics
    failures = df[df['class'] == 1]
    healthy = df[df['class'] == 0]
    
    failures_gt_0 = (failures[feature_name] > 0).mean()
    healthy_gt_0 = (healthy[feature_name] > 0).mean()
    
    print(f"Failures (Class 1) with {feature_name} > 0: {failures_gt_0:.4%}")
    print(f"Healthy (Class 0) with {feature_name} > 0: {healthy_gt_0:.4%}")
    print(f"Max value: {df[feature_name].max()}")
    
    # 2. Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature_name, hue='class', element="step", common_norm=False, log_scale=True)
    plt.title(f'Distribution of {feature_name} by Class (Log Scale)')
    plt.savefig(f'{feature_name}_distribution.png')
    print(f"Saved plot to {feature_name}_distribution.png")
    
    if failures_gt_0 > 0.99:
        print(f"\n[VERDICT] {feature_name} captures > 99% of failures. It is likely a Symptom/Mediator.")
    else:
        print(f"\n[VERDICT] {feature_name} does NOT capture > 99% of failures.")

if __name__ == "__main__":
    investigate_feature('cn_001')
