import os
import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from challenge.data.feature_selection import (engineer_histogram_features,
                                              get_wasserstein_features,
                                              select_features_kruskal,
                                              split_histogram_features)
from challenge.data.ingest import load_data


def get_real_data():
    """Helper to load real data or return None."""
    data_path = "src/dataset/"
    file_name = "aps_failure_training_set.csv"
    try:
        # print(f"Attempting to load real dataset from {data_path}{file_name}...")
        X, y = load_data(data_path, file_name)
        return X, y
    except Exception as e:
        # print(f"Could not load real data: {e}")
        return None, None


def test_histogram_physics():
    print("\n--- Testing Histogram Physics (Real Data) ---")

    X, y = get_real_data()

    if X is None:
        print("Falling back to TOY data for physics test.")
        df = pd.DataFrame({"ag_000": [10, 0], "ag_001": [1, 1], "ag_002": [0, 10]})
        hist_groups = {"ag": ["ag_000", "ag_001", "ag_002"]}
    else:
        print(f"Using REAL data subset (first 100 rows).")
        df = X.head(100).copy()
        # Dynamically find groups
        _, _, _, hist_groups = split_histogram_features(df)
        if "ag" not in hist_groups:
            print(
                "Warning: 'ag' family not found in real data. Using 'ay' or first available."
            )
            fam = list(hist_groups.keys())[0]
            hist_groups = {fam: hist_groups[fam]}
        else:
            hist_groups = {"ag": hist_groups["ag"]}

    engineered = engineer_histogram_features(df, hist_groups)

    # Just check if we ran without error and produced skew features
    fam = list(hist_groups.keys())[0]
    skew_col = f"{fam}_skew"

    if skew_col in engineered.columns:
        print(f"PASS: Generated {skew_col} from real data.")
        # print(f"Sample Skew values:\n{engineered[skew_col].head()}")

        # Physics Check: Skew should not be strictly 0 for all rows
        if engineered[skew_col].abs().sum() > 0:
            print("PASS: Real data shows physical variance (non-zero skew).")
    else:
        print(f"FAIL: {skew_col} not generated.")


def test_kruskal():
    print("\n--- Testing Kruskal-Wallis (Real Data) ---")
    X, y = get_real_data()

    if X is None:
        print("Falling back to TOY data.")
        X = pd.DataFrame(
            {"good_feat": [1, 2, 3, 10, 11, 12], "bad_feat": [1, 1, 1, 1, 1, 1]}
        )
        y = pd.Series([0, 0, 0, 1, 1, 1])
        n_top = 1
    else:
        print(f"Using REAL data ({X.shape}). Selecting top 5 features.")
        n_top = 5
        # Fill NA for Kruskal (simple fill)
        X = X.fillna(0)

    selected = select_features_kruskal(X, y, top_n=n_top)
    print(f"Selected Top {n_top} via Kruskal: {selected}")

    if len(selected) == n_top:
        print("PASS: Kruskal selection returned correct number of features.")
        if X is not None and "aa_000" in selected:
            print("PASS: Kruskal correctly identified 'aa_000' as a top discriminator.")
    else:
        print("FAIL: Count mismatch.")


def test_wasserstein():
    print("\n--- Testing Wasserstein (Real Data) ---")
    X, y = get_real_data()

    if X is None:
        print("Falling back to TOY data.")
        X = pd.DataFrame({"ag_000": [10, 0], "ag_001": [0, 10]})
        hist_groups = {"ag": ["ag_000", "ag_001"]}
        refs = {"ag": np.array([0.5, 0.5])}
    else:
        print("Using REAL data subset (first 50 rows).")
        X = X.head(50).copy()
        _, _, _, hist_groups = split_histogram_features(X)

        # Create a "Reference" from the mean of the first 10 rows (simulating 'Healthy' average)
        refs = {}
        for fam, cols in hist_groups.items():
            # Sum columns to get mass
            data = X[cols].fillna(0).values
            total_mass = data.sum(axis=1)[:, np.newaxis] + 1e-9
            pmfs = data / total_mass
            mean_pmf = pmfs.mean(axis=0)
            refs[fam] = mean_pmf

    w_feats = get_wasserstein_features(X, hist_groups, refs)

    if not w_feats.empty:
        print(f"PASS: Wasserstein distance computed for {w_feats.shape[1]} families.")
    else:
        print("FAIL: No features returned.")


if __name__ == "__main__":
    test_histogram_physics()
    test_kruskal()
    test_wasserstein()
