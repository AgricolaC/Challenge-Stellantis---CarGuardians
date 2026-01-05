import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add 'src' to path (relative to src/challenge/tests/test_ironclad.py)
# We need to go up 2 levels: tests -> challenge -> src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
import time

from challenge.data.feature_selection import (engineer_histogram_features,
                                              select_features_fast_consensus,
                                              split_histogram_features)
from challenge.data.ingest import load_data
from challenge.data.preprocess import ScaniaPreprocessor
from challenge.modelling.models import get_models
from challenge.modelling.train_eval import cv_cost
from challenge.visualization.forensics import (plot_chaos_cloud,
                                               plot_distribution_drift,
                                               plot_fan_of_death,
                                               plot_hardware_fingerprint,
                                               plot_killer_bins,
                                               plot_law_of_physics,
                                               plot_overload_tail,
                                               plot_shift_spectrum)


def test_ironclad_protocol():
    print("\n=== IRONCLAD FORENSIC PROTOCOL VERIFICATION ===")

    # Create Results Directory
    # Create Results Directory
    # Path: src/forensic_results/forensic_run_YYYYMMDD-HHMMSS
    base_results_dir = os.path.join(
        os.path.dirname(__file__), "../../results/forensic_results"
    )
    results_dir = os.path.join(
        base_results_dir, "forensic_run_" + time.strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(results_dir, exist_ok=True)
    print(f"Artifacts will be saved to: {results_dir}")

    # 1. Load Real Data
    # Try dynamic paths
    possible_paths = [
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../dataset/")
        ),  # Standard path under src/dataset
        "src/dataset/",  # If running from root
        "../../dataset/",  # Fallback
    ]

    file_name = "aps_failure_training_set.csv"
    X, y = None, None

    for data_path in possible_paths:
        try:
            print(f"Trying to load data from: {data_path}")
            if os.path.exists(os.path.join(data_path, file_name)):
                X, y = load_data(data_path, file_name)
                print(f"Success! Loaded from {data_path}")
                break
        except Exception as e:
            print(f"Failed path {data_path}: {e}")
            continue

    if X is None:
        print("Error: Could not load data.")
        return

    # Subsample for speed if needed (10k is good)
    print("Subsampling to 10,000 stratisfied samples...")
    from sklearn.model_selection import train_test_split

    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=24000, stratify=y, random_state=42
    )

    # 2. Run CV with Internal Feature Selection (The Ironclad Rule)
    print("Running Ironclad CV (Impute -> Select -> Balance -> Train)...")
    model = get_models()["LightGBM"]  # Use Standard LGBM

    results = cv_cost(
        model,
        X_sub,
        y_sub,
        folds=5,
        tune_threshold=True,
        feature_selector=select_features_fast_consensus,
        n_features=10,
        verbose=True,
    )

    # Save Results to JSON
    # Convert ndarrays to list for JSON serialization
    results_serializable = {
        "AUC_mean": float(results["AUC_mean"]),
        "AUC_std": float(results["AUC_std"]),
        "Cost_mean": float(results["Cost_mean"]),
        "Cost_std": float(results["Cost_std"]),
        "F1_mean": float(results["F1_mean"]),
        "selected_features": results["selected_features"],
    }
    with open(os.path.join(results_dir, "cv_results.json"), "w") as f:
        json.dump(results_serializable, f, indent=4)
    print("Saved CV results to cv_results.json")

    # 3. Verify Stability
    print("\n--- Forensic Verification ---")
    selected_features = results["selected_features"]

    # Check if aa_000 is present
    flattened = [f for fold in selected_features for f in fold]
    if "aa_000" in flattened:
        print("PASS: 'aa_000' (Mileage) detected as key physical driver.")
    else:
        print("WARNING: 'aa_000' not selected. Check selection logic.")

    # 4. Generate Plots
    print("Generating Forensic Plots...")
    try:
        plot_law_of_physics(
            results, save_path=os.path.join(results_dir, "law_of_physics.png")
        )

        # Hardware Fingerprint needs the preprocessor fitted
        prep = ScaniaPreprocessor(reduce_missingness=True)
        prep.fit(X_sub)
        plot_hardware_fingerprint(
            prep,
            save_path=os.path.join(results_dir, "hardware_fingerprint_cluster.png"),
        )

        # --- NEW SCALED VISUALIZATIONS ---
        print("Generating Advanced Physical Interaction Plots...")

        # We need engineered features for these plots
        print("Engineering Histogram Features for Visualization...")
        single_val_cols, _, _, hist_groups = split_histogram_features(X_sub)
        X_eng = engineer_histogram_features(X_sub, hist_groups)

        # Add aa_000 to X_eng for correlation analysis
        if "aa_000" in X_sub.columns:
            X_eng["aa_000"] = X_sub["aa_000"].values

        # 1. The Fan of Death (Intensity vs Mileage)
        # 2. The Shift Spectrum (Center of Mass)
        # 3. Killer Bin Analysis

        # Iterate over ALL histogram families found
        for family in hist_groups.keys():
            print(f"Generating Forensic Plots for family: {family}...")

            # Fan of Death (DCI / LLI / Sum)
            # Prioritize DCI (High Load), then LLI (Low Load), then Sum (Quantity)
            feature_candidates = [f"{family}_DCI", f"{family}_LLI", f"{family}_sum"]
            plot_col = next((c for c in feature_candidates if c in X_eng.columns), None)

            if plot_col:
                plot_fan_of_death(
                    X_eng,
                    y_sub,
                    feature_col=plot_col,
                    save_path=os.path.join(results_dir, f"fan_of_death_{family}.png"),
                )

            # Shift Spectrum (Center of Mass)
            com_col = f"{family}_center_mass"
            if com_col in X_eng.columns:
                plot_shift_spectrum(
                    X_eng,
                    y_sub,
                    feature_col=com_col,
                    save_path=os.path.join(results_dir, f"shift_spectrum_{family}.png"),
                )

            # Killer Bin (Raw Bins)
            plot_killer_bins(
                X_sub,
                y_sub,
                family=family,
                save_path=os.path.join(results_dir, f"killer_bin_{family}.png"),
            )

        # 3. The Chaos Cloud (Entropy vs Mileage)
        plot_chaos_cloud(
            X_eng, y_sub, save_path=os.path.join(results_dir, "chaos_cloud.png")
        )

        # 4. The Overload Tail (Skewness)
        plot_overload_tail(
            X_eng, y_sub, save_path=os.path.join(results_dir, "overload_tail.png")
        )

        # Legacy Plots (optional, commented out for focus or kept if useful)
        # plot_pressure_regimes(X_eng, y_sub, save_path=os.path.join(results_dir, "pressure_regimes_peaks.png"))

        # --- NEW PHYSICS VISUALIZATIONS ---
        # 5. True Fan of Death (Intensity - DCI - vs Mileage)
        # Note: DCI features are in X_eng. aa_000 was added to X_eng in line 127
        plot_fan_of_death(
            X_eng, y_sub, save_path=os.path.join(results_dir, "fan_of_death_dci.png")
        )

        # 6. Killer Bin Analysis (Raw Bins)
        # We need X_sub (raw bins) for this
        plot_killer_bins(
            X_sub,
            y_sub,
            family="ag",
            save_path=os.path.join(results_dir, "killer_bin_ag.png"),
        )
        plot_killer_bins(
            X_sub,
            y_sub,
            family="ay",
            save_path=os.path.join(results_dir, "killer_bin_ay.png"),
        )

        # 5. Wasserstein
        from challenge.data.feature_selection import get_wasserstein_features

        # Calculate Healthy Reference (Class 0)
        healthy_ref = {}
        for prefix, cols in hist_groups.items():
            # Get mean distribution of healthy trucks
            # 1. Sum cols roughly represents mass? No, we need PMF.
            # We treat the MEAN histogram of class 0 as the reference.
            X_healthy = X_sub.loc[y_sub == 0, cols]
            # Sum counts across all healthy trucks
            total_counts = X_healthy.sum(axis=0).values
            # Normalize to PMF
            if total_counts.sum() > 0:
                healthy_ref[prefix] = total_counts / total_counts.sum()
            else:
                # Uniform fallback
                healthy_ref[prefix] = np.ones(len(cols)) / len(cols)

        print(
            f"Calculated Healthy Reference Distributions for {len(healthy_ref)} families."
        )
        X_dist = get_wasserstein_features(X_sub, hist_groups, healthy_ref)

        # Add to X_eng for plotting
        X_eng = pd.concat([X_eng, X_dist], axis=1)

        plot_distribution_drift(
            X_eng, save_path=os.path.join(results_dir, "wasserstein_drift.png")
        )

        print("PASS: All forensic plots generated.")

        # --- NON-VISUAL STATISTICAL ANALYSIS ---
        print("\n=== FORENSIC STATISTICS ===")
        stats_report = []
        stats_report.append("=== FORENSIC STATISTICS ===\n")

        # 1. Mileage vs Skew Correlation
        if "ag_skew" in X_eng.columns and "aa_000" in X_eng.columns:
            corr = X_eng["aa_000"].corr(X_eng["ag_skew"])
            msg = f"Mileage (aa_000) vs Skew (ag_skew) Correlation: {corr:.4f}"
            print(msg)
            stats_report.append(msg)
            if abs(corr) > 0.3:
                print(
                    "-> SIGNIFICANT: Aging trucks show shifted pressure distributions."
                )
                stats_report.append(
                    "-> SIGNIFICANT: Aging trucks show shifted pressure distributions."
                )
        else:
            print("Skipping Correlation: aa_000 or ag_skew missing.")

        # 2. Entropy Delta
        entropy_cols = [c for c in X_eng.columns if "entropy" in c]
        if entropy_cols:
            # Just take mean of all entropies for a quick "System Entropy" metric
            sys_entropy = X_eng[entropy_cols].mean(axis=1)
            mean_0 = sys_entropy[y_sub == 0].mean()
            mean_1 = sys_entropy[y_sub == 1].mean()

            msg1 = (
                f"Mean System Entropy | Healthy: {mean_0:.4f} | Failing: {mean_1:.4f}"
            )
            msg2 = f"Disorder Ratio (Fail/Safe): {mean_1/mean_0:.2f}x"
            print(msg1)
            print(msg2)
            stats_report.append(msg1)
            stats_report.append(msg2)

        # 3. Peak Regimes
        peak_cols = [c for c in X_eng.columns if "peaks" in c]
        if peak_cols:
            # Let's check 'ag_peaks' (Engine RPM peaks)
            # If ag_peaks is not there, take first one
            p_col = "ag_peaks" if "ag_peaks" in peak_cols else peak_cols[0]
            mode_0 = X_eng.loc[y_sub == 0, p_col].mode()[0]
            mode_1 = X_eng.loc[y_sub == 1, p_col].mode()[0]
            msg = f"Pressure Regimes ({p_col}) | Most Common Peak Count: Healthy={mode_0}, Failing={mode_1}"
            print(msg)
            stats_report.append(msg)

        # Save stats to file
        with open(os.path.join(results_dir, "forensic_report.txt"), "w") as f:
            f.write("\n".join(stats_report))
        print("Saved statistical report to forensic_report.txt")

    except Exception as e:
        print(f"FAIL: Plotting error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_ironclad_protocol()
