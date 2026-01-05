import os
import sys

# Add src to python path to allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

print("DEBUG: File loaded.")
import os
import sys
import warnings

import arviz as az
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import shap
import statsmodels.api as sm
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam import DirectLiNGAM
from causallearn.utils.cit import chisq
# Causal Imports
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from lightgbm import LGBMClassifier
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import ks_2samp, kurtosis, skew, spearmanr
# Statistical Imports
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from challenge.data.feature_selection import (create_engineered_feature_set,
                                              select_features_fast_consensus,
                                              select_features_ks)
from challenge.data.ingest import load_data
from challenge.data.outliers import (analyze_isolation_forest_outliers,
                                     fit_predict_isolation_forest)
from challenge.data.preprocess import ScaniaPreprocessor

# Suppress specific scipy warnings that occur with constant/near-constant features





def configure_priors(
    labels,
    feature_groups=None,
    target_col="class",
    time_col="aa_000",
    rule_class_sink=True,
    rule_time_source=True,
    rule_mediation=True,
):
    n = len(labels)
    # Matrix convention: priors[i, j] = -1 means j -> i is FORBIDDEN
    # Row (i) = Effect, Col (j) = Cause
    priors = np.zeros((n, n))

    # Mapping for Hierarchy: Global (0) -> Secondary (1) -> Diagnostic (2)
    # Lower ID cannot be caused by Higher ID (e.g., 2 cannot cause 0)
    hierarchy_map = {"Global": 0, "Secondary": 1, "Diagnostic": 2, "Unclassified": 3}

    try:
        t_idx = labels.index(target_col)
        time_idx = labels.index(time_col) if time_col in labels else -1

        # 1. Standard Physics Rules
        if rule_class_sink:
            priors[:, t_idx] = -1  # Class cannot cause anything

        if time_idx != -1:
            if rule_time_source:
                priors[time_idx, :] = -1  # Nothing causes Time
            if rule_mediation:
                priors[t_idx, time_idx] = -1  # Time cannot cause Class directly

        # 2. Correlation-Based Hierarchy Rules
        if feature_groups:
            print("  Applying Correlation Hierarchy Rules...")
            for i, feat_i in enumerate(labels):
                if feat_i == target_col or feat_i == time_col:
                    continue

                group_i = feature_groups.get(feat_i, "Unclassified")
                rank_i = hierarchy_map.get(group_i, 3)

                for j, feat_j in enumerate(labels):
                    if i == j:
                        continue
                    if feat_j == target_col or feat_j == time_col:
                        continue

                    group_j = feature_groups.get(feat_j, "Unclassified")
                    rank_j = hierarchy_map.get(group_j, 3)

                    # RULE: Higher Rank cannot cause Lower Rank
                    # e.g. Diagnostic (2) cannot cause Global (0)
                    if rank_j > rank_i:
                        # j -> i is FORBIDDEN
                        priors[i, j] = -1

    except ValueError:
        pass

    return priors.astype(int)


def smart_transform_collapsed(df_collapsed, skew_threshold=2.0):
    """
    Applies Log1p to columns with high skewness (> threshold).
    """
    print(f"--- Applying Smart Transformation (Log1p if Skew > {skew_threshold}) ---")
    df_trans = df_collapsed.copy()

    transform_count = 0
    for col in df_trans.columns:
        # Calculate Skewness
        s = skew(df_trans[col])

        if s > skew_threshold:
            # Check for negative values
            if df_trans[col].min() < 0:
                df_trans[col] = df_trans[col].clip(lower=0)

            # Apply Log1p
            df_trans[col] = np.log1p(df_trans[col])
            transform_count += 1
            # print(f"  [Transform] {col:<15} | Skew: {s:.2f} -> Log1p")

    print(f" Transformed {transform_count} features.")
    return df_trans


def balance_for_structure_learning(X, y, ratio=1.0):
    """
    Performs Random Undersampling to balance the dataset for Causal Structure Learning.
    Keeps all failures (class=1) and samples healthy (class=0) to match ratio.
    """
    print(f"--- Balancing Data for Structure Learning (Ratio: {ratio}) ---")

    # Indices
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg = int(n_pos * ratio)

    print(f"  Failures: {n_pos}, Healthy (Sampled): {n_neg}")

    # Random Sample Healthy
    np.random.seed(42)
    neg_sampled = np.random.choice(neg_idx, n_neg, replace=False)

    # Combine & Shuffle
    indices = np.concatenate([pos_idx, neg_sampled])
    np.random.shuffle(indices)

    X_balanced = X.iloc[indices].copy()
    y_balanced = y.iloc[indices].copy()

    return X_balanced, y_balanced


def analyze_feature_relationships(X, y, reference_col, families=None):
    """
    Analyzes the relationship of a reference column (e.g., 'ay_002') against
    cumulative feature families and the target class.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        reference_col (str): The column to analyze (e.g., 'ay_002', 'bg_000').
        families (list): List of feature family prefixes to aggregate (e.g., ['ag', 'ay']).
    """
    if families is None:
        families = ["ag", "ay", "ba", "cn", "cs", "ee"]

    print(f"\n=== Analysis for Reference Column: {reference_col} ===")

    # --- TEST 1: Family Correlations & Scatter Plots ---
    n_families = len(families)
    n_cols = 3
    n_rows = (n_families + n_cols - 1) // n_cols

    plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    for i, family in enumerate(families):
        # Get all columns for this family
        cols = [c for c in X.columns if c.startswith(family + "_") and c[-1].isdigit()]

        if not cols:
            continue

        # Calculate Sum of the family for each truck
        family_sum = X[cols].sum(axis=1)

        # Calculate Correlation with reference_col
        corr, _ = spearmanr(X[reference_col], family_sum, nan_policy="omit")

        # Plot
        plt.subplot(n_rows, n_cols, i + 1)
        plt.scatter(X[reference_col], family_sum, alpha=0.1, s=1)
        plt.title(f"{reference_col} vs Sum({family}_*)\nSpearman Corr: {corr:.4f}")
        plt.xlabel(reference_col)
        plt.ylabel(f"Total Events ({family})")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- TEST 2: The "Correlation King" Test ---
    print(f"\n--- Top 10 Features Correlated with {reference_col} ---")
    # Calculate correlations of EVERYTHING against reference_col
    correlations = X.corrwith(X[reference_col], method="spearman").abs()
    top_corrs = correlations.sort_values(ascending=False).head(11)  # Top 10 + itself
    print(top_corrs)

    # --- TEST 3: The "Wear-Out" Test ---
    print(f"\n--- Class Distribution by {reference_col} ---")
    plt.figure(figsize=(10, 6))

    # Combine for plotting
    data_plot = pd.concat([X[reference_col], y.rename("class")], axis=1)

    sns.kdeplot(
        data=data_plot, x=reference_col, hue="class", common_norm=False, fill=True
    )
    plt.title(f"Distribution of {reference_col}: Healthy (0) vs Failure (1)")
    plt.show()

    # Check descriptive stats
    print(X[reference_col].groupby(y).describe())


def cluster_features_by_correlation(X, features, families=None, X_raw=None):
    """
    Clusters features into 'Global', 'Secondary', or 'Diagnostic'
    based on their mean Spearman correlation with cumulative feature families.

    Args:
        X (pd.DataFrame): The feature matrix to calculate correlations against (e.g. X_transformed).
        features (list): List of features to cluster.
        families (list): Feature families to sum.
        X_raw (pd.DataFrame): Optional. Reference dataframe containing raw family columns
                              (e.g. X_clean). Use this to calculate family sums if X
                              no longer contains them (e.g. after transform).
    """
    if families is None:
        families = ["ag", "ay", "ba", "cn", "cs", "ee"]

    print(f"\n--- Clustering {len(features)} Features by Correlation ---")

    # 1. Determine Source for Family Sums
    df_source = X_raw if X_raw is not None else X

    # 2. Pre-calculate Family Sums
    family_sums = {}
    for fam in families:
        # Look in source df
        cols = [
            c for c in df_source.columns if c.startswith(fam + "_") and c[-1].isdigit()
        ]
        if cols:
            family_sums[fam] = df_source[cols].sum(axis=1)

    if not family_sums:
        print(
            "  Warning: No families found for correlation. Defaulting all to 'Secondary'."
        )
        return {f: "Secondary" for f in features}

    feature_groups = {}

    for feat in features:
        if feat not in X.columns:
            continue

        corrs = []
        for fam_sum in family_sums.values():
            c, _ = spearmanr(X[feat], fam_sum, nan_policy="omit")
            if not np.isnan(c):
                corrs.append(abs(c))

        avg_corr = np.mean(corrs) if corrs else 0

        if avg_corr > 0.75:
            group = "Global"  # Age/Mileage Proxy
        elif avg_corr > 0.25:
            group = "Secondary"  # Duty Cycle
        else:
            group = "Diagnostic"  # Specific Fault

        feature_groups[feat] = group
        # print(f"  {feat:<12} | Corr: {avg_corr:.3f} -> {group}")

    # Stats
    counts = pd.Series(feature_groups.values()).value_counts()
    print(f"  Clustering Results: {counts.to_dict()}")

    return feature_groups


def filter_redundant_global_features(
    features, feature_groups, preferred_globals=["aa_000", "ci_000"]
):
    """
    Ensures ONLY ONE 'Global' feature is kept to represent system age/usage.
    Prioritizes 'aa_000' (Time), then 'ci_000', then others.
    """

    # 1. Identify all Global features
    global_features = [
        f for f in features if feature_groups.get(f, "Unclassified") == "Global"
    ]

    if len(global_features) <= 1:
        return features  # No redundancy

    print(f"  [Filter] Found {len(global_features)} Global Features: {global_features}")

    # 2. Select the Best Global
    best_global = None

    # Check preferences
    for pref in preferred_globals:
        if pref in global_features:
            best_global = pref
            break

    # Fallback if no preferred global is found (pick random/first)
    if best_global is None:
        best_global = global_features[0]

    print(f"  [Filter] Selected Global: {best_global}. Dropping others.")

    # 3. Construct Final List
    final_features = []
    dropped_count = 0
    for feat in features:
        # If it's a global feature but NOT the best one, skip it
        if feat in global_features and feat != best_global:
            dropped_count += 1
            continue
        final_features.append(feat)

    return final_features


def run_pc_algorithm(
    X,
    y,
    features,
    feature_groups,
    rule_class_sink=True,
    rule_time_source=True,
    rule_mediation=True,
):
    print(
        f"\n--- [Engine A] PC Algorithm (Discrete) | Rules: Class={rule_class_sink}, Time={rule_time_source}, Med={rule_mediation} ---"
    )

    # --- Feature Filtering (Added Step) ---
    # We must ensure only 1 global feature exists for PC as well
    # feature_groups is now PASSED IN, so we don't need to recalculate it here.
    features_filtered = filter_redundant_global_features(features, feature_groups)

    est = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")

    df = X[features_filtered].copy()

    # Discretize continuous
    for c in df.columns:
        if df[c].nunique() > 3:
            try:
                df[c] = (
                    pd.qcut(df[c], q=3, duplicates="drop", labels=False)
                    .fillna(0)
                    .astype(int)
                )
            except ValueError:
                # Fallback
                df[c] = pd.cut(df[c], bins=3, labels=False).fillna(0).astype(int)

    data = df.values.astype(int)
    labels = df.columns.tolist() + ["class"]
    data_full = np.column_stack((data, y.values.astype(int)))

    bk = BackgroundKnowledge()
    nodes = [GraphNode(f"X{i+1}") for i in range(len(labels))]

    # Apply Basic Rules to PC Background Knowledge
    t_idx = len(labels) - 1  # Class index

    # Rule 1: Class cannot cause Features
    if rule_class_sink:
        for i in range(t_idx):
            # Forbidden: Class -> Feature_i
            bk.add_forbidden_by_node(nodes[t_idx], nodes[i])

    if "aa_000" in labels:
        aa_idx = labels.index("aa_000")
        # Rule 2: Features cannot cause Time
        if rule_time_source:
            for i in range(len(labels) - 1):
                if i != aa_idx:
                    # Forbidden: Feature_i -> Time
                    bk.add_forbidden_by_node(nodes[i], nodes[aa_idx])

        # Rule 3: Time cannot cause Class directly (Mediation Rule)
        if rule_mediation:
            # Forbidden: Time -> Class
            bk.add_forbidden_by_node(nodes[aa_idx], nodes[t_idx])

    cg = pc(data_full, 0.05, chisq, True, 0, -1, background_knowledge=bk)

    adj = np.zeros((len(labels), len(labels)))
    for edge in cg.G.get_graph_edges():
        n1 = int(edge.node1.name.replace("X", "")) - 1
        n2 = int(edge.node2.name.replace("X", "")) - 1
        if "-->" in str(edge):
            adj[n1, n2] = 1
        elif "<--" in str(edge):
            adj[n2, n1] = 1

    return adj, labels


def run_lingam_algorithm(
    X,
    y,
    features,
    feature_groups,
    rule_class_sink=True,
    rule_time_source=True,
    rule_mediation=True,
    priors=None,
):
    print(
        f"\n--- [Engine B] DirectLiNGAM (Continuous) | Rules: Class={rule_class_sink}, Time={rule_time_source}, Med={rule_mediation} ---"
    )
    df = X[features].copy()

    # If we are using collapsed features, we don't need to filter _is_missing here necessarily,
    # but let's keep it safe if they exist.
    phys_features = [f for f in df.columns if not f.endswith("_is_missing")]
    df = df[phys_features]
    df["class"] = y.values
    labels = df.columns.tolist()

    if priors is None:

        # --- CRITICAL FIX: Drop Redundant Globals ---
        # We only want one "Global/Age" factor.
        features_to_keep = filter_redundant_global_features(labels, feature_groups)

        # Update Data and Labels
        df = df[features_to_keep]
        labels = features_to_keep

        # Update Priors based on cleaned list
        prior_knowledge = configure_priors(
            labels,
            feature_groups=feature_groups,
            rule_class_sink=rule_class_sink,
            rule_time_source=rule_time_source,
            rule_mediation=rule_mediation,
        )
    else:
        # If priors are passed manually, we strictly respect them
        prior_knowledge = priors

    model = DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(df)
    print("DEBUG: LiNGAM Fit Complete")
    print(f"DEBUG: Adjacency Matrix Shape: {model.adjacency_matrix_.shape}")
    # Check if Class is Source (Column has non-zeros)
    class_idx = labels.index("class")
    print(
        f"DEBUG: Class Column (As Cause) Sum: {np.sum(np.abs(model.adjacency_matrix_[:, class_idx]))}"
    )
    print(
        f"DEBUG: Class Row (As Effect) Sum: {np.sum(np.abs(model.adjacency_matrix_[class_idx, :]))}"
    )

    # Transpose: LiNGAM gives Col->Row, we want Row->Col standard
    adj = model.adjacency_matrix_.T
    clean_adj = np.zeros_like(adj)

    t_idx = labels.index("class")
    time_idx = labels.index("aa_000") if "aa_000" in labels else -1

    # --- SURGICAL CLEANUP ---
    for i in range(len(labels)):  # i is Source
        for j in range(len(labels)):  # j is Target

            weight = abs(adj[i, j])

            # 1. Thresholding (Lower to 0.01 to catch weak mediators)
            if weight < 0.01:
                continue

            # 2. Enforce Physics: Class is Sink
            # If Class -> j, we flip it to j -> Class
            if rule_class_sink and i == t_idx:
                # SPECIAL CASE: If j is Time, flipping creates Time -> Class, which is forbidden.
                # So if j == time_idx, we must DELETE if mediation is enforced.
                if j == time_idx and rule_mediation:
                    print(
                        f"  [Physics] Pruned forbidden edge (Class -> Time): {labels[i]} -> {labels[j]}"
                    )
                    continue

                clean_adj[j, t_idx] = 1
                continue

            # 3. Enforce Physics: Time is Source
            # If i -> Time, we flip it to Time -> i
            if rule_time_source and j == time_idx:
                clean_adj[time_idx, i] = 1
                continue

            # 4. Enforce Physics: MEDIATION RULE
            # If Time -> Class, we DELETE it (Force logic through components)
            if rule_mediation and i == time_idx and j == t_idx:
                print(
                    f"  [Physics] Pruned forbidden edge: {labels[i]} -> {labels[j]} (Weight: {weight:.4f})"
                )
                continue

            # 5. Keep valid edge
            clean_adj[i, j] = 1

    return clean_adj, labels


def analyze_graph_structure(adj, labels, target_node="class", force_target_sink=False):
    G = nx.DiGraph()
    for i, label in enumerate(labels):
        G.add_node(label)
    rows, cols = np.where(adj > 0)
    for r, c in zip(rows, cols):
        source = labels[r]
        target = labels[c]
        # G.add_edge(target, source) # Flip
        # print(f"  [Physics Fix] Flipped: {target} -> {source}")
        G.add_edge(source, target)

    if target_node not in G:
        return [], [], [], G

    parents = list(G.predecessors(target_node))
    if not parents:
        return [], [], [], G

    ancestors = list(nx.ancestors(G, target_node))
    roots = [n for n in ancestors if G.in_degree(n) == 0 and n != target_node]
    mediators = [n for n in ancestors if G.in_degree(n) > 0 and n != target_node]

    return roots, mediators, parents, G


def classify_causal_nodes(adj, labels, target="class", time_node="aa_000"):
    """
    Classifies nodes into Physical Roots, Mediators, or Unconnected.
    """
    G = nx.DiGraph()
    for i, lbl in enumerate(labels):
        G.add_node(lbl)

    rows, cols = np.where(adj > 0)
    for r, c in zip(rows, cols):
        G.add_edge(labels[r], labels[c])  # Source -> Target

    results = []

    for node in labels:
        if node == target or node == time_node:
            continue

        role = "Unconnected"
        if nx.has_path(G, node, target):
            parents = list(G.predecessors(node))

            # Is it a Root? (No parents OR only parent is Time)
            is_root = False
            if len(parents) == 0:
                is_root = True
            elif len(parents) == 1 and parents[0] == time_node:
                is_root = True

            if is_root:
                role = "Physical Root"
            else:
                role = "Mediator"

        results.append({"Feature": node, "Role": role})

    df_res = pd.DataFrame(results)
    if df_res.empty:
        df_res = pd.DataFrame(columns=["Feature", "Role"])

    return df_res, G


def run_pymc_validation(X, y, features, name):
    if not features:
        return None, []
    print(f"Running PyMC for {name} on {features}...")

    # Collinearity Pruning
    df_feat = X[features]
    corr = df_feat.corr().abs()
    drop = set()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            if f1 in drop or f2 in drop:
                continue
            if corr.loc[f1, f2] > 0.80:
                c1 = abs(df_feat[f1].corr(y))
                c2 = abs(df_feat[f2].corr(y))
                drop.add(f1 if c1 < c2 else f2)

    final_feats = [f for f in features if f not in drop]

    # Standardize
    X_data = (X[final_feats] - X[final_feats].mean()) / (X[final_feats].std() + 1e-6)

    with pm.Model() as model:
        # Switch to Normal for better stability (StudentT tails can cause divergences)
        beta = pm.Normal("beta", mu=0, sigma=2.5, shape=len(final_feats))
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        mu = alpha + pm.math.dot(X_data.values, beta)
        theta = pm.math.sigmoid(mu)
        y_obs = pm.Bernoulli("y_obs", p=theta, observed=y.values)

        trace = pm.sample(1000, tune=1000, target_accept=0.94, progressbar=True)

    return az.summary(trace, var_names=["beta"]), final_feats


# --- 5. Age-Adjusted Bayesian Validation ---
def run_pymc_age_adjusted(X, y, features, name, control_feat="aa_000"):
    """
    Runs Logistic Regression for each feature independently,
    CONDITIONED on the control feature (Time).
    """
    if not features:
        return None, []
    print(f"Running Age-Adjusted PyMC for {name}...")

    # If control feature is missing, fallback to standard run
    if control_feat not in X.columns:
        print(f"Control feature {control_feat} not found. Running standard.")
        # ... (fallback logic) ...
        return None, []

    results = []

    # We run a SEPARATE model for each feature to isolate its marginal contribution
    # Model: Logit(p) = alpha + beta_1 * Feature + beta_2 * Age

    # Standardize Age once
    age_data = X[control_feat].values
    age_scaled = (age_data - age_data.mean()) / (age_data.std() + 1e-6)

    for feat in features:
        print(f"  Modeling {feat} | {control_feat}...")

        # Prep Feature
        feat_data = X[feat].values
        feat_scaled = (feat_data - feat_data.mean()) / (feat_data.std() + 1e-6)

        # Combine into design matrix [N, 2]
        X_mat = np.column_stack([feat_scaled, age_scaled])

        with pm.Model() as model:
            # Priors
            # Switch to Normal for better stability/regularization (StudentT can be too heavy-tailed)
            beta = pm.Normal("beta", mu=0, sigma=2.5, shape=2)
            alpha = pm.Normal("alpha", mu=0, sigma=1)

            mu = alpha + pm.math.dot(X_mat, beta)
            theta = pm.math.sigmoid(mu)

            y_obs = pm.Bernoulli("y_obs", p=theta, observed=y.values)

            # Fast sampling, single core to prevent multiprocessing overhead/crashes in loop
            trace = pm.sample(
                600, tune=600, target_accept=0.95, progressbar=False, cores=1
            )

        # Extract ONLY the coefficient for the feature (index 0), ignore Age (index 1)
        summary = az.summary(trace, var_names=["beta"])
        mean_val = summary.iloc[0]["mean"]  # Beta_Feature
        hdi_3 = summary.iloc[0]["hdi_3%"]
        hdi_97 = summary.iloc[0]["hdi_97%"]

        results.append(
            {
                "Feature": feat,
                "Mean": mean_val,
                "Odds_Ratio": np.exp(mean_val),
                "HDI_3%": hdi_3,
                "HDI_97%": hdi_97,
            }
        )

    return pd.DataFrame(results)


def rank_features_age_adjusted(X, y, features, control_feat="aa_000"):
    """
    Rapidly ranks features by their impact on Failure,
    controlling for Age (aa_000).
    Method: Frequentist Logistic Regression (MLE)
    Formula: Logit(P) = Beta_0 + Beta_1 * Feature + Beta_2 * Age
    """
    print(f"\n--- Rapid Age-Adjusted Screening ({len(features)} candidates) ---")

    if control_feat not in X.columns:
        print(f"CRITICAL: Control feature {control_feat} missing. Aborting control.")
        return None

    results = []

    age_scaled = X[control_feat].values

    # Add constant for intercept
    # Design Matrix base: [Intercept, Age]
    X_base = np.column_stack([np.ones(len(age_scaled)), age_scaled])

    for feat in features:
        # Skip the control feature itself to avoid Singular Matrix (Age vs Age)
        if feat == control_feat:
            continue

        # Assume Feature is already standardized
        f_scaled = X[feat].values

        # specific Design Matrix: [Intercept, Age, Feature]
        # We put Feature last to easily grab its coefficient
        X_design = np.column_stack([X_base, f_scaled])

        try:
            # Fit Logit
            model = sm.Logit(y, X_design)
            result = model.fit(disp=0)  # disp=0 suppresses convergence messages

            # Extract Feature Stats (Index 2 corresponding to the Feature)
            coef = result.params.iloc[2]
            p_value = result.pvalues.iloc[2]

            # Confidence Intervals (2.5%, 97.5%)
            conf = result.conf_int().iloc[2]

            results.append(
                {
                    "Feature": feat,
                    "Coeff (LogOdds)": coef,
                    "Odds_Ratio": np.exp(coef),  # The Multiplier of Risk
                    "P_Value": p_value,
                    "Lower_CI": conf.iloc[0],
                    "Upper_CI": conf.iloc[1],
                    "Age_Conf_Factor": result.params.iloc[1],  # How much Age mattered
                }
            )

        except Exception as e:
            print(f" Failed to fit {feat}: {type(e).__name__} {e}")

    # Convert to DataFrame and Rank
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        # Sort by Absolute Coefficient (Impact) - Biggest drivers first
        df_results = df_results.sort_values(
            by="Coeff (LogOdds)", key=abs, ascending=False
        )

    return df_results


def create_hybrid_graph(
    lingam_adj,
    pc_adj,
    labels_lingam,
    labels_pc,
    target="class",
    time_node="aa_000",
    rule_time_source=True,
):
    print("\n--- Creating Hybrid Ensemble Graph (LiNGAM + PC) ---")
    G = nx.DiGraph()
    # Use LiNGAM labels as the base set of nodes
    for lbl in labels_lingam:
        G.add_node(lbl)

    # 1. Add LiNGAM Edges (High Confidence in Direction)
    rows_l, cols_l = np.where(lingam_adj > 0)
    for r, c in zip(rows_l, cols_l):
        if r < len(labels_lingam) and c < len(labels_lingam):
            src, tgt = labels_lingam[r], labels_lingam[c]
            G.add_edge(src, tgt, type="LiNGAM", color="black", weight=2)

    # 2. Add PC Edges (High Confidence in Connection)
    rows_p, cols_p = np.where(pc_adj > 0)
    for r, c in zip(rows_p, cols_p):
        if r < len(labels_pc) and c < len(labels_pc):
            src, tgt = labels_pc[r], labels_pc[c]

            # Check if connection exists in LiNGAM (in ANY direction)
            if not G.has_edge(src, tgt) and not G.has_edge(tgt, src):
                # Physics Check: Never allow Child -> Time (Only if Time is Source)
                if rule_time_source and tgt == time_node:
                    src, tgt = tgt, src  # Flip to Time -> Child

                # Physics Check: Never allow Class -> Parent
                if src == target:
                    src, tgt = tgt, src  # Flip to Parent -> Class

                print(f" [Hybrid] Adding PC-only edge: {src} -> {tgt}")
                G.add_edge(src, tgt, type="PC_Rescue", color="blue", weight=1)

    return G


def analyze_hybrid_layers(G, time_node="aa_000", target="class"):
    """
    Validates if we achieved the 3-4 layer structure.
    """
    print("\n--- Analyzing Hybrid Graph Layers ---")

    # Layer 1: Source (Time)
    layer_1 = [time_node]

    # Layer 2: Roots (Children of Time, but not Mediators)
    # Nodes caused by Time, but also cause other things (not just class)
    layer_2 = []
    if time_node in G:
        children_of_time = list(G.successors(time_node))
        for node in children_of_time:
            if node == target:
                continue

            # If it has children that are NOT class, it's Layer 2
            grand_children = list(G.successors(node))
            if len(grand_children) > 0 and any(gc != target for gc in grand_children):
                layer_2.append(node)

    # Layer 3: Mediators (Children of Layer 2)
    layer_3 = []
    for node in layer_2:
        children = list(G.successors(node))
        for child in children:
            if child != target and child not in layer_2 and child not in layer_3:
                layer_3.append(child)

    print(f"Layer 1 (Source): {layer_1}")
    print(f"Layer 2 (Work/Counters): {layer_2}")
    print(f"Layer 3 (Stress/Mediators): {layer_3}")
    print(f"Layer 4 (Failure): ['{target}']")

    return layer_1, layer_2, layer_3


def visualize_hybrid(G, filename="rca_hybrid_graph.png"):
    plt.figure(figsize=(24, 14))

    try:
        pos = nx.spring_layout(G, k=0.6, seed=42)
    except:
        pos = nx.shell_layout(G)

    edges = G.edges(data=True)
    colors = [d.get("color", "gray") for u, v, d in edges]
    weights = [d.get("weight", 1) for u, v, d in edges]

    # Draw Nodes
    node_colors = []
    for n in G.nodes():
        if n == "class":
            node_colors.append("red")
        elif n == "aa_000":
            node_colors.append("green")
        else:
            node_colors.append("skyblue")

    nx.draw(
        G,
        pos,
        node_color=node_colors,
        edge_color=colors,
        width=weights,
        with_labels=True,
        node_size=2500,
        font_size=10,
        arrowsize=20,
    )

    # Legend
    from matplotlib.lines import Line2D

    legend_lines = [
        Line2D([0], [0], color="black", lw=2, label="LiNGAM Edge (Strong)"),
        Line2D([0], [0], color="blue", lw=1, label="PC Edge (Rescue)"),
    ]
    plt.legend(handles=legend_lines)

    plt.title("Scania Hybrid Causal Graph (3-4 Layers)")
    plt.savefig(filename)
    print(f"Graph saved to {filename}")
    plt.close()


def visualize_hierarchy_final(G, title, filename, pymc_results=None):
    if len(G.edges) == 0:
        return

    plt.figure(figsize=(26, 18))  # Optimized Size (Bigger)
    target = "class"

    # --- FIX: Filter PyMC results to prevent leakage ---
    # Only use results matching the algorithm in the title
    current_pymc_results = None
    if pymc_results is not None:
        current_pymc_results = pymc_results.copy()
        if "Algo" in current_pymc_results.columns:
            if "PC" in title:
                current_pymc_results = current_pymc_results[
                    current_pymc_results["Algo"] == "PC"
                ]
            elif "LiNGAM" in title:
                current_pymc_results = current_pymc_results[
                    current_pymc_results["Algo"] == "LiNGAM"
                ]
    # ---------------------------------------------------

    # Depth calculation for layout
    layers = {}
    for node in G.nodes():
        if node == target:
            layers[node] = 10
        elif nx.has_path(G, node, target):
            try:
                layers[node] = 10 - len(nx.shortest_path(G, node, target))
            except:
                layers[node] = 0
        else:
            layers[node] = 0

    # Assign layers to nodes for multipartite_layout
    for node, layer in layers.items():
        G.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(G, subset_key="layer")

    # Color & Labeling logic
    node_colors = []
    labels = {}

    for n in G.nodes():
        labels[n] = n
        if n == target:
            node_colors.append("black")
            continue

        weight = 0
        # Use the filtered results
        if (
            current_pymc_results is not None
            and not current_pymc_results.empty
            and "Feature" in current_pymc_results.columns
        ):
            if n in current_pymc_results["Feature"].values:
                row = current_pymc_results[current_pymc_results["Feature"] == n]
                if "Mean" in row.columns:
                    weight = row["Mean"].values[0]
                elif "Coeff (LogOdds)" in row.columns:
                    weight = row["Coeff (LogOdds)"].values[0]

        # Color mapping based on PyMC weights
        if weight >= 0.05:
            node_colors.append("#e74c3c")  # Red (Risk +)
            labels[n] = f"{n}\n(+{weight:.2f})"
        elif weight <= -0.05:
            node_colors.append("#3498db")  # Blue (Safety -)
            labels[n] = f"{n}\n({weight:.2f})"
        else:
            node_colors.append("#95a5a6")  # Grey

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=4000, alpha=0.9)
    nx.draw_networkx_edges(
        G,
        pos,
        arrowsize=25,
        edge_color="#7f8c8d",
        width=2,
        alpha=0.7,
        node_size=4000,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_color="lightgreen", font_weight="bold", font_size=12
    )

    # Custom Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=15,
            label="Risk Driver (+Coeff)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#3498db",
            markersize=15,
            label="Safety Factor (-Coeff)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=15,
            label="Failure Event",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#95a5a6",
            markersize=15,
            label="Mediator (Uncertain Impact)",
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=12)
    plt.title(title, fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved High-Res {filename}")
    plt.close()

    from scipy.stats import kurtosis, skew


def check_gaussianity(df, features):
    print(f"{'Feature':<15} | {'Skew':<10} | {'Kurtosis':<10} | {'Verdict'}")
    print("-" * 50)
    for col in features:
        k = kurtosis(df[col])
        s = skew(df[col])
        is_gaussian = abs(k) < 1 and abs(s) < 0.5
        verdict = "Gaussian 🔴" if is_gaussian else "Non-Gaussian 🟢"
        print(f"{col:<15} | {s:.2f}       | {k:.2f}       | {verdict}")


def merge_results(output_dir, experiments, n_features):
    print("\n--- Merging Experimental Results ---")
    merged_nodes = []

    for exp_name, _, _ in experiments:
        path = f"{output_dir}/rca_node_details_{exp_name}_n={n_features}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Experiment"] = exp_name
            merged_nodes.append(df)

    if merged_nodes:
        final_df = pd.concat(merged_nodes, ignore_index=True)
        final_df.to_csv(
            f"{output_dir}/rca_experiment_comparison_nodes_n={n_features}.csv",
            index=False,
        )
        print(
            f"Saved Merged Node Results: {output_dir}/rca_experiment_comparison_nodes_n={n_features}.csv"
        )
        consolidate_results(output_dir, n_features)
    else:
        print("No results found to merge.")


def main():
    print("DEBUG: Starting main function...")
    DATA_PATH = "src/dataset/"
    TRAIN_FILE = "aps_failure_training_set.csv"
    n_features = 8
    # 1. Prep (Load Once)
    print("Loading Data...")
    # 1. Prep (Load Once)
    try:
        X, y_raw = load_data(DATA_PATH, TRAIN_FILE)
    except (FileNotFoundError, IndexError):
        # Fallback for dev environment pathing
        print("  Warning: Standard path failed. Trying relative path...")
        X, y_raw = load_data("../../" + DATA_PATH, TRAIN_FILE)

    # --- 1b. Preprocessing (Imputation & Outlier Removal) ---
    print("\n--- Preprocessing (Imputation) ---")
    # We must impute missing values before feature engineering,
    # otherwise numerical base features will have NaNs.
    preproc = ScaniaPreprocessor(reduce_missingness=True)
    X_clean = preproc.fit_transform(X)
    print("\n--- Preprocessing Done ---")

    # --- 1c. Outlier Detection (Isolation Forest) ---
    # Smart anomaly detection on clean data
    mask = fit_predict_isolation_forest(X_clean, y=y_raw, contamination=0.02)
    analyze_isolation_forest_outliers(X_clean, y_raw, mask)

    X_clean = X_clean[mask].reset_index(drop=True)
    y_clean = y_raw[mask].reset_index(drop=True)
    # Update y_raw reference for downstream compatibility if necessary,
    # but strictly we should use y_clean now.
    y_raw = y_clean
    print(f"  Data Shape after Outlier Removal: {X_clean.shape}")

    # --- 2. Feature Engineering & Selection (Consolidated & Modular) ---
    # Replaces old 'collapse_histograms' with production-grade engineering

    # A. Feature Engineering (Vectorized)
    # Note: create_engineered_feature_set handles collapsing and advanced physics features
    # (Entropy, Bimodality, Peaks, etc.) automatically.
    # We use X_clean which has imputed values.
    X_transformed = create_engineered_feature_set(X_clean, healthy_references=None)

    # B. Smart Transform (Log1p for Skewed)
    X_transformed = smart_transform_collapsed(X_transformed)

    # C. Consensus Feature Selection
    # Replaces old ad-hoc selector. Uses KS + LightGBM + Lasso (Tri-Method Consensus).
    final_features = select_features_fast_consensus(
        X_transformed, y_raw, n_features=n_features
    )

    print(
        f"  Final Feature Set for Causal Discovery ({len(final_features)}): {final_features}"
    )

    # --- 2b. Cluster Features & Enforce Hierarchy (Global/Secondary/Diagnostic) ---
    # Run on X_transformed which has Family Sums missing... so we pass X_clean (X_raw)
    # to calculate family sums correctly!
    feature_groups = cluster_features_by_correlation(
        X_transformed, final_features, X_raw=X_clean
    )
    print("\n--- Gaussianity Check (Consensus Features) ---")
    check_gaussianity(X_transformed, final_features)

    # 3. Structure Learning (Balanced)
    X_bal, y_bal = balance_for_structure_learning(X_transformed, y_raw, ratio=1.0)

    scaler = StandardScaler()
    # Prepare Continuous Data for LiNGAM and Validation
    X_cont = pd.DataFrame(
        scaler.fit_transform(X_transformed),
        columns=X_transformed.columns,
        index=X_transformed.index,
    )
    X_bal_cont, _ = balance_for_structure_learning(
        X_cont, y_raw, ratio=1.0
    )  # For LiNGAM (Continuous)
    print("\n--- Gaussianity Check (Balanced Continuous) ---")
    check_gaussianity(X_bal_cont, final_features)
    print("\n--- Gaussianity Check (Balanced) ---")
    check_gaussianity(X_bal, final_features)

    # --- EXPERIMENTAL LOOP ---
    output_dir = "results/experimental_rca_results"
    # --- 3. EXPERIMENTATION GRID ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    experiments = [
        ("EXP_FFF", False, False, False),
        ("EXP_TFF", True, False, False),
        ("EXP_TTT", True, True, True),
    ]

    all_node_results = []

    for exp_name, r_sink, r_source, r_med in experiments:
        print(f"DEBUG: Processing experiment {exp_name}")
        print(f"\n\n========================================")
        print(f"RUNNING EXPERIMENT: {exp_name}")
        print(
            f"Rules -> Class Sink: {r_sink}, Time Source: {r_source}, Mediation: {r_med}"
        )
        print(f"========================================")

        # Balance Data (Once per experiment to ensure fairness, or re-sample)
        X_bal, y_bal = balance_for_structure_learning(
            X_transformed, y_raw, ratio=1.0
        )  # Changed y_clean to y_raw

        # --- A. PC Algorithm ---
        adj_pc, labels_pc = run_pc_algorithm(
            X_bal,
            y_bal,
            final_features,
            feature_groups,  # Pass feature_groups
            rule_class_sink=r_sink,
            rule_time_source=r_source,
            rule_mediation=r_med,
        )

        # --- B. LiNGAM Algorithm ---
        adj_lingam, labels_lingam = run_lingam_algorithm(
            X_bal_cont,
            y_bal,
            final_features,
            feature_groups,  # Pass feature_groups
            rule_class_sink=r_sink,
            rule_time_source=r_source,
            rule_mediation=r_med,
        )
        # --- C. Analyze Structure ---
        print("\n--- Analyzing Graph Structure ---")
        df_pc, G_pc = classify_causal_nodes(adj_pc, labels_pc)
        df_lingam, G_lingam = classify_causal_nodes(adj_lingam, labels_lingam)

        df_pc["Algo"] = "PC"
        df_lingam["Algo"] = "LiNGAM"
        df_combined = pd.concat([df_pc, df_lingam])
        df_combined["Experiment"] = exp_name

        # Initialize PyMC columns in df_combined
        if "PyMC_Mean" not in df_combined.columns:
            df_combined["PyMC_Mean"] = np.nan
        if "PyMC_Odds_Ratio" not in df_combined.columns:
            df_combined["PyMC_Odds_Ratio"] = np.nan

        # --- D. Validation (Run for ALL experiments) ---
        screen_results_for_viz = None  # To pass to visualize_hierarchy_final
        print(f"\n--- Running Validation ({exp_name}) ---")

        # Identify candidates from LiNGAM roles
        roots = df_lingam[df_lingam["Role"] == "Physical Root"]["Feature"].tolist()
        mediators = df_lingam[df_lingam["Role"] == "Mediator"]["Feature"].tolist()
        candidates = list(set(roots + mediators))

        if r_source:
            # --- Age-Adjusted Validation (Time is Source) ---
            # 1. Rapid Screening
            screen_results = rank_features_age_adjusted(X_cont, y_raw, final_features)

            # FIX: If Mediation is False, we allow Age -> Class.
            # rank_features_age_adjusted skips aa_000. We must add it back if it's a feature.
            if not r_med and "aa_000" in final_features:
                try:
                    # Calculate univariate impact of aa_000
                    age_data = X_cont["aa_000"].values
                    X_age = sm.add_constant(age_data)
                    model_age = sm.Logit(y_raw, X_age)
                    res_age = model_age.fit(disp=0)

                    coef_age = res_age.params.iloc[1]
                    p_age = res_age.pvalues.iloc[1]
                    conf_age = res_age.conf_int().iloc[1]

                    row_age = {
                        "Feature": "aa_000",
                        "Coeff (LogOdds)": coef_age,
                        "Odds_Ratio": np.exp(coef_age),
                        "P_Value": p_age,
                        "Lower_CI": conf_age[0],
                        "Upper_CI": conf_age[1],
                        "Age_Conf_Factor": 0.0,  # It is Age itself
                    }
                    # Append to screen_results
                    if screen_results is None:
                        screen_results = pd.DataFrame([row_age])
                    else:
                        screen_results = pd.concat(
                            [screen_results, pd.DataFrame([row_age])], ignore_index=True
                        )

                    # Re-sort
                    screen_results = screen_results.sort_values(
                        by="Coeff (LogOdds)", key=abs, ascending=False
                    )

                except Exception as e:
                    print(f"Warning: Could not calculate aa_000 coefficient: {e}")

            screen_results_for_viz = screen_results  # Use for visualization

            # Filter screening results for candidates
            if candidates and screen_results is not None and not screen_results.empty:
                screen_subset = screen_results[
                    screen_results["Feature"].isin(candidates)
                ].head(5)

                # 2. Deep Dive PyMC on top candidates
                top_candidates = screen_subset["Feature"].tolist()[:3]
                if top_candidates:
                    print(
                        f"Running Deep Dive PyMC on Top 3 Candidates: {top_candidates}"
                    )
                    pymc_deep_dive_df = run_pymc_age_adjusted(
                        X_cont,
                        y_raw,
                        top_candidates,
                        "Top Candidates",
                        control_feat="aa_000",
                    )

                    if pymc_deep_dive_df is not None and not pymc_deep_dive_df.empty:
                        for feat in top_candidates:
                            row = pymc_deep_dive_df[
                                pymc_deep_dive_df["Feature"] == feat
                            ]
                            if not row.empty:
                                res = row.iloc[0]
                                mask = (df_combined["Feature"] == feat) & (
                                    df_combined["Experiment"] == exp_name
                                )
                                df_combined.loc[mask, "PyMC_Mean"] = res["Mean"]
                                df_combined.loc[mask, "PyMC_Odds_Ratio"] = res[
                                    "Odds_Ratio"
                                ]
        else:
            # --- Default Validation (Standard PyMC) ---
            if candidates:
                # Run standard PyMC on candidates
                summary, valid_feats = run_pymc_validation(
                    X_cont, y_raw, candidates, "Candidates"
                )

                if summary is not None and not summary.empty:
                    # Map summary rows to features
                    # Summary index is beta[0], beta[1]... corresponding to valid_feats
                    pymc_results_list = []
                    for i, feat in enumerate(valid_feats):
                        try:
                            mean_val = summary.iloc[i]["mean"]
                            pymc_results_list.append(
                                {"Feature": feat, "Mean": mean_val}
                            )

                            # Update df_combined
                            mask = (df_combined["Feature"] == feat) & (
                                df_combined["Experiment"] == exp_name
                            )
                            df_combined.loc[mask, "PyMC_Mean"] = mean_val
                            df_combined.loc[mask, "PyMC_Odds_Ratio"] = np.exp(mean_val)
                        except IndexError:
                            pass

                    # Create DataFrame for visualization
                    screen_results_for_viz = pd.DataFrame(pymc_results_list)

        # Visualize (always, but screen_results_for_viz only if validation ran)
        visualize_hierarchy_final(
            G_lingam,
            f"LiNGAM - {exp_name}",
            f"{output_dir}/rca_lingam_{exp_name}_n={n_features}.png",
            screen_results_for_viz,
        )
        visualize_hierarchy_final(
            G_pc,
            f"PC - {exp_name}",
            f"{output_dir}/rca_pc_{exp_name}_n={n_features}.png",
            screen_results_for_viz,
        )

        # --- HYBRID GRAPH (Ensemble) ---
        # Run for ALL experiments as requested
        G_hybrid = create_hybrid_graph(
            adj_lingam, adj_pc, labels_lingam, labels_pc, rule_time_source=r_source
        )  # Pass rule_time_source
        analyze_hybrid_layers(G_hybrid)
        visualize_hybrid(
            G_hybrid, f"{output_dir}/rca_hybrid_{exp_name}_n={n_features}.png"
        )

        # Export Hybrid Edges
        hybrid_edges = []
        for u, v, d in G_hybrid.edges(data=True):
            hybrid_edges.append(
                {
                    "Algo": "Hybrid",
                    "Source": u,
                    "Target": v,
                    "Type": d.get("type", "Unknown"),
                }
            )
        pd.DataFrame(hybrid_edges).to_csv(
            f"{output_dir}/rca_edges_hybrid_{exp_name}_n={n_features}.csv", index=False
        )

        # Export Edges (per experiment)
        edges_data = []
        for u, v in G_pc.edges():
            edges_data.append({"Algo": "PC", "Source": u, "Target": v})
        for u, v in G_lingam.edges():
            edges_data.append({"Algo": "LiNGAM", "Source": u, "Target": v})
        pd.DataFrame(edges_data).to_csv(
            f"{output_dir}/rca_edges_{exp_name}_n={n_features}.csv", index=False
        )

        df_combined.to_csv(
            f"{output_dir}/rca_node_details_{exp_name}_n={n_features}.csv", index=False
        )
        all_node_results.append(df_combined)

    consolidate_results(output_dir, n_features)


def consolidate_results(output_dir, n_features):
    print("\n--- Consolidating Experimental Results ---")

    # 1. Consolidate Node Details
    node_files = [
        f
        for f in os.listdir(output_dir)
        if f.startswith("rca_node_details_") and f.endswith(f"_n={n_features}.csv")
    ]
    merged_nodes = []
    for f in node_files:
        try:
            df = pd.read_csv(os.path.join(output_dir, f))
            # Extract experiment name if not present (assuming filename format rca_node_details_{EXP}_n={N}.csv)
            if "Experiment" not in df.columns:
                exp_name = f.replace("rca_node_details_", "").replace(
                    f"_n={n_features}.csv", ""
                )
                df["Experiment"] = exp_name
            merged_nodes.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if merged_nodes:
        final_nodes_df = pd.concat(merged_nodes, ignore_index=True)
        node_out_path = f"{output_dir}/rca_all_nodes_summary_n={n_features}.csv"
        final_nodes_df.to_csv(node_out_path, index=False)
        print(f"Saved Consolidated Node Results: {node_out_path}")
    else:
        print("No node results found to merge.")

    # 2. Consolidate Edges (Standard & Hybrid)
    edge_files = [
        f
        for f in os.listdir(output_dir)
        if f.startswith("rca_edges_") and f.endswith(f"_n={n_features}.csv")
    ]
    merged_edges = []
    for f in edge_files:
        try:
            df = pd.read_csv(os.path.join(output_dir, f))
            # Extract experiment name
            # Format: rca_edges_{EXP}_n={N}.csv or rca_edges_hybrid_{EXP}_n={N}.csv
            base_name = f.replace(f"_n={n_features}.csv", "")
            if "hybrid" in base_name:
                exp_name = base_name.replace("rca_edges_hybrid_", "")
                df["Graph_Type"] = "Hybrid"
            else:
                exp_name = base_name.replace("rca_edges_", "")
                df["Graph_Type"] = "Standard"

            if "Experiment" not in df.columns:
                df["Experiment"] = exp_name
            merged_edges.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if merged_edges:
        final_edges_df = pd.concat(merged_edges, ignore_index=True)
        edge_out_path = f"{output_dir}/rca_all_edges_summary_n={n_features}.csv"
        final_edges_df.to_csv(edge_out_path, index=False)
        print(f"Saved Consolidated Edge Results: {edge_out_path}")
    else:
        print("No edge results found to merge.")


if __name__ == "__main__":
    print("DEBUG: Script execution started.")
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()
