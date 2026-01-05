from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


class KruskalSelector(BaseEstimator, TransformerMixin):
    """
    Selects top features using the Kruskal-Wallis H-test.
    Robust to outliers unlike the standard ANOVA F-test (f_classif).
    """

    def __init__(self, top_n=50):
        self.top_n = top_n
        self.selected_columns_ = None
        self.p_values_ = None

    def fit(self, X, y):
        """
        Computes the Kruskal-Wallis H-statistic for each feature.
        Only sees the provided X and y (Training fold).
        """
        # X can be DataFrame or numpy array. We prefer DataFrame for column names.
        if not isinstance(X, pd.DataFrame):
            # Try to recover if X was a DataFrame before (e.g. via column preservation wrapper)
            # or just use indices if we must.
            X = pd.DataFrame(X)

        p_values = []
        feature_names = X.columns.tolist()

        # Ensure y is proper format
        y = np.array(y).flatten()

        for col in feature_names:
            try:
                # Group 0 vs Group 1 for this feature
                col_data = X[col]

                # Separation
                group0 = col_data[y == 0]
                group1 = col_data[y == 1]

                if len(group0) == 0 or len(group1) == 0:
                    p_values.append(1.0)
                    continue

                # KW Test
                stat, p = kruskal(group0, group1)
                p_values.append(p)
            except ValueError:
                # Handle constant columns or errors (e.g. all NaNs)
                p_values.append(1.0)

        self.p_values_ = np.array(p_values)

        # Select indices of lowest p-values
        best_indices = np.argsort(self.p_values_)[: self.top_n]
        self.selected_columns_ = [feature_names[i] for i in best_indices]

        return self

    def transform(self, X):
        """
        Reduces X to the selected features.
        """
        if self.selected_columns_ is None:
            raise RuntimeError("You must fit the selector before transforming.")

        if isinstance(X, pd.DataFrame):
            # Check if columns exist
            missing_cols = [c for c in self.selected_columns_ if c not in X.columns]
            if missing_cols:
                # If X lost columns (e.g. numpy conversion upstream), we might be in trouble.
                # But our pipeline ensures DataFrame preservation.
                raise ValueError(f"Selected columns not found in X: {missing_cols}")
            return X[self.selected_columns_]
        else:
            # Fallback if X is numpy array (column indices)
            # We need to map selected_columns_ names back to indices of the original X...
            # This is hard if we don't know the original order.
            # Assuming X matches the fit structure
            raise NotImplementedError(
                "KruskalSelector supports pandas DataFrame input only to ensure correct feature mapping."
            )


class ConsensusSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=100, consensus_thresh=2, n_jobs=-1, random_state=42):
        self.top_n = top_n
        self.consensus_thresh = consensus_thresh
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.selected_indices_ = None
        self.feature_names_in_ = None

    def fit(self, X, y):
        # Handle DataFrame vs Numpy
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_data = X.values
        else:
            X_data = X
            self.feature_names_in_ = [f"feat_{i}" for i in range(X.shape[1])]

        y = np.array(y).flatten()
        n_features = X_data.shape[1]

        # --- VOTER 1: Kruskal-Wallis (Statistical) ---
        kw_p_values = []
        for i in range(n_features):
            try:
                # Compare distributions of Class 0 vs Class 1
                g0 = X_data[y == 0, i]
                g1 = X_data[y == 1, i]
                stat, p = kruskal(g0, g1)
                kw_p_values.append(p)
            except ValueError:
                kw_p_values.append(1.0)

        # Lowest p-value is best
        kw_indices = np.argsort(kw_p_values)[: self.top_n]

        # --- VOTER 2: Mutual Information (Information Theory) ---
        # SPEED OPTIMIZATION: Downsample if data is huge (>5k rows)
        if X_data.shape[0] > 10000:
            idx = np.random.RandomState(self.random_state).choice(
                X_data.shape[0], 10000, replace=False
            )
            X_mi = X_data[idx]
            y_mi = y[idx]
        else:
            X_mi, y_mi = X_data, y

        # Fill NaNs with 0 for MI calculation only (KNN doesn't like NaNs)
        X_mi_filled = np.nan_to_num(X_mi, nan=0.0)

        mi_scores = mutual_info_classif(
            X_mi_filled,
            y_mi,
            discrete_features="auto",
            random_state=self.random_state,
            n_neighbors=3,
        )
        mi_indices = np.argsort(mi_scores)[-self.top_n :]

        # --- VOTER 3: LightGBM (Interactions) ---
        lgbm = lgb.LGBMClassifier(
            n_estimators=100,
            n_jobs=self.n_jobs,
            verbose=-1,
            random_state=self.random_state,
        )
        lgbm.fit(X_data, y)
        lgb_indices = np.argsort(lgbm.feature_importances_)[-self.top_n :]

        # --- AGGREGATION ---
        votes = np.zeros(n_features, dtype=int)
        votes[kw_indices] += 1
        votes[mi_indices] += 1
        votes[lgb_indices] += 1

        # Selection Logic
        selected_mask = votes >= self.consensus_thresh
        self.selected_indices_ = np.where(selected_mask)[0]

        # Fallback: If consensus is too strict, pick top N by vote count + LGBM score tiebreak
        if len(self.selected_indices_) == 0:
            print("Consensus strict. Fallback to ranked voting.")
            # Create a score: Votes * 1000 + Normalized LGBM Importance
            # This ensures features with equal votes are broken by model importance
            norm_imp = lgbm.feature_importances_ / (
                lgbm.feature_importances_.max() + 1e-9
            )
            hybrid_score = votes + norm_imp
            self.selected_indices_ = np.argsort(hybrid_score)[-self.top_n :]

        print(
            f"ConsensusSelector: Selected {len(self.selected_indices_)} features from {n_features} total."
        )

        return self

    def transform(self, X):
        if self.selected_indices_ is None:
            raise RuntimeError("ConsensusSelector not fitted.")

        # Handle DataFrame vs Numpy
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices_]
        return X[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_in_ is None:
            return [f"x{i}" for i in self.selected_indices_]
        return [self.feature_names_in_[i] for i in self.selected_indices_]
