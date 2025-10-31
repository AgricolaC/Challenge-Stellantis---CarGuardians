from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd

def split_histogram_features(X):
    """Separate 7×10 histogram families from numerical ones."""
    hist_families = ['ag', 'ay', 'ba', 'cn', 'cs', 'ee', 'cn']
    hist_cols = [c for c in X.columns if any(c.startswith(f) for f in hist_families)]
    num_cols = [c for c in X.columns if c not in hist_cols]
    return X[hist_cols], X[num_cols], hist_cols, num_cols

def select_top_features(X, y, n_features=15):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    selector = RFE(rf, n_features_to_select=n_features, step=0.1)
    selector.fit(X, y)
    selected_cols = X.columns[selector.support_]
    print(f"Selected top {n_features} features: {list(selected_cols)}")
    return X[selected_cols]

