import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def remove_na(df: pd.DataFrame, nan_count: dict):
    """
    Removes columns with >drop_thresh missing data, 
    and rows with missing data in <4% missing columns.
    Keeps 'class' column intact until after cleaning.
    """
    
    constant = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant, errors="ignore")
    print(f"Removed constant columns: {constant} ")    
    print(f"Current Shape: {df.shape} ")    
    
    # Columns with >70% missing
    na_70 = [k for k, v in nan_count.items() if v > 70 and k in df.columns]
    df = df.drop(columns=na_70, errors="ignore")
    print(f"Removed columns with >%70 Missing Values: {na_70}")
    print(f"Current Shape: {df.shape} ")    

    na_4 = [k for k, v in nan_count.items() if v < 4 and k in df.columns]
    print(f"Removed missing rows of Columns <%4 Missing Values: {na_4}")
    # Drop rows that have NA in those low-missing columns
    df = df.dropna(subset=na_4)
    print(f"Current Shape: {df.shape} ")  
    
    features_to_remove = list(set(na_70 + constant))
          
    return df, features_to_remove


def imputation(df, nan_count):
    """
    Median imputation for 5–15%, MICE for 15–70%.
    Returns full imputed DataFrame and both imputers for reuse on test.
    """
    # Median features (5–15%)
    median_feats = [k for k, v in nan_count.items() if 4 <= v < 15 and k in df.columns]
    median_imputer = SimpleImputer(strategy='median')
    df[median_feats] = median_imputer.fit_transform(df[median_feats])

    # Model-based imputation (15–70%)
    ridge_pipe = Pipeline([
    ("scale", StandardScaler(with_mean=True, with_std=True)),
    ("ridge", Ridge(alpha=10.0, random_state=0))  # try alpha=10–100
])
    mice_imputer = IterativeImputer(
        estimator=ridge_pipe,
        max_iter=10,
        random_state=0,
        n_nearest_features=50,     
        imputation_order="ascending",
        verbose=1
    )
    df_imputed = pd.DataFrame(mice_imputer.fit_transform(df), columns=df.columns)

    print(f"Median-imputed features: {len(median_feats)} | "
          f"MICE features: {sum((15 <= v < 70) for v in nan_count.values())}")
    return df_imputed, median_imputer, mice_imputer, median_feats

def preprocess_test_data(x_test, removed_features, median_feats, median_imputer, mice_imputer):
    x_test = x_test.copy()
    y_test = x_test["class"].map({"neg": 0, "pos": 1}).astype("int8")
    x_test = x_test.drop(columns=["class"], errors="ignore")
    x_test = x_test.drop(columns=removed_features, errors="ignore")
    x_test[median_feats] = median_imputer.transform(x_test[median_feats])
    x_test = pd.DataFrame(mice_imputer.transform(x_test), columns=x_test.columns)
    print(f"X_test Shape: {x_test.shape}")
    return x_test, y_test