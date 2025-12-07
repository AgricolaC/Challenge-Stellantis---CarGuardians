import pandas as pd
import numpy as np
import warnings
import sys
import os

# Causal Imports
try:
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
except ImportError:
    print("CRITICAL: causal-learn not found. pip install causal-learn")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path if needed, but if running from src, it should be fine.
# The user provided script adds '..', let's keep it but also ensure current dir is handled.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from challenge.data.ingest import load_data
    from challenge.data.preprocess import ScaniaPreprocessor
except ImportError:
    # Fallback if running from src and challenge is a subdir
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.challenge.data.ingest import load_data
    from src.challenge.data.preprocess import ScaniaPreprocessor

# --- RE-USE YOUR COLLAPSE LOGIC ---
def collapse_histograms(df):
    print("--- Collapsing Histograms ---")
    families = ['ag', 'ay', 'az', 'ba', 'cn', 'cs', 'ee']
    keep_cols = [c for c in df.columns if not any(c.startswith(f + '_') and c[-1].isdigit() for f in families)]
    df_new = df[keep_cols].copy()
    
    for fam in families:
        bins = sorted([c for c in df.columns if c.startswith(fam + '_') and c[-1].isdigit()])
        if not bins: continue
        
        data = df[bins].values
        total_counts = data.sum(axis=1)
        df_new[f'{fam}_sum'] = total_counts
        
        mask = total_counts > 0
        probs = np.zeros_like(data, dtype=float)
        probs[mask] = data[mask] / total_counts[mask, None]
        
        bin_indices = np.arange(len(bins))
        mu = np.sum(probs * bin_indices, axis=1)
        df_new[f'{fam}_mean'] = mu
        
        mu2 = np.sum(probs * (bin_indices**2), axis=1)
        var = mu2 - (mu**2)
        var[var < 0] = 0
        df_new[f'{fam}_var'] = var
        
    return df_new

def smart_transform_collapsed(df_collapsed):
    print("--- Applying Smart Transformation ---")
    df_trans = df_collapsed.copy()
    for col in df_trans.columns:
        if col.endswith('_sum') or col.endswith('_var'):
            if df_trans[col].min() < 0: df_trans[col] = df_trans[col].clip(lower=0)
            df_trans[col] = np.log1p(df_trans[col])
    return df_trans

def balance_data(X, y):
    print("--- Balancing Data (1:1) ---")
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    np.random.seed(42)
    neg_sampled = np.random.choice(neg_idx, len(pos_idx), replace=False)
    indices = np.concatenate([pos_idx, neg_sampled])
    np.random.shuffle(indices)
    return X.iloc[indices].copy(), y.iloc[indices].copy()

# --- THE DEBUG CONFIGURATION ---
def configure_debug_priors(labels, enforce_mediation=False):
    n = len(labels)
    # priors[i, j] = -1 means j -> i is FORBIDDEN
    # Row (i) = Effect, Col (j) = Cause
    priors = np.zeros((n, n))
    target_col = 'class'
    time_col = 'aa_000'
    
    try:
        t_idx = labels.index(target_col)
        time_idx = labels.index(time_col)
        
        print(f"\n[PRIOR CONFIG] Class Idx: {t_idx}, Time Idx: {time_idx}")
        
        # 1. Class is Sink (Target) -> Cannot be Cause (Column)
        priors[:, t_idx] = -1 
        print(f"  Constraint: Class cannot be Cause (Col {t_idx} is -1)")
        
        # 2. Time is Source -> Cannot be Effect (Row)
        priors[time_idx, :] = -1
        print(f"  Constraint: Time cannot be Effect (Row {time_idx} is -1)")
        
        # 3. MEDIATION RULE
        if enforce_mediation:
            # Forbidden: Cause=Time(Col), Effect=Class(Row)
            priors[t_idx, time_idx] = -1
            print(f"  Constraint: Time -> Class is FORBIDDEN (priors[{t_idx}, {time_idx}] = -1)")
        else:
            print(f"  Constraint: Time -> Class is ALLOWED (priors[{t_idx}, {time_idx}] = 0)")
            
    except ValueError:
        print("Error: Columns not found")
        pass
        
    return priors

def run_debug():
    DATA_PATH = 'dataset/'
    TRAIN_FILE = 'aps_failure_training_set.csv'
    
    # Load & Prep
    try:
        X_raw, y_raw = load_data(DATA_PATH, TRAIN_FILE)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Try adjusting path if running from src
        DATA_PATH = '../dataset/'
        X_raw, y_raw = load_data(DATA_PATH, TRAIN_FILE)

    preproc = ScaniaPreprocessor()
    X_clean = preproc.fit_transform(X_raw)
    
    X_coll = collapse_histograms(X_clean)
    X_trans = smart_transform_collapsed(X_coll)
    
    # Select Top Features + Time
    # (Hardcoded top features for debugging to ensure we have signal)
    debug_features = [
        'aa_000', 'ag_mean', 'ag_var', 'ay_mean', 'ay_var', 
        'cn_mean', 'cn_var', 'cs_mean', 'cs_var', 'bk_000'
    ]
    
    # Ensure they exist
    feat_subset = [f for f in debug_features if f in X_trans.columns]
    
    # Balance
    X_bal, y_bal = balance_data(X_trans[feat_subset], y_raw)
    
    # Prep for LiNGAM
    df = X_bal.copy()
    df['class'] = y_bal.values
    labels = df.columns.tolist()
    
    # --- RUN 1: MEDIATION DISABLED (Sanity Check) ---
    print("\n\n" + "="*50)
    print("RUN 1: Mediation Rule DISABLED (Should see Time->Class)")
    print("="*50)
    
    priors = configure_debug_priors(labels, enforce_mediation=False)
    model = DirectLiNGAM(prior_knowledge=priors)
    model.fit(df)
    
    # Analyze Matrix
    # LiNGAM Adjacency B: B[i, j] implies x_j -> x_i (Col causes Row)
    # We want standard format: Row causes Col
    # So we take Transpose. adj[i, j] means i -> j
    adj = model.adjacency_matrix_.T
    
    print("\n[RAW EDGE WEIGHTS] (Abs value > 0.001)")
    print(f"{'Source':<15} -> {'Target':<15} | {'Weight':<10}")
    print("-" * 45)
    
    rows, cols = np.where(np.abs(adj) > 0.001)
    
    # Sort by weight strength
    edges = []
    for r, c in zip(rows, cols):
        edges.append((labels[r], labels[c], abs(adj[r, c])))
    
    edges.sort(key=lambda x: x[2], reverse=True)
    
    for src, tgt, w in edges:
        mark = ""
        if src == 'aa_000' and tgt == 'class': mark = " <--- DIRECT TIME LINK"
        if tgt == 'class': mark += " (To Class)"
        print(f"{src:<15} -> {tgt:<15} | {w:.4f}{mark}")
        
    # --- RUN 2: MEDIATION ENABLED (The Real Test) ---
    print("\n\n" + "="*50)
    print("RUN 2: Mediation Rule ENABLED (Should see Time->Comp->Class)")
    print("="*50)
    
    priors = configure_debug_priors(labels, enforce_mediation=True)
    model = DirectLiNGAM(prior_knowledge=priors)
    model.fit(df)
    
    adj = model.adjacency_matrix_.T
    
    print("\n[RAW EDGE WEIGHTS] (Abs value > 0.001)")
    print(f"{'Source':<15} -> {'Target':<15} | {'Weight':<10}")
    print("-" * 45)
    
    rows, cols = np.where(np.abs(adj) > 0.001)
    
    edges = []
    for r, c in zip(rows, cols):
        edges.append((labels[r], labels[c], abs(adj[r, c])))
    
    edges.sort(key=lambda x: x[2], reverse=True)
    
    for src, tgt, w in edges:
        mark = ""
        if src == 'aa_000' and tgt == 'class': mark = " <--- VIOLATION?"
        if tgt == 'class': mark += " (To Class)"
        print(f"{src:<15} -> {tgt:<15} | {w:.4f}{mark}")

if __name__ == "__main__":
    run_debug()
