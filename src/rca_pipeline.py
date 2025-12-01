import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
import networkx as nx
import pymc as pm
import arviz as az

# Statistical Imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import shap
from scipy.stats import ks_2samp
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Causal Imports
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import chisq
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    from causallearn.graph.GraphNode import GraphNode
except ImportError:
    print("CRITICAL: causal-learn not found. pip install causal-learn")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from challenge.data.ingest import load_data
from challenge.data.feature_selection import select_features_ks

# --- 1. Data Engineering (Robust) ---
def load_and_prep_data():
    print("--- [1] Data Engineering ---")
    X_raw, y_raw = load_data('dataset/', 'aps_failure_training_set.csv')
    
    # 1. Row Cleaning (Low NA)
    na_pct = X_raw.isna().mean()
    low_na_cols = na_pct[(na_pct > 0) & (na_pct <= 0.04)].index
    if not low_na_cols.empty:
        rows_to_drop = X_raw[low_na_cols].isna().any(axis=1)
        indices_to_drop = X_raw[rows_to_drop].index
        X_raw = X_raw.drop(index=indices_to_drop)
        y_raw = y_raw.drop(index=indices_to_drop)
        
    # 2. Drop Outlier
    if 20683 in X_raw.index:
        X_raw = X_raw.drop(index=20683)
        y_raw = y_raw.drop(index=20683)
    
    # 3. Robust Imputation (Missingness Flags)
    X_clean = X_raw.copy()
    missing_cols = [c for c in X_clean.columns if X_clean[c].isnull().mean() >= 0.05]
    for col in missing_cols:
        X_clean[f'{col}_is_missing'] = X_clean[col].isnull().astype(int)
        
    imputer = SimpleImputer(strategy='median')
    X_clean_imp = pd.DataFrame(imputer.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index)
    
    # 4. Remove constant columns
    X_clean_imp = X_clean_imp.loc[:, X_clean_imp.nunique() > 1]
    
    return X_clean_imp, y_raw


# --- 2. Feature Selection (Expanded n=14) ---

def get_shap_importance(X, y):
    model = LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X.sample(min(1000, len(X)), random_state=42))
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]
    return pd.DataFrame({'feat': X.columns, 'shap': np.abs(shap_vals).mean(axis=0)})

def get_lasso_features(X, y, top_n):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, class_weight='balanced', random_state=42)
    lasso.fit(X_scaled, y)
    return pd.Series(np.abs(lasso.coef_[0]), index=X.columns).sort_values(ascending=False).head(top_n).index.tolist()

def get_consensus_features(X, y, n_features=14):
    print(f"\n--- [2] Consensus Feature Selection (Top {n_features} per method) ---")
    
    # 1. Run Selection Methods
    ks_feats = select_features_ks(X, y, top_n_by_stat=n_features, p_value_threshold=0.05)
    shap_df = get_shap_importance(X, y)
    shap_feats = shap_df.sort_values('shap', ascending=False).head(n_features)['feat'].tolist()
    lasso_feats = get_lasso_features(X, y, top_n=n_features)
    
    feature_pool = list(set(ks_feats) | set(shap_feats) | set(lasso_feats))
    print(f"Pooled Candidates: {len(feature_pool)}")
    
    # 2. Aggressive Clustering (The Filter)
    # We use Spearman distance. Features with corr > 0.5 (dist < 0.5) will likely cluster.
    corr = X[feature_pool].corr(method='spearman').abs()
    dist = 1 - corr.fillna(0)
    dist = np.clip(dist, 0, None)
    
    linkage = hierarchy.linkage(squareform(dist), method='ward')
    clusters = hierarchy.fcluster(linkage, t=0.5, criterion='distance')
    
    final_features = []
    for cid in np.unique(clusters):
        cluster_members = np.array(feature_pool)[clusters == cid]
        # Pick the one with highest SHAP score as the "Representative"
        best = shap_df[shap_df['feat'].isin(cluster_members)].sort_values('shap', ascending=False).iloc[0]['feat']
        final_features.append(best)
        
    print(f"Final Representatives after Clustering: {len(final_features)}")
    return final_features, ks_feats, shap_feats, lasso_feats

# --- 3. Causal Engines ---

def run_pc_algorithm(X, y, features):
    print("\n--- [Engine A] PC Algorithm ---")
    est = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
    df = X[features].copy()
    cont_cols = [c for c in df.columns if df[c].nunique() > 5]
    if cont_cols:
        df[cont_cols] = est.fit_transform(df[cont_cols]).astype(int)
    
    df['class'] = y.values
    data = df.values.astype(int)
    labels = df.columns.tolist()
    
    bk = BackgroundKnowledge()
    class_idx = len(labels) - 1
    # GraphNode names must match what causal-learn expects (X1..Xn)
    nodes = [GraphNode(f"X{i+1}") for i in range(len(labels))]
    
    for i in range(class_idx):
        bk.add_forbidden_by_node(nodes[class_idx], nodes[i])
        
    cg = pc(data, 0.05, chisq, True, 0, -1, background_knowledge=bk)
    
    adj = np.zeros((len(labels), len(labels)))
    for edge in cg.G.get_graph_edges():
        n1 = int(edge.node1.name.replace('X', '')) - 1
        n2 = int(edge.node2.name.replace('X', '')) - 1
        if '-->' in str(edge): adj[n1, n2] = 1
        elif '<--' in str(edge): adj[n2, n1] = 1
        
    return adj, labels

def run_lingam_algorithm(X, y, features):
    print("\n--- [Engine B] DirectLiNGAM ---")
    df = X[features].copy()
    df['class'] = y.values
    labels = df.columns.tolist()
    class_idx = len(labels) - 1
    
    model = DirectLiNGAM()
    model.fit(df)
    
    # Transpose for i->j adjacency
    adj = model.adjacency_matrix_.T 
    clean_adj = np.zeros_like(adj)
    
    # Optimized Thresholding
    for i in range(len(labels)):
        for j in range(len(labels)):
            w = abs(adj[i, j])
            # Strict for Sensor->Sensor, Sensitive for Sensor->Class
            if j == class_idx or i == class_idx:
                if w > 0.001: clean_adj[i, j] = 1
            else:
                if w > 0.05: clean_adj[i, j] = 1
                
    return clean_adj, labels

# --- 4. Visualization (Optimized) ---

def analyze_graph_structure(adj, labels, target_node='class', force_target_sink=False):
    G = nx.DiGraph()
    for i, label in enumerate(labels): G.add_node(label)
    rows, cols = np.where(adj > 0)
    for r, c in zip(rows, cols): 
        source = labels[r]
        target = labels[c]
        
        # --- LOGIC FIX: Force Class to be the Sink ---
        if force_target_sink and source == target_node:
            # Logic: Failure cannot cause Sensor. Sensor must cause Failure.
            # Action: Flip direction (Source <-> Target)
            G.add_edge(target, source)
            print(f"  [Physics Constraint] Flipped edge: {target} -> {source}")
        else:
            G.add_edge(source, target)
        
    if target_node not in G: return [], [], [], G
    
    parents = list(G.predecessors(target_node))
    if not parents: return [], [], [], G
    
    ancestors = list(nx.ancestors(G, target_node))
    roots = [n for n in ancestors if G.in_degree(n) == 0 and n != target_node]
    mediators = [n for n in ancestors if G.in_degree(n) > 0 and n != target_node]
    
    return roots, mediators, parents, G

def visualize_hierarchy_final(G, title, filename, pymc_results=None):
    if len(G.edges) == 0: return

    plt.figure(figsize=(26, 18)) # Optimized Size (Bigger)
    target = 'class'
    
    # Depth calculation for layout
    layers = {}
    for node in G.nodes():
        if node == target: layers[node] = 10
        elif nx.has_path(G, node, target):
            try: layers[node] = 10 - len(nx.shortest_path(G, node, target))
            except: layers[node] = 0
        else: layers[node] = 0
            
    # Assign layers to nodes for multipartite_layout
    for node, layer in layers.items():
        G.nodes[node]['layer'] = layer
            
    pos = nx.multipartite_layout(G, subset_key='layer')
    
    # Color & Labeling logic
    node_colors = []
    labels = {}
    
    for n in G.nodes():
        labels[n] = n
        if n == target:
            node_colors.append('black')
            continue
            
        weight = 0
        if pymc_results is not None and n in pymc_results['Feature'].values:
            weight = pymc_results[pymc_results['Feature'] == n]['Mean'].values[0]
            
        # Color mapping based on PyMC weights
        if weight >= 0.05: 
            node_colors.append('#e74c3c') # Red (Risk +)
            labels[n] = f"{n}\n(+{weight:.2f})"
        elif weight <= -0.05: 
            node_colors.append('#3498db') # Blue (Safety -)
            labels[n] = f"{n}\n({weight:.2f})"
        else: 
            node_colors.append('#95a5a6') # Grey

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=4000, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrowsize=25, edge_color='#7f8c8d', width=2, alpha=0.7, 
                           node_size=4000, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, labels=labels, font_color='white', font_weight='bold', font_size=12)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=15, label='Risk Driver (+Coeff)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=15, label='Safety Factor (-Coeff)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=15, label='Failure Event'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6', markersize=15, label='Mediator (Uncertain Impact)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.title(title, fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved High-Res {filename}")
    plt.close()

# --- 5. Bayesian Validation (Expanded Output) ---

def run_pymc(X, y, features, name):
    if not features: return None, []
    print(f"Running PyMC for {name} on {features}...")
    
    # Collinearity Guardrail
    df_feat = X[features]
    corr = df_feat.corr().abs()
    drop = set()
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            f1, f2 = features[i], features[j]
            if f1 in drop or f2 in drop: continue
            if corr.loc[f1, f2] > 0.80: # Strict 0.80 threshold
                c1 = abs(df_feat[f1].corr(y))
                c2 = abs(df_feat[f2].corr(y))
                drop.add(f1 if c1 < c2 else f2)
    
    final_feats = [f for f in features if f not in drop]
    if len(final_feats) < len(features):
        print(f"  Pruned {len(drop)} features. Analyzing: {final_feats}")

    # Model
    X_data = (X[final_feats] - X[final_feats].mean()) / (X[final_feats].std() + 1e-6)
    with pm.Model() as model:
        #beta = pm.Normal("beta", mu=0, sigma=0.25, shape=len(final_feats))
        beta = pm.StudentT("beta", nu=4, mu=0, sigma=0.5, shape=len(final_feats))
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        mu = alpha + pm.math.dot(X_data.values, beta)
        theta = pm.math.sigmoid(mu)
        y_obs = pm.Bernoulli("y_obs", p=theta, observed=y.values)
        
        # MAP init for stability
        # start = pm.find_MAP(progressbar=False)
        trace = pm.sample(1000, tune=1000, target_accept=0.96, progressbar=True, cores=1)
        
    return az.summary(trace, var_names=['beta']), final_feats

# --- Main ---

def main():
    # 1. Prep
    X_clean, y_raw = load_and_prep_data()
    # X_clean is already imputed and cleaned by load_and_prep_data
    
    # 2. Feature Selection 
    n_features = 24
    final_features, ks_f, shap_f, lasso_f = get_consensus_features(X_clean, y_raw, n_features=n_features)
    
    # --- Skewness Correction (Log Transform) ---
    print("\n--- Checking for Skewness ---")
    # Always transform cn_001 if present
    skewed_candidates = set(final_features)
    if 'cn_001' in X_clean.columns: skewed_candidates.add('cn_001')
    
    for col in skewed_candidates:
        if col not in X_clean.columns: continue
        
        # Check skewness
        skew_val = X_clean[col].skew()
        max_val = X_clean[col].max()
        
        # Transform if explicitly requested (cn_001) or highly skewed
        if col == 'cn_001' or (abs(skew_val) > 2 and max_val > 100):
            print(f"  Applying Log1p to {col} (Skew: {skew_val:.2f}, Max: {max_val})")
            X_clean[col] = np.log1p(X_clean[col])

    # 3. Fork
    scaler = StandardScaler()
    X_cont = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index)
    # Discrete done inside PC function now
    
    # 4. Engines
    adj_pc, lbl_pc = run_pc_algorithm(X_clean, y_raw, final_features)
    adj_lin, lbl_lin = run_lingam_algorithm(X_cont, y_raw, final_features)
    
    # 5. Analyze Structure
    print("\n--- Analyzing PC Structure ---")
    roots_pc, meds_pc, pars_pc, G_pc = analyze_graph_structure(adj_pc, lbl_pc, force_target_sink=False)
    
    print("\n--- Analyzing LiNGAM Structure (With Physics Fix) ---")
    roots_lin, meds_lin, pars_lin, G_lin = analyze_graph_structure(adj_lin, lbl_lin, force_target_sink=True)
    
    # 6. Bayesian Stats
    results = []
    
    # LiNGAM (Primary)
    summ, feats = run_pymc(X_cont, y_raw, roots_lin, "LiNGAM Roots")
    if summ is not None:
        for i, f in enumerate(feats):
            mean_val = summ.iloc[i]['mean']
            results.append({'Algo': 'LiNGAM', 'Role': 'Root', 'Feature': f, 
                            'Mean': mean_val, 'Odds_Ratio': np.exp(mean_val)})
    
    # PC (Comparison)
    summ, feats = run_pymc(X_cont, y_raw, roots_pc, "PC Roots")
    if summ is not None:
        for i, f in enumerate(feats):
            mean_val = summ.iloc[i]['mean']
            results.append({'Algo': 'PC', 'Role': 'Root', 'Feature': f, 
                            'Mean': mean_val, 'Odds_Ratio': np.exp(mean_val)})

    # 7. Final Visuals & Save
    df_res = pd.DataFrame(results)
    
    # Create output directory
    output_dir = "rca_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Pass DataFrame for coloring
    visualize_hierarchy_final(G_lin, "LiNGAM Causal Hierarchy (Final)", f"{output_dir}/rca_lingam_optimized_n_features={n_features}.png", df_res)
    visualize_hierarchy_final(G_pc, "PC Causal Hierarchy (Final)", f"{output_dir}/rca_pc_optimized_n_features={n_features}.png", df_res)
    
    # Export Graph Structure (Nodes & Edges)
    print("\n--- Exporting Graph Data ---")
    
    # 1. Edges
    edges_data = []
    for u, v in G_pc.edges(): edges_data.append({'Algo': 'PC', 'Source': u, 'Target': v})
    for u, v in G_lin.edges(): edges_data.append({'Algo': 'LiNGAM', 'Source': u, 'Target': v})
    pd.DataFrame(edges_data).to_csv(f"{output_dir}/rca_edges_n_features={n_features}.csv", index=False)
    print(f"Saved {output_dir}/rca_edges_n_features={n_features}.csv")
    
    # 2. Nodes (with Roles & Stats)
    nodes_data = []
    
    def process_nodes(G, algo, roots, meds, pars, pymc_df):
        for n in G.nodes():
            role = 'Unconnected'
            if n == 'class': role = 'Target'
            elif n in roots: role = 'Root'
            elif n in meds: role = 'Mediator'
            elif n in pars: role = 'Direct Parent' # Note: Parents can also be Roots or Mediators, strict hierarchy might vary
            
            # Refine Role Priority
            if n == 'class': role = 'Target'
            elif n in roots: role = 'Root'
            elif n in meds: role = 'Mediator'
            
            stats = {}
            if not pymc_df.empty:
                row = pymc_df[(pymc_df['Algo'] == algo) & (pymc_df['Feature'] == n)]
                if not row.empty:
                    stats = row.iloc[0].to_dict()
            
            nodes_data.append({
                'Algo': algo,
                'Node': n,
                'Role': role,
                'PyMC_Mean': stats.get('Mean', np.nan),
                'PyMC_Odds_Ratio': stats.get('Odds_Ratio', np.nan)
            })

    process_nodes(G_pc, 'PC', roots_pc, meds_pc, pars_pc, df_res)
    process_nodes(G_lin, 'LiNGAM', roots_lin, meds_lin, pars_lin, df_res)
    
    pd.DataFrame(nodes_data).to_csv(f"{output_dir}/rca_node_details_n_features={n_features}.csv", index=False)
    print(f"Saved {output_dir}/rca_node_details_n_features={n_features}.csv")

    df_res.to_csv(f"{output_dir}/rca_final_optimized_results_n_features={n_features}.csv", index=False)
    print("\n--- OPTIMIZED RCA RESULTS ---")
    if not df_res.empty:
        print(df_res.sort_values('Odds_Ratio', ascending=False))
    else:
        print("No significant results found.")

if __name__ == "__main__":
    main()
