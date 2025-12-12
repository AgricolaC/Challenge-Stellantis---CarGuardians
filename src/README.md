# 🚛 Scania APS Failure Prediction (CarGuardians)

> **A robust, cost-sensitive, and physics-aware machine learning pipeline for predictive maintenance on heavy trucks.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Executive Summary

The Air Pressure System (APS) generates pressurized air utilized in various functions in a truck, such as braking and gear changes. This project implements an industrial-grade pipeline to predict APS failures.

**The Business Constraint:**
This is not a standard classification problem. We optimize for **Total Cost**, not Accuracy.

*   **False Positive (Type I):** Unnecessary check. Cost: **$10**
*   **False Negative (Type II):** Missed failure (Breakdown). Cost: **$500**

Our solution minimizes this financial risk through **Custom Objective Functions**, **Synthetic Data Generation**, **Physics-Informed Features**, and **Bayesian Root Cause Analysis**.

---

## 🏗️ Architecture

The project is structured as a modular Python package (`challenge`) ensuring reproducibility and scalability.

```text
src/challenge/
├── data/
│   ├── ingest.py             # Loading & schema validation
│   ├── preprocess.py         # MICE Imputation & Hidden Missingness Clustering
│   ├── outliers.py           # IsolationForest with "Failure Rescue" logic
│   ├── feature_selection.py  # Physics-based feature extraction (Moments, Entropy)
│   └── balancing.py          # SMOTEENN & Gaussian Copula synthesis
├── modelling/
│   ├── models.py             # LightGBM with custom Weighted Logistic Loss
│   ├── selectors.py          # Consensus Voting (Mutual Info + KS Test + LGBM)
│   ├── train_eval.py         # Cost-sensitive CV & Evaluation
│   ├── experiment.py         # Grid search orchestration
│   └── threshold_tuning.py   # Dynamic cost-based threshold optimization
├── rca/
│   ├── pipeline.py           # Full Root Cause Analysis orchestration
│   ├── surrogate.py          # Surrogate Decision Trees & Rule Extraction
│   ├── explanation_shap.py   # Global & Local SHAP analysis
│   ├── comparison.py         # Decision Matrix & Radar Charts
│   ├── importance.py         # Permutation Importance
│   ├── reference_model.py    # Baseline models for RCA
│   └── pdp_ice.py            # Partial Dependence Plots
├── visualization/
│   ├── eda_plots.py          # Exploratory Data Analysis
│   ├── feature_correlation.py# Correlation analysis
│   └── performance_viz.py    # Cost curves and ROC plots
└── analysis/
    └── pca_analysis.py       # Dimensionality reduction analysis
```

---

## 🚀 Key Innovations

### 1. Physics-Aware Feature Engineering
Instead of treating sensor histograms (e.g., `ag_000` to `ag_009`) as independent columns, we treat them as **Probability Mass Functions**. We extract statistical moments and physics descriptors:
*   **Center of Mass (Mean pressure)**
*   **Variance (Stability)**
*   **Skewness/Kurtosis (Tail risks)**
*   **Vectorized Entropy (System Disorder)**
*   **Bimodality Coefficients & Peak Counts**

This reduces dimensionality while preserving physical signals that indicate component degradation.

### 2. "Failure Rescue" Outlier Detection
Standard anomaly detection often removes minority class samples because they look "weird."
*   **Our Logic:** Failures *are* anomalies.
*   **Implementation:** We predict outliers but forcefully **retain all positive samples** (`y=1`), cleaning only the noisy majority class.

### 3. Missingness Clustering
Missing data in this dataset is systemic (e.g., a specific ECU goes offline). We cluster binary missingness flags (`_is_missing`) to create new features representing **"Module Health,"** capturing non-random sensor failures.

### 4. Generative Balancing (Gaussian Copula)
Beyond standard SMOTE, we utilize **Gaussian Copulas** (via SDV) to learn the multivariate distribution of the failure class and generate physically valid synthetic failures with enforced constraints (e.g., non-negative pressures).

### 5. Automated Root Cause Analysis (RCA)
We go beyond prediction to explanation. Our RCA pipeline (`challenge.rca`) automatically:
*   Computes **SHAP** values for local explanations.
*   Trains **Surrogate Decision Trees** to extract human-readable failure rules.
*   Generates **Decision Matrices** and Radar Charts to compare multiple failure modes.

### 6. Cost-Driven Optimization
We implemented a **Custom Weighted Logistic Loss** for LightGBM. Furthermore, during inference, we scan the ROC curve to find the specific probability threshold that minimizes the business cost function:
$$Cost = (FP \times 10) + (FN \times 500)$$

---

## 💻 Installation & Usage

### Prerequisites
*   Python 3.10+
*   Git LFS (for large datasets)

### Setup
```bash
# 1. Clone the repo
git clone https://github.com/your-org/carguardians.git
cd carguardians

# 2. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Install Project in Editable Mode
pip install -e src/
```

### Running the Pipeline
The core analysis is orchestrated via the master notebook:

1.  **Open the Notebook:**
    ```bash
    jupyter notebook src/04_master_analysis.ipynb
    ```
2.  **Execution Flow:**
    *   **Data Ingestion:** Loads `aps_failure_training_set.csv`.
    *   **Preprocessing:** Applies ScaniaPreprocessor (MICE + Clustering).
    *   **Experiment Grid:** Runs cross-validation across Feature Sets (Raw vs. Physics) and Samplers.
    *   **Evaluation:** Reports Cost, AUC, and Macro-F1.
    *   **RCA:** Triggers the RCA pipeline for the best models.

---

## 📊 Performance Results

Based on our extensive grid search (see `04_master_analysis.ipynb`), the top-performing configuration was:

| Metric | Score | Configuration |
| :--- | :--- | :--- |
| **Total Cost** | **$9,500** | Best on Test Set |
| **ROC-AUC** | 0.9958 | Excellent Separation |
| **Macro F1** | 0.8129 | Balanced Precision/Recall |

**Winning Strategy:**
*   **Model:** LightGBM (Weighted Cost Objective)
*   **Features:** Clean Raw Features (Missingness Clustered)
*   **Selection:** Consensus Selector (Union of KS-Test, Mutual Info, LightGBM Importance)
*   **Sampling:** Gaussian Copula (0.25 ratio)
*   **Threshold:** Dynamically Tuned

*Note: While Physics-based features provided strong signals, the Clean Raw Features generalized better to the test set, capturing the "long tail" of sensor data.*

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
