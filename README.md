# 🚛 Scania APS Failure Prediction (CarGuardians)

> **A robust, cost-sensitive, and physics-aware machine learning pipeline for predictive maintenance on heavy trucks.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Executive Summary

The Air Pressure System (APS) generates pressurized air utilized in various functions in a truck, such as braking and gear changes. This project implements an industrial-grade pipeline to predict APS failures.

**The Business Constraint:**
This is not a standard classification problem. We optimize for **Total Cost**, not Accuracy.

* **False Positive (Type I):** Unnecessary check. Cost: **$10**
* **False Negative (Type II):** Missed failure (Breakdown). Cost: **$500**

Our solution minimizes this financial risk through **Custom Objective Functions**, **Synthetic Data Generation**, **Physics-Informed Features**, and **Bayesian Root Cause Analysis**.

---

## ⚡ Quick Start (Interactive Mode)

We have created a unified entry point for professionals to easily review the project, manage dependencies, and execute pipelines.

1.  **Clone the repository.**
2.  **Run the main driver:**

```bash
python main.py
```

This interactive menu will allow you to:

* ✅ **Auto-configure the Environment:** Checks and installs `requirements.txt`.
* 📊 **Run Master Notebooks:** Launches the Jupyter analysis.
* 🕵️ **Perform Forensics:** Generates "Fan of Death" and "Hardware Fingerprint" plots.
* 🔍 **Run Causal Discovery:** Executes the experimental PC/LiNGAM algorithms.
* 📉 **Execute Prediction Pipeline:** Runs the cost-sensitive training and evaluation.

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
│   ├── models.py             # Defining used models
│   ├── selectors.py          # Consensus Voting (Mutual Info + KS Test + LGBM)
│   ├── train_eval.py         # CV & Evaluation
│   ├── experiment.py         # Grid search orchestration
│   └── threshold_tuning.py   # Dynamic cost-based threshold optimization
├── rca/
│   ├── pipeline.py           # Standard RCA (SHAP, Surrogate Trees)
│   └── explanation_shap.py   # Global & Local SHAP analysis
├── experimental_rca/         # NEW: Advanced Causal Discovery
│   └── experimental_rca_pipeline.py # PC & LiNGAM algorithms with Physics Constraints
└── visualization/
    ├── forensics.py          # "Fan of Death" and Physics verification plots
    └── performance_viz.py    # Cost curves and ROC plots
...
```

---

## 🚀 Key Innovations

### 1. Physics-Aware Feature Engineering

We inferred the meaning of anonymized columns (e.g., `aa_000` as Odometer, and certain histogram families indicating similarity with Engine Load). We treated histogram families as **Probability Mass Functions**. We extracted:

* **Center of Mass (Mean pressure/RPM)**
* **Vectorized Entropy (System Disorder)**
* **Skewness/Kurtosis (Tail risks)**
This allowed us to achieve strong predictive performance using only **~10% of the original features** (20-25 out of 170).

### 2. "Failure Rescue" Outlier Detection

Standard anomaly detection often removes minority class samples because they look "weird."

* **Our Logic:** Failures *are* anomalies.
* **Implementation:** We predict outliers but forcefully **retain all positive samples (which are already 1:59 minority)** (`y=1`), cleaning only the noisy majority class.

### 3. Missingness Clustering

We discovered that missing data was not random but systemic (e.g., a specific ECU goes offline). We clustered correlated `_is_missing` flags to create new features representing **"Module Health,"** turning data gaps into diagnostic signals.

### 4. Generative Balancing (Gaussian Copula)

Beyond standard SMOTE, we utilize **Gaussian Copulas** (via SDV) to learn the multivariate distribution of the failure class and generate physically valid synthetic failures with enforced constraints (e.g., non-negative pressures).

### 5. Experimental Causal Discovery (New)

Moving beyond correlation, we implemented **Causal Inference** using **PC** (Constraint-based) and **LiNGAM** (Functional-based) algorithms.

* **Physics Constraints:** We enforced "Hard Rules" (e.g., Time is a Source, Failure is a Sink) to generate Directed Acyclic Graphs (DAGs).
* **Outcome:** A 4-layer causal hierarchy distinguishing between *Root Causes* and *Mediators*.

### 6. Cost-Driven Optimization

We implemented a **Custom Weighted Logistic Loss** for LightGBM. Furthermore, during inference, we scan the ROC curve to find the specific probability threshold that minimizes the business cost function:


---

## 💻 Manual Installation

If you prefer not to use `main.py`, you can set up the environment manually:

```bash
# 1. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Install Project in Editable Mode
pip install -e src/

```

### Running the Analysis

The core analysis is orchestrated via the master notebook:

```bash
jupyter notebook src/04_master_analysis.ipynb

```

**Execution Flow:**

1. **Data Ingestion:** Loads `aps_failure_training_set.csv`.
2. **Preprocessing:** Applies ScaniaPreprocessor (MICE + Clustering).
3. **Experiment Grid:** Runs cross-validation across Feature Sets (Raw vs. Physics) and Samplers.
4. **Evaluation:** Reports Cost, AUC, and Macro-F1.
5. **RCA:** Triggers the Causal Discovery pipeline.

---

## 📊 Performance Results

Based on our extensive grid search, the top-performing configuration was:

| Metric | Score | Configuration |
| --- | --- | --- |
| **Total Cost** | **$9,500** | Best on Test Set |
| **ROC-AUC** | 0.9958 | Excellent Separation |
| **Macro F1** | 0.8129 | Balanced Precision/Recall |

**Winning Strategy:**

* **Model:** LightGBM (Weighted Cost Objective)
* **Features:** Clean Raw Features (Missingness Clustered)
* **Selection:** Consensus Selector (Union of KS-Test, Mutual Info, LightGBM Importance)
* **Sampling:** Gaussian Copula (0.25 ratio)
* **Threshold:** Dynamically Tuned

*Note: While Physics-based features provided strong signals and explainability, the Clean Raw Features generalized slightly better to the test set by capturing the "long tail" of sensor data.*

---

## 📄 License

<<<<<<< HEAD
see the LICENSE file for details.

---
=======
This project is licensed under the MIT License - see the LICENSE file for details.
>>>>>>> 39bd718 (feat: Add `main.py` entry point, restructure test and analysis modules, and introduce forensic verification.)
