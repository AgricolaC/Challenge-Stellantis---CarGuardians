# Challenge@Stellantis – Minimal Modular Repository

This repository hosts the **Challenge@Stellantis** project for explainable diagnostics and novelty detection in automotive systems.  
The goal is to build a **clean, modular, and reproducible codebase** where all team members can ingest, preprocess, visualize, and analyze the dataset collaboratively.

---

## 🎯 Goal
Ingest → Preprocess (optional if using provided processed files) → Visualize → Test  
All within a minimal, Pythonic structure designed for easy iteration and CI/CD integration.

---

## Quickstart

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests to confirm setup
pytest -q

# (Optional) Explore the notebooks
jupyter notebook notebooks/
```

## Data

The dataset consists of TDMS and Excel files describing engine combustion cycles and derived measurements.

Data folders are divided to ensure version control hygiene:

```bash
data/
├─ raw/              # Original raw files (TDMS, raw spreadsheets) — LFS tracked
├─ processed/        # Provided processed files from the dataset — LFS tracked
└─ house_processed/  # In-house processed or generated artifacts — small, ≤10 MB
```

Notes:
- Large files (e.g. .tdms, .xlsx, .pdf, .docx) are tracked with Git LFS.
- Use data/house_processed/ for any derived results created by your scripts or notebooks.
- The repository avoids committing heavy files directly.

## Modules
The code lives under /src/challenge/, designed to be installable and importable as a Python package:
| Module            | Purpose                                                                                                                          |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **`ingest/`**     | Readers for TDMS and Excel data. Handles loading of sensor signals, injector data, and averages.                                 |
| **`preprocess/`** | Optional preprocessing pipeline (cleaning, normalization, or reformatting). Can be skipped if using the provided processed data. |
| **`novelty/`**    | Novelty-detection algorithms such as PCA- and Mahalanobis-based indices.                                                         |
| **`visualize/`**  | Quick plotting utilities for EDA and reporting (pressure traces, knock signals, feature scatterplots, etc.).                     |
| **`utils/`**      | Shared helpers (paths, environment variables, and constants).                                                                    |
                                                              |
Testing lives under /src/tests/, with lightweight placeholders ready for student expansion.

## How To Initialize
```bash
# Clone repository
git clone <your_repo_url>
cd challenge-stellantis

# Initialize Git LFS (needed once per machine)
git lfs install

# Pull large data files
git lfs pull

# Set up environment and dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify installation and run tests
pytest -q
```

## Workflow Overview

1. Ingest: Use ingest/ loaders to read TDMS or Excel data into DataFrames or NumPy arrays.

2. Preprocess (optional): Apply cleaning or feature extraction if you generate in-house processed datasets.

3. Visualize: Explore patterns, anomalies, and cycle variability in notebooks/01_eda.ipynb.

4. Novelty Detection: Implement and test PCA or Mahalanobis-based indices to identify deviations.

5. Test: Add or expand tests in /src/tests/ to maintain reliability and CI consistency.


## Continuous Integration

Each push or pull request triggers GitHub Actions (CI) which:

- Installs dependencies

- Runs tests automatically (pytest)

- Verifies that all code executes and imports cleanly

- You’ll see a ✅ or ❌ badge under your PRs, keeping the main branch stable.

## Conventions

- Use Python ≥3.10

- Keep notebooks lightweight; core logic belongs in /src/

- Add clear docstrings and small, modular commits
