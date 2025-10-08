# Challenge@Stellantis – Minimal Repo

**Goal:** Ingest data → preprocess → visualize → test with the simplest setup.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import challenge; print('ok')"
```

## Data

- Put raw files in data/raw/ (tracked via Git LFS).

- Save small processed outputs to data/processed/.

## Modules

### src/challenge/io.py : TDMS/Excel readers

### src/challenge/preprocess.py : cleaning & basic features

### src/challenge/viz.py : quick plots

## src/challenge/test_preprocess.py : tests after cleaning

## How to Initialize
```bash
git init
git lfs install
git add .
git commit -m "chore: minimal scaffold"
# git remote add origin <URL> && git push -u origin main
```
