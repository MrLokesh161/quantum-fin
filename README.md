# Quantum-Enhanced Topological Intelligence — Phase 1

This repository implements the classical, reproducible baseline for early financial-market collapse detection. It loads every CSV in `datasets/`, merges indices on `Date`, produces technical indicators, creates future-drawdown risk labels, then trains and evaluates a leakage-safe chronological Random Forest baseline.

## Run

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
```

Tune the target market and research assumptions in `config.py`. By default, an S&P 500 decline of at least 5% within the next 10 trading days is labelled `1` (high risk). The model uses 20-day windows, while fitting the scaler only on the chronological training partition.

## Outputs

`outputs/` receives cleaned and engineered data, IQR diagnostics, model artefact, metrics, predictions, and the progress report. `visualizations/` receives EDA, ROC, confusion-matrix, and feature-importance graphics.

## Look up risk for a date

After training, query any trading date in the chronological test period:

```powershell
python check_risk_by_date.py 2018-11-06
```

The command displays the saved high-risk probability, the model classification at the 0.50 threshold, and the evaluation-only ground-truth label. It reports the valid test-date range if the requested date is unavailable.

## Phase 2 interfaces

`models/future_extensions.py` provides deliberately stable placeholders for Persistent Homology, topological feature extraction (Betti numbers, persistence entropy, landscapes, Euler characteristics), and a Qiskit/PennyLane Variational Quantum Classifier. Each accepts outputs from the existing window pipeline, so preprocessing need not change.
