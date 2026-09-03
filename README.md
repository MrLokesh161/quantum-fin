# QETI: Classical Financial Instability Research

QETI is a research prototype for early market-instability warning. It is not a reliable collapse predictor, investment system, or claim of quantum computing. The current production pipeline is classical and uses financial indicators, optional persistent-homology features, chronological model evaluation, and derived risk scores.

## Requirements

- Windows PowerShell, macOS, or Linux
- Python 3.10 or newer
- Node.js 18 or newer and npm
- Internet access when refreshing market data with `dataset.py`

The commands below use the Windows virtual environment path. On macOS/Linux, replace `env\\Scripts\\python.exe` with `env/bin/python` and `npm.cmd` with `npm`.

## Complete setup

From the repository root:

```powershell
py -3 -m venv env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\\env\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the frontend dependencies:

```powershell
Push-Location frontend
npm.cmd install
Pop-Location
```

Run the lightweight checks:

```powershell
python -m py_compile dataset.py main.py src\\*.py
Push-Location frontend
npm.cmd run build
Pop-Location
```

## Refresh daily market data

`dataset.py` downloads the configured market indices from Yahoo Finance. It starts at 1990-01-01 and uses tomorrow as the exclusive end date, so every run refreshes through the latest available completed trading session. A market day that has not closed yet will not be present.

```powershell
env\\Scripts\\python.exe dataset.py
```

The default primary dataset is `datasets/sp500.csv`. The data downloader also refreshes NASDAQ, Dow Jones, NIFTY 50, and FTSE 100 files. The event file under `data/crisis_events.csv` is evaluation metadata and is not used as a model feature.

## Train and regenerate outputs

Run the main research pipeline after every data refresh:

```powershell
env\\Scripts\\python.exe main.py
```

The pipeline:

1. Loads the S&P 500 CSV and cleans the data.
2. Builds technical indicators and future-drawdown crash labels.
3. Builds sliding windows and optional classical TDA features.
4. Splits data chronologically into 70% training, 15% validation, and 15% testing.
5. Trains Logistic Regression, Random Forest, SVM, Gradient Boosting, and XGBoost when available.
6. Evaluates on the held-out test period.
7. Uses Random Forest with the `Financial + TDA` feature set for saved dashboard predictions.
8. Writes models, predictions, metrics, figures, and experiment reports under `outputs/`.

The crash label is `1` when the market falls at least 5% within the next 10 trading days. The final 10 available trading rows cannot receive a confirmed label until those future prices exist. They are therefore excluded from evaluation, although they can be used by a future live-prediction workflow.

## Start the application

Build the frontend, then start the FastAPI server:

```powershell
Push-Location frontend
npm.cmd run build
Pop-Location
env\\Scripts\\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The backend serves `frontend/dist` and reads only generated artifacts; it does not train models on request.

For frontend development, keep the API running in one terminal and use another:

```powershell
Push-Location frontend
npm.cmd run dev
Pop-Location
```

The Vite development server normally runs at `http://localhost:5173`.

## Useful API endpoints

| Endpoint | Purpose |
| --- | --- |
| `/api/health` | Check API and artifact availability |
| `/api/summary` | Latest prediction and experiment metadata |
| `/api/predictions` | Saved chronological test predictions |
| `/api/risk/YYYY-MM-DD` | Retrieve one saved test-date prediction |
| `/api/model-comparison` | Actual held-out metrics for trained models |
| `/api/early-warning` | Historical event backtest results |

## Model interpretation

The saved comparison is based on real fitted models and chronological held-out test data. The currently deployed dashboard prediction is **Random Forest + Financial/TDA features**.

- **ARIMA:** not implemented.
- **LSTM:** optional helper only; it is not called by `main.py` and is not in the current comparison.
- **XGBoost:** trained when the package is available.
- **TDA:** classical features added to the feature set, not a standalone classifier.
- **QTSI:** a classical 0-100 derived stability score calculated after prediction, not a quantum model.

Do not select a final model from accuracy alone. Review recall, F1, PR-AUC, ROC-AUC, false-alarm rate, and the class balance in `outputs/metrics/model_comparison.csv`.

## Risk scores

The dashboard uses these probability bands from `config.yaml`:

| Probability | Label |
| --- | --- |
| `< 25%` | Stable |
| `25% - 50%` | Caution |
| `50% - 75%` | High Risk |
| `>= 75%` | Critical |

Crash probability is the Random Forest estimate for the 5% future-drawdown target. Systemic risk is a relative 0-100 instability score combining probability, volatility, drawdown, and TDA signals. QTSI is a relative 0-100 stability score; higher values indicate greater estimated stability. Neither score is calibrated as a real-world failure probability.

## Date lookup and current data

`check_risk_by_date.py` queries dates in the saved test predictions:

```powershell
env\\Scripts\\python.exe check_risk_by_date.py 2025-10-03
```

Raw prices can reach the latest completed market day, but confirmed labels and independent test predictions end 10 trading days earlier. This is expected behavior for a forward-looking target. The dashboard calendar is limited to dates with saved test predictions.

## Generated files and version control

Generated model artifacts, figures, reports, caches, virtual environments, frontend dependencies, and frontend builds are ignored by `.gitignore`. Source code, configuration, event metadata, and the tracked dataset CSVs remain visible to version control. Recreate ignored experiment outputs by running `main.py`.

## Project structure

```text
backend/             FastAPI application
data/                Event metadata
datasets/             Market CSV inputs
frontend/             React + Vite dashboard
models/              Model helpers and future extension interfaces
outputs/              Generated models, metrics, reports, and predictions
src/                 Main research pipeline modules
tests/               Core tests, when pytest is installed
visualizations/      Generated plots from the legacy pipeline
```
