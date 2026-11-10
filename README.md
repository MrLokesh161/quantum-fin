# QETI: Local Financial Instability Research

QETI is a reproducible local research prototype for studying market instability with public financial data, classical machine learning, persistent-homology features, and a simulator-based variational quantum classifier. It does not provide investment advice, a calibrated real-world crash probability, or evidence of quantum advantage. The VQC runs on a classical statevector simulator; no quantum hardware is used.

## Free local stack

- Python 3.10+
- pandas, NumPy, scikit-learn, Ripser, SQLite
- yfinance for Yahoo Finance data
- Binance public market-data endpoints for BTC/ETH daily candles
- FRED public graph CSV endpoint for Treasury rates and macroeconomic series
- Streamlit and Plotly for the local dashboard
- NumPy statevector simulation for the experimental variational quantum classifier (VQC)
- Existing React/FastAPI research interface remains available

No paid data APIs, API keys, cloud services, paid hosting, or quantum hardware are required. Provider access and terms can change; sources may be delayed, rate-limited, or unavailable.

## Setup

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
```

The setup script installs dependencies into `env` and installs the existing frontend packages. Build the React frontend once:

```powershell
cd frontend
npm.cmd run build
cd ..
```

Start the backend, which serves both the API and the built frontend:

```powershell
.\env\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Startup does not download data or train models. Use **Refresh data + retrain** in the React interface when you want to run those processes; the backend runs them locally and reports their status in the UI.

The on-demand refresh downloads daily Yahoo Finance OHLCV for `^GSPC`, `^IXIC`, `^DJI`, `^NSEI`, and `^VIX`; Binance BTCUSDT and ETHUSDT candles; and FRED DGS10, DGS2, CPIAUCSL, UNRATE, and FEDFUNDS. The end date defaults to tomorrow and is exclusive.

Provider errors and rate limits are retried and recorded in `data/raw/source_status.json`, `outputs/reports/data_source_status.csv`, and SQLite. Successful downloads are cached under `data/raw/` and persisted in `data/qeti.sqlite3`. Failed downloads use existing SQLite observations for the requested date range when available; the status is marked `cached`. Missing observations are never fabricated. The project keeps tracked datasets untouched during ingestion.

### Source and data limitations

- Yahoo Finance access through yfinance is unofficial and data is delayed/end-of-day, not real-time. Availability, rate limits, and terms may change.
- Binance data is from public daily UTC klines for supported symbols, without API keys.
- FRED's public CSV endpoint requires no API key for the configured downloads. FRED publication frequency varies by series; values are aligned backward and forward-filled only after their observation date. Historical revisions mean the downloaded series is not a release-vintage dataset and may contain revised values.
- Raw downloads, source attribution, frequency, observed date range, fetch status, and limitations are saved in SQLite and the generated metadata/report files.

## Crisis event labels

`data/crisis_events.csv` records the event interval, anchor date, description, source references, and event type/relevance. Sources include Federal Reserve History, the FRED S&P 500 series, SEC press releases, and the FDIC.

The S&P 500 future-drawdown model is compared with sourced broad-market stress periods including the dot-com bust, 2008 financial crisis, 2011 debt-ceiling stress, 2018 fourth-quarter selloff, 2020 COVID-19 crash, and 2022 bear market. Terra-Luna and FTX are crypto stress events; SVB is an institution-specific event. They remain documented for review but are not scored as equivalent broad-market crashes. Event counts are too small for statistically reliable crisis-generalization claims.

## Processing and models

The pipeline:

1. Selects the configured S&P 500 CSV (preferring a locally refreshed `^GSPC` file), validates and sorts dates, removes duplicate timestamps, and forward-fills only previously observed values.
2. Builds return, volatility, drawdown, and technical-indicator features. If the SQLite market store is present, it backward-aligns market, crypto, and FRED observations and derives cross-market returns/correlations.
3. Defines the binary target as a decline of at least 5% within the next 10 trading sessions. Rows without a completed future label are excluded from evaluation.
4. Creates sliding windows and Ripser persistence summaries (Betti counts, persistence lifetimes, variance, and entropy). The topology point cloud uses a reduced market-state feature set and scales each historical window locally.
5. Splits data chronologically into train, validation, and test sets, with a configurable 30-row purge gap between them to reduce overlapping-window and forward-label leakage.
6. Fits feature scaling on training rows only. Chronological validation is divided into probability calibration and threshold-tuning partitions with a purge gap. Temperature scaling is evaluated on held-out data; it does not guarantee reliable real-world probabilities.
7. Compares conventional financial models, topological-only and financial-plus-topological feature sets, and a compact 2-qubit/1-layer variational quantum classifier (VQC). The VQC uses PCA-reduced inputs, angle encoding, a trainable RY/RZ ansatz with entangling CNOT gates, and SPSA optimization on a local NumPy statevector simulator. Model comparisons report precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, false-alarm rate, and runtime.
8. Runs expanding-window walk-forward validation with a purge gap and fold-local calibration, and uses out-of-fold Random Forest probabilities for the crisis-event backtest.
9. Trains a separate five-class next-session downside model using the next session's low-to-current-close drawdown: Stable (> -0.5%), Low (<= -0.5% and > -1.0%), Moderate (<= -1.0% and > -1.5%), High (<= -1.5% and > -2.5%), or Critical (<= -2.5%). This is not a next-day closing-price or return forecast.
10. Saves historical test predictions and current estimates from the latest completed S&P 500 daily bar. The binary Random Forest continues to estimate the chance of a 5% drawdown within 10 trading sessions; the separate next-session classifier estimates the downside category.

Experiments, models, predictions, confusion matrices, feature plots, data source status, and the JSON/Markdown report are generated under `outputs/`. You can configure date inputs, features, train/validation proportions, TDA stride, and model settings in `config.yaml`.

## Risk indicators

For the model's estimated risk score `P_risk`:

```text
SRS  = 100 × P_risk
QTSI = 100 × (1 − P_risk)
```

Therefore `SRS + QTSI = 100`. Risk bands use `config.yaml` and the displayed classification is Low, Moderate, High, or Critical. QTSI is a project-defined indicator, not an industry-standard index. Calibration metrics describe held-out historical behavior; they do not prove calibrated real-world crisis probabilities.

## Dashboard

The React/FastAPI market risk dashboard is the primary interface. It shows the 10-session drawdown estimate separately from the five-class next-session downside estimate, market context, daily source availability, historical test predictions, model comparisons, walk-forward metrics, and event backtests. Use **Refresh data + retrain** in the UI to manually run the public-data refresh and model pipeline; starting the backend does not trigger those processes.

The application uses daily public market data, not a real-time intraday feed. Yahoo Finance data can be delayed/end-of-day. Calibrated scores and risk classes are research estimates, not trading signals or guaranteed real-world probabilities. The VQC is simulated in software on a classical CPU; it does not use a quantum computer or establish quantum advantage.

Start the React/FastAPI dashboard with the backend command in Setup. If you use the helper script, it only starts the backend and does not refresh data, train models, or build the frontend:

```powershell
.\run_project.ps1
```

The optional Streamlit research interface remains available:

```powershell
.\run_dashboard.ps1
```

## Tests and limitations

Run the project tests with:

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Research results depend on data coverage, source revisions, label definitions, the limited number of crisis periods, and model calibration. The VQC is a small simulator-based research baseline, not evidence of quantum advantage. This work does not establish that TDA or VQC improves out-of-sample performance; compare the saved held-out test metrics before drawing conclusions.
