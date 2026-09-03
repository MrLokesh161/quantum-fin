"""FastAPI backend. It never trains models: run main.py to create artifacts first."""
from pathlib import Path
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PREDICTIONS = OUTPUTS / "predictions" / "predictions.csv"
METRICS = OUTPUTS / "metrics" / "model_comparison.csv"
EVENTS = OUTPUTS / "reports" / "early_warning_performance.csv"
REPORT = OUTPUTS / "reports" / "experiment_report.json"
FRONTEND_DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="QETI Classical Research API", version="1.0")

def _csv(path: Path, dates=()):
    if not path.exists():
        raise HTTPException(404, "Experiment artifact missing. Run `python main.py` first.")
    return pd.read_csv(path, parse_dates=list(dates))

@app.get("/api/health")
def health(): return {"status": "ok", "trained_artifacts_available": PREDICTIONS.exists()}

@app.get("/api/summary")
def summary():
    if not REPORT.exists(): raise HTTPException(404, "Experiment report missing. Run `python main.py` first.")
    predictions = _csv(PREDICTIONS, ["Date"])
    latest = predictions.iloc[-1].to_dict()
    latest["Date"] = str(latest["Date"].date())
    return {"experiment": json.loads(REPORT.read_text(encoding="utf-8")), "latest": latest}

@app.get("/api/model-comparison")
def model_comparison():
    """Only reports models that were actually trained by main.py."""
    return _csv(METRICS).to_dict(orient="records")

@app.get("/api/predictions")
def predictions(start: str | None = None, end: str | None = None):
    data = _csv(PREDICTIONS, ["Date"])
    if start: data = data[data.Date >= pd.Timestamp(start)]
    if end: data = data[data.Date <= pd.Timestamp(end)]
    data["Date"] = data.Date.dt.strftime("%Y-%m-%d")
    return data.to_dict(orient="records")

@app.get("/api/risk/{date}")
def risk_for_date(date: str):
    data = _csv(PREDICTIONS, ["Date"]); selected = data[data.Date == pd.Timestamp(date)]
    if selected.empty: raise HTTPException(404, "No trained test prediction exists for this date.")
    result = selected.iloc[0].to_dict(); result["Date"] = str(result["Date"].date()); return result

@app.get("/api/early-warning")
def early_warning():
    return _csv(EVENTS, ["warning_date", "crisis_date"]).fillna("N/A").astype(str).to_dict(orient="records")

@app.get("/")
def frontend():
    if (FRONTEND_DIST / "index.html").exists(): return FileResponse(FRONTEND_DIST / "index.html")
    return JSONResponse({"message":"React frontend is not built. Run npm.cmd install and npm.cmd run build in frontend/."},status_code=503)

if FRONTEND_DIST.exists(): app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
