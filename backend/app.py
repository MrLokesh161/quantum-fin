"""FastAPI backend for the local QETI application."""
from datetime import date
from pathlib import Path
import json
import subprocess
import sys
from threading import Lock, Thread

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PREDICTIONS = OUTPUTS / "predictions" / "predictions.csv"
CURRENT_PREDICTION = OUTPUTS / "predictions" / "current_prediction.csv"
METRICS = OUTPUTS / "metrics" / "model_comparison.csv"
EVENTS = OUTPUTS / "reports" / "early_warning_performance.csv"
REPORT = OUTPUTS / "reports" / "experiment_report.json"
NEXT_SESSION_METRICS = OUTPUTS / "reports" / "next_session_risk_metrics.json"
WALK_FORWARD_SUMMARY = OUTPUTS / "reports" / "walk_forward_summary.json"
NEXT_SESSION_PREDICTIONS = OUTPUTS / "predictions" / "next_session_risk_predictions.csv"
DATA_STATUS = OUTPUTS / "reports" / "data_source_status.csv"
EVENT_LABELS = ROOT / "data" / "crisis_events.csv"
DATABASE = ROOT / "data" / "qeti.sqlite3"
FRONTEND_DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="QETI Classical Research API", version="1.0")


class RefreshRequest(BaseModel):
    start: date = date(1990, 1, 1)


_job_lock = Lock()
_job_status = {
    "state": "idle",
    "phase": "",
    "message": "No refresh or training run has been started.",
    "output": "",
    "error": "",
}


def _job_snapshot():
    with _job_lock:
        return dict(_job_status)


def _run_refresh_and_train(start: str):
    commands = [
        ("Refreshing public data", [sys.executable, "-m", "src.ingestion", "--start", start]),
        ("Training models and generating estimates", [sys.executable, "main.py"]),
    ]
    output = []
    errors = []
    try:
        for phase, command in commands:
            with _job_lock:
                _job_status.update(
                    state="running",
                    phase=phase,
                    message=phase + "…",
                    output="\n".join(output),
                    error="\n".join(errors),
                )
            result = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
            )
            if result.stdout:
                output.append(result.stdout)
            if result.stderr:
                errors.append(result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"{phase} failed with exit code {result.returncode}."
                )
        with _job_lock:
            _job_status.update(
                state="completed",
                phase="",
                message="Public data refreshed and the model estimate was regenerated.",
                output="\n".join(output),
                error="\n".join(errors),
            )
    except Exception as exc:
        with _job_lock:
            _job_status.update(
                state="failed",
                phase="",
                message="Refresh and training failed.",
                output="\n".join(output),
                error="\n".join(errors + [str(exc)]),
            )


def _csv(path: Path, dates=()):
    if not path.exists():
        raise HTTPException(
            404,
            "Research artifacts are not available yet. Use Refresh data + retrain in the dashboard.",
        )
    return pd.read_csv(path, parse_dates=list(dates))

@app.get("/api/health")
def health(): return {"status": "ok", "trained_artifacts_available": PREDICTIONS.exists()}


@app.get("/api/refresh-and-train/status")
def refresh_and_train_status():
    return _job_snapshot()


@app.post("/api/refresh-and-train", status_code=202)
def start_refresh_and_train(payload: RefreshRequest):
    start = payload.start.isoformat()

    with _job_lock:
        if _job_status["state"] in {"queued", "running"}:
            raise HTTPException(409, "A refresh and training job is already running.")
        _job_status.update(
            state="queued",
            phase="",
            message="Refresh and training queued.",
            output="",
            error="",
        )
    worker = Thread(target=_run_refresh_and_train, args=(start,), daemon=True)
    worker.start()
    return _job_snapshot()


@app.get("/api/summary")
def summary():
    if not REPORT.exists():
        raise HTTPException(
            404,
            "Experiment report is not available yet. Use Refresh data + retrain in the dashboard.",
        )
    predictions = _csv(PREDICTIONS, ["Date"])
    latest_test = predictions.iloc[-1].to_dict()
    latest_test["Date"] = str(latest_test["Date"].date())
    current = _csv(CURRENT_PREDICTION, ["Date"]) if CURRENT_PREDICTION.exists() else predictions.tail(1)
    latest = current.iloc[-1].to_dict()
    latest["Date"] = str(pd.Timestamp(latest["Date"]).date())
    return {
        "experiment": json.loads(REPORT.read_text(encoding="utf-8")),
        "latest": latest,
        "latest_test": latest_test,
    }

@app.get("/api/current-risk")
def current_risk():
    if not CURRENT_PREDICTION.exists():
        raise HTTPException(404, "Current estimate unavailable. Use Refresh data + retrain in the app.")
    result = _csv(CURRENT_PREDICTION, ["Date"]).iloc[-1].to_dict()
    result["Date"] = str(pd.Timestamp(result["Date"]).date())
    return result

@app.get("/api/next-session-prediction")
def next_session_prediction():
    if not CURRENT_PREDICTION.exists():
        raise HTTPException(404, "Next-session estimate unavailable. Use Refresh data + retrain in the app.")
    result = _csv(CURRENT_PREDICTION, ["Date"]).iloc[-1].to_dict()
    result["Date"] = str(pd.Timestamp(result["Date"]).date())
    return result

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

@app.get("/api/next-session-metrics")
def next_session_metrics():
    if not NEXT_SESSION_METRICS.exists():
        raise HTTPException(404, "Next-session model metrics are unavailable. Retrain from the dashboard.")
    return json.loads(NEXT_SESSION_METRICS.read_text(encoding="utf-8"))

@app.get("/api/next-session-predictions")
def next_session_predictions():
    return _csv(NEXT_SESSION_PREDICTIONS, ["Date"]).assign(
        Date=lambda data: data.Date.dt.strftime("%Y-%m-%d")
    ).to_dict(orient="records")

@app.get("/api/walk-forward")
def walk_forward():
    if not WALK_FORWARD_SUMMARY.exists():
        raise HTTPException(404, "Walk-forward metrics are unavailable. Retrain from the dashboard.")
    return json.loads(WALK_FORWARD_SUMMARY.read_text(encoding="utf-8"))

@app.get("/api/data-status")
def data_status():
    if DATABASE.exists():
        from src.ingestion import MarketDataStore

        store = MarketDataStore(DATABASE)
        statuses = store.statuses()
        metadata = store.metadata()
        return {
            "sources": statuses.fillna("").to_dict(orient="records"),
            "metadata": metadata.fillna("").to_dict(orient="records"),
        }
    if DATA_STATUS.exists():
        statuses = _csv(DATA_STATUS).fillna("")
        return {"sources": statuses.to_dict(orient="records"), "metadata": []}
    return {"sources": [], "metadata": []}

@app.get("/api/event-labels")
def event_labels():
    if not EVENT_LABELS.exists():
        raise HTTPException(404, "Documented event labels are unavailable.")
    return _csv(EVENT_LABELS).fillna("").to_dict(orient="records")

@app.get("/")
def frontend():
    if (FRONTEND_DIST / "index.html").exists(): return FileResponse(FRONTEND_DIST / "index.html")
    return JSONResponse({"message":"React frontend is not built. Run npm.cmd install and npm.cmd run build in frontend/."},status_code=503)

if FRONTEND_DIST.exists(): app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
