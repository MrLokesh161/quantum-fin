"""Local Streamlit dashboard for saved QETI research artifacts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PREDICTIONS = OUTPUTS / "predictions"
METRICS = OUTPUTS / "metrics"
REPORTS = OUTPUTS / "reports"
DATABASE = ROOT / "data" / "qeti.sqlite3"
EVENTS_PATH = ROOT / "data" / "crisis_events.csv"

st.set_page_config(page_title="QETI Local Research Dashboard", layout="wide")
st.title("QETI · Financial Instability Research")
st.caption(
    "Local research prototype. Yahoo Finance values are delayed/end-of-day, not real-time. "
    "Model probabilities are not calibrated real-world crash probabilities."
)


def read_csv(path: Path, parse_dates=()) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=list(parse_dates))


with st.sidebar:
    st.subheader("Local controls")
    start_date = st.date_input("Refresh start date", value=pd.Timestamp("1990-01-01").date())
    if st.button("Refresh data + train next-session estimate", use_container_width=True):
        with st.spinner("Refreshing public sources, training models, and estimating next-session risk…"):
            result = subprocess.run(
                [
                    sys.executable,
                    "refresh_and_train.py",
                    "--start",
                    start_date.isoformat(),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
            )
        if result.stdout:
            st.code(result.stdout)
        if result.returncode:
            st.error(result.stderr or "Refresh and retraining failed.")
        else:
            st.success("Public data refreshed and the next-session estimate was regenerated.")
            if result.stderr:
                st.warning(result.stderr)
            st.rerun()
    if st.button("Refresh free public data", use_container_width=True):
        with st.spinner("Refreshing Yahoo Finance, Binance, and FRED data…"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.ingestion",
                    "--start",
                    start_date.isoformat(),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
            )
        if result.stdout:
            st.code(result.stdout)
        if result.returncode:
            st.error(result.stderr or "Data refresh reported unavailable sources.")
        elif result.stderr:
            st.warning(result.stderr)
        else:
            st.success("Data refresh completed. Unavailable sources are shown in the status table.")
    if st.button("Run model pipeline", use_container_width=True):
        with st.spinner("Training chronological baselines and generating artifacts…"):
            result = subprocess.run(
                [sys.executable, "main.py"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
            )
        if result.stdout:
            st.code(result.stdout)
        if result.returncode:
            st.error(result.stderr or "The pipeline did not complete successfully.")
        else:
            st.success("Model run completed.")
            if result.stderr:
                st.warning(result.stderr)

current = read_csv(PREDICTIONS / "current_prediction.csv", ["Date"])
predictions = read_csv(PREDICTIONS / "predictions.csv", ["Date"])
metrics = read_csv(METRICS / "model_comparison.csv")
events = read_csv(EVENTS_PATH, ["start_date", "peak_date", "end_date"])
engineered = read_csv(OUTPUTS / "engineered_dataset.csv", ["Date"])
source_status = read_csv(REPORTS / "data_source_status.csv")
dataset_metadata = read_csv(REPORTS / "dataset_metadata.csv")
early_warning = read_csv(REPORTS / "early_warning_performance.csv", ["warning_date", "crisis_date"])
report_path = REPORTS / "experiment_report.json"
if DATABASE.exists():
    from src.ingestion import MarketDataStore

    store = MarketDataStore(DATABASE)
    database_status = store.statuses()
    if not database_status.empty:
        source_status = database_status
    database_metadata = store.metadata()
    if not database_metadata.empty:
        dataset_metadata = database_metadata
if DATABASE.exists():
    from src.ingestion import MarketDataStore

    store = MarketDataStore(DATABASE)
    database_status = store.statuses()
    if not database_status.empty:
        source_status = database_status
    database_metadata = store.metadata()
    if not database_metadata.empty:
        dataset_metadata = database_metadata

if not current.empty:
    latest = current.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("Risk probability", f"{latest['crash_probability']:.1%}")
    cols[1].metric("Risk classification", latest.get("risk_classification", "Unavailable"))
    cols[2].metric("Systemic Risk Score", f"{latest['systemic_risk_score']:.1f} / 100")
    cols[3].metric("QTSI", f"{latest['qtsi']:.1f} / 100")
    st.caption(
        f"Risk estimate as of {latest['Date'].date()} for the "
        f"{int(latest.get('prediction_horizon_sessions', 10))} trading-session "
        "drawdown horizon beginning at the next available S&P 500 session · "
        f"Model: {latest.get('model', 'Random Forest')} · "
        "SRS + QTSI = 100. QTSI is a project-defined indicator."
    )
else:
    st.warning("No current prediction found. Run the model pipeline to generate one.")

st.subheader("Data sources and freshness")
if source_status.empty:
    st.info("No source status report found. Use Refresh free public data to create one.")
else:
    st.dataframe(source_status, use_container_width=True, hide_index=True)
    unavailable = source_status[source_status["status"].eq("unavailable")]
    if not unavailable.empty:
        st.warning(
            "One or more sources are unavailable. Cached data may be in use; "
            "review the status messages before interpreting results."
        )
    if "latest_observation" in source_status:
        now = pd.Timestamp.now().normalize()
        stale_rows = []
        for _, row in source_status.dropna(subset=["latest_observation"]).iterrows():
            latest_date = pd.to_datetime(row["latest_observation"], errors="coerce")
            if pd.isna(latest_date):
                continue
            limit_days = 60 if row["source"] == "fred" else 8
            if (now - latest_date.normalize()).days > limit_days:
                stale_rows.append(f"{row['source']} {row['symbol']} ({latest_date.date()})")
        if stale_rows:
            st.warning("Potentially stale observations: " + ", ".join(stale_rows))
    if not dataset_metadata.empty:
        with st.expander("Source attribution and dataset metadata"):
            st.dataframe(dataset_metadata, use_container_width=True, hide_index=True)
st.caption("Provider access is public/free but can be delayed, rate-limited, or unavailable.")

left, right = st.columns(2)
with left:
    st.subheader("Market indicators")
    if not engineered.empty and {"Date", "Close"}.issubset(engineered.columns):
        chart_data = engineered.tail(2520)
        price = px.line(chart_data, x="Date", y="Close", title="S&P 500 close (last 10 years)")
        broad_events = (
            events[events["market_relevance"].eq("broad_market_crash")]
            if "market_relevance" in events
            else events[events["event_type"].eq("market crash")]
        )
        for _, event in broad_events.iterrows():
            if chart_data["Date"].min() <= event["peak_date"] <= chart_data["Date"].max():
                price.add_vline(
                    x=event["peak_date"],
                    line_dash="dot",
                    opacity=0.5,
                    annotation_text=event["event_name"],
                    annotation_position="top",
                )
        st.plotly_chart(price, use_container_width=True)
        if "volatility_20" in chart_data:
            st.plotly_chart(
                px.line(chart_data, x="Date", y="volatility_20", title="20-day annualized volatility"),
                use_container_width=True,
            )
    else:
        st.info("Engineered market history is not available yet.")

with right:
    st.subheader("Risk history")
    if not predictions.empty:
        risk = px.line(
            predictions,
            x="Date",
            y="crash_probability",
            title="Held-out chronological risk probability",
        )
        for _, event in events.iterrows():
            if predictions["Date"].min() <= event["peak_date"] <= predictions["Date"].max():
                risk.add_vline(x=event["peak_date"], line_dash="dot", opacity=0.5)
        st.plotly_chart(risk, use_container_width=True)
        if not early_warning.empty:
            st.subheader("Historical warning outcomes")
            st.dataframe(early_warning, use_container_width=True, hide_index=True)
    else:
        st.info("Test-period predictions are not available yet.")

st.subheader("Chronological model comparison")
if metrics.empty:
    st.info("No model comparison has been generated. Run the model pipeline.")
else:
    preferred = [
        column for column in (
            "feature_set", "model", "threshold", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "false_alarm_rate", "runtime_seconds",
        ) if column in metrics
    ]
    st.dataframe(
        metrics[preferred].sort_values(["feature_set", "model"]),
        use_container_width=True,
        hide_index=True,
    )
    confusion_files = sorted((OUTPUTS / "figures").glob("confusion_*.png"))
    if confusion_files:
        selected = st.selectbox("Confusion matrix", confusion_files, format_func=lambda path: path.stem.replace("confusion_", ""))
        st.image(str(selected), use_container_width=True)

st.subheader("Topological features")
topological = [column for column in predictions.columns if column.startswith("tda_")]
if not predictions.empty and topological:
    feature = st.selectbox("Persistence summary", topological)
    st.plotly_chart(
        px.line(predictions, x="Date", y=feature, title=feature.replace("_", " ").title()),
        use_container_width=True,
    )
else:
    st.info("Topological features are unavailable. Check the experiment report for the TDA status.")

st.subheader("Documented financial events")
if events.empty:
    st.info("No event-label dataset is available.")
else:
    st.dataframe(events, use_container_width=True, hide_index=True)

with st.expander("Experiment limitations"):
    if report_path.exists():
        st.json(__import__("json").loads(report_path.read_text(encoding="utf-8")))
    else:
        st.write(
            "The pipeline uses forward-drawdown labels, a small number of historical crisis events, "
            "and uncalibrated classifier scores. It does not establish a quantum or investment advantage."
        )
