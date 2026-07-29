"""Look up the saved crash-risk probability for a test-set trading date.

Usage:
    env/Scripts/python.exe check_risk_by_date.py 2018-11-06
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PREDICTIONS_PATH = Path(__file__).resolve().parent / "outputs" / "predictions.csv"


def lookup_risk(date_text: str) -> None:
    """Print model risk probability for one ISO-8601 trading date."""
    try:
        requested_date = pd.Timestamp(date_text).normalize()
    except ValueError as error:
        raise SystemExit("Invalid date. Use ISO format YYYY-MM-DD, for example 2018-11-06.") from error

    if not PREDICTIONS_PATH.exists():
        raise SystemExit("Predictions do not exist yet. Train first: python run_pipeline.py")

    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["Date"])
    predictions["Date"] = predictions["Date"].dt.normalize()
    match = predictions.loc[predictions["Date"] == requested_date]
    if match.empty:
        first_date, last_date = predictions["Date"].min().date(), predictions["Date"].max().date()
        raise SystemExit(
            f"No test prediction for {requested_date.date()}. "
            f"Available test dates: {first_date} to {last_date}; non-trading days are excluded."
        )

    row = match.iloc[0]
    probability = float(row["high_risk_probability"])
    label = "HIGH RISK / CRASH" if probability >= 0.50 else "STABLE"
    print(f"Date: {requested_date.date()}")
    print(f"High-risk probability: {probability:.2%}")
    print(f"Model classification (0.50 threshold): {label}")
    print(f"Actual future-drawdown label: {int(row['actual'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a saved crash-risk probability by date.")
    parser.add_argument("date", help="Trading date in YYYY-MM-DD format")
    lookup_risk(parser.parse_args().date)
