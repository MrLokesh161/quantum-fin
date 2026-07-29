"""CSV ingestion and date-based market-data merging."""
from pathlib import Path
import pandas as pd


def _read_market_csv(path: Path) -> pd.DataFrame:
    """Read standard or yfinance's three-row-header CSV export."""
    preview = pd.read_csv(path, header=None, nrows=3)
    # yfinance exports: row 0=Price, row 1=Ticker, row 2=Date.
    if len(preview) >= 3 and str(preview.iloc[0, 0]).strip().lower() == "price" and str(preview.iloc[2, 0]).strip().lower() == "date":
        frame = pd.read_csv(path, skiprows=[1, 2])
    else:
        frame = pd.read_csv(path)
    frame = frame.rename(columns={frame.columns[0]: "Date"})
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date")
    for column in frame.columns.drop("Date"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_and_merge_datasets(dataset_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load all CSV files and outer-merge them on Date with source prefixes."""
    paths = sorted(dataset_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    individual: dict[str, pd.DataFrame] = {}
    merged: pd.DataFrame | None = None
    for path in paths:
        name = path.stem.lower()
        frame = _read_market_csv(path)
        individual[name] = frame
        renamed = frame.rename(columns={c: f"{name}_{c}" for c in frame.columns if c != "Date"})
        merged = renamed if merged is None else merged.merge(renamed, on="Date", how="outer")
    return merged.sort_values("Date").reset_index(drop=True), individual


def dataset_report(frame: pd.DataFrame) -> str:
    """Return reproducible dataset diagnostics for the research record."""
    lines = [f"Shape: {frame.shape}", "\nColumn types:", frame.dtypes.to_string(), "\n\nMissing values:", frame.isna().sum().to_string(), "\n\nSummary statistics:", frame.describe(include="all").to_string()]
    return "\n".join(lines)
