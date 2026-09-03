"""Robust loading for ordinary and yfinance three-row-header CSVs."""
from pathlib import Path
import pandas as pd

def discover_dataset(root, configured_dir="data/raw", configured_file=None):
    """Find a market CSV without relying on a fixed filename or demo data."""
    root=Path(root)
    if configured_file:
        candidate=root/configured_dir/configured_file
        if candidate.exists(): return candidate,False
    candidates=[]
    for folder in (root/configured_dir,root/"data",root/"datasets"):
        if folder.exists(): candidates.extend(p for p in folder.glob("*.csv") if "event" not in p.name.lower())
    if not candidates: raise FileNotFoundError("No market CSV found in data/raw, data, or datasets. Add a Date/Close CSV before running research experiments.")
    # Prefer S&P 500 when several market files are present; otherwise deterministic order.
    return sorted(set(candidates),key=lambda p:(0 if any(x in p.name.lower() for x in ("sp500","s&p","gspc")) else 1,p.name.lower()))[0],False

def load_csv(path, date_column="Date"):
    preview = pd.read_csv(path, header=None, nrows=3)
    data = pd.read_csv(path, skiprows=[1, 2]) if len(preview) == 3 and str(preview.iloc[0,0]).lower() == "price" and str(preview.iloc[2,0]).lower() == "date" else pd.read_csv(path)
    if date_column not in data: data = data.rename(columns={data.columns[0]: date_column})
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna(subset=[date_column]).sort_values(date_column).drop_duplicates(date_column).reset_index(drop=True)
    for col in data.columns.drop(date_column): data[col] = pd.to_numeric(data[col], errors="coerce")
    return data

def dataset_summary(data, date_column):
    return {"rows":len(data),"columns":len(data.columns),"date_range":[str(data[date_column].min().date()),str(data[date_column].max().date())],"missing_values":data.isna().sum().to_dict()}
