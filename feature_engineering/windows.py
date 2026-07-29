"""Sliding-window transformation for sequential-learning-ready datasets."""
import numpy as np
import pandas as pd


def create_sliding_windows(data: pd.DataFrame, feature_columns: list[str], target_column: str, window_size: int) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Build windows ending at t; labels refer to the future relative to t."""
    values = data[feature_columns].to_numpy(dtype=float)
    targets = data[target_column].to_numpy(dtype=int)
    X = np.stack([values[end - window_size:end] for end in range(window_size, len(data))])
    y = targets[window_size:]
    dates = data["Date"].iloc[window_size:].reset_index(drop=True)
    return X, y, dates
