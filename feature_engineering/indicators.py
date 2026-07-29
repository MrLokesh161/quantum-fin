"""Financial technical indicators and future crash labels."""
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def engineer_features(frame: pd.DataFrame, market: str, horizon: int, threshold: float) -> tuple[pd.DataFrame, str]:
    """Add technical indicators using the selected market's closing price."""
    data = frame.copy()
    close_col = f"{market}_Close"
    if close_col not in data:
        candidates = [c for c in data if c.lower().endswith("_close")]
        if not candidates:
            raise ValueError("No close-price column found; expected '<market>_Close'.")
        close_col = candidates[0]
    close = data[close_col]
    data["daily_return"] = close.pct_change()
    data["log_return"] = np.log(close / close.shift(1))
    data["ma_20"] = close.rolling(20).mean()
    data["ma_50"] = close.rolling(50).mean()
    data["ema_20"] = close.ewm(span=20, adjust=False).mean()
    data["rolling_volatility_20"] = data["daily_return"].rolling(20).std() * np.sqrt(252)
    data["momentum_10"] = close / close.shift(10) - 1
    data["rsi_14"] = _rsi(close)
    ema_fast, ema_slow = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema_fast - ema_slow
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    rolling_mean, rolling_std = close.rolling(20).mean(), close.rolling(20).std()
    data["bollinger_upper"] = rolling_mean + 2 * rolling_std
    data["bollinger_lower"] = rolling_mean - 2 * rolling_std
    data["percentage_change"] = close.pct_change()
    for lag in (1, 2, 3, 5, 10):
        data[f"return_lag_{lag}"] = data["daily_return"].shift(lag)
    future_minimum = pd.concat([close.shift(-step) for step in range(1, horizon + 1)], axis=1).min(axis=1)
    data["future_max_drawdown"] = future_minimum / close - 1
    data["crash_target"] = (data["future_max_drawdown"] <= threshold).astype("int64")
    # The last horizon has unknown targets, not stable labels.
    data.loc[data.index[-horizon:], "crash_target"] = np.nan
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return data, close_col
