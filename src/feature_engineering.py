import numpy as np

def build_features(data, close_column="Close", volume_column="Volume"):
    d=data.copy(); c=d[close_column]; returns=c.pct_change()
    d["daily_return"]=returns; d["log_return"]=np.log(c/c.shift()); d["percentage_change"]=returns
    d["weekly_return"]=c.pct_change(5);d["monthly_return"]=c.pct_change(21)
    for n in (5,20,50,200): d[f"sma_{n}"]=c.rolling(n).mean()
    for n in (5,20,60): d[f"volatility_{n}"]=returns.rolling(n).std()*np.sqrt(252)
    for n in (12,26): d[f"ema_{n}"]=c.ewm(span=n,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean(); d["rsi_14"]=100-100/(1+gain/loss.replace(0,np.nan))
    for n in (5,10,20):d[f"momentum_{n}"]=c/c.shift(n)-1
    d["macd"]=d.ema_12-d.ema_26; d["macd_signal"]=d.macd.ewm(span=9,adjust=False).mean()
    mean=c.rolling(20).mean(); d["bollinger_width"]=4*c.rolling(20).std()/mean; d["rolling_max"]=c.cummax(); d["drawdown"]=c/d.rolling_max-1; d["distance_sma_20"]=c/d.sma_20-1;d["downside_volatility_20"]=returns.clip(upper=0).rolling(20).std()*np.sqrt(252);d["sharpe_like_20"]=returns.rolling(20).mean()/returns.rolling(20).std().replace(0,np.nan)*np.sqrt(252)
    if "High" in d and "Low" in d: d["high_low_range"]=(d.High-d.Low)/c;d["normalized_range"]=(d.High-d.Low)/(d.High.rolling(20).max()-d.Low.rolling(20).min()).replace(0,np.nan)
    if "Open" in d: d["close_open_range"]=(c-d.Open)/d.Open
    if volume_column in d: d["volume_change"]=d[volume_column].pct_change(); d["volume_ma_20"]=d[volume_column].rolling(20).mean(); d["volume_ratio"]=d[volume_column]/d.volume_ma_20.replace(0,np.nan)
    return d.replace([np.inf,-np.inf],np.nan)
