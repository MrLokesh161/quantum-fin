import pandas as pd

def create_labels(data, close_column, horizon, crash_threshold, warning_threshold):
    """Only label construction looks forward; features never use future rows."""
    d=data.copy(); close=d[close_column]; future_min=pd.concat([close.shift(-i) for i in range(1,horizon+1)],axis=1).min(axis=1)
    d["future_drawdown"]=future_min/close-1; d["crash_target"]=(d.future_drawdown<=crash_threshold).astype(int); d["risk_level_actual"]=0
    d.loc[d.future_drawdown<=warning_threshold,"risk_level_actual"]=1; d.loc[d.future_drawdown<=crash_threshold,"risk_level_actual"]=2; d.loc[d.future_drawdown<=crash_threshold*1.5,"risk_level_actual"]=3
    return d.iloc[:-horizon].copy()
