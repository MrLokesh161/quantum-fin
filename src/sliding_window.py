import numpy as np
import pandas as pd

def window_descriptors(data, feature_columns, window):
    """Creates sequential point clouds and flattened statistically enriched rows."""
    rows=[]; clouds=[]
    for end in range(window,len(data)):
        s=data.iloc[end-window:end]; r=s.daily_return; row={"Date":data.iloc[end].Date,"target":int(data.iloc[end].crash_target)}
        row.update({f"last_{c}":s.iloc[-1][c] for c in feature_columns})
        row.update({"window_mean_return":r.mean(),"window_return_volatility":r.std(),"window_max_drawdown":s.drawdown.min(),"window_return_skew":r.skew(),"window_return_kurtosis":r.kurt(),"volatility_trend":s.volatility_20.iloc[-1]-s.volatility_20.iloc[0],"momentum_trend":s.momentum_10.iloc[-1]-s.momentum_10.iloc[0]})
        rows.append(row); clouds.append(s[feature_columns].to_numpy(float))
    return pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True),np.array(clouds)
