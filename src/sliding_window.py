import numpy as np
import pandas as pd

def window_descriptors(data, feature_columns, window, point_cloud_columns=None):
    """Creates sequential point clouds and flattened statistically enriched rows."""
    cloud_columns=point_cloud_columns or feature_columns
    missing=set(cloud_columns).difference(data.columns)
    if missing:
        raise ValueError(f"Point-cloud features are missing from the input data: {sorted(missing)}")
    rows=[]; clouds=[]
    for end in range(window-1,len(data)):
        s=data.iloc[end-window+1:end+1]; r=s.daily_return; row={"Date":data.iloc[end].Date,"target":int(data.iloc[end].crash_target)}
        row.update({f"last_{c}":s.iloc[-1][c] for c in feature_columns})
        row.update({"window_mean_return":r.mean(),"window_return_volatility":r.std(),"window_max_drawdown":s.drawdown.min(),"window_return_skew":r.skew(),"window_return_kurtosis":r.kurt(),"volatility_trend":s.volatility_20.iloc[-1]-s.volatility_20.iloc[0],"momentum_trend":s.momentum_10.iloc[-1]-s.momentum_10.iloc[0]})
        rows.append(row); clouds.append(s[cloud_columns].to_numpy(float))
    frame=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
    valid=frame.notna().all(axis=1).to_numpy()
    frame=frame.loc[valid].reset_index(drop=True)
    cloud_array=np.asarray(clouds,dtype=float)
    return frame,cloud_array[valid]
