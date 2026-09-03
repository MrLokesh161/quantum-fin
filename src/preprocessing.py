import numpy as np
import pandas as pd

def clean_data(data, date_column="Date"):
    """Retain IQR extremes: they may be financially meaningful collapse signals."""
    frame=data.copy().replace([np.inf,-np.inf],np.nan); numeric=frame.select_dtypes("number").columns
    frame[numeric]=frame[numeric].ffill().bfill(); report=[]
    for name in numeric:
        q1,q3=frame[name].quantile([.25,.75]); iqr=q3-q1
        report.append({"feature":name,"outlier_count":int(((frame[name]<q1-1.5*iqr)|(frame[name]>q3+1.5*iqr)).sum())})
    return frame.dropna(subset=numeric,how="all"),pd.DataFrame(report)
