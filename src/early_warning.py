import pandas as pd

def evaluate_early_warnings(predictions,event_path,threshold=.4):
    events=pd.read_csv(event_path)
    if "peak_date" in events.columns: events=events.rename(columns={"peak_date":"crisis_date"})
    for col in ("start_date","crisis_date","end_date"): events[col]=pd.to_datetime(events[col])
    rows=[]
    for _,e in events.iterrows():
        if e.crisis_date < predictions.Date.min() or e.start_date > predictions.Date.max():
            rows.append({"event":e.event_name,"event_type":e.event_type,"warning_date":pd.NaT,"crisis_date":e.crisis_date,"lead_time_days":None,"status":"Not evaluable (outside test period)"})
            continue
        warning=predictions[(predictions.Date>=e.start_date)&(predictions.Date<=e.crisis_date)&(predictions.crash_probability>=threshold)].Date.min()
        rows.append({"event":e.event_name,"event_type":e.event_type,"warning_date":warning,"crisis_date":e.crisis_date,"lead_time_days":(e.crisis_date-warning).days if pd.notna(warning) else None,"status":"Detected" if pd.notna(warning) else "Missed"})
    return pd.DataFrame(rows)
