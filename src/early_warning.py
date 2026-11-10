import pandas as pd

def evaluate_early_warnings(predictions,event_path,threshold=.4):
    events=pd.read_csv(event_path)
    if "peak_date" in events.columns: events=events.rename(columns={"peak_date":"crisis_date"})
    for col in ("start_date","crisis_date","end_date"): events[col]=pd.to_datetime(events[col])
    rows=[]
    for _,e in events.iterrows():
        row={"event":e.event_name,"event_type":e.event_type,"warning_date":pd.NaT,"crisis_date":e.crisis_date,"lead_time_days":None,"source_reference":e.get("source_reference","")}
        if str(e.get("market_relevance","")).lower() not in {"broad_market_crash","equity_market"}:
            row["status"]="Not comparable (event is not a broad-market crash)"
            rows.append(row)
            continue
        if e.crisis_date < predictions.Date.min() or e.start_date > predictions.Date.max():
            row["status"]="Not evaluable (outside test period)"
            rows.append(row)
            continue
        warning=predictions[(predictions.Date>=e.start_date)&(predictions.Date<=e.crisis_date)&(predictions.crash_probability>=threshold)].Date.min()
        row.update({"warning_date":warning,"lead_time_days":(e.crisis_date-warning).days if pd.notna(warning) else None,"status":"Detected" if pd.notna(warning) else "Missed"})
        rows.append(row)
    return pd.DataFrame(rows)
