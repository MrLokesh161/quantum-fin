def classify_risk(probability, thresholds):
    labels=["Stable","Caution","High Risk","Critical"] if len(thresholds)==3 else ["Stable","Watch","Warning","High Risk","Critical"]
    for i,threshold in enumerate(thresholds):
        if probability<threshold:return i,labels[i]
    return 4,"Critical"

def add_risk_predictions(predictions, thresholds):
    d=predictions.copy();v=[classify_risk(x,thresholds) for x in d.crash_probability];d["risk_level"]=[x[0] for x in v];d["risk_label"]=[x[1] for x in v];d["risk_classification"]=[classify_research_risk(x,thresholds) for x in d.crash_probability];d["early_warning_alert"]=d.risk_level>=2;return d

def add_systemic_risk(predictions):
    """Compute project-defined SRS directly from the model risk probability."""
    d=predictions.copy()
    probability=d["crash_probability"].clip(0,1)
    d["risk_probability"]=probability
    d["systemic_instability_probability"]=probability
    d["systemic_risk_score"]=100*probability
    return d

def classify_research_risk(probability, thresholds):
    labels=["Low","Moderate","High","Critical"]
    for index,threshold in enumerate(thresholds):
        if probability < threshold:
            return labels[index]
    return labels[-1]
