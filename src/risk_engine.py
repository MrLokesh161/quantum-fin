def classify_risk(probability, thresholds):
    labels=["Stable","Caution","High Risk","Critical"] if len(thresholds)==3 else ["Stable","Watch","Warning","High Risk","Critical"]
    for i,threshold in enumerate(thresholds):
        if probability<threshold:return i,labels[i]
    return 4,"Critical"

def add_risk_predictions(predictions, thresholds):
    d=predictions.copy();v=[classify_risk(x,thresholds) for x in d.crash_probability];d["risk_level"]=[x[0] for x in v];d["risk_label"]=[x[1] for x in v];d["early_warning_alert"]=d.risk_level>=2;return d

def add_systemic_risk(predictions, weights):
    """Prototype normalized systemic-risk score, 0=stable and 100=unstable."""
    d=predictions.copy()
    def norm(s): return (s-s.min())/(s.max()-s.min()) if s.max()>s.min() else s*0
    entropy=norm(d.get("tda_h0_persistence_entropy",d.crash_probability*0));topology=norm(d.get("tda_h0_max_persistence",d.crash_probability*0))
    score=(weights["probability"]*norm(d.crash_probability)+weights["volatility"]*norm(d.volatility_20)+weights["drawdown"]*norm(-d.drawdown)+weights["persistence_entropy"]*entropy+weights["topology"]*topology)
    d["systemic_risk_score"]=(100*score).clip(0,100);return d
