def add_qtsi(predictions):
    """Project-defined QTSI, complementary to SRS from the same risk probability."""
    d=predictions.copy()
    probability=d.get("risk_probability",d.crash_probability).clip(0,1)
    d["qtsi"]=100*(1-probability)
    d["systemic_risk_score"]=100*probability
    return d
