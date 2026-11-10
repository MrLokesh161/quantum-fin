"""Named, auditable feature sets for financial-versus-topological experiments."""
def build_feature_sets(windows):
    numeric=[c for c in windows.select_dtypes("number") if c!="target"]
    raw=[c for c in numeric if c.startswith("last_") and any(x in c.lower() for x in ("open","high","low","close","volume"))]
    engineered=[c for c in numeric if not c.startswith("tda_")]
    topological=[c for c in numeric if c.startswith("tda_")]
    feature_sets={"Raw financial":raw or engineered,"Engineered financial":engineered}
    if topological:
        feature_sets["Topological only"]=topological
        feature_sets["Financial + TDA"]=engineered+topological
    return feature_sets
