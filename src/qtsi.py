def add_qtsi(predictions):
    """Classical QTSI prototype: 100 means greater estimated stability."""
    d=predictions.copy()
    def norm(s): return (s-s.min())/(s.max()-s.min()) if s.max()>s.min() else s*0
    entropy=norm(d.get("tda_h0_persistence_entropy",d.crash_probability*0)); strength=norm(d.get("tda_h0_max_persistence",d.crash_probability*0))
    instability=(norm(d.volatility_20)+norm(-d.drawdown)+entropy+strength+norm(d.crash_probability))/5;d["qtsi"]=(100*(1-instability)).clip(0,100);return d
