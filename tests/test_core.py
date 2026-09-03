import pandas as pd
from src.labeling import create_labels
from src.risk_engine import classify_risk
from src.qtsi import add_qtsi

def test_label_generation():
    d=pd.DataFrame({"Close":[100,98,94,90,91,92]});out=create_labels(d,"Close",2,-.05,-.03);assert out.crash_target.iloc[0]==1
def test_risk_classification(): assert classify_risk(.61,[.2,.4,.6,.8])[1]=="High Risk"
def test_qtsi_bounds():
    d=pd.DataFrame({"volatility_20":[.1,.2],"drawdown":[-.1,-.2],"crash_probability":[.1,.8]});out=add_qtsi(d);assert out.qtsi.between(0,100).all()
