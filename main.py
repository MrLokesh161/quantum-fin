"""Primary executable for the classical QETI early-warning research prototype."""
import json
from pathlib import Path
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils import ROOT,load_config,prepare_outputs,set_seed
from src.data_loader import load_csv,dataset_summary
from src.preprocessing import clean_data
from src.feature_engineering import build_features
from src.labeling import create_labels
from src.sliding_window import window_descriptors
from src.tda import tda_features
from src.feature_fusion import build_feature_sets
from src.baseline_models import chronological_splits,train_models
from src.evaluation import evaluate,timeline_plots
from src.risk_engine import add_risk_predictions,add_systemic_risk
from src.qtsi import add_qtsi
from src.early_warning import evaluate_early_warnings

def main():
    cfg=load_config();prepare_outputs(cfg);set_seed(cfg["models"]["random_seed"]);out={k:ROOT/v for k,v in cfg["outputs"].items()};fig=out["figures"]
    from src.data_loader import discover_dataset
    dataset_path,is_demo=discover_dataset(ROOT,cfg["data"]["dataset_dir"],cfg["data"]["primary_dataset"]);raw=load_csv(dataset_path,cfg["data"]["date_column"]);summary=dataset_summary(raw,"Date");summary["dataset_path"]=str(dataset_path.relative_to(ROOT));summary["demo_data"]=is_demo
    clean,outliers=clean_data(raw);clean.to_csv(out["root"] / "cleaned_dataset.csv",index=False);outliers.to_csv(out["reports"] / "iqr_outlier_report.csv",index=False)
    engineered=create_labels(build_features(clean,cfg["data"]["close_column"]),cfg["data"]["close_column"],cfg["features"]["horizon_days"],cfg["features"]["crash_threshold"],cfg["features"]["warning_threshold"]).dropna().reset_index(drop=True);engineered.to_csv(out["root"] / "engineered_dataset.csv",index=False)
    base_features=[c for c in engineered.select_dtypes("number").columns if c not in {"future_drawdown","crash_target","risk_level_actual"}]
    windows,clouds=window_descriptors(engineered,base_features,cfg["features"]["window_size"])
    tda_status="Disabled by configuration"
    if cfg["models"]["enable_tda"]:
        tda,tda_status=tda_features(clouds,fig,cfg["models"]["tda_stride"]);windows=pd.concat([windows,tda],axis=1).dropna().reset_index(drop=True)
    feature_sets=build_feature_sets(windows);train,val,test=chronological_splits(windows,cfg["splits"]["train"],cfg["splits"]["validation"]);all_metrics=[];runs={}
    for set_name,feature_cols in feature_sets.items():
        models,scaler,x_test,y_test=train_models(train,val,test,feature_cols,cfg);metrics,probs=evaluate(models,x_test,y_test,feature_cols,fig,set_name);all_metrics.append(metrics);runs[set_name]=(models,scaler,feature_cols,probs,y_test)
    metrics=pd.concat(all_metrics,ignore_index=True);metrics.to_csv(out["metrics"] / "model_comparison.csv",index=False)
    ax=metrics.pivot_table(index="model",columns="feature_set",values="pr_auc").plot.bar(figsize=(10,5));ax.set_ylabel("PR-AUC");ax.set_title("Model comparison (actual PR-AUC)");plt.tight_layout();plt.savefig(fig/"model_comparison.png",dpi=180);plt.close()
    models,scaler,feature_cols,probs,y_test=runs["Financial + TDA"] if "Financial + TDA" in runs else runs["Engineered financial"]
    prediction=test[["Date"]].copy();prediction["actual"]=y_test;prediction["crash_probability"]=probs["Random Forest"];prediction=prediction.merge(engineered[["Date","Close","volatility_20","drawdown"]],on="Date",how="left");tda_cols=[c for c in windows if c.startswith("tda_")];prediction=prediction.join(test[tda_cols].reset_index(drop=True));prediction=add_qtsi(add_systemic_risk(add_risk_predictions(prediction,cfg["risk_thresholds"]),cfg["risk_weights"]));prediction.to_csv(out["predictions"] / "predictions.csv",index=False);prediction.to_csv(out["predictions"] / "risk_predictions.csv",index=False);timeline_plots(prediction,fig)
    prediction.plot(x="Date",y=["crash_probability","systemic_risk_score","qtsi"],figsize=(12,5),title="Risk timeline");plt.tight_layout();plt.savefig(fig/"risk_timeline.png",dpi=180);plt.close()
    if tda_cols: prediction.plot(x="Date",y=tda_cols,figsize=(12,5),title="Topological feature time series");plt.tight_layout();plt.savefig(fig/"topological_features.png",dpi=180);plt.close()
    early=evaluate_early_warnings(prediction,ROOT/"data/crisis_events.csv");early.to_csv(out["reports"] / "early_warning_performance.csv",index=False);early.to_csv(out["metrics"] / "early_warning_results.csv",index=False);early.to_csv(out["metrics"] / "crisis_detection.csv",index=False)
    joblib.dump({"model":models["Random Forest"],"scaler":scaler,"features":feature_cols},out["models"] / "random_forest_tda.joblib")
    report={"dataset":summary,"feature_sets":{k:len(v) for k,v in feature_sets.items()},"tda_feature_count":len(tda_cols),"training_period":[str(train.Date.min().date()),str(train.Date.max().date())],"validation_period":[str(val.Date.min().date()),str(val.Date.max().date())],"testing_period":[str(test.Date.min().date()),str(test.Date.max().date())],"tda_status":tda_status,"lstm_status":"Optional module; unavailable unless TensorFlow is installed.","limitations":"Research prototype only. Historical patterns do not establish reliable real-world collapse prediction."}
    (out["reports"] / "experiment_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");(out["reports"] / "experiment_report.md").write_text("# Experiment Report\n\n```json\n"+json.dumps(report,indent=2)+"\n```\n",encoding="utf-8")
    print(f"DATASET: {summary['rows']} rows, {summary['columns']} columns\nPREPROCESSING: Completed\nFEATURE ENGINEERING: Completed\nSLIDING WINDOW: {len(windows)} samples\nTDA: {tda_status}\nRANDOM FOREST: Completed\nXGBOOST: {'Completed' if 'XGBoost' in models else 'Fallback used'}\nLSTM: {report['lstm_status']}\nEARLY WARNING: Completed\nDASHBOARD: Available via streamlit run dashboard/app.py\nOutputs: {out['root']}")
if __name__=="__main__":main()
