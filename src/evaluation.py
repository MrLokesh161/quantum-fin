import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,RocCurveDisplay,PrecisionRecallDisplay
from sklearn.metrics import brier_score_loss, log_loss

from src.calibration import calibration_metrics


def tune_thresholds(models, scaler, validation, features, calibrators=None):
    """Choose an F1 threshold using validation only; default to 0.5 without positives."""
    if validation.empty or validation.target.nunique() < 2:
        return {name: 0.5 for name in models}
    x_validation=scaler.transform(validation[features])
    y_validation=validation.target.to_numpy()
    thresholds={}
    for name,model in models.items():
        probabilities=model.predict_proba(x_validation)[:,1]
        if calibrators and name in calibrators:
            probabilities=calibrators[name].predict_proba(
                np.column_stack((1.0-probabilities, probabilities))
            )[:,1]
        candidates=np.linspace(0.05,0.95,91)
        scores=[f1_score(y_validation,(probabilities>=threshold).astype(int),zero_division=0) for threshold in candidates]
        best=max(scores)
        thresholds[name]=float(max(threshold for threshold,score in zip(candidates,scores) if score==best))
    return thresholds

def evaluate(models,x_test,y_test,features,figures,feature_set="Engineered financial",thresholds=None,calibrators=None):
    rows=[];probs={}
    thresholds=thresholds or {}
    for name,model in models.items():
        probability_matrix=model.predict_proba(x_test)
        if calibrators and name in calibrators:
            probability_matrix=calibrators[name].predict_proba(probability_matrix)
        p=probability_matrix[:,1];threshold=float(thresholds.get(name,.5));pred=(p>=threshold).astype(int);probs[name]=p
        tn,fp,_fn,_tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
        model_classes = getattr(
            model, "classes_", np.arange(probability_matrix.shape[1])
        )
        calibration=calibration_metrics(
            probability_matrix, y_test, classes=model_classes
        )
        rows.append({"model":name,"feature_set":feature_set,"threshold":threshold,"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,zero_division=0),"recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),"roc_auc":roc_auc_score(y_test,p) if len(set(y_test))>1 else None,"pr_auc":average_precision_score(y_test,p) if np.any(np.asarray(y_test)==1) else None,"specificity":tn/max(tn+fp,1),"false_positive_rate":fp/max(fp+tn,1),"false_alarm_rate":fp/max(fp+tn,1),"brier_score":brier_score_loss(y_test,p),"log_loss":log_loss(y_test,probability_matrix,labels=[0,1]),"expected_calibration_error":calibration["expected_calibration_error"],"calibration_method":calibrators[name].status_ if calibrators and name in calibrators else "none"})
    if len(np.unique(y_test)) < 2:
        for fname,title in [
            ("roc_curve.png","ROC curve unavailable: test period contains one class"),
            ("precision_recall_curve.png","Precision-recall curve unavailable: test period contains one class"),
        ]:
            fig,ax=plt.subplots(figsize=(6,5));ax.text(.5,.5,title,ha="center",va="center",wrap=True);ax.set_axis_off()
            fig.tight_layout();fig.savefig(figures/fname,dpi=180);plt.close(fig)
    else:
        for fname,display in [("roc_curve.png",RocCurveDisplay),("precision_recall_curve.png",PrecisionRecallDisplay)]:
            fig,ax=plt.subplots(figsize=(6,5))
            for name,p in probs.items():display.from_predictions(y_test,p,name=name,ax=ax)
            fig.tight_layout();fig.savefig(figures/fname,dpi=180);plt.close(fig)
    for name,p in probs.items():
        threshold=float(thresholds.get(name,.5))
        matrix=confusion_matrix(y_test,(p>=threshold).astype(int),labels=[0,1])
        slug="_".join(part.lower() for part in f"{feature_set}_{name}".replace("+"," ").split())
        pd.DataFrame(matrix,index=["actual_0","actual_1"],columns=["predicted_0","predicted_1"]).to_csv(figures/f"confusion_{slug}.csv")
        fig,ax=plt.subplots(figsize=(6,5));sns.heatmap(matrix,annot=True,fmt="d",cmap="Blues",ax=ax);ax.set_title(f"{feature_set} / {name}");fig.tight_layout();fig.savefig(figures/f"confusion_{slug}.png",dpi=180);plt.close(fig)
    model=models["Random Forest"]
    pd.Series(model.feature_importances_,index=features).nlargest(25).sort_values().plot.barh(figsize=(10,7));plt.tight_layout();plt.savefig(figures/"feature_importance.png",dpi=180);plt.close()
    return pd.DataFrame(rows),probs

def timeline_plots(data,figures):
    for col,title,name in [("Close","Market price","market_price"),("daily_return","Daily returns","daily_returns"),("volatility_20","Rolling volatility","rolling_volatility"),("drawdown","Drawdown","drawdown"),("crash_probability","Crash probability","crash_probability"),("qtsi","Classical QTSI","qtsi")]:
        if col in data:
            plt.figure(figsize=(12,4));plt.plot(data.Date,data[col]);plt.title(title);plt.tight_layout();plt.savefig(figures/f"{name}.png",dpi=180);plt.close()
