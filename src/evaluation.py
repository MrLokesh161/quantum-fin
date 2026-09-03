import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,RocCurveDisplay,PrecisionRecallDisplay

def evaluate(models,x_test,y_test,features,figures,feature_set="Engineered financial"):
    rows=[];probs={}
    for name,model in models.items():
        p=model.predict_proba(x_test)[:,1];pred=(p>=.5).astype(int);probs[name]=p
        tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel();rows.append({"model":name,"feature_set":feature_set,"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,zero_division=0),"recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),"roc_auc":roc_auc_score(y_test,p) if len(set(y_test))>1 else None,"pr_auc":average_precision_score(y_test,p),"specificity":tn/max(tn+fp,1),"false_positive_rate":fp/max(fp+tn,1),"false_alarm_rate":fp/max(fp+tn,1)})
    for fname,display in [("roc_curve.png",RocCurveDisplay),("precision_recall_curve.png",PrecisionRecallDisplay)]:
        fig,ax=plt.subplots(figsize=(6,5))
        for name,p in probs.items():display.from_predictions(y_test,p,name=name,ax=ax)
        fig.tight_layout();fig.savefig(figures/fname,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));sns.heatmap(confusion_matrix(y_test,(probs["Random Forest"]>=.5).astype(int)),annot=True,fmt="d",cmap="Blues",ax=ax);fig.tight_layout();fig.savefig(figures/"confusion_matrix.png",dpi=180);plt.close(fig)
    model=models["Random Forest"]
    pd.Series(model.feature_importances_,index=features).nlargest(25).sort_values().plot.barh(figsize=(10,7));plt.tight_layout();plt.savefig(figures/"feature_importance.png",dpi=180);plt.close()
    return pd.DataFrame(rows),probs

def timeline_plots(data,figures):
    for col,title,name in [("Close","Market price","market_price"),("daily_return","Daily returns","daily_returns"),("volatility_20","Rolling volatility","rolling_volatility"),("drawdown","Drawdown","drawdown"),("crash_probability","Crash probability","crash_probability"),("qtsi","Classical QTSI","qtsi")]:
        if col in data:
            plt.figure(figsize=(12,4));plt.plot(data.Date,data[col]);plt.title(title);plt.tight_layout();plt.savefig(figures/f"{name}.png",dpi=180);plt.close()
