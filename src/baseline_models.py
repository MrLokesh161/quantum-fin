from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC

def chronological_splits(data, train_ratio, val_ratio):
    a=int(len(data)*train_ratio);b=int(len(data)*(train_ratio+val_ratio));return data.iloc[:a],data.iloc[a:b],data.iloc[b:]

def train_models(train, validation, test, features, config):
    scaler=StandardScaler(); xtr=scaler.fit_transform(train[features]);xte=scaler.transform(test[features]);seed=config["models"]["random_seed"]
    models={"Logistic Regression":LogisticRegression(max_iter=2000,class_weight="balanced",random_state=seed),"Random Forest":RandomForestClassifier(**config["models"]["rf"],class_weight="balanced",random_state=seed,n_jobs=-1),"SVM":SVC(probability=True,class_weight="balanced",random_state=seed),"Gradient Boosting (XGBoost fallback)":GradientBoostingClassifier(random_state=seed)}
    if config["models"]["enable_xgboost"]:
        try:
            from xgboost import XGBClassifier
            models["XGBoost"]=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,subsample=.8,eval_metric="logloss",random_state=seed)
        except ImportError: pass
    for model in models.values(): model.fit(xtr,train.target)
    return models,scaler,xte,test.target.to_numpy()
