from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from src.vqc import VQCClassifier

def chronological_splits(data, train_ratio, val_ratio, purge_gap=0):
    """Split in time order and purge nearby rows to prevent window/label overlap."""
    if purge_gap < 0:
        raise ValueError("purge_gap cannot be negative.")
    a=int(len(data)*train_ratio);b=int(len(data)*(train_ratio+val_ratio))
    train=data.iloc[:a]
    validation=data.iloc[min(a+purge_gap,b):b]
    test=data.iloc[min(b+purge_gap,len(data)):]
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Chronological split is empty after purging. Reduce the purge gap or "
            "provide more labeled windows."
        )
    return train,validation,test

def train_models(train, test, features, config, model_names=None):
    if train["target"].nunique()<2:
        raise ValueError("Training data must contain both instability classes to fit classifiers.")
    scaler=StandardScaler(); xtr=scaler.fit_transform(train[features]);xte=scaler.transform(test[features]);seed=config["models"]["random_seed"]
    models={"Logistic Regression":LogisticRegression(max_iter=2000,class_weight="balanced",random_state=seed),"Random Forest":RandomForestClassifier(**config["models"]["rf"],class_weight="balanced",random_state=seed,n_jobs=-1),"SVM":SVC(probability=True,class_weight="balanced",random_state=seed),"Gradient Boosting (XGBoost fallback)":GradientBoostingClassifier(random_state=seed)}
    if config["models"]["enable_xgboost"]:
        try:
            from xgboost import XGBClassifier
            models["XGBoost"]=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,subsample=.8,eval_metric="logloss",random_state=seed)
        except ImportError: pass
    if config["models"].get("enable_vqc", False):
        vqc_config = config["models"].get("vqc", {})
        models["VQC (Statevector Simulator)"] = VQCClassifier(
            n_qubits=int(vqc_config.get("n_qubits", 4)),
            n_layers=int(vqc_config.get("n_layers", 2)),
            iterations=int(vqc_config.get("iterations", 30)),
            batch_size=int(vqc_config.get("batch_size", 64)),
            max_samples=int(vqc_config.get("max_samples", 512)),
            learning_rate=float(vqc_config.get("learning_rate", 0.15)),
            perturbation=float(vqc_config.get("perturbation", 0.1)),
            random_state=seed,
        )
    if model_names is not None:
        missing_models = set(model_names).difference(models)
        if missing_models:
            raise ValueError(f"Requested models are unavailable: {sorted(missing_models)}")
        models = {name: models[name] for name in model_names}
    for model in models.values(): model.fit(xtr,train.target)
    return models,scaler,xte,test.target.to_numpy()
