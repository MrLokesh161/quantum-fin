"""Run the reproducible QETI financial instability research pipeline."""
from __future__ import annotations

import json
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.baseline_models import chronological_splits, train_models
from src.calibration import (
    chronological_calibration_split,
    fit_model_calibrators,
)
from src.data_loader import dataset_summary, discover_dataset, load_csv
from src.early_warning import evaluate_early_warnings
from src.evaluation import evaluate, timeline_plots, tune_thresholds
from src.feature_engineering import build_features
from src.feature_fusion import build_feature_sets
from src.ingestion import MarketDataStore, load_market_feature_frame
from src.labeling import create_labels
from src.next_session import (
    RISK_LABELS,
    fit_next_session_model,
    next_session_metrics,
    next_session_targets,
)
from src.preprocessing import clean_data
from src.qtsi import add_qtsi
from src.risk_engine import add_risk_predictions, add_systemic_risk
from src.sliding_window import window_descriptors
from src.tda import tda_features
from src.utils import ROOT, load_config, prepare_outputs, set_seed
from src.vqc import VQCClassifier
from src.walk_forward import walk_forward_evaluate

VQC_NAME = "VQC (Statevector Simulator)"


def _json_safe_records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def main() -> None:
    config = load_config()
    prepare_outputs(config)
    set_seed(config["models"]["random_seed"])
    outputs = {key: ROOT / value for key, value in config["outputs"].items()}
    figures = outputs["figures"]

    dataset_path, is_demo = discover_dataset(
        ROOT, config["data"]["dataset_dir"], config["data"]["primary_dataset"]
    )
    raw = load_csv(dataset_path, config["data"]["date_column"])
    db_path = ROOT / config["data"].get("database", "data/qeti.sqlite3")
    store = MarketDataStore(db_path)
    downloaded_primary = store.read_observations("yahoo", "^GSPC")
    if not downloaded_primary.empty and downloaded_primary["Close"].notna().any():
        raw = downloaded_primary.dropna(subset=["Close"]).reset_index(drop=True)
        summary_path = "SQLite: Yahoo Finance ^GSPC"
    else:
        summary_path = str(dataset_path.relative_to(ROOT))
    summary = dataset_summary(raw, config["data"]["date_column"])
    summary["dataset_path"] = summary_path
    summary["demo_data"] = is_demo

    clean, outliers = clean_data(raw, config["data"]["date_column"])
    clean.to_csv(outputs["root"] / "cleaned_dataset.csv", index=False)
    outliers.to_csv(outputs["reports"] / "iqr_outlier_report.csv", index=False)

    market_features = build_features(clean, config["data"]["close_column"])
    event_path = ROOT / "data" / "crisis_events.csv"
    if event_path.exists():
        store.sync_events(event_path)
    if db_path.exists():
        external = load_market_feature_frame(market_features["Date"], db_path)
        external_columns = [column for column in external.columns if column != "Date"]
        if external_columns:
            market_features = market_features.merge(external, on="Date", how="left")
            missing_columns = []
            for column in external_columns:
                missing_column = f"{column}_missing"
                if market_features[column].isna().any():
                    market_features[missing_column] = market_features[column].isna().astype(int)
                    market_features[column] = market_features[column].fillna(0.0)
                    missing_columns.append(missing_column)
            external_columns.extend(missing_columns)
    else:
        external_columns = []

    engineered = create_labels(
        market_features,
        config["data"]["close_column"],
        config["features"]["horizon_days"],
        config["features"]["crash_threshold"],
        config["features"]["warning_threshold"],
    ).dropna().reset_index(drop=True)
    engineered.to_csv(outputs["root"] / "engineered_dataset.csv", index=False)

    excluded = {"future_drawdown", "crash_target", "risk_level_actual"}
    base_features = [
        column for column in engineered.select_dtypes("number").columns
        if column not in excluded
    ]
    tda_state_features = [
        column for column in config["models"].get("tda_state_features", [])
        if column in base_features
    ]
    if len(tda_state_features) < 2:
        tda_state_features = [
            column for column in ("daily_return", "volatility_20", "drawdown", "momentum_10")
            if column in base_features
        ]
    windows, clouds = window_descriptors(
        engineered,
        base_features,
        config["features"]["window_size"],
        point_cloud_columns=tda_state_features,
    )
    tda_status = "Disabled by configuration"
    if config["models"]["enable_tda"]:
        tda, tda_status = tda_features(
            clouds, figures, config["models"]["tda_stride"]
        )
        if not tda.empty:
            windows = pd.concat([windows, tda.reset_index(drop=True)], axis=1)
            windows = windows.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    windows.to_csv(outputs["root"] / "model_windows.csv", index=False)
    tda_columns = [column for column in windows if column.startswith("tda_")]
    if tda_columns:
        windows[["Date", "target", *tda_columns]].to_csv(
            outputs["root"] / "topological_features.csv", index=False
        )

    feature_sets = build_feature_sets(windows)
    purge_gap = int(config["features"].get(
        "purge_gap",
        config["features"]["window_size"] + config["features"]["horizon_days"],
    ))
    train, validation, test = chronological_splits(
        windows,
        config["splits"]["train"],
        config["splits"]["validation"],
        purge_gap=purge_gap,
    )
    calibration_rows, threshold_rows = chronological_calibration_split(
        validation, purge_gap
    )

    metric_frames = []
    runs = {}
    for feature_set, feature_columns in feature_sets.items():
        started = time.perf_counter()
        models, scaler, x_test, y_test = train_models(
            train, test, feature_columns, config
        )
        calibrators = fit_model_calibrators(
            models, scaler, calibration_rows, feature_columns
        )
        thresholds = tune_thresholds(
            models, scaler, threshold_rows, feature_columns, calibrators
        )
        metrics, probabilities = evaluate(
            models,
            x_test,
            y_test,
            feature_columns,
            figures,
            feature_set,
            thresholds,
            calibrators,
        )
        metrics["runtime_seconds"] = time.perf_counter() - started
        metric_frames.append(metrics)
        runs[feature_set] = {
            "models": models,
            "scaler": scaler,
            "features": feature_columns,
            "probabilities": probabilities,
            "thresholds": thresholds,
            "calibrators": calibrators,
        }

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(outputs["metrics"] / "model_comparison.csv", index=False)
    ax = metrics.pivot_table(
        index="model", columns="feature_set", values="pr_auc"
    ).plot.bar(figsize=(11, 5))
    ax.set_ylabel("PR-AUC")
    ax.set_title("Chronological held-out model comparison")
    plt.tight_layout()
    plt.savefig(figures / "model_comparison.png", dpi=180)
    plt.close()

    deployed_set = (
        "Financial + TDA" if "Financial + TDA" in runs else "Engineered financial"
    )
    deployed = runs[deployed_set]
    random_forest = deployed["models"]["Random Forest"]
    random_forest_threshold = deployed["thresholds"]["Random Forest"]
    random_forest_calibrator = deployed["calibrators"]["Random Forest"]
    walk_forward_predictions, walk_forward_metrics = walk_forward_evaluate(
        windows,
        deployed["features"],
        config,
        purge_gap,
        int(config["validation"].get("walk_forward_folds", 5)),
    )
    walk_forward_metrics["feature_set"] = deployed_set
    walk_forward_predictions.to_csv(
        outputs["predictions"] / "walk_forward_predictions.csv", index=False
    )
    walk_forward_metrics.to_csv(
        outputs["metrics"] / "walk_forward_metrics.csv", index=False
    )
    walk_forward_summary = (
        walk_forward_metrics.groupby("model")
        .agg(
            folds=("fold", "nunique"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            brier_score_mean=("brier_score", "mean"),
            expected_calibration_error_mean=("expected_calibration_error", "mean"),
        )
        .reset_index()
    )
    walk_forward_summary.to_csv(
        outputs["metrics"] / "walk_forward_summary.csv", index=False
    )
    (outputs["reports"] / "walk_forward_summary.json").write_text(
        json.dumps(_json_safe_records(walk_forward_summary), indent=2),
        encoding="utf-8",
    )

    next_targets = next_session_targets(market_features)
    next_session_data = windows.merge(next_targets, on="Date", how="inner")
    next_session_data = next_session_data.dropna(
        subset=[*deployed["features"], "next_session_risk_target"]
    ).reset_index(drop=True)
    next_train, next_validation, next_test = chronological_splits(
        next_session_data,
        config["splits"]["train"],
        config["splits"]["validation"],
        purge_gap=purge_gap,
    )
    next_model, next_scaler, next_calibrator, next_probabilities = fit_next_session_model(
        next_train,
        next_validation,
        next_test,
        deployed["features"],
        config["models"]["random_seed"],
        config["models"]["rf"],
    )
    next_classes = next_model.classes_.astype(int)
    next_test_actual = next_test["next_session_risk_target"].to_numpy(dtype=int)
    next_metrics = next_session_metrics(
        next_test_actual, next_probabilities, next_classes
    )
    next_predicted_indices = next_probabilities.argmax(axis=1)
    next_predicted = next_classes[next_predicted_indices]
    next_predictions = pd.DataFrame({
        "Date": next_test["Date"].to_numpy(),
        "actual_risk_class": next_test_actual,
        "actual_risk_label": [RISK_LABELS[value] for value in next_test_actual],
        "predicted_risk_class": next_predicted,
        "predicted_risk_label": [RISK_LABELS[value] for value in next_predicted],
        "next_session_max_drawdown": next_test[
            "next_session_max_drawdown"
        ].to_numpy(),
    })
    for column, risk_class in enumerate(next_classes):
        next_predictions[
            f"probability_{RISK_LABELS[int(risk_class)].lower()}"
        ] = next_probabilities[:, column]
    next_predictions.to_csv(
        outputs["predictions"] / "next_session_risk_predictions.csv", index=False
    )
    (outputs["reports"] / "next_session_risk_metrics.json").write_text(
        json.dumps({
            **next_metrics,
            "test_period": [
                str(next_test["Date"].min().date()),
                str(next_test["Date"].max().date()),
            ],
            "class_labels": list(RISK_LABELS),
            "class_boundaries": {
                "Stable": "drawdown > -0.5%",
                "Low": "-1.0% < drawdown <= -0.5%",
                "Moderate": "-1.5% < drawdown <= -1.0%",
                "High": "-2.5% < drawdown <= -1.5%",
                "Critical": "2.5% or more below prior close",
            },
            "calibration_method": next_calibrator.status_,
        }, indent=2),
        encoding="utf-8",
    )
    prediction = test[["Date"]].copy().reset_index(drop=True)
    prediction["actual"] = test["target"].to_numpy()
    prediction["crash_probability"] = deployed["probabilities"]["Random Forest"]
    if VQC_NAME in deployed["probabilities"]:
        prediction["vqc_probability"] = deployed["probabilities"][VQC_NAME]
        prediction["vqc_threshold"] = deployed["thresholds"][VQC_NAME]
    prediction["prediction_horizon_sessions"] = int(config["features"]["horizon_days"])
    prediction["drawdown_threshold"] = float(config["features"]["crash_threshold"])
    prediction = prediction.merge(
        engineered[["Date", "Close", "volatility_20", "drawdown"]],
        on="Date",
        how="left",
    )
    if tda_columns:
        prediction = prediction.merge(
            test[["Date", *tda_columns]], on="Date", how="left"
        )
    prediction = add_qtsi(
        add_systemic_risk(
            add_risk_predictions(prediction, config["risk_thresholds"]),
        )
    )
    prediction.to_csv(outputs["predictions"] / "predictions.csv", index=False)
    prediction.to_csv(outputs["predictions"] / "risk_predictions.csv", index=False)
    timeline_plots(prediction, figures)
    prediction.plot(
        x="Date",
        y=["crash_probability", "systemic_risk_score", "qtsi"],
        figsize=(12, 5),
        title="Risk probability, SRS and QTSI",
    )
    plt.tight_layout()
    plt.savefig(figures / "risk_timeline.png", dpi=180)
    plt.close()
    if tda_columns:
        prediction.plot(
            x="Date", y=tda_columns, figsize=(12, 5), title="Topological features"
        )
        plt.tight_layout()
        plt.savefig(figures / "topological_features.png", dpi=180)
        plt.close()

    early = evaluate_early_warnings(
        walk_forward_predictions,
        event_path,
        threshold=float(config["risk_thresholds"][1]),
    )
    early.to_csv(outputs["reports"] / "early_warning_performance.csv", index=False)
    early.to_csv(outputs["metrics"] / "early_warning_results.csv", index=False)

    # Current estimate is generated on the latest completed market-data row using
    # the training-only transformer/model; no future target label is needed.
    live_data = market_features.copy()
    live_data["crash_target"] = 0
    live_windows, live_clouds = window_descriptors(
        live_data,
        base_features,
        config["features"]["window_size"],
        point_cloud_columns=tda_state_features,
    )
    if live_windows.empty:
        raise ValueError("No complete latest feature window is available for current risk.")
    current_window = live_windows.tail(1).reset_index(drop=True)
    if tda_columns and len(live_clouds):
        current_tda, _ = tda_features([live_clouds[-1]], figures, stride=1)
        if not current_tda.empty:
            current_window = pd.concat(
                [current_window, current_tda.reset_index(drop=True)], axis=1
            )
    current_window = current_window.reindex(columns=deployed["features"])
    if current_window.isna().any().any() or not np.isfinite(current_window.to_numpy(dtype=float)).all():
        raise ValueError("Latest feature window is incomplete or non-finite; current risk was not generated.")
    current_x = deployed["scaler"].transform(current_window)
    current_raw_probability = random_forest.predict_proba(current_x)
    current_probability = float(
        random_forest_calibrator.predict_proba(current_raw_probability)[0, 1]
    )
    vqc_probability = None
    if VQC_NAME in deployed["models"]:
        vqc_raw_probability = deployed["models"][VQC_NAME].predict_proba(current_x)
        vqc_probability = float(
            deployed["calibrators"][VQC_NAME].predict_proba(
                vqc_raw_probability
            )[0, 1]
        )
    next_current_x = next_scaler.transform(current_window[deployed["features"]])
    next_raw_probability = next_model.predict_proba(next_current_x)
    next_current_probabilities = next_calibrator.predict_proba(next_raw_probability)[0]
    next_current_index = int(next_current_probabilities.argmax())
    next_current_class = int(next_classes[next_current_index])
    current_date = pd.Timestamp(live_windows["Date"].iloc[-1])
    current = pd.DataFrame({
        "Date": [current_date],
        "crash_probability": [current_probability],
        "vqc_probability": [vqc_probability],
        "next_session_risk_class": [next_current_class],
        "next_session_risk_label": [RISK_LABELS[next_current_class]],
        "next_session_risk_probability": [
            float(next_current_probabilities[next_current_index])
        ],
    }).merge(
        market_features[["Date", "Close", "volatility_20", "drawdown"]],
        on="Date",
        how="left",
    )
    current = add_qtsi(
        add_systemic_risk(
            add_risk_predictions(current, config["risk_thresholds"]),
        )
    )
    current["model"] = "Random Forest"
    current["feature_set"] = deployed_set
    current["threshold"] = random_forest_threshold
    current["probability_calibrated"] = (
        random_forest_calibrator.status_ == "temperature-scaled"
    )
    current["vqc_probability_calibrated"] = (
        deployed["calibrators"][VQC_NAME].status_ == "temperature-scaled"
        if VQC_NAME in deployed["calibrators"]
        else False
    )
    current["next_session_risk_calibrated"] = (
        next_calibrator.status_ == "temperature-scaled"
    )
    current["prediction_scope"] = "Next available S&P 500 session; >=5% drawdown risk over the following 10 trading sessions."
    current["prediction_as_of_date"] = current_date.date().isoformat()
    current["prediction_start"] = "Next available S&P 500 trading session"
    current["prediction_horizon_sessions"] = int(config["features"]["horizon_days"])
    current["drawdown_threshold"] = float(config["features"]["crash_threshold"])
    for column, risk_class in enumerate(next_classes):
        current[
            f"next_session_probability_{RISK_LABELS[int(risk_class)].lower()}"
        ] = float(next_current_probabilities[column])
    current.to_csv(outputs["predictions"] / "current_prediction.csv", index=False)
    current.to_csv(outputs["predictions"] / "next_session_prediction.csv", index=False)

    joblib.dump(
        {
            "model": random_forest,
            "scaler": deployed["scaler"],
            "features": deployed["features"],
            "threshold": random_forest_threshold,
            "feature_set": deployed_set,
            "probability_calibrated": current["probability_calibrated"].iloc[0],
            "calibrator": random_forest_calibrator,
        },
        outputs["models"] / "random_forest_tda.joblib",
    )
    if VQC_NAME in deployed["models"]:
        joblib.dump(
            {
                "model": deployed["models"][VQC_NAME],
                "scaler": deployed["scaler"],
                "features": deployed["features"],
                "threshold": deployed["thresholds"][VQC_NAME],
                "feature_set": deployed_set,
                "calibrator": deployed["calibrators"][VQC_NAME],
                "simulator": "NumPy statevector",
                "physical_quantum_hardware": False,
            },
            outputs["models"] / "vqc_statevector_tda.joblib",
        )
    joblib.dump(
        {
            "model": next_model,
            "scaler": next_scaler,
            "calibrator": next_calibrator,
            "features": deployed["features"],
            "classes": RISK_LABELS,
            "feature_set": deployed_set,
            "risk_cutoffs": [-0.005, -0.01, -0.015, -0.025],
        },
        outputs["models"] / "next_session_risk.joblib",
    )
    if db_path.exists():
        source_status = MarketDataStore(db_path).statuses()
        dataset_metadata = MarketDataStore(db_path).metadata()
    else:
        source_status = pd.DataFrame(columns=[
            "source", "symbol", "status", "row_count", "latest_observation",
            "fetched_at", "message",
        ])
        dataset_metadata = pd.DataFrame()
    source_status.to_csv(outputs["reports"] / "data_source_status.csv", index=False)
    if not dataset_metadata.empty:
        dataset_metadata.to_csv(outputs["reports"] / "dataset_metadata.csv", index=False)

    report = {
        "dataset": summary,
        "external_feature_count": len(external_columns),
        "external_features": external_columns,
        "external_missingness": "External values unavailable before first observation or during warm-up are explicitly flagged with companion *_missing features and neutral-zero encoded; raw source observations remain unchanged.",
        "feature_sets": {name: len(columns) for name, columns in feature_sets.items()},
        "target_definition": (
            f"At least {abs(float(config['features']['crash_threshold'])):.1%} "
            f"drawdown within the next {int(config['features']['horizon_days'])} "
            "trading sessions."
        ),
        "target_horizon_sessions": int(config["features"]["horizon_days"]),
        "tda_feature_count": len(tda_columns),
        "training_period": [str(train.Date.min().date()), str(train.Date.max().date())],
        "validation_period": [str(validation.Date.min().date()), str(validation.Date.max().date())],
        "testing_period": [str(test.Date.min().date()), str(test.Date.max().date())],
        "purge_gap_rows": purge_gap,
        "tda_status": tda_status,
        "deployed_model": "Random Forest",
        "deployed_feature_set": deployed_set,
        "current_prediction_date": current_date.date().isoformat(),
        "next_session_prediction": {
            "as_of_date": current_date.date().isoformat(),
            "prediction_start": "Next available S&P 500 trading session",
            "horizon_sessions": int(config["features"]["horizon_days"]),
            "drawdown_threshold": float(config["features"]["crash_threshold"]),
            "probability": current_probability,
        },
        "one_session_risk_prediction": {
            "target": "Classify the next session's intraday low versus the current close",
            "as_of_date": current_date.date().isoformat(),
            "class_labels": list(RISK_LABELS),
            "predicted_class": RISK_LABELS[next_current_class],
            "probabilities": {
                RISK_LABELS[int(risk_class)]: float(next_current_probabilities[column])
                for column, risk_class in enumerate(next_classes)
            },
            "test_metrics": next_metrics,
            "test_period": [
                str(next_test["Date"].min().date()),
                str(next_test["Date"].max().date()),
            ],
            "calibration_method": next_calibrator.status_,
        },
        "probability_calibration": {
            "method": "Chronological temperature scaling",
            "calibration_partition": "First half of validation data",
            "threshold_partition": "Second half of validation data after purge gap",
            "random_forest_status": random_forest_calibrator.status_,
            "random_forest_temperature": random_forest_calibrator.temperature_,
            "random_forest_test_metrics": metrics.loc[
                (metrics["model"] == "Random Forest")
                & (metrics["feature_set"] == deployed_set),
                [
                    "brier_score",
                    "log_loss",
                    "expected_calibration_error",
                ],
            ].iloc[0].to_dict(),
        },
        "walk_forward_validation": {
            "fold_count": int(config["validation"].get("walk_forward_folds", 5)),
            "metrics": _json_safe_records(walk_forward_summary),
            "out_of_fold_prediction_count": len(walk_forward_predictions),
            "event_backtest_uses_out_of_fold_predictions": True,
        },
        "current_probability_calibrated": (
            random_forest_calibrator.status_ == "temperature-scaled"
        ),
        "event_evaluation": "Only broad-equity-market crash events are compared with the S&P 500 drawdown target; crypto and institution-specific events remain documented but are not scored as equivalent crashes.",
        "data_sources": _json_safe_records(source_status),
        "quantum_status": (
            "VQC evaluated with a local NumPy statevector simulator. "
            "No physical quantum hardware or quantum-advantage claim."
            if VQC_NAME in deployed["models"]
            else "VQC disabled by configuration."
        ),
        "vqc": {
            "model": VQC_NAME,
            "feature_set": deployed_set,
            "next_session_score": vqc_probability,
            "probability_calibrated": (
                deployed["calibrators"][VQC_NAME].status_ == "temperature-scaled"
                if VQC_NAME in deployed["calibrators"]
                else False
            ),
            "simulator": "NumPy statevector",
            "physical_quantum_hardware": False,
            "config": config["models"].get("vqc", {}),
        },
        "limitations": "Research prototype only. Calibration is statistical and out-of-sample, not proof of real-world probability accuracy. Historical event counts remain too small for strong crisis-generalization claims.",
    }
    (outputs["reports"] / "experiment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (outputs["reports"] / "experiment_report.md").write_text(
        "# Experiment Report\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )

    print(
        f"DATASET: {summary['rows']} rows, {summary['columns']} columns\n"
        f"EXTERNAL FEATURES: {len(external_columns)}\n"
        f"SLIDING WINDOWS: {len(windows)} (purge gap: {purge_gap})\n"
        f"TDA: {tda_status}\n"
        f"MODELS: {len(metrics)} model/feature-set evaluations\n"
        f"CURRENT RISK: {current_probability:.3f} ({current_date.date()})\n"
        f"DATA SOURCES: {len(source_status)} status records\n"
        f"VQC: {vqc_probability if vqc_probability is not None else 'Disabled'} (local statevector simulator; no hardware)\n"
        f"Outputs: {outputs['root']}"
    )


if __name__ == "__main__":
    main()
