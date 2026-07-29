"""Run the complete Phase 1 collapse-risk baseline pipeline."""
from pathlib import Path
import joblib
import numpy as np

from config import PipelineConfig
from preprocessing.data_loader import dataset_report, load_and_merge_datasets
from preprocessing.cleaning import clean_market_data
from feature_engineering.indicators import engineer_features
from feature_engineering.windows import create_sliding_windows
from models.baseline import fit_baseline
from evaluation.reporting import evaluate_model, generate_eda_plots, generate_evaluation_plots, save_metrics


def main() -> None:
    config = PipelineConfig()
    config.output_dir.mkdir(exist_ok=True); config.visualizations_dir.mkdir(exist_ok=True)
    raw, _ = load_and_merge_datasets(config.dataset_dir)
    (config.output_dir / "dataset_report.txt").write_text(dataset_report(raw), encoding="utf-8")
    cleaned, outlier_report = clean_market_data(raw)
    cleaned.to_csv(config.cleaned_path, index=False); outlier_report.to_csv(config.output_dir / "iqr_outlier_report.csv", index=False)
    engineered, close_column = engineer_features(cleaned, config.primary_market, config.forecast_horizon, config.crash_threshold)
    engineered.to_csv(config.engineered_path, index=False)
    generate_eda_plots(engineered, close_column, config.visualizations_dir)
    excluded = {"Date", "crash_target", "future_max_drawdown"}
    feature_columns = [c for c in engineered.select_dtypes(include="number").columns if c not in excluded]
    X_windows, y, dates = create_sliding_windows(engineered, feature_columns, "crash_target", config.window_size)
    model, scaler, _, X_test, _, y_test, split = fit_baseline(X_windows, y, config.test_size, config.scaler, config.random_state)
    metrics, predictions, probabilities = evaluate_model(model, X_test, y_test)
    save_metrics({**metrics, "best_parameters": model.best_params_, "best_cv_f1": model.best_score_}, config.output_dir / "evaluation_metrics.json")
    window_names = [f"t-{config.window_size - i}_{feature}" for i in range(config.window_size) for feature in feature_columns]
    generate_evaluation_plots(model.best_estimator_, y_test, probabilities, window_names, config.visualizations_dir)
    joblib.dump({"model": model.best_estimator_, "scaler": scaler, "features": feature_columns, "window_size": config.window_size}, config.output_dir / "random_forest_baseline.joblib")
    test_dates = dates.iloc[split:].to_numpy()
    import pandas as pd
    pd.DataFrame({"Date": test_dates, "actual": y_test, "prediction": predictions, "high_risk_probability": probabilities}).to_csv(config.output_dir / "predictions.csv", index=False)
    report = """Phase 1 Progress Report\n=======================\n✔ Dataset Loaded\n✔ Data Preprocessing Completed\n✔ Feature Engineering Completed\n✔ Sliding Window Generation Completed\n✔ Base Model Trained\n✔ Evaluation Completed\n⏳ Persistent Homology Pending\n⏳ Quantum Model Pending\n"""
    (config.output_dir / "progress_report.txt").write_text(report, encoding="utf-8")
    # Some Windows terminals still default to cp1252 and cannot render symbols.
    print(report.encode("ascii", "replace").decode("ascii")); print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
