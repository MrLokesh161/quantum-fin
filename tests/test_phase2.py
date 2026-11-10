from datetime import date
from subprocess import CompletedProcess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import backend.app as backend_app
from src.baseline_models import chronological_splits
from src.calibration import (
    TemperatureScaler,
    calibration_metrics,
    chronological_calibration_split,
)
from src.early_warning import evaluate_early_warnings
from src.evaluation import evaluate
from src.ingestion import MarketDataStore, _binance_frames, load_market_feature_frame
from src.next_session import next_session_targets
from src.qtsi import add_qtsi
from src.risk_engine import add_systemic_risk, classify_research_risk
from src.sliding_window import window_descriptors
from src.tda import tda_features
from src.vqc import VQCClassifier
from src.walk_forward import walk_forward_indices


class FixedProbabilityModel:
    feature_importances_ = np.asarray([1.0])

    def predict_proba(self, X):
        probabilities = np.full(len(X), 0.2)
        return np.column_stack([1 - probabilities, probabilities])


class FakeBinanceSession:
    def __init__(self, candles):
        self.candles = candles

    def get(self, *args, **kwargs):
        return FakeBinanceResponse(self.candles)


class FakeBinanceResponse:
    def __init__(self, candles):
        self.candles = candles

    def raise_for_status(self):
        return None

    def json(self):
        return self.candles


class PhaseTwoPipelineTests(unittest.TestCase):
    def test_temperature_scaler_and_metrics_respect_noncontiguous_class_ids(self):
        probabilities = np.asarray([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.1, 0.9],
        ])
        targets = np.asarray([2, 5, 2, 5])
        calibrator = TemperatureScaler().fit(probabilities, targets, [2, 5])
        calibrated = calibrator.predict_proba(probabilities)

        self.assertEqual(calibrator.status_, "temperature-scaled")
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
        self.assertTrue(np.isfinite(calibration_metrics(
            calibrated, targets, classes=[2, 5]
        )["expected_calibration_error"]))

    def test_calibration_and_threshold_partitions_are_chronological_and_purged(self):
        validation = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=20)})

        calibration, threshold = chronological_calibration_split(validation, 3)

        self.assertLess(calibration.Date.max(), threshold.Date.min())
        self.assertEqual((threshold.Date.min() - calibration.Date.max()).days, 4)

    def test_next_session_targets_use_five_downside_bands(self):
        data = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=6),
            "Close": [100.0] * 6,
            "Low": [100.0, 99.6, 99.5, 99.0, 98.5, 97.5],
        })

        targets = next_session_targets(data)

        self.assertEqual(
            targets["next_session_risk_target"].tolist(),
            [0, 1, 2, 3, 4],
        )

    def test_next_session_targets_reject_unsorted_or_duplicate_dates(self):
        data = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-02", "2024-01-01"]),
            "Close": [100.0, 100.0],
            "Low": [100.0, 100.0],
        })

        with self.assertRaisesRegex(ValueError, "sorted"):
            next_session_targets(data)

        data["Date"] = pd.to_datetime(["2024-01-01", "2024-01-01"])
        with self.assertRaisesRegex(ValueError, "unique"):
            next_session_targets(data)

    def test_walk_forward_folds_preserve_chronology_and_purge_gap(self):
        folds = walk_forward_indices(row_count=50, folds=4, purge_gap=3)

        self.assertEqual(len(folds), 4)
        for train_indices, test_indices in folds:
            self.assertLess(train_indices[-1] + 3, test_indices[0])

    def test_vqc_statevector_classifier_returns_binary_probabilities(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(24, 5))
        y = np.asarray([0, 1] * 12)
        model = VQCClassifier(
            n_qubits=3,
            n_layers=1,
            iterations=4,
            batch_size=8,
            max_samples=24,
            random_state=7,
        ).fit(X, y)

        probabilities = model.predict_proba(X[:6])

        self.assertEqual(probabilities.shape, (6, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
        self.assertEqual(model.n_qubits_, 3)

    def test_refresh_and_train_runs_only_when_requested(self):
        class ImmediateThread:
            def __init__(self, target, args, daemon):
                self.target = target
                self.args = args

            def start(self):
                self.target(*self.args)

        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return CompletedProcess(command, 0, "mock output", "")

        with patch.object(backend_app, "Thread", ImmediateThread), patch.object(
            backend_app.subprocess, "run", side_effect=fake_run
        ):
            status = backend_app.start_refresh_and_train(
                backend_app.RefreshRequest(start=date(2020, 1, 1))
            )

        self.assertEqual(status["state"], "completed")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1:3], ["-m", "src.ingestion"])
        self.assertEqual(calls[1][1], "main.py")

    def test_new_research_artifact_endpoints_return_saved_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "next_session_metrics.json"
            walk_forward_path = Path(directory) / "walk_forward.json"
            predictions_path = Path(directory) / "next_session_predictions.csv"
            metrics_path.write_text('{"accuracy": 0.75}', encoding="utf-8")
            walk_forward_path.write_text('[{"model": "Random Forest"}]', encoding="utf-8")
            pd.DataFrame({
                "Date": ["2024-01-02"],
                "predicted_risk_label": ["Low"],
            }).to_csv(predictions_path, index=False)
            with (
                patch.object(backend_app, "NEXT_SESSION_METRICS", metrics_path),
                patch.object(backend_app, "WALK_FORWARD_SUMMARY", walk_forward_path),
                patch.object(
                    backend_app, "NEXT_SESSION_PREDICTIONS", predictions_path
                ),
            ):
                next_metrics = backend_app.next_session_metrics()
                walk_forward = backend_app.walk_forward()
                predictions = backend_app.next_session_predictions()

        self.assertEqual(next_metrics["accuracy"], 0.75)
        self.assertEqual(walk_forward[0]["model"], "Random Forest")
        self.assertEqual(predictions[0]["Date"], "2024-01-02")
        self.assertEqual(predictions[0]["predicted_risk_label"], "Low")

    def test_sqlite_store_and_backward_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MarketDataStore(Path(directory) / "market.sqlite3")
            dates = pd.to_datetime(["2024-01-02", "2024-01-04"])
            market = pd.DataFrame(
                {"Date": dates, "Close": [100.0, 110.0], "Volume": [10.0, 11.0]}
            )
            store.upsert_observations("yahoo", "^IXIC", market)
            store.upsert_observations(
                "yahoo",
                "^GSPC",
                pd.DataFrame({"Date": dates, "Close": [200.0, 220.0]}),
            )
            primary_observations = store.read_observations("yahoo", "^GSPC")
            self.assertNotIn("Value", primary_observations.columns)
            self.assertNotIn("Units", primary_observations.columns)
            store.write_status("yahoo", "^IXIC", "downloaded", 2, "2024-01-04", "test")
            metadata = store.metadata()
            self.assertEqual(metadata.loc[0, "source_url"], "https://finance.yahoo.com/")
            self.assertEqual(metadata.loc[0, "observation_count"], 2)
            events_path = Path(__file__).resolve().parents[1] / "data" / "crisis_events.csv"
            self.assertEqual(store.sync_events(events_path), 9)
            dates_to_align = pd.date_range("2024-01-01", "2024-01-05", freq="D")
            panel = load_market_feature_frame(
                dates_to_align, Path(directory) / "market.sqlite3"
            )
            self.assertTrue(pd.isna(panel.loc[0, "cross_nasdaq_level"]))
            self.assertEqual(panel.loc[1, "cross_nasdaq_level"], 100.0)
            self.assertTrue(pd.isna(panel.loc[2, "cross_nasdaq_return"]))
            self.assertEqual(panel.loc[3, "cross_nasdaq_level"], 110.0)
            self.assertEqual(store.read_observations("yahoo", "^IXIC").shape[0], 2)

    def test_fred_values_are_not_available_before_publication_lag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "market.sqlite3"
            store = MarketDataStore(path)
            store.upsert_observations(
                "fred",
                "CPIAUCSL",
                pd.DataFrame({
                    "Date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                    "Value": [100.0, 101.0],
                }),
            )
            target_dates = pd.to_datetime(["2024-02-14", "2024-02-15", "2024-03-16", "2024-03-17"])
            panel = load_market_feature_frame(target_dates, path)
            self.assertTrue(pd.isna(panel.loc[0, "cross_fred_cpi_level"]))
            self.assertEqual(panel.loc[1, "cross_fred_cpi_level"], 100.0)
            self.assertEqual(panel.loc[2, "cross_fred_cpi_level"], 100.0)
            self.assertEqual(panel.loc[3, "cross_fred_cpi_level"], 101.0)

    def test_binance_daily_ingestion_excludes_unclosed_candle(self):
        utc_today = pd.Timestamp.now(tz="UTC").normalize()
        complete_open = int((utc_today - pd.Timedelta(days=1)).timestamp() * 1000)
        incomplete_open = int(utc_today.timestamp() * 1000)
        candles = []
        for opened_at in (complete_open, incomplete_open):
            candles.append([
                opened_at, "1", "2", "0.5", "1.5", "100",
                opened_at + 86_400_000 - 1, "150", 5, "50", "75", "0",
            ])
        frame = _binance_frames(
            "BTCUSDT",
            (utc_today - pd.Timedelta(days=2)).date().isoformat(),
            (utc_today + pd.Timedelta(days=2)).date().isoformat(),
            FakeBinanceSession(candles),
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame["Date"].iloc[0], utc_today.tz_localize(None))

    def test_risk_scores_are_complementary(self):
        frame = pd.DataFrame({"crash_probability": [0.0, 0.25, 0.9]})
        scores = add_qtsi(add_systemic_risk(frame))
        np.testing.assert_allclose(scores["systemic_risk_score"] + scores["qtsi"], 100.0)
        self.assertEqual(classify_research_risk(0.0, [0.25, 0.5, 0.75]), "Low")
        self.assertEqual(classify_research_risk(0.9, [0.25, 0.5, 0.75]), "Critical")

    def test_chronological_split_purges_boundaries(self):
        data = pd.DataFrame({"value": range(20)})
        train, validation, test = chronological_splits(data, 0.6, 0.2, purge_gap=2)
        self.assertEqual(train["value"].max(), 11)
        self.assertEqual(validation["value"].min(), 14)
        self.assertEqual(validation["value"].max(), 15)
        self.assertEqual(test["value"].min(), 18)

    def test_sliding_windows_include_their_prediction_date_and_keep_clouds_aligned(self):
        data = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "crash_target": [0] * 8,
            "daily_return": [0.01, -0.02, 0.015, -0.01, 0.03, -0.025, 0.02, 0.005],
            "volatility_20": np.linspace(0.1, 0.8, 8),
            "drawdown": np.linspace(0.0, -0.3, 8),
            "momentum_10": np.linspace(-0.2, 0.2, 8),
        })
        windows, clouds = window_descriptors(
            data,
            ["daily_return", "volatility_20", "drawdown", "momentum_10"],
            window=5,
        )
        self.assertEqual(windows["Date"].iloc[0], data["Date"].iloc[4])
        self.assertEqual(len(windows), len(clouds))
        self.assertEqual(clouds.shape, (4, 5, 4))

    def test_metrics_handle_single_class_test_period(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, _ = evaluate(
                {"Random Forest": FixedProbabilityModel()},
                np.asarray([[0.0], [1.0], [2.0]]),
                np.asarray([0, 0, 0]),
                ["feature"],
                Path(directory),
                thresholds={"Random Forest": 0.4},
            )
            self.assertTrue(pd.isna(metrics.loc[0, "roc_auc"]))
            self.assertTrue(pd.isna(metrics.loc[0, "pr_auc"]))
            self.assertEqual(metrics.loc[0, "threshold"], 0.4)
            self.assertTrue((Path(directory) / "roc_curve.png").exists())

    def test_tda_handles_empty_clouds_and_returns_finite_features(self):
        clouds = [
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.empty((0, 2)),
            np.asarray([[0.1, 0.0], [1.1, 0.2], [0.0, 1.2]]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            features, status = tda_features(clouds, Path(directory), stride=2)
        self.assertIn("Ripser", status)
        self.assertEqual(len(features), len(clouds))
        self.assertTrue(np.isfinite(features.to_numpy(dtype=float)).all())

    def test_only_broad_market_events_are_compared_to_equity_crash_target(self):
        predictions = pd.DataFrame(
            {
                "Date": pd.date_range("2022-05-01", periods=30, freq="D"),
                "crash_probability": np.zeros(30),
            }
        )
        events = Path(__file__).resolve().parents[1] / "data" / "crisis_events.csv"
        result = evaluate_early_warnings(predictions, events)
        event_labels = pd.read_csv(events)
        non_equity_names = event_labels.loc[
            ~event_labels["market_relevance"].isin(
                ["broad_market_crash", "equity_market"]
            ),
            "event_name",
        ]
        non_equity = result[result["event"].isin(non_equity_names)]
        self.assertEqual(len(non_equity), 3)
        self.assertTrue(
            non_equity["status"].str.startswith("Not comparable").all()
        )
        self.assertTrue(result["source_reference"].notna().all())


if __name__ == "__main__":
    unittest.main()
