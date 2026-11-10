import React, { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  LineChart,
  Line,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";
import "./styles.css";
import "./pipeline.css";
import "./dashboard.css";

const get = async (path) => {
  const response = await fetch(path);
  if (!response.ok) throw Error(await response.text());
  return response.json();
};
const pct = (value) => `${(Number(value) * 100).toFixed(2)}%`;
const number = (value, digits = 2) =>
  Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "N/A";
const formatDate = (value) =>
  value
    ? new Intl.DateTimeFormat("en", {
        day: "2-digit",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
      }).format(new Date(`${String(value).slice(0, 10)}T00:00:00Z`))
    : "Unavailable";
const riskLabel = (row) => {
  const probability = Number(row.crash_probability);
  if (probability >= 0.75) return "Critical";
  if (probability >= 0.5) return "High";
  return row.risk_classification || "Unknown";
};
const riskBands = [
  ["Low", "< 25%", "stable"],
  ["Moderate", "25% - 50%", "caution"],
  ["High", "50% - 75%", "high"],
  ["Critical", ">= 75%", "critical"],
];

function RiskTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="risk-tooltip">
      <strong>{label}</strong>
      <div className="tooltip-values">
        <span className="tooltip-probability">
          10-session drawdown risk: {pct(row.crash_probability)}
        </span>
        <span className="tooltip-qtsi">
          QTSI: {Number(row.qtsi).toFixed(2)} / 100
        </span>
        <span className="tooltip-systemic">
          Systemic risk: {Number(row.systemic_risk_score).toFixed(2)} / 100
        </span>
        {row.vqc_probability != null && (
          <span className="tooltip-vqc">
            VQC simulator: {pct(row.vqc_probability)}
          </span>
        )}
      </div>
      <div className="tooltip-status">
        <span>Risk status</span>
        <b>{row.risk_classification || riskLabel(row)}</b>
      </div>
      <div className="thresholds">
        <span>Risk thresholds</span>
        {riskBands.map(([name, value, tone]) => (
          <div
            key={name}
            className={
              riskLabel(row) === name ? `threshold-active ${tone}` : ""
            }
          >
            <i className={tone} />
            {name}
            <em>{value}</em>
          </div>
        ))}
      </div>
    </div>
  );
}

const metricColumns = new Set([
  "accuracy",
  "accuracy_at_0_5",
  "balanced_accuracy",
  "precision",
  "recall",
  "f1",
  "f1_at_0_5",
  "macro_f1",
  "roc_auc",
  "pr_auc",
  "roc_auc_mean",
  "pr_auc_mean",
  "specificity",
  "false_positive_rate",
  "false_alarm_rate",
]);
const percentageMetrics = new Set([
  "accuracy",
  "accuracy_at_0_5",
  "balanced_accuracy",
  "precision",
  "recall",
  "f1",
  "f1_at_0_5",
  "macro_f1",
  "roc_auc",
  "pr_auc",
  "roc_auc_mean",
  "pr_auc_mean",
  "specificity",
  "false_positive_rate",
  "false_alarm_rate",
]);
const formatColumn = (column) =>
  column === "roc_auc"
    ? "ROC-AUC"
    : column === "pr_auc"
      ? "PR-AUC"
      : column.replaceAll("_", " ");
const formatCell = (column, value) => {
  if (value == null || value === "") return "N/A";
  if (metricColumns.has(column)) return pct(value);
  return value;
};

function Table({ rows }) {
  if (!rows?.length) return <p className="muted">No records available.</p>;
  const columns = Object.keys(rows[0]);
  return (
    <div className="table">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{formatColumn(column)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>
                  {column === "value" && percentageMetrics.has(row.metric)
                    ? pct(row.value)
                    : formatCell(column, row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Header({ page, setPage }) {
  return (
    <header className="app-header">
      <div className="brand">
        <b className="brand-mark">Q</b>
        <div className="brand-copy">
          <strong>QETI</strong>
          <span>MARKET RISK TERMINAL</span>
        </div>
      </div>
      <nav>
        <button
          className={page === "overview" ? "nav-active" : ""}
          onClick={() => setPage("overview")}
        >
          Overview
        </button>
        <button
          className={page === "research" ? "nav-active" : ""}
          onClick={() => setPage("research")}
        >
          Research outputs
        </button>
      </nav>
      <div className="header-meta">
        <span className="local-indicator" />
        <span>LOCAL RESEARCH</span>
        <i>·</i>
        <span>DAILY DATA</span>
      </div>
    </header>
  );
}

function ResearchPage({
  models,
  events,
  eventLabels,
  nextSessionMetrics,
  walkForwardMetrics,
  setPage,
}) {
  const benchmarkRows = models;
  return (
    <main className="research-page">
      <button className="back-button" onClick={() => setPage("overview")}>
        ← Back to overview
      </button>
      <div className="page-intro">
        <span className="eyebrow">RESEARCH OUTPUTS / 02</span>
        <h1>
          Benchmarks &<br />
          <em>retrospective checks.</em>
        </h1>
        <p>
          Read the trained models and historical warning performance as separate
          evidence sets. Tables scroll within their panels so the research
          record stays legible at every width.
        </p>
      </div>
      <div className="research-grid">
        <section className="panel research-card">
          <div className="card-index">01</div>
          <div className="section-heading">
            <div>
              <span className="eyebrow">BENCHMARKS</span>
              <h2>10-session model comparison</h2>
            </div>
            <span className="date-chip">{benchmarkRows.length} models</span>
          </div>
          <p className="muted">
            Rows contain models evaluated by the chronological experiment.
            Metrics use the chronological held-out test period and
            validation-selected thresholds. VQC runs on a local statevector
            simulator, not quantum hardware.
          </p>
          <Table rows={benchmarkRows} />
        </section>
        <section className="panel research-card">
          <div className="card-index">02</div>
          <div className="section-heading">
            <div>
              <span className="eyebrow">SEPARATE NEXT-SESSION TARGET</span>
              <h2>One-session risk classifier</h2>
            </div>
            <span className="date-chip">
              {nextSessionMetrics?.calibration_method || "Unavailable"}
            </span>
          </div>
          <p className="muted">
            Predicts five downside-risk categories from the next session's low
            relative to the current close. The table shows performance on a
            chronological held-out period; it is not the 10-session crash model.
          </p>
          {nextSessionMetrics ? (
            <Table
              rows={Object.entries(nextSessionMetrics)
                .filter(([key]) =>
                  [
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "log_loss",
                    "brier_score",
                    "expected_calibration_error",
                  ].includes(key),
                )
                .map(([metric, value]) => ({ metric, value }))}
            />
          ) : (
            <p className="muted">Next-session metrics are unavailable.</p>
          )}
          {nextSessionMetrics?.class_boundaries && (
            <Table
              rows={Object.entries(nextSessionMetrics.class_boundaries).map(
                ([risk, boundary]) => ({ risk, boundary }),
              )}
            />
          )}
        </section>
        <section className="panel research-card">
          <div className="card-index">03</div>
          <div className="section-heading">
            <div>
              <span className="eyebrow">EXPANDING-WINDOW VALIDATION</span>
              <h2>Walk-forward model performance</h2>
            </div>
            <span className="date-chip">
              {walkForwardMetrics.length
                ? `${walkForwardMetrics[0].folds} folds`
                : "Unavailable"}
            </span>
          </div>
          <p className="muted">
            Fold-wise chronological estimates use a purge gap and calibration
            fitted within each training window. Crisis-event backtesting uses
            the corresponding out-of-fold Random Forest predictions.
          </p>
          <Table rows={walkForwardMetrics} />
        </section>
        <section className="panel research-card">
          <div className="card-index">04</div>
          <div className="section-heading">
            <div>
              <span className="eyebrow">RETROSPECTIVE CHECK</span>
              <h2>Early-warning backtest</h2>
            </div>
            <span className="date-chip">
              {events.length} events
            </span>
          </div>
          <p className="muted">
            Only broad equity-market crash events are compared with the S&P 500
            drawdown target. Other events are documented but not treated as
            equivalent crashes.
          </p>
          <Table rows={events} />
        </section>
        <section className="panel research-card">
          <div className="card-index">05</div>
          <div className="section-heading">
            <div>
              <span className="eyebrow">EVENT DEFINITIONS</span>
              <h2>Sources and labels</h2>
            </div>
            <span className="date-chip">{eventLabels.length} events</span>
          </div>
          <p className="muted">
            Event dates and relevance labels are based on the cited public
            sources. They are evaluation metadata, not model input features.
          </p>
          <Table rows={eventLabels} />
        </section>
      </div>
    </main>
  );
}

function Overview({ summary, predictions, sourceStatus, sourceMetadata, setPage }) {
  const [date, setDate] = useState("");
  const [lookup, setLookup] = useState("");
  const [range, setRange] = useState("all");
  const [series, setSeries] = useState({
    probability: true,
    systemic: true,
    qtsi: true,
    vqc: true,
  });
  const filteredPredictions =
    range === "all"
      ? predictions
      : predictions.filter(
          (row) =>
            new Date(row.Date) >=
            new Date(
              new Date(predictions.at(-1)?.Date).setFullYear(
                new Date(predictions.at(-1)?.Date).getFullYear() -
                  Number(range),
              ),
            ),
        );
  const check = async () => {
    if (!date) return;
    try {
      const result = await get(`/api/risk/${date}`);
      setLookup(
        `${formatDate(result.Date)} · ${pct(result.crash_probability)} estimated 10-session drawdown risk · ${result.risk_classification || riskLabel(result)}`,
      );
    } catch {
      setLookup("No trained test prediction exists for this date.");
    }
  };
  const latest = summary.latest;
  const horizon = latest.prediction_horizon_sessions || 10;
  const target = Math.abs(Number(latest.drawdown_threshold || -0.05));
  const nextRiskLabel = latest.next_session_risk_label || "Unavailable";
  const nextRiskProbability = Number(latest.next_session_risk_probability);
  const nextRiskDistribution = ["Stable", "Low", "Moderate", "High", "Critical"];
  const risk = Math.max(0, Math.min(1, Number(latest.crash_probability)));
  const riskStatus = riskLabel(latest);
  const sourceCounts = sourceStatus.reduce(
    (counts, row) => {
      const state = row.status === "downloaded" || row.status === "loaded"
        ? "available"
        : row.status === "cached"
          ? "cached"
          : "unavailable";
      counts[state] += 1;
      return counts;
    },
    { available: 0, cached: 0, unavailable: 0 },
  );
  const latestFetch = sourceStatus
    .map((row) => row.fetched_at)
    .filter(Boolean)
    .sort()
    .at(-1);
  const chartPredictions = filteredPredictions.map((row) => ({
    ...row,
    risk_percent: Number(row.crash_probability) * 100,
    vqc_percent: Number(row.vqc_probability) * 100,
  }));
  const highRiskDates = predictions
    .filter((row) => Number(row.crash_probability) >= 0.5)
    .sort(
      (left, right) =>
        Number(right.crash_probability) - Number(left.crash_probability),
    )
    .slice(0, 6);
  const toggleSeries = (key) =>
    setSeries((current) => ({ ...current, [key]: !current[key] }));
  return (
    <main className="dashboard-main">
      <div className="page-heading">
        <div>
          <span className="eyebrow">MARKET OVERVIEW / DAILY RESEARCH BRIEF</span>
          <h1>Market risk overview</h1>
          <p>Current estimate, market context, and source health in one view.</p>
        </div>
        <div className="asof-chip">
          <span>ESTIMATE AS OF</span>
          <strong>{formatDate(latest.Date)}</strong>
        </div>
      </div>

      <section className={`risk-summary-card risk-${riskStatus.toLowerCase()}`}>
        <div className="risk-copy">
          <span className="eyebrow">10-SESSION DRAWDOWN RISK</span>
          <div className="risk-title-row">
            <h2>{riskStatus} risk</h2>
            <span className={`risk-pill risk-pill-${riskStatus.toLowerCase()}`}>
              {riskStatus}
            </span>
          </div>
          <p>
            Estimated risk of a {pct(target)} or greater S&amp;P 500 drawdown
            over the {horizon} trading sessions beginning at the next available
            session. This is separate from the next-session downside category below.
          </p>
          <div className="model-meta">
            <span>{summary.experiment.deployed_model || latest.model}</span>
            <i />
            <span>{summary.experiment.deployed_feature_set || latest.feature_set}</span>
            <i />
            <span>{latest.probability_calibrated ? "Calibrated" : "Uncalibrated estimate"}</span>
            {latest.vqc_probability != null && (
              <>
                <i />
                <span>VQC simulator score {pct(latest.vqc_probability)}</span>
              </>
            )}
          </div>
        </div>
        <div className="risk-gauge" aria-label={`${pct(risk)} estimated drawdown risk`}>
          <svg viewBox="0 0 140 140" role="img">
            <circle className="gauge-track" cx="70" cy="70" r="55" />
            <circle
              className="gauge-value"
              cx="70"
              cy="70"
              r="55"
              strokeDasharray={`${risk * 345.6} 345.6`}
            />
          </svg>
          <div className="gauge-label">
            <strong>{pct(risk)}</strong>
            <span>MODEL SCORE</span>
          </div>
        </div>
      </section>

      <section className="metric-grid" aria-label="Latest market indicators">
        <article className={`metric-card next-session-risk-card next-risk-${nextRiskLabel.toLowerCase()}`}>
          <span className="metric-label">NEXT-SESSION DOWNSIDE CLASS</span>
          <strong>{nextRiskLabel}</strong>
          <small>
            {Number.isFinite(nextRiskProbability)
              ? `${pct(nextRiskProbability)} model confidence in the selected class`
              : "Estimate unavailable"}
          </small>
          <small>Based on next session's low vs. today's close</small>
          <div className="next-risk-probabilities" aria-label="Next-session risk class probabilities">
            {nextRiskDistribution.map((label) => {
              const probability = Number(
                latest[`next_session_probability_${label.toLowerCase()}`],
              );
              return (
                <span
                  className={label === nextRiskLabel ? "probability-active" : ""}
                  key={label}
                >
                  <em>{label}</em>
                  {Number.isFinite(probability) ? pct(probability) : "N/A"}
                </span>
              );
            })}
          </div>
          <small>
            {latest.next_session_risk_calibrated
              ? "Probability-calibrated classifier"
              : "Uncalibrated classifier output"}
          </small>
        </article>
        <article className="metric-card">
          <span className="metric-label">S&amp;P 500 CLOSE</span>
          <strong>${Number(latest.Close).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</strong>
          <small>Latest completed daily bar</small>
        </article>
        <article className="metric-card">
          <span className="metric-label">CURRENT DRAWDOWN</span>
          <strong>{pct(latest.drawdown)}</strong>
          <small>From recent rolling peak</small>
        </article>
        <article className="metric-card">
          <span className="metric-label">20-DAY VOLATILITY</span>
          <strong>{pct(latest.volatility_20)}</strong>
          <small>Annualized estimate</small>
        </article>
        <article className="metric-card metric-card-coral">
          <span className="metric-label">SYSTEMIC RISK SCORE</span>
          <strong>{number(latest.systemic_risk_score, 1)}<small> / 100</small></strong>
          <small>Model-derived indicator</small>
        </article>
        <article className="metric-card metric-card-mint">
          <span className="metric-label">QTSI</span>
          <strong>{number(latest.qtsi, 1)}<small> / 100</small></strong>
          <small>Project-defined complementary score</small>
        </article>
        {latest.vqc_probability != null && (
          <article className="metric-card metric-card-vqc">
            <span className="metric-label">QUANTUM VQC · SIMULATOR</span>
            <strong>{pct(latest.vqc_probability)}</strong>
            <small>
              {summary.experiment.vqc?.config?.n_qubits || 2}-qubit ·{" "}
              {summary.experiment.vqc?.config?.n_layers || 1}-layer circuit ·
              experimental score
            </small>
          </article>
        )}
      </section>

      <section className="dashboard-grid">
        <article className="panel timeline-panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">HISTORICAL TEST SERIES</span>
              <h2>Risk signals over time</h2>
            </div>
            <span className="date-chip">{filteredPredictions.length.toLocaleString()} observations</span>
          </div>
          <p className="panel-description">
            Chronological held-out test predictions. Scores share a 0–100
            scale; the next-session estimate above is a separate latest-data run.
          </p>
          <div className="chart-toolbar">
            <div className="control-group">
              <span>PERIOD</span>
              {[
                ["all", "All"],
                ["1", "1Y"],
                ["3", "3Y"],
                ["5", "5Y"],
              ].map(([value, label]) => (
                <button
                  className={range === value ? "control-active" : ""}
                  key={value}
                  onClick={() => setRange(value)}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="control-group">
              <span>SERIES</span>
              {[
                ["probability", "Drawdown risk"],
                ["systemic", "SRS"],
                ["qtsi", "QTSI"],
                ...(chartPredictions.some((row) => Number.isFinite(row.vqc_percent))
                  ? [["vqc", "VQC simulator"]]
                  : []),
              ].map(([key, label]) => (
                <button
                  className={series[key] ? "control-active" : ""}
                  key={key}
                  onClick={() => toggleSeries(key)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="chart">
            <ResponsiveContainer>
              <LineChart
                data={chartPredictions}
                margin={{ top: 12, right: 12, left: -18, bottom: 0 }}
              >
                <XAxis dataKey="Date" minTickGap={70} tickLine={false} axisLine={false} />
                <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tickFormatter={(value) => `${value}`} tickLine={false} axisLine={false} />
                <Tooltip content={<RiskTooltip />} />
                <Legend />
                <Line hide={!series.probability} name="10-session drawdown risk" dataKey="risk_percent" stroke="#eab46b" strokeWidth={2.5} dot={false} activeDot={{ r: 4 }} />
                <Line hide={!series.systemic} name="Systemic risk score" dataKey="systemic_risk_score" stroke="#e98570" strokeWidth={1.8} dot={false} activeDot={{ r: 4 }} />
                <Line hide={!series.qtsi} name="QTSI" dataKey="qtsi" stroke="#72c4ad" strokeWidth={1.8} dot={false} activeDot={{ r: 4 }} />
                <Line hide={!series.vqc} name="VQC simulator score" dataKey="vqc_percent" stroke="#9c8df5" strokeWidth={1.8} dot={false} activeDot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <aside className="side-column">
          <section className="panel source-health-card">
            <div className="section-heading">
              <div>
                <span className="eyebrow">DATA CONNECTIONS</span>
                <h2>Source health</h2>
              </div>
              <span className={`health-dot ${sourceCounts.unavailable ? "health-warning" : ""}`} />
            </div>
            <p className="panel-description">
              Public daily data, refreshed manually from this dashboard.
            </p>
            <div className="source-counts">
              <div><strong>{sourceCounts.available}</strong><span>Available</span></div>
              <div><strong>{sourceCounts.cached}</strong><span>Cached</span></div>
              <div><strong>{sourceCounts.unavailable}</strong><span>Unavailable</span></div>
            </div>
            <div className="source-latest">
              <span>Latest provider refresh</span>
              <strong>{latestFetch ? new Date(latestFetch).toLocaleString() : "No refresh recorded"}</strong>
            </div>
            <details className="source-details">
              <summary>View {sourceStatus.length} source records</summary>
              {sourceStatus.some((row) => row.status === "unavailable" || row.status === "cached") && (
                <p className="source-warning">
                  Some feeds are cached or unavailable. Review their dates before interpreting this estimate.
                </p>
              )}
              <Table rows={sourceStatus} />
              {sourceMetadata.length > 0 && (
                <details>
                  <summary>Attribution and limitations</summary>
                  <Table rows={sourceMetadata} />
                </details>
              )}
            </details>
          </section>
          <section className="panel interpretation-card">
            <span className="eyebrow">HOW TO READ THIS</span>
            <h2>Research estimate, not a trading signal</h2>
            <p>
              Probability calibration is estimated from earlier validation data;
              it does not guarantee real-world accuracy.
              QTSI is a project-defined indicator; it is not an industry
              benchmark or a guarantee of safety. VQC is evaluated on a
              classical statevector simulator and does not demonstrate
              quantum hardware performance or quantum advantage.
            </p>
            <button className="text-button" onClick={() => setPage("research")}>
              Explore validation and backtests <span>→</span>
            </button>
          </section>
        </aside>
      </section>

      <section className="dashboard-bottom-grid">
        <section className="panel lookup-panel">
          <div>
            <span className="eyebrow">HISTORICAL LOOKUP</span>
            <h2>Inspect a test date</h2>
            <p className="panel-description">
              Query saved estimates inside the held-out test period.
            </p>
          </div>
          <div className="lookup-controls">
            <input
              aria-label="Test date"
              type="date"
              min={predictions[0]?.Date}
              max={predictions.at(-1)?.Date}
              value={date}
              onChange={(event) => {
                setDate(event.target.value);
                setLookup("");
              }}
            />
            <button onClick={check} disabled={!date}>Inspect date</button>
            {lookup && <output>{lookup}</output>}
          </div>
        </section>
        <section className="panel high-risk-panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">BACKTEST REVIEW</span>
              <h2>Highest historical test scores</h2>
            </div>
            <span className="date-chip">Top {highRiskDates.length}</span>
          </div>
          <p className="panel-description">
            Highest risk estimates in the held-out test history; these are not
            current alerts.
          </p>
          {highRiskDates.length ? (
            <div className="high-risk-list">
              {highRiskDates.map((row) => (
                <div className="high-risk-row" key={row.Date}>
                  <span>{formatDate(row.Date)}</span>
                  <b className={riskLabel(row) === "Critical" ? "critical-text" : ""}>
                    {riskLabel(row)}
                  </b>
                  <strong>{pct(row.crash_probability)}</strong>
                </div>
              ))}
            </div>
          ) : (
            <p className="empty-alert">No High Risk or Critical estimates in the test period.</p>
          )}
        </section>
      </section>

      <details className="experiment-details">
        <summary>Experiment configuration and provenance</summary>
        <pre>{JSON.stringify(summary.experiment, null, 2)}</pre>
      </details>
    </main>
  );
}

function RefreshControls({ status, startDate, setStartDate, onStart }) {
  const busy = status.state === "queued" || status.state === "running";
  return (
    <section className="pipeline-controls">
      <div>
        <span className="eyebrow">ON-DEMAND UPDATE</span>
        <p>
          Refresh daily public feeds and rebuild the estimate when you choose.
        </p>
      </div>
      <div className="pipeline-actions">
        <label>
          DATA HISTORY FROM
          <input
            type="date"
            value={startDate}
            onChange={(event) => setStartDate(event.target.value)}
            disabled={busy}
          />
        </label>
        <button onClick={onStart} disabled={busy || !startDate}>
          {busy ? "Update in progress..." : "Refresh data + retrain"}
        </button>
      </div>
      {(busy || status.state === "completed" || status.state === "failed") && (
        <div className={`pipeline-status ${status.state}`} role="status">
          <strong>{status.phase || status.message}</strong>
          {status.error && <pre>{status.error}</pre>}
          {status.output && (
            <details>
              <summary>Show process output</summary>
              <pre>{status.output}</pre>
            </details>
          )}
        </div>
      )}
    </section>
  );
}

function App() {
  const [summary, setSummary] = useState();
  const [predictions, setPredictions] = useState([]);
  const [models, setModels] = useState([]);
  const [events, setEvents] = useState([]);
  const [eventLabels, setEventLabels] = useState([]);
  const [nextSessionMetrics, setNextSessionMetrics] = useState();
  const [walkForwardMetrics, setWalkForwardMetrics] = useState([]);
  const [dataStatus, setDataStatus] = useState({ sources: [], metadata: [] });
  const [page, setPage] = useState(
    window.location.hash === "#research" ? "research" : "overview",
  );
  const [error, setError] = useState("");
  const [startDate, setStartDate] = useState("1990-01-01");
  const [jobStatus, setJobStatus] = useState({
    state: "idle",
    phase: "",
    message: "",
    output: "",
    error: "",
  });
  const refreshArtifacts = useCallback(() => {
    setError("");
    return Promise.all([
      get("/api/summary"),
      get("/api/predictions"),
      get("/api/model-comparison"),
      get("/api/early-warning"),
      get("/api/event-labels"),
      get("/api/data-status"),
      get("/api/next-session-metrics"),
      get("/api/walk-forward"),
    ])
      .then(([report, history, comparison, backtest, labels, sourceData, nextMetrics, walkForward]) => {
        setSummary(report);
        setPredictions(history);
        setModels(comparison);
        setEvents(backtest);
        setEventLabels(labels);
        setDataStatus(sourceData);
        setNextSessionMetrics(nextMetrics);
        setWalkForwardMetrics(walkForward);
      })
      .catch((reason) => setError(reason.message));
  }, []);
  useEffect(() => {
    refreshArtifacts();
    get("/api/refresh-and-train/status")
      .then(setJobStatus)
      .catch((reason) =>
        setJobStatus((current) => ({
          ...current,
          state: "failed",
          message: "Unable to check backend job status.",
          error: reason.message,
        })),
      );
  }, [refreshArtifacts]);
  useEffect(() => {
    if (jobStatus.state !== "queued" && jobStatus.state !== "running") return;
    const timer = window.setInterval(() => {
      get("/api/refresh-and-train/status").then(setJobStatus).catch((reason) =>
        setJobStatus((current) => ({
          ...current,
          state: "failed",
          message: "Unable to check backend job status.",
          error: reason.message,
        })),
      );
    }, 2000);
    return () => window.clearInterval(timer);
  }, [jobStatus.state]);
  useEffect(() => {
    if (jobStatus.state === "completed") refreshArtifacts();
  }, [jobStatus.state, refreshArtifacts]);
  const startRefreshAndTrain = async () => {
    try {
      const response = await fetch("/api/refresh-and-train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start: startDate }),
      });
      if (!response.ok) throw Error(await response.text());
      setJobStatus(await response.json());
    } catch (reason) {
      setJobStatus({
        state: "failed",
        phase: "",
        message: "Could not start refresh and training.",
        output: "",
        error: reason.message,
      });
    }
  };
  const navigate = (nextPage) => {
    setPage(nextPage);
    window.location.hash = nextPage === "research" ? "research" : "";
    window.scrollTo(0, 0);
  };
  const controls = (
    <RefreshControls
      status={jobStatus}
      startDate={startDate}
      setStartDate={setStartDate}
      onStart={startRefreshAndTrain}
    />
  );
  if (error)
    return (
      <>
        <Header page={page} setPage={navigate} />
        {controls}
        <main className="state">
          <span className="eyebrow">QETI / SYSTEM STATUS</span>
          <h1>Artifacts unavailable</h1>
          <p>Use the control above to refresh data and generate them.</p>
          <code>{error}</code>
        </main>
      </>
    );
  if (!summary)
    return (
      <>
        <Header page={page} setPage={navigate} />
        {controls}
        <main className="state">
          <span className="eyebrow">QETI / INITIALIZING</span>
          <h1>Loading research artifacts</h1>
          <p>Connecting to the local experiment record...</p>
        </main>
      </>
    );
  return (
    <>
      <Header page={page} setPage={navigate} />
      {controls}
      {page === "research" ? (
        <ResearchPage
          models={models}
          events={events}
          eventLabels={eventLabels}
          nextSessionMetrics={nextSessionMetrics}
          walkForwardMetrics={walkForwardMetrics}
          setPage={navigate}
        />
      ) : (
        <Overview
          summary={summary}
          predictions={predictions}
          sourceStatus={dataStatus.sources}
          sourceMetadata={dataStatus.metadata}
          setPage={navigate}
        />
      )}
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
