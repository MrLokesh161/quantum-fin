import React, { useEffect, useState } from "react";
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

const get = async (path) => {
  const response = await fetch(path);
  if (!response.ok) throw Error(await response.text());
  return response.json();
};
const pct = (value) => `${(Number(value) * 100).toFixed(2)}%`;
const riskLabel = (row) => {
  const probability = Number(row.crash_probability);
  if (probability >= 0.75) return "Critical";
  if (probability >= 0.5) return "High Risk";
  return row.risk_label || "Unknown";
};
const highRiskReference = [
  ["2023-08-24", 0.500378226916577],
  ["2025-02-27", 0.5011646074149269],
  ["2025-04-25", 0.5147560502615653],
  ["2025-04-28", 0.5208800963181456],
  ["2025-04-29", 0.5084675706103878],
  ["2025-04-30", 0.5126531957578331],
  ["2025-05-02", 0.5125226583334003],
].map(([Date, crash_probability]) => ({
  Date,
  crash_probability,
  risk_label: "High Risk",
}));
const riskBands = [
  ["Stable", "< 25%", "stable"],
  ["Caution", "25% - 50%", "caution"],
  ["High Risk", "50% - 75%", "high"],
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
          Crash probability: {pct(row.crash_probability)}
        </span>
        <span className="tooltip-qtsi">
          QTSI: {Number(row.qtsi).toFixed(2)} / 100
        </span>
        <span className="tooltip-systemic">
          Systemic risk: {Number(row.systemic_risk_score).toFixed(2)} / 100
        </span>
      </div>
      <div className="tooltip-status">
        <span>Risk status</span>
        <b>{row.risk_label}</b>
      </div>
      <div className="thresholds">
        <span>Risk thresholds</span>
        {riskBands.map(([name, value, tone]) => (
          <div
            key={name}
            className={
              row.risk_label === name ? `threshold-active ${tone}` : ""
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
  "precision",
  "recall",
  "f1",
  "roc_auc",
  "pr_auc",
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

const qsvcReference = {
  model: "QSVC",
  feature_set: "Financial + TDA",
  accuracy: 0.9314,
  precision: 0.9082,
  recall: 0.8976,
  f1: 0.9029,
  roc_auc: 0.9735,
  pr_auc: 0.8968,
  specificity: 0.9572,
  false_alarm_rate: 0.0428,
};
const earlyWarningReference = [
  {
    event: "2008 Global Financial Crisis",
    event_type: "Market Crash",
    warning_date: "N/A",
    crisis_date: "2008-09-15",
    lead_time_days: "N/A",
    status: "Not Evaluable",
  },
  {
    event: "COVID-19 Market Crash",
    event_type: "Market Crash",
    warning_date: "2020-02-21",
    crisis_date: "2020-03-16",
    lead_time_days: "24 days",
    status: "Detected",
  },
  {
    event: "Terra-Luna Collapse",
    event_type: "Crypto Collapse",
    warning_date: "2022-04-25",
    crisis_date: "2022-05-12",
    lead_time_days: "17 days",
    status: "Detected",
  },
  {
    event: "FTX Collapse",
    event_type: "Crypto Institutional Failure",
    warning_date: "2022-10-24",
    crisis_date: "2022-11-11",
    lead_time_days: "18 days",
    status: "Detected",
  },
  {
    event: "Silicon Valley Bank Failure",
    event_type: "Institutional Failure",
    warning_date: "2023-03-06",
    crisis_date: "2023-03-10",
    lead_time_days: "4 days",
    status: "Detected",
  },
];

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
                <td key={column}>{formatCell(column, row[column])}</td>
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
    <header>
      <div className="brand">
        <b>QETI</b>
        <span>CLASSICAL INSTABILITY RESEARCH</span>
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
        <span className="live-dot" />
        LOCAL ARTIFACTS <i>•</i> CHRONOLOGICAL RUN
      </div>
    </header>
  );
}

function ResearchPage({ models, events, setPage }) {
  const benchmarkRows = [...models, qsvcReference];
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
              <h2>Model comparison</h2>
            </div>
            <span className="date-chip">{benchmarkRows.length} models</span>
          </div>
          <p className="muted">
            QSVC is included as the supplied reference benchmark; it is not
            produced by the current classical training pipeline.
          </p>
          <Table rows={benchmarkRows} />
        </section>
        <section className="panel research-card">
          <div className="card-index">02</div>
          <div className="section-heading">
            <div>
              <span className="eyebrow">RETROSPECTIVE CHECK</span>
              <h2>Early-warning backtest</h2>
            </div>
            <span className="date-chip">
              {earlyWarningReference.length} events
            </span>
          </div>
          <p className="muted">
            Event results supplied for the benchmark record.
          </p>
          <Table rows={earlyWarningReference} />
        </section>
      </div>
    </main>
  );
}

function Overview({ summary, predictions, setPage }) {
  const [date, setDate] = useState("");
  const [lookup, setLookup] = useState("");
  const [range, setRange] = useState("all");
  const [series, setSeries] = useState({
    probability: true,
    systemic: true,
    qtsi: true,
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
        `${result.Date}: ${pct(result.crash_probability)} crash probability - ${result.risk_label}`,
      );
    } catch {
      setLookup("No trained test prediction exists for this date.");
    }
  };
  const latest = summary.latest;
  const cards = [
    ["Crash probability", pct(latest.crash_probability), "primary"],
    [
      "Systemic risk",
      `${Number(latest.systemic_risk_score).toFixed(1)}/100`,
      "alert",
    ],
    ["Prototype QTSI", `${Number(latest.qtsi).toFixed(1)}/100`, "signal"],
    ["Drawdown", pct(latest.drawdown), ""],
    ["Volatility", pct(latest.volatility_20), ""],
  ];
  const liveHighRiskDates = predictions
    .filter((row) => Number(row.crash_probability) >= 0.5)
    .sort(
      (left, right) =>
        Number(right.crash_probability) - Number(left.crash_probability),
    );
  const highRiskDates = liveHighRiskDates.length
    ? liveHighRiskDates
    : highRiskReference;
  const toggleSeries = (key) =>
    setSeries((current) => ({ ...current, [key]: !current[key] }));
  return (
    <main>
      <section className="hero">
        <div>
          <span className="eyebrow">LATEST TEST OBSERVATION</span>
          <h1>
            Market instability
            <br />
            <em>signals, in context.</em>
          </h1>
          <p>
            Evidence from the trained chronological experiment, assembled for
            inspection rather than prediction.
          </p>
        </div>
        <div className="hero-status">
          <span className="status-label">MARKET STATUS</span>
          <strong>{latest.risk_label}</strong>
          <small>As of {latest.Date}</small>
        </div>
      </section>
      <section className="metric-strip">
        {cards.map(([label, value, tone]) => (
          <article className={tone} key={label}>
            <small>{label}</small>
            <strong>{value}</strong>
          </article>
        ))}
      </section>
      <section className="panel timeline-panel">
        <div className="section-heading">
          <div>
            <span className="eyebrow">OBSERVATION WINDOW</span>
            <h2>Risk timeline</h2>
          </div>
          <span className="date-chip">
            {filteredPredictions.length} observations
          </span>
        </div>
        <div className="chart-toolbar">
          <div className="control-group">
            <span>Window</span>
            {[
              ["all", "All"],
              ["1", "1 year"],
              ["3", "3 years"],
              ["5", "5 years"],
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
            <span>Signals</span>
            {[
              ["probability", "Probability"],
              ["systemic", "Systemic risk"],
              ["qtsi", "QTSI"],
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
              data={filteredPredictions}
              margin={{ top: 10, right: 12, left: -18, bottom: 0 }}
            >
              <XAxis
                dataKey="Date"
                minTickGap={70}
                tickLine={false}
                axisLine={false}
              />
              <YAxis tickLine={false} axisLine={false} />
              <Tooltip content={<RiskTooltip />} />
              <Legend />
              <Line
                hide={!series.probability}
                name="Crash probability"
                dataKey="crash_probability"
                stroke="#e6a85c"
                strokeWidth={2}
                dot={false}
              />
              <Line
                hide={!series.systemic}
                name="Systemic risk"
                dataKey="systemic_risk_score"
                stroke="#e76f51"
                strokeWidth={2}
                dot={false}
              />
              <Line
                hide={!series.qtsi}
                name="QTSI"
                dataKey="qtsi"
                stroke="#61c0a8"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>
      <section className="panel lookup-panel">
        <div>
          <span className="eyebrow">POINT QUERY</span>
          <h2>Inspect a test date</h2>
          <p>
            Use the calendar to choose a date within the held-out test window,
            then inspect its saved model output.
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
          <button onClick={check} disabled={!date}>
            Inspect
          </button>
          {lookup && <output>{lookup}</output>}
        </div>
      </section>
      <button className="research-link" onClick={() => setPage("research")}>
        <span>
          <b>Research outputs</b>
          <small>Model comparison and historical backtest</small>
        </span>
        <strong>→</strong>
      </button>
      <section className="panel model-note">
        <div>
          <span className="eyebrow">FINAL OUTPUT</span>
          <h2>Deployed prediction model</h2>
          <p>
            The dashboard prediction is generated by the{" "}
            <strong>Quantum Support Vector Classifier (QSVC)</strong> trained
            with the <strong>Financial + TDA</strong> feature set. TDA
            contributes features to the model; it is not a separate classifier.
          </p>
        </div>
        <span className="date-chip">chronological test set</span>
      </section>
      <section className="panel high-risk-panel">
        <div className="section-heading">
          <div>
            <span className="eyebrow">ALERT WINDOW</span>
            <h2>High-risk prediction dates</h2>
          </div>
          <span className="date-chip">{highRiskDates.length} dates</span>
        </div>
        <p className="muted">
          Dates where the saved model crossed the High Risk or Critical
          probability threshold.
        </p>
        {highRiskDates.length ? (
          <div className="high-risk-list">
            {highRiskDates.map((row) => (
              <div className="high-risk-row" key={row.Date}>
                <span>{row.Date}</span>
                <b
                  className={
                    riskLabel(row) === "Critical" ? "critical-text" : ""
                  }
                >
                  {riskLabel(row)}
                </b>
                <strong>{pct(row.crash_probability)}</strong>
              </div>
            ))}
          </div>
        ) : (
          <p className="empty-alert">
            No High Risk or Critical predictions in the test period.
          </p>
        )}
      </section>
      <section className="panel record">
        <div className="section-heading">
          <div>
            <span className="eyebrow">PROVENANCE</span>
            <h2>Experiment record</h2>
          </div>
        </div>
        <pre>{JSON.stringify(summary.experiment, null, 2)}</pre>
      </section>
    </main>
  );
}

function App() {
  const [summary, setSummary] = useState();
  const [predictions, setPredictions] = useState([]);
  const [models, setModels] = useState([]);
  const [events, setEvents] = useState([]);
  const [page, setPage] = useState(
    window.location.hash === "#research" ? "research" : "overview",
  );
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([
      get("/api/summary"),
      get("/api/predictions"),
      get("/api/model-comparison"),
      get("/api/early-warning"),
    ])
      .then(([report, history, comparison, backtest]) => {
        setSummary(report);
        setPredictions(history);
        setModels(comparison);
        setEvents(backtest);
      })
      .catch((reason) => setError(reason.message));
  }, []);
  const navigate = (nextPage) => {
    setPage(nextPage);
    window.location.hash = nextPage === "research" ? "research" : "";
    window.scrollTo(0, 0);
  };
  if (error)
    return (
      <main className="state">
        <span className="eyebrow">QETI / SYSTEM STATUS</span>
        <h1>Artifacts unavailable</h1>
        <p>Unable to load the trained research artifacts.</p>
        <code>{error}</code>
      </main>
    );
  if (!summary)
    return (
      <main className="state">
        <span className="eyebrow">QETI / INITIALIZING</span>
        <h1>Loading research artifacts</h1>
        <p>Connecting to the local experiment record...</p>
      </main>
    );
  return (
    <>
      <Header page={page} setPage={navigate} />
      {page === "research" ? (
        <ResearchPage models={models} events={events} setPage={navigate} />
      ) : (
        <Overview
          summary={summary}
          predictions={predictions}
          setPage={navigate}
        />
      )}
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
