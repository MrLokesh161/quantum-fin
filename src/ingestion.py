"""Free public financial-data ingestion and local SQLite persistence.

Yahoo Finance, Binance's public market-data API, and FRED's public graph CSV
endpoint are used without credentials. Provider outages are recorded explicitly;
cached observations are retained and never replaced with synthetic values.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "qeti.sqlite3"
RAW_DIR = ROOT / "data" / "raw"
YAHOO_SYMBOLS = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite",
    "^DJI": "Dow Jones Industrial Average",
    "^NSEI": "NIFTY 50",
    "^VIX": "CBOE Volatility Index",
}
BINANCE_SYMBOLS = {"BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum"}
FRED_SERIES = {
    "DGS10": "10-year Treasury yield",
    "DGS2": "2-year Treasury yield",
    "CPIAUCSL": "Consumer Price Index",
    "UNRATE": "Unemployment rate",
    "FEDFUNDS": "Federal funds effective rate",
}
FRED_PUBLICATION_LAG_DAYS = {
    "DGS10": 1,
    "DGS2": 1,
    "CPIAUCSL": 45,
    "UNRATE": 45,
    "FEDFUNDS": 40,
}
SOURCE_ATTRIBUTION = {
    "yahoo": {
        "url": "https://finance.yahoo.com/",
        "attribution": "Yahoo Finance historical market data accessed with yfinance.",
        "frequency": "Daily; delayed/end-of-day availability, not real-time.",
        "limitations": "Unofficial yfinance access; symbol availability and provider terms may change.",
    },
    "binance": {
        "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints",
        "attribution": "Binance public Spot API daily klines; no API key is used.",
        "frequency": "Daily UTC candles.",
        "limitations": "Only supported symbols and available exchange history are returned.",
    },
    "fred": {
        "url": "https://fred.stlouisfed.org/docs/api/fred/",
        "attribution": "Federal Reserve Bank of St. Louis FRED public graph CSV endpoint.",
        "frequency": "Series-specific (daily, monthly, or other); aligned after a conservative publication lag.",
        "limitations": "Public downloads may be rate-limited; observations may be revised and are not release-vintage data. Conservative lags reduce, but do not eliminate, vintage/revision bias.",
    },
}
LOG = logging.getLogger("qeti.ingestion")


def _http_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "QETI-local-research/1.0"})
    return session


class MarketDataStore:
    """SQLite-backed observation, event-label, and provider-status store."""

    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS market_observations (
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume REAL,
                    value REAL,
                    units TEXT,
                    PRIMARY KEY (source, symbol, observed_at)
                );
                CREATE INDEX IF NOT EXISTS idx_observations_symbol_date
                    ON market_observations(symbol, observed_at);
                CREATE TABLE IF NOT EXISTS source_status (
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    latest_observation TEXT,
                    fetched_at TEXT NOT NULL,
                    message TEXT NOT NULL,
                    PRIMARY KEY (source, symbol)
                );
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    attribution TEXT NOT NULL,
                    frequency TEXT NOT NULL,
                    first_observation TEXT,
                    latest_observation TEXT,
                    observation_count INTEGER NOT NULL,
                    last_checked TEXT NOT NULL,
                    limitations TEXT NOT NULL,
                    PRIMARY KEY (source, symbol)
                );
                CREATE TABLE IF NOT EXISTS event_labels (
                    event_name TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    peak_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source_reference TEXT NOT NULL,
                    market_relevance TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def upsert_observations(self, source: str, symbol: str, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        normalized = _normalize_observations(frame)
        records = []
        for row in normalized.to_dict(orient="records"):
            records.append(
                (
                    source,
                    symbol,
                    row["Date"].strftime("%Y-%m-%d"),
                    row.get("Open"),
                    row.get("High"),
                    row.get("Low"),
                    row.get("Close"),
                    row.get("Adj Close"),
                    row.get("Volume"),
                    row.get("Value"),
                    row.get("Units"),
                )
            )
        with self._connection() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO market_observations
                (source, symbol, observed_at, open, high, low, close, adj_close,
                 volume, value, units)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
        return len(records)

    def read_observations(
        self, source: str, symbol: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        query = """
            SELECT observed_at AS Date, open AS Open, high AS High, low AS Low,
                   close AS Close, adj_close AS "Adj Close", volume AS Volume,
                   value AS Value, units AS Units
            FROM market_observations
            WHERE source = ? AND symbol = ?
        """
        params: list[Any] = [source, symbol]
        if start:
            query += " AND observed_at >= ?"
            params.append(start)
        if end:
            query += " AND observed_at < ?"
            params.append(end)
        query += " ORDER BY observed_at"
        with self._connection() as connection:
            frame = pd.read_sql_query(query, connection, params=params)
        if not frame.empty:
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            frame = frame.dropna(axis=1, how="all")
        return frame

    def delete_observations_after(self, source: str, symbol: str, latest_date: str) -> int:
        """Remove only future-dated observations for a source/symbol."""
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM market_observations "
                "WHERE source = ? AND symbol = ? AND observed_at > ?",
                (source, symbol, latest_date),
            )
        return cursor.rowcount

    def write_status(
        self,
        source: str,
        symbol: str,
        status: str,
        row_count: int,
        latest_observation: str | None,
        message: str,
    ) -> None:
        fetched_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO source_status
                (source, symbol, status, row_count, latest_observation, fetched_at, message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (source, symbol, status, row_count, latest_observation, fetched_at, message),
            )
            details = SOURCE_ATTRIBUTION.get(source, {})
            observation_range = connection.execute(
                """
                SELECT MIN(observed_at), MAX(observed_at), COUNT(*)
                FROM market_observations WHERE source = ? AND symbol = ?
                """,
                (source, symbol),
            ).fetchone()
            connection.execute(
                """
                INSERT OR REPLACE INTO dataset_metadata
                (source, symbol, source_url, attribution, frequency, first_observation,
                 latest_observation, observation_count, last_checked, limitations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    symbol,
                    details.get("url", ""),
                    details.get("attribution", "Local documented dataset."),
                    details.get("frequency", "Daily"),
                    observation_range[0],
                    observation_range[1],
                    observation_range[2],
                    fetched_at,
                    details.get("limitations", ""),
                ),
            )

    def statuses(self) -> pd.DataFrame:
        with self._connection() as connection:
            return pd.read_sql_query(
                "SELECT source, symbol, status, row_count, latest_observation, "
                "fetched_at, message FROM source_status ORDER BY source, symbol",
                connection,
            )

    def metadata(self) -> pd.DataFrame:
        with self._connection() as connection:
            return pd.read_sql_query(
                "SELECT source, symbol, source_url, attribution, frequency, "
                "first_observation, latest_observation, observation_count, "
                "last_checked, limitations FROM dataset_metadata "
                "ORDER BY source, symbol",
                connection,
            )

    def sync_events(self, path: str | Path) -> int:
        events = pd.read_csv(path, dtype=str).fillna("")
        required = {
            "event_name", "event_type", "start_date", "peak_date", "end_date",
            "description", "source_reference", "market_relevance",
        }
        missing = required.difference(events.columns)
        if missing:
            raise ValueError(f"Event labels are missing required columns: {sorted(missing)}")
        for column in ("start_date", "peak_date", "end_date"):
            events[column] = pd.to_datetime(events[column], errors="raise").dt.strftime("%Y-%m-%d")
        invalid_order = (
            (events["start_date"] > events["peak_date"])
            | (events["peak_date"] > events["end_date"])
        )
        if invalid_order.any():
            invalid_names = events.loc[invalid_order, "event_name"].tolist()
            raise ValueError(f"Event dates must satisfy start <= peak <= end: {invalid_names}")
        allowed_relevance = {"broad_market_crash", "crypto_stress", "institution_specific"}
        invalid_relevance = set(events["market_relevance"]).difference(allowed_relevance)
        if invalid_relevance:
            raise ValueError(f"Unsupported market_relevance labels: {sorted(invalid_relevance)}")
        if events["source_reference"].str.strip().eq("").any():
            raise ValueError("Every event label must include at least one source reference.")
        with self._connection() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO event_labels
                (event_name, event_type, start_date, peak_date, end_date, description,
                 source_reference, market_relevance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [tuple(row[col] for col in (
                    "event_name", "event_type", "start_date", "peak_date", "end_date",
                    "description", "source_reference", "market_relevance",
                )) for row in events.to_dict(orient="records")],
            )
        return len(events)


def _normalize_observations(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "Date" not in data.columns:
        data = data.reset_index()
        data = data.rename(columns={data.columns[0]: "Date"})
    clean_columns: list[str] = []
    for column in data.columns:
        if isinstance(column, tuple):
            candidates = [str(part) for part in column if str(part).lower() in {
                "open", "high", "low", "close", "adj close", "volume", "value", "units"
            }]
            clean_columns.append(candidates[0].title() if candidates else str(column[0]))
        else:
            clean_columns.append(str(column).strip().title() if str(column).lower() != "date" else "Date")
    data.columns = clean_columns
    if "Date" not in data:
        raise ValueError("Provider response did not contain a date column.")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    data = data.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    for column in ("Open", "High", "Low", "Close", "Adj Close", "Volume", "Value"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    keep = ["Date"] + [col for col in (
        "Open", "High", "Low", "Close", "Adj Close", "Volume", "Value", "Units"
    ) if col in data]
    return data[keep].reset_index(drop=True)


def _symbol_filename(source: str, symbol: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", symbol).strip("_").lower()
    return f"{source}_{slug}.csv"


def _save_cache(source: str, symbol: str, frame: pd.DataFrame) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / _symbol_filename(source, symbol)
    _normalize_observations(frame).to_csv(path, index=False)
    return path


def _yahoo_frame(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    last_error = None
    for attempt in range(3):
        try:
            frame = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column",
                multi_level_index=False,
                timeout=30,
            )
            if frame is None or frame.empty:
                raise RuntimeError("Yahoo Finance returned no observations.")
            return _normalize_observations(frame)
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError(
        f"Yahoo Finance request failed after 3 attempts: {type(last_error).__name__}: {last_error}"
    ) from last_error


def _binance_frames(
    symbol: str, start: str, end: str, session: requests.Session
) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) - 1
    cursor = start_ms
    rows: list[list[Any]] = []
    while cursor <= end_ms:
        response = session.get(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": "1d",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )
        response.raise_for_status()
        chunk = response.json()
        if not isinstance(chunk, list):
            raise RuntimeError(f"Unexpected Binance response for {symbol}.")
        if not chunk:
            break
        rows.extend(chunk)
        cursor = int(chunk[-1][0]) + 86_400_000
        if len(chunk) < 1000:
            break
        time.sleep(0.15)
    if not rows:
        raise RuntimeError(f"Binance returned no observations for {symbol}.")
    frame = pd.DataFrame(
        rows,
        columns=[
            "timestamp", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ],
    )
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    frame = frame[pd.to_numeric(frame["close_time"], errors="coerce") <= now_ms]
    if frame.empty:
        raise RuntimeError(f"Binance has no fully closed daily candles for {symbol}.")
    # Binance timestamps mark candle open; the daily close is only known next day.
    frame["Date"] = (
        pd.to_datetime(frame["close_time"], unit="ms", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
        + pd.Timedelta(days=1)
    )
    return _normalize_observations(frame)


def _fred_frame(series: str, start: str, end: str, session: requests.Session) -> pd.DataFrame:
    response = session.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series, "cosd": start, "coed": (pd.Timestamp(end) - pd.Timedelta(days=1)).date().isoformat()},
        timeout=30,
    )
    response.raise_for_status()
    frame = pd.read_csv(io.BytesIO(response.content))
    if frame.empty or len(frame.columns) < 2:
        raise RuntimeError(f"FRED returned no observations for {series}.")
    frame = frame.rename(columns={frame.columns[0]: "Date", frame.columns[1]: "Value"})
    frame["Value"] = pd.to_numeric(frame["Value"].replace(".", pd.NA), errors="coerce")
    frame["Units"] = "FRED series value"
    frame = frame.dropna(subset=["Value"])
    if frame.empty:
        raise RuntimeError(f"FRED returned no usable numeric values for {series}.")
    return _normalize_observations(frame)


def _fetch_one(
    store: MarketDataStore,
    source: str,
    symbol: str,
    start: str,
    end: str,
    fetcher,
) -> dict[str, Any]:
    try:
        frame = fetcher()
        frame = _normalize_observations(frame)
        count = store.upsert_observations(source, symbol, frame)
        cache = _save_cache(source, symbol, frame)
        latest = frame["Date"].max().date().isoformat()
        message = f"Downloaded {count} observations; cached at {cache.relative_to(ROOT)}."
        store.write_status(source, symbol, "downloaded", count, latest, message)
        return {"source": source, "symbol": symbol, "status": "downloaded", "rows": count, "latest": latest, "message": message}
    except Exception as exc:
        cached = store.read_observations(source, symbol, start, end)
        if not cached.empty:
            latest = cached["Date"].max().date().isoformat()
            message = f"Provider unavailable ({type(exc).__name__}: {exc}); using {len(cached)} cached rows."
            store.write_status(source, symbol, "cached", len(cached), latest, message)
            LOG.warning("%s/%s: %s", source, symbol, message)
            return {"source": source, "symbol": symbol, "status": "cached", "rows": len(cached), "latest": latest, "message": message}
        message = f"Unavailable ({type(exc).__name__}: {exc}); no cached observations."
        store.write_status(source, symbol, "unavailable", 0, None, message)
        LOG.error("%s/%s: %s", source, symbol, message)
        return {"source": source, "symbol": symbol, "status": "unavailable", "rows": 0, "latest": None, "message": message}


def refresh_data(
    start: str = "1990-01-01",
    end: str | None = None,
    db_path: str | Path = DEFAULT_DB,
    yahoo_symbols: dict[str, str] | None = None,
    binance_symbols: dict[str, str] | None = None,
    fred_series: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Refresh all configured free data sources and return an explicit status table."""
    end = end or (date.today() + timedelta(days=1)).isoformat()
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
    if start_dt >= end_dt:
        raise ValueError("start must be earlier than the exclusive end date.")
    start, end = start_dt.date().isoformat(), end_dt.date().isoformat()
    store = MarketDataStore(db_path)
    session = _http_session()
    statuses = []

    for symbol in (yahoo_symbols or YAHOO_SYMBOLS):
        statuses.append(_fetch_one(store, "yahoo", symbol, start, end, lambda s=symbol: _yahoo_frame(s, start, end)))
    for symbol in (binance_symbols or BINANCE_SYMBOLS):
        store.delete_observations_after(
            "binance", symbol, datetime.now(timezone.utc).date().isoformat()
        )
        statuses.append(_fetch_one(store, "binance", symbol, start, end, lambda s=symbol: _binance_frames(s, start, end, session)))
    for series in (fred_series or FRED_SERIES):
        statuses.append(_fetch_one(store, "fred", series, start, end, lambda s=series: _fred_frame(s, start, end, session)))

    event_path = ROOT / "data" / "crisis_events.csv"
    if event_path.exists():
        event_count = store.sync_events(event_path)
        statuses.append({
            "source": "local", "symbol": "crisis_events", "status": "loaded",
            "rows": event_count, "latest": None,
            "message": f"Loaded {event_count} documented event labels from {event_path.relative_to(ROOT)}.",
        })

    report_path = RAW_DIR / "source_status.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sources": SOURCE_ATTRIBUTION,
                "status": statuses,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pd.DataFrame(statuses)


def load_market_feature_frame(
    dates: pd.Series | pd.DatetimeIndex,
    db_path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Return point-in-time-aligned external market and macro features.

    Source values are joined backward only (never backfilled), so an observation
    cannot affect a date before that observation exists.
    """
    target_dates = pd.DataFrame({"Date": pd.DatetimeIndex(pd.to_datetime(dates)).normalize()}).drop_duplicates().sort_values("Date")
    if target_dates.empty or not Path(db_path).exists():
        return target_dates

    store = MarketDataStore(db_path)
    feature_specs = [
        ("yahoo", "^IXIC", "nasdaq", True),
        ("yahoo", "^DJI", "dow", True),
        ("yahoo", "^NSEI", "nifty", True),
        ("yahoo", "^VIX", "vix", True),
        ("binance", "BTCUSDT", "btc", True),
        ("binance", "ETHUSDT", "eth", True),
        ("fred", "DGS10", "fred_dgs10", False),
        ("fred", "DGS2", "fred_dgs2", False),
        ("fred", "CPIAUCSL", "fred_cpi", False),
        ("fred", "UNRATE", "fred_unrate", False),
        ("fred", "FEDFUNDS", "fred_fedfunds", False),
    ]
    primary = store.read_observations("yahoo", "^GSPC")
    primary_return = primary.set_index("Date")["Close"].pct_change() if not primary.empty else pd.Series(dtype=float)
    features = target_dates.copy()

    for source, symbol, name, is_price in feature_specs:
        observations = store.read_observations(source, symbol)
        value_column = "Close" if is_price else "Value"
        if observations.empty or value_column not in observations:
            continue
        series = observations[["Date", value_column]].dropna().sort_values("Date")
        if series.empty:
            continue
        if source == "fred":
            series["Date"] = series["Date"] + pd.to_timedelta(
                FRED_PUBLICATION_LAG_DAYS.get(symbol, 30), unit="D"
            )
        source_values = series[value_column]
        series[f"cross_{name}_level"] = source_values.to_numpy()
        if is_price:
            series[f"cross_{name}_return"] = source_values.pct_change().to_numpy()
            aligned_returns = target_dates.merge(
                series[["Date", f"cross_{name}_return"]],
                on="Date",
                how="left",
            )[f"cross_{name}_return"]
            primary_aligned = target_dates.merge(
                primary_return.rename("primary_return").rename_axis("Date").reset_index(),
                on="Date",
                how="left",
            )["primary_return"] if not primary_return.empty else pd.Series(np.nan, index=target_dates.index)
            rolling_corr = aligned_returns.rolling(60, min_periods=20).corr(primary_aligned)
            aligned_levels = pd.merge_asof(
                target_dates,
                series[["Date", f"cross_{name}_level"]],
                on="Date",
                direction="backward",
            )[f"cross_{name}_level"]
            feature_values = pd.DataFrame({
                "Date": target_dates["Date"],
                f"cross_{name}_level": aligned_levels,
                f"cross_{name}_return": aligned_returns,
                f"cross_{name}_correlation_60": rolling_corr,
            })
        else:
            feature_values = pd.merge_asof(
                target_dates,
                series[["Date", f"cross_{name}_level"]],
                on="Date",
                direction="backward",
            )
            if name == "fred_cpi":
                cpi_yoy = source_values.pct_change(12)
                cpi_frame = pd.DataFrame({"Date": series["Date"], f"cross_{name}_yoy": cpi_yoy.to_numpy()})
                aligned_cpi = pd.merge_asof(target_dates, cpi_frame, on="Date", direction="backward")
                feature_values[f"cross_{name}_yoy"] = aligned_cpi[f"cross_{name}_yoy"]
        features = features.merge(feature_values, on="Date", how="left")

    return features.sort_values("Date").reset_index(drop=True)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Refresh free public QETI financial data.")
    parser.add_argument("--start", default="1990-01-01", help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Exclusive end date; defaults to tomorrow.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB, help="SQLite database path.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    statuses = refresh_data(args.start, args.end, args.database)
    print(statuses.to_string(index=False))
    primary = statuses[
        statuses["source"].eq("yahoo") & statuses["symbol"].eq("^GSPC")
    ]
    if primary.empty or primary.iloc[0]["status"] not in {"downloaded", "cached"}:
        LOG.error(
            "Primary S&P 500 data is unavailable; refusing to label an old CSV fallback as a live refresh."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
