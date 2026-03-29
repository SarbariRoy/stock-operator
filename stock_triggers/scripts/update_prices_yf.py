"""Download daily OHLCV data from Yahoo chart API using requests.

This avoids yfinance internals and fetches chart data directly from:
https://query1.finance.yahoo.com/v8/finance/chart/<TICKER>

Output is written to stock_triggers/data/prices_eod.csv in long format:
Date, Ticker, Open, High, Low, Close, AdjClose, Volume
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
TRIGGERS_DIR = ROOT / "stock_triggers"
DATA_DIR = TRIGGERS_DIR / "data"
PRICES_CSV = DATA_DIR / "prices_eod.csv"
DEFAULT_BENCHMARK_TICKERS = ("^NSEI",)


def _is_benchmark_ticker(ticker: str) -> bool:
    return str(ticker).strip().upper() in {t.upper() for t in DEFAULT_BENCHMARK_TICKERS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily OHLCV from Yahoo chart API and save to prices_eod.csv",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="List of tickers (e.g., RELIANCE.NS TCS.NS). Optional if --universe-file is provided.",
    )
    parser.add_argument(
        "--universe-file",
        type=str,
        help="Path to a text file with one ticker per line (comments starting with # are ignored). If provided, tickers are read from this file unless --tickers is also given.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1200,
        help="Number of calendar days of history to download (default: 1200, enough to retain data back to 2023 in the current workflow).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite prices_eod.csv instead of appending/merging.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="Brilliant",
        help="HTTP User-Agent for Yahoo requests (default: Brilliant).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (not recommended).",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.5,
        help="Pause between ticker requests to reduce throttling (default: 0.5).",
    )
    return parser.parse_args()


def _build_session(*, insecure: bool, user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json,text/plain,*/*",
        }
    )
    session.verify = not insecure

    if insecure:
        try:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

    return session


def _fetch_ticker_chart(
    session: requests.Session,
    *,
    ticker: str,
    period1: int,
    period2: int,
) -> dict | None:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "div,splits",
    }

    try:
        resp = session.get(url, params=params, timeout=30)
    except requests.RequestException:
        return None

    if resp.status_code != 200:
        return None

    try:
        payload = resp.json()
    except ValueError:
        return None

    chart = payload.get("chart", {})
    result = chart.get("result")
    if not result:
        return None
    return result[0]


def download_prices(
    tickers: Iterable[str],
    days: int,
    *,
    insecure: bool,
    user_agent: str,
    pause_seconds: float,
) -> pd.DataFrame:
    end = datetime.now().date()
    start = end - timedelta(days=days)

    period1 = int(datetime.combine(start, datetime.min.time()).timestamp())
    period2 = int(datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp())

    session = _build_session(insecure=insecure, user_agent=user_agent)
    rows: List[pd.DataFrame] = []
    failed: List[str] = []

    for ticker in tickers:
        ticker = str(ticker).strip()
        if not ticker:
            continue
        result = _fetch_ticker_chart(
            session,
            ticker=ticker,
            period1=period1,
            period2=period2,
        )
        if result is None:
            failed.append(ticker)
            time.sleep(max(0.0, pause_seconds))
            continue

        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators", {})
        quote_list = indicators.get("quote") or []
        if not timestamps or not quote_list:
            failed.append(ticker)
            time.sleep(max(0.0, pause_seconds))
            continue

        quote = quote_list[0]
        adj_list = indicators.get("adjclose") or []
        adjclose_values = adj_list[0].get("adjclose") if adj_list else None

        n = len(timestamps)
        open_v = (quote.get("open") or [None] * n)[:n]
        high_v = (quote.get("high") or [None] * n)[:n]
        low_v = (quote.get("low") or [None] * n)[:n]
        close_v = (quote.get("close") or [None] * n)[:n]
        vol_v = (quote.get("volume") or [None] * n)[:n]
        if adjclose_values is None:
            adj_v = close_v
        else:
            adj_v = adjclose_values[:n]

        sub = pd.DataFrame(
            {
                "Date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).date,
                "Ticker": ticker,
                "Open": open_v,
                "High": high_v,
                "Low": low_v,
                "Close": close_v,
                "AdjClose": adj_v,
                "Volume": vol_v,
            }
        )
        for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        if _is_benchmark_ticker(ticker):
            sub["Volume"] = sub["Volume"].fillna(0.0)
            required_cols = ["Open", "High", "Low", "Close"]
        else:
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
        sub.dropna(subset=required_cols, inplace=True)
        if not sub.empty:
            rows.append(sub)
        else:
            failed.append(ticker)

        time.sleep(max(0.0, pause_seconds))

    if not rows:
        raise SystemExit("No data returned from Yahoo API. Check tickers, CA bundle, or network.")

    all_prices = pd.concat(rows, ignore_index=True)
    all_prices.sort_values(["Date", "Ticker"], inplace=True)
    all_prices.drop_duplicates(subset=["Date", "Ticker"], keep="last", inplace=True)

    if failed:
        failed_str = ", ".join(sorted(set(failed)))
        print(f"Warning: no rows for tickers: {failed_str}")

    return all_prices


def merge_with_existing(new_df: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    if not existing_path.exists():
        return new_df
    old = pd.read_csv(existing_path, parse_dates=["Date"])
    old["Date"] = old["Date"].dt.date
    combined = pd.concat([old, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["Date", "Ticker"], keep="last", inplace=True)
    combined.sort_values(["Date", "Ticker"], inplace=True)
    return combined


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve tickers: explicit CLI list wins; otherwise read from universe file if given.
    if args.tickers:
        tickers = args.tickers
    elif args.universe_file:
        universe_path = Path(args.universe_file)
        if not universe_path.is_file():
            raise SystemExit(f"Universe file not found: {universe_path}")
        lines = universe_path.read_text(encoding="utf-8").splitlines()
        tickers = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not tickers:
            raise SystemExit(f"Universe file {universe_path} contained no valid tickers.")
    else:
        raise SystemExit("You must provide either --tickers ... or --universe-file path/to/file.txt")

    tickers = list(dict.fromkeys([*tickers, *DEFAULT_BENCHMARK_TICKERS]))

    prices = download_prices(
        tickers,
        args.days,
        insecure=args.insecure,
        user_agent=args.user_agent,
        pause_seconds=args.pause_seconds,
    )

    if args.overwrite:
        final = prices
    else:
        final = merge_with_existing(prices, PRICES_CSV)

    final.to_csv(PRICES_CSV, index=False)
    print(f"Saved {len(final)} rows to {PRICES_CSV}")


if __name__ == "__main__":
    main()
