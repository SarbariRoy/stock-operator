"""Build EOD OHLCV prices from NSE bhavcopy CSV files.

This script switches the trigger data source away from Yahoo Finance.
It downloads daily bhavcopy files from NSE archives and writes a normalized
prices file used by the trigger system:

  stock_triggers/data/prices_eod.csv

Output schema:
- Date (YYYY-MM-DD)
- Ticker (e.g., RELIANCE.NS)
- Open
- High
- Low
- Close
- AdjClose (same as Close)
- Volume
"""

from __future__ import annotations

import argparse
import io
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_OUT = DATA_DIR / "prices_eod.csv"
DEFAULT_TICKERS_FILE = ROOT / "stock_selector" / "data" / "stocks.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NSE bhavcopy and build prices_eod.csv")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (default: 365 days ago)")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--tickers", nargs="+", default=None, help="Optional ticker list, e.g. RELIANCE.NS TCS.NS")
    parser.add_argument(
        "--tickers-file",
        type=str,
        default=str(DEFAULT_TICKERS_FILE),
        help="CSV with a Ticker column (default: stock_selector/data/stocks.csv)",
    )
    parser.add_argument("--suffix", type=str, default=".NS", help="Ticker suffix to append (default: .NS)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of merge")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output CSV path")
    return parser.parse_args()


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _bhav_url(d: date) -> str:
    mon = d.strftime("%b").upper()
    return f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{d.day:02d}{mon}{d.year}.csv"


def load_tickers(args: argparse.Namespace) -> set[str]:
    if args.tickers:
        tickers = [str(t).strip().upper() for t in args.tickers if str(t).strip()]
    else:
        src = Path(args.tickers_file)
        if not src.exists():
            raise SystemExit(f"Tickers file not found: {src}")
        df = pd.read_csv(src)
        if "Ticker" not in df.columns:
            raise SystemExit(f"Ticker column not found in {src}")
        tickers = [str(t).strip().upper() for t in df["Ticker"].dropna().tolist()]

    symbols = set()
    suffix = str(args.suffix).upper()
    for t in tickers:
        if suffix and t.endswith(suffix):
            symbols.add(t[: -len(suffix)])
        else:
            symbols.add(t)
    return symbols


def fetch_day(session: requests.Session, d: date, symbols: set[str], suffix: str) -> pd.DataFrame:
    if d.weekday() >= 5:
        return pd.DataFrame()

    url = _bhav_url(d)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*;q=0.8",
    }

    try:
        resp = session.get(url, headers=headers, timeout=30)
    except requests.RequestException:
        return pd.DataFrame()

    if resp.status_code != 200 or not resp.text.strip():
        return pd.DataFrame()

    try:
        raw = resp.content.decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(raw))
    except Exception:
        return pd.DataFrame()

    df.columns = [str(c).strip().upper() for c in df.columns]

    symbol_col = _pick_col(df, ["SYMBOL"])
    series_col = _pick_col(df, ["SERIES"])
    open_col = _pick_col(df, ["OPEN_PRICE", "OPEN"])
    high_col = _pick_col(df, ["HIGH_PRICE", "HIGH"])
    low_col = _pick_col(df, ["LOW_PRICE", "LOW"])
    close_col = _pick_col(df, ["CLOSE_PRICE", "CLOSE"])
    volume_col = _pick_col(df, ["TTL_TRD_QNTY", "TOTTRDQTY", "TOTTRD_QTY", "VOLUME"])

    required = [symbol_col, series_col, open_col, high_col, low_col, close_col, volume_col]
    if any(c is None for c in required):
        return pd.DataFrame()

    out = df[[symbol_col, series_col, open_col, high_col, low_col, close_col, volume_col]].copy()
    out.rename(
        columns={
            symbol_col: "SYMBOL",
            series_col: "SERIES",
            open_col: "Open",
            high_col: "High",
            low_col: "Low",
            close_col: "Close",
            volume_col: "Volume",
        },
        inplace=True,
    )

    out["SYMBOL"] = out["SYMBOL"].astype(str).str.strip().str.upper()
    out["SERIES"] = out["SERIES"].astype(str).str.strip().str.upper()
    out = out[out["SERIES"] == "EQ"]
    out = out[out["SYMBOL"].isin(symbols)]

    if out.empty:
        return out

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    if out.empty:
        return out

    out["Date"] = pd.to_datetime(d)
    out["Ticker"] = out["SYMBOL"] + suffix
    out["AdjClose"] = out["Close"]

    return out[["Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"]]


def build_prices(start_d: date, end_d: date, symbols: set[str], suffix: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    session = requests.Session()

    d = start_d
    while d <= end_d:
        day_df = fetch_day(session, d, symbols, suffix)
        if not day_df.empty:
            rows.append(day_df)
        d += timedelta(days=1)

    if not rows:
        raise SystemExit("No bhavcopy rows found in date range. Check dates, tickers, or network access.")

    out = pd.concat(rows, ignore_index=True)
    out.sort_values(["Date", "Ticker"], inplace=True)
    out.drop_duplicates(subset=["Date", "Ticker"], keep="last", inplace=True)
    return out


def merge_with_existing(new_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    if not out_path.exists():
        return new_df
    old = pd.read_csv(out_path, parse_dates=["Date"])
    combo = pd.concat([old, new_df], ignore_index=True)
    combo.drop_duplicates(subset=["Date", "Ticker"], keep="last", inplace=True)
    combo.sort_values(["Date", "Ticker"], inplace=True)
    return combo


def main() -> None:
    args = parse_args()

    today = date.today()
    start_d = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else today - timedelta(days=365)
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else today

    if start_d > end_d:
        raise SystemExit("--start cannot be after --end")

    symbols = load_tickers(args)
    suffix = str(args.suffix)

    prices = build_prices(start_d, end_d, symbols, suffix)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    final = prices if args.overwrite else merge_with_existing(prices, out_path)
    final.to_csv(out_path, index=False)

    print(f"Saved {len(final)} rows to {out_path}")
    print(f"Tickers covered: {final['Ticker'].nunique()} | Date range: {final['Date'].min().date()} -> {final['Date'].max().date()}")


if __name__ == "__main__":
    main()
