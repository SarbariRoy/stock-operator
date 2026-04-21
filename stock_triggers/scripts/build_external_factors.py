"""Build external factor datasets for backtesting filters.

Outputs:
- stock_triggers/data/external_factors.csv
- stock_triggers/data/ticker_sector_map.csv

Data source for market factors uses Yahoo Chart API via requests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
TRIGGERS_DIR = ROOT / "stock_triggers"
DATA_DIR = TRIGGERS_DIR / "data"
UNIVERSE_FILE = DATA_DIR / "universe_tickers.txt"
EXTERNAL_FACTORS_CSV = DATA_DIR / "external_factors.csv"
TICKER_SECTOR_MAP_CSV = DATA_DIR / "ticker_sector_map.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build external factors and sector map CSVs.")
    p.add_argument("--days", type=int, default=730, help="Calendar days of history to fetch.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs.")
    p.add_argument("--user-agent", type=str, default="Brilliant", help="HTTP User-Agent.")
    p.add_argument("--insecure", action="store_true", help="Disable TLS verification (not recommended).")
    return p.parse_args()


def _build_session(*, insecure: bool, user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept": "application/json,text/plain,*/*"})
    s.verify = not insecure
    if insecure:
        try:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
    return s


def _fetch_series_close(
    session: requests.Session,
    *,
    ticker: str,
    period1: int,
    period2: int,
    out_col: str,
) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "div,splits",
    }
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        return pd.DataFrame(columns=["Date", out_col])
    obj = result[0]
    ts = obj.get("timestamp") or []
    quote = (obj.get("indicators", {}).get("quote") or [{}])[0]
    close = quote.get("close") or []
    if not ts or not close:
        return pd.DataFrame(columns=["Date", out_col])
    n = min(len(ts), len(close))
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(ts[:n], unit="s", utc=True).tz_convert(None).normalize(),
            out_col: pd.to_numeric(close[:n], errors="coerce"),
        }
    )
    df.dropna(subset=[out_col], inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    return df


def build_external_factors(*, days: int, user_agent: str, insecure: bool) -> pd.DataFrame:
    end = datetime.now().date()
    start = end - timedelta(days=int(days))
    period1 = int(datetime.combine(start, datetime.min.time()).timestamp())
    period2 = int(datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp())

    s = _build_session(insecure=insecure, user_agent=user_agent)
    frames: list[pd.DataFrame] = []

    pulls = [
        ("^INDIAVIX", "india_vix_close"),
        ("INR=X", "usdinr_close"),
        ("BZ=F", "brent_close"),
    ]

    for ticker, col in pulls:
        try:
            df = _fetch_series_close(
                s,
                ticker=ticker,
                period1=period1,
                period2=period2,
                out_col=col,
            )
            if not df.empty:
                frames.append(df)
        except requests.RequestException:
            if ticker == "BZ=F":
                # Fallback to WTI if Brent pull fails.
                try:
                    df = _fetch_series_close(
                        s,
                        ticker="CL=F",
                        period1=period1,
                        period2=period2,
                        out_col=col,
                    )
                    if not df.empty:
                        frames.append(df)
                except requests.RequestException:
                    pass

    if not frames:
        raise SystemExit("Could not fetch any external factor series from Yahoo API.")

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="Date", how="outer")
    out.sort_values("Date", inplace=True)

    # Placeholder flow column; can be replaced later by exchange-sourced data.
    if "fii_dii_net_cr" not in out.columns:
        out["fii_dii_net_cr"] = pd.NA

    # Compute derived features.
    out["vix_change_1d_pct"] = out["india_vix_close"].pct_change() * 100
    out["usdinr_ret_5d_pct"] = out["usdinr_close"].pct_change(5) * 100
    out["brent_ret_5d_pct"] = out["brent_close"].pct_change(5) * 100

    out = out[["Date", "india_vix_close", "usdinr_close", "brent_close", "fii_dii_net_cr", 
               "vix_change_1d_pct", "usdinr_ret_5d_pct", "brent_ret_5d_pct"]]
    return out


def _load_universe(path: Path) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def build_sector_map(universe: list[str]) -> pd.DataFrame:
    # Best-effort mapping for common Indian symbols; unmatched tickers fall back to "Other".
    sector_of: dict[str, str] = {
        "RELIANCE.NS": "Energy",
        "ONGC.NS": "Energy",
        "IOC.NS": "Energy",
        "BPCL.NS": "Energy",
        "TCS.NS": "IT",
        "INFY.NS": "IT",
        "HCLTECH.NS": "IT",
        "WIPRO.NS": "IT",
        "TECHM.NS": "IT",
        "LT.NS": "Industrials",
        "SIEMENS.NS": "Industrials",
        "INDUSTOWER.NS": "Telecom Infra",
        "HDFCBANK.NS": "Banking",
        "ICICIBANK.NS": "Banking",
        "SBIN.NS": "Banking",
        "AXISBANK.NS": "Banking",
        "KOTAKBANK.NS": "Banking",
        "INDUSINDBK.NS": "Banking",
        "SBILIFE.NS": "Insurance",
        "BAJFINANCE.NS": "NBFC",
        "BAJAJFINSV.NS": "NBFC",
        "ITC.NS": "FMCG",
        "HINDUNILVR.NS": "FMCG",
        "NESTLEIND.NS": "FMCG",
        "BRITANNIA.NS": "FMCG",
        "DABUR.NS": "FMCG",
        "GODREJCP.NS": "FMCG",
        "TATACONSUM.NS": "FMCG",
        "ASIANPAINT.NS": "Consumer",
        "TITAN.NS": "Consumer",
        "MARUTI.NS": "Auto",
        "TATAMOTORS.NS": "Auto",
        "M&M.NS": "Auto",
        "HEROMOTOCO.NS": "Auto",
        "TVSMOTOR.NS": "Auto",
        "SUNPHARMA.NS": "Pharma",
        "DRREDDY.NS": "Pharma",
        "CIPLA.NS": "Pharma",
        "DIVISLAB.NS": "Pharma",
        "APOLLOHOSP.NS": "Healthcare",
        "ULTRACEMCO.NS": "Cement",
        "SHREECEM.NS": "Cement",
        "AMBUJACEM.NS": "Cement",
        "JSWSTEEL.NS": "Metals",
        "TATASTEEL.NS": "Metals",
        "HINDALCO.NS": "Metals",
        "VEDL.NS": "Metals",
        "COALINDIA.NS": "Mining",
        "NTPC.NS": "Utilities",
        "POWERGRID.NS": "Utilities",
        "TATAPOWER.NS": "Utilities",
        "ADANIPORTS.NS": "Logistics",
        "ADANIENT.NS": "Diversified",
        "ADANIGREEN.NS": "Utilities",
        "ADANIPOWER.NS": "Utilities",
        "GRASIM.NS": "Materials",
        "EICHERMOT.NS": "Auto",
        "PIDILITIND.NS": "Chemicals",
        "DLF.NS": "Real Estate",
        "BHARTIARTL.NS": "Telecom",
        "INDIGO.NS": "Airlines",
        "SPICEJET.NS": "Airlines",
        "GOLDBEES.NS": "Commodity ETF",
        "SILVERBEES.NS": "Commodity ETF",
    }

    rows = []
    for t in universe:
        tt = t.strip().upper()
        if not tt:
            continue
        rows.append({"ticker": tt, "sector": sector_of.get(tt, "Other")})

    out = pd.DataFrame(rows).drop_duplicates()
    out.sort_values(["sector", "ticker"], inplace=True)
    return out


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    external = build_external_factors(days=int(args.days), user_agent=args.user_agent, insecure=args.insecure)
    if EXTERNAL_FACTORS_CSV.is_file() and not args.overwrite:
        old = pd.read_csv(EXTERNAL_FACTORS_CSV, parse_dates=["Date"])
        combined = pd.concat([old, external], ignore_index=True)
        combined.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        combined.sort_values("Date", inplace=True)
        external = combined
    external.to_csv(EXTERNAL_FACTORS_CSV, index=False)

    universe = _load_universe(UNIVERSE_FILE)
    sector_map = build_sector_map(universe)
    sector_map.to_csv(TICKER_SECTOR_MAP_CSV, index=False)

    print(f"Saved {len(external)} rows to {EXTERNAL_FACTORS_CSV}")
    print(f"Saved {len(sector_map)} rows to {TICKER_SECTOR_MAP_CSV}")


if __name__ == "__main__":
    main()
