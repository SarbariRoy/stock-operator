"""Phase 1: Build/refresh event calendar for NSE tickers.

Downloads earnings and dividend event dates from Yahoo Finance; normalizes to trading-day windows.

Output:
- stock_triggers/data/event_calendar.csv — ticker, event_date, event_type, trading_days_to_signal.

Usage:
  python build_event_calendar.py [--days 730] [--universe-file path/to/universe.txt]
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "stock_triggers" / "data"
UNIVERSE_FILE = DATA_DIR / "universe_tickers.txt"
EVENT_CALENDAR_CSV = DATA_DIR / "event_calendar.csv"
PRICES_CSV = DATA_DIR / "prices_eod.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build event calendar from Yahoo Finance events data.")
    p.add_argument("--days", type=int, default=730, help="Calendar days of history to fetch.")
    p.add_argument("--universe-file", type=str, default=str(UNIVERSE_FILE), help="Path to universe ticker file.")
    p.add_argument(
        "--manual-events-csv",
        type=str,
        default="",
        help=(
            "Optional manual events CSV path. Expected columns: "
            "ticker,event_date,event_type[,amount_or_eps]. "
            "event_type should be earnings or dividend."
        ),
    )
    p.add_argument("--user-agent", type=str, default="Brilliant", help="HTTP User-Agent.")
    p.add_argument("--insecure", action="store_true", help="Disable TLS verification (not recommended).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing event calendar.")
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


def _fetch_events(
    session: requests.Session,
    *,
    ticker: str,
    period1: int,
    period2: int,
) -> list[dict]:
    """Fetch earnings and dividend events from Yahoo Finance."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "div,earnings",
    }
    try:
        r = session.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        result = payload.get("chart", {}).get("result")
        if not result:
            return []
        obj = result[0]
        events_data = obj.get("events", {})
        divisions = events_data.get("dividends", [])
        earnings_list = events_data.get("earnings", [])

        events = []
        for div in divisions:
            try:
                # Handle both dict and other formats
                if isinstance(div, dict):
                    date_val = div.get("date")
                else:
                    date_val = None
                
                if date_val is None:
                    continue
                    
                if isinstance(date_val, (int, float)):
                    event_date = pd.Timestamp(date_val, unit="s", tz=None)
                else:
                    event_date = pd.Timestamp(date_val)
                    
                amount = div.get("amount", 0) if isinstance(div, dict) else 0
                events.append(
                    {
                        "ticker": ticker,
                        "event_date": event_date,
                        "event_type": "dividend",
                        "amount_or_eps": amount,
                    }
                )
            except Exception:
                continue
        
        for earning in earnings_list:
            try:
                max_age = earning.get("maxAge", 0) if isinstance(earning, dict) else 0
                if max_age > 0:
                    event_date = pd.Timestamp(max_age, unit="s", tz=None)
                else:
                    eps_est = earning.get("epsEstimate", {}) if isinstance(earning, dict) else {}
                    raw_date = eps_est.get("raw") if isinstance(eps_est, dict) else None
                    event_date = pd.Timestamp(raw_date) if raw_date else None
                
                if pd.notna(event_date):
                    eps_val = earning.get("epsEstimate", {}).get("raw", None) if isinstance(earning, dict) else None
                    events.append(
                        {
                            "ticker": ticker,
                            "event_date": event_date,
                            "event_type": "earnings",
                            "amount_or_eps": eps_val,
                        }
                    )
            except Exception:
                continue
        
        return events
    except requests.RequestException as ex:
        print(f"Warning: Failed to fetch events for {ticker}: {ex}", file=sys.stderr)
        return []


def _ticker_to_nse_symbol(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".NS", "")


def _init_nse_session(session: requests.Session) -> None:
    """Warm NSE cookies. NSE API often needs a homepage request first."""
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/",
            "Origin": "https://www.nseindia.com",
        }
    )
    try:
        session.get("https://www.nseindia.com", timeout=20)
    except requests.RequestException:
        pass


def _extract_amount_from_text(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"rs\.?\s*([0-9]+(?:\.[0-9]+)?)", str(text), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _fetch_events_nse(
    session: requests.Session,
    *,
    symbol: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[dict]:
    """Fetch earnings/dividend events from NSE APIs."""
    symbol = _ticker_to_nse_symbol(symbol)
    events: list[dict] = []

    # 1) Corporate announcements -> infer earnings from results/board-meeting texts.
    ann_url = "https://www.nseindia.com/api/corporate-announcements"
    try:
        r = session.get(ann_url, params={"index": "equities", "symbol": symbol}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, list):
            for row in payload:
                txt = " ".join(
                    [
                        str(row.get("desc", "")),
                        str(row.get("attchmntText", "")),
                        str(row.get("subject", "")),
                    ]
                ).lower()
                is_earnings = any(
                    k in txt
                    for k in [
                        "financial results",
                        "quarterly results",
                        "board meeting",
                        "results",
                        "audited",
                        "unaudited",
                    ]
                )
                is_dividend = "dividend" in txt
                if not (is_earnings or is_dividend):
                    continue

                dt_raw = row.get("an_dt") or row.get("sort_date") or row.get("dt")
                dt = pd.to_datetime(dt_raw, errors="coerce", dayfirst=True)
                if pd.isna(dt):
                    continue
                d = dt.date()
                if d < start_date or d > end_date:
                    continue

                ev_type = "dividend" if is_dividend else "earnings"
                amt = _extract_amount_from_text(txt)
                events.append(
                    {
                        "ticker": f"{symbol}.NS",
                        "event_date": pd.Timestamp(dt).normalize(),
                        "event_type": ev_type,
                        "amount_or_eps": amt,
                    }
                )
    except requests.RequestException:
        pass

    # 2) Corporate actions -> robust dividend ex-date source.
    actions_url = "https://www.nseindia.com/api/corporates-corporateActions"
    try:
        r = session.get(actions_url, params={"index": "equities", "symbol": symbol}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, list):
            for row in payload:
                subject = str(row.get("subject", ""))
                if "dividend" not in subject.lower():
                    continue
                dt_raw = row.get("exDate") or row.get("recDate")
                dt = pd.to_datetime(dt_raw, errors="coerce", dayfirst=True)
                if pd.isna(dt):
                    continue
                d = dt.date()
                if d < start_date or d > end_date:
                    continue
                amt = _extract_amount_from_text(subject)
                events.append(
                    {
                        "ticker": f"{symbol}.NS",
                        "event_date": pd.Timestamp(dt).normalize(),
                        "event_type": "dividend",
                        "amount_or_eps": amt,
                    }
                )
    except requests.RequestException:
        pass

    return events


def _load_universe(path: Path) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _load_trading_calendar(prices_path: Path) -> set[pd.Timestamp]:
    """Extract unique trading dates from prices file."""
    if not prices_path.exists():
        return set()
    df = pd.read_csv(prices_path, parse_dates=["Date"], usecols=["Date"])
    return set(pd.to_datetime(df["Date"]).dt.normalize())


def _get_nearest_trading_day(event_date: pd.Timestamp, trading_dates: set[pd.Timestamp], direction: str = "nearest") -> pd.Timestamp | None:
    """Find nearest trading day to event date."""
    if not trading_dates:
        return event_date  # Fallback if no trading calendar available.
    trading_sorted = sorted(trading_dates)
    event_norm = event_date.normalize()

    if event_norm in trading_dates:
        return event_norm

    if direction == "nearest":
        # Find closest trading date (before or after).
        idx = next((i for i, d in enumerate(trading_sorted) if d > event_norm), None)
        if idx is None:
            return trading_sorted[-1]
        before = trading_sorted[idx - 1] if idx > 0 else None
        after = trading_sorted[idx]
        if before and (event_norm - before) < (after - event_norm):
            return before
        return after
    elif direction == "before":
        idx = next((i for i, d in enumerate(trading_sorted) if d >= event_norm), None)
        return trading_sorted[idx - 1] if idx and idx > 0 else None
    elif direction == "after":
        return next((d for d in trading_sorted if d >= event_norm), None)
    return None


def _load_manual_events(path: Path, trading_dates: set[pd.Timestamp]) -> pd.DataFrame:
    """Load manually maintained earnings/dividend events from CSV.

    Expected columns:
      - ticker
      - event_date
      - event_type (earnings|dividend)
      - amount_or_eps (optional)
    """
    if not path or not path.is_file():
        return pd.DataFrame(columns=["ticker", "event_date", "event_type", "amount_or_eps"])

    df = pd.read_csv(path)
    required = {"ticker", "event_date", "event_type"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Manual events CSV missing required columns: {sorted(missing)}. "
            "Expected: ticker,event_date,event_type[,amount_or_eps]"
        )

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.normalize()
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower().replace({
        "earning": "earnings",
        "div": "dividend",
        "dividends": "dividend",
    })
    if "amount_or_eps" not in df.columns:
        df["amount_or_eps"] = pd.NA

    df = df[df["event_type"].isin(["earnings", "dividend"])].copy()
    df = df[df["ticker"].ne("") & df["event_date"].notna()].copy()

    # Snap to nearest trading day to align with signal_date joins.
    if trading_dates:
        snapped = df["event_date"].apply(lambda x: _get_nearest_trading_day(x, trading_dates, direction="nearest"))
        df["event_date"] = snapped.fillna(df["event_date"])

    df = df[["ticker", "event_date", "event_type", "amount_or_eps"]].drop_duplicates(
        subset=["ticker", "event_date", "event_type"], keep="last"
    )
    df.sort_values(["ticker", "event_date"], inplace=True)
    return df


def build_event_calendar(
    *,
    days: int,
    universe: list[str],
    user_agent: str,
    insecure: bool,
    trading_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    end = datetime.now().date()
    start = end - timedelta(days=int(days))
    period1 = int(datetime.combine(start, datetime.min.time()).timestamp())
    period2 = int(datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp())

    s = _build_session(insecure=insecure, user_agent=user_agent)
    _init_nse_session(s)
    all_events = []

    for i, ticker in enumerate(universe):
        ticker = ticker.strip().upper()
        if not ticker:
            continue
        if i % 10 == 0:
            print(f"Fetching events for {ticker} ({i}/{len(universe)})", file=sys.stderr)

        # Primary: NSE APIs (better for Indian symbols), fallback: Yahoo.
        events = _fetch_events_nse(
            s,
            symbol=ticker,
            start_date=start,
            end_date=end,
        )
        if not events:
            events = _fetch_events(s, ticker=ticker, period1=period1, period2=period2)
        all_events.extend(events)

    if not all_events:
        print("No events fetched from Yahoo Finance.", file=sys.stderr)
        return pd.DataFrame(columns=["ticker", "event_date", "event_type", "amount_or_eps"])

    df = pd.DataFrame(all_events)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
    
    # Snap events to nearest trading day.
    if trading_dates:
        df["event_date"] = df["event_date"].apply(
            lambda x: _get_nearest_trading_day(x, trading_dates, direction="nearest")
        )

    df = df[["ticker", "event_date", "event_type", "amount_or_eps"]].copy()
    df.drop_duplicates(inplace=True)
    df.sort_values(["ticker", "event_date"], inplace=True)

    return df


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    universe = _load_universe(Path(args.universe_file))
    if not universe:
        raise SystemExit(f"No tickers found in universe file: {args.universe_file}")

    trading_dates = _load_trading_calendar(PRICES_CSV)
    print(f"Loaded {len(trading_dates)} trading dates from {PRICES_CSV}", file=sys.stderr)

    calendar = build_event_calendar(
        days=int(args.days),
        universe=universe,
        user_agent=args.user_agent,
        insecure=args.insecure,
        trading_dates=trading_dates,
    )

    manual_events = pd.DataFrame(columns=["ticker", "event_date", "event_type", "amount_or_eps"])
    if str(args.manual_events_csv).strip():
        manual_path = Path(str(args.manual_events_csv).strip())
        manual_events = _load_manual_events(manual_path, trading_dates)
        if not manual_events.empty:
            print(f"Loaded {len(manual_events)} manual events from {manual_path}", file=sys.stderr)

    if not manual_events.empty:
        calendar = pd.concat([calendar, manual_events], ignore_index=True)
        calendar.drop_duplicates(subset=["ticker", "event_date", "event_type"], keep="last", inplace=True)
        calendar.sort_values(["ticker", "event_date"], inplace=True)

    if EVENT_CALENDAR_CSV.is_file() and not args.overwrite:
        old = pd.read_csv(EVENT_CALENDAR_CSV, parse_dates=["event_date"])
        combined = pd.concat([old, calendar], ignore_index=True)
        combined.drop_duplicates(subset=["ticker", "event_date", "event_type"], keep="last", inplace=True)
        combined.sort_values(["ticker", "event_date"], inplace=True)
        calendar = combined

    calendar.to_csv(EVENT_CALENDAR_CSV, index=False)
    print(f"Saved {len(calendar)} events to {EVENT_CALENDAR_CSV}")


if __name__ == "__main__":
    main()
