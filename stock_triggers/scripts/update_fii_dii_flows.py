"""Fetch FII/DII flows from NSE API and update external_factors.csv.

Writes/updates the `fii_dii_net_cr` column in:
stock_triggers/data/external_factors.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "stock_triggers" / "data"
EXTERNAL_FACTORS_CSV = DATA_DIR / "external_factors.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update fii_dii_net_cr in external_factors.csv from NSE and optional historical CSV.")
    p.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Start date in YYYY-MM-DD. Defaults to 730 days before today.",
    )
    p.add_argument(
        "--to-date",
        type=str,
        default=datetime.now().date().isoformat(),
        help="End date in YYYY-MM-DD (default: today).",
    )
    p.add_argument("--insecure", action="store_true", help="Disable TLS verification (not recommended).")
    p.add_argument(
        "--historical-csv",
        type=str,
        default=None,
        help=(
            "Optional CSV path for historical flows. Expected columns include Date and either "
            "fii_dii_net_cr OR (fii_net_cr and dii_net_cr) OR netValue."
        ),
    )
    return p.parse_args()


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null", "-"}:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date_any(v: Any) -> pd.Timestamp | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y", "%d %b %Y", "%d/%m/%Y"]:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt).date())
        except ValueError:
            continue
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt.date())


def _extract_net_from_record(r: dict[str, Any]) -> float | None:
    # Try common keys for net values.
    net_keys = [
        "netValue",
        "net",
        "netCash",
        "fiiNet",
        "diiNet",
        "fiiNetValue",
        "diiNetValue",
    ]
    for k in net_keys:
        if k in r:
            val = _to_float(r.get(k))
            if val is not None:
                return val

    # Fallback: buy - sell.
    buy_keys = ["buyValue", "buy", "fiiBuyValue", "diiBuyValue"]
    sell_keys = ["sellValue", "sell", "fiiSellValue", "diiSellValue"]
    buy_val = None
    sell_val = None
    for k in buy_keys:
        if k in r:
            buy_val = _to_float(r.get(k))
            if buy_val is not None:
                break
    for k in sell_keys:
        if k in r:
            sell_val = _to_float(r.get(k))
            if sell_val is not None:
                break
    if buy_val is not None and sell_val is not None:
        return buy_val - sell_val
    return None


def _extract_date_from_record(r: dict[str, Any]) -> pd.Timestamp | None:
    date_keys = ["date", "Date", "tradeDate", "tradedDate"]
    for k in date_keys:
        if k in r:
            dt = _parse_date_any(r.get(k))
            if dt is not None:
                return dt
    return None


def _fetch_nse_payload(from_date: str, to_date: str, *, insecure: bool) -> Any:
    session = requests.Session()
    session.verify = not insecure
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/",
        }
    )

    # Warm-up call for cookies often required by NSE API.
    session.get("https://www.nseindia.com", timeout=30)

    from_nse = pd.Timestamp(from_date).strftime("%d-%m-%Y")
    to_nse = pd.Timestamp(to_date).strftime("%d-%m-%Y")
    url = (
        "https://www.nseindia.com/api/fiidiiTradeReact"
        f"?fromDate={from_nse}&toDate={to_nse}"
    )
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_flows(payload: Any) -> pd.DataFrame:
    # Output: Date, fii_dii_net_cr (combined net)
    by_date: dict[pd.Timestamp, float] = {}

    def add_record(rec: dict[str, Any], *, sign: float = 1.0) -> None:
        dt = _extract_date_from_record(rec)
        if dt is None:
            return
        net = _extract_net_from_record(rec)
        if net is None:
            return
        by_date[dt] = by_date.get(dt, 0.0) + sign * float(net)

    if isinstance(payload, list):
        for rec in payload:
            if isinstance(rec, dict):
                add_record(rec)

    elif isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            for rec in payload["data"]:
                if isinstance(rec, dict):
                    # If each row already has both fii and dii fields, this still works through key probing.
                    add_record(rec)

        # Alternate shape: separate arrays for fii and dii.
        fii_rows = payload.get("fii") if isinstance(payload.get("fii"), list) else []
        dii_rows = payload.get("dii") if isinstance(payload.get("dii"), list) else []
        if fii_rows or dii_rows:
            for rec in fii_rows:
                if isinstance(rec, dict):
                    add_record(rec)
            for rec in dii_rows:
                if isinstance(rec, dict):
                    add_record(rec)

    if not by_date:
        return pd.DataFrame(columns=["Date", "fii_dii_net_cr"])

    out = pd.DataFrame(
        {"Date": list(by_date.keys()), "fii_dii_net_cr": list(by_date.values())}
    )
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out.sort_values("Date", inplace=True)
    out.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    return out


def load_historical_flows_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["Date", "fii_dii_net_cr"])

    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame(columns=["Date", "fii_dii_net_cr"])

    date_col = None
    for c in ["Date", "date", "tradeDate", "tradedDate"]:
        if c in raw.columns:
            date_col = c
            break
    if date_col is None:
        return pd.DataFrame(columns=["Date", "fii_dii_net_cr"])

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(raw[date_col], errors="coerce").dt.normalize()

    if "fii_dii_net_cr" in raw.columns:
        out["fii_dii_net_cr"] = pd.to_numeric(raw["fii_dii_net_cr"], errors="coerce")
    elif "netValue" in raw.columns:
        out["fii_dii_net_cr"] = pd.to_numeric(raw["netValue"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    elif "net" in raw.columns:
        out["fii_dii_net_cr"] = pd.to_numeric(raw["net"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    elif "fii_net_cr" in raw.columns and "dii_net_cr" in raw.columns:
        fii = pd.to_numeric(raw["fii_net_cr"], errors="coerce")
        dii = pd.to_numeric(raw["dii_net_cr"], errors="coerce")
        out["fii_dii_net_cr"] = fii + dii
    else:
        return pd.DataFrame(columns=["Date", "fii_dii_net_cr"])

    out = out[out["Date"].notna() & out["fii_dii_net_cr"].notna()].copy()
    out.sort_values("Date", inplace=True)
    out.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    return out[["Date", "fii_dii_net_cr"]]


def main() -> None:
    args = parse_args()
    to_date = pd.Timestamp(args.to_date).date()
    if args.from_date:
        from_date = pd.Timestamp(args.from_date).date()
    else:
        from_date = to_date - timedelta(days=730)

    payload = _fetch_nse_payload(from_date.isoformat(), to_date.isoformat(), insecure=args.insecure)
    flows_live = parse_flows(payload)

    flows_hist = pd.DataFrame(columns=["Date", "fii_dii_net_cr"])
    if args.historical_csv:
        flows_hist = load_historical_flows_csv(Path(args.historical_csv))

    if flows_hist.empty:
        flows = flows_live.copy()
    elif flows_live.empty:
        flows = flows_hist.copy()
    else:
        flows = pd.concat([flows_hist, flows_live], ignore_index=True)
    flows.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    flows.sort_values("Date", inplace=True)
    if flows.empty:
        raise SystemExit("No FII/DII flow rows parsed from NSE or historical CSV.")

    if EXTERNAL_FACTORS_CSV.is_file():
        ext = pd.read_csv(EXTERNAL_FACTORS_CSV, parse_dates=["Date"])
    else:
        ext = pd.DataFrame(columns=["Date", "india_vix_close", "usdinr_close", "brent_close", "fii_dii_net_cr"])

    ext["Date"] = pd.to_datetime(ext["Date"], errors="coerce").dt.normalize()
    ext = ext[ext["Date"].notna()].copy()

    merged = ext.merge(flows, on="Date", how="outer", suffixes=("", "_new"))
    if "fii_dii_net_cr" not in merged.columns:
        merged["fii_dii_net_cr"] = pd.NA
    if "fii_dii_net_cr_new" in merged.columns:
        merged["fii_dii_net_cr"] = merged["fii_dii_net_cr_new"].combine_first(merged["fii_dii_net_cr"])
        merged.drop(columns=["fii_dii_net_cr_new"], inplace=True)

    for c in ["india_vix_close", "usdinr_close", "brent_close", "fii_dii_net_cr"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    ordered_cols = ["Date", "india_vix_close", "usdinr_close", "brent_close", "fii_dii_net_cr"]
    for c in ordered_cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged = merged[ordered_cols]
    merged.sort_values("Date", inplace=True)
    merged.drop_duplicates(subset=["Date"], keep="last", inplace=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(EXTERNAL_FACTORS_CSV, index=False)

    filled = int(merged["fii_dii_net_cr"].notna().sum())
    print(f"Saved {len(merged)} rows to {EXTERNAL_FACTORS_CSV}")
    print(f"fii_dii_net_cr non-null rows: {filled}")
    if len(flows_live) <= 2:
        print(
            "Note: NSE fiidiiTradeReact currently returns latest-day rows only; "
            "use --historical-csv to backfill full history."
        )


if __name__ == "__main__":
    main()
