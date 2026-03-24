"""Simple Streamlit UI for viewing Pattern A signals.

This is a starting point. It reads signals_pattern_a.csv and displays
signals in a table with basic filters.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import date, datetime
import subprocess
import sys

import pandas as pd
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
TRIGGERS_DIR = ROOT / "stock_triggers"
SCRIPTS_DIR = TRIGGERS_DIR / "scripts"
DATA_DIR = TRIGGERS_DIR / "data"
SIGNALS_CSV = DATA_DIR / "signals_pattern_a.csv"
SELL_SIGNALS_CSV = DATA_DIR / "sell_signals_pattern_a.csv"
PORTFOLIO_CSV = DATA_DIR / "portfolio_positions.csv"
DUMMY_LAB_CSV = DATA_DIR / "backtesting_lab_positions.csv"
PRICES_CSV = DATA_DIR / "prices_eod.csv"
EXTERNAL_FACTORS_CSV = DATA_DIR / "external_factors.csv"
TICKER_SECTOR_MAP_CSV = DATA_DIR / "ticker_sector_map.csv"
SECRETS_FILE = ROOT / "secrets.yml"
IS_STREAMLIT_CLOUD = bool(os.getenv("STREAMLIT_SHARING_MODE")) or bool(os.getenv("STREAMLIT_CLOUD"))
PRODUCTION_APP_URL = "https://stock-operator-roy.streamlit.app/"


st.set_page_config(page_title="Stock Triggers – Pattern A", layout="wide")
st.markdown(
    "<div class='brand-title'>Stock Triggers by <span class='brand-roy'>Roy</span></div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='small-muted'>Production link: <a href='{PRODUCTION_APP_URL}' target='_blank'>{PRODUCTION_APP_URL}</a></div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;600;700&display=swap');
    .block-container {padding-top: 3.0rem;}
    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.2px;
    }
    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.2px;
        font-size: 2.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.3rem;
        margin-bottom: 0.45rem;
        line-height: 1.15;
        padding-top: 0.25rem;
        display: block;
    }
    .brand-roy {
        font-style: italic;
    }
    .stApp {
        background: radial-gradient(circle at 15% 0%, #fff9ed 0%, #f8fbff 40%, #f4f8fb 100%);
    }
    .card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        background: #f8fafc;
        margin-bottom: 0.8rem;
    }
    .small-muted {color: #4b5563; font-size: 0.9rem;}
    .stat-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 0.7rem 0.9rem;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        min-height: 84px;
    }
    .stat-label {
        color: #475569;
        font-size: 0.82rem;
        margin-bottom: 0.2rem;
    }
    .stat-value {
        color: #0f172a;
        font-size: 1.3rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-ok {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .status-warn {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }
    .hero {
        border: 1px solid #fed7aa;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        background: linear-gradient(120deg, #fff7ed 0%, #ecfeff 100%);
    }
    .hero-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #7c2d12;
    }
    .hero-sub {
        color: #334155;
        font-size: 0.9rem;
    }
    .action-item {
        border: 1px solid #dbeafe;
        border-radius: 12px;
        background: #f8fbff;
        padding: 0.65rem 0.8rem;
        margin-bottom: 0.55rem;
    }
    .action-title {
        font-size: 0.85rem;
        color: #1e3a8a;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .action-value {
        font-size: 1.1rem;
        color: #0f172a;
        font-weight: 700;
    }
    .flow-wrap {
        border: 1px solid #dbe4ef;
        border-radius: 14px;
        background: #ffffff;
        padding: 0.55rem 0.7rem;
        margin-bottom: 0.7rem;
    }
    .flow-title {
        font-size: 0.82rem;
        color: #475569;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .flow-step {
        display: inline-block;
        border: 1px solid #cbd5e1;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        color: #334155;
        padding: 0.16rem 0.52rem;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
        background: #f8fafc;
    }
    .flow-step-done {
        background: #dcfce7;
        border-color: #86efac;
        color: #166534;
    }
    .flow-step-next {
        background: #dbeafe;
        border-color: #93c5fd;
        color: #1e3a8a;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #dbe4ef;
        border-radius: 12px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        overflow: hidden;
    }
    div[data-testid="stDataFrame"] [role="columnheader"] {
        background: #f8fafc;
        color: #334155;
        font-weight: 700;
        border-bottom: 1px solid #e2e8f0;
    }
    div[data-testid="stDataFrame"] [role="gridcell"] {
        border-bottom: 1px solid #f1f5f9;
    }
    div[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] {
        background: #f8fbff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_stat_card(label: str, value: str) -> None:
    st.markdown(
        (
            "<div class='stat-card'>"
            f"<div class='stat-label'>{label}</div>"
            f"<div class='stat-value'>{value}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_flow_header(*, step1_done: bool, step2_done: bool, step3_done: bool, step4_done: bool) -> None:
    state = [step1_done, step2_done, step3_done, step4_done]
    next_idx = None
    for i, done in enumerate(state):
        if not done:
            next_idx = i
            break

    labels = [
        "1. Refresh Data",
        "2. Generate Signals",
        "3. Review Action List",
        "4. Send Summary",
    ]

    parts: list[str] = ["<div class='flow-wrap'><div class='flow-title'>Today Flow</div>"]
    for i, label in enumerate(labels):
        css = "flow-step"
        if state[i]:
            css += " flow-step-done"
        elif next_idx == i:
            css += " flow-step-next"
        parts.append(f"<span class='{css}'>{label}</span>")
    parts.append("</div>")

    st.markdown("".join(parts), unsafe_allow_html=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def render_table(df: pd.DataFrame | pd.io.formats.style.Styler, *, height: int = 320) -> None:
    if isinstance(df, pd.DataFrame):
        display = df.copy()
        float_cols = display.select_dtypes(include=["float64", "float32"]).columns.tolist()
        for c in float_cols:
            display[c] = display[c].round(2)
        st.dataframe(display, width="stretch", hide_index=True, height=height)
    else:
        st.dataframe(df, width="stretch", hide_index=True, height=height)


def humanize_outcome(value: str) -> str:
    mapping = {
        "stop_hit": "Stopped out",
        "held_to_window_end": "Held to end",
        "time_stop": "Timed exit",
        "no_future_data": "No future data",
    }
    return mapping.get(str(value), str(value))


def render_glossary(*, section: str = "general") -> None:
    with st.expander("Glossary", expanded=False):
        st.markdown("- **Signal date**: The date when setup conditions were met.")
        st.markdown("- **Breakout**: Price closing above a recent high close.")
        st.markdown("- **Volume strength**: Today's volume versus 20-day average volume.")
        st.markdown("- **Stop / Initial risk limit**: Exit level used to cap downside risk.")
        st.markdown("- **Stop hit**: Price touched or crossed the stop level.")
        st.markdown("- **Hold window**: Number of forward days used for evaluation.")
        if section in {"signals", "general"}:
            st.markdown("- **Pattern**: The exact rule-set that generated the signal.")
            st.markdown("- **Current-only view**: Shows only today's actionable rows.")
        if section in {"backtest", "general"}:
            st.markdown("- **Strict mode (Pattern A+)**: Adds extra filters and dynamic exits to reduce weak setups.")
            st.markdown("- **ATR stop**: Volatility-based stop distance using Average True Range.")
            st.markdown("- **Break-even trigger**: Moves stop to entry after a minimum gain.")
            st.markdown("- **Time-stop**: Forced exit after a fixed number of days.")
            st.markdown("- **Pattern score**: Combined quality score from win-rate and average return.")


@st.cache_data(show_spinner=False)
def load_signals() -> pd.DataFrame:
    if not SIGNALS_CSV.is_file():
        return pd.DataFrame()
    return pd.read_csv(SIGNALS_CSV)


@st.cache_data(show_spinner=False)
def load_sell_signals() -> pd.DataFrame:
    if not SELL_SIGNALS_CSV.is_file():
        return pd.DataFrame()
    return pd.read_csv(SELL_SIGNALS_CSV)


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    if not PRICES_CSV.is_file():
        return pd.DataFrame()
    df = pd.read_csv(PRICES_CSV, parse_dates=["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_external_factors() -> pd.DataFrame:
    if not EXTERNAL_FACTORS_CSV.is_file():
        return pd.DataFrame()
    df = pd.read_csv(EXTERNAL_FACTORS_CSV)
    if "Date" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    if df.empty:
        return df
    df["Date"] = df["Date"].dt.normalize()

    # Normalize common column names for easier user-provided CSVs.
    rename_map = {
        "india_vix": "india_vix_close",
        "vix": "india_vix_close",
        "usdinr": "usdinr_close",
        "brent": "brent_close",
        "fii_dii": "fii_dii_net_cr",
        "fii_dii_net": "fii_dii_net_cr",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    for c in ["india_vix_close", "usdinr_close", "brent_close", "fii_dii_net_cr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df.sort_values("Date", inplace=True)
    if "india_vix_close" in df.columns:
        df["vix_change_1d_pct"] = df["india_vix_close"].pct_change() * 100.0
    if "usdinr_close" in df.columns:
        df["usdinr_ret_5d_pct"] = df["usdinr_close"].pct_change(5) * 100.0
    if "brent_close" in df.columns:
        df["brent_ret_5d_pct"] = df["brent_close"].pct_change(5) * 100.0
    return df


@st.cache_data(show_spinner=False)
def load_ticker_sector_map() -> pd.DataFrame:
    if not TICKER_SECTOR_MAP_CSV.is_file():
        return pd.DataFrame(columns=["ticker", "sector"])
    df = pd.read_csv(TICKER_SECTOR_MAP_CSV)
    if "ticker" not in df.columns or "sector" not in df.columns:
        return pd.DataFrame(columns=["ticker", "sector"])
    out = df[["ticker", "sector"]].copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["sector"] = out["sector"].astype(str).str.strip()
    out = out[(out["ticker"] != "") & (out["sector"] != "")].drop_duplicates()
    return out


def filter_eligible_dates_by_external_factors(
    eligible_dates: list[pd.Timestamp],
    factors_df: pd.DataFrame,
    *,
    max_vix: float | None = None,
    max_vix_1d_spike_pct: float | None = None,
    max_usdinr_5d_pct: float | None = None,
    max_brent_5d_pct: float | None = None,
    min_fii_dii_net_cr: float | None = None,
) -> tuple[list[pd.Timestamp], dict[str, int | str | bool]]:
    if not eligible_dates:
        return [], {"applied": False, "dates_kept": 0, "dates_total": 0}
    if factors_df.empty:
        return eligible_dates, {
            "applied": False,
            "dates_total": len(eligible_dates),
            "dates_kept": len(eligible_dates),
            "reason": "external_factors_missing",
        }

    tmp = factors_df.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"]).dt.normalize()
    fac = tmp.set_index("Date", drop=False)

    kept: list[pd.Timestamp] = []
    blocked_vix = 0
    blocked_vix_spike = 0
    blocked_usdinr = 0
    blocked_brent = 0
    blocked_flows = 0

    for d in eligible_dates:
        dn = pd.to_datetime(d).normalize()
        if dn not in fac.index:
            kept.append(d)
            continue

        row = fac.loc[dn]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        blocked = False
        vix_val = row.get("india_vix_close")
        if max_vix is not None and pd.notna(vix_val) and float(vix_val) > float(max_vix):
            blocked = True
            blocked_vix += 1

        vix_spike_val = row.get("vix_change_1d_pct")
        if (
            (not blocked)
            and max_vix_1d_spike_pct is not None
            and pd.notna(vix_spike_val)
            and float(vix_spike_val) > float(max_vix_1d_spike_pct)
        ):
            blocked = True
            blocked_vix_spike += 1

        usdinr_ret_val = row.get("usdinr_ret_5d_pct")
        if (
            (not blocked)
            and max_usdinr_5d_pct is not None
            and pd.notna(usdinr_ret_val)
            and float(usdinr_ret_val) > float(max_usdinr_5d_pct)
        ):
            blocked = True
            blocked_usdinr += 1

        brent_ret_val = row.get("brent_ret_5d_pct")
        if (
            (not blocked)
            and max_brent_5d_pct is not None
            and pd.notna(brent_ret_val)
            and float(brent_ret_val) > float(max_brent_5d_pct)
        ):
            blocked = True
            blocked_brent += 1

        flows_val = row.get("fii_dii_net_cr")
        if (
            (not blocked)
            and min_fii_dii_net_cr is not None
            and pd.notna(flows_val)
            and float(flows_val) < float(min_fii_dii_net_cr)
        ):
            blocked = True
            blocked_flows += 1

        if not blocked:
            kept.append(d)

    summary: dict[str, int | str | bool] = {
        "applied": True,
        "dates_total": len(eligible_dates),
        "dates_kept": len(kept),
        "blocked_vix": blocked_vix,
        "blocked_vix_spike": blocked_vix_spike,
        "blocked_usdinr": blocked_usdinr,
        "blocked_brent": blocked_brent,
        "blocked_flows": blocked_flows,
    }
    return kept, summary


def build_ticker_sector_rs_table(
    prices_df: pd.DataFrame,
    ticker_sector_df: pd.DataFrame,
    *,
    lookback_days: int = 20,
) -> pd.DataFrame:
    cols = ["Date", "ticker", "sector", "sector_rs20"]
    if prices_df.empty or ticker_sector_df.empty:
        return pd.DataFrame(columns=cols)

    p = prices_df[["Date", "Ticker", "Close"]].copy()
    p["Date"] = pd.to_datetime(p["Date"]).dt.normalize()
    p["ticker"] = p["Ticker"].astype(str).str.upper()
    p.sort_values(["ticker", "Date"], inplace=True)
    p["ret_lb_pct"] = p.groupby("ticker")["Close"].pct_change(int(lookback_days)) * 100.0

    m = ticker_sector_df.copy()
    m["ticker"] = m["ticker"].astype(str).str.upper()

    pr = p.merge(m, on="ticker", how="inner")
    pr = pr[pr["ret_lb_pct"].notna()].copy()
    if pr.empty:
        return pd.DataFrame(columns=cols)

    market_ret = pr.groupby("Date", as_index=False)["ret_lb_pct"].mean().rename(
        columns={"ret_lb_pct": "market_ret_lb_pct"}
    )
    sector_ret = pr.groupby(["Date", "sector"], as_index=False)["ret_lb_pct"].mean().rename(
        columns={"ret_lb_pct": "sector_ret_lb_pct"}
    )
    rs = sector_ret.merge(market_ret, on="Date", how="left")
    rs["sector_rs20"] = rs["sector_ret_lb_pct"] - rs["market_ret_lb_pct"]

    out = pr[["Date", "ticker", "sector"]].drop_duplicates().merge(
        rs[["Date", "sector", "sector_rs20"]], on=["Date", "sector"], how="left"
    )
    return out[cols]


def get_prices_refresh_info(prices_df: pd.DataFrame) -> dict[str, str]:
    """Return persistent refresh info from prices file and content."""
    if not PRICES_CSV.is_file():
        return {
            "file_updated": "-",
            "latest_market_date": "-",
            "rows": "0",
        }

    updated_dt = datetime.fromtimestamp(PRICES_CSV.stat().st_mtime)
    updated_str = updated_dt.strftime("%Y-%m-%d %H:%M")

    latest_market_date = "-"
    row_count = "0"
    if not prices_df.empty:
        latest_market_date = prices_df["Date"].max().date().isoformat()
        row_count = f"{len(prices_df):,}"

    return {
        "file_updated": updated_str,
        "latest_market_date": latest_market_date,
        "rows": row_count,
    }


def is_refreshed_today() -> bool:
    if not PRICES_CSV.is_file():
        return False
    updated_dt = datetime.fromtimestamp(PRICES_CSV.stat().st_mtime)
    return updated_dt.date() == date.today()


def load_local_secrets(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}

    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get_telegram_credentials() -> tuple[str, str]:
    secrets = load_local_secrets(SECRETS_FILE)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "") or secrets.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "") or secrets.get("TELEGRAM_CHAT_ID", "")
    return token, chat_id


def is_remote_runtime() -> bool:
    """Allow Telegram sending only from hosted runtimes, never from local hosts."""
    return bool(os.getenv("GITHUB_ACTIONS")) or bool(os.getenv("STREAMLIT_CLOUD")) or bool(os.getenv("STREAMLIT_SHARING_MODE"))


def build_telegram_message_for_date(signals_df: pd.DataFrame, signal_date: str) -> str:
    if signals_df.empty:
        return "Stock Trigger Update\n\nNo trigger today."

    rows = signals_df[signals_df["signal_date"] == signal_date].copy()
    rows.sort_values(["ticker"], inplace=True)

    if rows.empty:
        return f"Stock Trigger Update\n\nDate: {signal_date}\nNo trigger today."

    lines = [
        "Stock Trigger Update",
        "",
        f"Date: {signal_date}",
        f"Signals: {len(rows)}",
        "",
    ]
    for _, r in rows.iterrows():
        lines.append(
            f"- {r['ticker']} | {r['pattern']} | Entry {r['entry_price']} | Stop {r['stop_price']}"
        )
    return "\n".join(lines)


def build_sell_telegram_message(sell_df: pd.DataFrame) -> str:
    if sell_df.empty:
        return "Stock Trigger Update\n\nNo sell signal today."

    latest_sell_date = sell_df["sell_signal_date"].max()
    latest = sell_df[sell_df["sell_signal_date"] == latest_sell_date].copy()
    latest.sort_values(["ticker"], inplace=True)

    lines = [
        "Stock Trigger Update",
        "",
        f"Sell date: {latest_sell_date}",
        f"Sell signals: {len(latest)}",
        "",
    ]
    for _, r in latest.iterrows():
        lines.append(
            f"- SELL {r['ticker']} | Exit {r['sell_price']} | Return {r['realized_return_pct']}%"
        )
    return "\n".join(lines)


def send_telegram_message(token: str, chat_id: str, text: str) -> tuple[bool, str]:
    if not is_remote_runtime():
        return False, "Telegram send is blocked on local runtime by policy."

    if not token or not chat_id:
        return False, "Missing Telegram credentials (token/chat_id)."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
    except requests.RequestException as exc:
        return False, str(exc)

    if resp.status_code != 200:
        return False, f"Telegram API error {resp.status_code}: {resp.text[:500]}"

    return True, "sent"


def refresh_prices() -> tuple[bool, str]:
    """Run only the price updater step.

    Returns (ok, message).
    """

    if is_refreshed_today():
        return True, "Refresh skipped: prices file was already updated today."

    update_script = SCRIPTS_DIR / "update_prices_yf.py"
    if not update_script.is_file():
        return False, "Price updater script not found under stock_triggers/scripts/."

    # 1) Refresh prices for the configured universe (overwrite prices_eod.csv)
    update_cmd = [
        sys.executable,
        str(update_script),
        "--user-agent",
        "Brilliant",
        "--days",
        "365",
        "--pause-seconds",
        "0.8",
        "--overwrite",
        "--universe-file",
        str(DATA_DIR / "universe_tickers.txt"),
    ]

    try:
        res1 = subprocess.run(update_cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # pragma: no cover
        return False, f"Error running price updater: {exc}"

    if res1.returncode != 0:
        return False, f"Price updater failed (exit {res1.returncode}): {res1.stderr.strip()}"

    # Clear cached data so subsequent calls see fresh files
    load_prices.clear()
    load_signals.clear()

    return True, res1.stdout.strip()


def generate_triggers(
    *,
    breakout_days: int | None = None,
    volume_multiplier: float | None = None,
    stop_pct: float | None = None,
    as_of_date: str | None = None,
) -> tuple[bool, str]:
    """Run Pattern A trigger generation using latest prices.

    If parameters are provided, pass them through to the generator.
    """

    pattern_script = SCRIPTS_DIR / "generate_triggers_pattern_a.py"
    if not pattern_script.is_file():
        return False, "Pattern A script not found under stock_triggers/scripts/."

    cmd = [sys.executable, str(pattern_script)]
    if as_of_date:
        cmd.extend(["--as-of-date", as_of_date])
    if breakout_days is not None:
        cmd.extend(["--breakout-days", str(breakout_days)])
    if volume_multiplier is not None:
        cmd.extend(["--volume-multiplier", str(volume_multiplier)])
    if stop_pct is not None:
        cmd.extend(["--stop-pct", str(stop_pct)])
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # pragma: no cover
        return False, f"Error running Pattern A generator: {exc}"

    if res.returncode != 0:
        return False, f"Pattern A generator failed (exit {res.returncode}): {res.stderr.strip()}"

    load_signals.clear()
    load_sell_signals.clear()
    return True, res.stdout.strip()


def render_refresh_summary(prices: pd.DataFrame, signals: pd.DataFrame) -> None:
    """Show a short summary after refresh.

    Includes latest date, coverage vs universe, and signal counts.
    """

    st.subheader("Refresh Summary")

    if prices.empty:
        st.error("prices_eod.csv is empty after refresh.")
        return

    latest_date = prices["Date"].max()
    latest_date_obj = latest_date.date() if hasattr(latest_date, "date") else pd.to_datetime(latest_date).date()
    latest_date_str = latest_date_obj.isoformat()

    n_rows = len(prices)
    n_tickers = prices["Ticker"].nunique()

    universe_path = DATA_DIR / "universe_tickers.txt"
    universe: list[str] = []
    if universe_path.is_file():
        lines = universe_path.read_text(encoding="utf-8").splitlines()
        universe = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]

    have_latest_set: set[str] = set()
    n_universe = 0
    n_with_latest = 0
    missing_latest: list[str] = []

    if universe:
        have_latest = prices[prices["Date"] == latest_date]["Ticker"].unique().tolist()
        have_latest_set = set(have_latest)
        universe_set = set(universe)
        n_universe = len(universe_set)
        n_with_latest = len(universe_set & have_latest_set)
        missing_latest = sorted(universe_set - have_latest_set)

    latest_sig_date = None
    latest_sig_count = 0
    total_signals = len(signals)
    if not signals.empty:
        latest_sig_date = signals["signal_date"].max()
        latest_sig_count = int(
            signals[signals["signal_date"] == latest_sig_date]["ticker"].nunique()
        )

    top = st.columns(4)
    with top[0]:
        render_stat_card("Latest Trading Date (EOD available)", latest_date_str)
    with top[1]:
        render_stat_card("Price Rows", f"{n_rows:,}")
    with top[2]:
        render_stat_card("Tickers With History", f"{n_tickers}")
    with top[3]:
        render_stat_card("Signals Rows", f"{total_signals:,}")

    bottom = st.columns(3)
    with bottom[0]:
        render_stat_card("Universe Size", str(n_universe) if n_universe else "-")
    with bottom[1]:
        render_stat_card("Data On Latest Date", str(n_with_latest) if n_universe else "-")
    with bottom[2]:
        render_stat_card("Latest Signal Date", latest_sig_date if latest_sig_date else "-")

    today = date.today()
    gap_days = (today - latest_date_obj).days
    if gap_days == 0:
        st.markdown(
            "<span class='status-pill status-ok'>Up to date</span> "
            "Latest available EOD bar is for today.",
            unsafe_allow_html=True,
        )
    elif gap_days <= 3:
        if today.weekday() >= 5:
            st.info(
                "Latest trading date can be earlier than calendar date on weekends/holidays. "
                f"Current gap: {gap_days} day(s)."
            )
        else:
            st.info(
                "Latest trading date can be earlier than calendar date during market hours "
                "or before EOD publication from data source. "
                f"Current gap: {gap_days} day(s)."
            )
    else:
        st.warning(
            "Latest trading date appears older than expected "
            f"({gap_days} day(s) behind today). Check refresh run status and data-source availability."
        )

    if universe:
        if n_with_latest == n_universe:
            st.markdown(
                "<span class='status-pill status-ok'>Coverage OK</span> "
                "All universe tickers have data on the latest date.",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='status-pill status-warn'>Coverage Warning</span> "
                f"{n_universe - n_with_latest} ticker(s) are missing latest-date prices.",
                unsafe_allow_html=True,
            )
            with st.expander("Show missing tickers"):
                st.write(", ".join(missing_latest) if missing_latest else "None")
    else:
        st.info("Universe file not found/empty, so coverage vs configured universe cannot be validated.")

    if not signals.empty:
        st.info(
            f"Latest signal_date {latest_sig_date} has {latest_sig_count} ticker(s) with Pattern A signals."
        )
    else:
        st.warning("signals_pattern_a.csv has no rows currently.")


def compute_pattern_a_signals_for_date(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
    breakout_buffer_pct: float = 0.0,
    use_atr_stop: bool = False,
    atr_period: int = 14,
    atr_multiplier: float = 2.5,
) -> pd.DataFrame:
    """Compute Pattern A signals for one date from the provided price history."""

    all_rows: list[dict] = []
    for ticker, g in prices.groupby("Ticker", sort=True):
        g = g.copy().sort_values("Date")

        g["SMA50"] = g["Close"].rolling(50).mean()
        g["SMA200"] = g["Close"].rolling(200).mean()
        g["VolAvg20"] = g["Volume"].rolling(20).mean()
        g["PrevNHighClose"] = g["Close"].shift(1).rolling(breakout_days).max()
        tr1 = g["High"] - g["Low"]
        tr2 = (g["High"] - g["Close"].shift(1)).abs()
        tr3 = (g["Low"] - g["Close"].shift(1)).abs()
        g["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        g["ATR"] = g["TR"].rolling(int(atr_period)).mean()

        row = g[g["Date"] == as_of_date]
        if row.empty:
            continue
        r = row.iloc[0]

        needed = [r["SMA50"], r["SMA200"], r["VolAvg20"], r["PrevNHighClose"]]
        if use_atr_stop:
            needed.append(r["ATR"])
        if any(pd.isna(v) for v in needed):
            continue

        cond_trend = bool(r["SMA50"] > r["SMA200"])
        cond_price = bool((r["Close"] > r["SMA50"]) and (r["Close"] > r["SMA200"]))
        breakout_level = float(r["PrevNHighClose"]) * (1.0 + float(breakout_buffer_pct) / 100.0)
        cond_breakout = bool(r["Close"] > breakout_level)
        cond_volume = bool(r["Volume"] >= volume_multiplier * r["VolAvg20"])

        if not (cond_trend and cond_price and cond_breakout and cond_volume):
            continue

        entry_price = float(r["Close"])
        if use_atr_stop:
            stop_price = entry_price - float(r["ATR"]) * float(atr_multiplier)
            stop_pct_eff = ((entry_price - stop_price) / entry_price) * 100.0
        else:
            stop_price = entry_price * (1.0 - stop_pct / 100.0)
            stop_pct_eff = float(stop_pct)
        all_rows.append(
            {
                "signal_date": as_of_date.date().isoformat(),
                "ticker": ticker,
                "pattern": (
                    f"A_plus_breakout_{breakout_days}d"
                    if use_atr_stop or float(breakout_buffer_pct) > 0
                    else f"A_breakout_{breakout_days}d"
                ),
                "entry_price": round(entry_price, 4),
                "stop_pct": round(float(stop_pct_eff), 2),
                "stop_price": round(stop_price, 4),
            }
        )

    cols = [
        "signal_date",
        "ticker",
        "pattern",
        "entry_price",
        "stop_pct",
        "stop_price",
    ]
    if not all_rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(all_rows, columns=cols)
    out["pattern_family"] = "A"
    out["score_trend"] = pd.NA
    out["score_setup"] = pd.NA
    out["score_volume"] = pd.NA
    out["score_risk"] = pd.NA
    out["signal_score"] = pd.NA
    out["consensus_count"] = 1
    return out


def _clip_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _build_score_components(
    *,
    trend_strength_pct: float,
    setup_strength_pct: float,
    volume_ratio: float,
    stop_pct_eff: float,
) -> tuple[float, float, float, float, float]:
    score_trend = _clip_score(50.0 + trend_strength_pct * 5.0)
    score_setup = _clip_score(50.0 + setup_strength_pct * 8.0)
    score_volume = _clip_score(40.0 + volume_ratio * 20.0)
    score_risk = _clip_score(100.0 - stop_pct_eff * 6.0)
    signal_score = round(
        (0.3 * score_trend) + (0.3 * score_setup) + (0.2 * score_volume) + (0.2 * score_risk),
        1,
    )
    return (
        round(score_trend, 1),
        round(score_setup, 1),
        round(score_volume, 1),
        round(score_risk, 1),
        signal_score,
    )


def compute_pattern_b_signals_for_date(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    volume_multiplier: float,
    stop_pct: float,
    pullback_buffer_pct: float = 1.5,
    rebound_min_pct: float = 0.2,
) -> pd.DataFrame:
    """Pattern B: trend pullback and rebound near SMA20 within an uptrend."""

    all_rows: list[dict] = []
    for ticker, g in prices.groupby("Ticker", sort=True):
        g = g.copy().sort_values("Date")

        g["SMA20"] = g["Close"].rolling(20).mean()
        g["SMA50"] = g["Close"].rolling(50).mean()
        g["SMA200"] = g["Close"].rolling(200).mean()
        g["VolAvg20"] = g["Volume"].rolling(20).mean()
        g["ClosePrev1"] = g["Close"].shift(1)
        g["SwingLow10"] = g["Low"].shift(1).rolling(10).min()

        row = g[g["Date"] == as_of_date]
        if row.empty:
            continue
        r = row.iloc[0]

        needed = [
            r["SMA20"],
            r["SMA50"],
            r["SMA200"],
            r["VolAvg20"],
            r["ClosePrev1"],
            r["SwingLow10"],
        ]
        if any(pd.isna(v) for v in needed):
            continue

        cond_trend = bool((r["SMA50"] > r["SMA200"]) and (r["Close"] > r["SMA50"]))
        cond_pullback = bool(r["Close"] <= float(r["SMA20"]) * (1.0 + float(pullback_buffer_pct) / 100.0))
        cond_rebound = bool(
            r["Close"] >= float(r["ClosePrev1"]) * (1.0 + float(rebound_min_pct) / 100.0)
        )
        cond_volume = bool(r["Volume"] >= max(1.0, float(volume_multiplier) * 0.8) * float(r["VolAvg20"]))

        if not (cond_trend and cond_pullback and cond_rebound and cond_volume):
            continue

        entry_price = float(r["Close"])
        fixed_stop = entry_price * (1.0 - float(stop_pct) / 100.0)
        stop_price = min(entry_price * 0.995, max(fixed_stop, float(r["SwingLow10"])))
        stop_pct_eff = ((entry_price - stop_price) / entry_price) * 100.0

        trend_strength_pct = ((float(r["SMA50"]) / float(r["SMA200"])) - 1.0) * 100.0
        setup_strength_pct = ((float(r["SMA20"]) / float(r["Close"])) - 1.0) * 100.0
        volume_ratio = float(r["Volume"]) / float(r["VolAvg20"])

        score_trend, score_setup, score_volume, score_risk, signal_score = _build_score_components(
            trend_strength_pct=trend_strength_pct,
            setup_strength_pct=setup_strength_pct,
            volume_ratio=volume_ratio,
            stop_pct_eff=stop_pct_eff,
        )

        all_rows.append(
            {
                "signal_date": as_of_date.date().isoformat(),
                "ticker": ticker,
                "pattern": "B_pullback_rebound",
                "pattern_family": "B",
                "entry_price": round(entry_price, 4),
                "stop_pct": round(float(stop_pct_eff), 2),
                "stop_price": round(stop_price, 4),
                "score_trend": score_trend,
                "score_setup": score_setup,
                "score_volume": score_volume,
                "score_risk": score_risk,
                "signal_score": signal_score,
                "consensus_count": 1,
            }
        )

    cols = [
        "signal_date",
        "ticker",
        "pattern",
        "pattern_family",
        "entry_price",
        "stop_pct",
        "stop_price",
        "score_trend",
        "score_setup",
        "score_volume",
        "score_risk",
        "signal_score",
        "consensus_count",
    ]
    if not all_rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(all_rows, columns=cols)


def compute_scored_signals_for_date(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
    use_pattern_a: bool,
    use_pattern_b: bool,
    breakout_buffer_pct: float = 0.0,
    use_atr_stop: bool = False,
    atr_period: int = 14,
    atr_multiplier: float = 2.5,
    pullback_buffer_pct: float = 1.5,
    rebound_min_pct: float = 0.2,
    min_signal_score: float = 0.0,
    consensus_bonus: float = 5.0,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    if use_pattern_a:
        a_df = compute_pattern_a_signals_for_date(
            prices,
            as_of_date=as_of_date,
            breakout_days=int(breakout_days),
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            breakout_buffer_pct=float(breakout_buffer_pct),
            use_atr_stop=bool(use_atr_stop),
            atr_period=int(atr_period),
            atr_multiplier=float(atr_multiplier),
        )
        if not a_df.empty:
            for i in a_df.index:
                ticker = a_df.at[i, "ticker"]
                g = prices[prices["Ticker"] == ticker].copy().sort_values("Date")
                g = g[g["Date"] <= as_of_date].copy()
                g["SMA50"] = g["Close"].rolling(50).mean()
                g["SMA200"] = g["Close"].rolling(200).mean()
                g["VolAvg20"] = g["Volume"].rolling(20).mean()
                g["PrevNHighClose"] = g["Close"].shift(1).rolling(int(breakout_days)).max()
                r = g.iloc[-1]
                if pd.isna(r["SMA50"]) or pd.isna(r["SMA200"]) or pd.isna(r["VolAvg20"]) or pd.isna(r["PrevNHighClose"]):
                    continue
                trend_strength_pct = ((float(r["SMA50"]) / float(r["SMA200"])) - 1.0) * 100.0
                setup_strength_pct = ((float(r["Close"]) / float(r["PrevNHighClose"])) - 1.0) * 100.0
                volume_ratio = float(r["Volume"]) / float(r["VolAvg20"])
                stop_pct_eff = float(a_df.at[i, "stop_pct"])
                score_trend, score_setup, score_volume, score_risk, signal_score = _build_score_components(
                    trend_strength_pct=trend_strength_pct,
                    setup_strength_pct=setup_strength_pct,
                    volume_ratio=volume_ratio,
                    stop_pct_eff=stop_pct_eff,
                )
                a_df.at[i, "score_trend"] = score_trend
                a_df.at[i, "score_setup"] = score_setup
                a_df.at[i, "score_volume"] = score_volume
                a_df.at[i, "score_risk"] = score_risk
                a_df.at[i, "signal_score"] = signal_score
            rows.append(a_df)

    if use_pattern_b:
        b_df = compute_pattern_b_signals_for_date(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            pullback_buffer_pct=float(pullback_buffer_pct),
            rebound_min_pct=float(rebound_min_pct),
        )
        if not b_df.empty:
            rows.append(b_df)

    cols = [
        "signal_date",
        "ticker",
        "pattern",
        "pattern_family",
        "entry_price",
        "stop_pct",
        "stop_price",
        "score_trend",
        "score_setup",
        "score_volume",
        "score_risk",
        "signal_score",
        "consensus_count",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.concat(rows, ignore_index=True)
    out["consensus_count"] = out.groupby(["signal_date", "ticker"])["pattern_family"].transform("nunique")

    if float(consensus_bonus) > 0:
        bonus_mask = out["consensus_count"] > 1
        out.loc[bonus_mask, "signal_score"] = out.loc[bonus_mask, "signal_score"].astype(float) + float(consensus_bonus)
        out["signal_score"] = out["signal_score"].astype(float).map(_clip_score)

    out.sort_values(["signal_date", "ticker", "signal_score"], ascending=[True, True, False], inplace=True)
    out = out.drop_duplicates(subset=["signal_date", "ticker"], keep="first")

    out = out[out["signal_score"].astype(float) >= float(min_signal_score)].copy()
    out.sort_values(["signal_date", "ticker"], inplace=True)
    return out[cols]


def backtest_signals_forward(
    signals_df: pd.DataFrame,
    prices_full: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    hold_days: int,
) -> pd.DataFrame:
    """Evaluate generated signals on future data for a fixed holding window."""

    if signals_df.empty:
        return pd.DataFrame()

    out_rows: list[dict] = []
    end_date = as_of_date + pd.Timedelta(days=hold_days)

    for _, sig in signals_df.iterrows():
        ticker = sig["ticker"]
        entry_price = float(sig["entry_price"])
        stop_price = float(sig["stop_price"])

        fut = prices_full[
            (prices_full["Ticker"] == ticker)
            & (prices_full["Date"] > as_of_date)
            & (prices_full["Date"] <= end_date)
        ].copy()
        fut.sort_values("Date", inplace=True)

        if fut.empty:
            out_rows.append(
                {
                    "ticker": ticker,
                    "entry_price": entry_price,
                    "exit_date": None,
                    "exit_price": None,
                    "outcome": "no_future_data",
                    "return_pct": None,
                }
            )
            continue

        stop_hit_rows = fut[fut["Low"] <= stop_price]
        if not stop_hit_rows.empty:
            exit_date = stop_hit_rows.iloc[0]["Date"]
            exit_price = stop_price
            outcome = "stop_hit"
            ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
        else:
            last_row = fut.iloc[-1]
            exit_date = last_row["Date"]
            exit_price = float(last_row["Close"])
            outcome = "held_to_window_end"
            ret_pct = ((exit_price - entry_price) / entry_price) * 100.0

        out_rows.append(
            {
                "ticker": ticker,
                "entry_price": round(entry_price, 4),
                "exit_date": exit_date.date().isoformat() if hasattr(exit_date, "date") else str(exit_date),
                "exit_price": round(float(exit_price), 4),
                "outcome": outcome,
                "return_pct": round(float(ret_pct), 2),
            }
        )

    return pd.DataFrame(out_rows)


def evaluate_generated_triggers(
    signals_df: pd.DataFrame,
    prices_full: pd.DataFrame,
    *,
    hold_days: int,
    break_even_trigger_pct: float | None = None,
    time_stop_days: int | None = None,
) -> pd.DataFrame:
    """Evaluate each generated trigger using future data from its own signal_date."""

    cols = [
        "signal_date",
        "ticker",
        "pattern",
        "pattern_family",
        "signal_score",
        "consensus_count",
        "entry_price",
        "stop_price",
        "exit_date",
        "exit_price",
        "outcome",
        "return_pct",
        "max_upside_pct",
        "max_drawdown_pct",
        "quality",
    ]
    if signals_df.empty:
        return pd.DataFrame(columns=cols)

    out: list[dict] = []
    for _, sig in signals_df.iterrows():
        ticker = sig["ticker"]
        entry_price = float(sig["entry_price"])
        stop_price = float(sig["stop_price"])
        sig_dt = pd.to_datetime(sig["signal_date"])
        end_dt = sig_dt + pd.Timedelta(days=int(hold_days))

        fut = prices_full[
            (prices_full["Ticker"] == ticker)
            & (prices_full["Date"] > sig_dt)
            & (prices_full["Date"] <= end_dt)
        ].copy()
        fut.sort_values("Date", inplace=True)

        if fut.empty:
            out.append(
                {
                    "signal_date": sig["signal_date"],
                    "ticker": ticker,
                    "pattern": sig["pattern"],
                    "pattern_family": sig.get("pattern_family", "A"),
                    "signal_score": sig.get("signal_score", pd.NA),
                    "consensus_count": sig.get("consensus_count", 1),
                    "entry_price": round(entry_price, 4),
                    "stop_price": round(stop_price, 4),
                    "exit_date": None,
                    "exit_price": None,
                    "outcome": "no_future_data",
                    "return_pct": None,
                    "max_upside_pct": None,
                    "max_drawdown_pct": None,
                    "quality": "?",
                }
            )
            continue

        max_upside = ((float(fut["High"].max()) - entry_price) / entry_price) * 100.0
        max_drawdown = ((float(fut["Low"].min()) - entry_price) / entry_price) * 100.0

        dynamic_stop = float(stop_price)
        moved_to_be = False
        exit_row = fut.iloc[-1]
        exit_price = float(exit_row["Close"])
        exit_date = exit_row["Date"]
        outcome = "held_to_window_end"

        for i, (_, row) in enumerate(fut.iterrows(), start=1):
            if (
                break_even_trigger_pct is not None
                and not moved_to_be
                and float(row["High"]) >= entry_price * (1.0 + float(break_even_trigger_pct) / 100.0)
            ):
                dynamic_stop = max(dynamic_stop, entry_price)
                moved_to_be = True

            if float(row["Low"]) <= dynamic_stop:
                exit_row = row
                exit_price = dynamic_stop
                exit_date = row["Date"]
                outcome = "stop_hit"
                break

            if time_stop_days is not None and i >= int(time_stop_days):
                exit_row = row
                exit_price = float(row["Close"])
                exit_date = row["Date"]
                outcome = "time_stop"
                break

        ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
        if outcome == "stop_hit":
            quality = "--"
        elif outcome == "time_stop" and ret_pct > 0:
            quality = "+"
        elif outcome == "time_stop":
            quality = "-"
        elif ret_pct >= 5.0:
            quality = "++"
        elif ret_pct > 0:
            quality = "+"
        else:
            quality = "-"

        out.append(
            {
                "signal_date": sig["signal_date"],
                "ticker": ticker,
                "pattern": sig["pattern"],
                "pattern_family": sig.get("pattern_family", "A"),
                "signal_score": sig.get("signal_score", pd.NA),
                "consensus_count": sig.get("consensus_count", 1),
                "entry_price": round(entry_price, 4),
                "stop_price": round(stop_price, 4),
                "exit_date": exit_date.date().isoformat(),
                "exit_price": round(float(exit_price), 4),
                "outcome": outcome,
                "return_pct": round(float(ret_pct), 2),
                "max_upside_pct": round(float(max_upside), 2),
                "max_drawdown_pct": round(float(max_drawdown), 2),
                "quality": quality,
            }
        )

    df = pd.DataFrame(out, columns=cols)
    df.sort_values(["signal_date", "ticker"], inplace=True)
    return df


def run_backtest_for_params(
    prices: pd.DataFrame,
    *,
    eligible_dates: list[pd.Timestamp],
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
    hold_days: int,
    breakout_buffer_pct: float = 0.0,
    use_atr_stop: bool = False,
    atr_period: int = 14,
    atr_multiplier: float = 2.5,
    break_even_trigger_pct: float | None = None,
    time_stop_days: int | None = None,
    ticker_sector_rs_df: pd.DataFrame | None = None,
    min_sector_rs20: float | None = None,
    use_pattern_a: bool = True,
    use_pattern_b: bool = False,
    pullback_buffer_pct: float = 1.5,
    rebound_min_pct: float = 0.2,
    min_signal_score: float = 0.0,
    consensus_bonus: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_signals: list[pd.DataFrame] = []
    for d in eligible_dates:
        hist_to_date = prices[prices["Date"] <= d].copy()
        day_signals = compute_scored_signals_for_date(
            hist_to_date,
            as_of_date=d,
            breakout_days=int(breakout_days),
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            use_pattern_a=bool(use_pattern_a),
            use_pattern_b=bool(use_pattern_b),
            breakout_buffer_pct=float(breakout_buffer_pct),
            use_atr_stop=bool(use_atr_stop),
            atr_period=int(atr_period),
            atr_multiplier=float(atr_multiplier),
            pullback_buffer_pct=float(pullback_buffer_pct),
            rebound_min_pct=float(rebound_min_pct),
            min_signal_score=float(min_signal_score),
            consensus_bonus=float(consensus_bonus),
        )

        if (
            min_sector_rs20 is not None
            and ticker_sector_rs_df is not None
            and not ticker_sector_rs_df.empty
            and not day_signals.empty
        ):
            day_rs = ticker_sector_rs_df[
                ticker_sector_rs_df["Date"] == pd.to_datetime(d).normalize()
            ][["ticker", "sector_rs20"]].copy()
            if not day_rs.empty:
                day_signals = day_signals.merge(day_rs, on="ticker", how="left")
                day_signals = day_signals[
                    day_signals["sector_rs20"].notna()
                    & (day_signals["sector_rs20"] >= float(min_sector_rs20))
                ].copy()
                day_signals.drop(columns=["sector_rs20"], inplace=True)

        if not day_signals.empty:
            all_signals.append(day_signals)

    if all_signals:
        bt_signals = pd.concat(all_signals, ignore_index=True)
        bt_signals.sort_values(["signal_date", "ticker"], inplace=True)
    else:
        bt_signals = pd.DataFrame(
            columns=[
                "signal_date",
                "ticker",
                "pattern",
                "pattern_family",
                "entry_price",
                "stop_pct",
                "stop_price",
                "score_trend",
                "score_setup",
                "score_volume",
                "score_risk",
                "signal_score",
                "consensus_count",
            ]
        )

    bt_eval = evaluate_generated_triggers(
        bt_signals,
        prices,
        hold_days=int(hold_days),
        break_even_trigger_pct=break_even_trigger_pct,
        time_stop_days=time_stop_days,
    )
    return bt_signals, bt_eval


def load_portfolio(path: Path = PORTFOLIO_CSV) -> pd.DataFrame:
    cols = [
        "buy_signal_date",
        "ticker",
        "pattern",
        "entry_price",
        "stop_price",
        "status",
        "entered_date",
        "closed_date",
        "last_updated",
    ]
    if not path.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


def save_portfolio(df: pd.DataFrame, path: Path = PORTFOLIO_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_dummy_lab(path: Path = DUMMY_LAB_CSV) -> pd.DataFrame:
    cols = [
        "lab_id",
        "created_at",
        "source_signal_date",
        "ticker",
        "pattern",
        "entry_price",
        "stop_price",
        "capital",
        "status",
        "note",
    ]
    if not path.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


def save_dummy_lab(df: pd.DataFrame, path: Path = DUMMY_LAB_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def enrich_dummy_lab_with_live_metrics(lab_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    if lab_df.empty:
        return lab_df.copy()

    out = lab_df.copy()
    for c in ["entry_price", "stop_price", "capital"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if prices_df.empty:
        out["latest_price_date"] = pd.NA
        out["latest_close"] = pd.NA
        out["qty"] = pd.NA
        out["current_value"] = pd.NA
        out["pnl"] = pd.NA
        out["current_return_pct"] = pd.NA
        out["distance_to_stop_pct"] = pd.NA
        return out

    latest_prices = prices_df.sort_values("Date").groupby("Ticker", as_index=False).tail(1)
    latest_prices = latest_prices[["Ticker", "Date", "Close"]].rename(
        columns={"Ticker": "ticker", "Date": "latest_price_date", "Close": "latest_close"}
    )

    out = out.merge(latest_prices, on="ticker", how="left")
    out["qty"] = out["capital"] / out["entry_price"]
    out["current_value"] = out["qty"] * out["latest_close"]
    out["pnl"] = out["current_value"] - out["capital"]
    out["current_return_pct"] = (out["pnl"] / out["capital"]) * 100.0
    out["distance_to_stop_pct"] = ((out["latest_close"] - out["stop_price"]) / out["stop_price"]) * 100.0
    return out


def sync_portfolio_with_buys(buy_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if buy_df.empty:
        return portfolio_df, 0

    out = portfolio_df.copy()
    added = 0
    existing_keys = set(
        out["buy_signal_date"].astype(str) + "|" + out["ticker"].astype(str) + "|" + out["pattern"].astype(str)
    ) if not out.empty else set()

    for _, r in buy_df.iterrows():
        k = f"{r['signal_date']}|{r['ticker']}|{r['pattern']}"
        if k in existing_keys:
            continue
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "buy_signal_date": r["signal_date"],
                            "ticker": r["ticker"],
                            "pattern": r["pattern"],
                            "entry_price": r["entry_price"],
                            "stop_price": r.get("stop_price", pd.NA),
                            "status": "New",
                            "entered_date": pd.NA,
                            "closed_date": pd.NA,
                            "last_updated": date.today().isoformat(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        existing_keys.add(k)
        added += 1

    if not out.empty:
        out.sort_values(["buy_signal_date", "ticker"], inplace=True)
    return out, added


def apply_portfolio_status(
    portfolio_df: pd.DataFrame,
    *,
    buy_signal_date: str,
    ticker: str,
    pattern: str,
    new_status: str,
) -> pd.DataFrame:
    out = portfolio_df.copy()
    mask = (
        (out["buy_signal_date"].astype(str) == str(buy_signal_date))
        & (out["ticker"].astype(str) == str(ticker))
        & (out["pattern"].astype(str) == str(pattern))
    )
    if not mask.any():
        return out

    out.loc[mask, "status"] = new_status
    out.loc[mask, "last_updated"] = date.today().isoformat()
    if new_status == "Entered":
        out.loc[mask, "entered_date"] = date.today().isoformat()
    if new_status == "Closed":
        out.loc[mask, "closed_date"] = date.today().isoformat()
    return out


def auto_close_portfolio_with_sells(portfolio_df: pd.DataFrame, sell_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if portfolio_df.empty or sell_df.empty:
        return portfolio_df, 0

    out = portfolio_df.copy()
    changed = 0

    sell_keys = set(
        sell_df["buy_signal_date"].astype(str)
        + "|"
        + sell_df["ticker"].astype(str)
        + "|"
        + sell_df["pattern"].astype(str)
    )

    for idx, row in out.iterrows():
        key = f"{row['buy_signal_date']}|{row['ticker']}|{row['pattern']}"
        if key in sell_keys and str(row.get("status", "")) != "Closed":
            out.at[idx, "status"] = "Closed"
            out.at[idx, "closed_date"] = date.today().isoformat()
            out.at[idx, "last_updated"] = date.today().isoformat()
            changed += 1

    return out, changed


def style_portfolio_status(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _row_style(row: pd.Series) -> list[str]:
        status = str(row.get("status", "")).strip()
        if status == "New":
            color = "#fef3c7"
        elif status == "Entered":
            color = "#dbeafe"
        elif status == "Closed":
            color = "#dcfce7"
        else:
            color = "#f1f5f9"
        return [f"background-color: {color}"] * len(row)

    return df.style.apply(_row_style, axis=1)


def enrich_portfolio_with_live_metrics(portfolio_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    if portfolio_df.empty or prices_df.empty:
        return portfolio_df.copy()

    latest_prices = prices_df.sort_values("Date").groupby("Ticker", as_index=False).tail(1)
    latest_prices = latest_prices[["Ticker", "Date", "Close"]].rename(
        columns={"Ticker": "ticker", "Date": "latest_price_date", "Close": "latest_close"}
    )

    out = portfolio_df.copy()
    out = out.merge(latest_prices, on="ticker", how="left")
    out["current_return_pct"] = ((out["latest_close"] - out["entry_price"]) / out["entry_price"]) * 100.0
    out["to_target_6pct"] = 6.0 - out["current_return_pct"]

    if "stop_price" in out.columns:
        out["distance_to_stop_pct"] = ((out["latest_close"] - out["stop_price"]) / out["stop_price"]) * 100.0
    else:
        out["distance_to_stop_pct"] = pd.NA

    return out


def build_needs_action_rows(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    if portfolio_df.empty:
        return portfolio_df.copy()

    out = portfolio_df.copy()
    if "to_target_6pct" not in out.columns:
        out["to_target_6pct"] = pd.NA
    if "distance_to_stop_pct" not in out.columns:
        out["distance_to_stop_pct"] = pd.NA

    needs_mask = (
        (out["status"] == "New")
        | (
            (out["status"] == "Entered")
            & (
                (out["to_target_6pct"] <= 1.0)
                | (out["distance_to_stop_pct"] <= 1.0)
            )
        )
    )
    out = out[needs_mask].copy()
    if out.empty:
        return out

    out["priority_reason"] = "Review"
    out.loc[out["status"] == "New", "priority_reason"] = "New signal"
    out.loc[(out["status"] == "Entered") & (out["to_target_6pct"] <= 1.0), "priority_reason"] = "Near +6% target"
    out.loc[(out["status"] == "Entered") & (out["distance_to_stop_pct"] <= 1.0), "priority_reason"] = "Near stop"
    out.sort_values(["status", "to_target_6pct", "distance_to_stop_pct", "buy_signal_date"], inplace=True)
    return out


def _pct_return_from_offset(series: pd.Series, offset: int) -> float | None:
    if series.empty or len(series) <= offset:
        return None
    latest = float(series.iloc[-1])
    old = float(series.iloc[-1 - offset])
    if old == 0:
        return None
    return ((latest / old) - 1.0) * 100.0


def build_market_dashboard(prices_df: pd.DataFrame) -> pd.DataFrame:
    if prices_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for ticker, g in prices_df.groupby("Ticker", sort=True):
        g = g.copy().sort_values("Date")
        close = g["Close"].astype(float)

        sma20 = close.rolling(20).mean().iloc[-1] if len(g) >= 20 else None
        sma50 = close.rolling(50).mean().iloc[-1] if len(g) >= 50 else None
        sma200 = close.rolling(200).mean().iloc[-1] if len(g) >= 200 else None

        latest_date = pd.to_datetime(g["Date"].iloc[-1]).date().isoformat()
        latest_close = float(close.iloc[-1])
        high_52w = float(g["High"].tail(252).max()) if "High" in g.columns else float(close.tail(252).max())
        low_52w = float(g["Low"].tail(252).min()) if "Low" in g.columns else float(close.tail(252).min())
        dist_high_pct = ((latest_close / high_52w) - 1.0) * 100.0 if high_52w else None

        ret_1d = _pct_return_from_offset(close, 1)
        ret_5d = _pct_return_from_offset(close, 5)
        ret_20d = _pct_return_from_offset(close, 20)
        ret_60d = _pct_return_from_offset(close, 60)

        score = 0
        if sma50 is not None and sma200 is not None and sma50 > sma200:
            score += 1
        if ret_20d is not None and ret_20d > 0:
            score += 1
        if ret_60d is not None and ret_60d > 0:
            score += 1
        if dist_high_pct is not None and dist_high_pct >= -12:
            score += 1

        if score >= 3:
            health = "Doing well"
            insight = "Trend is strong and price behavior is healthy. Keep on watchlist for future opportunities."
        elif score == 2:
            health = "Mixed"
            insight = "Signals are mixed. Wait for trend and momentum to align before fresh allocation."
        else:
            health = "Weak"
            insight = "Trend is weak right now. Better to avoid fresh long entries until structure improves."

        rows.append(
            {
                "ticker": ticker,
                "latest_date": latest_date,
                "latest_close": round(latest_close, 2),
                "ret_1d_pct": ret_1d,
                "ret_5d_pct": ret_5d,
                "ret_20d_pct": ret_20d,
                "ret_60d_pct": ret_60d,
                "sma20": round(float(sma20), 2) if sma20 is not None and pd.notna(sma20) else None,
                "sma50": round(float(sma50), 2) if sma50 is not None and pd.notna(sma50) else None,
                "sma200": round(float(sma200), 2) if sma200 is not None and pd.notna(sma200) else None,
                "high_52w": round(high_52w, 2),
                "low_52w": round(low_52w, 2),
                "dist_from_52w_high_pct": dist_high_pct,
                "health": health,
                "score": score,
                "insight": insight,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    for c in ["ret_1d_pct", "ret_5d_pct", "ret_20d_pct", "ret_60d_pct", "dist_from_52w_high_pct"]:
        if c in out.columns:
            out[c] = out[c].round(2)
    out.sort_values(["score", "ret_20d_pct", "ret_60d_pct"], ascending=False, inplace=True)
    return out


def explain_buy_signal(row: pd.Series) -> list[str]:
    checks: list[str] = []

    close = float(row.get("close", row.get("entry_price", 0.0)) or 0.0)
    sma50 = float(row.get("sma50", 0.0) or 0.0)
    sma200 = float(row.get("sma200", 0.0) or 0.0)
    prev_high = float(row.get("prev_high_close", 0.0) or 0.0)
    vol = float(row.get("volume", 0.0) or 0.0)
    vol_avg20 = float(row.get("vol_avg20", 0.0) or 0.0)

    checks.append(
        "Trend is up: SMA50 is above SMA200." if sma50 > sma200 else "Trend check failed: SMA50 is not above SMA200."
    )
    checks.append(
        "Price is above both moving averages."
        if close > sma50 and close > sma200
        else "Price check failed: close is not above both averages."
    )
    checks.append(
        "Price broke above recent high close."
        if close > prev_high
        else "Breakout check failed: close did not beat recent high close."
    )

    if vol_avg20 > 0:
        ratio = vol / vol_avg20
        checks.append(f"Volume strength: {ratio:.2f}x of 20-day average.")
    else:
        checks.append("Volume check not available.")

    return checks


def build_open_positions(buy_df: pd.DataFrame, sell_df: pd.DataFrame) -> pd.DataFrame:
    if buy_df.empty:
        return pd.DataFrame()

    buy = buy_df.copy()
    buy["buy_key"] = buy["signal_date"].astype(str) + "|" + buy["ticker"].astype(str) + "|" + buy["pattern"].astype(str)

    if sell_df.empty:
        out = buy.drop(columns=["buy_key"])
        out.sort_values(["signal_date", "ticker"], inplace=True)
        return out

    sell = sell_df.copy()
    sell["buy_key"] = sell["buy_signal_date"].astype(str) + "|" + sell["ticker"].astype(str) + "|" + sell["pattern"].astype(str)
    sold_keys = set(sell["buy_key"].tolist())

    open_df = buy[~buy["buy_key"].isin(sold_keys)].copy()
    open_df.drop(columns=["buy_key"], inplace=True)
    open_df.sort_values(["signal_date", "ticker"], inplace=True)
    return open_df


def enrich_open_positions_with_latest_return(open_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    if open_df.empty or prices_df.empty:
        return open_df

    latest_prices = prices_df.sort_values("Date").groupby("Ticker", as_index=False).tail(1)
    latest_prices = latest_prices[["Ticker", "Date", "Close"]].rename(
        columns={"Ticker": "ticker", "Date": "latest_price_date", "Close": "latest_close"}
    )
    out = open_df.merge(latest_prices, on="ticker", how="left")
    out["current_return_pct"] = ((out["latest_close"] - out["entry_price"]) / out["entry_price"]) * 100.0
    out["to_target_6pct"] = 6.0 - out["current_return_pct"]
    return out


def _init_tomorrow_ui_state() -> None:
    defaults = {
        "selected_stock": None,
        "min_score": 55,
        "sort_by": "Score (high to low)",
        "show_chart": False,
        "show_past_results": False,
        "show_watchouts": False,
        "hold_days": 15,
        "mode": "Tomorrow",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _format_pattern_name(pattern: str) -> str:
    p = str(pattern).lower()
    if "pullback" in p:
        return "Pullback rebound"
    if "breakout" in p:
        return "Breakout"
    return str(pattern).replace("_", " ").strip().title()


def _plain_reason(score: float, risk_pct: float, pattern: str) -> str:
    if score >= 75 and risk_pct <= 7:
        return "Strong setup with controlled risk."
    if "pullback" in str(pattern).lower():
        return "Uptrend stock near a pullback zone."
    if risk_pct > 9:
        return "Setup looks okay, but risk is wide."
    return "Trend and price action are still supportive."


def _build_tags(score: float, risk_pct: float, pattern: str) -> list[str]:
    tags = ["Uptrend"]
    if "breakout" in str(pattern).lower():
        tags.append("Breakout")
    if "pullback" in str(pattern).lower():
        tags.append("Pullback")
    if score >= 65:
        tags.append("Volume okay")
    if risk_pct <= 7.0:
        tags.append("Low risk")
    return tags


def _decorate_stock_rows(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return base

    out = base.copy()
    if "signal_score" not in out.columns:
        out["signal_score"] = 0.0
    out["signal_score"] = pd.to_numeric(out["signal_score"], errors="coerce").fillna(0.0)
    out["entry_price"] = pd.to_numeric(out.get("entry_price"), errors="coerce")
    out["stop_price"] = pd.to_numeric(out.get("stop_price"), errors="coerce")

    if "stop_pct" in out.columns:
        out["risk_pct"] = pd.to_numeric(out["stop_pct"], errors="coerce")
    else:
        out["risk_pct"] = ((out["entry_price"] - out["stop_price"]) / out["entry_price"]) * 100.0

    out["pattern_simple"] = out["pattern"].astype(str).map(_format_pattern_name)
    out["reason_short"] = out.apply(
        lambda r: _plain_reason(float(r.get("signal_score", 0.0)), float(r.get("risk_pct", 0.0)), str(r.get("pattern", ""))),
        axis=1,
    )
    out["tags"] = out.apply(
        lambda r: _build_tags(float(r.get("signal_score", 0.0)), float(r.get("risk_pct", 0.0)), str(r.get("pattern", ""))),
        axis=1,
    )
    return out


def _prepare_tomorrow_list(signals_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if signals_df.empty:
        return pd.DataFrame(), None

    latest_signal_date = str(signals_df["signal_date"].max())
    base = signals_df[signals_df["signal_date"] == latest_signal_date].copy()
    if base.empty:
        return pd.DataFrame(), latest_signal_date

    return _decorate_stock_rows(base), latest_signal_date


def _prepare_recent_recommendations(signals_df: pd.DataFrame, *, days: int = 7) -> pd.DataFrame:
    if signals_df.empty:
        return pd.DataFrame()

    tmp = signals_df.copy()
    tmp["signal_date_dt"] = pd.to_datetime(tmp["signal_date"], errors="coerce")
    tmp = tmp[tmp["signal_date_dt"].notna()].copy()
    if tmp.empty:
        return pd.DataFrame()

    max_dt = tmp["signal_date_dt"].max()
    min_dt = max_dt - pd.Timedelta(days=max(1, int(days) - 1))
    recent = tmp[(tmp["signal_date_dt"] >= min_dt) & (tmp["signal_date_dt"] <= max_dt)].copy()
    if recent.empty:
        return pd.DataFrame()

    if "signal_score" in recent.columns:
        recent.sort_values(["signal_date_dt", "ticker", "signal_score"], ascending=[False, True, False], inplace=True)
    else:
        recent.sort_values(["signal_date_dt", "ticker"], ascending=[False, True], inplace=True)
    recent = recent.drop_duplicates(subset=["ticker"], keep="first")
    recent.drop(columns=["signal_date_dt"], inplace=True)
    return _decorate_stock_rows(recent)


def render_header(*, latest_signal_date: str | None, total_count: int) -> None:
    st.markdown(
        """
        <style>
        .tomorrow-sticky {
            position: sticky;
            top: 0.25rem;
            z-index: 50;
            background: rgba(248, 251, 255, 0.94);
            backdrop-filter: blur(6px);
            border: 1px solid #dbe4ef;
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
        }
        .tomorrow-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            color: #0f172a;
            font-size: 1.45rem;
            margin-bottom: 0.2rem;
        }
        .tomorrow-sub {
            color: #475569;
            font-size: 0.9rem;
            margin-bottom: 0.1rem;
        }
        .tomorrow-left-list div[data-testid="stButton"] > button {
            text-align: left;
            border-radius: 14px;
            border: 1px solid #dbe4ef;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            padding-top: 0.7rem;
            padding-bottom: 0.7rem;
            white-space: pre-line;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            animation: cardIn 0.28s ease;
        }
        .tomorrow-left-list div[data-testid="stButton"] > button[kind="primary"] {
            border: 1px solid #7dd3fc;
            background: linear-gradient(180deg, #ecfeff 0%, #f8fafc 100%);
            box-shadow: 0 10px 24px rgba(2, 132, 199, 0.16);
        }
        .tomorrow-left-list div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
            border-color: #bfdbfe;
        }
        .stock-card-meta {
            border: 1px solid #dbe4ef;
            background: #ffffff;
            border-radius: 12px;
            padding: 0.55rem 0.65rem;
            margin-bottom: 0.3rem;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        }
        .stock-card-meta-selected {
            border-color: #7dd3fc;
            background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 100%);
        }
        .stock-card-line {
            color: #334155;
            font-size: 0.82rem;
            margin-top: 0.2rem;
        }
        .stock-card-reason {
            color: #1f2937;
            font-size: 0.83rem;
            margin-top: 0.22rem;
        }
        .chip-row {
            margin-top: 0.25rem;
            margin-bottom: 0.35rem;
        }
        .chip {
            display: inline-block;
            font-size: 0.74rem;
            color: #1e3a8a;
            background: #dbeafe;
            border: 1px solid #93c5fd;
            border-radius: 999px;
            padding: 0.08rem 0.45rem;
            margin-right: 0.25rem;
            margin-bottom: 0.2rem;
        }
        .reveal-wrap {
            border: 1px solid #dbe4ef;
            border-radius: 12px;
            background: #ffffff;
            padding: 0.7rem 0.8rem;
            margin-top: 0.6rem;
            animation: revealIn 0.24s ease;
        }
        @keyframes cardIn {
            from {opacity: 0; transform: translateY(5px);} 
            to {opacity: 1; transform: translateY(0);} 
        }
        @keyframes revealIn {
            from {opacity: 0; transform: translateY(8px);} 
            to {opacity: 1; transform: translateY(0);} 
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            "<div class='tomorrow-sticky'>"
            "<div class='tomorrow-title'>Stocks to check for tomorrow</div>"
            f"<div class='tomorrow-sub'>Latest signal date: {latest_signal_date or '-'} | Stocks found: {total_count}</div>"
            f"<div class='tomorrow-sub'>Production link: <a href='{PRODUCTION_APP_URL}' target='_blank'>{PRODUCTION_APP_URL}</a></div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    h1, h2, h3 = st.columns([1.0, 1.1, 1.15])
    with h1:
        mode_options = ["Tomorrow", "Backtest Lab"]
        current_mode = str(st.session_state.get("mode", "Tomorrow"))
        mode_index = mode_options.index(current_mode) if current_mode in mode_options else 0
        st.selectbox(
            "View",
            options=mode_options,
            index=mode_index,
            key="mode_selector",
        )
        st.session_state["mode"] = st.session_state.get("mode_selector", current_mode)
    with h2:
        st.slider("Minimum signal score", min_value=0, max_value=100, step=1, key="min_score")
    with h3:
        st.selectbox(
            "Sort",
            options=["Score (high to low)", "Risk (low to high)", "Ticker (A to Z)"],
            key="sort_by",
        )


def render_stock_card(row: pd.Series, *, selected: bool) -> bool:
    ticker = str(row.get("ticker", ""))
    score = float(row.get("signal_score", 0.0))
    raw_recommended_date = row.get("signal_date", "")
    recommended_date = "-"
    if pd.notna(raw_recommended_date) and str(raw_recommended_date).strip():
        parsed_date = pd.to_datetime(raw_recommended_date, errors="coerce")
        if pd.notna(parsed_date):
            recommended_date = parsed_date.strftime("%d %b %Y")
        else:
            recommended_date = str(raw_recommended_date)
    entry = float(row.get("entry_price", 0.0)) if pd.notna(row.get("entry_price")) else 0.0
    stop = float(row.get("stop_price", 0.0)) if pd.notna(row.get("stop_price")) else 0.0
    risk = float(row.get("risk_pct", 0.0)) if pd.notna(row.get("risk_pct")) else 0.0
    pattern_simple = str(row.get("pattern_simple", "-"))
    reason = str(row.get("reason_short", ""))
    tags = row.get("tags", [])

    if isinstance(tags, list):
        chips = "".join([f"<span class='chip'>{t}</span>" for t in tags])
    else:
        chips = ""

    card_css = "stock-card-meta stock-card-meta-selected" if selected else "stock-card-meta"
    st.markdown(
        (
            f"<div class='{card_css}'>"
            f"<div><strong>{ticker}</strong> | {pattern_simple}</div>"
            f"<div class='stock-card-line'>Recommended {recommended_date}</div>"
            f"<div class='stock-card-line'>Score {score:.1f} | Entry {entry:.2f} | Stop {stop:.2f} | Risk {risk:.2f}%</div>"
            f"<div class='stock-card-reason'>{reason}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='chip-row'>{chips}</div>", unsafe_allow_html=True)
    button_label = f"Selected: {ticker}" if selected else f"Select {ticker}"
    return st.button(button_label, key=f"card_{ticker}", type=("primary" if selected else "secondary"), width="stretch")


def render_stock_list(stocks_df: pd.DataFrame) -> None:
    st.markdown("### Tomorrow's stock list")
    st.markdown("<div class='tomorrow-left-list'>", unsafe_allow_html=True)
    for _, row in stocks_df.iterrows():
        ticker = str(row["ticker"])
        is_selected = str(st.session_state.get("selected_stock")) == ticker
        clicked = render_stock_card(row, selected=is_selected)
        if clicked:
            prev = st.session_state.get("selected_stock")
            st.session_state["selected_stock"] = ticker
            if prev != ticker:
                st.session_state["show_chart"] = False
                st.session_state["show_past_results"] = False
                st.session_state["show_watchouts"] = False
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _quick_check_data(ticker: str, prices_df: pd.DataFrame, selected_row: pd.Series) -> dict[str, str]:
    out = {
        "Trend": "Not enough data",
        "Above moving averages": "Not enough data",
        "Above recent high": "Not enough data",
        "Volume": "Not enough data",
        "Price stretched": "Not enough data",
        "Stop wide": "No",
    }
    risk_pct = float(selected_row.get("risk_pct", 0.0)) if pd.notna(selected_row.get("risk_pct")) else 0.0
    out["Stop wide"] = "Yes" if risk_pct > 8.0 else "No"

    t = prices_df[prices_df["Ticker"] == ticker].copy().sort_values("Date")
    if t.empty:
        return out

    t["SMA20"] = t["Close"].rolling(20).mean()
    t["SMA50"] = t["Close"].rolling(50).mean()
    t["SMA200"] = t["Close"].rolling(200).mean()
    t["VolAvg20"] = t["Volume"].rolling(20).mean()
    t["Prev40High"] = t["Close"].shift(1).rolling(40).max()
    r = t.iloc[-1]

    if pd.notna(r.get("SMA50")) and pd.notna(r.get("SMA200")):
        out["Trend"] = "Yes" if float(r["SMA50"]) > float(r["SMA200"]) else "No"
    if pd.notna(r.get("SMA50")) and pd.notna(r.get("SMA200")):
        out["Above moving averages"] = (
            "Yes" if float(r["Close"]) > float(r["SMA50"]) and float(r["Close"]) > float(r["SMA200"]) else "No"
        )
    if pd.notna(r.get("Prev40High")):
        out["Above recent high"] = "Yes" if float(r["Close"]) > float(r["Prev40High"]) else "No"
    if pd.notna(r.get("VolAvg20")) and float(r["VolAvg20"]) > 0:
        vol_ratio = float(r["Volume"]) / float(r["VolAvg20"])
        out["Volume"] = f"{vol_ratio:.2f}x"
    if pd.notna(r.get("SMA20")) and float(r["SMA20"]) > 0:
        stretched = ((float(r["Close"]) / float(r["SMA20"])) - 1.0) * 100.0
        out["Price stretched"] = "Yes" if stretched > 5.0 else "No"

    return out


def render_overview(selected_row: pd.Series) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entry", f"{float(selected_row.get('entry_price', 0.0)):.2f}")
    c2.metric("Stop", f"{float(selected_row.get('stop_price', 0.0)):.2f}")
    risk_pct = float(selected_row.get("risk_pct", 0.0)) if pd.notna(selected_row.get("risk_pct")) else 0.0
    c3.metric("Risk", f"{risk_pct:.2f}%")
    c4.metric("Score", f"{float(selected_row.get('signal_score', 0.0)):.1f}")
    st.caption(f"Why this is here: {selected_row.get('reason_short', '')}")


def render_quick_check(selected_row: pd.Series, prices_df: pd.DataFrame) -> dict[str, str]:
    st.markdown("### Quick check")
    checks = _quick_check_data(str(selected_row.get("ticker", "")), prices_df, selected_row)
    show_df = pd.DataFrame(
        [{"Item": k, "Status": v} for k, v in checks.items()]
    )
    render_table(show_df, height=250)
    return checks


def render_chart(selected_row: pd.Series, prices_df: pd.DataFrame) -> None:
    st.markdown("<div class='reveal-wrap'>", unsafe_allow_html=True)
    st.markdown("### Chart")
    ticker = str(selected_row.get("ticker", ""))
    t = prices_df[prices_df["Ticker"] == ticker].copy().sort_values("Date")
    if t.empty:
        st.info("No chart data for this stock.")
    else:
        t["SMA50"] = t["Close"].rolling(50).mean()
        t["SMA200"] = t["Close"].rolling(200).mean()
        chart_df = t.tail(220)[["Date", "Close", "SMA50", "SMA200"]].set_index("Date")
        st.line_chart(chart_df, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_past_results(selected_row: pd.Series, all_signals: pd.DataFrame, prices_df: pd.DataFrame) -> None:
    st.markdown("<div class='reveal-wrap'>", unsafe_allow_html=True)
    st.markdown("### Past results")
    ticker = str(selected_row.get("ticker", ""))
    hist = all_signals[all_signals["ticker"].astype(str) == ticker].copy().sort_values("signal_date")
    if hist.empty:
        st.info("No past rows for this stock.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.slider("Hold days", min_value=5, max_value=60, step=1, key="hold_days")
    tail_hist = hist.tail(8).copy()
    eval_df = evaluate_generated_triggers(
        tail_hist,
        prices_df,
        hold_days=int(st.session_state["hold_days"]),
    )
    if eval_df.empty:
        st.info("Not enough future bars yet for past-result view.")
    else:
        view = eval_df[["signal_date", "outcome", "return_pct", "exit_date"]].copy()
        view["outcome"] = view["outcome"].map(humanize_outcome)
        render_table(view, height=240)
    st.markdown("</div>", unsafe_allow_html=True)


def render_watchouts(selected_row: pd.Series, checks: dict[str, str]) -> None:
    st.markdown("<div class='reveal-wrap'>", unsafe_allow_html=True)
    st.markdown("### Things to watch")
    notes: list[str] = []
    if checks.get("Trend") == "No":
        notes.append("Trend is not clean right now.")
    if checks.get("Above moving averages") == "No":
        notes.append("Price is below one or both moving averages.")
    if checks.get("Above recent high") == "No":
        notes.append("Price has not cleared recent high yet.")
    if checks.get("Stop wide") == "Yes":
        notes.append("Risk is wide, so position size may need to be smaller.")
    if not notes:
        notes.append("No major warning right now. Keep normal discipline.")

    for line in notes:
        st.write(f"- {line}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_telegram_action(selected_row: pd.Series, *, allow_actions: bool) -> None:
    st.markdown("### Send to Telegram")
    ticker = str(selected_row.get("ticker", ""))
    token, chat_id = get_telegram_credentials()
    msg = (
        "Stocks to check for tomorrow\n\n"
        f"{ticker}\n"
        f"Entry: {float(selected_row.get('entry_price', 0.0)):.2f}\n"
        f"Stop: {float(selected_row.get('stop_price', 0.0)):.2f}\n"
        f"Risk: {float(selected_row.get('risk_pct', 0.0)):.2f}%\n"
        f"Score: {float(selected_row.get('signal_score', 0.0)):.1f}"
    )
    if st.button("Send to Telegram", key=f"send_selected_{ticker}", disabled=not allow_actions):
        with st.spinner("Sending..."):
            ok, out = send_telegram_message(token, chat_id, msg)
        if ok:
            st.success("Sent.")
        else:
            st.error(out)


def render_score_breakdown(selected_row: pd.Series) -> None:
    total_score = float(selected_row.get("signal_score", 0.0)) if pd.notna(selected_row.get("signal_score")) else 0.0
    trend = selected_row.get("score_trend")
    setup = selected_row.get("score_setup")
    volume = selected_row.get("score_volume")
    risk = selected_row.get("score_risk")

    has_components = all(pd.notna(v) for v in [trend, setup, volume, risk])
    if not has_components:
        st.markdown(
            "- Component scores are not available for this row.\n"
            f"- Final signal score shown: {total_score:.1f}",
        )
        return

    trend = float(trend)
    setup = float(setup)
    volume = float(volume)
    risk = float(risk)

    sma50 = selected_row.get("sma50")
    sma200 = selected_row.get("sma200")
    close = selected_row.get("close", selected_row.get("entry_price"))
    prev_high_close = selected_row.get("prev_high_close")
    volume_raw = selected_row.get("volume")
    vol_avg20 = selected_row.get("vol_avg20")
    entry_price = selected_row.get("entry_price")
    stop_price = selected_row.get("stop_price")

    trend_strength_pct = None
    setup_strength_pct = None
    volume_ratio = None
    stop_pct_eff = None

    if pd.notna(sma50) and pd.notna(sma200) and float(sma200) != 0:
        trend_strength_pct = ((float(sma50) / float(sma200)) - 1.0) * 100.0
    if pd.notna(close) and pd.notna(prev_high_close) and float(prev_high_close) != 0:
        setup_strength_pct = ((float(close) / float(prev_high_close)) - 1.0) * 100.0
    if pd.notna(volume_raw) and pd.notna(vol_avg20) and float(vol_avg20) != 0:
        volume_ratio = float(volume_raw) / float(vol_avg20)
    if pd.notna(entry_price) and pd.notna(stop_price) and float(entry_price) != 0:
        stop_pct_eff = ((float(entry_price) - float(stop_price)) / float(entry_price)) * 100.0

    c_trend = round(trend * 0.3, 1)
    c_setup = round(setup * 0.3, 1)
    c_volume = round(volume * 0.2, 1)
    c_risk = round(risk * 0.2, 1)

    running = 0.0
    lines: list[str] = []

    running = round(running + c_trend, 1)
    if trend_strength_pct is not None:
        trend_label = "high" if trend_strength_pct >= 8 else ("moderate" if trend_strength_pct >= 2 else "low")
        lines.append(
            f"- Trend strength is {trend_label} ({trend_strength_pct:.2f}% gap between SMA50 and SMA200). Trend score is {trend:.1f} after clipping to the 0-100 band, adding +{c_trend:.1f} (30%), running total {running:.1f}."
        )
    else:
        lines.append(
            f"- Trend score is {trend:.1f}. Trend inputs are limited for this row, and this still adds +{c_trend:.1f} (30%), running total {running:.1f}."
        )

    running = round(running + c_setup, 1)
    if setup_strength_pct is not None:
        setup_label = "strong" if setup_strength_pct >= 3 else ("decent" if setup_strength_pct >= 1 else "soft")
        lines.append(
            f"- Breakout setup is {setup_label} ({setup_strength_pct:.2f}% above recent reference high). Setup score is {setup:.1f} after clipping to 0-100, adding +{c_setup:.1f} (30%), running total {running:.1f}."
        )
    else:
        lines.append(
            f"- Setup score is {setup:.1f}. Setup inputs are limited for this row, and this adds +{c_setup:.1f} (30%), running total {running:.1f}."
        )

    running = round(running + c_volume, 1)
    if volume_ratio is not None:
        volume_label = "strong" if volume_ratio >= 1.8 else ("healthy" if volume_ratio >= 1.2 else "light")
        lines.append(
            f"- Volume support is {volume_label} ({volume_ratio:.2f}x of 20-day average volume). Volume score is {volume:.1f} after clipping to 0-100, adding +{c_volume:.1f} (20%), running total {running:.1f}."
        )
    else:
        lines.append(
            f"- Volume score is {volume:.1f}. Volume inputs are limited for this row, and this adds +{c_volume:.1f} (20%), running total {running:.1f}."
        )

    running = round(running + c_risk, 1)
    if stop_pct_eff is not None:
        risk_label = "tight" if stop_pct_eff <= 5 else ("balanced" if stop_pct_eff <= 8 else "wide")
        lines.append(
            f"- Stop risk is {risk_label} ({stop_pct_eff:.2f}% distance from entry to stop). Risk score is {risk:.1f} after clipping to 0-100, adding +{c_risk:.1f} (20%), running total {running:.1f}."
        )
    else:
        lines.append(
            f"- Risk score is {risk:.1f}. Risk inputs are limited for this row, and this adds +{c_risk:.1f} (20%), running total {running:.1f}."
        )

    lines.append(f"- Final signal score: {total_score:.1f}")
    st.markdown("\n".join(lines))


def render_selected_stock(
    selected_row: pd.Series,
    *,
    all_signals: pd.DataFrame,
    prices_df: pd.DataFrame,
    allow_actions: bool,
) -> None:
    ticker = str(selected_row.get("ticker", ""))
    st.markdown(f"## {ticker}")
    render_score_breakdown(selected_row)
    render_overview(selected_row)
    checks = render_quick_check(selected_row, prices_df)

    if st.button("Put dummy money in Backtesting Lab", key=f"put_dummy_money_{ticker}", width="stretch"):
        st.session_state["lab_prefill"] = {
            "ticker": ticker,
            "pattern": str(selected_row.get("pattern", "")),
            "source_signal_date": str(selected_row.get("signal_date", "")),
            "entry_price": float(selected_row.get("entry_price", 0.0)) if pd.notna(selected_row.get("entry_price")) else 0.0,
            "stop_price": float(selected_row.get("stop_price", 0.0)) if pd.notna(selected_row.get("stop_price")) else 0.0,
        }
        st.session_state["mode"] = "Backtest Lab"
        st.rerun()

    st.markdown("### Action buttons")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Show chart", key="show_chart_btn", width="stretch"):
            st.session_state["show_chart"] = not bool(st.session_state.get("show_chart", False))
            st.rerun()
    with a2:
        if st.button("Show past results", key="show_past_btn", width="stretch"):
            st.session_state["show_past_results"] = not bool(st.session_state.get("show_past_results", False))
            st.rerun()
    with a3:
        if st.button("Show things to watch", key="show_watch_btn", width="stretch"):
            st.session_state["show_watchouts"] = not bool(st.session_state.get("show_watchouts", False))
            st.rerun()

    if st.session_state.get("show_chart"):
        render_chart(selected_row, prices_df)
    if st.session_state.get("show_past_results"):
        render_past_results(selected_row, all_signals, prices_df)
    if st.session_state.get("show_watchouts"):
        render_watchouts(selected_row, checks)

    if is_remote_runtime():
        render_telegram_action(selected_row, allow_actions=allow_actions)


def render_tomorrow_screen(signals_df: pd.DataFrame, prices_df: pd.DataFrame, *, allow_actions: bool) -> None:
    stocks_df, latest_signal_date = _prepare_tomorrow_list(signals_df)
    render_header(latest_signal_date=latest_signal_date, total_count=len(stocks_df))

    if stocks_df.empty:
        st.info("No stocks found for tomorrow yet.")
        return

    stocks_df = stocks_df[stocks_df["signal_score"] >= float(st.session_state.get("min_score", 0))].copy()
    if stocks_df.empty:
        fallback_df = _prepare_recent_recommendations(signals_df, days=7)
        fallback_df = fallback_df[fallback_df["signal_score"] >= float(st.session_state.get("min_score", 0))].copy()
        if fallback_df.empty:
            st.info("No stocks match your score filter.")
            return
        stocks_df = fallback_df
        st.warning("Tomorrow picks are zero on the latest date. Showing recommended stocks from the last 7 days.")

    sort_by = str(st.session_state.get("sort_by", "Score (high to low)"))
    if sort_by == "Risk (low to high)":
        stocks_df.sort_values(["risk_pct", "signal_score"], ascending=[True, False], inplace=True)
    elif sort_by == "Ticker (A to Z)":
        stocks_df.sort_values(["ticker"], inplace=True)
    else:
        stocks_df.sort_values(["signal_score", "risk_pct"], ascending=[False, True], inplace=True)

    selected = st.session_state.get("selected_stock")
    options = stocks_df["ticker"].astype(str).tolist()
    if selected not in options:
        st.session_state["selected_stock"] = options[0]
        st.session_state["show_chart"] = False
        st.session_state["show_past_results"] = False
        st.session_state["show_watchouts"] = False

    selected_ticker = str(st.session_state.get("selected_stock"))
    selected_row = stocks_df[stocks_df["ticker"].astype(str) == selected_ticker].iloc[0]

    left, right = st.columns([1, 1.35])
    with left:
        render_stock_list(stocks_df)
    with right:
        render_selected_stock(
            selected_row,
            all_signals=signals_df,
            prices_df=prices_df,
            allow_actions=allow_actions,
        )


signals = load_signals()
sell_signals = load_sell_signals()
prices = load_prices()

# Single summary placeholder so refresh summary appears only once on page.
summary_panel = st.container()


def update_summary_panel(prices_df: pd.DataFrame, signals_df: pd.DataFrame) -> None:
    summary_panel.empty()
    with summary_panel:
        render_refresh_summary(prices_df, signals_df)


_init_tomorrow_ui_state()

# Keep tomorrow mode clean; legacy tabs remain available under other modes.
tomorrow_allow_actions = not IS_STREAMLIT_CLOUD
if st.session_state.get("mode") == "Tomorrow":
    render_tomorrow_screen(signals, prices, allow_actions=tomorrow_allow_actions)
    st.stop()

# In-page controls (kept in main area, not sidebar)
st.markdown("### Control Center")
c0, c1 = st.columns([1.1, 1.1])
with c0:
    allow_actions = st.toggle(
        "Enable refresh/trigger actions",
        value=(not IS_STREAMLIT_CLOUD),
        help="Keep OFF on Streamlit Cloud for read-only dashboard mode. Turn ON when you want this app to run local scripts.",
    )
with c1:
    compact_mode = st.toggle(
        "Compact mobile mode",
        value=False,
        help="Use tighter spacing and smaller cards for phone screens.",
    )
if compact_mode:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.8rem; padding-bottom: 0.8rem;}
        .brand-title {font-size: 1.55rem; line-height: 1.2; margin-top: 0.2rem; margin-bottom: 0.5rem; padding-top: 0.2rem;}
        .brand-roy {font-style: italic;}
        .stat-card {min-height: 66px; padding: 0.45rem 0.6rem;}
        .stat-label {font-size: 0.72rem;}
        .stat-value {font-size: 1.0rem;}
        .hero {padding: 0.65rem 0.7rem;}
        .action-item {padding: 0.45rem 0.55rem; margin-bottom: 0.35rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
if not allow_actions:
    st.info("Read-only mode: refresh and trigger generation are disabled.")

if st.session_state.get("mode") != "Tomorrow":
    if st.button("Return to Tomorrow view", key="return_tomorrow_view"):
        st.session_state["mode"] = "Tomorrow"
        st.rerun()

today_str = date.today().isoformat()
last_refresh_date = st.session_state.get("last_refresh_date")
refresh_info = get_prices_refresh_info(prices)
latest_market_date = refresh_info["latest_market_date"]

sidebar_step1_done = bool(st.session_state.get("flow_step_1_date") == today_str) or bool(last_refresh_date == today_str)
sidebar_step1_done = sidebar_step1_done or is_refreshed_today()
sidebar_step2_done = bool(st.session_state.get("flow_step_2_date") == today_str)
sidebar_step3_done = bool(st.session_state.get("flow_step_3_date") == today_str)
sidebar_step4_done = bool(st.session_state.get("flow_step_4_date") == today_str)

st.markdown("### Today")
st.caption(f"Market date: {latest_market_date}")
st.caption(f"Data file updated: {refresh_info['file_updated']}")
st.caption(f"Price rows: {refresh_info['rows']}")
if last_refresh_date:
    st.caption(f"Last app refresh click: {last_refresh_date}")

done_count = int(sidebar_step1_done) + int(sidebar_step2_done) + int(sidebar_step3_done) + int(sidebar_step4_done)
st.progress(done_count / 4.0, text=f"Flow progress: {done_count}/4")

if not sidebar_step1_done:
    next_label = "Run Step 1: Refresh data"
    next_action = "refresh"
elif not sidebar_step2_done:
    next_label = "Run Step 2: Generate signals"
    next_action = "generate"
elif not sidebar_step3_done:
    next_label = "Run Step 3: Mark review done"
    next_action = "review"
else:
    next_label = "Go to Step 4: Send summary"
    next_action = "send"

st.caption("Next step")
if st.button(next_label, disabled=(not allow_actions and next_action in {"refresh", "generate"})):
    if next_action == "refresh":
        with st.spinner("Refreshing prices..."):
            ok, msg = refresh_prices()
        if ok:
            st.session_state["last_refresh_date"] = today_str
            st.session_state["flow_step_1_date"] = today_str
            st.session_state["action_feedback"] = {
                "level": "success",
                "title": "Price refresh completed.",
                "output": msg,
            }
            st.session_state["show_refresh_actions"] = True
            st.rerun()
        else:
            st.session_state["action_feedback"] = {
                "level": "error",
                "title": msg or "Price refresh failed.",
                "output": "",
            }
            st.session_state["show_refresh_actions"] = True
            st.rerun()
    elif next_action == "generate":
        with st.spinner("Generating Pattern A triggers..."):
            ok, msg = generate_triggers()
        if ok:
            st.session_state["flow_step_2_date"] = today_str
            st.session_state["action_feedback"] = {
                "level": "success",
                "title": "Trigger generation completed.",
                "output": msg,
            }
            st.rerun()
        else:
            st.session_state["action_feedback"] = {
                "level": "error",
                "title": msg or "Trigger generation failed.",
                "output": "",
            }
            st.rerun()
    elif next_action == "review":
        st.session_state["flow_step_3_date"] = today_str
        st.success("Review marked done.")
    else:
        st.info("Open Telegram tab to send summary.")

with st.expander("Manual tools", expanded=False):
    if st.button("Show refresh details", key="show_refresh_details_sidebar", disabled=not allow_actions):
        st.session_state["show_refresh_actions"] = True
        st.rerun()
    if st.button("Open custom trigger panel", key="open_custom_trigger_panel_sidebar", disabled=not allow_actions):
        st.session_state["show_trigger_panel"] = True
        st.rerun()

with st.expander("Filters for all tabs", expanded=True):
    if "global_health_filter" not in st.session_state:
        st.session_state["global_health_filter"] = "All"
    st.session_state["global_health_filter"] = st.selectbox(
        "Category",
        options=["All", "Doing well", "Mixed", "Weak"],
        index=["All", "Doing well", "Mixed", "Weak"].index(st.session_state.get("global_health_filter", "All")),
        key="global_health_filter_select",
    )
    st.session_state["global_ticker_search"] = st.text_input(
        "Ticker search",
        value=st.session_state.get("global_ticker_search", ""),
        key="global_ticker_search_input",
    )

    focus_source: list[str] = []
    if not prices.empty:
        focus_source = sorted(prices["Ticker"].dropna().unique().tolist())
    elif not signals.empty:
        focus_source = sorted(signals["ticker"].dropna().unique().tolist())

    if focus_source:
        current_focus = st.session_state.get("focus_ticker")
        focus_idx = focus_source.index(current_focus) if current_focus in focus_source else 0
        st.session_state["focus_ticker"] = st.selectbox(
            "Focus ticker",
            options=focus_source,
            index=focus_idx,
            key="global_focus_ticker_select",
        )

if "show_refresh_actions" not in st.session_state:
    st.session_state["show_refresh_actions"] = False

if "show_trigger_panel" not in st.session_state:
    st.session_state["show_trigger_panel"] = False

if not allow_actions:
    st.session_state["show_refresh_actions"] = False
    st.session_state["show_trigger_panel"] = False

if st.session_state["show_refresh_actions"]:
    prices = load_prices()
    signals = load_signals()

    refreshed_today = is_refreshed_today()

    if refreshed_today:
        st.info("Prices were already refreshed today.")
    else:
        st.info("No app refresh action recorded for today yet (latest EOD market date may still be earlier than today).")

    update_summary_panel(prices, signals)

    feedback = st.session_state.pop("action_feedback", None)
    if feedback:
        if feedback.get("level") == "success":
            st.success(feedback.get("title", "Completed."))
        else:
            st.error(feedback.get("title", "Failed."))
        output_text = feedback.get("output", "")
        if output_text:
            with st.expander("Command output"):
                st.code(output_text, language="text")

    st.markdown("### Next action")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Repeat data refresh", key="repeat_data_refresh_btn", disabled=(not allow_actions) or refreshed_today):
            with st.spinner("Refreshing prices..."):
                ok, msg = refresh_prices()
            if ok:
                st.session_state["last_refresh_date"] = today_str
                st.session_state["flow_step_1_date"] = today_str
                st.session_state["action_feedback"] = {
                    "level": "success",
                    "title": "Price refresh completed.",
                    "output": msg,
                }
                st.rerun()
            else:
                st.session_state["action_feedback"] = {
                    "level": "error",
                    "title": msg or "Price refresh failed.",
                    "output": "",
                }
                st.rerun()
    with c2:
        if st.button("Generate trigger", key="generate_trigger_from_refresh_flow_btn", disabled=not allow_actions):
            with st.spinner("Generating Pattern A triggers..."):
                ok, msg = generate_triggers()
            if ok:
                st.session_state["flow_step_2_date"] = today_str
                st.session_state["action_feedback"] = {
                    "level": "success",
                    "title": "Trigger generation completed.",
                    "output": msg,
                }
                st.rerun()
            else:
                st.session_state["action_feedback"] = {
                    "level": "error",
                    "title": msg or "Trigger generation failed.",
                    "output": "",
                }
                st.rerun()

if st.session_state["show_trigger_panel"]:
    st.markdown("## Pattern A Trigger Parameters")
    st.markdown(
        "Adjust parameters if needed, then click Run. If unchanged, defaults are used."
    )

    p1, p2 = st.columns(2)
    with p1:
        ui_breakout_days = st.number_input(
            "Breakout days",
            min_value=5,
            max_value=200,
            value=40,
            step=1,
            key="step2_breakout_days",
        )
        ui_volume_multiplier = st.number_input(
            "Volume multiplier",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            format="%.2f",
            key="step2_volume_multiplier",
        )
    with p2:
        ui_stop_pct = st.number_input(
            "Stop %",
            min_value=1.0,
            max_value=20.0,
            value=7.0,
            step=0.5,
            format="%.1f",
            key="step2_stop_pct",
        )

        prices_for_dates = load_prices()
        use_custom_as_of = st.checkbox("Use custom As-of date", value=False, key="step2_use_custom_as_of")
        if use_custom_as_of:
            if prices_for_dates.empty:
                st.warning("prices_eod.csv not available; date picker uses today as fallback.")
                picked_date = st.date_input("As-of date", value=date.today(), key="step2_as_of_date_fallback")
                ui_as_of_date = picked_date.isoformat()
            else:
                min_date = prices_for_dates["Date"].min().date()
                max_date = prices_for_dates["Date"].max().date()
                picked_date = st.date_input(
                    "As-of date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    help="Calendar lookup for trigger run date.",
                    key="step2_as_of_date",
                )
                ui_as_of_date = picked_date.isoformat()
        else:
            ui_as_of_date = ""

    defaults_unchanged = (
        int(ui_breakout_days) == 40
        and abs(float(ui_volume_multiplier) - 1.5) < 1e-9
        and abs(float(ui_stop_pct) - 7.0) < 1e-9
        and not ui_as_of_date
    )

    if st.button("Run", key="run_trigger_btn", width="stretch", disabled=not allow_actions):
        if defaults_unchanged:
            with st.spinner("Generating Pattern A triggers (defaults)..."):
                ok, msg = generate_triggers()
            mode = "defaults"
        else:
            with st.spinner("Generating Pattern A triggers (with selected parameters)..."):
                ok, msg = generate_triggers(
                    breakout_days=int(ui_breakout_days),
                    volume_multiplier=float(ui_volume_multiplier),
                    stop_pct=float(ui_stop_pct),
                    as_of_date=ui_as_of_date or None,
                )
            mode = "custom"

        if ok:
            st.session_state["flow_step_2_date"] = today_str
            if mode == "defaults":
                st.success("Trigger generation completed using default parameters.")
            else:
                st.success("Trigger generation completed using your selected parameters.")
            if msg:
                with st.expander("Trigger command output"):
                    st.code(msg, language="text")
            signals = load_signals()
        else:
            st.error(msg or "Trigger generation failed.")

latest_trading_date_str = None
if not prices.empty:
    latest_trading_date_str = prices["Date"].max().date().isoformat()

filtered = pd.DataFrame()
selected_date = None

portfolio = load_portfolio()
portfolio, added_positions = sync_portfolio_with_buys(signals, portfolio)
portfolio, auto_closed = auto_close_portfolio_with_sells(portfolio, sell_signals)
if added_positions > 0 or auto_closed > 0:
    save_portfolio(portfolio)

portfolio_live = enrich_portfolio_with_live_metrics(portfolio, prices)
needs_action_rows = build_needs_action_rows(portfolio_live)
dummy_lab = load_dummy_lab()
dummy_lab_live = enrich_dummy_lab_with_live_metrics(dummy_lab, prices)

if st.session_state.get("mode") == "Backtest Lab":
    st.subheader("Backtesting Lab")
    st.caption("Track your own dummy-money positions. This is separate from Pattern A Backtest.")
    if st.button("Return to Tomorrow view", key="lab_return_to_tomorrow"):
        st.session_state["mode"] = "Tomorrow"
        st.rerun()

    prefill = st.session_state.get("lab_prefill", {})
    with st.form("backtesting_lab_form_direct"):
        f1, f2 = st.columns(2)
        with f1:
            ticker_in = st.text_input("Ticker", value=str(prefill.get("ticker", ""))).strip().upper()
            signal_date_in = st.text_input("Signal date", value=str(prefill.get("source_signal_date", ""))).strip()
            entry_in = st.number_input(
                "1 stock price (entry)",
                min_value=0.0,
                value=float(prefill.get("entry_price", 0.0) or 0.0),
                step=0.1,
                key="lab_direct_entry",
            )
        with f2:
            pattern_in = st.text_input("Pattern", value=str(prefill.get("pattern", ""))).strip()
            stop_in = st.number_input(
                "Stop loss",
                min_value=0.0,
                value=float(prefill.get("stop_price", 0.0) or 0.0),
                step=0.1,
                key="lab_direct_stop",
            )
            capital_in = st.number_input("Dummy money to put", min_value=100.0, value=10000.0, step=100.0, key="lab_direct_capital")

        note_in = st.text_input("Note (optional)", value="")
        submit = st.form_submit_button("See how it performs")

    if submit:
        if not ticker_in:
            st.warning("Ticker is required.")
        elif entry_in <= 0:
            st.warning("Entry price must be greater than 0.")
        elif stop_in <= 0:
            st.warning("Stop loss must be greater than 0.")
        else:
            new_row = pd.DataFrame(
                [
                    {
                        "lab_id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source_signal_date": signal_date_in or pd.NA,
                        "ticker": ticker_in,
                        "pattern": pattern_in or pd.NA,
                        "entry_price": float(entry_in),
                        "stop_price": float(stop_in),
                        "capital": float(capital_in),
                        "status": "Watching",
                        "note": note_in or pd.NA,
                    }
                ]
            )
            dummy_lab = pd.concat([dummy_lab, new_row], ignore_index=True)
            save_dummy_lab(dummy_lab)
            st.session_state.pop("lab_prefill", None)
            st.success("Added to Backtesting Lab.")
            st.rerun()

    if dummy_lab_live.empty:
        st.info("No dummy-money positions yet. Add one using the form above or from stock details.")
    else:
        open_lab = dummy_lab_live[dummy_lab_live["status"].astype(str) == "Watching"].copy()
        if open_lab.empty:
            open_lab = dummy_lab_live.copy()

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Positions", len(open_lab))
        t2.metric("Capital", f"{open_lab['capital'].sum():,.0f}")
        t3.metric("Current value", f"{open_lab['current_value'].sum():,.0f}" if "current_value" in open_lab.columns else "-")
        total_pnl = float(open_lab["pnl"].sum()) if "pnl" in open_lab.columns else 0.0
        t4.metric("P&L", f"{total_pnl:,.2f}")

        show_cols = [
            "created_at",
            "source_signal_date",
            "ticker",
            "pattern",
            "entry_price",
            "stop_price",
            "latest_close",
            "capital",
            "current_value",
            "pnl",
            "current_return_pct",
            "distance_to_stop_pct",
            "status",
            "note",
        ]
        show_cols = [c for c in show_cols if c in open_lab.columns]
        view_df = open_lab[show_cols].copy()
        for c in ["entry_price", "stop_price", "latest_close", "capital", "current_value", "pnl", "current_return_pct", "distance_to_stop_pct"]:
            if c in view_df.columns:
                view_df[c] = pd.to_numeric(view_df[c], errors="coerce").round(2)
        render_table(view_df.sort_values(["created_at", "ticker"], ascending=[False, True]), height=360)

        st.markdown("### Manage lab positions")
        sel_df = open_lab.copy()
        sel_df["label"] = sel_df["created_at"].astype(str) + " | " + sel_df["ticker"].astype(str) + " | " + sel_df["status"].astype(str)
        selected_label = st.selectbox("Choose row", options=sel_df["label"].tolist(), key="lab_row_select_direct")
        selected_row = sel_df[sel_df["label"] == selected_label].iloc[0]

        c_close, c_reopen = st.columns(2)
        with c_close:
            if st.button("Mark Closed", key="lab_mark_closed_direct"):
                mask = dummy_lab["lab_id"].astype(str) == str(selected_row["lab_id"])
                dummy_lab.loc[mask, "status"] = "Closed"
                save_dummy_lab(dummy_lab)
                st.success("Marked as Closed.")
                st.rerun()
        with c_reopen:
            if st.button("Mark Watching", key="lab_mark_watching_direct"):
                mask = dummy_lab["lab_id"].astype(str) == str(selected_row["lab_id"])
                dummy_lab.loc[mask, "status"] = "Watching"
                save_dummy_lab(dummy_lab)
                st.success("Marked as Watching.")
                st.rerun()

    st.stop()

if "focus_ticker" not in st.session_state and not needs_action_rows.empty:
    st.session_state["focus_ticker"] = str(needs_action_rows.iloc[0]["ticker"])

step1_done = bool(st.session_state.get("flow_step_1_date") == today_str) or bool(last_refresh_date == today_str)
step2_done = bool(st.session_state.get("flow_step_2_date") == today_str)
step3_done = bool(st.session_state.get("flow_step_3_date") == today_str)
step4_done = bool(st.session_state.get("flow_step_4_date") == today_str)

render_flow_header(
    step1_done=step1_done,
    step2_done=step2_done,
    step3_done=step3_done,
    step4_done=step4_done,
)

market_tab, dashboard_tab, signals_tab, portfolio_tab, backtest_tab, backtest_lab_tab, telegram_tab = st.tabs(["Market Dashboard", "Dashboard", "Signals", "Portfolio", "Backtesting", "Backtesting Lab", "Telegram"])

with market_tab:
    st.subheader("All Stocks Dashboard")
    st.caption("Simple view: which stocks are strong, mixed, or weak based on trend and momentum.")
    st.info("Next step: narrow with Category/Ticker filters, then pick one focus stock for Signals and Portfolio tabs.")

    if prices.empty:
        st.warning("Price data is not available. Run refresh first.")
    else:
        market_df = build_market_dashboard(prices)
        if market_df.empty:
            st.info("No stock rows available.")
        else:
            total_stocks = len(market_df)
            doing_well = int((market_df["health"] == "Doing well").sum())
            mixed = int((market_df["health"] == "Mixed").sum())
            weak = int((market_df["health"] == "Weak").sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total stocks", total_stocks)
            m2.metric("Doing well", doing_well)
            m3.metric("Mixed", mixed)
            m4.metric("Weak", weak)

            top_winners = market_df.sort_values("ret_20d_pct", ascending=False).head(5)
            top_losers = market_df.sort_values("ret_20d_pct", ascending=True).head(5)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Top 20-day Winners")
                render_table(top_winners[["ticker", "ret_20d_pct", "ret_60d_pct", "health", "score"]], height=240)
            with c2:
                st.markdown("### Top 20-day Laggards")
                render_table(top_losers[["ticker", "ret_20d_pct", "ret_60d_pct", "health", "score"]], height=240)

            filter_col1, filter_col2 = st.columns([1.2, 1.8])
            with filter_col1:
                health_filter = st.session_state.get("global_health_filter", "All")
                st.caption(f"Category filter: {health_filter}")
            with filter_col2:
                ticker_search = st.session_state.get("global_ticker_search", "")
                st.caption(f"Ticker search: {ticker_search if ticker_search else 'None'}")

            market_view = market_df.copy()
            if health_filter != "All":
                market_view = market_view[market_view["health"] == health_filter]
            if ticker_search.strip():
                market_view = market_view[
                    market_view["ticker"].str.contains(ticker_search.strip(), case=False, na=False)
                ]

            st.markdown("### Full Stock List")
            view_cols = [
                "ticker",
                "health",
                "score",
                "latest_close",
                "ret_1d_pct",
                "ret_5d_pct",
                "ret_20d_pct",
                "ret_60d_pct",
                "dist_from_52w_high_pct",
                "insight",
            ]
            render_table(market_view[view_cols], height=420)

            st.markdown("### Stock Insight")
            pick = st.selectbox(
                "Choose stock",
                options=market_df["ticker"].tolist(),
                key="market_pick_stock",
                help="Pick a stock to view plain-language insight and trend chart below.",
            )
            pick_row = market_df[market_df["ticker"] == pick].iloc[0]
            st.info(
                f"{pick}: {pick_row['health']} | 20D return {pick_row['ret_20d_pct']}% | "
                f"60D return {pick_row['ret_60d_pct']}% | Insight: {pick_row['insight']}"
            )

            stock_hist = prices[prices["Ticker"] == pick].copy().sort_values("Date")
            if not stock_hist.empty:
                stock_hist["SMA50"] = stock_hist["Close"].rolling(50).mean()
                stock_hist["SMA200"] = stock_hist["Close"].rolling(200).mean()
                recent = stock_hist.tail(200)
                chart_df = recent[["Date", "Close", "SMA50", "SMA200"]].set_index("Date")
                st.line_chart(chart_df, width="stretch")

            st.caption("For research and learning only. This is not financial advice.")

with dashboard_tab:
    st.markdown(
        """
        <div class='hero'>
            <div class='hero-title'>Today at a glance</div>
            <div class='hero-sub'>See new buys, new sells, and open positions in one place.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Data file updated: {refresh_info['file_updated']} | "
        f"Latest market date: {refresh_info['latest_market_date']}"
    )
    st.info("Next step: review Today Action List, set one Focus ticker, then validate in Signals tab.")
    if auto_closed > 0:
        st.info(f"Auto update: {auto_closed} position(s) moved to Closed because sell signals were found.")

    latest_price_date = "-"
    latest_buy_date = "-"
    latest_sell_date = "-"
    if not prices.empty:
        latest_price_date = prices["Date"].max().date().isoformat()
    if not signals.empty:
        latest_buy_date = str(signals["signal_date"].max())
    if not sell_signals.empty:
        latest_sell_date = str(sell_signals["sell_signal_date"].max())

    latest_buy_rows = signals[signals["signal_date"] == latest_buy_date].copy() if not signals.empty else pd.DataFrame()
    latest_sell_rows = sell_signals[sell_signals["sell_signal_date"] == latest_sell_date].copy() if not sell_signals.empty else pd.DataFrame()

    open_positions = build_open_positions(signals, sell_signals)
    open_positions = enrich_open_positions_with_latest_return(open_positions, prices)
    nearing_target = 0
    if not open_positions.empty and "to_target_6pct" in open_positions.columns:
        nearing_target = int((open_positions["to_target_6pct"] <= 1.0).sum())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_stat_card("Latest market date", latest_price_date)
    with m2:
        render_stat_card("New buy signals", str(len(latest_buy_rows)))
    with m3:
        render_stat_card("New sell signals", str(len(latest_sell_rows)))
    with m4:
        render_stat_card("Open positions", str(len(open_positions)))

    left, right = st.columns([1.2, 1.0])
    with left:
        st.subheader("Action center")
        st.markdown(
            (
                "<div class='action-item'><div class='action-title'>Sell now</div>"
                f"<div class='action-value'>{len(latest_sell_rows)}</div></div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                "<div class='action-item'><div class='action-title'>New buy ideas</div>"
                f"<div class='action-value'>{len(latest_buy_rows)}</div></div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                "<div class='action-item'><div class='action-title'>Close to +6% target</div>"
                f"<div class='action-value'>{nearing_target}</div></div>"
            ),
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Changes")
        st.markdown(f"- Latest buy date: **{latest_buy_date}**")
        st.markdown(f"- Latest sell date: **{latest_sell_date}**")
        st.markdown(f"- Open positions tracked: **{len(open_positions)}**")
        if prices.empty:
            st.warning("Price data is missing. Run refresh first.")

    row_a, row_b = st.columns(2)
    with row_a:
        st.markdown("### Buy signals (latest date)")
        if latest_buy_rows.empty:
            st.info("No buy signals on latest date.")
        else:
            latest_buy_rows = latest_buy_rows.sort_values(["ticker"])
            render_table(latest_buy_rows, height=260)
    with row_b:
        st.markdown("### Sell signals (latest date)")
        if latest_sell_rows.empty:
            st.info("No sell signals yet.")
        else:
            latest_sell_rows = latest_sell_rows.sort_values(["ticker"])
            render_table(latest_sell_rows, height=260)

    st.markdown("### Open positions")
    if open_positions.empty:
        st.info("No open positions.")
    else:
        view_cols = [
            "signal_date",
            "ticker",
            "entry_price",
            "stop_price",
            "latest_close",
            "current_return_pct",
            "to_target_6pct",
        ]
        view_cols = [c for c in view_cols if c in open_positions.columns]
        show_open = open_positions[view_cols].copy().sort_values(["signal_date", "ticker"])
        if "current_return_pct" in show_open.columns:
            show_open["current_return_pct"] = show_open["current_return_pct"].round(2)
        if "to_target_6pct" in show_open.columns:
            show_open["to_target_6pct"] = show_open["to_target_6pct"].round(2)
        render_table(show_open, height=300)

    st.markdown("### Today Action List")
    if needs_action_rows.empty:
        st.info("No urgent rows right now.")
    else:
        top_cols = [
            "buy_signal_date",
            "ticker",
            "status",
            "priority_reason",
            "current_return_pct",
            "to_target_6pct",
            "distance_to_stop_pct",
        ]
        top_cols = [c for c in top_cols if c in needs_action_rows.columns]
        top5 = needs_action_rows[top_cols].head(10).copy()
        for c in ["current_return_pct", "to_target_6pct", "distance_to_stop_pct"]:
            if c in top5.columns:
                top5[c] = top5[c].round(2)

        action_left, action_right = st.columns([1.3, 1.0])
        with action_left:
            render_table(top5, height=280)
        with action_right:
            options = top5["ticker"].astype(str).unique().tolist()
            current_focus = st.session_state.get("focus_ticker")
            default_focus = 0
            if current_focus in options:
                default_focus = options.index(current_focus)
            focus_pick = st.selectbox(
                "Focus ticker for other tabs",
                options=options,
                index=default_focus,
                key="dashboard_focus_ticker_pick",
                help="This keeps one ticker synced across Dashboard, Signals, and Portfolio flows.",
            )
            if st.button("Use this ticker in Signals/Portfolio", key="set_global_focus_ticker"):
                st.session_state["focus_ticker"] = focus_pick
                st.success(f"Focus ticker set to {focus_pick}.")
            if st.button("Mark action review done today", key="mark_review_done"):
                st.session_state["flow_step_3_date"] = today_str
                st.success("Step 3 completed for today.")

with signals_tab:
    if signals.empty:
        st.warning(
            "No signals yet. Run refresh and trigger steps first."
        )
    else:
        render_glossary(section="signals")
        st.markdown("#### Signal filters")
        st.info("Tip: default view shows only current-date signals. Turn on historical mode when you want context.")
        sf1, sf2, sf3, sf4 = st.columns([1.0, 1.2, 1.8, 1.2])
        with sf1:
            include_historical_signals = st.checkbox(
                "Include historical signals",
                value=False,
                key="signals_show_old_signals",
                help="Turn on to browse older signal dates. Keep off for clean daily-action view.",
            )

        signal_dates = sorted(signals["signal_date"].unique())
        current_signal_date = latest_trading_date_str or today_str

        # Default behavior: focus only on current date signals.
        if include_historical_signals:
            date_options = ["All signal dates"] + signal_dates
            default_date = "All signal dates"
        else:
            date_options = [current_signal_date]
            default_date = current_signal_date

        with sf2:
            selected_date = st.selectbox(
                "Signal date",
                options=date_options,
                index=date_options.index(default_date),
                key="signals_date_filter",
                help="Choose one date for precise review, or select all dates in historical mode.",
            )

        all_tickers = sorted(signals["ticker"].unique())
        global_search = st.session_state.get("global_ticker_search", "").strip()
        if global_search:
            all_tickers = [t for t in all_tickers if global_search.lower() in str(t).lower()]
            if not all_tickers:
                all_tickers = sorted(signals["ticker"].unique())
        with sf3:
            selected_tickers = st.multiselect(
                "Tickers",
                options=all_tickers,
                default=all_tickers,
                key="signals_tickers_filter",
                help="Filter down to specific symbols for a focused action list.",
            )

        all_patterns = sorted(signals["pattern"].unique())
        with sf4:
            selected_patterns = st.multiselect(
                "Patterns",
                options=all_patterns,
                default=all_patterns,
                key="signals_patterns_filter",
                help="Limit to one setup type when comparing consistency.",
            )

        filtered = signals.copy()
        if selected_date != "All signal dates":
            filtered = filtered[filtered["signal_date"] == selected_date]
        if selected_tickers:
            filtered = filtered[filtered["ticker"].isin(selected_tickers)]
        if selected_patterns:
            filtered = filtered[filtered["pattern"].isin(selected_patterns)]

        if not include_historical_signals:
            st.caption("Showing current date only. Turn on 'Include historical signals' to browse older dates.")
        else:
            st.caption("Showing historical signal view. Choose a date or use 'All signal dates'.")

        buy_view_tab, sell_view_tab, chart_view_tab = st.tabs(["Buy Signals", "Sell Signals", "Price Chart"])

        with buy_view_tab:
            buy_title = selected_date if selected_date != "All signal dates" else "all signal dates"
            st.subheader(f"Buy signals for {buy_title}")
            if latest_trading_date_str and selected_date == latest_trading_date_str and filtered.empty:
                st.info("No buy signal on latest market date.")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("# Signals", len(filtered))
            with col2:
                st.metric("# Tickers", filtered["ticker"].nunique())
            with col3:
                st.metric("Patterns", ", ".join(sorted(filtered["pattern"].unique())) or "-")
            buy_out = filtered.sort_values(["ticker"]).copy()
            render_table(buy_out, height=360)
            st.download_button(
                "Download buy signals CSV",
                data=to_csv_bytes(buy_out),
                file_name=(
                    "buy_signals_all_dates.csv"
                    if selected_date == "All signal dates"
                    else f"buy_signals_{selected_date}.csv"
                ),
                mime="text/csv",
                key="download_buy_signals_csv",
            )

            st.markdown("#### Why this buy signal?")
            if filtered.empty:
                st.info("No rows to explain.")
            else:
                explain_options = sorted(filtered["ticker"].unique())
                focus = st.session_state.get("focus_ticker")
                explain_idx = explain_options.index(focus) if focus in explain_options else 0
                explain_ticker = st.selectbox(
                    "Choose ticker",
                    options=explain_options,
                    index=explain_idx,
                    key="explain_buy_ticker",
                )
                explain_row = filtered[filtered["ticker"] == explain_ticker].iloc[0]
                for line in explain_buy_signal(explain_row):
                    st.write(f"- {line}")

        with sell_view_tab:
            st.subheader("Sell signal history (+6% target)")
            if sell_signals.empty:
                st.info("No sell signals yet.")
            else:
                include_historical_sell_signals = st.checkbox(
                    "Include historical sell signals",
                    value=False,
                    key="signals_show_old_sell_signals",
                    help="Turn on to analyze older sell outcomes and exit behavior.",
                )

                sell_dates_all = sorted(sell_signals["sell_signal_date"].unique())
                current_sell_date = latest_trading_date_str or today_str

                if include_historical_sell_signals:
                    sell_dates = sell_dates_all.copy()
                    if current_sell_date not in sell_dates:
                        sell_dates.append(current_sell_date)
                        sell_dates = sorted(sell_dates)
                    default_sell_date = current_sell_date if current_sell_date in sell_dates else sell_dates[-1]
                else:
                    sell_dates = [current_sell_date]
                    default_sell_date = current_sell_date

                chosen_sell_date = st.selectbox(
                    "Sell signal date",
                    options=sell_dates,
                    index=sell_dates.index(default_sell_date),
                    key="sell_signal_date_filter",
                    help="Review exits for a specific day to understand realized outcomes.",
                )
                sell_filtered = sell_signals[sell_signals["sell_signal_date"] == chosen_sell_date].copy()
                if not include_historical_sell_signals:
                    st.caption("Showing current date only. Turn on 'Include historical sell signals' to browse older dates.")
                s1, s2, s3 = st.columns(3)
                s1.metric("# Sell Signals", len(sell_filtered))
                s2.metric("# Tickers", sell_filtered["ticker"].nunique())
                s3.metric("Avg Realized Return %", f"{sell_filtered['realized_return_pct'].mean():.2f}")
                render_table(sell_filtered.sort_values(["ticker"]), height=340)
                st.download_button(
                    "Download sell signals CSV",
                    data=to_csv_bytes(sell_filtered.sort_values(["ticker"])),
                    file_name=f"sell_signals_{chosen_sell_date}.csv",
                    mime="text/csv",
                    key="download_sell_signals_csv",
                )

        with chart_view_tab:
            st.subheader("Price chart for a selected signal")

            if prices.empty or filtered.empty:
                st.info("Price history or filtered buy signals are not available for charting.")
            else:
                tickers_for_chart = sorted(filtered["ticker"].unique())
                focus = st.session_state.get("focus_ticker")
                chart_idx = tickers_for_chart.index(focus) if focus in tickers_for_chart else 0
                chart_ticker = st.selectbox("Ticker", options=tickers_for_chart, index=chart_idx)

                t_prices = prices[prices["Ticker"] == chart_ticker].copy()
                if not t_prices.empty:
                    t_prices.sort_values("Date", inplace=True)
                    recent = t_prices.tail(120)
                    st.line_chart(
                        recent.set_index("Date")["Close"],
                        width="stretch",
                    )
                else:
                    st.info("No price history found for this ticker in prices_eod.csv.")

with portfolio_tab:
    st.subheader("Portfolio")
    st.caption("Track each buy signal as New, Entered, or Closed.")
    st.info("Flow: New -> Entered -> Closed. Use Quick filter to prioritize only rows needing attention.")

    if portfolio.empty:
        st.info("No portfolio rows yet. New rows appear after buy signals are generated.")
    else:
        p1, p2, p3 = st.columns(3)
        p1.metric("New", int((portfolio["status"] == "New").sum()))
        p2.metric("Entered", int((portfolio["status"] == "Entered").sum()))
        p3.metric("Closed", int((portfolio["status"] == "Closed").sum()))

        status_filter = st.multiselect(
            "Show status",
            options=["New", "Entered", "Closed"],
            default=["New", "Entered", "Closed"],
            key="portfolio_status_filter",
            help="Pick which lifecycle states to include in the table.",
        )
        shown = portfolio_live[portfolio_live["status"].isin(status_filter)].copy()

        quick_filter = st.selectbox(
            "Quick filter",
            options=["All", "Needs action", "Near target", "Stop risk"],
            index=0,
            key="portfolio_quick_filter",
            help="Needs action: New rows, or Entered rows near target/stop.",
        )

        if quick_filter == "Needs action":
            base_needs = needs_action_rows.copy()
            shown = base_needs[base_needs["status"].isin(status_filter)].copy()
        elif quick_filter == "Near target":
            shown = shown[(shown["status"] == "Entered") & (shown["to_target_6pct"] <= 1.0)]
        elif quick_filter == "Stop risk":
            shown = shown[(shown["status"] == "Entered") & (shown["distance_to_stop_pct"] <= 1.0)]

        shown.sort_values(["buy_signal_date", "ticker"], inplace=True)
        render_table(style_portfolio_status(shown), height=360)
        st.download_button(
            "Download portfolio CSV",
            data=to_csv_bytes(shown),
            file_name="portfolio_view.csv",
            mime="text/csv",
            key="download_portfolio_csv",
        )
        st.download_button(
            "Download needs action CSV",
            data=to_csv_bytes(needs_action_rows),
            file_name="portfolio_needs_action.csv",
            mime="text/csv",
            key="download_portfolio_needs_action_csv",
        )

        st.markdown("### Update Position Status")
        if shown.empty:
            st.info("No rows for selected status filter.")
        else:
            shown = shown.copy()
            shown["label"] = (
                shown["buy_signal_date"].astype(str)
                + " | "
                + shown["ticker"].astype(str)
                + " | "
                + shown["pattern"].astype(str)
                + " | "
                + shown["status"].astype(str)
            )
            labels = shown["label"].tolist()
            focus = st.session_state.get("focus_ticker")
            chosen_idx = 0
            if focus:
                for i, label in enumerate(labels):
                    if f" | {focus} | " in label:
                        chosen_idx = i
                        break
            chosen = st.selectbox(
                "Choose row",
                options=labels,
                index=chosen_idx,
                key="portfolio_row",
                help="Pick one position row, then update status to match your actual trade state.",
            )
            selected = shown[shown["label"] == chosen].iloc[0]

            q1, q2, q3 = st.columns(3)
            with q1:
                if st.button("Mark Entered", key="mark_entered_btn", disabled=not allow_actions):
                    portfolio = apply_portfolio_status(
                        portfolio,
                        buy_signal_date=str(selected["buy_signal_date"]),
                        ticker=str(selected["ticker"]),
                        pattern=str(selected["pattern"]),
                        new_status="Entered",
                    )
                    save_portfolio(portfolio)
                    st.success("Updated to Entered.")
                    st.rerun()
            with q2:
                if st.button("Mark Closed", key="mark_closed_btn", disabled=not allow_actions):
                    portfolio = apply_portfolio_status(
                        portfolio,
                        buy_signal_date=str(selected["buy_signal_date"]),
                        ticker=str(selected["ticker"]),
                        pattern=str(selected["pattern"]),
                        new_status="Closed",
                    )
                    save_portfolio(portfolio)
                    st.success("Updated to Closed.")
                    st.rerun()
            with q3:
                if st.button("Mark New", key="mark_new_btn", disabled=not allow_actions):
                    portfolio = apply_portfolio_status(
                        portfolio,
                        buy_signal_date=str(selected["buy_signal_date"]),
                        ticker=str(selected["ticker"]),
                        pattern=str(selected["pattern"]),
                        new_status="New",
                    )
                    save_portfolio(portfolio)
                    st.success("Updated to New.")
                    st.rerun()

with backtest_tab:
    st.subheader("Pattern A Backtest")
    st.info("Use strict mode for cleaner setups (often fewer trades, potentially lower stop-hit rate).")
    render_glossary(section="backtest")

    if prices.empty:
        st.warning("prices_eod.csv is missing or empty. Refresh prices first.")
    else:
        b1, b2 = st.columns(2)
        with b1:
            bt_hide_months = st.slider(
                "Hide latest months",
                min_value=1,
                max_value=12,
                value=2,
                step=1,
                help="How many recent months to hide before generating historical trigger(s).",
            )
            bt_breakout_days = st.number_input(
                "Lookback window (breakout days)", min_value=5, max_value=200, value=40, step=1, key="bt_breakout_days",
                help="Price must break above the highest close in this lookback window.",
            )
            bt_volume_multiplier = st.number_input(
                "Volume strength (x of 20D avg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1, format="%.2f", key="bt_volume_multiplier",
                help="Higher value means stricter confirmation from participation.",
            )
            bt_use_strict_mode = st.toggle(
                "Use Pattern A+ strict mode",
                value=True,
                key="bt_use_strict_mode",
                help="Adds breakout buffer, ATR stop, break-even protection, and time-stop exit.",
            )
            bt_use_pattern_a = st.checkbox(
                "Enable Pattern A (breakout)",
                value=True,
                key="bt_use_pattern_a",
                help="Classic trend breakout pattern.",
            )
            bt_use_pattern_b = st.checkbox(
                "Enable Pattern B (pullback rebound)",
                value=True,
                key="bt_use_pattern_b",
                help="Trend pullback near SMA20 followed by a rebound day.",
            )
        with b2:
            bt_stop_pct = st.number_input(
                "Initial risk limit % (stop)",
                min_value=1.0,
                max_value=20.0,
                value=7.0,
                step=0.5,
                format="%.1f",
                key="bt_stop_pct",
                help="Fallback fixed stop. In strict mode with ATR stop, effective stop may be wider/tighter.",
            )
            bt_hold_days = st.slider(
                "Forward evaluation window (days)", min_value=5, max_value=60, value=15, step=1, key="bt_hold_days"
            )
            bt_breakout_buffer_pct = st.number_input(
                "Breakout buffer %",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.1,
                format="%.1f",
                key="bt_breakout_buffer_pct",
                disabled=not bt_use_strict_mode,
                help="Requires breakout to clear the prior high by this extra margin.",
            )
            bt_use_atr_stop = st.checkbox(
                "Use ATR stop",
                value=True,
                key="bt_use_atr_stop",
                disabled=not bt_use_strict_mode,
                help="Adaptive stop based on volatility instead of fixed-only stop distance.",
            )
            bt_atr_period = st.number_input(
                "ATR period",
                min_value=5,
                max_value=50,
                value=14,
                step=1,
                key="bt_atr_period",
                disabled=not bt_use_strict_mode,
                help="Volatility lookback used for ATR stop calculation.",
            )
            bt_atr_multiplier = st.number_input(
                "ATR multiplier",
                min_value=1.0,
                max_value=5.0,
                value=2.5,
                step=0.1,
                format="%.1f",
                key="bt_atr_multiplier",
                disabled=not bt_use_strict_mode,
                help="Higher multiplier gives wider ATR-based stop distance.",
            )
            bt_break_even_trigger_pct = st.number_input(
                "Break-even trigger %",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                format="%.1f",
                key="bt_break_even_trigger_pct",
                disabled=not bt_use_strict_mode,
                help="When price rises this much, stop moves up to entry price.",
            )
            bt_time_stop_days = st.number_input(
                "Time-stop days",
                min_value=3,
                max_value=60,
                value=10,
                step=1,
                key="bt_time_stop_days",
                disabled=not bt_use_strict_mode,
                help="Force exit after this many forward bars if still open.",
            )
            bt_pullback_buffer_pct = st.number_input(
                "Pattern B pullback buffer %",
                min_value=0.2,
                max_value=5.0,
                value=1.5,
                step=0.1,
                format="%.1f",
                key="bt_pullback_buffer_pct",
                disabled=not bt_use_pattern_b,
                help="Pattern B requires close to be near/under SMA20 within this tolerance.",
            )
            bt_rebound_min_pct = st.number_input(
                "Pattern B rebound min %",
                min_value=0.0,
                max_value=3.0,
                value=0.2,
                step=0.1,
                format="%.1f",
                key="bt_rebound_min_pct",
                disabled=not bt_use_pattern_b,
                help="Pattern B rebound confirmation versus prior close.",
            )
            bt_min_signal_score = st.slider(
                "Minimum signal score",
                min_value=0,
                max_value=100,
                value=55,
                step=1,
                key="bt_min_signal_score",
                help="Only keep signals with score at or above this threshold.",
            )
            bt_consensus_bonus = st.number_input(
                "Consensus bonus (A and B agree)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                format="%.1f",
                key="bt_consensus_bonus",
                disabled=not (bt_use_pattern_a and bt_use_pattern_b),
                help="Adds bonus points when both patterns trigger on the same ticker/date.",
            )
            st.caption("Trigger generation uses only data up to each trigger date (no look-ahead).")

        st.markdown("### Optional External Filters")
        bt_use_external_filters = st.toggle(
            "Use external-factor regime filters",
            value=False,
            key="bt_use_external_filters",
            help=(
                "Applies date-level market regime filters from stock_triggers/data/external_factors.csv "
                "and optional sector-strength filter from stock_triggers/data/ticker_sector_map.csv."
            ),
        )

        bt_max_vix = 99.0
        bt_max_vix_spike = 100.0
        bt_max_usdinr_5d = 100.0
        bt_max_brent_5d = 100.0
        bt_min_fii_dii = -999999.0
        bt_use_sector_rs_filter = False
        bt_sector_lookback_days = 20
        bt_min_sector_rs20 = -100.0

        if bt_use_external_filters:
            ef1, ef2 = st.columns(2)
            with ef1:
                bt_max_vix = st.number_input(
                    "Max India VIX close",
                    min_value=5.0,
                    max_value=100.0,
                    value=22.0,
                    step=0.5,
                    format="%.1f",
                    key="bt_max_vix",
                    help="Skip signal dates when India VIX is above this level.",
                )
                bt_max_vix_spike = st.number_input(
                    "Max VIX 1D spike %",
                    min_value=0.0,
                    max_value=100.0,
                    value=8.0,
                    step=0.5,
                    format="%.1f",
                    key="bt_max_vix_spike",
                    help="Skip dates with sharp one-day volatility jumps.",
                )
                bt_max_usdinr_5d = st.number_input(
                    "Max USDINR 5D change %",
                    min_value=-10.0,
                    max_value=10.0,
                    value=1.2,
                    step=0.1,
                    format="%.1f",
                    key="bt_max_usdinr_5d",
                    help="Skip dates when INR weakens too much over 5 days.",
                )
            with ef2:
                bt_max_brent_5d = st.number_input(
                    "Max Brent 5D change %",
                    min_value=-20.0,
                    max_value=20.0,
                    value=6.0,
                    step=0.5,
                    format="%.1f",
                    key="bt_max_brent_5d",
                    help="Skip dates with sharp crude rises that can stress risk assets.",
                )
                bt_min_fii_dii = st.number_input(
                    "Min FII+DII net flow (Cr)",
                    min_value=-50000.0,
                    max_value=50000.0,
                    value=-1000.0,
                    step=100.0,
                    format="%.0f",
                    key="bt_min_fii_dii",
                    help="Skip dates with very negative combined institutional flow.",
                )
                bt_use_sector_rs_filter = st.checkbox(
                    "Use sector relative-strength filter",
                    value=False,
                    key="bt_use_sector_rs_filter",
                    help="Requires ticker_sector_map.csv with columns: ticker, sector.",
                )
                bt_sector_lookback_days = st.number_input(
                    "Sector RS lookback days",
                    min_value=10,
                    max_value=120,
                    value=20,
                    step=1,
                    key="bt_sector_lookback_days",
                    disabled=not bt_use_sector_rs_filter,
                    help="Lookback used to compare each sector return vs market average return.",
                )
                bt_min_sector_rs20 = st.number_input(
                    "Min sector RS vs market %",
                    min_value=-20.0,
                    max_value=20.0,
                    value=0.0,
                    step=0.2,
                    format="%.1f",
                    key="bt_min_sector_rs20",
                    disabled=not bt_use_sector_rs_filter,
                    help="Keep signals only when sector outperforms market by at least this amount.",
                )

            st.caption(
                "Expected files: stock_triggers/data/external_factors.csv (Date + factor columns), "
                "and optional stock_triggers/data/ticker_sector_map.csv (ticker, sector)."
            )

        if bt_use_strict_mode:
            st.markdown(
                f"""
**Trigger definition used in this backtest: Pattern A+ (strict)**

- SMA50 > SMA200
- Close > SMA50 and Close > SMA200
- Close > previous {int(bt_breakout_days)}-day high close by at least {float(bt_breakout_buffer_pct):.1f}%
- Volume >= {float(bt_volume_multiplier):.2f} * 20-day average volume
- Initial stop = max({float(bt_stop_pct):.1f}%, {float(bt_atr_multiplier):.1f} x ATR{int(bt_atr_period)}) when ATR stop is enabled
- Move stop to break-even after +{float(bt_break_even_trigger_pct):.1f}%
- Exit at day {int(bt_time_stop_days)} if still open
"""
            )
        else:
            st.markdown(
                f"""
**Trigger definition used in this backtest: Pattern A (base)**

- SMA50 > SMA200
- Close > SMA50 and Close > SMA200
- Close > previous {int(bt_breakout_days)}-day high close
- Volume >= {float(bt_volume_multiplier):.2f} * 20-day average volume
- Stop loss = {float(bt_stop_pct):.1f}% below entry
"""
            )

        st.caption(
            f"Pattern blend: A={'ON' if bt_use_pattern_a else 'OFF'}, "
            f"B={'ON' if bt_use_pattern_b else 'OFF'} | "
            f"Score threshold: {int(bt_min_signal_score)} | "
            f"Consensus bonus: {float(bt_consensus_bonus):.1f}"
        )

        latest_dt = prices["Date"].max()
        target_dt = latest_dt - pd.DateOffset(months=int(bt_hide_months))
        eligible_dates = sorted(prices.loc[prices["Date"] <= target_dt, "Date"].drop_duplicates())

        bt_dates_for_run = eligible_dates
        bt_external_summary: dict[str, int | str | bool] = {"applied": False}
        bt_ticker_sector_rs_df = pd.DataFrame()
        bt_min_sector_rs_for_run: float | None = None

        if bt_use_external_filters and eligible_dates:
            ext_df = load_external_factors()
            bt_dates_for_run, bt_external_summary = filter_eligible_dates_by_external_factors(
                eligible_dates,
                ext_df,
                max_vix=float(bt_max_vix),
                max_vix_1d_spike_pct=float(bt_max_vix_spike),
                max_usdinr_5d_pct=float(bt_max_usdinr_5d),
                max_brent_5d_pct=float(bt_max_brent_5d),
                min_fii_dii_net_cr=float(bt_min_fii_dii),
            )

            if bt_use_sector_rs_filter:
                sector_map = load_ticker_sector_map()
                bt_ticker_sector_rs_df = build_ticker_sector_rs_table(
                    prices,
                    sector_map,
                    lookback_days=int(bt_sector_lookback_days),
                )
                if bt_ticker_sector_rs_df.empty:
                    st.warning(
                        "Sector RS filter requested, but ticker_sector_map.csv is missing/invalid or insufficient price history. "
                        "Sector filter will be skipped."
                    )
                else:
                    bt_min_sector_rs_for_run = float(bt_min_sector_rs20)

            if bt_external_summary.get("applied"):
                st.caption(
                    "External filters kept "
                    f"{bt_external_summary.get('dates_kept', 0)} / {bt_external_summary.get('dates_total', 0)} "
                    "eligible signal dates."
                )
            elif bt_external_summary.get("reason") == "external_factors_missing":
                st.warning("external_factors.csv not found or invalid. External date filters are skipped.")

        if not eligible_dates:
            st.error("Not enough history after hiding selected months. Reduce hidden months.")
        else:
            bt_as_of = bt_dates_for_run[-1] if bt_dates_for_run else eligible_dates[-1]
            st.info(
                f"Latest visible date after hiding {bt_hide_months} month(s): {bt_as_of.date().isoformat()} (latest full date is {latest_dt.date().isoformat()})."
            )

            if st.button("Run Backtest", key="run_backtest_btn", width="stretch"):
                if not bt_use_pattern_a and not bt_use_pattern_b:
                    st.warning("Enable at least one pattern before running backtest.")
                    st.stop()

                if not bt_dates_for_run:
                    st.warning("All eligible dates were filtered out by external constraints. Relax thresholds and retry.")
                    st.stop()

                bt_signals, bt_eval = run_backtest_for_params(
                    prices,
                    eligible_dates=bt_dates_for_run,
                    breakout_days=int(bt_breakout_days),
                    volume_multiplier=float(bt_volume_multiplier),
                    stop_pct=float(bt_stop_pct),
                    hold_days=int(bt_hold_days),
                    breakout_buffer_pct=float(bt_breakout_buffer_pct if bt_use_strict_mode else 0.0),
                    use_atr_stop=bool(bt_use_atr_stop and bt_use_strict_mode),
                    atr_period=int(bt_atr_period),
                    atr_multiplier=float(bt_atr_multiplier),
                    break_even_trigger_pct=(
                        float(bt_break_even_trigger_pct) if bt_use_strict_mode else None
                    ),
                    time_stop_days=(int(bt_time_stop_days) if bt_use_strict_mode else None),
                    ticker_sector_rs_df=bt_ticker_sector_rs_df,
                    min_sector_rs20=bt_min_sector_rs_for_run,
                    use_pattern_a=bool(bt_use_pattern_a),
                    use_pattern_b=bool(bt_use_pattern_b),
                    pullback_buffer_pct=float(bt_pullback_buffer_pct),
                    rebound_min_pct=float(bt_rebound_min_pct),
                    min_signal_score=float(bt_min_signal_score),
                    consensus_bonus=float(bt_consensus_bonus),
                )

                st.session_state["bt_result"] = {
                    "as_of": bt_as_of.date().isoformat(),
                    "signals": bt_signals,
                    "evaluated": bt_eval,
                    "hide_months": int(bt_hide_months),
                    "hold_days": int(bt_hold_days),
                    "external_summary": bt_external_summary,
                }

        bt_result = st.session_state.get("bt_result")
        if bt_result:
            bt_signals = bt_result["signals"]
            bt_eval = bt_result.get("evaluated", pd.DataFrame())

            st.markdown("### Backtest Result")
            s1, s2, s3, s4 = st.columns(4)
            total_bt_signals = len(bt_signals)
            unique_tickers = bt_signals["ticker"].nunique() if not bt_signals.empty else 0
            unique_dates = bt_signals["signal_date"].nunique() if not bt_signals.empty else 0
            eval_valid = bt_eval[bt_eval["return_pct"].notna()] if not bt_eval.empty else bt_eval
            win_rate = (
                float((eval_valid["return_pct"] > 0).mean() * 100.0) if not eval_valid.empty else 0.0
            )
            avg_return = float(eval_valid["return_pct"].mean()) if not eval_valid.empty else 0.0
            stop_rate = (
                float((bt_eval["outcome"] == "stop_hit").mean() * 100.0) if not bt_eval.empty else 0.0
            )
            # Simple 0-100 quality score combining win-rate and average return.
            return_component = max(0.0, min(100.0, 50.0 + avg_return * 5.0))
            pattern_score = round(0.6 * win_rate + 0.4 * return_component, 1)

            s1.metric("Signals Generated", total_bt_signals)
            s2.metric("Tickers Triggered", unique_tickers)
            s3.metric("Signal Dates", unique_dates)
            s4.metric("Hidden Months", bt_result.get("hide_months", "-"))

            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Win Rate %", f"{win_rate:.1f}")
            o2.metric("Avg Return %", f"{avg_return:.2f}")
            o3.metric("Stop Hit %", f"{stop_rate:.1f}")
            o4.metric("Pattern Score / 100", f"{pattern_score:.1f}")
            st.caption("How to read: higher Win Rate and Avg Return are better; lower Stop Hit is safer; compare scores only within the same hold window.")

            if pattern_score >= 65:
                st.success("Overall view: pattern quality looks strong on this backtest setup.")
            elif pattern_score >= 50:
                st.info("Overall view: pattern quality is mixed/average on this backtest setup.")
            else:
                st.warning("Overall view: pattern quality looks weak on this backtest setup.")

            with st.expander("Show generated trigger(s)", expanded=True):
                render_table(bt_signals, height=360)

            ext_summary = bt_result.get("external_summary", {})
            if isinstance(ext_summary, dict) and ext_summary.get("applied"):
                st.caption(
                    "External filter impact: "
                    f"kept {ext_summary.get('dates_kept', 0)} / {ext_summary.get('dates_total', 0)} dates | "
                    f"blocked by VIX: {ext_summary.get('blocked_vix', 0)}, "
                    f"VIX spike: {ext_summary.get('blocked_vix_spike', 0)}, "
                    f"USDINR: {ext_summary.get('blocked_usdinr', 0)}, "
                    f"Brent: {ext_summary.get('blocked_brent', 0)}, "
                    f"Flows: {ext_summary.get('blocked_flows', 0)}"
                )

            with st.expander("Show trigger quality details", expanded=True):
                if bt_eval.empty:
                    st.info("No evaluated trigger rows available.")
                else:
                    view_cols = [
                        "signal_date",
                        "ticker",
                        "pattern_family",
                        "signal_score",
                        "quality",
                        "outcome",
                        "return_pct",
                        "max_upside_pct",
                        "max_drawdown_pct",
                        "entry_price",
                        "exit_price",
                        "exit_date",
                    ]

                    def _row_style(row: pd.Series) -> list[str]:
                        outcome = row.get("outcome")
                        if outcome in {"stop_hit", "Stopped out"}:
                            color = "#fee2e2"
                        elif outcome in {"held_to_window_end", "Held to end"} and float(row.get("return_pct") or 0) > 0:
                            color = "#dcfce7"
                        elif outcome in {"held_to_window_end", "Held to end"}:
                            color = "#fef3c7"
                        elif outcome in {"time_stop", "Timed exit"} and float(row.get("return_pct") or 0) > 0:
                            color = "#e0f2fe"
                        else:
                            color = "#f1f5f9"
                        return [f"background-color: {color}"] * len(row)

                    eval_view = bt_eval[view_cols].copy()
                    eval_view["outcome"] = eval_view["outcome"].map(humanize_outcome)
                    styled = eval_view.style.apply(_row_style, axis=1)
                    render_table(styled, height=360)

        st.markdown("---")
        st.subheader("Compare Settings")
        st.caption("Edit values below, then run compare.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Setup 1**")
            s1_name = st.text_input("Name", value="Safe", key="cmp_1_name")
            s1_breakout = st.number_input("Breakout days", min_value=5, max_value=200, value=50, step=1, key="cmp_1_breakout")
            s1_volume = st.number_input("Volume x", min_value=0.5, max_value=5.0, value=1.8, step=0.1, format="%.2f", key="cmp_1_volume")
            s1_stop = st.number_input("Stop %", min_value=1.0, max_value=20.0, value=6.0, step=0.5, format="%.1f", key="cmp_1_stop")
        with c2:
            st.markdown("**Setup 2**")
            s2_name = st.text_input("Name ", value="Balanced", key="cmp_2_name")
            s2_breakout = st.number_input("Breakout days ", min_value=5, max_value=200, value=40, step=1, key="cmp_2_breakout")
            s2_volume = st.number_input("Volume x ", min_value=0.5, max_value=5.0, value=1.5, step=0.1, format="%.2f", key="cmp_2_volume")
            s2_stop = st.number_input("Stop % ", min_value=1.0, max_value=20.0, value=7.0, step=0.5, format="%.1f", key="cmp_2_stop")
        with c3:
            st.markdown("**Setup 3**")
            s3_name = st.text_input("Name  ", value="Fast", key="cmp_3_name")
            s3_breakout = st.number_input("Breakout days  ", min_value=5, max_value=200, value=25, step=1, key="cmp_3_breakout")
            s3_volume = st.number_input("Volume x  ", min_value=0.5, max_value=5.0, value=1.2, step=0.1, format="%.2f", key="cmp_3_volume")
            s3_stop = st.number_input("Stop %  ", min_value=1.0, max_value=20.0, value=8.0, step=0.5, format="%.1f", key="cmp_3_stop")

        presets = [
            {
                "name": s1_name.strip() or "Setup 1",
                "breakout_days": int(s1_breakout),
                "volume_multiplier": float(s1_volume),
                "stop_pct": float(s1_stop),
                "breakout_buffer_pct": 0.5,
                "use_atr_stop": True,
                "atr_period": 14,
                "atr_multiplier": 2.5,
                "break_even_trigger_pct": 2.0,
                "time_stop_days": 10,
                "pullback_buffer_pct": float(bt_pullback_buffer_pct),
                "rebound_min_pct": float(bt_rebound_min_pct),
            },
            {
                "name": s2_name.strip() or "Setup 2",
                "breakout_days": int(s2_breakout),
                "volume_multiplier": float(s2_volume),
                "stop_pct": float(s2_stop),
                "breakout_buffer_pct": 0.0,
                "use_atr_stop": False,
                "atr_period": 14,
                "atr_multiplier": 2.5,
                "break_even_trigger_pct": None,
                "time_stop_days": None,
                "pullback_buffer_pct": float(bt_pullback_buffer_pct),
                "rebound_min_pct": float(bt_rebound_min_pct),
            },
            {
                "name": s3_name.strip() or "Setup 3",
                "breakout_days": int(s3_breakout),
                "volume_multiplier": float(s3_volume),
                "stop_pct": float(s3_stop),
                "breakout_buffer_pct": 0.0,
                "use_atr_stop": False,
                "atr_period": 14,
                "atr_multiplier": 2.5,
                "break_even_trigger_pct": None,
                "time_stop_days": None,
                "pullback_buffer_pct": float(bt_pullback_buffer_pct),
                "rebound_min_pct": float(bt_rebound_min_pct),
            },
        ]
        render_table(pd.DataFrame(presets), height=220)

        if st.button("Run Compare", key="run_compare_btn", width="stretch"):
            compare_rows: list[dict] = []
            compare_runs: dict[str, pd.DataFrame] = {}

            for p in presets:
                cmp_signals, cmp_eval = run_backtest_for_params(
                    prices,
                    eligible_dates=bt_dates_for_run,
                    breakout_days=int(p["breakout_days"]),
                    volume_multiplier=float(p["volume_multiplier"]),
                    stop_pct=float(p["stop_pct"]),
                    hold_days=int(bt_hold_days),
                    breakout_buffer_pct=float(p["breakout_buffer_pct"]),
                    use_atr_stop=bool(p["use_atr_stop"]),
                    atr_period=int(p["atr_period"]),
                    atr_multiplier=float(p["atr_multiplier"]),
                    break_even_trigger_pct=p["break_even_trigger_pct"],
                    time_stop_days=p["time_stop_days"],
                    ticker_sector_rs_df=bt_ticker_sector_rs_df,
                    min_sector_rs20=bt_min_sector_rs_for_run,
                    use_pattern_a=bool(bt_use_pattern_a),
                    use_pattern_b=bool(bt_use_pattern_b),
                    pullback_buffer_pct=float(p["pullback_buffer_pct"]),
                    rebound_min_pct=float(p["rebound_min_pct"]),
                    min_signal_score=float(bt_min_signal_score),
                    consensus_bonus=float(bt_consensus_bonus),
                )
                valid = cmp_eval[cmp_eval["return_pct"].notna()] if not cmp_eval.empty else cmp_eval
                win_rate = float((valid["return_pct"] > 0).mean() * 100.0) if not valid.empty else 0.0
                avg_return = float(valid["return_pct"].mean()) if not valid.empty else 0.0
                stop_rate = float((cmp_eval["outcome"] == "stop_hit").mean() * 100.0) if not cmp_eval.empty else 0.0
                score = round(0.6 * win_rate + 0.4 * max(0.0, min(100.0, 50.0 + avg_return * 5.0)), 1)

                compare_rows.append(
                    {
                        "setup": p["name"],
                        "signals": len(cmp_signals),
                        "win_rate_pct": round(win_rate, 1),
                        "avg_return_pct": round(avg_return, 2),
                        "stop_hit_pct": round(stop_rate, 1),
                        "score": score,
                    }
                )
                if not cmp_eval.empty:
                    tmp = cmp_eval.copy()
                    tmp["signal_date"] = pd.to_datetime(tmp["signal_date"])
                    tmp["setup"] = p["name"]
                    compare_runs[p["name"]] = tmp

            compare_table = pd.DataFrame(compare_rows).sort_values(["score", "win_rate_pct"], ascending=False)
            st.session_state["bt_compare_table"] = compare_table
            st.session_state["bt_compare_runs"] = compare_runs

        compare_table = st.session_state.get("bt_compare_table")
        compare_runs = st.session_state.get("bt_compare_runs", {})

        if isinstance(compare_table, pd.DataFrame) and not compare_table.empty:
            st.markdown("### Compare Result")
            render_table(compare_table, height=220)

            setup_names = compare_table["setup"].tolist()
            selected_setup = st.selectbox(
                "Choose setup for details",
                options=setup_names,
                key="bt_compare_setup",
                help="Drill into one setup to inspect return curve, monthly behavior, and trade log.",
            )
            sel_eval = compare_runs.get(selected_setup, pd.DataFrame()).copy()

            if sel_eval.empty:
                st.info("No trade rows for this setup.")
            else:
                valid = sel_eval[sel_eval["return_pct"].notna()].copy()
                if not valid.empty:
                    valid.sort_values("signal_date", inplace=True)
                    valid["cum_return_pct"] = valid["return_pct"].cumsum()
                    curve = valid[["signal_date", "cum_return_pct"]].set_index("signal_date")
                    st.markdown("### Return Curve")
                    st.line_chart(curve, width="stretch")

                    valid["month"] = valid["signal_date"].dt.to_period("M").astype(str)
                    monthly = (
                        valid.groupby("month", as_index=False)
                        .agg(avg_return_pct=("return_pct", "mean"), trades=("ticker", "count"))
                        .sort_values("month")
                    )
                    st.markdown("### Month by Month")
                    render_table(monthly, height=240)

                st.markdown("### Trade Log")
                log_cols = [
                    "signal_date",
                    "ticker",
                    "pattern_family",
                    "signal_score",
                    "outcome",
                    "return_pct",
                    "max_upside_pct",
                    "max_drawdown_pct",
                    "entry_price",
                    "exit_price",
                    "exit_date",
                ]
                log_cols = [c for c in log_cols if c in sel_eval.columns]
                show_log = sel_eval[log_cols].copy()
                if "outcome" in show_log.columns:
                    show_log["outcome"] = show_log["outcome"].map(humanize_outcome)
                render_table(show_log.sort_values(["signal_date", "ticker"]), height=360)

with backtest_lab_tab:
    st.subheader("Backtesting Lab")
    st.caption("Track your own dummy-money positions. This is separate from Pattern A Backtest.")

    prefill = st.session_state.get("lab_prefill", {})
    with st.form("backtesting_lab_form"):
        f1, f2 = st.columns(2)
        with f1:
            ticker_in = st.text_input("Ticker", value=str(prefill.get("ticker", ""))).strip().upper()
            signal_date_in = st.text_input("Signal date", value=str(prefill.get("source_signal_date", ""))).strip()
            entry_in = st.number_input(
                "1 stock price (entry)",
                min_value=0.0,
                value=float(prefill.get("entry_price", 0.0) or 0.0),
                step=0.1,
            )
        with f2:
            pattern_in = st.text_input("Pattern", value=str(prefill.get("pattern", ""))).strip()
            stop_in = st.number_input(
                "Stop loss",
                min_value=0.0,
                value=float(prefill.get("stop_price", 0.0) or 0.0),
                step=0.1,
            )
            capital_in = st.number_input("Dummy money to put", min_value=100.0, value=10000.0, step=100.0)

        note_in = st.text_input("Note (optional)", value="")
        submit = st.form_submit_button("See how it performs")

    if submit:
        if not ticker_in:
            st.warning("Ticker is required.")
        elif entry_in <= 0:
            st.warning("Entry price must be greater than 0.")
        elif stop_in <= 0:
            st.warning("Stop loss must be greater than 0.")
        else:
            new_row = pd.DataFrame(
                [
                    {
                        "lab_id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source_signal_date": signal_date_in or pd.NA,
                        "ticker": ticker_in,
                        "pattern": pattern_in or pd.NA,
                        "entry_price": float(entry_in),
                        "stop_price": float(stop_in),
                        "capital": float(capital_in),
                        "status": "Watching",
                        "note": note_in or pd.NA,
                    }
                ]
            )
            dummy_lab = pd.concat([dummy_lab, new_row], ignore_index=True)
            save_dummy_lab(dummy_lab)
            st.session_state.pop("lab_prefill", None)
            st.success("Added to Backtesting Lab.")
            st.rerun()

    if dummy_lab_live.empty:
        st.info("No dummy-money positions yet. Add one using the form above or from stock details.")
    else:
        open_lab = dummy_lab_live[dummy_lab_live["status"].astype(str) == "Watching"].copy()
        if open_lab.empty:
            open_lab = dummy_lab_live.copy()

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Positions", len(open_lab))
        t2.metric("Capital", f"{open_lab['capital'].sum():,.0f}")
        t3.metric("Current value", f"{open_lab['current_value'].sum():,.0f}" if "current_value" in open_lab.columns else "-")
        total_pnl = float(open_lab["pnl"].sum()) if "pnl" in open_lab.columns else 0.0
        t4.metric("P&L", f"{total_pnl:,.2f}")

        show_cols = [
            "created_at",
            "source_signal_date",
            "ticker",
            "pattern",
            "entry_price",
            "stop_price",
            "latest_close",
            "capital",
            "current_value",
            "pnl",
            "current_return_pct",
            "distance_to_stop_pct",
            "status",
            "note",
        ]
        show_cols = [c for c in show_cols if c in open_lab.columns]
        view_df = open_lab[show_cols].copy()
        for c in ["entry_price", "stop_price", "latest_close", "capital", "current_value", "pnl", "current_return_pct", "distance_to_stop_pct"]:
            if c in view_df.columns:
                view_df[c] = pd.to_numeric(view_df[c], errors="coerce").round(2)
        render_table(view_df.sort_values(["created_at", "ticker"], ascending=[False, True]), height=360)

        st.markdown("### Manage lab positions")
        sel_df = open_lab.copy()
        sel_df["label"] = sel_df["created_at"].astype(str) + " | " + sel_df["ticker"].astype(str) + " | " + sel_df["status"].astype(str)
        selected_label = st.selectbox("Choose row", options=sel_df["label"].tolist(), key="lab_row_select")
        selected_row = sel_df[sel_df["label"] == selected_label].iloc[0]

        c_close, c_reopen = st.columns(2)
        with c_close:
            if st.button("Mark Closed", key="lab_mark_closed"):
                mask = dummy_lab["lab_id"].astype(str) == str(selected_row["lab_id"])
                dummy_lab.loc[mask, "status"] = "Closed"
                save_dummy_lab(dummy_lab)
                st.success("Marked as Closed.")
                st.rerun()
        with c_reopen:
            if st.button("Mark Watching", key="lab_mark_watching"):
                mask = dummy_lab["lab_id"].astype(str) == str(selected_row["lab_id"])
                dummy_lab.loc[mask, "status"] = "Watching"
                save_dummy_lab(dummy_lab)
                st.success("Marked as Watching.")
                st.rerun()

with telegram_tab:
    st.subheader("Send to Telegram")
    st.caption("This uses TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env or secrets.yml.")
    st.info("Recommended flow: send only after you validate buys/sells in Signals and Backtesting tabs.")
    if not is_remote_runtime():
        st.warning("Telegram sending is disabled on local machine by security policy.")

    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        st.warning("Telegram credentials not found. Add them in env or secrets.yml.")

    st.markdown("### Quick send")
    sell_message = build_sell_telegram_message(sell_signals)
    if st.button("Send latest sell signals", key="send_latest_sells_btn", disabled=(not allow_actions)):
        with st.spinner("Sending latest sell signals..."):
            ok, msg = send_telegram_message(token, chat_id, sell_message)
        if ok:
            st.session_state["flow_step_4_date"] = today_str
            st.success("Latest sell signals sent.")
        else:
            st.error(msg)

    if signals.empty:
        st.info("No buy signals file rows found. You can still send a no-signal message.")
        telegram_date_options = [date.today().isoformat()]
    else:
        telegram_date_options = sorted(signals["signal_date"].unique())

    tg_date = st.selectbox(
        "Buy signal date to send",
        options=telegram_date_options,
        index=len(telegram_date_options) - 1,
        key="telegram_signal_date",
    )

    tg_message = build_telegram_message_for_date(signals, tg_date)
    st.text_area("Telegram message preview", value=tg_message, height=180, key="telegram_preview")

    if st.button("Send to Telegram", key="send_telegram_btn", disabled=(not allow_actions)):
        with st.spinner("Sending Telegram message..."):
            ok, msg = send_telegram_message(token, chat_id, tg_message)
        if ok:
            st.session_state["flow_step_4_date"] = today_str
            st.success("Message sent.")
        else:
            st.error(msg)

st.caption(
    "Data files used: prices_eod.csv, signals_pattern_a.csv, sell_signals_pattern_a.csv, portfolio_positions.csv. "
    f"Production app: {PRODUCTION_APP_URL}"
)
