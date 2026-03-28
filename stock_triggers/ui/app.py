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
import importlib.util

import pandas as pd
import requests
import streamlit as st
from streamlit_navigation_bar import st_navbar


ROOT = Path(__file__).resolve().parents[2]
TRIGGERS_DIR = ROOT / "stock_triggers"
LOGO_SVG = str(Path(__file__).resolve().parent / "logo.svg")
SCRIPTS_DIR = TRIGGERS_DIR / "scripts"
DATA_DIR = TRIGGERS_DIR / "data"
SIGNALS_CSV = DATA_DIR / "signals_pattern_a.csv"
SELL_SIGNALS_CSV = DATA_DIR / "sell_signals_pattern_a.csv"
PORTFOLIO_CSV = DATA_DIR / "portfolio_positions.csv"
DUMMY_LAB_CSV = DATA_DIR / "backtesting_lab_positions.csv"
PRICES_CSV = DATA_DIR / "prices_eod.csv"
EXTERNAL_FACTORS_CSV = DATA_DIR / "external_factors.csv"
TICKER_SECTOR_MAP_CSV = DATA_DIR / "ticker_sector_map.csv"
STOCK_SCORES_CSV = DATA_DIR / "stock_scores.csv"
CANDIDATE_STOCKS_CSV = DATA_DIR / "candidate_stocks.csv"
STOCK_UNIVERSE_DIR = DATA_DIR / "stock_universe"
SECRETS_FILE = ROOT / "secrets.yml"
IS_STREAMLIT_CLOUD = bool(os.getenv("STREAMLIT_SHARING_MODE")) or bool(os.getenv("STREAMLIT_CLOUD"))
PRODUCTION_APP_URL = "https://stock-operator-roy.streamlit.app/"

# Reuse the main RSI implementation from generate_stock_scores so any change
# in the core indicator logic is picked up automatically.
_compute_rsi_shared = None
_scores_module_path = SCRIPTS_DIR / "generate_stock_scores.py"
if _scores_module_path.is_file():  # pragma: no cover - simple import wiring
    spec = importlib.util.spec_from_file_location("_stock_scores_module", _scores_module_path)
    if spec and spec.loader:
        _mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(_mod)
            _compute_rsi_shared = getattr(_mod, "compute_rsi", None)
        except Exception:
            _compute_rsi_shared = None


st.set_page_config(page_title="Stock Triggers – Pattern A", layout="wide")

# ── Navigation bar ──
_nav_styles = {
    "nav": {
        "background-color": "#1e293b",
        "font-family": "'Space Grotesk', sans-serif",
        "justify-content": "left",
        "padding": "0.35rem 0.8rem",
        "box-shadow": "0 4px 20px rgba(15,23,42,0.25)",
    },
    "img": {
        "padding-right": "14px",
        "height": "26px",
    },
    "span": {
        "color": "#94a3b8",
        "font-weight": "600",
        "font-size": "0.85rem",
        "padding": "0.45rem 0.9rem",
        "border-radius": "10px",
    },
    "active": {
        "color": "#f8fafc",
        "background-color": "rgba(59,130,246,0.25)",
    },
    "hover": {
        "color": "#e2e8f0",
        "background-color": "rgba(255,255,255,0.06)",
    },
}

_nav_options = {
    "show_menu": False,
    "show_sidebar": False,
    "fix_shadow": True,
    "use_padding": True,
}

# Map navbar page names → internal mode names
_NAV_PAGES = ["Tomorrow's Picks", "Backtesting Lab"]
_NAV_TO_MODE = {"Tomorrow's Picks": "Tomorrow", "Backtesting Lab": "Backtest Lab"}

# Resolve which page to pre-select based on current session mode
if "mode" not in st.session_state:
    st.session_state["mode"] = "Tomorrow"
_mode_to_nav = {v: k for k, v in _NAV_TO_MODE.items()}
_preselected = _mode_to_nav.get(st.session_state["mode"], _NAV_PAGES[0])

_selected_page = st_navbar(
    _NAV_PAGES,
    selected=_preselected,
    logo_path=LOGO_SVG,
    logo_page="Tomorrow's Picks",
    styles=_nav_styles,
    options=_nav_options,
    adjust=False,
    key="main_nav",
)

# Position the navbar at the top of the page
st.markdown(
    """
    <style>
    /* Hide default Streamlit chrome */
    header[data-testid="stHeader"] {
        background-color: #1e293b !important;
        height: 2.875rem !important;
        z-index: 0 !important;
    }
    #MainMenu, footer, #stDecoration { visibility: hidden !important; }
    div[class="stDeployButton"] { visibility: hidden !important; }
    div[class="stStatusWidget"] { visibility: hidden !important; }

    /* Navbar iframe — fixed to top */
    iframe[title="streamlit_navigation_bar.st_navbar"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 2.875rem !important;
        z-index: 999999 !important;
        margin-top: 0 !important;
        border: none !important;
    }

    /* Push main content below navbar */
    section.main {
        position: relative !important;
        top: 2.875rem !important;
    }
    /* Navbar iframe needs pointer-events */
    iframe[title="streamlit_navigation_bar.st_navbar"] {
        pointer-events: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sync navbar selection → session state mode
if _selected_page and _NAV_TO_MODE.get(_selected_page) != st.session_state["mode"]:
    st.session_state["mode"] = _NAV_TO_MODE[_selected_page]
    st.rerun()

_curr_mode = st.session_state["mode"]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;600;700&display=swap');
    .block-container {padding-top: 0.3rem;}
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


@st.cache_data(show_spinner=False, ttl=120)
def load_stock_scores() -> pd.DataFrame:
    if not STOCK_SCORES_CSV.is_file():
        return pd.DataFrame()
    df = pd.read_csv(STOCK_SCORES_CSV)
    if "ticker" not in df.columns:
        return pd.DataFrame()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df


@st.cache_data(show_spinner=False)
def load_candidate_stocks() -> pd.DataFrame:
    """Load external candidate stocks for the add-stocks dropdown.

    Sources (merged & deduplicated):
      1. candidate_stocks.csv  – manual overrides / custom picks.
      2. stock_universe/*.csv  – index constituent files (e.g. ind_nifty50list.csv).
         Recognises columns named ticker, Ticker, or Symbol.
    """

    all_tickers: list[str] = []

    # --- source 1: candidate_stocks.csv ---
    if CANDIDATE_STOCKS_CSV.is_file():
        df = pd.read_csv(CANDIDATE_STOCKS_CSV)
        col = next((c for c in ("ticker", "Ticker", "Symbol") if c in df.columns), None)
        if col is not None:
            all_tickers.extend(df[col].astype(str).str.strip().str.upper().tolist())

    # --- source 2: stock_universe/ folder ---
    if STOCK_UNIVERSE_DIR.is_dir():
        for csv_path in sorted(STOCK_UNIVERSE_DIR.glob("*.csv")):
            try:
                udf = pd.read_csv(csv_path)
            except Exception:
                continue
            col = next((c for c in ("Symbol", "ticker", "Ticker") if c in udf.columns), None)
            if col is not None:
                all_tickers.extend(udf[col].astype(str).str.strip().str.upper().tolist())

    if not all_tickers:
        return pd.DataFrame()

    out = pd.DataFrame({"ticker": all_tickers})
    out = out[out["ticker"] != ""].drop_duplicates().reset_index(drop=True)
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
    backfill: bool = False,
) -> tuple[bool, str]:
    """Run Pattern A trigger generation and stock score refresh.

    If parameters are provided, pass them through to the generator.
    When *backfill* is True, regenerate signals for all historical dates.
    Also regenerates stock_scores.csv so the All Scores panel stays fresh.
    """

    pattern_script = SCRIPTS_DIR / "generate_triggers_pattern_a.py"
    if not pattern_script.is_file():
        return False, "Pattern A script not found under stock_triggers/scripts/."

    cmd = [sys.executable, str(pattern_script)]
    if backfill:
        cmd.append("--backfill-history")
    elif as_of_date:
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

    # Also regenerate stock health scores so the All Scores panel is fresh
    scores_script = SCRIPTS_DIR / "generate_stock_scores.py"
    if scores_script.is_file():
        try:
            res2 = subprocess.run(
                [sys.executable, str(scores_script)],
                capture_output=True, text=True, check=False,
            )
        except Exception:
            pass  # Non-fatal: signals were generated successfully
        else:
            if res2.returncode != 0:
                pass  # Non-fatal

    load_signals.clear()
    load_sell_signals.clear()
    load_stock_scores.clear()
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
    out["score_rsi"] = pd.NA
    out["score_risk"] = pd.NA
    out["signal_score"] = pd.NA
    out["consensus_count"] = 1
    return out


def _clip_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


# Component weights for Pattern A/B scoring when computing signal_score
WEIGHT_TREND = 0.28
WEIGHT_SETUP = 0.28
WEIGHT_VOLUME = 0.19
WEIGHT_RISK = 0.20
WEIGHT_RSI = 0.05


def _build_score_components(
    *,
    trend_strength_pct: float,
    setup_strength_pct: float,
    volume_ratio: float,
    stop_pct_eff: float,
    rsi_value: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    score_trend = _clip_score(50.0 + trend_strength_pct * 5.0)
    score_setup = _clip_score(50.0 + setup_strength_pct * 8.0)
    score_volume = _clip_score(40.0 + volume_ratio * 20.0)
    score_risk = _clip_score(100.0 - stop_pct_eff * 6.0)

    # RSI component: map latest RSI value (0-100) into a 0-100 score.
    # If RSI is missing, treat it as neutral (50).
    if rsi_value is None or pd.isna(rsi_value):
        score_rsi = 50.0
    else:
        score_rsi = _clip_score(rsi_value)

    signal_score = round(
        (WEIGHT_TREND * score_trend)
        + (WEIGHT_SETUP * score_setup)
        + (WEIGHT_VOLUME * score_volume)
        + (WEIGHT_RISK * score_risk)
        + (WEIGHT_RSI * score_rsi),
        1,
    )
    return (
        round(score_trend, 1),
        round(score_setup, 1),
        round(score_volume, 1),
        round(score_risk, 1),
        round(score_rsi, 1),
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
        rsi_value = None
        if _compute_rsi_shared is not None:
            try:
                hist_close = g[g["Date"] <= as_of_date]["Close"].astype(float)
                rsi_value = _compute_rsi_shared(hist_close, period=14)
            except Exception:
                rsi_value = None

        score_trend, score_setup, score_volume, score_risk, score_rsi, signal_score = _build_score_components(
            trend_strength_pct=trend_strength_pct,
            setup_strength_pct=setup_strength_pct,
            volume_ratio=volume_ratio,
            stop_pct_eff=stop_pct_eff,
            rsi_value=rsi_value,
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
                "score_rsi": score_rsi,
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
        "score_rsi",
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
                rsi_value = None
                if _compute_rsi_shared is not None:
                    try:
                        hist_close = g["Close"].astype(float)
                        rsi_value = _compute_rsi_shared(hist_close, period=14)
                    except Exception:
                        rsi_value = None

                score_trend, score_setup, score_volume, score_risk, score_rsi, signal_score = _build_score_components(
                    trend_strength_pct=trend_strength_pct,
                    setup_strength_pct=setup_strength_pct,
                    volume_ratio=volume_ratio,
                    stop_pct_eff=stop_pct_eff,
                    rsi_value=rsi_value,
                )
                a_df.at[i, "score_trend"] = score_trend
                a_df.at[i, "score_setup"] = score_setup
                a_df.at[i, "score_volume"] = score_volume
                a_df.at[i, "score_risk"] = score_risk
                a_df.at[i, "score_rsi"] = score_rsi
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
        "score_rsi",
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


def build_signal_tracker(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float = 6.0,
    stop_pct: float = 7.0,
    capital_per_trade: float = 10000.0,
) -> pd.DataFrame:
    """Build a tracker showing each buy signal's current status.

    For every signal, simulate buying 1 qty at entry_price on signal_date.
    Walk through subsequent price bars to determine outcome:
      - Target Hit: Close >= entry * (1 + target_pct/100)
      - Stop Hit:   Close <= entry * (1 - stop_pct/100)
      - Holding:    Neither triggered yet

    Returns a DataFrame with one row per signal.
    """
    if signals_df.empty or prices_df.empty:
        return pd.DataFrame()

    prices_df = prices_df.copy()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"])

    rows: list[dict] = []
    for _, sig in signals_df.iterrows():
        ticker = str(sig["ticker"])
        sig_date = pd.to_datetime(sig["signal_date"])
        entry_price = float(sig["entry_price"])
        stop_price_sig = float(sig.get("stop_price", entry_price * (1.0 - stop_pct / 100.0)))
        target_price = entry_price * (1.0 + target_pct / 100.0)
        stop_price_calc = entry_price * (1.0 - stop_pct / 100.0)

        qty = int(capital_per_trade // entry_price) if entry_price > 0 else 0
        if qty == 0:
            continue
        invested = round(qty * entry_price, 2)

        future = prices_df[(prices_df["Ticker"] == ticker) & (prices_df["Date"] > sig_date)].sort_values("Date")

        status = "Holding"
        exit_date = None
        exit_price = None
        latest_close = entry_price

        for _, bar in future.iterrows():
            close = float(bar["Close"])
            high = float(bar["High"])
            low = float(bar["Low"])
            latest_close = close
            if high >= target_price:
                status = "Target Hit ✅"
                exit_date = bar["Date"]
                exit_price = target_price
                break
            if low <= stop_price_calc:
                status = "Stop Hit 🛑"
                exit_date = bar["Date"]
                exit_price = stop_price_calc
                break

        if exit_price is not None:
            current_val = round(qty * exit_price, 2)
        else:
            current_val = round(qty * latest_close, 2)

        pnl = round(current_val - invested, 2)
        return_pct = round(((current_val / invested) - 1) * 100, 2) if invested > 0 else 0.0

        # Days held
        if exit_date is not None:
            days_held = (pd.to_datetime(exit_date) - sig_date).days
        elif not future.empty:
            days_held = (future["Date"].max() - sig_date).days
        else:
            days_held = 0

        rows.append({
            "signal_date": sig_date.date().isoformat(),
            "ticker": ticker.replace(".NS", ""),
            "pattern": str(sig.get("pattern", "")),
            "entry_price": round(entry_price, 2),
            "qty": qty,
            "invested": invested,
            "target_price": round(target_price, 2),
            "stop_price": round(stop_price_calc, 2),
            "latest_close": round(latest_close, 2),
            "current_value": current_val,
            "pnl": pnl,
            "return_pct": return_pct,
            "days_held": days_held,
            "exit_date": exit_date.date().isoformat() if exit_date is not None and hasattr(exit_date, "date") else (str(exit_date)[:10] if exit_date else "-"),
            "status": status,
            "signal_score": round(float(sig["signal_score"]), 1) if pd.notna(sig.get("signal_score")) else None,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.sort_values(["signal_date", "ticker"], ascending=[False, True], inplace=True)
    return out


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
                "score_rsi",
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

        # 14-day RSI for display in the Market dashboard, reusing core implementation.
        rsi14 = _compute_rsi_shared(close, period=14) if _compute_rsi_shared is not None else None

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
                "rsi14": rsi14,
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


def _decorate_stock_rows(base: pd.DataFrame, prices_df: pd.DataFrame | None = None) -> pd.DataFrame:
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

    # Default RSI-related fields; will be populated when price history is available.
    out["rsi_14"] = pd.NA
    out["rsi_state"] = pd.NA
    out["rsi_bonus"] = 0.0
    out["rsi_note"] = pd.NA
    out["ui_score"] = out["signal_score"]

    if (
        prices_df is not None
        and not prices_df.empty
        and "Ticker" in prices_df.columns
        and _compute_rsi_shared is not None
    ):
        prices_local = prices_df.copy()
        prices_local["Ticker"] = prices_local["Ticker"].astype(str).str.upper()

        rsi_cache: dict[str, dict] = {}

        def _compute_rsi_for_ticker(ticker_str: str) -> dict:
            t = prices_local[prices_local["Ticker"] == ticker_str].copy().sort_values("Date")
            if t.empty:
                return {}

            close = t["Close"].astype(float)
            rsi_val = _compute_rsi_shared(close, period=14)
            if rsi_val is None:
                return {}

            state = "unknown"
            bonus = 0.0
            note = ""
            tag = None
            if rsi_val < 45.0:
                state = "weak"
                bonus = -5.0
                note = "RSI is weak for a breakout."
                tag = "RSI weak"
            elif 45.0 <= rsi_val < 52.0:
                state = "cooling"
                bonus = 1.0
                note = "RSI is cooling after a stronger move."
                tag = "RSI cooling"
            elif 52.0 <= rsi_val <= 68.0:
                state = "healthy"
                bonus = 3.0
                note = "Momentum looks healthy."
                tag = "RSI healthy"
            elif 68.0 < rsi_val <= 78.0:
                state = "strong"
                bonus = 1.0
                note = "Momentum is strong; watch for stretch."
                tag = "RSI strong"
            elif rsi_val > 78.0:
                state = "stretched"
                bonus = -4.0
                note = "RSI is stretched; entry may be late."
                tag = "RSI stretched"

            return {
                "rsi_14": rsi_val,
                "rsi_state": state,
                "rsi_bonus": bonus,
                "rsi_note": note,
                "rsi_tag": tag,
            }

        tickers = out["ticker"].astype(str).str.upper()
        for idx, tkr in tickers.items():
            if tkr not in rsi_cache:
                rsi_cache[tkr] = _compute_rsi_for_ticker(tkr)
            info = rsi_cache.get(tkr) or {}
            if not info:
                continue

            out.at[idx, "rsi_14"] = info["rsi_14"]
            out.at[idx, "rsi_state"] = info["rsi_state"]
            out.at[idx, "rsi_bonus"] = info["rsi_bonus"]
            out.at[idx, "rsi_note"] = info["rsi_note"]
            out.at[idx, "ui_score"] = float(out.at[idx, "signal_score"]) + float(info["rsi_bonus"])

            rsi_tag = info.get("rsi_tag")
            if rsi_tag:
                existing_tags = out.at[idx, "tags"]
                if isinstance(existing_tags, list):
                    if rsi_tag not in existing_tags:
                        existing_tags.append(rsi_tag)
                    out.at[idx, "tags"] = existing_tags
                else:
                    out.at[idx, "tags"] = [existing_tags, rsi_tag] if existing_tags else [rsi_tag]

    return out


def _prepare_tomorrow_list(signals_df: pd.DataFrame, prices_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str | None]:
    if signals_df.empty:
        return pd.DataFrame(), None

    latest_signal_date = str(signals_df["signal_date"].max())
    base = signals_df[signals_df["signal_date"] == latest_signal_date].copy()
    if base.empty:
        return pd.DataFrame(), latest_signal_date

    return _decorate_stock_rows(base, prices_df), latest_signal_date


def _prepare_recent_recommendations(signals_df: pd.DataFrame, *, days: int = 7, prices_df: pd.DataFrame | None = None) -> pd.DataFrame:
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
    return _decorate_stock_rows(recent, prices_df)


def render_header(
    *,
    latest_signal_date: str | None,
    total_count: int,
    total_considered: int | None = None,
    data_updated: str | None = None,
    signals_generated: str | None = None,
    fallback_note: str | None = None,
) -> None:
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
            background: #e0e7ff;
            border: 1px solid #c7d2fe;
            border-radius: 999px;
            padding: 0.08rem 0.45rem;
            margin-right: 0.25rem;
            margin-bottom: 0.2rem;
        }
        .chip-good {
            color: #166534;
            background: #dcfce7;
            border-color: #86efac;
        }
        .chip-bad {
            color: #b91c1c;
            background: #fee2e2;
            border-color: #fecaca;
        }
        .chip-neutral {
            color: #92400e;
            background: #fef3c7;
            border-color: #fde68a;
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

    note_html = f"<div class='tomorrow-sub'><strong>{fallback_note}</strong></div>" if fallback_note else ""

    considered_str = ""
    if total_considered is not None:
        try:
            considered_val = int(total_considered)
            if considered_val > 0:
                considered_str = f" | Considered: {considered_val}"
        except Exception:
            considered_str = ""

    # Check staleness for inline refresh link.
    _stale_header = False
    if data_updated and data_updated != "-":
        try:
            _du_dt = datetime.strptime(data_updated, "%Y-%m-%d %H:%M")
            _stale_header = (datetime.now() - _du_dt).total_seconds() / 3600.0 >= 24.0
        except Exception:
            pass

    _refreshing = st.session_state.get("_header_refreshing", False)
    _generating = st.session_state.get("_header_generating", False)

    # --- Status dots ---
    def _dot(color: str) -> str:
        return (
            f"<span style='display:inline-block; width:7px; height:7px; "
            f"border-radius:50%; background:{color}; margin-right:0.3rem; "
            f"vertical-align:middle;"
            f"{"animation:pulse 1.2s ease-in-out infinite;" if color == "#eab308" else ""}'></span>"
        )

    if _refreshing:
        price_dot = _dot("#eab308")
        price_status = "Refreshing…"
    elif _stale_header:
        price_dot = _dot("#f59e0b")
        price_status = f"{data_updated or '-'}"
    else:
        price_dot = _dot("#22c55e")
        price_status = f"{data_updated or '-'}"

    if _generating:
        sig_dot = _dot("#eab308")
        sig_status = "Generating…"
    else:
        sig_dot = _dot("#22c55e") if signals_generated and signals_generated != "-" else _dot("#94a3b8")
        sig_status = signals_generated or "-"

    signals_gen_str = signals_generated or "-"

    st.markdown(
        (
            "<style>"
            "@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.35;} }"
            # ---- action bar inside sticky header ----
            ".hdr-action-bar {"
            "  display:flex; gap:0.4rem; margin-top:0.55rem; flex-wrap:wrap;"
            "}"
            ".hdr-pill {"
            "  display:inline-flex; align-items:center; gap:0.3rem;"
            "  font-size:0.76rem; font-weight:600; line-height:1;"
            "  border-radius:999px; padding:0.35rem 0.75rem;"
            "  cursor:pointer; border:none; text-decoration:none;"
            "  transition: transform 0.15s ease, box-shadow 0.15s ease;"
            "  box-shadow: 0 1px 4px rgba(15,23,42,0.08);"
            "}"
            ".hdr-pill:hover {"
            "  transform:translateY(-1px); box-shadow:0 4px 12px rgba(15,23,42,0.12);"
            "}"
            ".hdr-pill-primary {"
            "  color:#fff; background:linear-gradient(135deg,#059669 0%,#10b981 100%);"
            "}"
            ".hdr-pill-primary:hover { background:linear-gradient(135deg,#047857 0%,#059669 100%); }"
            ".hdr-pill-secondary {"
            "  color:#0369a1; background:#e0f2fe; border:1px solid #bae6fd;"
            "}"
            ".hdr-pill-secondary:hover { background:#bae6fd; }"
            ".hdr-pill-disabled {"
            "  color:#94a3b8; background:#f1f5f9; border:1px solid #e2e8f0;"
            "  cursor:not-allowed; opacity:0.6; pointer-events:none;"
            "}"
            ".hdr-pill-busy {"
            "  color:#92400e; background:#fefce8; border:1px solid #fde68a;"
            "  cursor:wait; animation:pulse 1.2s ease-in-out infinite;"
            "}"
            ".hdr-pill-icon { font-size:0.85rem; }"
            # ---- Streamlit button override inside action-bar wrapper ----
            ".action-bar-wrap div[data-testid='stHorizontalBlock'] { gap:0.4rem !important; }"
            ".action-bar-wrap button {"
            "  font-size:0.76rem !important; font-weight:600 !important;"
            "  border-radius:999px !important; padding:0.35rem 0.8rem !important;"
            "  line-height:1.1 !important; min-height:0 !important; height:auto !important;"
            "  transition: transform 0.15s ease, box-shadow 0.15s ease !important;"
            "  box-shadow: 0 1px 4px rgba(15,23,42,0.08) !important;"
            "}"
            ".action-bar-wrap button:hover {"
            "  transform:translateY(-1px) !important;"
            "  box-shadow:0 4px 12px rgba(15,23,42,0.12) !important;"
            "}"
            ".act-generate button {"
            "  color:#fff !important; background:linear-gradient(135deg,#059669 0%,#10b981 100%) !important;"
            "  border:none !important;"
            "}"
            ".act-generate button:hover { background:linear-gradient(135deg,#047857 0%,#059669 100%) !important; }"
            ".act-refresh button {"
            "  color:#0369a1 !important; background:#e0f2fe !important;"
            "  border:1px solid #bae6fd !important;"
            "}"
            ".act-refresh button:hover { background:#bae6fd !important; }"
            ".act-busy button {"
            "  color:#92400e !important; background:#fefce8 !important;"
            "  border:1px solid #fde68a !important;"
            "  animation:pulse 1.2s ease-in-out infinite !important;"
            "  cursor:wait !important;"
            "}"
            "</style>"
            "<div class='tomorrow-sticky'>"
            f"<div class='tomorrow-sub'>Latest signal date: {latest_signal_date or '-'}{considered_str} | Stocks found: {total_count}</div>"
            f"<div class='tomorrow-sub'>{price_dot}Prices: {price_status}</div>"
            f"<div class='tomorrow-sub'>{sig_dot}Signals: {sig_status}</div>"
            f"{note_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    # --- Action bar: compact pill buttons inside a styled wrapper ---
    if _stale_header or _refreshing:
        st.markdown("<div class='action-bar-wrap'>", unsafe_allow_html=True)
        ab1, ab2 = st.columns([1, 4])
        with ab1:
            if _refreshing:
                st.markdown("<div class='act-busy'>", unsafe_allow_html=True)
                st.button("⏳ Refreshing…", key="tomorrow_refresh_now", disabled=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='act-refresh'>", unsafe_allow_html=True)
                if st.button("🔄 Refresh prices", key="tomorrow_refresh_now"):
                    st.session_state["_header_refreshing"] = True
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    h1, h2 = st.columns([1.2, 1.0])
    with h1:
        st.slider("Minimum signal score", min_value=0, max_value=100, step=1, key="min_score")
    with h2:
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

    rsi_val = row.get("rsi_14")
    rsi_state = str(row.get("rsi_state", "") or "")
    rsi_display = ""
    if pd.notna(rsi_val):
        try:
            rsi_num = float(rsi_val)
            state_label = rsi_state.capitalize() if rsi_state else ""
            if state_label:
                rsi_display = f" | RSI {rsi_num:.0f} ({state_label})"
            else:
                rsi_display = f" | RSI {rsi_num:.0f}"
        except Exception:
            rsi_display = ""

    if isinstance(tags, list):
        def _chip_class(tag: str) -> str:
            t = str(tag).lower()
            # Clearly positive / supportive signals
            if t in {"uptrend", "breakout", "pullback", "volume okay", "low risk", "rsi healthy"}:
                return "chip chip-good"
            # Clearly negative / cautionary signals
            if t in {"rsi weak", "rsi stretched"}:
                return "chip chip-bad"
            # Mild caution / in-between states
            if t in {"rsi cooling", "rsi strong"}:
                return "chip chip-neutral"
            return "chip"

        chips = "".join([f"<span class='{_chip_class(t)}'>{t}</span>" for t in tags])
    else:
        chips = ""

    # Score color based on value
    if score >= 70:
        _sc_color = "#059669"   # green
    elif score >= 45:
        _sc_color = "#d97706"   # amber
    else:
        _sc_color = "#dc2626"   # red

    card_css = "stock-card-meta stock-card-meta-selected" if selected else "stock-card-meta"
    st.markdown(
        (
            f"<div class='{card_css}'>"
            f"<div><strong>{ticker}</strong> | {pattern_simple}</div>"
            f"<div class='stock-card-line'>Recommended {recommended_date}</div>"
            f"<div class='stock-card-line'>"
            f"Score <span style='font-weight:800; color:{_sc_color}; font-size:0.9rem;'>{score:.0f}</span>"
            f" | Entry {entry:.2f} | Stop {stop:.2f} | Risk {risk:.2f}%{rsi_display}</div>"
            f"<div class='stock-card-reason'>{reason}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='chip-row'>{chips}</div>", unsafe_allow_html=True)
    button_label = f"Selected: {ticker}" if selected else f"Select {ticker}"
    return st.button(button_label, key=f"card_{ticker}", type=("primary" if selected else "secondary"), width="stretch")


def _render_scores_panel() -> None:
    """Render a beautiful grid of all stock scores from stock_scores.csv."""
    scores_df = load_stock_scores()
    if scores_df.empty:
        st.info("No stock scores available yet. Run the scoring pipeline to generate them.")
        return

    # Sort by score descending, then ticker
    if "score" in scores_df.columns:
        scores_df["_sort"] = pd.to_numeric(scores_df["score"], errors="coerce")
        scores_df.sort_values(["_sort", "ticker"], ascending=[False, True], inplace=True)

    tiles_html = []
    for _, r in scores_df.iterrows():
        ticker = str(r.get("ticker", "")).replace(".NS", "")
        score_val = r.get("score")
        health = str(r.get("health", "")).strip() if pd.notna(r.get("health")) else ""
        rsi = r.get("rsi14")
        ret1d = r.get("ret_1d_pct")
        ret5d = r.get("ret_5d_pct")
        dist52 = r.get("dist_from_52w_high_pct")
        insight = str(r.get("insight", "")).strip() if pd.notna(r.get("insight")) else ""

        # Badge class + score color
        h_lc = health.lower()
        if h_lc.startswith("doing"):
            badge_cls = "score-tile-good"
            num_cls = "score-num-good"
        elif h_lc.startswith("mixed"):
            badge_cls = "score-tile-mixed"
            num_cls = "score-num-mixed"
        elif h_lc.startswith("weak"):
            badge_cls = "score-tile-weak"
            num_cls = "score-num-weak"
        else:
            badge_cls = "score-tile-na"
            num_cls = "score-num-na"

        score_str = str(int(score_val)) if pd.notna(score_val) else "-"
        health_str = health or "N/A"

        # Meta line
        meta_parts = []
        if pd.notna(rsi):
            meta_parts.append(f"RSI {rsi:.0f}")
        if pd.notna(ret1d):
            meta_parts.append(f"1d {ret1d:+.1f}%")
        if pd.notna(ret5d):
            meta_parts.append(f"5d {ret5d:+.1f}%")
        if pd.notna(dist52):
            meta_parts.append(f"52wH {dist52:+.1f}%")
        meta_str = " · ".join(meta_parts) if meta_parts else ""

        # Truncate insight
        if len(insight) > 80:
            insight = insight[:77] + "…"

        tile = (
            "<div class='score-tile'>"
            f"<span class='score-tile-ticker'>{ticker}</span>"
            f"<span class='score-tile-badge {badge_cls}'>{health_str}</span>"
            f"<span class='score-tile-num {num_cls}'>{score_str}</span>"
        )
        if meta_str:
            tile += f"<div class='score-tile-meta'>{meta_str}</div>"
        if insight:
            tile += f"<div class='score-tile-insight'>{insight}</div>"
        tile += "</div>"
        tiles_html.append(tile)

    st.markdown(
        "<div class='scores-panel'>"
        "<div style='font-weight:600; font-size:0.9rem; color:#0f172a; margin-bottom:0.3rem;'>"
        f"📊 Universe Health — {len(scores_df)} stocks scored</div>"
        "<div class='scores-grid'>"
        + "".join(tiles_html)
        + "</div></div>",
        unsafe_allow_html=True,
    )


def render_stock_list(stocks_df: pd.DataFrame) -> None:
    st.markdown("### Tomorrow's Picks")
    fallback_note = st.session_state.get("tomorrow_fallback_note")
    if fallback_note:
        # Styled fallback banner with an inline "Show all scores" toggle
        st.markdown(
            "<style>"
            ".fallback-bar {"
            "  display:flex; align-items:center; justify-content:space-between;"
            "  flex-wrap:wrap; gap:0.4rem;"
            "  background:linear-gradient(135deg,#fffbeb 0%,#fef3c7 100%);"
            "  border:1px solid #fde68a; border-radius:12px;"
            "  padding:0.55rem 0.85rem; margin-bottom:0.6rem;"
            "  box-shadow:0 2px 8px rgba(234,179,8,0.08);"
            "}"
            ".fallback-bar-text {"
            "  color:#92400e; font-size:0.85rem; font-weight:500;"
            "}"
            # Scores panel
            ".scores-panel {"
            "  border:1px solid #dbe4ef; border-radius:14px;"
            "  background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);"
            "  padding:0.7rem 0.8rem; margin-bottom:0.8rem;"
            "  box-shadow:0 4px 16px rgba(15,23,42,0.05);"
            "  animation:revealIn 0.24s ease;"
            "}"
            ".scores-grid {"
            "  display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr));"
            "  gap:0.5rem; margin-top:0.5rem;"
            "}"
            ".score-tile {"
            "  border:1px solid #e2e8f0; border-radius:10px;"
            "  padding:0.5rem 0.65rem; background:#fff;"
            "  transition:transform 0.15s ease, box-shadow 0.15s ease;"
            "}"
            ".score-tile:hover {"
            "  transform:translateY(-1px); box-shadow:0 4px 12px rgba(15,23,42,0.08);"
            "}"
            ".score-tile-ticker { font-weight:700; font-size:0.88rem; color:#0f172a; }"
            ".score-tile-badge {"
            "  display:inline-block; font-size:0.68rem; font-weight:600;"
            "  border-radius:999px; padding:0.08rem 0.4rem; margin-left:0.3rem;"
            "  vertical-align:middle;"
            "}"
            ".score-tile-good { color:#166534; background:#dcfce7; border:1px solid #86efac; }"
            ".score-tile-mixed { color:#92400e; background:#fef3c7; border:1px solid #fde68a; }"
            ".score-tile-weak { color:#b91c1c; background:#fee2e2; border:1px solid #fecaca; }"
            ".score-tile-na { color:#64748b; background:#f1f5f9; border:1px solid #e2e8f0; }"
            ".score-tile-num {"
            "  font-weight:800; font-size:0.82rem; margin-left:0.25rem;"
            "  vertical-align:middle;"
            "}"
            ".score-num-good { color:#059669; }"
            ".score-num-mixed { color:#d97706; }"
            ".score-num-weak { color:#dc2626; }"
            ".score-num-na { color:#94a3b8; }"
            ".score-tile-meta { font-size:0.76rem; color:#64748b; margin-top:0.2rem; }"
            ".score-tile-insight { font-size:0.74rem; color:#475569; margin-top:0.15rem; font-style:italic; }"
            "</style>",
            unsafe_allow_html=True,
        )

        # Fallback banner with toggle + generate toggle
        _generating = st.session_state.get("_header_generating", False)
        fb_cols = st.columns([3.5, 1, 1])
        with fb_cols[0]:
            st.markdown(
                f"<div class='fallback-bar'><span class='fallback-bar-text'>⚠️ {fallback_note}</span></div>",
                unsafe_allow_html=True,
            )
        with fb_cols[1]:
            show_scores = st.toggle("📊 All scores", key="show_all_scores", value=False)
        with fb_cols[2]:
            if _generating:
                st.toggle("⏳ Generating…", key="_gen_toggle_busy", value=True, disabled=True)
            else:
                def _on_gen_toggle():
                    if st.session_state.get("_gen_toggle"):
                        st.session_state["_header_generating"] = True
                        st.session_state["_gen_toggle"] = False
                st.toggle("⚡ Generate", key="_gen_toggle", value=False, on_change=_on_gen_toggle)

        if show_scores:
            _render_scores_panel()
            return
    else:
        # No fallback — show generate toggle right-aligned
        _generating = st.session_state.get("_header_generating", False)
        nf_cols = st.columns([4, 1])
        with nf_cols[1]:
            if _generating:
                st.toggle("⏳ Generating…", key="_gen_toggle_busy", value=True, disabled=True)
            else:
                def _on_gen_toggle_nf():
                    if st.session_state.get("_gen_toggle"):
                        st.session_state["_header_generating"] = True
                        st.session_state["_gen_toggle"] = False
                st.toggle("⚡ Generate", key="_gen_toggle", value=False, on_change=_on_gen_toggle_nf)

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
        "RSI": "Not enough data",
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

    rsi_state = str(selected_row.get("rsi_state", "") or "")
    if rsi_state:
        out["RSI"] = rsi_state.capitalize()

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


def render_chart(selected_row: pd.Series, prices_df: pd.DataFrame, *, signal_date: str | None = None, exit_date: str | None = None) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ticker = str(selected_row.get("ticker", ""))
    t = prices_df[prices_df["Ticker"] == ticker].copy().sort_values("Date")
    if t.empty:
        st.info("No chart data for this stock.")
    else:
        t["SMA50"] = t["Close"].rolling(50).mean()
        t["SMA200"] = t["Close"].rolling(200).mean()
        t = t.tail(120)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )

        fig.add_trace(go.Candlestick(
            x=t["Date"], open=t["Open"], high=t["High"],
            low=t["Low"], close=t["Close"], name="Price",
            increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=t["Date"], y=t["SMA50"], name="SMA 50",
            line=dict(color="#3b82f6", width=1.5),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=t["Date"], y=t["SMA200"], name="SMA 200",
            line=dict(color="#f59e0b", width=1.5),
        ), row=1, col=1)

        colors = [
            "#22c55e" if c >= o else "#ef4444"
            for c, o in zip(t["Close"], t["Open"])
        ]
        fig.add_trace(go.Bar(
            x=t["Date"], y=t["Volume"], name="Volume",
            marker_color=colors, opacity=0.5,
        ), row=2, col=1)

        # Vertical marker lines for signal/exit dates
        if signal_date:
            _sd = str(pd.to_datetime(signal_date).date())
            fig.add_vline(x=_sd, line_width=1.5, line_dash="dash", line_color="#38bdf8", row="all", col=1)
            fig.add_annotation(x=_sd, y=1.06, yref="paper", text="Signal", showarrow=False,
                               font=dict(color="#38bdf8", size=10), xanchor="left")
        if exit_date and str(exit_date) not in ("-", "", "nan", "None", "NaT"):
            _ed = str(pd.to_datetime(exit_date).date())
            fig.add_vline(x=_ed, line_width=1.5, line_dash="dash", line_color="#f472b6", row="all", col=1)
            fig.add_annotation(x=_ed, y=1.06, yref="paper", text="Exit", showarrow=False,
                               font=dict(color="#f472b6", size=10), xanchor="right")

        fig.update_layout(
            height=480,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            xaxis2=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e293b"),
            yaxis2=dict(showgrid=False),
        )

        st.plotly_chart(fig, use_container_width=True)
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
    rsi_state = str(selected_row.get("rsi_state", "") or "").lower()
    if rsi_state in {"stretched", "strong"}:
        notes.append("RSI is high, so entry may be stretched.")
    elif rsi_state == "weak":
        notes.append("RSI is still weak for a breakout.")
    elif rsi_state == "healthy":
        notes.append("RSI is in a healthy range, but normal risk rules still apply.")
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
    rsi = selected_row.get("score_rsi")

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
    rsi = float(rsi) if pd.notna(rsi) else 50.0

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

    c_trend = round(trend * WEIGHT_TREND, 1)
    c_setup = round(setup * WEIGHT_SETUP, 1)
    c_volume = round(volume * WEIGHT_VOLUME, 1)
    c_risk = round(risk * WEIGHT_RISK, 1)
    c_rsi = round(rsi * WEIGHT_RSI, 1)

    running = 0.0
    lines: list[str] = []

    running = round(running + c_trend, 1)
    if trend_strength_pct is not None:
        trend_label = "high" if trend_strength_pct >= 8 else ("moderate" if trend_strength_pct >= 2 else "low")
        lines.append(
            f"- Trend strength is {trend_label} ({trend_strength_pct:.2f}% gap between SMA50 and SMA200). Trend score is {trend:.1f} after clipping to the 0-100 band, adding +{c_trend:.1f} (28%), running total {running:.1f}."
        )
    else:
        lines.append(
            f"- Trend score is {trend:.1f}. Trend inputs are limited for this row, and this still adds +{c_trend:.1f} (28%), running total {running:.1f}."
        )

    running = round(running + c_setup, 1)
    if setup_strength_pct is not None:
        setup_label = "strong" if setup_strength_pct >= 3 else ("decent" if setup_strength_pct >= 1 else "soft")
        lines.append(
            f"- Breakout setup is {setup_label} ({setup_strength_pct:.2f}% above recent reference high). Setup score is {setup:.1f} after clipping to 0-100, adding +{c_setup:.1f} (28%), running total {running:.1f}."
        )
    else:
        lines.append(
            f"- Setup score is {setup:.1f}. Setup inputs are limited for this row, and this adds +{c_setup:.1f} (28%), running total {running:.1f}."
        )

    running = round(running + c_volume, 1)
    if volume_ratio is not None:
        volume_label = "strong" if volume_ratio >= 1.8 else ("healthy" if volume_ratio >= 1.2 else "light")
        lines.append(
            f"- Volume support is {volume_label} ({volume_ratio:.2f}x of 20-day average volume). Volume score is {volume:.1f} after clipping to 0-100, adding +{c_volume:.1f} (19%), running total {running:.1f}."
        )
    else:
        lines.append(
            f"- Volume score is {volume:.1f}. Volume inputs are limited for this row, and this adds +{c_volume:.1f} (19%), running total {running:.1f}."
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

    rsi_val = selected_row.get("rsi_14")
    rsi_state = selected_row.get("rsi_state")
    if pd.notna(rsi_val) and pd.notna(rsi_state):
        try:
            rsi_num = float(rsi_val)
            state_label = str(rsi_state).capitalize()
            running = round(running + c_rsi, 1)
            lines.append(
                f"- RSI component is {state_label} at {rsi_num:.0f}. RSI score is {rsi:.1f} after clipping to 0-100, adding +{c_rsi:.1f} (5%), running total {running:.1f}."
            )
        except Exception:
            pass

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

    # Candlestick chart — always shown right under the name
    render_chart(selected_row, prices_df)

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
    a1, a2 = st.columns(2)
    with a1:
        if st.button("Show past results", key="show_past_btn", width="stretch"):
            st.session_state["show_past_results"] = not bool(st.session_state.get("show_past_results", False))
            st.rerun()
    with a2:
        if st.button("Show things to watch", key="show_watch_btn", width="stretch"):
            st.session_state["show_watchouts"] = not bool(st.session_state.get("show_watchouts", False))
            st.rerun()

    if st.session_state.get("show_past_results"):
        render_past_results(selected_row, all_signals, prices_df)
    if st.session_state.get("show_watchouts"):
        render_watchouts(selected_row, checks)

    if is_remote_runtime():
        render_telegram_action(selected_row, allow_actions=allow_actions)


def render_tomorrow_screen(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    allow_actions: bool,
    data_updated: str | None,
) -> None:
    stocks_df, latest_signal_date = _prepare_tomorrow_list(signals_df, prices_df)

    min_score = float(st.session_state.get("min_score", 0))

    # Total stocks considered in the whole setup:
    # - Prefer configured universe (universe_tickers.txt)
    # - Fallback to all tickers in prices_eod.csv
    # - Fallback to all tickers present in signals
    total_considered: int | None = None
    try:
        universe_path = DATA_DIR / "universe_tickers.txt"
        if universe_path.is_file():
            lines = universe_path.read_text(encoding="utf-8").splitlines()
            universe = [
                line.strip()
                for line in lines
                if line.strip() and not line.lstrip().startswith("#")
            ]
            total_considered = len(set(universe)) if universe else None
        elif not prices_df.empty and "Ticker" in prices_df.columns:
            total_considered = int(prices_df["Ticker"].astype(str).nunique())
        elif not signals_df.empty and "ticker" in signals_df.columns:
            total_considered = int(signals_df["ticker"].astype(str).nunique())
    except Exception:
        total_considered = None

    # Get the timestamp of when signals were last generated (file modification time).
    signals_generated: str | None = None
    try:
        if SIGNALS_CSV.is_file():
            _sig_mtime = SIGNALS_CSV.stat().st_mtime
            signals_generated = datetime.fromtimestamp(_sig_mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    # Treat stale signal dates as "no triggers for tomorrow" and fall back.
    stale_for_tomorrow = False
    if latest_signal_date:
        latest_dt = pd.to_datetime(latest_signal_date, errors="coerce")
        if pd.notna(latest_dt):
            days_diff = (date.today() - latest_dt.date()).days
            if days_diff > 1:
                stale_for_tomorrow = True

    fallback_note: str | None = None

    # --- Execute pending refresh / generate actions before rendering anything ---
    if st.session_state.get("_header_refreshing"):
        ok, msg = refresh_prices()
        st.session_state["_header_refreshing"] = False
        if ok:
            load_stock_scores.clear()
            st.rerun()
        else:
            st.error(msg or "Price refresh failed.")

    if st.session_state.get("_header_generating"):
        ok, msg = generate_triggers(backfill=True)
        st.session_state["_header_generating"] = False
        if ok:
            load_stock_scores.clear()
            st.rerun()
        else:
            st.error(msg or "Signal generation failed.")

    if stocks_df.empty or stale_for_tomorrow:
        fallback_df = _prepare_recent_recommendations(signals_df, days=7, prices_df=prices_df)
        fallback_df = fallback_df[fallback_df["signal_score"] >= min_score].copy()
        if fallback_df.empty:
            if stale_for_tomorrow:
                fallback_note = "No triggers for tomorrow and none in the last 7 days."
            else:
                fallback_note = "No triggers for tomorrow or in the last 7 days."
            render_header(
                latest_signal_date=latest_signal_date,
                total_count=0,
                total_considered=total_considered,
                data_updated=data_updated,
                signals_generated=signals_generated,
                fallback_note=fallback_note,
            )
            return
        stocks_df = fallback_df
        if stale_for_tomorrow:
            fallback_note = "Latest signals are not from today. Showing signals from the last 7 days."
        else:
            fallback_note = "No triggers for tomorrow. Showing signals from the last 7 days."
    else:
        stocks_df = stocks_df[stocks_df["signal_score"] >= min_score].copy()
        if stocks_df.empty:
            fallback_df = _prepare_recent_recommendations(signals_df, days=7, prices_df=prices_df)
            fallback_df = fallback_df[fallback_df["signal_score"] >= min_score].copy()
            if fallback_df.empty:
                fallback_note = "No stocks match your score filter in the last 7 days."
                render_header(
                    latest_signal_date=latest_signal_date,
                    total_count=0,
                    total_considered=total_considered,
                    data_updated=data_updated,
                    signals_generated=signals_generated,
                    fallback_note=fallback_note,
                )
                return
            stocks_df = fallback_df
            fallback_note = "No triggers for tomorrow. Showing signals from the last 7 days."

    # Re-render header with the correct total_count and optional fallback note.
    render_header(
        latest_signal_date=latest_signal_date,
        total_count=len(stocks_df),
        total_considered=total_considered,
        data_updated=data_updated,
        signals_generated=signals_generated,
        fallback_note=fallback_note,
    )
    # Store note for use directly above the Tomorrow's stock list section.
    st.session_state["tomorrow_fallback_note"] = fallback_note

    sort_by = str(st.session_state.get("sort_by", "Score (high to low)"))
    if sort_by == "Risk (low to high)":
        sort_cols = ["risk_pct"]
        asc = [True]
        if "ui_score" in stocks_df.columns:
            sort_cols.append("ui_score")
            asc.append(False)
        sort_cols.append("ticker")
        asc.append(True)
        stocks_df.sort_values(sort_cols, ascending=asc, inplace=True)
    elif sort_by == "Ticker (A to Z)":
        stocks_df.sort_values(["ticker"], inplace=True)
    else:
        sort_cols = []
        asc = []
        if "ui_score" in stocks_df.columns:
            sort_cols.append("ui_score")
            asc.append(False)
        else:
            sort_cols.append("signal_score")
            asc.append(False)
        sort_cols.extend(["signal_score", "risk_pct", "ticker"])
        asc.extend([False, True, True])
        stocks_df.sort_values(sort_cols, ascending=asc, inplace=True)

    selected = st.session_state.get("selected_stock")
    options = stocks_df["ticker"].astype(str).tolist()
    if selected not in options:
        st.session_state["selected_stock"] = options[0]
        st.session_state["show_chart"] = False
        st.session_state["show_past_results"] = False
        st.session_state["show_watchouts"] = False

    selected_ticker = str(st.session_state.get("selected_stock"))
    selected_row = stocks_df[stocks_df["ticker"].astype(str) == selected_ticker].iloc[0]

    # When "All scores" panel is open, show full-width scores grid instead of split layout.
    if st.session_state.get("show_all_scores"):
        render_stock_list(stocks_df)
    else:
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

refresh_info = get_prices_refresh_info(prices)

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
    render_tomorrow_screen(
        signals,
        prices,
        allow_actions=tomorrow_allow_actions,
        data_updated=refresh_info["file_updated"],
    )
    st.stop()

# Non-Tomorrow mode: set defaults
allow_actions = not IS_STREAMLIT_CLOUD

today_str = date.today().isoformat()
last_refresh_date = st.session_state.get("last_refresh_date")

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

    # --- Signal Performance Tracker ---
    if not signals.empty and not prices.empty:

        # ── Rescore toggle inline with description ──
        _bt_desc_col, _bt_rescore_col = st.columns([5, 1.5])
        with _bt_desc_col:
            st.caption("Auto-track every generated buy signal: buy 1 lot at entry, target +6%, stop −7%.")
        with _bt_rescore_col:
            _rescore_on = st.toggle("🔄 Refresh scores", key="lab_rescore_toggle", value="_lab_rescored_signals" in st.session_state)
        def _rescore_signals(sigs: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
            """Recompute signal_score for every row using the current algo."""
            sigs = sigs.copy()
            px = px.copy()
            px["Date"] = pd.to_datetime(px["Date"])
            for i in sigs.index:
                ticker = str(sigs.at[i, "ticker"])
                sig_date = pd.to_datetime(sigs.at[i, "signal_date"])
                g = px[px["Ticker"] == ticker].sort_values("Date")
                g = g[g["Date"] <= sig_date].copy()
                if len(g) < 200:
                    continue
                g["SMA50"] = g["Close"].rolling(50).mean()
                g["SMA200"] = g["Close"].rolling(200).mean()
                g["VolAvg20"] = g["Volume"].rolling(20).mean()
                breakout_days = 40
                g["PrevNHighClose"] = g["Close"].shift(1).rolling(breakout_days).max()
                r = g.iloc[-1]
                if any(pd.isna(r[c]) for c in ["SMA50", "SMA200", "VolAvg20", "PrevNHighClose"]):
                    continue
                trend_strength_pct = ((float(r["SMA50"]) / float(r["SMA200"])) - 1.0) * 100.0
                setup_strength_pct = ((float(r["Close"]) / float(r["PrevNHighClose"])) - 1.0) * 100.0
                volume_ratio = float(r["Volume"]) / float(r["VolAvg20"]) if float(r["VolAvg20"]) > 0 else 1.0
                stop_pct_eff = float(sigs.at[i, "stop_pct"]) if pd.notna(sigs.at[i, "stop_pct"]) else 7.0
                rsi_value = None
                if _compute_rsi_shared is not None:
                    try:
                        rsi_value = _compute_rsi_shared(g["Close"].astype(float), period=14)
                    except Exception:
                        pass
                _, _, _, _, _, new_score = _build_score_components(
                    trend_strength_pct=trend_strength_pct,
                    setup_strength_pct=setup_strength_pct,
                    volume_ratio=volume_ratio,
                    stop_pct_eff=stop_pct_eff,
                    rsi_value=rsi_value,
                )
                sigs.at[i, "signal_score"] = new_score
            return sigs

        if _rescore_on:
            if "_lab_rescored_signals" not in st.session_state:
                st.session_state["_lab_rescored_signals"] = _rescore_signals(signals, prices)
                st.rerun()
            _lab_signals = st.session_state["_lab_rescored_signals"]
        else:
            if "_lab_rescored_signals" in st.session_state:
                del st.session_state["_lab_rescored_signals"]
            _lab_signals = signals

        _lab_c1, _lab_c2, _lab_c3, _lab_c4 = st.columns(4)
        with _lab_c1:
            _lab_tgt = st.number_input("Target %", min_value=1.0, max_value=50.0, value=6.0, step=0.5, key="lab_d_target")
        with _lab_c2:
            _lab_stp = st.number_input("Stop %", min_value=1.0, max_value=50.0, value=7.0, step=0.5, key="lab_d_stop")
        with _lab_c3:
            _lab_cap = st.number_input("₹ per trade", min_value=1000.0, max_value=500000.0, value=10000.0, step=1000.0, key="lab_d_capital")
        with _lab_c4:
            _lab_min_score = st.number_input("Min score", min_value=0, max_value=100, value=0, step=5, key="lab_d_min_score")

        _filtered_signals = _lab_signals if _lab_min_score == 0 else _lab_signals[_lab_signals["signal_score"].fillna(0) >= _lab_min_score]
        _tracker = build_signal_tracker(_filtered_signals, prices, target_pct=_lab_tgt, stop_pct=_lab_stp, capital_per_trade=_lab_cap)
        if not _tracker.empty:
            _n_total = len(_tracker)
            _n_tgt = int((_tracker["status"] == "Target Hit ✅").sum())
            _n_stp = int((_tracker["status"] == "Stop Hit 🛑").sum())
            _n_hold = int((_tracker["status"] == "Holding").sum())
            _t_inv = _tracker["invested"].sum()
            _t_cur = _tracker["current_value"].sum()
            _t_pnl = _tracker["pnl"].sum()
            _ov_ret = ((_t_cur / _t_inv) - 1) * 100 if _t_inv > 0 else 0.0

            _m1, _m2, _m3, _m4, _m5 = st.columns(5)
            _m1.metric("Total Signals", _n_total)
            _m2.metric("Target Hit ✅", _n_tgt)
            _m3.metric("Stop Hit 🛑", _n_stp)
            _m4.metric("Holding", _n_hold)
            _m5.metric("Overall Return", f"{_ov_ret:.1f}%", delta=f"₹{_t_pnl:,.0f}")

            _m6, _m7, _m8 = st.columns(3)
            _m6.metric("Total Invested", f"₹{_t_inv:,.0f}")
            _m7.metric("Current Value", f"₹{_t_cur:,.0f}")
            _wr = (_n_tgt / (_n_tgt + _n_stp) * 100) if (_n_tgt + _n_stp) > 0 else 0.0
            _m8.metric("Win Rate", f"{_wr:.0f}%")

            _lab_sf = st.selectbox("Filter by status", ["All", "Target Hit ✅", "Stop Hit 🛑", "Holding"], key="lab_d_sf")
            _view = _tracker if _lab_sf == "All" else _tracker[_tracker["status"] == _lab_sf]
            _sc = [c for c in ["signal_date", "ticker", "entry_price", "qty", "invested", "target_price", "stop_price",
                                "latest_close", "current_value", "pnl", "return_pct", "days_held", "exit_date", "status", "signal_score"] if c in _view.columns]
            _view_display = _view[_sc].copy()
            _float_cols = _view_display.select_dtypes(include=["float64", "float32"]).columns.tolist()
            for _fc in _float_cols:
                _view_display[_fc] = _view_display[_fc].round(2)

            _had_sel = st.session_state.get("_lab_d_had_sel", False)
            if _had_sel:
                _tbl_col, _chart_col = st.columns([3, 2])
            else:
                _tbl_col = st.container()
                _chart_col = None
            with _tbl_col:
                _sel_ev = st.dataframe(
                    _view_display,
                    width="stretch",
                    hide_index=True,
                    height=500,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="lab_d_tracker_sel",
                )
            _sel_rows = _sel_ev.selection.rows if _sel_ev and _sel_ev.selection else []
            if _sel_rows:
                st.session_state["_lab_d_had_sel"] = True
                _picked = _view.iloc[_sel_rows[0]]
                _chart_row = pd.Series({"ticker": str(_picked["ticker"]) + ".NS"})
                if _chart_col is not None:
                    with _chart_col:
                        st.markdown(f"### 📈 {_picked['ticker']}")
                        render_chart(_chart_row, prices,
                                     signal_date=str(_picked.get("signal_date", "")),
                                     exit_date=str(_picked.get("exit_date", "")))
                else:
                    st.rerun()
            else:
                if _had_sel:
                    st.session_state["_lab_d_had_sel"] = False
                    st.rerun()
        st.divider()

    # --- Manual add form ---
    with st.expander("➕ Add manual position"):
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
            submit = st.form_submit_button("Add position")

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

    if not dummy_lab_live.empty:
        with st.expander("📋 Manual positions"):
            open_lab = dummy_lab_live[dummy_lab_live["status"].astype(str) == "Watching"].copy()
            if open_lab.empty:
                open_lab = dummy_lab_live.copy()

            show_cols = [
                "created_at", "source_signal_date", "ticker", "pattern",
                "entry_price", "stop_price", "latest_close", "capital",
                "current_value", "pnl", "current_return_pct", "distance_to_stop_pct",
                "status", "note",
            ]
            show_cols = [c for c in show_cols if c in open_lab.columns]
            view_df = open_lab[show_cols].copy()
            for c in ["entry_price", "stop_price", "latest_close", "capital", "current_value", "pnl", "current_return_pct", "distance_to_stop_pct"]:
                if c in view_df.columns:
                    view_df[c] = pd.to_numeric(view_df[c], errors="coerce").round(2)
            render_table(view_df.sort_values(["created_at", "ticker"], ascending=[False, True]), height=360)

            st.markdown("### Manage positions")
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
                winner_cols = ["ticker", "ret_20d_pct", "ret_60d_pct", "health", "score"]
                if "rsi14" in top_winners.columns:
                    winner_cols.append("rsi14")
                render_table(top_winners[winner_cols], height=240)
            with c2:
                st.markdown("### Top 20-day Laggards")
                loser_cols = ["ticker", "ret_20d_pct", "ret_60d_pct", "health", "score"]
                if "rsi14" in top_losers.columns:
                    loser_cols.append("rsi14")
                render_table(top_losers[loser_cols], height=240)

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
            if "rsi14" in market_view.columns:
                view_cols.insert(3, "rsi14")
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
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                stock_hist["SMA50"] = stock_hist["Close"].rolling(50).mean()
                stock_hist["SMA200"] = stock_hist["Close"].rolling(200).mean()
                recent = stock_hist.tail(120)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
                fig.add_trace(go.Candlestick(
                    x=recent["Date"], open=recent["Open"], high=recent["High"],
                    low=recent["Low"], close=recent["Close"], name="Price",
                    increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
                ), row=1, col=1)
                fig.add_trace(go.Scatter(x=recent["Date"], y=recent["SMA50"], name="SMA 50", line=dict(color="#3b82f6", width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=recent["Date"], y=recent["SMA200"], name="SMA 200", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
                colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(recent["Close"], recent["Open"])]
                fig.add_trace(go.Bar(x=recent["Date"], y=recent["Volume"], name="Volume", marker_color=colors, opacity=0.5), row=2, col=1)
                fig.update_layout(
                    height=480, margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#fafafa"),
                    xaxis2=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e293b"), yaxis2=dict(showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True)

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
                    import plotly.graph_objects as go
                    t_prices.sort_values("Date", inplace=True)
                    recent = t_prices.tail(120)
                    recent["SMA50"] = t_prices["Close"].rolling(50).mean().iloc[-120:]
                    recent["SMA200"] = t_prices["Close"].rolling(200).mean().iloc[-120:]
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=recent["Date"], open=recent["Open"], high=recent["High"],
                        low=recent["Low"], close=recent["Close"], name="Price",
                        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
                    ))
                    fig.add_trace(go.Scatter(x=recent["Date"], y=recent["SMA50"], name="SMA 50", line=dict(color="#3b82f6", width=1.5)))
                    fig.add_trace(go.Scatter(x=recent["Date"], y=recent["SMA200"], name="SMA 200", line=dict(color="#f59e0b", width=1.5)))
                    fig.update_layout(
                        height=420, margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_rangeslider_visible=False,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#fafafa"),
                        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
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
    st.caption("Auto-track every generated buy signal: buy 1 lot at entry, target +6%, stop −7%.")

    # --- Signal Performance Tracker (auto-generated) ---
    if signals.empty:
        st.info("No buy signals generated yet. Run 'Generate' from the Tomorrow view first.")
    elif prices.empty:
        st.warning("Price data not available. Refresh prices first.")
    else:
        lab_c1, lab_c2, lab_c3 = st.columns(3)
        with lab_c1:
            lab_target = st.number_input("Target %", min_value=1.0, max_value=50.0, value=6.0, step=0.5, key="lab_target_pct")
        with lab_c2:
            lab_stop = st.number_input("Stop %", min_value=1.0, max_value=50.0, value=7.0, step=0.5, key="lab_stop_pct")
        with lab_c3:
            lab_capital = st.number_input("₹ per trade", min_value=1000.0, max_value=500000.0, value=10000.0, step=1000.0, key="lab_capital")

        tracker_df = build_signal_tracker(
            signals, prices,
            target_pct=lab_target,
            stop_pct=lab_stop,
            capital_per_trade=lab_capital,
        )

        if tracker_df.empty:
            st.info("No signal data to track.")
        else:
            # Summary metrics
            n_total = len(tracker_df)
            n_target = int((tracker_df["status"] == "Target Hit ✅").sum())
            n_stop = int((tracker_df["status"] == "Stop Hit 🛑").sum())
            n_holding = int((tracker_df["status"] == "Holding").sum())
            total_invested = tracker_df["invested"].sum()
            total_current = tracker_df["current_value"].sum()
            total_pnl = tracker_df["pnl"].sum()
            overall_return = ((total_current / total_invested) - 1) * 100 if total_invested > 0 else 0.0

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Signals", n_total)
            m2.metric("Target Hit ✅", n_target)
            m3.metric("Stop Hit 🛑", n_stop)
            m4.metric("Holding", n_holding)
            m5.metric("Overall Return", f"{overall_return:.1f}%", delta=f"₹{total_pnl:,.0f}")

            m6, m7, m8 = st.columns(3)
            m6.metric("Total Invested", f"₹{total_invested:,.0f}")
            m7.metric("Current Value", f"₹{total_current:,.0f}")
            win_rate = (n_target / (n_target + n_stop) * 100) if (n_target + n_stop) > 0 else 0.0
            m8.metric("Win Rate", f"{win_rate:.0f}%", help="Target hit / (Target hit + Stop hit)")

            # Filter
            status_opts = ["All", "Target Hit ✅", "Stop Hit 🛑", "Holding"]
            lab_status_filter = st.selectbox("Filter by status", options=status_opts, key="lab_status_filter")
            view = tracker_df.copy()
            if lab_status_filter != "All":
                view = view[view["status"] == lab_status_filter]

            show_cols = [
                "signal_date", "ticker", "entry_price", "qty", "invested",
                "target_price", "stop_price", "latest_close", "current_value",
                "pnl", "return_pct", "days_held", "exit_date", "status", "signal_score",
            ]
            show_cols = [c for c in show_cols if c in view.columns]
            _view_tab = view[show_cols].copy()
            _float_cols_tab = _view_tab.select_dtypes(include=["float64", "float32"]).columns.tolist()
            for _fc_t in _float_cols_tab:
                _view_tab[_fc_t] = _view_tab[_fc_t].round(2)

            _had_sel_tab = st.session_state.get("_lab_tab_had_sel", False)
            if _had_sel_tab:
                _tbl_col_tab, _chart_col_tab = st.columns([3, 2])
            else:
                _tbl_col_tab = st.container()
                _chart_col_tab = None
            with _tbl_col_tab:
                _sel_ev_tab = st.dataframe(
                    _view_tab,
                    width="stretch",
                    hide_index=True,
                    height=500,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="lab_tab_tracker_sel",
                )
            _sel_rows_tab = _sel_ev_tab.selection.rows if _sel_ev_tab and _sel_ev_tab.selection else []
            if _sel_rows_tab:
                st.session_state["_lab_tab_had_sel"] = True
                _picked_tab = view.iloc[_sel_rows_tab[0]]
                _chart_row_tab = pd.Series({"ticker": str(_picked_tab["ticker"]) + ".NS"})
                if _chart_col_tab is not None:
                    with _chart_col_tab:
                        st.markdown(f"### 📈 {_picked_tab['ticker']}")
                        render_chart(_chart_row_tab, prices,
                                     signal_date=str(_picked_tab.get("signal_date", "")),
                                     exit_date=str(_picked_tab.get("exit_date", "")))
                else:
                    st.rerun()
            else:
                if _had_sel_tab:
                    st.session_state["_lab_tab_had_sel"] = False
                    st.rerun()

            st.download_button(
                "Download tracker CSV",
                data=to_csv_bytes(view[show_cols]),
                file_name="signal_tracker.csv",
                mime="text/csv",
                key="download_signal_tracker",
            )

    # --- Manual positions (kept as expander) ---
    with st.expander("➕ Add manual position"):
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
            submit = st.form_submit_button("Add position")

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

    if not dummy_lab_live.empty:
        with st.expander("📋 Manual positions"):
            open_lab = dummy_lab_live[dummy_lab_live["status"].astype(str) == "Watching"].copy()
            if open_lab.empty:
                open_lab = dummy_lab_live.copy()

            show_cols = [
                "created_at", "source_signal_date", "ticker", "pattern",
                "entry_price", "stop_price", "latest_close", "capital",
                "current_value", "pnl", "current_return_pct", "distance_to_stop_pct",
                "status", "note",
            ]
            show_cols = [c for c in show_cols if c in open_lab.columns]
            view_df = open_lab[show_cols].copy()
            for c in ["entry_price", "stop_price", "latest_close", "capital", "current_value", "pnl", "current_return_pct", "distance_to_stop_pct"]:
                if c in view_df.columns:
                    view_df[c] = pd.to_numeric(view_df[c], errors="coerce").round(2)
            render_table(view_df.sort_values(["created_at", "ticker"], ascending=[False, True]), height=360)

            st.markdown("### Manage positions")
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
