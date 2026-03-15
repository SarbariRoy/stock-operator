"""Simple Streamlit UI for viewing Pattern A signals.

This is a starting point. It reads signals_pattern_a.csv and displays
signals in a table with basic filters.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import date
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
PRICES_CSV = DATA_DIR / "prices_eod.csv"
SECRETS_FILE = ROOT / "secrets.yml"
IS_STREAMLIT_CLOUD = bool(os.getenv("STREAMLIT_SHARING_MODE")) or bool(os.getenv("STREAMLIT_CLOUD"))


st.set_page_config(page_title="Stock Triggers – Pattern A", layout="wide")
st.markdown(
    "<div class='brand-title'>Stock Triggers by <span class='brand-roy'>Roy</span></div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;600;700&display=swap');
    .block-container {padding-top: 2.2rem;}
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


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


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
) -> pd.DataFrame:
    """Compute Pattern A signals for one date from the provided price history."""

    all_rows: list[dict] = []
    for ticker, g in prices.groupby("Ticker", sort=True):
        g = g.copy().sort_values("Date")

        g["SMA50"] = g["Close"].rolling(50).mean()
        g["SMA200"] = g["Close"].rolling(200).mean()
        g["VolAvg20"] = g["Volume"].rolling(20).mean()
        g["PrevNHighClose"] = g["Close"].shift(1).rolling(breakout_days).max()

        row = g[g["Date"] == as_of_date]
        if row.empty:
            continue
        r = row.iloc[0]

        needed = [r["SMA50"], r["SMA200"], r["VolAvg20"], r["PrevNHighClose"]]
        if any(pd.isna(v) for v in needed):
            continue

        cond_trend = bool(r["SMA50"] > r["SMA200"])
        cond_price = bool((r["Close"] > r["SMA50"]) and (r["Close"] > r["SMA200"]))
        cond_breakout = bool(r["Close"] > r["PrevNHighClose"])
        cond_volume = bool(r["Volume"] >= volume_multiplier * r["VolAvg20"])

        if not (cond_trend and cond_price and cond_breakout and cond_volume):
            continue

        entry_price = float(r["Close"])
        stop_price = entry_price * (1.0 - stop_pct / 100.0)
        all_rows.append(
            {
                "signal_date": as_of_date.date().isoformat(),
                "ticker": ticker,
                "pattern": f"A_breakout_{breakout_days}d",
                "entry_price": round(entry_price, 4),
                "stop_pct": float(stop_pct),
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
    return pd.DataFrame(all_rows, columns=cols)


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
) -> pd.DataFrame:
    """Evaluate each generated trigger using future data from its own signal_date."""

    cols = [
        "signal_date",
        "ticker",
        "pattern",
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

        stop_rows = fut[fut["Low"] <= stop_price]
        if not stop_rows.empty:
            exit_row = stop_rows.iloc[0]
            exit_price = stop_price
            exit_date = exit_row["Date"]
            outcome = "stop_hit"
        else:
            exit_row = fut.iloc[-1]
            exit_price = float(exit_row["Close"])
            exit_date = exit_row["Date"]
            outcome = "held_to_window_end"

        ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
        if outcome == "stop_hit":
            quality = "--"
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_signals: list[pd.DataFrame] = []
    for d in eligible_dates:
        hist_to_date = prices[prices["Date"] <= d].copy()
        day_signals = compute_pattern_a_signals_for_date(
            hist_to_date,
            as_of_date=d,
            breakout_days=int(breakout_days),
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
        )
        if not day_signals.empty:
            all_signals.append(day_signals)

    if all_signals:
        bt_signals = pd.concat(all_signals, ignore_index=True)
        bt_signals.sort_values(["signal_date", "ticker"], inplace=True)
    else:
        bt_signals = pd.DataFrame(
            columns=["signal_date", "ticker", "pattern", "entry_price", "stop_pct", "stop_price"]
        )

    bt_eval = evaluate_generated_triggers(
        bt_signals,
        prices,
        hold_days=int(hold_days),
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


signals = load_signals()
sell_signals = load_sell_signals()

# Single summary placeholder so refresh summary appears only once on page.
summary_panel = st.container()


def update_summary_panel(prices_df: pd.DataFrame, signals_df: pd.DataFrame) -> None:
    summary_panel.empty()
    with summary_panel:
        render_refresh_summary(prices_df, signals_df)

# Sidebar – data actions and filters (always visible)
st.sidebar.header("Execution Mode")
allow_actions = st.sidebar.toggle(
    "Enable refresh/trigger actions",
    value=(not IS_STREAMLIT_CLOUD),
    help="Keep OFF on Streamlit Cloud for read-only dashboard mode. Turn ON when you want this app to run local scripts.",
)
compact_mode = st.sidebar.toggle(
    "Compact mobile mode",
    value=False,
    help="Use tighter spacing and smaller cards for phone screens.",
)
if compact_mode:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.1rem; padding-bottom: 0.8rem;}
        .brand-title {font-size: 1.55rem; line-height: 1.2; margin-top: 0.2rem; margin-bottom: 0.5rem;}
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
    st.sidebar.info("Read-only mode: refresh and trigger generation are disabled.")

st.sidebar.header("Step 1: Refresh Prices")

today_str = date.today().isoformat()
last_refresh_date = st.session_state.get("last_refresh_date")

st.sidebar.caption(f"Today: {today_str}")
if last_refresh_date:
    st.sidebar.caption(f"Last refresh: {last_refresh_date}")

do_refresh = st.sidebar.button("Refresh prices", disabled=not allow_actions)

if "show_refresh_actions" not in st.session_state:
    st.session_state["show_refresh_actions"] = False

if "show_trigger_panel" not in st.session_state:
    st.session_state["show_trigger_panel"] = False

if not allow_actions:
    st.session_state["show_refresh_actions"] = False
    st.session_state["show_trigger_panel"] = False

if do_refresh:
    # Only check and show status/options; do not auto-run refresh.
    st.session_state["show_refresh_actions"] = True

if st.session_state["show_refresh_actions"]:
    prices = load_prices()
    signals = load_signals()

    if last_refresh_date == today_str:
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
        if st.button("Repeat data refresh", key="repeat_data_refresh_btn", disabled=not allow_actions):
            with st.spinner("Refreshing prices..."):
                ok, msg = refresh_prices()
            if ok:
                st.session_state["last_refresh_date"] = today_str
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

st.sidebar.header("Step 2: Generate Trigger")
if st.sidebar.button("Generate Pattern A trigger", disabled=not allow_actions):
    st.session_state["show_trigger_panel"] = True

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

prices = load_prices()
latest_trading_date_str = None
if not prices.empty:
    latest_trading_date_str = prices["Date"].max().date().isoformat()

filtered = pd.DataFrame()
selected_date = None
if not signals.empty:
    st.sidebar.header("Filters")
    st.sidebar.markdown(
        "Use these filters to narrow down Pattern A signals by date, ticker, and pattern."
    )

    only_signal_dates = st.sidebar.checkbox(
        "Show only dates with buy signals",
        value=False,
        key="only_signal_dates",
        help="Turn on to hide market dates that have no buy signal.",
    )

    dates = sorted(signals["signal_date"].unique())
    if (not only_signal_dates) and latest_trading_date_str and latest_trading_date_str not in dates:
        dates.append(latest_trading_date_str)
        dates = sorted(dates)

    default_date = latest_trading_date_str if latest_trading_date_str in dates else dates[-1]
    selected_date = st.sidebar.selectbox(
        "Signal date",
        options=dates,
        index=dates.index(default_date),
    )

    all_tickers = sorted(signals["ticker"].unique())
    selected_tickers = st.sidebar.multiselect(
        "Tickers",
        options=all_tickers,
        default=all_tickers,
    )

    all_patterns = sorted(signals["pattern"].unique())
    selected_patterns = st.sidebar.multiselect(
        "Patterns",
        options=all_patterns,
        default=all_patterns,
    )

    filtered = signals.copy()
    filtered = filtered[filtered["signal_date"] == selected_date]
    if selected_tickers:
        filtered = filtered[filtered["ticker"].isin(selected_tickers)]
    if selected_patterns:
        filtered = filtered[filtered["pattern"].isin(selected_patterns)]

portfolio = load_portfolio()
portfolio, added_positions = sync_portfolio_with_buys(signals, portfolio)
portfolio, auto_closed = auto_close_portfolio_with_sells(portfolio, sell_signals)
if added_positions > 0 or auto_closed > 0:
    save_portfolio(portfolio)

portfolio_live = enrich_portfolio_with_live_metrics(portfolio, prices)
needs_action_rows = build_needs_action_rows(portfolio_live)

dashboard_tab, signals_tab, portfolio_tab, backtest_tab, telegram_tab = st.tabs(["Dashboard", "Signals", "Portfolio", "Backtesting", "Telegram"])

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
            st.dataframe(latest_buy_rows, width="stretch")
    with row_b:
        st.markdown("### Sell signals (latest date)")
        if latest_sell_rows.empty:
            st.info("No sell signals yet.")
        else:
            latest_sell_rows = latest_sell_rows.sort_values(["ticker"])
            st.dataframe(latest_sell_rows, width="stretch")

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
        st.dataframe(show_open, width="stretch")

    st.markdown("### Top priorities")
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
        top5 = needs_action_rows[top_cols].head(5).copy()
        for c in ["current_return_pct", "to_target_6pct", "distance_to_stop_pct"]:
            if c in top5.columns:
                top5[c] = top5[c].round(2)
        st.dataframe(top5, width="stretch")

with signals_tab:
    if signals.empty:
        st.warning(
            "No signals yet. Run refresh and trigger steps first."
        )
    else:
        buy_view_tab, sell_view_tab, chart_view_tab = st.tabs(["Buy Signals", "Sell Signals", "Price Chart"])

        with buy_view_tab:
            st.subheader(f"Buy signals for {selected_date}")
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
            st.dataframe(buy_out, width="stretch")
            st.download_button(
                "Download buy signals CSV",
                data=to_csv_bytes(buy_out),
                file_name=f"buy_signals_{selected_date}.csv",
                mime="text/csv",
                key="download_buy_signals_csv",
            )

            st.markdown("#### Why this buy signal?")
            if filtered.empty:
                st.info("No rows to explain.")
            else:
                explain_ticker = st.selectbox(
                    "Choose ticker",
                    options=sorted(filtered["ticker"].unique()),
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
                sell_dates = sorted(sell_signals["sell_signal_date"].unique())
                chosen_sell_date = st.selectbox(
                    "Sell signal date",
                    options=sell_dates,
                    index=len(sell_dates) - 1,
                    key="sell_signal_date_filter",
                )
                sell_filtered = sell_signals[sell_signals["sell_signal_date"] == chosen_sell_date].copy()
                s1, s2, s3 = st.columns(3)
                s1.metric("# Sell Signals", len(sell_filtered))
                s2.metric("# Tickers", sell_filtered["ticker"].nunique())
                s3.metric("Avg Realized Return %", f"{sell_filtered['realized_return_pct'].mean():.2f}")
                st.dataframe(sell_filtered.sort_values(["ticker"]), width="stretch")
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
                chart_ticker = st.selectbox("Ticker", options=tickers_for_chart)

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
        st.dataframe(style_portfolio_status(shown), width="stretch")
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

        st.markdown("### Quick actions")
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
            chosen = st.selectbox("Choose row", options=shown["label"].tolist(), key="portfolio_row")
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
                "Breakout days", min_value=5, max_value=200, value=40, step=1, key="bt_breakout_days"
            )
            bt_volume_multiplier = st.number_input(
                "Volume multiplier", min_value=0.5, max_value=5.0, value=1.5, step=0.1, format="%.2f", key="bt_volume_multiplier"
            )
        with b2:
            bt_stop_pct = st.number_input("Stop %", min_value=1.0, max_value=20.0, value=7.0, step=0.5, format="%.1f", key="bt_stop_pct")
            bt_hold_days = st.slider(
                "Forward evaluation window (days)", min_value=5, max_value=60, value=15, step=1, key="bt_hold_days"
            )
            st.caption("Trigger generation uses only data up to each trigger date (no look-ahead).")

        st.markdown(
            f"""
**Trigger definition used in this backtest**

- SMA50 > SMA200
- Close > SMA50 and Close > SMA200
- Close > previous {int(bt_breakout_days)}-day high close
- Volume >= {float(bt_volume_multiplier):.2f} * 20-day average volume
- Stop loss = {float(bt_stop_pct):.1f}% below entry
"""
        )

        latest_dt = prices["Date"].max()
        target_dt = latest_dt - pd.DateOffset(months=int(bt_hide_months))
        eligible_dates = sorted(prices.loc[prices["Date"] <= target_dt, "Date"].drop_duplicates())

        if not eligible_dates:
            st.error("Not enough history after hiding selected months. Reduce hidden months.")
        else:
            bt_as_of = eligible_dates[-1]
            st.info(
                f"Latest visible date after hiding {bt_hide_months} month(s): {bt_as_of.date().isoformat()} (latest full date is {latest_dt.date().isoformat()})."
            )

            if st.button("Run Backtest", key="run_backtest_btn", width="stretch"):
                bt_signals, bt_eval = run_backtest_for_params(
                    prices,
                    eligible_dates=eligible_dates,
                    breakout_days=int(bt_breakout_days),
                    volume_multiplier=float(bt_volume_multiplier),
                    stop_pct=float(bt_stop_pct),
                    hold_days=int(bt_hold_days),
                )

                st.session_state["bt_result"] = {
                    "as_of": bt_as_of.date().isoformat(),
                    "signals": bt_signals,
                    "evaluated": bt_eval,
                    "hide_months": int(bt_hide_months),
                    "hold_days": int(bt_hold_days),
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

            if pattern_score >= 65:
                st.success("Overall view: pattern quality looks strong on this backtest setup.")
            elif pattern_score >= 50:
                st.info("Overall view: pattern quality is mixed/average on this backtest setup.")
            else:
                st.warning("Overall view: pattern quality looks weak on this backtest setup.")

            with st.expander("Show generated trigger(s)", expanded=True):
                st.dataframe(bt_signals, width="stretch")

            with st.expander("Show trigger quality details", expanded=True):
                if bt_eval.empty:
                    st.info("No evaluated trigger rows available.")
                else:
                    view_cols = [
                        "signal_date",
                        "ticker",
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
                        if outcome == "stop_hit":
                            color = "#fee2e2"
                        elif outcome == "held_to_window_end" and float(row.get("return_pct") or 0) > 0:
                            color = "#dcfce7"
                        elif outcome == "held_to_window_end":
                            color = "#fef3c7"
                        else:
                            color = "#f1f5f9"
                        return [f"background-color: {color}"] * len(row)

                    styled = bt_eval[view_cols].style.apply(_row_style, axis=1)
                    st.dataframe(styled, width="stretch")

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
            {"name": s1_name.strip() or "Setup 1", "breakout_days": int(s1_breakout), "volume_multiplier": float(s1_volume), "stop_pct": float(s1_stop)},
            {"name": s2_name.strip() or "Setup 2", "breakout_days": int(s2_breakout), "volume_multiplier": float(s2_volume), "stop_pct": float(s2_stop)},
            {"name": s3_name.strip() or "Setup 3", "breakout_days": int(s3_breakout), "volume_multiplier": float(s3_volume), "stop_pct": float(s3_stop)},
        ]
        st.dataframe(pd.DataFrame(presets), width="stretch")

        if st.button("Run Compare", key="run_compare_btn", width="stretch"):
            compare_rows: list[dict] = []
            compare_runs: dict[str, pd.DataFrame] = {}

            for p in presets:
                cmp_signals, cmp_eval = run_backtest_for_params(
                    prices,
                    eligible_dates=eligible_dates,
                    breakout_days=int(p["breakout_days"]),
                    volume_multiplier=float(p["volume_multiplier"]),
                    stop_pct=float(p["stop_pct"]),
                    hold_days=int(bt_hold_days),
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
            st.dataframe(compare_table, width="stretch")

            setup_names = compare_table["setup"].tolist()
            selected_setup = st.selectbox("Choose setup for details", options=setup_names, key="bt_compare_setup")
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
                    st.dataframe(monthly, width="stretch")

                st.markdown("### Trade Log")
                log_cols = [
                    "signal_date",
                    "ticker",
                    "outcome",
                    "return_pct",
                    "max_upside_pct",
                    "max_drawdown_pct",
                    "entry_price",
                    "exit_price",
                    "exit_date",
                ]
                log_cols = [c for c in log_cols if c in sel_eval.columns]
                st.dataframe(sel_eval[log_cols].sort_values(["signal_date", "ticker"]), width="stretch")

with telegram_tab:
    st.subheader("Send to Telegram")
    st.caption("This uses TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env or secrets.yml.")

    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        st.warning("Telegram credentials not found. Add them in env or secrets.yml.")

    st.markdown("### Quick send")
    sell_message = build_sell_telegram_message(sell_signals)
    if st.button("Send latest sell signals", key="send_latest_sells_btn", disabled=(not allow_actions)):
        with st.spinner("Sending latest sell signals..."):
            ok, msg = send_telegram_message(token, chat_id, sell_message)
        if ok:
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
            st.success("Message sent.")
        else:
            st.error(msg)

st.caption(
    "Data files used: prices_eod.csv, signals_pattern_a.csv, sell_signals_pattern_a.csv, portfolio_positions.csv."
)
