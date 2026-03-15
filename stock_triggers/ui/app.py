"""Simple Streamlit UI for viewing Pattern A signals.

This is a starting point. It reads signals_pattern_a.csv and displays
signals in a table with basic filters.
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import subprocess
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
TRIGGERS_DIR = ROOT / "stock_triggers"
SCRIPTS_DIR = TRIGGERS_DIR / "scripts"
DATA_DIR = TRIGGERS_DIR / "data"
SIGNALS_CSV = DATA_DIR / "signals_pattern_a.csv"
PRICES_CSV = DATA_DIR / "prices_eod.csv"


st.set_page_config(page_title="Stock Triggers – Pattern A", layout="wide")
st.title("Stock Triggers – Pattern A Signals")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem;}
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


@st.cache_data(show_spinner=False)
def load_signals() -> pd.DataFrame:
    if not SIGNALS_CSV.is_file():
        return pd.DataFrame()
    return pd.read_csv(SIGNALS_CSV)


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    if not PRICES_CSV.is_file():
        return pd.DataFrame()
    df = pd.read_csv(PRICES_CSV, parse_dates=["Date"])
    return df


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
    latest_date_str = latest_date.date().isoformat() if hasattr(latest_date, "date") else str(latest_date)

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
        render_stat_card("Latest Price Date", latest_date_str)
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


signals = load_signals()

# Single summary placeholder so refresh summary appears only once on page.
summary_panel = st.container()


def update_summary_panel(prices_df: pd.DataFrame, signals_df: pd.DataFrame) -> None:
    summary_panel.empty()
    with summary_panel:
        render_refresh_summary(prices_df, signals_df)

# Sidebar – data actions and filters (always visible)
st.sidebar.header("Step 1: Refresh Prices")

today_str = date.today().isoformat()
last_refresh_date = st.session_state.get("last_refresh_date")

st.sidebar.caption(f"Today: {today_str}")
if last_refresh_date:
    st.sidebar.caption(f"Last refresh: {last_refresh_date}")

do_refresh = st.sidebar.button("Refresh prices")

if "show_refresh_actions" not in st.session_state:
    st.session_state["show_refresh_actions"] = False

if do_refresh:
    # Only check and show status/options; do not auto-run refresh.
    st.session_state["show_refresh_actions"] = True

if st.session_state["show_refresh_actions"]:
    prices = load_prices()
    signals = load_signals()

    if last_refresh_date == today_str:
        st.info("Prices were already refreshed today.")
    else:
        st.info("No refresh recorded for today yet.")

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
        if st.button("Repeat data refresh", key="repeat_data_refresh_btn"):
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
        if st.button("Generate trigger", key="generate_trigger_from_refresh_flow_btn"):
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
if "show_trigger_panel" not in st.session_state:
    st.session_state["show_trigger_panel"] = False

if st.sidebar.button("Generate Pattern A trigger"):
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

    if st.button("Run", key="run_trigger_btn", use_container_width=True):
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

filtered = pd.DataFrame()
selected_date = None
if not signals.empty:
    st.sidebar.header("Filters")
    st.sidebar.markdown(
        "Use these filters to narrow down Pattern A signals by date, ticker, and pattern."
    )

    dates = sorted(signals["signal_date"].unique())
    selected_date = st.sidebar.selectbox(
        "Signal date",
        options=dates,
        index=len(dates) - 1,
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

live_tab, backtest_tab = st.tabs(["Live Signals", "Backtesting"])

with live_tab:
    if signals.empty:
        st.warning(
            "No signals to display yet. Use Step 1/Step 2 to refresh data and generate signals."
        )
    else:
        st.subheader(f"Signals for {selected_date}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("# Signals", len(filtered))
        with col2:
            st.metric("# Tickers", filtered["ticker"].nunique())
        with col3:
            st.metric("Patterns", ", ".join(sorted(filtered["pattern"].unique())) or "-")

        st.dataframe(
            filtered,
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("Price chart for a selected signal")

        if prices.empty or filtered.empty:
            st.info("Price history or signals not available for charting.")
        else:
            tickers_for_chart = sorted(filtered["ticker"].unique())
            chart_ticker = st.selectbox("Ticker", options=tickers_for_chart)

            t_prices = prices[prices["Ticker"] == chart_ticker].copy()
            if not t_prices.empty:
                t_prices.sort_values("Date", inplace=True)
                recent = t_prices.tail(120)
                st.line_chart(
                    recent.set_index("Date")["Close"],
                    use_container_width=True,
                )
            else:
                st.info("No price history found for this ticker in prices_eod.csv.")

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

            if st.button("Run Backtest", key="run_backtest_btn", use_container_width=True):
                all_signals: list[pd.DataFrame] = []
                for d in eligible_dates:
                    hist_to_date = prices[prices["Date"] <= d].copy()
                    day_signals = compute_pattern_a_signals_for_date(
                        hist_to_date,
                        as_of_date=d,
                        breakout_days=int(bt_breakout_days),
                        volume_multiplier=float(bt_volume_multiplier),
                        stop_pct=float(bt_stop_pct),
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
                st.dataframe(bt_signals, use_container_width=True)

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
                    st.dataframe(styled, use_container_width=True)

st.caption(
    "Data source: stock_triggers/data/prices_eod.csv and "
    "stock_triggers/data/signals_pattern_a.csv – generated by the scripts in "
    "stock_triggers/scripts/."
)
