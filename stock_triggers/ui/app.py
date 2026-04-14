"""Streamlit UI for browsing stock signals, rankings, and backtests."""

from __future__ import annotations

import os
from pathlib import Path
from datetime import date, datetime
import html
import subprocess
import sys
import importlib.util

import pandas as pd
import requests
import streamlit as st
from streamlit_navigation_bar import st_navbar

# Ensure project root is on sys.path so stock_triggers.ui.* imports resolve
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from stock_triggers.ui.documentation_page import (
    build_dataframe_column_config,
    handle_help_query_param,
    render_caption_with_help,
    render_documentation_page,
    render_heading_with_help,
    render_help_button,
    render_table_help_glossary,
    table_help_map,
)
from stock_triggers.coverage_cache import (
    DEFAULT_FORWARD_DAYS as _COV_DEFAULT_FORWARD_DAYS,
    DEFAULT_PATTERN_FAMILIES as _COV_DEFAULT_PATTERN_FAMILIES,
    DEFAULT_RECOGNITION_THRESHOLD as _COV_DEFAULT_RECOGNITION_THRESHOLD,
    DEFAULT_TARGET_RETURN_PCT as _COV_DEFAULT_TARGET_RETURN_PCT,
    load_default_cache_if_valid as _load_default_coverage_cache_if_valid,
)

# Pattern detection modules
import importlib as _il
_pat_a = _il.import_module("stock_triggers.ui.patterns.pattern_a")
_pat_b = _il.import_module("stock_triggers.ui.patterns.pattern_b")
_pat_c = _il.import_module("stock_triggers.ui.patterns.pattern_c_macd")
_pat_d = _il.import_module("stock_triggers.ui.patterns.pattern_d_rsi")
_pat_e = _il.import_module("stock_triggers.ui.patterns.pattern_e_boll")
_pat_f = _il.import_module("stock_triggers.ui.patterns.pattern_f_vwap")
_pat_g = _il.import_module("stock_triggers.ui.patterns.pattern_g_vcp")
_scoring_mod = _il.import_module("stock_triggers.ui.patterns.scoring")

# Candle-shape enhancer modules
_enh_doji = _il.import_module("stock_triggers.ui.enhancers.dragonfly_doji")
_enh_hammer = _il.import_module("stock_triggers.ui.enhancers.hammer")
_enh_marubozu = _il.import_module("stock_triggers.ui.enhancers.bullish_marubozu")
_enh_mstar = _il.import_module("stock_triggers.ui.enhancers.morning_star")
_enh_engulf = _il.import_module("stock_triggers.ui.enhancers.bullish_engulfing")
_enh_harami = _il.import_module("stock_triggers.ui.enhancers.bullish_harami")
_enh_piercing = _il.import_module("stock_triggers.ui.enhancers.piercing_line")
_enh_piercing_variant = _il.import_module("stock_triggers.ui.enhancers.piercing_variant")
_enh_inv_hammer = _il.import_module("stock_triggers.ui.enhancers.inverted_hammer")
_enh_belt_hold = _il.import_module("stock_triggers.ui.enhancers.bullish_belt_hold")
_enh_three_white = _il.import_module("stock_triggers.ui.enhancers.three_white_soldiers")

_ENGULFING_POSITIVE_FAMILIES = {"A", "C", "G"}
_PIERCING_VARIANT_POSITIVE_FAMILIES = {"B"}


def _tag_candle_shapes_fast(
    df: pd.DataFrame,
    prices: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "signal_date",
    add_ns_suffix: bool = False,
) -> pd.DataFrame:
    """Add candle bool columns. Pre-groups prices by Ticker for speed."""
    for c in (
        "candle_doji", "candle_hammer", "candle_marubozu", "candle_morning_star", "candle_engulfing",
        "candle_engulfing_trend_combo",
        "candle_harami", "candle_piercing_line", "candle_piercing_variant", "candle_inverted_hammer",
        "candle_piercing_variant_b_combo",
        "candle_belt_hold", "candle_three_white_soldiers", "candle_confirmed_hammer_a",
    ):
        df[c] = False
    if prices.empty or df.empty:
        return df
    _checks = [
        ("candle_doji", _enh_doji.check),
        ("candle_hammer", _enh_hammer.check),
        ("candle_marubozu", _enh_marubozu.check),
        ("candle_morning_star", _enh_mstar.check),
        ("candle_engulfing", _enh_engulf.check),
        ("candle_harami", _enh_harami.check),
        ("candle_piercing_line", _enh_piercing.check),
        ("candle_piercing_variant", _enh_piercing_variant.check),
        ("candle_inverted_hammer", _enh_inv_hammer.check),
        ("candle_belt_hold", _enh_belt_hold.check),
        ("candle_three_white_soldiers", _enh_three_white.check),
    ]
    _grouped: dict[str, pd.DataFrame] = {}
    for tkr, grp in prices.groupby("Ticker", sort=False):
        _grouped[str(tkr)] = grp.sort_values("Date")
    _cache: dict[tuple[str, str], dict[str, bool]] = {}
    for idx in df.index:
        raw_tkr = str(df.at[idx, ticker_col])
        tkr_ns = raw_tkr if raw_tkr.endswith(".NS") else (raw_tkr + ".NS") if add_ns_suffix else raw_tkr
        sd = str(df.at[idx, date_col])
        key = (tkr_ns, sd)
        if key not in _cache:
            g = _grouped.get(tkr_ns)
            if g is None:
                _cache[key] = {c: False for c, _ in _checks}
            else:
                sd_dt = pd.to_datetime(sd, errors="coerce")
                g_slice = g[g["Date"] <= sd_dt] if pd.notna(sd_dt) else g
                _cache[key] = {c: fn(g_slice, tkr_ns) for c, fn in _checks}
        for c, _ in _checks:
            df.at[idx, c] = _cache[key][c]
    if "pattern_family" in df.columns:
        df["candle_confirmed_hammer_a"] = (
            df["candle_hammer"].astype(bool)
            & df["pattern_family"].astype(str).str.upper().eq("A")
        )
        df["candle_engulfing_trend_combo"] = (
            df["candle_engulfing"].astype(bool)
            & df["pattern_family"].astype(str).str.upper().isin(_ENGULFING_POSITIVE_FAMILIES)
        )
        df["candle_piercing_variant_b_combo"] = (
            df["candle_piercing_variant"].astype(bool)
            & df["pattern_family"].astype(str).str.upper().isin(_PIERCING_VARIANT_POSITIVE_FAMILIES)
        )
    return df


ROOT = Path(__file__).resolve().parents[2]
TRIGGERS_DIR = ROOT / "stock_triggers"
LOGO_SVG = str(Path(__file__).resolve().parent / "logo.svg")
SCRIPTS_DIR = TRIGGERS_DIR / "scripts"
DATA_DIR = TRIGGERS_DIR / "data"
SIGNALS_CSV = DATA_DIR / "signals_pattern_a.csv"
SIGNALS_ALL_PATTERNS_CSV = DATA_DIR / "signals_all_patterns.csv"
SELL_SIGNALS_CSV = DATA_DIR / "sell_signals_pattern_a.csv"
PORTFOLIO_CSV = DATA_DIR / "portfolio_positions.csv"
DUMMY_LAB_CSV = DATA_DIR / "backtesting_lab_positions.csv"
PRICES_CSV = DATA_DIR / "prices_eod.csv"
EXTERNAL_FACTORS_CSV = DATA_DIR / "external_factors.csv"
TICKER_SECTOR_MAP_CSV = DATA_DIR / "ticker_sector_map.csv"
STOCK_SCORES_CSV = DATA_DIR / "stock_scores.csv"
CANDLE_WEIGHTS_JSON = DATA_DIR / "candle_weights.json"
PATTERN_WEIGHTS_JSON = DATA_DIR / "pattern_weights.json"
WHATS_NEW_JSON = DATA_DIR / "whats_new.json"
BENCHMARK_TICKERS = {"^NSEI"}
TOMORROW_SCORE_METHODS = {
    "Heuristic score": {
        "column": "ui_score",
        "label": "Heuristic score",
        "short_label": "Score",
        "higher_is_better": True,
        "filter_label": "Minimum heuristic score",
        "default_filter": 75,
        "display_scale": 1.0,
        "display_suffix": "",
    },
    "Reliability score": {
        "column": "signal_reliability_score",
        "label": "Reliability score",
        "short_label": "Reliability",
        "higher_is_better": True,
        "filter_label": "Minimum reliability score",
        "default_filter": 70,
        "display_scale": 1.0,
        "display_suffix": "",
    },
    "Stop risk": {
        "column": "signal_stop_risk",
        "label": "Stop risk",
        "short_label": "Stop risk",
        "higher_is_better": False,
        "filter_label": "Maximum stop risk %",
        "default_filter": 40,
        "display_scale": 100.0,
        "display_suffix": "%",
    },
}


def _is_benchmark_ticker(ticker: str) -> bool:
    return str(ticker).strip().upper() in BENCHMARK_TICKERS


def _exclude_benchmark_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Ticker" not in df.columns:
        return df.copy()
    return df[~df["Ticker"].astype(str).map(_is_benchmark_ticker)].copy()


def _select_benchmark_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Ticker" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[df["Ticker"].astype(str).map(_is_benchmark_ticker)].copy()


CANDLE_WEIGHT_KEYS = (
    "doji",
    "hammer",
    "marubozu",
    "confirmed_hammer_a",
    "morning_star",
    "engulfing",
    "engulfing_trend_combo",
    "harami",
    "piercing_line",
    "piercing_variant",
    "piercing_variant_b_combo",
    "inverted_hammer",
    "belt_hold",
    "three_white_soldiers",
)

_WEIGHT_KEY_TO_CANDLE_COL = {
    "doji": "candle_doji",
    "hammer": "candle_hammer",
    "marubozu": "candle_marubozu",
    "confirmed_hammer_a": "candle_confirmed_hammer_a",
    "morning_star": "candle_morning_star",
    "engulfing": "candle_engulfing",
    "engulfing_trend_combo": "candle_engulfing_trend_combo",
    "harami": "candle_harami",
    "piercing_line": "candle_piercing_line",
    "piercing_variant": "candle_piercing_variant",
    "piercing_variant_b_combo": "candle_piercing_variant_b_combo",
    "inverted_hammer": "candle_inverted_hammer",
    "belt_hold": "candle_belt_hold",
    "three_white_soldiers": "candle_three_white_soldiers",
}


def _default_candle_weight_map() -> dict[str, float]:
    return {key: 0.0 for key in CANDLE_WEIGHT_KEYS}


def _load_candle_weights_payload() -> dict:
    try:
        import json
        with open(CANDLE_WEIGHTS_JSON) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return {}


def _flatten_candle_weights_payload(payload: dict | None) -> dict[str, float]:
    defaults = _default_candle_weight_map()
    if not isinstance(payload, dict):
        return defaults
    global_block = payload.get("global") if isinstance(payload.get("global"), dict) else None
    for key in defaults:
        value = payload.get(key, defaults[key])
        if isinstance(global_block, dict) and key in global_block:
            value = global_block.get(key, value)
        try:
            defaults[key] = float(value)
        except (TypeError, ValueError):
            defaults[key] = 0.0
    return defaults


def _load_candle_weights() -> dict[str, float]:
    """Load the global candle enhancer summary from JSON."""
    return _flatten_candle_weights_payload(_load_candle_weights_payload())


def _load_lab_default_candle_weights() -> dict[str, float]:
    return _load_candle_weights()


def _compute_family_learned_candle_bonus(
    df: pd.DataFrame,
    payload: dict | None,
) -> pd.Series:
    bonus = pd.Series(0.0, index=df.index, dtype=float)
    if df.empty or not isinstance(payload, dict):
        return bonus
    families = payload.get("families") if isinstance(payload.get("families"), dict) else {}
    global_weights = _flatten_candle_weights_payload(payload)
    if "pattern_family" in df.columns:
        family_series = df["pattern_family"].astype(str).str.strip().str.upper()
    else:
        family_series = pd.Series("", index=df.index, dtype=object)

    for weight_key, candle_col in _WEIGHT_KEY_TO_CANDLE_COL.items():
        if candle_col not in df.columns:
            continue
        active_mask = df[candle_col].fillna(False).astype(bool)
        if not active_mask.any():
            continue
        row_weights = family_series.map(
            lambda fam: float(
                families.get(fam, {}).get("weights", {}).get(weight_key, global_weights.get(weight_key, 0.0))
            )
        ).astype(float)
        bonus.loc[active_mask] += row_weights.loc[active_mask]
    return bonus


def _load_pattern_weights() -> dict[str, float]:
    _defaults = {key: 0.0 for key in ("A", "B", "C", "D", "E", "F", "G")}
    try:
        import json
        with open(PATTERN_WEIGHTS_JSON) as f:
            data = json.load(f)
        return {key: float(data.get(key, 0.0)) for key in _defaults}
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return _defaults


def _load_pattern_weights_payload() -> dict:
    try:
        import json
        with open(PATTERN_WEIGHTS_JSON) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return {}


@st.cache_data(show_spinner=False, ttl=120)
def _load_whats_new_entries(limit: int = 3) -> list[dict[str, str]]:
    try:
        import json

        with open(WHATS_NEW_JSON) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return []

    if isinstance(data, dict):
        raw_entries = data.get("entries", [])
    elif isinstance(data, list):
        raw_entries = data
    else:
        raw_entries = []

    entries: list[dict[str, str]] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", "")).strip()
        if not title:
            continue
        entries.append(
            {
                "title": title,
                "tag": str(raw.get("tag", "Update")).strip() or "Update",
                "date": str(raw.get("date", "")).strip(),
                "summary": str(raw.get("summary", "")).strip(),
                "details": str(raw.get("details", "")).strip(),
                "impact": str(raw.get("impact", "")).strip(),
            }
        )
        if len(entries) >= max(1, int(limit)):
            break
    return entries


def render_whats_new_panel(*, context_label: str) -> None:
    entries = _load_whats_new_entries(limit=3)
    if not entries:
        return

    latest = entries[0]
    secondary_entries = entries[1:3]
    ordinals = ("Latest", "2nd latest", "3rd latest")

    def _esc(value: str) -> str:
        return html.escape(str(value or ""), quote=True)

    latest_meta = " · ".join(part for part in (_esc(latest.get("tag", "")), _esc(latest.get("date", ""))) if part)
    latest_impact = _esc(latest.get("impact", ""))
    secondary_blocks: list[str] = []
    for idx, entry in enumerate(secondary_entries, start=1):
        ordinal = ordinals[idx] if idx < len(ordinals) else f"Update {idx + 1}"
        meta = " · ".join(part for part in (_esc(entry.get("tag", "")), _esc(entry.get("date", ""))) if part)
        secondary_blocks.append(
            "<article class='whats-new-mini'>"
            "<div class='whats-new-mini-top'>"
            f"<div class='whats-new-mini-kicker'>{ordinal}</div>"
            f"<div class='whats-new-mini-meta'>- {meta}</div>"
            "</div>"
            f"<div class='whats-new-mini-title'>{_esc(entry.get('title', ''))}</div>"
            f"<div class='whats-new-mini-text'>{_esc(entry.get('summary', ''))}</div>"
            "</article>"
        )

    panel_html = (
        "<style>"
        ".whats-new-wrap {"
        "  border:1px solid rgba(251,191,36,0.45); border-radius:22px;"
        "  background:linear-gradient(135deg,#fff7ed 0%,#fefce8 26%,#ecfeff 100%);"
        "  padding:1rem; margin:0.2rem 0 1rem 0;"
        "  box-shadow:0 14px 34px rgba(249,115,22,0.16), 0 8px 22px rgba(14,165,233,0.10);"
        "  overflow:hidden; position:relative;"
        "}"
        ".whats-new-wrap::before {"
        "  content:''; position:absolute; inset:-20% auto auto -8%; width:220px; height:220px;"
        "  background:radial-gradient(circle, rgba(251,191,36,0.26) 0%, rgba(251,191,36,0) 70%);"
        "  pointer-events:none;"
        "}"
        ".whats-new-wrap::after {"
        "  content:''; position:absolute; inset:auto -6% -36% auto; width:240px; height:240px;"
        "  background:radial-gradient(circle, rgba(34,211,238,0.18) 0%, rgba(34,211,238,0) 72%);"
        "  pointer-events:none;"
        "}"
        ".whats-new-head {"
        "  position:relative; z-index:1; display:flex; align-items:flex-end; justify-content:space-between;"
        "  gap:0.9rem; flex-wrap:wrap; margin-bottom:0.85rem;"
        "}"
        ".whats-new-badge {"
        "  display:inline-flex; align-items:center; gap:0.35rem; padding:0.3rem 0.65rem;"
        "  border-radius:999px; background:#0f172a; color:#f8fafc; font-size:0.72rem; font-weight:800; letter-spacing:0.05em; text-transform:uppercase;"
        "}"
        ".whats-new-title-main { font-size:1.35rem; font-weight:900; color:#7c2d12; line-height:1.1; margin-top:0.5rem; }"
        ".whats-new-sub { font-size:0.88rem; color:#7c2d12; margin-top:0.22rem; max-width:820px; }"
        ".whats-new-grid { position:relative; z-index:1; display:grid; grid-template-columns:minmax(0,1.45fr) minmax(280px,0.95fr); gap:0.8rem; align-items:stretch; }"
        ".whats-new-hero {"
        "  border:1px solid rgba(249,115,22,0.28); border-radius:18px; padding:1rem 1.05rem;"
        "  background:linear-gradient(135deg,#7c2d12 0%,#c2410c 55%,#0f766e 100%); color:#fff7ed;"
        "  box-shadow:0 14px 30px rgba(124,45,18,0.18);"
        "}"
        ".whats-new-hero-top { display:flex; align-items:baseline; gap:0.4rem; flex-wrap:wrap; }"
        ".whats-new-hero-kicker { font-size:0.72rem; font-weight:800; letter-spacing:0.05em; text-transform:uppercase; color:#fde68a; }"
        ".whats-new-hero-meta { font-size:0.76rem; color:#fed7aa; }"
        ".whats-new-hero-title { font-size:1.3rem; font-weight:900; line-height:1.15; margin-top:0.55rem; color:#fff7ed; }"
        ".whats-new-hero-summary { font-size:0.96rem; line-height:1.48; margin-top:0.6rem; color:#fff7ed; }"
        ".whats-new-hero-detail { font-size:0.86rem; line-height:1.52; margin-top:0.6rem; color:#ffedd5; }"
        ".whats-new-hero-impact { margin-top:0.8rem; padding:0.65rem 0.75rem; border-radius:14px; background:rgba(255,247,237,0.16); border:1px solid rgba(255,255,255,0.18); }"
        ".whats-new-hero-impact-label { font-size:0.7rem; font-weight:800; letter-spacing:0.05em; text-transform:uppercase; color:#fde68a; margin-bottom:0.22rem; }"
        ".whats-new-hero-impact-text { font-size:0.82rem; line-height:1.45; color:#fff7ed; }"
        ".whats-new-side { display:grid; grid-template-rows:1fr 1fr; gap:0.7rem; }"
        ".whats-new-mini { border:1px solid rgba(14,165,233,0.18); border-radius:16px; padding:0.85rem 0.9rem; background:rgba(255,255,255,0.82); backdrop-filter:blur(4px); }"
        ".whats-new-mini-top { display:flex; align-items:baseline; gap:0.38rem; flex-wrap:wrap; }"
        ".whats-new-mini-kicker { font-size:0.7rem; font-weight:800; letter-spacing:0.05em; text-transform:uppercase; color:#0369a1; }"
        ".whats-new-mini-meta { font-size:0.72rem; color:#64748b; }"
        ".whats-new-mini-title { font-size:0.98rem; font-weight:800; color:#0f172a; line-height:1.25; margin-top:0.45rem; }"
        ".whats-new-mini-text { font-size:0.82rem; color:#334155; line-height:1.45; margin-top:0.45rem; }"
        ".whats-new-mini-detail { color:#475569; }"
        "@media (max-width: 900px) {"
        "  .whats-new-grid { grid-template-columns:1fr; }"
        "  .whats-new-side { grid-template-rows:none; grid-template-columns:1fr; }"
        "  .whats-new-title-main { font-size:1.15rem; }"
        "  .whats-new-hero-title { font-size:1.1rem; }"
        "}"
        "</style>"
        "<div class='whats-new-wrap'>"
        "<div class='whats-new-head'>"
        "<div>"
        "<div class='whats-new-badge'>What's New Now</div>"
        "</div>"
        "</div>"
        "<div class='whats-new-grid'>"
        f"<article class='whats-new-hero'><div class='whats-new-hero-top'><div class='whats-new-hero-kicker'>{ordinals[0]}</div><div class='whats-new-hero-meta'>- {latest_meta}</div></div><div class='whats-new-hero-title'>{_esc(latest.get('title', ''))}</div><div class='whats-new-hero-summary'>{_esc(latest.get('summary', ''))}</div><div class='whats-new-hero-detail'>{_esc(latest.get('details', ''))}</div><div class='whats-new-hero-impact'><div class='whats-new-hero-impact-label'>Why this matters</div><div class='whats-new-hero-impact-text'>{latest_impact}</div></div></article>"
        f"<div class='whats-new-side'>{''.join(secondary_blocks)}</div>"
        "</div>"
        "</div>"
    )

    st.markdown(panel_html, unsafe_allow_html=True)


def _render_equity_curve(trades_df: pd.DataFrame) -> None:
    """Step 2: Cumulative return equity curve inside a Lab summary expander."""
    import plotly.graph_objects as go

    if trades_df is None or trades_df.empty:
        return
    if "return_pct" not in trades_df.columns or "signal_date" not in trades_df.columns:
        return

    t = trades_df.copy()
    t["signal_date"] = pd.to_datetime(t["signal_date"], errors="coerce")
    t = t.dropna(subset=["signal_date", "return_pct"]).sort_values("signal_date")
    t["return_pct"] = pd.to_numeric(t["return_pct"], errors="coerce").fillna(0.0)
    if t.empty:
        return

    t["cum_return"] = t["return_pct"].cumsum()
    best_idx = int(t["cum_return"].idxmax())
    worst_idx = int(t["cum_return"].idxmin())

    pos_mask = t["cum_return"] >= 0
    neg_mask = t["cum_return"] < 0

    fig = go.Figure()
    # Positive fill
    fig.add_trace(go.Scatter(
        x=t["signal_date"], y=t["cum_return"].where(pos_mask),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.18)",
        line=dict(color="#22c55e", width=2),
        name="Gain",
    ))
    # Negative fill
    fig.add_trace(go.Scatter(
        x=t["signal_date"], y=t["cum_return"].where(neg_mask),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.18)",
        line=dict(color="#ef4444", width=2),
        name="Loss",
    ))
    # Zero baseline
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#64748b")
    # Best / worst annotations
    fig.add_trace(go.Scatter(
        x=[t.loc[best_idx, "signal_date"]], y=[t.loc[best_idx, "cum_return"]],
        mode="markers+text",
        marker=dict(color="#22c55e", size=8),
        text=[f"Peak {t.loc[best_idx, 'cum_return']:.1f}%"],
        textposition="top center",
        textfont=dict(size=9, color="#22c55e"),
        name="Peak",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[t.loc[worst_idx, "signal_date"]], y=[t.loc[worst_idx, "cum_return"]],
        mode="markers+text",
        marker=dict(color="#ef4444", size=8),
        text=[f"Trough {t.loc[worst_idx, 'cum_return']:.1f}%"],
        textposition="bottom center",
        textfont=dict(size=9, color="#ef4444"),
        name="Trough",
        showlegend=False,
    ))
    fig.update_layout(
        height=220, margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", title="Cumulative return %"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    with st.expander("📈 Equity Curve", expanded=True):
        st.plotly_chart(fig, use_container_width=True)


def _render_risk_return_scatter(trades_df: pd.DataFrame) -> None:
    """Step 3: stop_pct vs return_pct scatter coloured by outcome."""
    import plotly.graph_objects as go

    if trades_df is None or trades_df.empty:
        return
    needed = {"stop_pct", "return_pct", "ticker"}
    if not needed.issubset(trades_df.columns):
        return

    t = trades_df.copy()
    for col in ("stop_pct", "return_pct"):
        t[col] = pd.to_numeric(t[col], errors="coerce")
    t = t.dropna(subset=["stop_pct", "return_pct"])
    if t.empty:
        return

    status_col = "status" if "status" in t.columns else None
    signal_score_col = "signal_score" if "signal_score" in t.columns else None

    color_map = {
        "Target Hit ✅": "#22c55e",
        "Stop Hit 🛑": "#ef4444",
        "Holding": "#94a3b8",
    }

    def _status_color(s: str) -> str:
        return color_map.get(str(s), "#94a3b8")

    colors = (
        t[status_col].map(_status_color).tolist()
        if status_col else ["#94a3b8"] * len(t)
    )

    sizes = [8] * len(t)
    if signal_score_col:
        raw = pd.to_numeric(t[signal_score_col], errors="coerce").fillna(60)
        sizes = ((raw - raw.min()) / max(raw.max() - raw.min(), 1) * 8 + 6).tolist()

    hover_parts = ["<b>%{customdata[0]}</b>"]
    if "signal_date" in t.columns:
        hover_parts.append("Date: %{customdata[1]}")
    if "pattern_family" in t.columns:
        hover_parts.append("Pattern: %{customdata[2]}")
    hover_parts += ["Risk: %{x:.1f}%", "Return: %{y:.1f}%"]
    if signal_score_col:
        hover_parts.append("Score: %{customdata[3]:.0f}")

    custom_cols = [
        t["ticker"].astype(str),
        t.get("signal_date", pd.Series([""] * len(t))).astype(str),
        t.get("pattern_family", pd.Series([""] * len(t))).astype(str),
        pd.to_numeric(t.get(signal_score_col, pd.Series([0] * len(t))), errors="coerce").fillna(0),
    ]
    customdata = list(zip(*[c.tolist() for c in custom_cols]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t["stop_pct"], y=t["return_pct"],
        mode="markers",
        marker=dict(color=colors, size=sizes, opacity=0.75, line=dict(width=0.5, color="#1e293b")),
        customdata=customdata,
        hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
        name="Trades",
    ))
    fig.add_hline(y=0, line_width=0.8, line_dash="dot", line_color="#64748b")
    fig.add_vline(x=0, line_width=0.8, line_dash="dot", line_color="#64748b")
    # Legend proxies
    for label, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, size=8),
            name=label,
        ))
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", title="Stop risk %"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False, title="Return %"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    with st.expander("📉 Risk vs Return", expanded=False):
        st.plotly_chart(fig, use_container_width=True)


def _render_pattern_performance_chart(pattern_weights_payload: dict) -> None:
    """Step 4: grouped bar chart A–G; clicking sets lab_family_filter."""
    import plotly.graph_objects as go

    if not pattern_weights_payload:
        return
    details = pattern_weights_payload.get("details", {})
    if not isinstance(details, dict) or not details:
        return

    baseline = float(pattern_weights_payload.get("baseline_win_rate", 46.0))
    family_colors = {
        "A": "#3b82f6", "B": "#8b5cf6", "C": "#f59e0b",
        "D": "#ef4444", "E": "#10b981", "F": "#06b6d4", "G": "#f97316",
    }
    families: list[str] = []
    win_rates: list[float] = []
    edges: list[float] = []
    weights: list[float] = []
    for fam in ("A", "B", "C", "D", "E", "F", "G"):
        d = details.get(fam, {})
        if not d:
            continue
        families.append(fam)
        win_rates.append(float(d.get("win_rate_with", 0.0) or 0.0))
        edges.append(float(d.get("edge_pp", 0.0) or 0.0))
        weights.append(float(pattern_weights_payload.get(fam, 0.0) or 0.0))

    if not families:
        return

    bar_colors = [family_colors.get(f, "#64748b") for f in families]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=families, y=win_rates, name="Win rate %",
        marker_color=bar_colors, opacity=0.85,
        customdata=families,
        hovertemplate="Pattern %{x}<br>Win rate: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=families, y=edges, name="Edge pp",
        marker_color=bar_colors, opacity=0.5,
        customdata=families,
        hovertemplate="Pattern %{x}<br>Edge: %{y:.1f} pp<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=families, y=weights, name="Weight /30",
        marker_color=bar_colors, opacity=0.35,
        customdata=families,
        hovertemplate="Pattern %{x}<br>Weight: %{y:.1f}/30<extra></extra>",
    ))
    fig.add_hline(y=baseline, line_width=1.2, line_dash="dash", line_color="#f472b6",
                  annotation_text=f"Baseline {baseline:.0f}%",
                  annotation_font_color="#f472b6", annotation_font_size=9,
                  annotation_position="bottom right")

    fig.update_layout(
        barmode="group",
        height=240, margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(showgrid=False, title="Pattern family"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )

    sel = st.plotly_chart(
        fig, use_container_width=True, on_select="rerun",
        key="pattern_perf_chart_sel",
    )
    # Handle click → lab filter
    if sel and hasattr(sel, "selection") and sel.selection:
        pts = getattr(sel.selection, "points", [])
        if pts:
            clicked_fam = str(pts[0].get("x", "")).strip().upper()
            if clicked_fam in ("A", "B", "C", "D", "E", "F", "G"):
                st.session_state["lab_family_filter"] = [clicked_fam]
                if st.session_state.get("mode") != "Backtest Lab":
                    st.session_state["mode"] = "Backtest Lab"
                    st.session_state["_nav_skip_sync"] = True
                st.rerun()

    st.caption("Click a bar to filter the Backtesting Lab to that pattern family. Dashed line = baseline win rate.")


def _render_score_distribution(signals_df: pd.DataFrame, min_score: float = 75.0) -> None:
    """Step 5: histogram of signal_score with min_score and median lines."""
    import plotly.graph_objects as go

    if signals_df is None or signals_df.empty or "signal_score" not in signals_df.columns:
        return

    scores = pd.to_numeric(signals_df["signal_score"], errors="coerce").dropna()
    if len(scores) < 5:
        return

    bins = list(range(0, 106, 5))
    fig = go.Figure()
    # Two traces to colour below/above min_score
    below = scores[scores < min_score]
    above = scores[scores >= min_score]
    if not below.empty:
        fig.add_trace(go.Histogram(
            x=below, xbins=dict(start=0, end=100, size=5),
            name=f"< {min_score:.0f}", marker_color="#475569", opacity=0.7,
        ))
    if not above.empty:
        fig.add_trace(go.Histogram(
            x=above, xbins=dict(start=0, end=100, size=5),
            name=f"≥ {min_score:.0f}", marker_color="#38bdf8", opacity=0.8,
        ))
    # Min score line
    fig.add_vline(x=min_score, line_width=1.5, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"Min {min_score:.0f}", annotation_font_size=9,
                  annotation_font_color="#ef4444", annotation_position="top right")
    # Median line
    med = float(scores.median())
    fig.add_vline(x=med, line_width=1.2, line_dash="dot", line_color="#3b82f6",
                  annotation_text=f"Median {med:.0f}", annotation_font_size=9,
                  annotation_font_color="#3b82f6", annotation_position="top left")

    fig.update_layout(
        barmode="overlay", height=200,
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(range=[0, 100], showgrid=False, title="Signal score"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", title="Count"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_candle_enhancer_chart(candle_weights: dict) -> None:
    """Step 6: horizontal bar chart of candle enhancer weights."""
    import plotly.graph_objects as go

    if not candle_weights:
        return
    # Filter out metadata keys and zero/absent entries
    _skip = {"baseline_win_rate", "computed_at", "total_signals", "details", "families", "model", "global", "comparison_details", "comparisons", "outcomes"}
    items = {k: float(v) for k, v in candle_weights.items() if k not in _skip and isinstance(v, (int, float))}
    if not items:
        return

    sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
    names = [k.replace("_", " ").title() for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["#38bdf8" if v > 0 else "#f87171" if v < 0 else "#475569" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=0.8, line_dash="dot", line_color="#64748b")
    fig.update_layout(
        height=max(200, len(names) * 28),
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", title="Weight bonus"),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pattern_bonus_expander() -> None:
    payload = _load_pattern_weights_payload()
    with st.expander("Learned Pattern Weights", expanded=False):
        render_caption_with_help(
            "How to read the learned pattern-weight table.",
            "learned_pattern_weights",
            key="pattern_weights_intro_help",
        )
        if not payload:
            st.caption("No historical pattern-weight file found yet.")
            return

        summary_parts: list[str] = []
        computed_at = str(payload.get("computed_at", "")).strip()
        total_signals = payload.get("total_signals")
        baseline_win_rate = payload.get("baseline_win_rate")
        if computed_at:
            summary_parts.append(f"Updated: {computed_at}")
        if total_signals is not None:
            summary_parts.append(f"Signals analyzed: {int(total_signals)}")
        if baseline_win_rate is not None:
            summary_parts.append(f"Baseline win rate: {float(baseline_win_rate):.1f}%")
        if summary_parts:
            st.caption(" | ".join(summary_parts))

        # Performance chart (Step 4) above the raw table
        _render_pattern_performance_chart(payload)

        rows: list[dict[str, object]] = []
        details = payload.get("details", {}) if isinstance(payload.get("details"), dict) else {}
        pattern_labels = {family: label for family, label in _LAB_PATTERN_OPTIONS}
        families_in_payload = {
            str(key).strip().upper()
            for key in payload.keys()
            if str(key).strip().upper() in pattern_labels
        }
        families_in_details = {
            str(key).strip().upper()
            for key in details.keys()
            if str(key).strip().upper() in pattern_labels
        }
        families_to_show = sorted(
            set(pattern_labels).union(families_in_payload).union(families_in_details)
        )

        for family in families_to_show:
            label = pattern_labels.get(family, family)
            detail = details.get(family, {}) if isinstance(details, dict) else {}
            rows.append(
                {
                    "Pattern": f"{family} · {label}",
                    "Score /100": float(detail.get("score_pattern", 0.0) or 0.0),
                    "Weight /30": float(payload.get(family, 0.0) or 0.0),
                    "Count": int(detail.get("count", 0) or 0),
                    "Win %": float(detail.get("win_rate_with", 0.0) or 0.0),
                    "Loss %": float(detail.get("loss_rate_with", 0.0) or 0.0),
                    "Edge pp": float(detail.get("edge_pp", 0.0) or 0.0),
                }
            )

        stats_df = pd.DataFrame(rows)
        render_table(
            stats_df,
            height=min(420, max(260, 40 * (len(stats_df) + 1))),
            column_help=table_help_map("pattern_weights", stats_df.columns),
            table_help_title="Learned Pattern Weights",
            table_help_key_prefix="pattern_weights_cols",
        )


def render_candle_enhancer_expander() -> None:
    """Step 6: wrap the candle enhancer bar chart in its own expander."""
    cw = _load_candle_weights()
    if not cw:
        return
    with st.expander("Candle Enhancer Weights", expanded=False):
        render_caption_with_help(
            "Global summary of the learned candle weights. Backtest Lab can additionally apply family-specific signed weights from the same artifact.",
            "enhancer_doji",
            key="candle_enh_chart_help",
        )
        _render_candle_enhancer_chart(cw)


_CANDLE_PATTERN_HELP = {
    "doji": "Small body with a long lower wick. Often shows sellers pushed price down but buyers pulled it back up.",
    "hammer": "Small body near the top with a long lower shadow. Often signals bullish rejection after weakness.",
    "marubozu": "Strong green candle with very small wicks. Best when it appears from tight consolidation with volume or at a breakout or pullback-support location.",
    "confirmed_hammer_a": "A confirmed hammer that only applies when the signal itself is Pattern A, so the reversal candle is aligned with the breakout family.",
    "morning_star": "Three-candle bullish reversal: a strong red candle, a pause candle, then a strong green recovery.",
    "engulfing": "Bullish two-candle reversal where the green candle fully covers the prior red candle body.",
    "engulfing_trend_combo": "A bullish engulfing that only earns the combo bonus when the signal family is A, C, or G, which are the families where engulfing currently shows positive edge.",
    "harami": "Bullish two-candle setup where a smaller candle sits inside the prior large red candle body.",
    "piercing_line": "Bullish two-candle reversal where the second candle pushes well back into the prior red candle.",
    "piercing_variant": "A practical piercing-line style recovery without needing a perfect textbook gap.",
    "piercing_variant_b_combo": "A practical piercing-line recovery that only earns the combo bonus when the signal family is B, where the pattern currently shows positive edge.",
    "inverted_hammer": "Small body with a long upper wick. Can signal buyers are starting to test control.",
    "belt_hold": "Strong green candle that opens near the low and closes near the high with little lower wick.",
    "three_white_soldiers": "Three strong green candles in a row, each closing higher. Often shows steady bullish control.",
}


def _candle_help(pattern_key: str) -> str:
    return _CANDLE_PATTERN_HELP.get(pattern_key, "")


CANDIDATE_STOCKS_CSV = DATA_DIR / "candidate_stocks.csv"
STOCK_UNIVERSE_DIR = DATA_DIR / "stock_universe"
SECRETS_FILE = ROOT / "secrets.yml"
IS_STREAMLIT_CLOUD = bool(os.getenv("STREAMLIT_SHARING_MODE")) or bool(os.getenv("STREAMLIT_CLOUD"))
PRODUCTION_APP_URL = "https://stock-operator-roy.streamlit.app/"


@st.cache_data(show_spinner=False, ttl=300)
def _load_build_marker() -> str:
    env_sha = str(
        os.getenv("STREAMLIT_GIT_COMMIT")
        or os.getenv("GIT_COMMIT")
        or os.getenv("COMMIT_SHA")
        or ""
    ).strip()
    env_branch = str(os.getenv("STREAMLIT_GIT_BRANCH") or os.getenv("GIT_BRANCH") or "").strip()
    if env_sha:
        short_sha = env_sha[:7]
        return f"Build {short_sha}" + (f" on {env_branch}" if env_branch else "")

    try:
        git_sha = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        git_branch = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        if git_sha:
            return f"Build {git_sha}" + (f" on {git_branch}" if git_branch else "")
    except Exception:
        pass

    return "Build unknown"

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


st.set_page_config(page_title="Stock Operator", layout="wide")

_theme_query_value = str(st.query_params.get("theme", "")).strip().lower()
_theme_query_enabled = _theme_query_value in {"night", "dark", "true", "1"}

if "ui_night_theme" not in st.session_state:
    st.session_state["ui_night_theme"] = _theme_query_enabled

_is_night_theme = bool(st.session_state.get("ui_night_theme", False))

_theme_tokens = {
    "nav_bg": "#020617" if _is_night_theme else "#1e293b",
    "nav_text": "#cbd5e1" if _is_night_theme else "#94a3b8",
    "nav_active_text": "#f8fafc",
    "nav_active_bg": "rgba(56, 189, 248, 0.22)" if _is_night_theme else "rgba(59,130,246,0.25)",
    "nav_hover_text": "#e2e8f0",
    "nav_hover_bg": "rgba(148, 163, 184, 0.12)" if _is_night_theme else "rgba(255,255,255,0.06)",
    "app_bg": "radial-gradient(circle at 15% 0%, #172033 0%, #0d1526 38%, #070b14 100%)" if _is_night_theme else "radial-gradient(circle at 15% 0%, #fff9ed 0%, #f8fbff 40%, #f4f8fb 100%)",
    "surface_bg": "#0f172a" if _is_night_theme else "#ffffff",
    "surface_alt": "#111c31" if _is_night_theme else "#f8fafc",
    "surface_soft": "#13203a" if _is_night_theme else "#f8fbff",
    "border": "#2a3b57" if _is_night_theme else "#dbe4ef",
    "border_soft": "#334155" if _is_night_theme else "#e5e7eb",
    "text": "#e5eefc" if _is_night_theme else "#0f172a",
    "text_muted": "#9fb0c8" if _is_night_theme else "#4b5563",
    "text_soft": "#94a3b8" if _is_night_theme else "#64748b",
    "heading": "#f8fafc" if _is_night_theme else "#0f172a",
    "hero_bg": "linear-gradient(120deg, #172033 0%, #10223d 100%)" if _is_night_theme else "linear-gradient(120deg, #fff7ed 0%, #ecfeff 100%)",
    "hero_border": "#314158" if _is_night_theme else "#fed7aa",
    "hero_title": "#fbbf24" if _is_night_theme else "#7c2d12",
    "hero_sub": "#cbd5e1" if _is_night_theme else "#334155",
    "widget_bg": "#0b1220" if _is_night_theme else "#ffffff",
    "widget_bg_alt": "#13203a" if _is_night_theme else "#f8fafc",
    "widget_border": "#314158" if _is_night_theme else "#cbd5e1",
    "table_header_bg": "#13203a" if _is_night_theme else "#f8fafc",
    "table_row_hover": "#17243b" if _is_night_theme else "#f8fbff",
    "shadow": "0 10px 26px rgba(2, 6, 23, 0.35)" if _is_night_theme else "0 10px 26px rgba(15, 23, 42, 0.06)",
    "panel_shadow": "0 4px 16px rgba(2, 6, 23, 0.28)" if _is_night_theme else "0 4px 16px rgba(15, 23, 42, 0.04)",
    "tone_pos_bg": "linear-gradient(180deg, #10261b 0%, #0d1f16 100%)" if _is_night_theme else "linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%)",
    "tone_pos_border": "#1f7a47" if _is_night_theme else "#bbf7d0",
    "tone_warn_bg": "linear-gradient(180deg, #2b2110 0%, #21190d 100%)" if _is_night_theme else "linear-gradient(180deg, #ffffff 0%, #fffbeb 100%)",
    "tone_warn_border": "#9a6b11" if _is_night_theme else "#fde68a",
    "tone_neg_bg": "linear-gradient(180deg, #2d1518 0%, #241013 100%)" if _is_night_theme else "linear-gradient(180deg, #ffffff 0%, #fef2f2 100%)",
    "tone_neg_border": "#a33a46" if _is_night_theme else "#fecaca",
}

# ── Navigation bar ──
_nav_styles = {
    "nav": {
        "background-color": _theme_tokens["nav_bg"],
        "font-family": "'Space Grotesk', sans-serif",
        "justify-content": "left",
        "align-items": "center",
        "padding": "0.35rem 0.6rem",
        "box-shadow": "0 4px 20px rgba(15,23,42,0.25)",
        "height": "auto",
        "min-height": "2.875rem",
    },
    "ul": {
        "flex-wrap": "wrap",
        "row-gap": "0.2rem",
        "column-gap": "0.15rem",
        "padding": "0",
        "margin": "0",
        "align-items": "center",
    },
    "li": {
        "display": "flex",
        "align-items": "center",
    },
    "img": {
        "padding-right": "6px",
        "height": "22px",
    },
    "span": {
        "color": _theme_tokens["nav_text"],
        "font-weight": "600",
        "font-size": "0.74rem",
        "padding": "0.35rem 0.5rem",
        "border-radius": "10px",
        "line-height": "1.2",
        "white-space": "normal",
        "text-align": "center",
    },
    "active": {
        "color": _theme_tokens["nav_active_text"],
        "background-color": _theme_tokens["nav_active_bg"],
    },
    "hover": {
        "color": _theme_tokens["nav_hover_text"],
        "background-color": _theme_tokens["nav_hover_bg"],
    },
}

_nav_options = {
    "show_menu": False,
    "show_sidebar": False,
    "fix_shadow": True,
    "use_padding": True,
}

# Map navbar page names → internal mode names
_NAV_PAGES = ["Tomorrow's Picks", "Backtesting Lab", "Coverage", "Documentation"]
_NAV_TO_MODE = {
    "Tomorrow's Picks": "Tomorrow",
    "Backtesting Lab": "Backtest Lab",
    "Coverage": "Coverage",
    "Documentation": "Documentation",
}

# Resolve which page to pre-select based on current session mode
if "mode" not in st.session_state:
    st.session_state["mode"] = "Tomorrow"
if st.session_state.get("mode") not in set(_NAV_TO_MODE.values()):
    st.session_state["mode"] = "Tomorrow"

# Intercept ?help=<key> links emitted by the HTML help chips
handle_help_query_param()

# Force-sync min_score to current config default when the default changes.
# Streamlit restores browser-cached widget values on reconnect, so without this
# the slider would stay at the old default (90) even after a server restart.
_min_score_cfg = int(TOMORROW_SCORE_METHODS["Heuristic score"]["default_filter"])
if st.session_state.get("_min_score_cfg") != _min_score_cfg:
    st.session_state["min_score"] = _min_score_cfg
    st.session_state["_min_score_cfg"] = _min_score_cfg
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
        background-color: __NAV_BG__ !important;
        height: 2.875rem !important;
        z-index: 0 !important;
    }
    footer, #stDecoration { visibility: hidden !important; }
    div[class="stDeployButton"] { visibility: hidden !important; }
    div[class="stStatusWidget"] { visibility: hidden !important; }

    /* Float the hamburger menu above the navbar iframe so it's clickable */
    div[data-testid="stToolbarActions"] {
        position: fixed !important;
        top: 0.35rem !important;
        right: 0.75rem !important;
        z-index: 9999999 !important;
    }

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

    @media (max-width: 720px) {
        iframe[title="streamlit_navigation_bar.st_navbar"] {
            height: 7rem !important;
        }

        header[data-testid="stHeader"] {
            height: 7rem !important;
        }

        section.main {
            top: 7rem !important;
        }
    }
    </style>
    """.replace("__NAV_BG__", _theme_tokens["nav_bg"]),
    unsafe_allow_html=True,
)

# Sync navbar selection → session state mode
# Skip when mode was set programmatically (button nav) — navbar component lags
if st.session_state.pop("_nav_skip_sync", False):
    pass  # programmatic navigation — let mode stand as-is this render
elif _selected_page and _NAV_TO_MODE.get(_selected_page) != st.session_state["mode"]:
    st.session_state["mode"] = _NAV_TO_MODE[_selected_page]
    st.rerun()

_curr_mode = st.session_state["mode"]

_theme_css_vars = "\n".join([
    f"        --app-bg: {_theme_tokens['app_bg']};",
    f"        --surface-bg: {_theme_tokens['surface_bg']};",
    f"        --surface-alt: {_theme_tokens['surface_alt']};",
    f"        --surface-soft: {_theme_tokens['surface_soft']};",
    f"        --border-color: {_theme_tokens['border']};",
    f"        --border-soft: {_theme_tokens['border_soft']};",
    f"        --text-primary: {_theme_tokens['text']};",
    f"        --text-muted: {_theme_tokens['text_muted']};",
    f"        --text-soft: {_theme_tokens['text_soft']};",
    f"        --heading-color: {_theme_tokens['heading']};",
    f"        --hero-bg: {_theme_tokens['hero_bg']};",
    f"        --hero-border: {_theme_tokens['hero_border']};",
    f"        --hero-title: {_theme_tokens['hero_title']};",
    f"        --hero-sub: {_theme_tokens['hero_sub']};",
    f"        --widget-bg: {_theme_tokens['widget_bg']};",
    f"        --widget-bg-alt: {_theme_tokens['widget_bg_alt']};",
    f"        --widget-border: {_theme_tokens['widget_border']};",
    f"        --table-header-bg: {_theme_tokens['table_header_bg']};",
    f"        --table-row-hover: {_theme_tokens['table_row_hover']};",
    f"        --panel-shadow: {_theme_tokens['panel_shadow']};",
    f"        --page-shadow: {_theme_tokens['shadow']};",
    f"        --tone-pos-bg: {_theme_tokens['tone_pos_bg']};",
    f"        --tone-pos-border: {_theme_tokens['tone_pos_border']};",
    f"        --tone-warn-bg: {_theme_tokens['tone_warn_bg']};",
    f"        --tone-warn-border: {_theme_tokens['tone_warn_border']};",
    f"        --tone-neg-bg: {_theme_tokens['tone_neg_bg']};",
    f"        --tone-neg-border: {_theme_tokens['tone_neg_border']};",
])

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;600;700&display=swap');
    :root {
__THEME_VARS__
    }
    .block-container {padding-top: 0.3rem;}
    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        color: var(--text-primary);
    }
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.2px;
        color: var(--heading-color);
    }
    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.2px;
        font-size: 2.05rem;
        font-weight: 700;
        color: var(--heading-color);
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
        background: var(--app-bg);
        color: var(--text-primary);
    }
    [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"], section.main {
        background: transparent !important;
    }
    p, li, label, .stMarkdown, .stCaption, [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] {
        color: var(--text-primary);
    }
    .card {
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        background: var(--surface-alt);
        margin-bottom: 0.8rem;
        box-shadow: var(--panel-shadow);
    }
    .small-muted {color: var(--text-muted); font-size: 0.9rem;}
    .stat-card {
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 0.7rem 0.9rem;
        background: linear-gradient(180deg, var(--surface-bg) 0%, var(--surface-alt) 100%);
        min-height: 84px;
        box-shadow: var(--panel-shadow);
    }
    .stat-label {
        color: var(--text-soft);
        font-size: 0.82rem;
        margin-bottom: 0.2rem;
    }
    .stat-value {
        color: var(--heading-color);
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
        border: 1px solid var(--hero-border);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        background: var(--hero-bg);
        box-shadow: var(--page-shadow);
    }
    .hero-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--hero-title);
    }
    .hero-sub {
        color: var(--hero-sub);
        font-size: 0.9rem;
    }
    .action-item {
        border: 1px solid var(--border-color);
        border-radius: 12px;
        background: var(--surface-soft);
        padding: 0.65rem 0.8rem;
        margin-bottom: 0.55rem;
    }
    .action-title {
        font-size: 0.85rem;
        color: var(--text-soft);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .action-value {
        font-size: 1.1rem;
        color: var(--heading-color);
        font-weight: 700;
    }
    .flow-wrap {
        border: 1px solid var(--border-color);
        border-radius: 14px;
        background: var(--surface-bg);
        padding: 0.55rem 0.7rem;
        margin-bottom: 0.7rem;
        box-shadow: var(--panel-shadow);
    }
    .flow-title {
        font-size: 0.82rem;
        color: var(--text-soft);
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .flow-step {
        display: inline-block;
        border: 1px solid var(--widget-border);
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        color: var(--text-primary);
        padding: 0.16rem 0.52rem;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
        background: var(--surface-alt);
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
        border: 1px solid var(--border-color);
        border-radius: 12px;
        background: var(--surface-bg);
        box-shadow: var(--panel-shadow);
        overflow: hidden;
    }
    div[data-testid="stDataFrame"] [role="columnheader"] {
        background: var(--table-header-bg);
        color: var(--text-primary);
        font-weight: 700;
        border-bottom: 1px solid var(--border-color);
    }
    div[data-testid="stDataFrame"] [role="gridcell"] {
        border-bottom: 1px solid var(--border-color);
    }
    div[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] {
        background: var(--table-row-hover);
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div,
    [data-testid="stDateInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    textarea {
        background: var(--widget-bg) !important;
        color: var(--text-primary) !important;
        border-color: var(--widget-border) !important;
    }
    [data-baseweb="tag"] {
        background: var(--widget-bg-alt) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--widget-border) !important;
    }
    [data-testid="stCheckbox"] label, [data-testid="stToggle"] label {
        color: var(--text-primary) !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    .theme-toggle-note {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 0.15rem;
        margin-bottom: 0.2rem;
    }

    /* ── Button polish ───────────────────────────────────────── */
    div[data-testid="stButton"] > button {
        font-family: 'Manrope', sans-serif;
        font-weight: 600;
        border-radius: 10px;
        transition: background 0.15s ease, border-color 0.15s ease,
                    transform 0.12s ease, box-shadow 0.15s ease,
                    filter 0.15s ease;
    }
    /* Secondary = ghost: transparent until hovered */
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
        color: var(--text-primary) !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover,
    div[data-testid="stButton"] > button[kind="secondary"]:focus-visible {
        background: var(--surface-alt) !important;
        border-color: var(--border-soft) !important;
        transform: translateY(-1px);
    }
    /* Primary: keep filled sky-500, add lift + glow on hover */
    div[data-testid="stButton"] > button[kind="primary"]:hover,
    div[data-testid="stButton"] > button[kind="primary"]:focus-visible {
        filter: brightness(1.08);
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.3) !important;
    }
    div[data-testid="stButton"] > button:active {
        transform: translateY(0) !important;
        filter: none !important;
    }
    div[data-testid="stButton"] > button:disabled,
    div[data-testid="stButton"] > button[disabled] {
        opacity: 0.45 !important;
        pointer-events: none !important;
    }

    /* ── Expander: refined header ────────────────────────────── */
    /* Streamlit 1.55 renders expanders as native <details>/<summary> elements;
       the built-in disclosure triangle is removed by Streamlit itself. */
    [data-testid="stExpander"] details {
        border-color: var(--border-color) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: var(--heading-color) !important;
    }
    /* The markdown label inside summary renders as <p> */
    [data-testid="stExpander"] summary p {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        margin: 0 !important;
        color: var(--heading-color) !important;
    }
    /* CSS-only chevron — animates with the open/close state */
    [data-testid="stExpander"] summary::after {
        content: "";
        display: inline-block;
        flex-shrink: 0;
        margin-left: auto;
        width: 0.42rem;
        height: 0.42rem;
        border-right: 2px solid rgba(14, 165, 233, 0.8);
        border-bottom: 2px solid rgba(14, 165, 233, 0.8);
        transform: rotate(45deg);  /* ↓ pointing down = "expand" */
        transition: transform 0.22s ease;
        position: relative;
        top: -2px;
    }
    [data-testid="stExpander"] details[open] > summary::after {
        transform: rotate(225deg);  /* ↑ pointing up = "collapse" */
        top: 1px;
    }
    </style>
    """.replace("__THEME_VARS__", _theme_css_vars),
    unsafe_allow_html=True,
)

_theme_query_target = "night" if st.session_state.get("ui_night_theme") else "light"
if _theme_query_value != _theme_query_target:
    st.query_params["theme"] = _theme_query_target


def render_theme_toggle_control() -> None:
    st.toggle(
        "Night theme",
        key="ui_night_theme",
        help="Switch between the default light palette and a darker night palette.",
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


def _slug_filename_part(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "na"
    chars: list[str] = []
    last_was_sep = False
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
            last_was_sep = False
        elif ch in {" ", "-", "_", "/", ",", "."}:
            if not last_was_sep:
                chars.append("-")
                last_was_sep = True
    out = "".join(chars).strip("-")
    return out or "na"


def build_lab_export_filename(
    *,
    pattern_families: tuple[str, ...],
    status_filter: str,
    candle_filters: list[str],
    min_score: int,
    max_days_held: int | None,
    sort_by: str,
    sort_desc: bool,
    row_count: int,
) -> str:
    families_part = "-".join(pattern_families) if pattern_families else "all"
    candles_part = "all" if not candle_filters else "-".join(_slug_filename_part(x) for x in candle_filters)
    status_part = _slug_filename_part(status_filter)
    sort_part = _slug_filename_part(sort_by)
    sort_dir_part = "desc" if sort_desc else "asc"
    max_days_part = "all" if max_days_held is None else str(int(max_days_held))
    date_part = date.today().isoformat()
    return (
        f"backtesting_lab_{families_part}"
        f"_status-{status_part}"
        f"_candles-{candles_part}"
        f"_minscore-{int(min_score)}"
        f"_maxdays-{max_days_part}"
        f"_sort-{sort_part}-{sort_dir_part}"
        f"_rows-{int(row_count)}"
        f"_{date_part}.csv"
    )


def render_table(
    df: pd.DataFrame | pd.io.formats.style.Styler,
    *,
    height: int = 320,
    column_help: dict[str, str] | None = None,
    table_help_title: str | None = None,
    table_help_key_prefix: str | None = None,
) -> None:
    column_help = column_help or {}
    if column_help and table_help_title and table_help_key_prefix:
        render_table_help_glossary(
            table_help_title,
            column_help,
            key_prefix=table_help_key_prefix,
        )
    column_config = build_dataframe_column_config(column_help) if column_help else None
    if isinstance(df, pd.DataFrame):
        display = df.copy()
        float_cols = display.select_dtypes(include=["float64", "float32"]).columns.tolist()
        for c in float_cols:
            display[c] = display[c].round(2)
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            height=height,
            column_config=column_config,
        )
    else:
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            height=height,
            column_config=column_config,
        )


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
    return _apply_pattern_family_bonus(pd.read_csv(SIGNALS_CSV), _load_pattern_weights())


@st.cache_data(show_spinner=False, ttl=120)
def load_all_pattern_signals() -> pd.DataFrame:
    if not SIGNALS_ALL_PATTERNS_CSV.is_file():
        return pd.DataFrame()
    return _apply_pattern_family_bonus(pd.read_csv(SIGNALS_ALL_PATTERNS_CSV), _load_pattern_weights())


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


@st.cache_data(show_spinner=False, ttl=120)
def load_default_coverage_cache() -> dict:
    payload = _load_default_coverage_cache_if_valid(
        prices_path=PRICES_CSV,
        signals_path=SIGNALS_ALL_PATTERNS_CSV,
    )
    return payload if isinstance(payload, dict) else {}


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
    if "score_100" not in df.columns and "score" in df.columns:
        raw_score = pd.to_numeric(df["score"], errors="coerce")
        if raw_score.notna().any() and float(raw_score.max()) <= 4.0:
            df["score_100"] = ((raw_score.clip(lower=0.0, upper=4.0) / 4.0) * 100.0).round()
        else:
            df["score_100"] = raw_score.round()
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


def build_ticker_index_rs_table(
    prices_df: pd.DataFrame,
    benchmark_prices_df: pd.DataFrame,
    *,
    lookback_days: int = 20,
    benchmark_ticker: str = "^NSEI",
) -> pd.DataFrame:
    rs_col = f"stock_rs{int(lookback_days)}"
    cols = ["Date", "ticker", rs_col]
    if prices_df.empty or benchmark_prices_df.empty:
        return pd.DataFrame(columns=cols)

    p = prices_df[["Date", "Ticker", "Close"]].copy()
    p["Date"] = pd.to_datetime(p["Date"]).dt.normalize()
    p["ticker"] = p["Ticker"].astype(str).str.upper()
    p.sort_values(["ticker", "Date"], inplace=True)
    p["ret_lb_pct"] = p.groupby("ticker")["Close"].pct_change(int(lookback_days)) * 100.0
    p = p[p["ret_lb_pct"].notna()].copy()
    if p.empty:
        return pd.DataFrame(columns=cols)

    b = benchmark_prices_df[["Date", "Ticker", "Close"]].copy()
    b["Date"] = pd.to_datetime(b["Date"]).dt.normalize()
    b["Ticker"] = b["Ticker"].astype(str).str.upper()
    preferred = str(benchmark_ticker).strip().upper()
    if preferred:
        chosen = b[b["Ticker"] == preferred].copy()
    else:
        chosen = b.copy()
    if chosen.empty and not b.empty:
        first_ticker = str(b["Ticker"].iloc[0])
        chosen = b[b["Ticker"] == first_ticker].copy()
    if chosen.empty:
        return pd.DataFrame(columns=cols)

    chosen.sort_values("Date", inplace=True)
    chosen["benchmark_ret_lb_pct"] = chosen["Close"].pct_change(int(lookback_days)) * 100.0
    chosen = chosen[chosen["benchmark_ret_lb_pct"].notna()][["Date", "benchmark_ret_lb_pct"]].copy()
    if chosen.empty:
        return pd.DataFrame(columns=cols)

    out = p[["Date", "ticker", "ret_lb_pct"]].merge(chosen, on="Date", how="inner")
    if out.empty:
        return pd.DataFrame(columns=cols)
    out[rs_col] = out["ret_lb_pct"] - out["benchmark_ret_lb_pct"]
    return out[["Date", "ticker", rs_col]]


def _apply_stock_rs_score_bonus(
    signals_df: pd.DataFrame,
    *,
    enabled: bool,
    max_bonus: float = 3.0,
    rs20_weight: float = 0.20,
    rs50_weight: float = 0.10,
) -> pd.DataFrame:
    out = signals_df.copy()
    out["rs_bonus"] = 0.0
    if out.empty or not enabled or "signal_score" not in out.columns:
        return out

    def _numeric_col_or_zeros(column_name: str) -> pd.Series:
        if column_name not in out.columns:
            return pd.Series(0.0, index=out.index, dtype="float64")
        values = pd.to_numeric(out[column_name], errors="coerce")
        if not isinstance(values, pd.Series):
            return pd.Series(0.0, index=out.index, dtype="float64")
        return values.fillna(0.0).clip(lower=0.0, upper=10.0)

    rs20 = _numeric_col_or_zeros("stock_rs20")
    rs50 = _numeric_col_or_zeros("stock_rs50")
    raw_bonus = (rs20 * float(rs20_weight)) + (rs50 * float(rs50_weight))
    if float(max_bonus) > 0:
        raw_bonus = raw_bonus.clip(upper=float(max_bonus))
    out["rs_bonus"] = raw_bonus.round(2)
    out["signal_score"] = (
        pd.to_numeric(out["signal_score"], errors="coerce").fillna(0.0) + raw_bonus
    ).map(_clip_score)
    return out


def _attach_stock_index_rs(
    signals_df: pd.DataFrame,
    ticker_index_rs_df: pd.DataFrame,
) -> pd.DataFrame:
    if signals_df.empty or ticker_index_rs_df.empty:
        return signals_df.copy()

    out = signals_df.copy()
    rs = ticker_index_rs_df.copy()

    out["_signal_date_key"] = pd.to_datetime(out.get("signal_date"), errors="coerce").dt.normalize()
    out["_ticker_key"] = out.get("ticker", pd.Series(index=out.index, dtype="object")).astype(str).str.strip().str.upper()

    rs["_signal_date_key"] = pd.to_datetime(rs.get("Date"), errors="coerce").dt.normalize()
    rs["_ticker_key"] = rs.get("ticker", pd.Series(index=rs.index, dtype="object")).astype(str).str.strip().str.upper()
    rs = rs.drop(columns=[c for c in ["Date", "ticker"] if c in rs.columns])

    for col in [c for c in ["stock_rs20", "stock_rs50"] if c in out.columns and c in rs.columns]:
        out.drop(columns=[col], inplace=True)

    out = out.merge(rs, on=["_signal_date_key", "_ticker_key"], how="left")
    out.drop(columns=["_signal_date_key", "_ticker_key"], inplace=True)
    return out


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
        return f"Daily Stock Trigger Update | {signal_date}\n\nNo signal generated today.\n\nProduction: {PRODUCTION_APP_URL}"

    rows = signals_df[signals_df["signal_date"] == signal_date].copy()

    if rows.empty:
        return f"Daily Stock Trigger Update | {signal_date}\n\nNo signal generated today.\n\nProduction: {PRODUCTION_APP_URL}"

    if "signal_score" in rows.columns:
        rows.sort_values(["signal_score", "ticker"], ascending=[False, True], inplace=True)
    else:
        rows.sort_values(["ticker"], inplace=True)

    lines = [
        f"Daily Stock Trigger Update | {signal_date}",
        "",
        f"Signals generated today: {len(rows)}",
        "",
    ]
    for _, r in rows.iterrows():
        score = int(round(float(r["signal_score"]))) if "signal_score" in rows.columns else None
        score_text = f" | Score {score}" if score is not None else ""
        lines.append(
            f"- {r['ticker']}{score_text} | Entry {float(r['entry_price']):.2f} | {r['pattern']}"
        )
    lines.extend(["", f"Production: {PRODUCTION_APP_URL}"])
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


def refresh_stock_scores() -> tuple[bool, str]:
    """Regenerate stock_scores.csv from the current prices file."""

    scores_script = SCRIPTS_DIR / "generate_stock_scores.py"
    if not scores_script.is_file():
        return False, "Stock scores script not found under stock_triggers/scripts/."

    try:
        res = subprocess.run(
            [sys.executable, str(scores_script)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover
        return False, f"Error running stock score refresh: {exc}"

    if res.returncode != 0:
        detail = res.stderr.strip() or res.stdout.strip()
        return False, f"Stock score refresh failed (exit {res.returncode}): {detail}"

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
            if line.strip() and not line.lstrip().startswith("#") and not _is_benchmark_ticker(line.strip())
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
    stop_mode: str = "fixed_pct",
    atr_period: int = 14,
    atr_multiplier: float = 2.5,
    structure_atr_buffer: float = 0.5,
) -> pd.DataFrame:
    """Compute Pattern A signals for one date (delegated to patterns.pattern_a)."""
    return _pat_a.detect(
        prices,
        as_of_date=as_of_date,
        breakout_days=breakout_days,
        volume_multiplier=volume_multiplier,
        stop_pct=stop_pct,
        breakout_buffer_pct=breakout_buffer_pct,
        use_atr_stop=use_atr_stop,
        stop_mode=stop_mode,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        structure_atr_buffer=structure_atr_buffer,
    )


# Scoring functions delegated to patterns.scoring module
_clip_score = _scoring_mod.clip_score
_build_score_components = _scoring_mod.build_score_components
_apply_ma_slope_bonus = _scoring_mod.apply_ma_slope_bonus
_compute_ma_slope_pct = _scoring_mod.compute_ma_slope_pct
_apply_pattern_family_bonus = _scoring_mod.apply_pattern_family_bonus
WEIGHT_TREND = _scoring_mod.WEIGHT_TREND
WEIGHT_SETUP = _scoring_mod.WEIGHT_SETUP
WEIGHT_VOLUME = _scoring_mod.WEIGHT_VOLUME
WEIGHT_RISK = _scoring_mod.WEIGHT_RISK
WEIGHT_RSI = _scoring_mod.WEIGHT_RSI


def compute_pattern_b_signals_for_date(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    volume_multiplier: float,
    stop_pct: float,
    pullback_buffer_pct: float = 1.5,
    rebound_min_pct: float = 0.2,
) -> pd.DataFrame:
    """Pattern B: pullback rebound (delegated to patterns.pattern_b)."""
    return _pat_b.detect(
        prices,
        as_of_date=as_of_date,
        volume_multiplier=volume_multiplier,
        stop_pct=stop_pct,
        pullback_buffer_pct=pullback_buffer_pct,
        rebound_min_pct=rebound_min_pct,
        compute_rsi_fn=_compute_rsi_shared,
    )


def compute_dragonfly_doji_signals_for_date(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    stop_pct: float,
    body_pct_max: float = 0.3,
    upper_shadow_max_pct: float = 0.15,
    sma50_proximity_pct: float = 3.0,
    reclaim_sma200_days: int = 5,
) -> pd.DataFrame:
    """Pattern C (legacy) – Dragonfly Doji standalone signals (kept for backward compat)."""
    # Standalone doji pattern is now superseded by the enhancer approach;
    # kept here so existing references don't break.
    return pd.DataFrame(columns=[
        "signal_date", "ticker", "pattern", "pattern_family",
        "entry_price", "stop_pct", "stop_price",
        "score_trend", "score_setup", "score_volume",
        "score_rsi", "score_risk", "score_pattern", "sma50_slope_pct", "ma_slope_bonus", "pattern_bonus", "signal_score", "consensus_count",
    ])

def _has_doji_enhancement(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
    body_pct_max: float = 0.30,
    upper_shadow_max_pct: float = 0.15,
) -> bool:
    """Delegate to enhancers.dragonfly_doji.check."""
    return _enh_doji.check(prices, ticker, lookback=lookback,
                           body_pct_max=body_pct_max,
                           upper_shadow_max_pct=upper_shadow_max_pct)


def compute_scored_signals_for_date(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
    use_pattern_a: bool,
    use_pattern_b: bool,
    use_pattern_c: bool = False,
    use_pattern_d: bool = False,
    use_pattern_e: bool = False,
    use_pattern_f: bool = False,
    use_pattern_g: bool = False,
    doji_enhancer_bonus: float = 0.0,
    hammer_enhancer_bonus: float = 0.0,
    marubozu_enhancer_bonus: float = 0.0,
    confirmed_hammer_a_enhancer_bonus: float = 0.0,
    morning_star_enhancer_bonus: float = 0.0,
    engulfing_enhancer_bonus: float = 0.0,
    engulfing_trend_combo_enhancer_bonus: float = 0.0,
    harami_enhancer_bonus: float = 0.0,
    piercing_line_enhancer_bonus: float = 0.0,
    piercing_variant_enhancer_bonus: float = 0.0,
    piercing_variant_b_combo_enhancer_bonus: float = 0.0,
    inverted_hammer_enhancer_bonus: float = 0.0,
    belt_hold_enhancer_bonus: float = 0.0,
    three_white_soldiers_enhancer_bonus: float = 0.0,
    max_enhancer_total: float = 20.0,
    breakout_buffer_pct: float = 0.0,
    use_atr_stop: bool = False,
    stop_mode: str = "fixed_pct",
    atr_period: int = 14,
    atr_multiplier: float = 2.5,
    structure_atr_buffer: float = 0.5,
    pullback_buffer_pct: float = 1.5,
    rebound_min_pct: float = 0.2,
    min_signal_score: float = 0.0,
    consensus_bonus: float = 5.0,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    # ── Pattern A: Trend Breakout ──
    if use_pattern_a:
        a_df = compute_pattern_a_signals_for_date(
            prices,
            as_of_date=as_of_date,
            breakout_days=int(breakout_days),
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            breakout_buffer_pct=float(breakout_buffer_pct),
            use_atr_stop=bool(use_atr_stop),
            stop_mode=str(stop_mode),
            atr_period=int(atr_period),
            atr_multiplier=float(atr_multiplier),
            structure_atr_buffer=float(structure_atr_buffer),
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
                sma50_slope_pct = _compute_ma_slope_pct(g["SMA50"])
                ma_slope_bonus, signal_score = _apply_ma_slope_bonus(signal_score, sma50_slope_pct)
                a_df.at[i, "score_trend"] = score_trend
                a_df.at[i, "score_setup"] = score_setup
                a_df.at[i, "score_volume"] = score_volume
                a_df.at[i, "score_risk"] = score_risk
                a_df.at[i, "score_rsi"] = score_rsi
                a_df.at[i, "sma50_slope_pct"] = round(float(sma50_slope_pct), 2) if sma50_slope_pct is not None else pd.NA
                a_df.at[i, "ma_slope_bonus"] = ma_slope_bonus
                a_df.at[i, "signal_score"] = signal_score
            rows.append(a_df)

    # ── Pattern B: Pullback Rebound ──
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

    # ── Pattern C: MACD Crossover ──
    if use_pattern_c:
        c_df = _pat_c.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            compute_rsi_fn=_compute_rsi_shared,
        )
        if not c_df.empty:
            rows.append(c_df)

    # ── Pattern D: RSI Oversold Bounce ──
    if use_pattern_d:
        d_df = _pat_d.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            compute_rsi_fn=_compute_rsi_shared,
        )
        if not d_df.empty:
            rows.append(d_df)

    # ── Pattern E: Bollinger Squeeze Breakout ──
    if use_pattern_e:
        e_df = _pat_e.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            compute_rsi_fn=_compute_rsi_shared,
        )
        if not e_df.empty:
            rows.append(e_df)

    # ── Pattern F: VWAP Reclaim ──
    if use_pattern_f:
        f_df = _pat_f.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            compute_rsi_fn=_compute_rsi_shared,
        )
        if not f_df.empty:
            rows.append(f_df)

    # ── Pattern G: Volatility Contraction Pattern ──
    if use_pattern_g:
        g_df = _pat_g.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=min(float(volume_multiplier), 1.2),
            stop_pct=float(stop_pct),
            base_lookback=100,
            dryup_volume_ratio=1.0,
            compute_rsi_fn=_compute_rsi_shared,
        )
        if not g_df.empty:
            rows.append(g_df)

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
        "score_pattern",
        "sma50_slope_pct",
        "ma_slope_bonus",
        "pattern_bonus",
        "signal_score",
        "consensus_count",
        "hold_to_target_only",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.concat(rows, ignore_index=True)
    out["consensus_count"] = out.groupby(["signal_date", "ticker"])["pattern_family"].transform("nunique")

    # Consensus bonus when multiple pattern families agree
    if float(consensus_bonus) > 0:
        bonus_mask = out["consensus_count"] > 1
        out.loc[bonus_mask, "signal_score"] = out.loc[bonus_mask, "signal_score"].astype(float) + float(consensus_bonus)
        out["signal_score"] = out["signal_score"].astype(float).map(_clip_score)

    out = _apply_pattern_family_bonus(out, _load_pattern_weights())

    # ── Candle-shape enhancers ──
    enhancers = {
        "candle_doji": float(doji_enhancer_bonus),
        "candle_hammer": float(hammer_enhancer_bonus),
        "candle_marubozu": float(marubozu_enhancer_bonus),
        "candle_confirmed_hammer_a": float(confirmed_hammer_a_enhancer_bonus),
        "candle_morning_star": float(morning_star_enhancer_bonus),
        "candle_engulfing": float(engulfing_enhancer_bonus),
        "candle_engulfing_trend_combo": float(engulfing_trend_combo_enhancer_bonus),
        "candle_harami": float(harami_enhancer_bonus),
        "candle_piercing_line": float(piercing_line_enhancer_bonus),
        "candle_piercing_variant": float(piercing_variant_enhancer_bonus),
        "candle_piercing_variant_b_combo": float(piercing_variant_b_combo_enhancer_bonus),
        "candle_inverted_hammer": float(inverted_hammer_enhancer_bonus),
        "candle_belt_hold": float(belt_hold_enhancer_bonus),
        "candle_three_white_soldiers": float(three_white_soldiers_enhancer_bonus),
    }
    if any(b > 0 for b in enhancers.values()) and not out.empty:
        _tag_candle_shapes_fast(out, prices, ticker_col="ticker", date_col="signal_date")
        enh_total = pd.Series(0.0, index=out.index)
        for col, bonus in enhancers.items():
            if bonus > 0 and col in out.columns:
                enh_total.loc[out[col].astype(bool)] += bonus
        if float(max_enhancer_total) > 0:
            enh_total = enh_total.clip(upper=float(max_enhancer_total))
        out["signal_score"] = (out["signal_score"].astype(float) + enh_total).map(_clip_score)

    out.sort_values(["signal_date", "ticker", "signal_score"], ascending=[True, True, False], inplace=True)
    out = out.drop_duplicates(subset=["signal_date", "ticker"], keep="first")

    out = _annotate_hold_to_target_only(out, stop_mode)
    out.sort_values(["signal_date", "ticker"], inplace=True)
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
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

        stop_hit_rows = fut[
            (fut["Close"] <= stop_price)
            & fut["Date"].map(lambda value: _stop_exit_allowed(as_of_date, value))
        ]
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


def _normalize_stop_mode(stop_mode: str) -> str:
    mode = str(stop_mode or "fixed_pct").strip().lower()
    if mode not in {"fixed_pct", "atr", "structure_atr", "score_gt_95_hold_to_target", "score_gt_90_hold_to_target"}:
        return "fixed_pct"
    return mode


def _annotate_hold_to_target_only(signals_df: pd.DataFrame, stop_mode: str) -> pd.DataFrame:
    out = signals_df.copy()
    out["hold_to_target_only"] = False
    normalized = _normalize_stop_mode(stop_mode)
    if normalized == "score_gt_95_hold_to_target":
        threshold = 95.0
    elif normalized == "score_gt_90_hold_to_target":
        threshold = 90.0
    else:
        return out
    if out.empty or "signal_score" not in out.columns:
        return out
    scores = pd.to_numeric(out["signal_score"], errors="coerce")
    out.loc[scores > threshold, "hold_to_target_only"] = True
    return out


def _add_atr_columns(price_history: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    hist = price_history.copy().sort_values("Date")
    tr1 = hist["High"] - hist["Low"]
    tr2 = (hist["High"] - hist["Close"].shift(1)).abs()
    tr3 = (hist["Low"] - hist["Close"].shift(1)).abs()
    hist["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    hist["ATR"] = hist["TR"].rolling(int(atr_period)).mean()
    return hist


STOP_EXIT_LOCKOUT_DAYS = 7


def _stop_exit_allowed(signal_date: pd.Timestamp, bar_date: pd.Timestamp, *, lockout_days: int = STOP_EXIT_LOCKOUT_DAYS) -> bool:
    sig_dt = pd.to_datetime(signal_date)
    bar_dt = pd.to_datetime(bar_date)
    return (bar_dt - sig_dt).days > int(lockout_days)


@st.cache_data(show_spinner=False)
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
        "score_pattern",
        "sma50_slope_pct",
        "ma_slope_bonus",
        "stock_rs20",
        "stock_rs50",
        "consensus_count",
        "hold_to_target_only",
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
        hold_to_target_only = bool(sig.get("hold_to_target_only", False))
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
                    "score_pattern": sig.get("score_pattern", pd.NA),
                    "sma50_slope_pct": sig.get("sma50_slope_pct", pd.NA),
                    "ma_slope_bonus": sig.get("ma_slope_bonus", 0.0),
                    "stock_rs20": sig.get("stock_rs20", pd.NA),
                    "stock_rs50": sig.get("stock_rs50", pd.NA),
                    "consensus_count": sig.get("consensus_count", 1),
                    "hold_to_target_only": hold_to_target_only,
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

            if (
                not hold_to_target_only
                and _stop_exit_allowed(sig_dt, row["Date"])
                and float(row["Close"]) <= dynamic_stop
            ):
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
                "score_pattern": sig.get("score_pattern", pd.NA),
                "sma50_slope_pct": sig.get("sma50_slope_pct", pd.NA),
                "ma_slope_bonus": sig.get("ma_slope_bonus", 0.0),
                "stock_rs20": sig.get("stock_rs20", pd.NA),
                "stock_rs50": sig.get("stock_rs50", pd.NA),
                "rs_bonus": sig.get("rs_bonus", 0.0),
                "consensus_count": sig.get("consensus_count", 1),
                "hold_to_target_only": hold_to_target_only,
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


@st.cache_data(show_spinner=False)
def _apply_lab_stop_mode(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    stop_mode: str,
    fixed_stop_pct: float = 7.0,
    atr_period: int = 14,
    atr_multiplier: float = 2.5,
    structure_lookback: int = 5,
    structure_atr_buffer: float = 0.5,
) -> pd.DataFrame:
    """Apply the selected stop-loss mode to lab signals."""
    if signals_df.empty or prices_df.empty:
        return signals_df.copy()

    out = signals_df.copy()
    px = prices_df.copy()
    px["Date"] = pd.to_datetime(px["Date"])
    grouped = {str(ticker): grp.sort_values("Date") for ticker, grp in px.groupby("Ticker", sort=False)}

    for idx in out.index:
        ticker = str(out.at[idx, "ticker"])
        ticker_alt = ticker[:-3] if ticker.endswith(".NS") else ticker + ".NS"
        hist = grouped.get(ticker)
        if hist is None:
            hist = grouped.get(ticker_alt)

        entry_price = float(out.at[idx, "entry_price"])
        fallback_stop = entry_price * (1.0 - float(fixed_stop_pct) / 100.0)
        if hist is None or entry_price <= 0:
            out.at[idx, "stop_price"] = round(fallback_stop, 4)
            out.at[idx, "stop_pct"] = round(float(fixed_stop_pct), 2)
            continue

        sig_date = pd.to_datetime(out.at[idx, "signal_date"])
        hist = hist[hist["Date"] <= sig_date].copy()
        if hist.empty:
            out.at[idx, "stop_price"] = round(fallback_stop, 4)
            out.at[idx, "stop_pct"] = round(float(fixed_stop_pct), 2)
            continue

        tr1 = hist["High"] - hist["Low"]
        tr2 = (hist["High"] - hist["Close"].shift(1)).abs()
        tr3 = (hist["Low"] - hist["Close"].shift(1)).abs()
        hist["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        hist["ATR"] = hist["TR"].rolling(int(atr_period)).mean()
        atr_value = float(hist.iloc[-1]["ATR"]) if pd.notna(hist.iloc[-1]["ATR"]) else None

        effective_stop_mode = _normalize_stop_mode(stop_mode)

        if effective_stop_mode == "atr" and atr_value is not None:
            stop_price = entry_price - atr_value * float(atr_multiplier)
        elif effective_stop_mode == "structure_atr" and atr_value is not None:
            recent_low = float(hist["Low"].tail(int(structure_lookback)).min())
            stop_price = recent_low - atr_value * float(structure_atr_buffer)
        else:
            stop_price = fallback_stop

        if effective_stop_mode in {"atr", "structure_atr"}:
            stop_price = max(float(stop_price), float(fallback_stop))

        if stop_price <= 0 or stop_price >= entry_price:
            stop_price = fallback_stop

        stop_pct_eff = ((entry_price - stop_price) / entry_price) * 100.0
        out.at[idx, "stop_price"] = round(float(stop_price), 4)
        out.at[idx, "stop_pct"] = round(float(stop_pct_eff), 2)

    return out


@st.cache_data(show_spinner=False)
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
            - Target Hit: intraday high reaches entry * (1 + target_pct/100)
            - Stop Hit:   daily close falls to or below stop_price
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
        stop_price_calc = stop_price_sig
        hold_to_target_only = bool(sig.get("hold_to_target_only", False))

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
            latest_close = close
            if high >= target_price:
                status = "Target Hit ✅"
                exit_date = bar["Date"]
                exit_price = target_price
                break
            if (
                not hold_to_target_only
                and _stop_exit_allowed(sig_date, bar["Date"])
                and close <= stop_price_calc
            ):
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
            "pattern_family": str(sig.get("pattern_family", "")),
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
            "score_pattern": round(float(sig["score_pattern"]), 1) if pd.notna(sig.get("score_pattern")) else None,
            "sma50_slope_pct": round(float(sig["sma50_slope_pct"]), 2) if pd.notna(sig.get("sma50_slope_pct")) else None,
            "ma_slope_bonus": round(float(sig["ma_slope_bonus"]), 2) if pd.notna(sig.get("ma_slope_bonus")) else 0.0,
            "pattern_bonus": round(float(sig["pattern_bonus"]), 2) if pd.notna(sig.get("pattern_bonus")) else 0.0,
            "stock_rs20": round(float(sig["stock_rs20"]), 2) if pd.notna(sig.get("stock_rs20")) else None,
            "stock_rs50": round(float(sig["stock_rs50"]), 2) if pd.notna(sig.get("stock_rs50")) else None,
            "rs_bonus": round(float(sig["rs_bonus"]), 2) if pd.notna(sig.get("rs_bonus")) else 0.0,
            "enhancer_bonus": round(float(sig["enhancer_bonus"]), 1) if "enhancer_bonus" in sig.index and pd.notna(sig.get("enhancer_bonus")) else 0.0,
            "hold_to_target_only": hold_to_target_only,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.sort_values(["signal_date", "ticker"], ascending=[False, True], inplace=True)
    return out


def summarize_signal_tracker(view: pd.DataFrame) -> dict[str, float | int]:
    if view.empty:
        return {
            "n_total": 0,
            "n_target": 0,
            "n_stop": 0,
            "n_holding": 0,
            "total_invested": 0.0,
            "total_current": 0.0,
            "total_pnl": 0.0,
            "overall_return": 0.0,
            "closed_invested": 0.0,
            "closed_current": 0.0,
            "closed_pnl": 0.0,
            "closed_return": 0.0,
            "win_rate": 0.0,
        }

    n_total = len(view)
    n_target = int((view["status"] == "Target Hit ✅").sum())
    n_stop = int((view["status"] == "Stop Hit 🛑").sum())
    n_holding = int((view["status"] == "Holding").sum())
    total_invested = float(view["invested"].sum())
    total_current = float(view["current_value"].sum())
    total_pnl = float(view["pnl"].sum())
    overall_return = ((total_current / total_invested) - 1) * 100 if total_invested > 0 else 0.0

    closed_view = view[view["status"].isin(["Target Hit ✅", "Stop Hit 🛑"])].copy()
    closed_invested = float(closed_view["invested"].sum()) if not closed_view.empty else 0.0
    closed_current = float(closed_view["current_value"].sum()) if not closed_view.empty else 0.0
    closed_pnl = float(closed_view["pnl"].sum()) if not closed_view.empty else 0.0
    closed_return = ((closed_current / closed_invested) - 1) * 100 if closed_invested > 0 else 0.0
    win_rate = (n_target / (n_target + n_stop) * 100) if (n_target + n_stop) > 0 else 0.0

    return {
        "n_total": n_total,
        "n_target": n_target,
        "n_stop": n_stop,
        "n_holding": n_holding,
        "total_invested": total_invested,
        "total_current": total_current,
        "total_pnl": total_pnl,
        "overall_return": overall_return,
        "closed_invested": closed_invested,
        "closed_current": closed_current,
        "closed_pnl": closed_pnl,
        "closed_return": closed_return,
        "win_rate": win_rate,
    }


def _freeze_session_cache_value(value):
    if isinstance(value, dict):
        return tuple((str(k), _freeze_session_cache_value(v)) for k, v in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze_session_cache_value(v) for v in value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.to_datetime(value).isoformat()
    if isinstance(value, float):
        return round(float(value), 6)
    return value


def _make_session_cache_key(prefix: str, params: dict) -> tuple:
    frozen = tuple((str(k), _freeze_session_cache_value(v)) for k, v in sorted(params.items(), key=lambda item: str(item[0])))
    return (prefix, frozen)


def _session_cache_get_df(bucket_name: str, cache_key: tuple) -> pd.DataFrame | None:
    bucket = st.session_state.setdefault(bucket_name, {})
    cached = bucket.get(cache_key)
    if isinstance(cached, pd.DataFrame):
        return cached.copy()
    return None


def _session_cache_set_df(bucket_name: str, cache_key: tuple, df: pd.DataFrame, *, max_items: int = 24) -> None:
    bucket = st.session_state.setdefault(bucket_name, {})
    if cache_key in bucket:
        bucket.pop(cache_key)
    bucket[cache_key] = df.copy()
    while len(bucket) > max_items:
        bucket.pop(next(iter(bucket)))


def _record_lab_session_snapshot(snapshot_key: tuple, params: dict, summary: dict[str, float | int], view: pd.DataFrame, *, max_items: int = 40) -> None:
    bucket = st.session_state.setdefault("_lab_session_dump", {})
    if snapshot_key in bucket:
        bucket.pop(snapshot_key)
    bucket[snapshot_key] = {
        "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": params.copy(),
        "summary": summary.copy(),
        "view": view.copy(),
    }
    while len(bucket) > max_items:
        bucket.pop(next(iter(bucket)))


def _build_lab_session_history_df() -> pd.DataFrame:
    bucket = st.session_state.get("_lab_session_dump", {})
    if not bucket:
        return pd.DataFrame()

    rows: list[dict] = []
    for item in reversed(list(bucket.values())):
        params = item.get("params", {})
        summary = item.get("summary", {})
        rows.append({
            "captured_at": item.get("captured_at", ""),
            "pattern_families": _format_lab_cache_param(params.get("pattern_families")),
            "source_mode": params.get("source_mode"),
            "eval_mode": params.get("evaluation_mode"),
            "target_pct": params.get("target_pct"),
            "stop_mode": params.get("stop_mode"),
            "stop_pct": params.get("stop_pct"),
            "atr_mult": params.get("atr_mult"),
            "min_score": params.get("min_score"),
            "rescore": params.get("rescore"),
            "rs_bonus": params.get("rs_bonus"),
            "enhancers": _summarize_lab_bonus_params(params),
            "max_enh": params.get("max_enh_bonus"),
            "status": params.get("status_filter"),
            "candles": params.get("candle_filter"),
            "sort_by": params.get("sort_by"),
            "desc": params.get("sort_desc"),
            "rows": summary.get("n_total", 0),
            "targets": summary.get("n_target", 0),
            "stops": summary.get("n_stop", 0),
            "holding": summary.get("n_holding", 0),
            "overall_return": summary.get("overall_return", 0.0),
            "closed_return": summary.get("closed_return", 0.0),
        })
    return pd.DataFrame(rows)


def _format_lab_cache_param(value) -> str:
    if isinstance(value, (list, tuple, set)):
        parts = [str(v) for v in value if str(v).strip()]
        return ", ".join(parts)
    if value is None:
        return ""
    return str(value)


def _summarize_lab_bonus_params(params: dict) -> str:
    bonus_labels = (
        ("doji_bonus", "Doji"),
        ("hammer_bonus", "Hammer"),
        ("marubozu_bonus", "Marubozu"),
        ("confirmed_hammer_a_bonus", "Ham+A"),
        ("morning_star_bonus", "M.Star"),
        ("engulf_bonus", "Engulf"),
        ("engulf_trend_combo_bonus", "Engulf A/C/G"),
        ("harami_bonus", "Harami"),
        ("piercing_bonus", "Pierce"),
        ("piercing_variant_bonus", "Pierce V"),
        ("piercing_variant_b_combo_bonus", "PierceV+B"),
        ("inv_hammer_bonus", "Inv Ham"),
        ("belt_hold_bonus", "Belt"),
        ("three_white_bonus", "3 White"),
    )
    active: list[str] = []
    for key, label in bonus_labels:
        value = params.get(key)
        try:
            amount = float(value or 0.0)
        except (TypeError, ValueError):
            amount = 0.0
        if amount > 0:
            active.append(f"{label}:{amount:g}")
    return ", ".join(active)


def _build_lab_session_dump_df() -> pd.DataFrame:
    bucket = st.session_state.get("_lab_session_dump", {})
    if not bucket:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for item in bucket.values():
        view = item.get("view")
        if not isinstance(view, pd.DataFrame) or view.empty:
            continue
        params = item.get("params", {})
        summary = item.get("summary", {})
        out = view.copy()
        out.insert(0, "cache_captured_at", item.get("captured_at", ""))
        out.insert(1, "cache_stop_mode", params.get("stop_mode"))
        out.insert(2, "cache_min_score", params.get("min_score"))
        out.insert(3, "cache_status", params.get("status_filter"))
        out.insert(4, "cache_candles", _format_lab_cache_param(params.get("candle_filter")))
        out.insert(5, "cache_sort_by", params.get("sort_by"))
        out.insert(6, "cache_sort_desc", params.get("sort_desc"))
        out.insert(7, "cache_max_days_held", params.get("max_days_held"))
        out.insert(8, "cache_overall_return", summary.get("overall_return", 0.0))
        out.insert(9, "cache_closed_return", summary.get("closed_return", 0.0))
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _clear_lab_session_cache() -> None:
    st.session_state["_lab_tracker_cache"] = {}
    st.session_state["_lab_view_cache"] = {}
    st.session_state["_lab_session_dump"] = {}


def _filter_signal_tracker_view(
    tracker_df: pd.DataFrame,
    *,
    status_filter: str,
    candle_filters: list[str],
    max_days_held: int | None,
    sort_by: str,
    sort_desc: bool,
) -> pd.DataFrame:
    view = tracker_df.copy()
    if status_filter != "All":
        view = view[view["status"] == status_filter].copy()
    if max_days_held is not None and "days_held" in view.columns:
        view["days_held"] = pd.to_numeric(view["days_held"], errors="coerce")
        view = view[view["days_held"].notna() & (view["days_held"] <= int(max_days_held))].copy()
    if candle_filters and not view.empty:
        candle_map = {
            "Doji": "candle_doji",
            "Hammer": "candle_hammer",
            "Bullish Marubozu": "candle_marubozu",
            "Confirmed Hammer + Pattern A": "candle_confirmed_hammer_a",
            "Morning Star": "candle_morning_star",
            "Engulfing": "candle_engulfing",
            "Engulf A/C/G": "candle_engulfing_trend_combo",
            "Harami": "candle_harami",
            "Piercing Line": "candle_piercing_line",
            "Piercing Variant": "candle_piercing_variant",
            "Pierce V+B": "candle_piercing_variant_b_combo",
            "Inverted Hammer": "candle_inverted_hammer",
            "Belt Hold": "candle_belt_hold",
            "Three White Soldiers": "candle_three_white_soldiers",
        }
        candle_mask = pd.Series(False, index=view.index)
        for label in candle_filters:
            col = candle_map.get(label)
            if col and col in view.columns:
                candle_mask = candle_mask | view[col].astype(bool)
        view = view[candle_mask].copy()
    if not view.empty and sort_by in view.columns:
        if sort_by == "ticker":
            view = view.sort_values([sort_by, "signal_date"], ascending=[not sort_desc, False]).copy()
        else:
            view = view.sort_values(sort_by, ascending=not sort_desc, na_position="last").copy()
    return view.reset_index(drop=True)


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
    stop_mode: str = "fixed_pct",
    atr_period: int = 14,
    atr_multiplier: float = 2.5,
    structure_atr_buffer: float = 0.5,
    break_even_trigger_pct: float | None = None,
    time_stop_days: int | None = None,
    ticker_sector_rs_df: pd.DataFrame | None = None,
    ticker_index_rs_df: pd.DataFrame | None = None,
    min_sector_rs20: float | None = None,
    min_stock_rs20: float | None = None,
    min_stock_rs50: float | None = None,
    use_stock_rs_score_bonus: bool = False,
    stock_rs_max_bonus: float = 3.0,
    use_pattern_a: bool = True,
    use_pattern_b: bool = False,
    use_pattern_c: bool = False,
    use_pattern_d: bool = False,
    use_pattern_e: bool = False,
    use_pattern_f: bool = False,
    use_pattern_g: bool = False,
    doji_enhancer_bonus: float = 0.0,
    hammer_enhancer_bonus: float = 0.0,
    marubozu_enhancer_bonus: float = 0.0,
    confirmed_hammer_a_enhancer_bonus: float = 0.0,
    morning_star_enhancer_bonus: float = 0.0,
    engulfing_enhancer_bonus: float = 0.0,
    engulfing_trend_combo_enhancer_bonus: float = 0.0,
    harami_enhancer_bonus: float = 0.0,
    piercing_line_enhancer_bonus: float = 0.0,
    piercing_variant_enhancer_bonus: float = 0.0,
    piercing_variant_b_combo_enhancer_bonus: float = 0.0,
    inverted_hammer_enhancer_bonus: float = 0.0,
    belt_hold_enhancer_bonus: float = 0.0,
    three_white_soldiers_enhancer_bonus: float = 0.0,
    max_enhancer_total: float = 20.0,
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
            use_pattern_c=bool(use_pattern_c),
            use_pattern_d=bool(use_pattern_d),
            use_pattern_e=bool(use_pattern_e),
            use_pattern_f=bool(use_pattern_f),
            use_pattern_g=bool(use_pattern_g),
            doji_enhancer_bonus=float(doji_enhancer_bonus),
            hammer_enhancer_bonus=float(hammer_enhancer_bonus),
            marubozu_enhancer_bonus=float(marubozu_enhancer_bonus),
            confirmed_hammer_a_enhancer_bonus=float(confirmed_hammer_a_enhancer_bonus),
            morning_star_enhancer_bonus=float(morning_star_enhancer_bonus),
            engulfing_enhancer_bonus=float(engulfing_enhancer_bonus),
            engulfing_trend_combo_enhancer_bonus=float(engulfing_trend_combo_enhancer_bonus),
            harami_enhancer_bonus=float(harami_enhancer_bonus),
            piercing_line_enhancer_bonus=float(piercing_line_enhancer_bonus),
            piercing_variant_enhancer_bonus=float(piercing_variant_enhancer_bonus),
            piercing_variant_b_combo_enhancer_bonus=float(piercing_variant_b_combo_enhancer_bonus),
            inverted_hammer_enhancer_bonus=float(inverted_hammer_enhancer_bonus),
            belt_hold_enhancer_bonus=float(belt_hold_enhancer_bonus),
            three_white_soldiers_enhancer_bonus=float(three_white_soldiers_enhancer_bonus),
            max_enhancer_total=float(max_enhancer_total),
            breakout_buffer_pct=float(breakout_buffer_pct),
            use_atr_stop=bool(use_atr_stop),
            stop_mode=str(stop_mode),
            atr_period=int(atr_period),
            atr_multiplier=float(atr_multiplier),
            structure_atr_buffer=float(structure_atr_buffer),
            pullback_buffer_pct=float(pullback_buffer_pct),
            rebound_min_pct=float(rebound_min_pct),
            min_signal_score=float(min_signal_score),
            consensus_bonus=float(consensus_bonus),
        )

        if ticker_index_rs_df is not None and not ticker_index_rs_df.empty and not day_signals.empty:
            day_idx_rs = ticker_index_rs_df[
                ticker_index_rs_df["Date"] == pd.to_datetime(d).normalize()
            ].drop(columns=["Date"], errors="ignore").copy()
            if not day_idx_rs.empty:
                for _c in [c for c in ["stock_rs20", "stock_rs50"] if c in day_signals.columns]:
                    day_signals.drop(columns=[_c], inplace=True)
                day_signals = day_signals.merge(day_idx_rs, on="ticker", how="left")

        if not day_signals.empty and (min_stock_rs20 is not None or min_stock_rs50 is not None):
            if min_stock_rs20 is not None:
                day_signals = day_signals[
                    day_signals["stock_rs20"].notna()
                    & (day_signals["stock_rs20"] >= float(min_stock_rs20))
                ].copy()
            if min_stock_rs50 is not None and not day_signals.empty:
                day_signals = day_signals[
                    day_signals["stock_rs50"].notna()
                    & (day_signals["stock_rs50"] >= float(min_stock_rs50))
                ].copy()

        if not day_signals.empty:
            day_signals = _apply_stock_rs_score_bonus(
                day_signals,
                enabled=bool(use_stock_rs_score_bonus),
                max_bonus=float(stock_rs_max_bonus),
            )
            day_signals = day_signals[
                pd.to_numeric(day_signals["signal_score"], errors="coerce").fillna(0.0) >= float(min_signal_score)
            ].copy()

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
                "sma50_slope_pct",
                "ma_slope_bonus",
                "signal_score",
                "rs_bonus",
                "consensus_count",
                "hold_to_target_only",
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
        "min_score": 90,
        "sort_by": "Selected method",
        "score_method": "Heuristic score",
        "lab_evaluation_mode": "walk-forward",
        "lab_train_end_date": date.today(),
        "lab_eval_hold_days": 30,
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
    if "macd" in p:
        return "MACD crossover"
    if "rsi" in p:
        return "RSI bounce"
    if "boll" in p:
        return "BB squeeze"
    if "vwap" in p:
        return "VWAP reclaim"
    return str(pattern).replace("_", " ").strip().title()


def _plain_reason(score: float, risk_pct: float, pattern: str) -> str:
    if score >= 75 and risk_pct <= 7:
        return "Strong setup with controlled risk."
    if "pullback" in str(pattern).lower():
        return "Uptrend stock near a pullback zone."
    if risk_pct > 9:
        return "Setup looks okay, but risk is wide."
    return "Trend and price action are still supportive."


def _get_tomorrow_score_method() -> dict:
    selected = str(st.session_state.get("score_method", "Heuristic score"))
    return TOMORROW_SCORE_METHODS.get(selected, TOMORROW_SCORE_METHODS["Heuristic score"])


def _get_backtest_train_end_date() -> date:
    value = st.session_state.get("lab_train_end_date", date.today())
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return date.today()
    return parsed.date()


_COVERAGE_RAW_COLS = [
    "signal_date",
    "ticker",
    "pattern",
    "pattern_family",
    "entry_price",
    "stop_pct",
    "stop_price",
]


def _scan_raw_breakouts_for_date(
    prices_df: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    pattern_families: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    selected = {str(family).strip().upper() for family in pattern_families if str(family).strip()}

    def _append(frame: pd.DataFrame) -> None:
        if not frame.empty:
            rows.append(frame[[col for col in _COVERAGE_RAW_COLS if col in frame.columns]].copy())

    try:
        if "A" in selected:
            _append(
                _pat_a.detect(
                    prices_df,
                    as_of_date=as_of_date,
                    breakout_days=40,
                    volume_multiplier=1.0,
                    stop_pct=7.0,
                )
            )
        if "B" in selected:
            _append(
                _pat_b.detect(
                    prices_df,
                    as_of_date=as_of_date,
                    volume_multiplier=1.0,
                    stop_pct=7.0,
                    pullback_buffer_pct=1.5,
                    rebound_min_pct=0.2,
                    compute_rsi_fn=_compute_rsi_shared,
                )
            )
        if "C" in selected:
            _append(
                _pat_c.detect(
                    prices_df,
                    as_of_date=as_of_date,
                    volume_multiplier=1.0,
                    stop_pct=7.0,
                    compute_rsi_fn=_compute_rsi_shared,
                )
            )
        if "D" in selected:
            _append(
                _pat_d.detect(
                    prices_df,
                    as_of_date=as_of_date,
                    volume_multiplier=1.0,
                    stop_pct=7.0,
                    compute_rsi_fn=_compute_rsi_shared,
                )
            )
        if "E" in selected:
            _append(
                _pat_e.detect(
                    prices_df,
                    as_of_date=as_of_date,
                    volume_multiplier=1.0,
                    stop_pct=7.0,
                    compute_rsi_fn=_compute_rsi_shared,
                )
            )
        if "F" in selected:
            _append(
                _pat_f.detect(
                    prices_df,
                    as_of_date=as_of_date,
                    volume_multiplier=1.0,
                    stop_pct=7.0,
                    compute_rsi_fn=_compute_rsi_shared,
                )
            )
        if "G" in selected:
            _append(
                _pat_g.detect(
                    prices_df,
                    as_of_date=as_of_date,
                    volume_multiplier=1.0,
                    stop_pct=7.0,
                    base_lookback=100,
                    dryup_volume_ratio=1.0,
                    compute_rsi_fn=_compute_rsi_shared,
                )
            )
    except Exception:
        return pd.DataFrame(columns=_COVERAGE_RAW_COLS)

    if not rows:
        return pd.DataFrame(columns=_COVERAGE_RAW_COLS)
    return pd.concat(rows, ignore_index=True)


@st.cache_data(show_spinner=False)
def _scan_raw_breakout_candidates(
    prices_df: pd.DataFrame,
    pattern_families: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Scan price history for raw detector events, independent of saved signals and scores."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_dates = pd.to_datetime(prices_df["Date"], errors="coerce").dropna()
    all_dates = sorted(d for d in all_dates.unique() if start_ts <= pd.Timestamp(d) <= end_ts)

    rows: list[pd.DataFrame] = []
    for as_of_date in all_dates:
        detected = _scan_raw_breakouts_for_date(
            prices_df,
            as_of_date=pd.Timestamp(as_of_date),
            pattern_families=pattern_families,
        )
        if not detected.empty:
            rows.append(detected)

    if not rows:
        return pd.DataFrame(columns=_COVERAGE_RAW_COLS)

    out = pd.concat(rows, ignore_index=True)
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out["pattern_family"] = out["pattern_family"].astype(str).str.strip().str.upper()
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out = out.dropna(subset=["signal_date", "ticker", "pattern_family", "entry_price"])
    out = out.drop_duplicates(subset=["signal_date", "ticker", "pattern_family", "pattern", "entry_price"])
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _evaluate_raw_breakout_targets(
    raw_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_return_pct: float,
    forward_days: int,
) -> pd.DataFrame:
    """Mark raw detector events that reach the target return within a forward window."""
    if raw_df.empty:
        return pd.DataFrame(columns=[
            *_COVERAGE_RAW_COLS,
            "target_return_pct",
            "target_price",
            "bars_available_forward",
            "first_target_hit_day",
            "is_breakout",
            "is_pending",
        ])

    grouped_prices = {
        str(ticker): grp.sort_values("Date").copy()
        for ticker, grp in prices_df.groupby("Ticker", sort=False)
    }
    out_rows: list[dict] = []

    for _, row in raw_df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        signal_date = pd.to_datetime(row.get("signal_date"), errors="coerce")
        entry_price = pd.to_numeric(row.get("entry_price"), errors="coerce")
        target_price = float(entry_price) * (1.0 + float(target_return_pct) / 100.0) if pd.notna(entry_price) else pd.NA

        future = pd.DataFrame()
        if ticker and pd.notna(signal_date) and ticker in grouped_prices:
            future = grouped_prices[ticker]
            future = future[future["Date"] > signal_date].head(int(forward_days)).copy()

        bars_available_forward = int(len(future))
        first_target_hit_day: int | None = None
        if pd.notna(target_price):
            for day_number, (_, bar) in enumerate(future.iterrows(), start=1):
                high_value = pd.to_numeric(bar.get("High"), errors="coerce")
                if pd.notna(high_value) and float(high_value) >= float(target_price):
                    first_target_hit_day = day_number
                    break

        is_breakout = first_target_hit_day is not None
        is_pending = (not is_breakout) and bars_available_forward < int(forward_days)
        out_rows.append(
            {
                **{col: row.get(col) for col in _COVERAGE_RAW_COLS},
                "target_return_pct": round(float(target_return_pct), 2),
                "target_price": round(float(target_price), 4) if pd.notna(target_price) else pd.NA,
                "bars_available_forward": bars_available_forward,
                "first_target_hit_day": first_target_hit_day,
                "is_breakout": bool(is_breakout),
                "is_pending": bool(is_pending),
            }
        )

    return pd.DataFrame(out_rows)


@st.cache_data(show_spinner=False)
def _scan_raw_pattern_a_breakouts(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility helper for the audit expander; scans raw Pattern A events only."""
    if prices_df.empty:
        return pd.DataFrame(columns=["signal_date", "ticker", "entry_price", "stop_pct"])

    all_dates = pd.to_datetime(prices_df["Date"], errors="coerce").dropna()
    if all_dates.empty:
        return pd.DataFrame(columns=["signal_date", "ticker", "entry_price", "stop_pct"])

    out = _scan_raw_breakout_candidates(
        prices_df,
        ("A",),
        pd.Timestamp(all_dates.min()).date().isoformat(),
        pd.Timestamp(all_dates.max()).date().isoformat(),
    )
    keep_cols = [col for col in ["signal_date", "ticker", "entry_price", "stop_pct"] if col in out.columns]
    return out[keep_cols].copy()


def _render_coverage_page(
    all_signals: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """Signal coverage analysis page."""
    import plotly.graph_objects as go

    _cov_header, _cov_theme_col = st.columns([4, 2])
    with _cov_header:
        st.subheader("Signal Coverage Analysis")
        st.caption("How many breakouts did the model recognise, and why did it miss the rest?")
    with _cov_theme_col:
        render_theme_toggle_control()

    if prices.empty:
        st.info("No price data available. Load price history first.")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    _ctrl1, _ctrl2, _ctrl3, _ctrl4, _ctrl5, _ctrl6 = st.columns([0.9, 0.9, 1.1, 0.8, 0.8, 0.8])
    with _ctrl1:
        target_return_pct = st.number_input(
            "Target return %", min_value=1.0, max_value=25.0, value=6.0, step=0.5,
            key="cov_target_return_pct",
            help="A breakout only counts if price reaches at least this return after the raw detector event.",
        )
    with _ctrl2:
        forward_days = st.number_input(
            "Forward days", min_value=5, max_value=120, value=int(_COV_DEFAULT_FORWARD_DAYS), step=5,
            key="cov_forward_days",
            help="Forward trading-day window used to confirm whether a raw event actually delivered the target return.",
        )
    with _ctrl3:
        score_threshold = st.slider(
            "Recognition threshold", min_value=50, max_value=95, value=80, step=5,
            key="cov_threshold",
            help="Saved signals scoring at or above this value are counted as 'recognised'.",
        )
    _families_avail = ["A", "B", "C", "D", "E", "F", "G"]
    with _ctrl4:
        selected_families = st.multiselect(
            "Pattern families", options=_families_avail, default=_families_avail,
            key="cov_families",
        )
    _all_dates_cov = pd.to_datetime(prices["Date"], errors="coerce").dropna()
    if _all_dates_cov.empty:
        st.info("No dated price rows available for coverage analysis.")
        return
    _min_date_cov = _all_dates_cov.min().date()
    _max_date_cov = _all_dates_cov.max().date()
    with _ctrl5:
        date_from = st.date_input(
            "From", value=_min_date_cov, min_value=_min_date_cov, max_value=_max_date_cov,
            key="cov_date_from",
        )
    with _ctrl6:
        date_to = st.date_input(
            "To", value=_max_date_cov, min_value=_min_date_cov, max_value=_max_date_cov,
            key="cov_date_to",
        )

    if date_from > date_to:
        st.warning("The start date must be earlier than or equal to the end date.")
        return

    # ── Raw breakout universe (independent of saved signals / scores) ─────────
    use_default_cache = (
        abs(float(target_return_pct) - float(_COV_DEFAULT_TARGET_RETURN_PCT)) < 1e-9
        and int(forward_days) == int(_COV_DEFAULT_FORWARD_DAYS)
        and int(score_threshold) == int(_COV_DEFAULT_RECOGNITION_THRESHOLD)
        and set(selected_families) == set(_COV_DEFAULT_PATTERN_FAMILIES)
        and date_from == _min_date_cov
        and date_to == _max_date_cov
    )

    cached_payload = load_default_coverage_cache() if use_default_cache else {}
    if cached_payload:
        df = cached_payload.get("df", pd.DataFrame()).copy()
        meta = cached_payload.get("meta") or {}
        pending_count = int(meta.get("pending_count", 0) or 0)
    else:
        raw_candidates = _scan_raw_breakout_candidates(
            prices,
            tuple(selected_families),
            date_from.isoformat(),
            date_to.isoformat(),
        )
        if raw_candidates.empty:
            st.warning("No raw detector events matched the selected families and date range.")
            return

        evaluated = _evaluate_raw_breakout_targets(
            raw_candidates,
            prices,
            target_return_pct=float(target_return_pct),
            forward_days=int(forward_days),
        )
        breakout_df = evaluated[evaluated["is_breakout"]].copy()
        pending_count = int(evaluated["is_pending"].sum()) if "is_pending" in evaluated.columns else 0
        if breakout_df.empty:
            st.warning(
                f"No raw detector events reached +{float(target_return_pct):.1f}% within {int(forward_days)} trading days in the selected scope."
            )
            if pending_count > 0:
                st.caption(f"{pending_count} recent raw candidates are still pending because they do not yet have the full forward window.")
            return

        saved_cols = [
            "signal_date",
            "ticker",
            "pattern_family",
            "pattern",
            "signal_score",
            "score_trend",
            "score_setup",
            "score_volume",
            "score_rsi",
            "score_risk",
            "pattern_bonus",
        ]
        if all_signals.empty:
            saved_best = pd.DataFrame(columns=saved_cols)
        else:
            saved_best = all_signals.copy()
            saved_best["signal_date"] = pd.to_datetime(saved_best["signal_date"], errors="coerce")
            saved_best["pattern_family"] = saved_best["pattern_family"].astype(str).str.strip().str.upper()
            saved_best["ticker"] = saved_best["ticker"].astype(str).str.strip()
            saved_best["signal_score"] = pd.to_numeric(saved_best.get("signal_score"), errors="coerce")
            saved_best = saved_best[saved_best["pattern_family"].isin(selected_families)]
            saved_best = saved_best[saved_best["signal_date"].dt.date >= date_from]
            saved_best = saved_best[saved_best["signal_date"].dt.date <= date_to]
            saved_best = saved_best.sort_values("signal_score", ascending=False, na_position="last")
            saved_best = saved_best[[col for col in saved_cols if col in saved_best.columns]].drop_duplicates(
                subset=["signal_date", "ticker", "pattern_family"],
                keep="first",
            )

        df = breakout_df.merge(
            saved_best,
            on=["signal_date", "ticker", "pattern_family"],
            how="left",
            suffixes=("", "_saved"),
        )
        df["signal_score"] = pd.to_numeric(df.get("signal_score"), errors="coerce")
        df["recognised"] = df["signal_score"].ge(score_threshold).fillna(False)
        df["captured"] = df["signal_score"].notna()

    if df.empty:
        st.warning("No confirmed breakouts are available for the selected scope.")
        return

    # ── KPI summary ───────────────────────────────────────────────────────────
    total = len(df)
    recognised = int(df["recognised"].sum())
    missed = total - recognised
    coverage_pct = round(recognised / total * 100, 1) if total > 0 else 0.0
    captured_low_score = int((df["captured"] & ~df["recognised"]).sum())
    uncaptured = int((~df["captured"]).sum())

    _render_backtest_kpi_cards([
        {"label": "Total Breakouts", "value": str(total), "tone": "neutral"},
        {"label": f"Recognised (\u2265{score_threshold})", "value": str(recognised), "tone": "positive"},
        {"label": "Missed", "value": str(missed), "tone": "warning" if missed > 0 else "positive"},
        {"label": "Coverage", "value": f"{coverage_pct}%",
         "tone": "positive" if coverage_pct >= 70 else "warning"},
    ], columns_per_row=4)
    st.caption(
        f"Breakout = a raw detector event that later reached +{float(target_return_pct):.1f}% within {int(forward_days)} trading days. "
        f"{captured_low_score} were captured but scored below the recognition threshold, {uncaptured} were never saved as signals, "
        f"and {pending_count} recent raw candidates are excluded because they do not yet have the full forward window."
    )

    # ── Monthly stacked bar + Family breakdown ─────────────────────────────
    df["ym"] = df["signal_date"].dt.to_period("M").astype(str)
    monthly_rec = df[df["recognised"]].groupby("ym").size()
    monthly_miss_grp = df[~df["recognised"]].groupby("ym").size()
    all_months = sorted(set(monthly_rec.index) | set(monthly_miss_grp.index))

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        name=f"Recognised (\u2265{score_threshold})",
        x=all_months,
        y=[int(monthly_rec.get(m, 0)) for m in all_months],
        marker_color="#22c55e",
    ))
    fig_monthly.add_trace(go.Bar(
        name="Missed",
        x=all_months,
        y=[int(monthly_miss_grp.get(m, 0)) for m in all_months],
        marker_color="#f97316",
    ))
    fig_monthly.update_layout(
        barmode="stack", title="Coverage by Month", height=280,
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )

    fam_stats = (
        df.groupby("pattern_family")
        .apply(lambda g: pd.Series({
            "total": len(g),
            "recognised": int(g["recognised"].sum()),
        }))
        .reset_index()
    )
    fam_stats["missed_n"] = fam_stats["total"] - fam_stats["recognised"]
    fam_stats = fam_stats.sort_values("recognised", ascending=True)

    fig_family = go.Figure()
    fig_family.add_trace(go.Bar(
        name="Recognised", x=fam_stats["recognised"], y=fam_stats["pattern_family"],
        orientation="h", marker_color="#22c55e",
    ))
    fig_family.add_trace(go.Bar(
        name="Missed", x=fam_stats["missed_n"], y=fam_stats["pattern_family"],
        orientation="h", marker_color="#f97316",
    ))
    fig_family.update_layout(
        barmode="stack", title="Coverage by Pattern Family", height=280,
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )

    _ch1, _ch2 = st.columns(2)
    with _ch1:
        st.plotly_chart(fig_monthly, use_container_width=True)
    with _ch2:
        st.plotly_chart(fig_family, use_container_width=True)

    # ── Miss Diagnosis ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Miss Diagnosis")
    st.caption(
        "Misses are analysed only after a raw event has already proven itself by hitting the target return. "
        "If no saved signal exists for that breakout, it is labelled **Not Captured**. Otherwise the **primary drag** is whichever score component fell furthest below a healthy level (70/100), scaled by that component's weight. "
        "The bar chart counts how many missed breakouts each factor was responsible for. The comparison chart uses only rows that were actually saved as signals."
    )
    missed_df = df[~df["recognised"]].copy()

    if missed_df.empty:
        st.success("All signals in this date range met the recognition threshold! \U0001f389")
        return

    _SCORE_COLS = {
        "score_trend": ("Weak Trend", 0.20),
        "score_setup": ("Weak Setup", 0.20),
        "score_volume": ("Low Volume", 0.13),
        "score_risk": ("Wide Stop", 0.14),
        "score_rsi": ("RSI Drag", 0.03),
    }
    avail_cols = {k: v for k, v in _SCORE_COLS.items() if k in missed_df.columns}

    _GOOD_BASELINE = 70.0  # score below this is considered "dragging"

    def _dominant_miss(row: pd.Series) -> str:
        """Return the component label with the largest weighted deficit below the good baseline."""
        if pd.isna(pd.to_numeric(row.get("signal_score"), errors="coerce")):
            return "Not Captured"
        deficit: dict[str, float] = {}
        for col, (label, weight) in avail_cols.items():
            val = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(val):
                d = weight * max(0.0, _GOOD_BASELINE - float(val))
                if d > 0:
                    deficit[label] = d
        pattern_bonus_val = pd.to_numeric(row.get("pattern_bonus"), errors="coerce")
        if pd.notna(pattern_bonus_val) and float(pattern_bonus_val) < 0:
            deficit["Pattern Drag"] = abs(float(pattern_bonus_val))
        if not deficit:
            return "Near Miss"
        return max(deficit, key=lambda k: deficit[k])

    missed_df = missed_df.copy()
    missed_df["dominant_miss_reason"] = missed_df.apply(_dominant_miss, axis=1)

    reason_counts = missed_df["dominant_miss_reason"].value_counts()
    fig_reasons = go.Figure(go.Bar(
        x=reason_counts.values, y=reason_counts.index,
        orientation="h", marker_color="#f97316",
        text=reason_counts.values, textposition="auto",
    ))
    fig_reasons.update_layout(
        title="Primary Drag — Signal Count by Factor",
        height=max(180, 44 * len(reason_counts) + 80),
        margin=dict(l=0, r=16, t=36, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="# missed signals"),
    )

    # ── Grouped bar: median score per component, recognised vs missed ─────────
    scored_df = df[df["signal_score"].notna()].copy()
    grp_rec = scored_df[scored_df["recognised"]]
    grp_miss = scored_df[~scored_df["recognised"]]
    comp_labels = [label for _, (label, _) in avail_cols.items()]
    rec_medians = [
        pd.to_numeric(grp_rec[col], errors="coerce").median()
        for col in avail_cols
    ]
    miss_medians = [
        pd.to_numeric(grp_miss[col], errors="coerce").median()
        for col in avail_cols
    ]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Recognised", x=comp_labels, y=rec_medians,
        marker_color="#22c55e", opacity=0.85,
    ))
    fig_comp.add_trace(go.Bar(
        name="Missed", x=comp_labels, y=miss_medians,
        marker_color="#f97316", opacity=0.85,
    ))
    # reference line at the good-baseline
    fig_comp.add_shape(
        type="line", x0=-0.5, x1=len(comp_labels) - 0.5,
        y0=_GOOD_BASELINE, y1=_GOOD_BASELINE,
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
    )
    fig_comp.add_annotation(
        x=len(comp_labels) - 0.5, y=_GOOD_BASELINE,
        text="target 70", showarrow=False,
        font=dict(size=10, color="#94a3b8"), xanchor="right", yanchor="bottom",
    )
    fig_comp.update_layout(
        title="Median Component Scores: Recognised vs Missed",
        barmode="group",
        height=280, margin=dict(l=0, r=0, t=36, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="median score", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    _diag1, _diag2 = st.columns([1, 1.8])
    with _diag1:
        st.plotly_chart(fig_reasons, use_container_width=True)
    with _diag2:
        if scored_df.empty or not comp_labels:
            st.info("No saved-signal score components are available for the missed breakouts in this view.")
        else:
            st.plotly_chart(fig_comp, use_container_width=True)

    # ── Drill-down Table + Chart ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Missed Signal Drill-down")

    _tbl_c1, _tbl_c2, _tbl_c3 = st.columns([1, 1, 1.5])
    with _tbl_c1:
        all_reasons = ["All"] + sorted(missed_df["dominant_miss_reason"].unique())
        filter_reason = st.selectbox("Miss reason", all_reasons, key="cov_reason_filter")
    with _tbl_c2:
        filter_ticker = st.text_input(
            "Ticker filter", placeholder="e.g. RELIANCE", key="cov_ticker_filter",
        ).strip().upper()
    with _tbl_c3:
        drill_families = sorted(missed_df["pattern_family"].unique())
        filter_family = st.multiselect(
            "Family", options=drill_families, default=drill_families, key="cov_drill_family",
        )

    view = missed_df.copy()
    if filter_reason != "All":
        view = view[view["dominant_miss_reason"] == filter_reason]
    if filter_ticker:
        view = view[view["ticker"].astype(str).str.upper().str.contains(filter_ticker, regex=False)]
    if filter_family:
        view = view[view["pattern_family"].isin(filter_family)]

    _display_cols = [c for c in
        [
            "signal_date",
            "ticker",
            "pattern_family",
            "pattern",
            "entry_price",
            "target_price",
            "first_target_hit_day",
            "signal_score",
        ]
        + list(avail_cols.keys()) + ["pattern_bonus", "dominant_miss_reason"]
        if c in view.columns]
    view_display = view[_display_cols].sort_values("signal_score", ascending=True).copy()
    view_display["signal_date"] = view_display["signal_date"].dt.date.astype(str)
    for _fc in view_display.select_dtypes("float64").columns:
        view_display[_fc] = view_display[_fc].round(1)

    _had_cov_sel = st.session_state.get("_cov_had_sel", False)
    if _had_cov_sel:
        _tbl_col, _chart_col = st.columns([3, 2])
    else:
        _tbl_col = st.container()
        _chart_col = None

    with _tbl_col:
        _sel_ev = st.dataframe(
            view_display, hide_index=True, height=420,
            on_select="rerun", selection_mode="single-row",
            key="cov_drill_table",
        )
        _sel_rows = (_sel_ev.selection.get("rows", []) if hasattr(_sel_ev, "selection") and _sel_ev.selection else [])

    if _sel_rows:
        st.session_state["_cov_had_sel"] = True
        _sel_row = view_display.iloc[_sel_rows[0]]
        if _chart_col:
            with _chart_col:
                st.markdown(f"**{_sel_row.get('ticker')}** \u2014 {_sel_row.get('signal_date')}")
                _score_parts = " \u00b7 ".join(
                    f"{lbl}: {_sel_row[col]}"
                    for col, (lbl, _) in avail_cols.items()
                    if col in _sel_row.index and pd.notna(_sel_row[col])
                )
                if _score_parts:
                    st.caption(_score_parts)
                render_chart(
                    _sel_row, prices,
                    signal_date=str(_sel_row.get("signal_date", "")),
                    chart_key=f"cov_{_sel_row.get('ticker')}_{_sel_row.get('signal_date', '')}",
                )
    else:
        st.session_state["_cov_had_sel"] = False

    # ── Deep Scanner (opt-in) ─────────────────────────────────────────────────
    st.markdown("---")
    with st.expander(
        "Deep scan: find Pattern A breakouts not captured in signal history",
        expanded=False,
    ):
        st.caption(
            "Re-runs Pattern A detection across the full price history and cross-references with the "
            "saved signals CSV to find breakout events that were never recorded as signals. "
            "May take 15\u201330 seconds on a full dataset."
        )
        if st.button("Run deep scan", key="cov_deep_scan_btn"):
            with st.spinner("Scanning full price history for Pattern A breakouts\u2026"):
                _raw = _scan_raw_pattern_a_breakouts(prices)
            if _raw.empty:
                st.warning("No Pattern A breakouts found in the price history.")
            else:
                _pa_saved = all_signals[all_signals["pattern_family"].astype(str) == "A"]
                _saved_keys = set(zip(
                    _pa_saved["signal_date"].astype(str),
                    _pa_saved["ticker"].astype(str),
                ))
                _raw = _raw.copy()
                _raw["_in_csv"] = _raw.apply(
                    lambda r: (str(r["signal_date"]), str(r["ticker"])) in _saved_keys, axis=1
                )
                _n_raw = len(_raw)
                _n_csv = int(_raw["_in_csv"].sum())
                _n_unseen = _n_raw - _n_csv
                _render_backtest_kpi_cards([
                    {"label": "Raw breakouts (all time)", "value": str(_n_raw), "tone": "neutral"},
                    {"label": "Captured in signals CSV", "value": str(_n_csv), "tone": "positive"},
                    {"label": "Not captured", "value": str(_n_unseen),
                     "tone": "warning" if _n_unseen > 0 else "positive"},
                    {"label": "Capture rate",
                     "value": f"{round(_n_csv / _n_raw * 100, 1)}%" if _n_raw else "N/A",
                     "tone": "positive"},
                ], columns_per_row=4)
                _unseen_df = _raw[~_raw["_in_csv"]].drop(columns=["_in_csv"])
                if not _unseen_df.empty:
                    st.markdown("**Uncaptured breakouts**")
                    st.dataframe(
                        _unseen_df.sort_values("signal_date", ascending=False).head(200),
                        hide_index=True, height=340, use_container_width=True,
                    )


def _render_backtest_lab_styles() -> None:
    st.markdown(
        "<style>"
        ".lab-hero {"
        "  border: 1px solid var(--border-color);"
        "  border-radius: 18px;"
        "  padding: 1rem 1.05rem 0.95rem;"
        "  background: linear-gradient(180deg, var(--surface-bg) 0%, var(--surface-soft) 100%);"
        "  box-shadow: var(--page-shadow);"
        "  margin-bottom: 0.85rem;"
        "}"
        ".lab-badge-row {"
        "  display: flex;"
        "  flex-wrap: wrap;"
        "  gap: 0.45rem;"
        "  margin-bottom: 0.7rem;"
        "}"
        ".lab-badge {"
        "  display: inline-flex;"
        "  align-items: center;"
        "  border-radius: 999px;"
        "  padding: 0.2rem 0.65rem;"
        "  font-size: 0.74rem;"
        "  font-weight: 700;"
        "  border: 1px solid transparent;"
        "}"
        ".lab-badge-blue { background: rgba(56, 189, 248, 0.15); color: #7dd3fc; border-color: rgba(56, 189, 248, 0.35); }"
        ".lab-badge-green { background: rgba(34, 197, 94, 0.16); color: #86efac; border-color: rgba(34, 197, 94, 0.35); }"
        ".lab-badge-amber { background: rgba(245, 158, 11, 0.16); color: #fcd34d; border-color: rgba(245, 158, 11, 0.35); }"
        ".lab-badge-slate { background: var(--surface-alt); color: var(--text-primary); border-color: var(--widget-border); }"
        ".lab-hero-title {"
        "  font-size: 1.05rem;"
        "  font-weight: 800;"
        "  color: var(--heading-color);"
        "  margin-bottom: 0.2rem;"
        "}"
        ".lab-hero-copy {"
        "  font-size: 0.84rem;"
        "  color: var(--text-muted);"
        "  line-height: 1.45;"
        "  margin-bottom: 0.15rem;"
        "}"
        ".lab-kpi-card {"
        "  border: 1px solid var(--border-color);"
        "  border-radius: 14px;"
        "  padding: 0.8rem 0.85rem;"
        "  background: var(--surface-bg);"
        "  box-shadow: var(--panel-shadow);"
        "  min-height: 104px;"
        "  margin-bottom: 0.55rem;"
        "}"
        ".lab-kpi-card-positive { border-color: var(--tone-pos-border); background: var(--tone-pos-bg); }"
        ".lab-kpi-card-warning { border-color: var(--tone-warn-border); background: var(--tone-warn-bg); }"
        ".lab-kpi-card-negative { border-color: var(--tone-neg-border); background: var(--tone-neg-bg); }"
        ".lab-kpi-label { font-size: 0.75rem; color: var(--text-soft); font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }"
        ".lab-kpi-value { font-size: 1.45rem; color: var(--heading-color); font-weight: 800; margin-top: 0.2rem; line-height: 1.1; }"
        ".lab-kpi-delta { font-size: 0.78rem; font-weight: 700; margin-top: 0.3rem; }"
        ".lab-kpi-delta-positive { color: #059669; }"
        ".lab-kpi-delta-negative { color: #dc2626; }"
        ".lab-kpi-delta-neutral { color: var(--text-muted); }"
        ".lab-section-note { font-size: 0.82rem; color: var(--text-soft); margin-top: -0.1rem; margin-bottom: 0.6rem; }"
        ".lab-compact-panel { border: 1px solid var(--border-color); border-radius: 12px; padding: 0.55rem 0.7rem 0.3rem; background: linear-gradient(180deg, var(--surface-bg) 0%, var(--surface-soft) 100%); box-shadow: var(--panel-shadow); margin-bottom: 0.35rem; }"
        ".lab-compact-title { font-size: 0.72rem; color: var(--heading-color); font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.08rem; }"
        ".lab-compact-copy { font-size: 0.72rem; color: var(--text-soft); line-height: 1.25; margin-bottom: 0.35rem; }"
        ".lab-kpi-strip { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.7rem; }"
        ".lab-kpi-chip { flex: 1 1 110px; border: 1px solid var(--border-color); border-radius: 10px; padding: 0.5rem 0.7rem 0.45rem; background: var(--surface-bg); position: relative; overflow: hidden; box-shadow: var(--panel-shadow); }"
        ".lab-kpi-chip::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2.5px; background: var(--border-soft); border-radius: 10px 10px 0 0; }"
        ".lab-kpi-chip-positive { border-color: var(--tone-pos-border); background: var(--tone-pos-bg); }"
        ".lab-kpi-chip-positive::before { background: #22c55e; }"
        ".lab-kpi-chip-warning { border-color: var(--tone-warn-border); background: var(--tone-warn-bg); }"
        ".lab-kpi-chip-warning::before { background: #f59e0b; }"
        ".lab-kpi-chip-negative { border-color: var(--tone-neg-border); background: var(--tone-neg-bg); }"
        ".lab-kpi-chip-negative::before { background: #ef4444; }"
        ".lab-kpi-chip-label { font-size: 0.62rem; color: var(--text-soft); font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.15rem; }"
        ".lab-kpi-chip-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; color: var(--heading-color); font-weight: 800; line-height: 1.1; }"
        ".lab-kpi-chip-delta { font-size: 0.7rem; font-weight: 700; margin-top: 0.12rem; }"
        ".lab-setup-rail-cap { height: 3px; background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%); border-radius: 3px; margin-bottom: 0.75rem; }"
        ".lab-setup-section-divider { border: none; border-top: 1px solid var(--border-soft); margin: 0.55rem 0 0.5rem; }"
        "[data-testid='column']:has(.lab-setup-rail-cap) [data-testid='stVerticalBlock'] { gap: 0 !important; }"
        "[data-testid='column']:has(.lab-setup-rail-cap) [data-testid='stMarkdownContainer'] { margin-top: 0.55rem !important; margin-bottom: 0px !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] label { font-size: 0.65rem !important; margin-bottom: 1px !important; line-height: 1.2 !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] .stCaption p { font-size: 0.62rem !important; margin-top: -2px !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stNumberInput'] { margin-bottom: -4px !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stNumberInput'] input { font-size: 0.70rem !important; padding-top: 3px !important; padding-bottom: 3px !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stSelectbox'] { margin-bottom: -4px !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-baseweb='select'] { font-size: 0.70rem !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-baseweb='select'] [data-testid='stMarkdown'] p { font-size: 0.70rem !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stButton'] button { font-size: 0.68rem !important; padding: 2px 6px !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stToggle'] p { font-size: 0.65rem !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stMultiSelect'] { margin-bottom: -4px !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stMultiSelect'] [data-baseweb='tag'] { font-size: 0.58rem !important; padding: 0px 3px !important; height: auto !important; line-height: 1.2 !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stMultiSelect'] [data-baseweb='tag'] span { font-size: 0.58rem !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stMultiSelect'] input { font-size: 0.62rem !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stTextInput'] { margin-bottom: -4px !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stTextInput'] input { font-size: 0.70rem !important; padding-top: 3px !important; padding-bottom: 3px !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stCheckbox'] { margin-bottom: -4px !important; }"
        "[data-testid='element-container']:has(.lab-setup-rail-cap) ~ [data-testid='element-container'] [data-testid='stToggle'] { margin-bottom: -2px !important; }"
        "</style>",
        unsafe_allow_html=True,
    )


def _render_summary_kpi_strip(metrics: list[dict]) -> None:
    """Compact flex-wrap chip strip for the main summary KPIs."""
    if not metrics:
        return
    chips: list[str] = []
    for m in metrics:
        tone = str(m.get("tone", "neutral"))
        tone_cls = {
            "positive": " lab-kpi-chip-positive",
            "warning": " lab-kpi-chip-warning",
            "negative": " lab-kpi-chip-negative",
        }.get(tone, "")
        delta = str(m.get("delta", "")).strip()
        delta_cls = (
            "lab-kpi-delta-positive" if delta.startswith("+")
            else ("lab-kpi-delta-negative" if delta.startswith("-")
                  else "lab-kpi-delta-neutral")
        )
        delta_html = f"<div class='lab-kpi-chip-delta {delta_cls}'>{delta}</div>" if delta else ""
        chips.append(
            f"<div class='lab-kpi-chip{tone_cls}'>"
            f"<div class='lab-kpi-chip-label'>{m.get('label', '')}</div>"
            f"<div class='lab-kpi-chip-value'>{m.get('value', '')}</div>"
            f"{delta_html}"
            f"</div>"
        )
    st.markdown(
        "<div class='lab-kpi-strip'>" + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


def _render_backtest_kpi_cards(metrics: list[dict], *, columns_per_row: int = 4) -> None:
    if not metrics:
        return

    columns_per_row = max(1, int(columns_per_row))
    for start in range(0, len(metrics), columns_per_row):
        row_metrics = metrics[start:start + columns_per_row]
        columns = st.columns(columns_per_row)
        for idx, metric in enumerate(row_metrics):
            tone = str(metric.get("tone", "neutral"))
            tone_class = {
                "positive": " lab-kpi-card-positive",
                "warning": " lab-kpi-card-warning",
                "negative": " lab-kpi-card-negative",
            }.get(tone, "")
            delta = str(metric.get("delta", "")).strip()
            delta_class = "lab-kpi-delta-neutral"
            if delta.startswith("+") or delta.startswith("up"):
                delta_class = "lab-kpi-delta-positive"
            elif delta.startswith("-"):
                delta_class = "lab-kpi-delta-negative"
            delta_html = f"<div class='lab-kpi-delta {delta_class}'>{delta}</div>" if delta else ""
            help_text = str(metric.get("help", "")).strip()
            columns[idx].markdown(
                f"<div class='lab-kpi-card{tone_class}'>"
                f"<div class='lab-kpi-label'>{metric.get('label', '')}</div>"
                f"<div class='lab-kpi-value'>{metric.get('value', '')}</div>"
                f"{delta_html}"
                "</div>",
                unsafe_allow_html=True,
            )
            if help_text:
                columns[idx].caption(help_text)

        for idx in range(len(row_metrics), columns_per_row):
            columns[idx].empty()


def _render_backtest_evaluation_controls(widget_prefix: str) -> None:
    current_mode = str(st.session_state.get("lab_evaluation_mode", "walk-forward"))
    if current_mode not in {"walk-forward", "holdout"}:
        current_mode = "walk-forward"
    current_hold_days = int(st.session_state.get("lab_eval_hold_days", 30) or 30)

    eval_col, hold_col, date_col = st.columns([1.0, 0.9, 1.1])
    with eval_col:
        selected_mode = st.selectbox(
            "Evaluation mode",
            options=["walk-forward", "holdout"],
            index=0 if current_mode == "walk-forward" else 1,
            key=f"{widget_prefix}_evaluation_mode",
            help="Walk-forward trains on prior months and tests the next unseen month. Holdout trains on all rows up to a cutoff and tests the rest.",
        )
    st.session_state["lab_evaluation_mode"] = str(selected_mode)

    with hold_col:
        selected_hold_days = st.number_input(
            "Evaluation hold days",
            min_value=5,
            max_value=60,
            value=int(current_hold_days),
            step=1,
            key=f"{widget_prefix}_eval_hold_days",
            help="Hold horizon used by the stop-risk evaluator labels and out-of-sample summaries.",
        )
    st.session_state["lab_eval_hold_days"] = int(selected_hold_days)

    if selected_mode == "holdout":
        with date_col:
            selected_date = st.date_input(
                "Train end date",
                value=_get_backtest_train_end_date(),
                format="YYYY-MM-DD",
                key=f"{widget_prefix}_train_end_date",
                help="Inclusive cutoff for holdout mode. Training uses rows on or before this date, and testing uses rows after it.",
            )
        st.session_state["lab_train_end_date"] = selected_date
        st.caption("Holdout uses a clean train-on-past, test-on-rest split based on the selected cutoff.")
    else:
        st.caption("Walk-forward trains on prior months and scores each next month as unseen out-of-sample data.")


@st.cache_data(show_spinner="Running stop-risk backtest evaluation...")
def _run_backtest_stop_risk_evaluation(
    evaluation_mode: str,
    train_end_date_iso: str,
    max_hold_days: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    from stock_triggers.scripts.evaluate_stop_risk_walk_forward import (
        CANDIDATE_SPECS,
        _compute_labels,
        _load_signals,
        evaluate_candidate,
    )

    if not PRICES_CSV.exists() or not SIGNALS_ALL_PATTERNS_CSV.exists():
        return {}, pd.DataFrame(), pd.DataFrame()

    prices_df = pd.read_csv(PRICES_CSV, parse_dates=["Date"])
    if prices_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    hold_days = int(max_hold_days)
    signals_df = _load_signals(SIGNALS_ALL_PATTERNS_CSV, prices_df, max_hold_days=hold_days)
    if signals_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    labels_df = _compute_labels(
        signals_df,
        prices_df,
        target_pct=6.0,
        stop_pct=7.0,
        max_hold_days=hold_days,
    )
    if labels_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    train_end_date = None
    if str(evaluation_mode) == "holdout":
        parsed = pd.to_datetime(train_end_date_iso, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Invalid holdout train end date: {train_end_date_iso}")
        train_end_date = parsed

    return evaluate_candidate(
        "scores_only",
        CANDIDATE_SPECS["scores_only"],
        signals_df,
        labels_df,
        prices_df,
        target_pct=6.0,
        stop_pct=7.0,
        max_hold_days=hold_days,
        breakout_days=40,
        recent_signal_lookback_days=5,
        min_train_rows=250,
        tail_quantile=0.2,
        evaluation_mode=str(evaluation_mode),
        train_end_date=train_end_date,
    )


def _render_backtest_stop_risk_results(widget_prefix: str) -> None:
    _render_backtest_lab_styles()
    evaluation_mode = str(st.session_state.get("lab_evaluation_mode", "walk-forward"))
    train_end_date = _get_backtest_train_end_date().isoformat()
    max_hold_days = int(st.session_state.get("lab_eval_hold_days", 30) or 30)

    try:
        summary, monthly_df, predictions_df = _run_backtest_stop_risk_evaluation(evaluation_mode, train_end_date, max_hold_days)
    except Exception as exc:
        st.warning(f"Could not run stop-risk evaluation: {exc}")
        return

    if not summary:
        st.info("Stop-risk evaluation data is not available yet.")
        return

    oos_rows = int(summary.get("oos_rows", 0) or 0)
    if oos_rows <= 0:
        if evaluation_mode == "holdout":
            st.info(f"No unseen test rows were available for holdout with train end date {train_end_date}.")
        else:
            st.info("No walk-forward evaluation rows were available with the current data window.")
        return

    mode_label = "Holdout" if evaluation_mode == "holdout" else "Walk-forward"
    scope_badge_class = "lab-badge-green" if evaluation_mode == "holdout" else "lab-badge-blue"
    st.markdown(
        "<div class='lab-hero'>"
        "<div class='lab-badge-row'>"
        f"<span class='lab-badge {scope_badge_class}'>{mode_label}</span>"
        f"<span class='lab-badge lab-badge-slate'>{max_hold_days}-day horizon</span>"
        "<span class='lab-badge lab-badge-amber'>Candidate: scores_only</span>"
        "</div>"
        "<div class='lab-hero-title'>Backtesting Snapshot</div>"
        f"<div class='lab-hero-copy'>Leakage-safe stop-risk evaluation is active for the current lab scope. Use the KPI layer to assess reliability first, then inspect individual records in the table below.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    hero_metrics = [
        {
            "label": "OOS rows",
            "value": f"{oos_rows}",
            "tone": "positive",
            "help": "Rows available in the out-of-sample evaluation window.",
        },
        {
            "label": "Months",
            "value": f"{int(summary.get('months', 0) or 0)}",
            "help": "Distinct unseen months represented in this evaluation scope.",
        },
        {
            "label": "Spearman",
            "value": f"{float(summary.get('spearman_stop_risk_vs_stop', float('nan'))):.3f}",
            "tone": "positive" if float(summary.get('spearman_stop_risk_vs_stop', 0.0) or 0.0) > 0 else "warning",
            "help": "Rank correlation between predicted stop risk and realized stop-before-target outcomes.",
        },
        {
            "label": "Lowest-risk 20%",
            "value": f"{100.0 * float(summary.get('low20_stop_rate', 0.0) or 0.0):.1f}%",
            "tone": "positive",
            "help": "Realized stop-hit rate among the lowest predicted-risk bucket.",
        },
        {
            "label": "Highest-risk 20%",
            "value": f"{100.0 * float(summary.get('high20_stop_rate', 0.0) or 0.0):.1f}%",
            "tone": "warning",
            "help": "Realized stop-hit rate among the highest predicted-risk bucket.",
        },
        {
            "label": "Risk bucket gap",
            "value": f"{100.0 * float(summary.get('stop_rate_gap_high20_minus_low20', 0.0) or 0.0):.1f} pp",
            "tone": "positive" if float(summary.get('stop_rate_gap_high20_minus_low20', 0.0) or 0.0) > 0 else "warning",
            "help": "How much stop-hit rates separate between high- and low-risk buckets.",
        },
    ]
    _render_backtest_kpi_cards(hero_metrics, columns_per_row=3)

    if evaluation_mode == "holdout":
        secondary_metrics = [
            {
                "label": "Train end date",
                "value": str(summary.get("train_end_date", train_end_date)),
                "help": "Inclusive holdout cutoff used to split train and test rows.",
            },
            {
                "label": "Train rows",
                "value": f"{int(summary.get('train_rows', 0) or 0)}",
                "help": "Rows used to fit the stop-risk model for this holdout run.",
            },
            {
                "label": "Test rows",
                "value": f"{int(summary.get('test_rows', 0) or 0)}",
                "help": "Rows scored after the holdout cutoff.",
            },
        ]
    else:
        secondary_metrics = [
            {
                "label": "Baseline high-score stop rate",
                "value": f"{100.0 * float(summary.get('baseline_top_stop_rate', 0.0) or 0.0):.1f}%",
                "help": "How the original heuristic score behaves in the highest-score bucket.",
            },
            {
                "label": "Baseline low-score stop rate",
                "value": f"{100.0 * float(summary.get('baseline_bottom_stop_rate', 0.0) or 0.0):.1f}%",
                "help": "How the original heuristic score behaves in the lowest-score bucket.",
            },
            {
                "label": "Tracker scope",
                "value": f"{int(summary.get('months', 0) or 0)} months",
                "help": "Unseen-month coverage currently feeding the record table scope.",
            },
        ]
    _render_backtest_kpi_cards(secondary_metrics, columns_per_row=3)

    if not monthly_df.empty:
        st.caption(f"Monthly evaluation summaries are available for {len(monthly_df)} months. Use the CSV downloads when you need the detailed slice-by-slice breakdown.")

    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button(
            "Download summary CSV",
            data=to_csv_bytes(pd.DataFrame([summary])),
            file_name=f"stop_risk_{evaluation_mode}_summary.csv",
            mime="text/csv",
            key=f"{widget_prefix}_download_stop_risk_summary",
        )
    with download_cols[1]:
        if not monthly_df.empty:
            st.download_button(
                "Download monthly CSV",
                data=to_csv_bytes(monthly_df),
                file_name=f"stop_risk_{evaluation_mode}_monthly.csv",
                mime="text/csv",
                key=f"{widget_prefix}_download_stop_risk_monthly",
            )
    with download_cols[2]:
        if not predictions_df.empty:
            st.download_button(
                "Download predictions CSV",
                data=to_csv_bytes(predictions_df),
                file_name=f"stop_risk_{evaluation_mode}_predictions.csv",
                mime="text/csv",
                key=f"{widget_prefix}_download_stop_risk_predictions",
            )


def _filter_lab_signals_for_evaluation_window(signals_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if signals_df.empty:
        return signals_df.copy(), None

    evaluation_mode = str(st.session_state.get("lab_evaluation_mode", "walk-forward"))
    train_end_date = _get_backtest_train_end_date().isoformat()
    max_hold_days = int(st.session_state.get("lab_eval_hold_days", 30) or 30)

    try:
        summary, _, predictions_df = _run_backtest_stop_risk_evaluation(evaluation_mode, train_end_date, max_hold_days)
    except Exception as exc:
        return signals_df.copy(), f"Tracker scope fallback: stop-risk evaluation could not run ({exc})."

    if not summary or predictions_df.empty or int(summary.get("oos_rows", 0) or 0) <= 0:
        if evaluation_mode == "holdout":
            return signals_df.iloc[0:0].copy(), f"Tracker scope: no holdout test rows available after {train_end_date}."
        return signals_df.iloc[0:0].copy(), "Tracker scope: no walk-forward out-of-sample rows available."

    merge_keys = ["ticker", "signal_date", "pattern_family"]
    base_signals = signals_df.copy()
    if "pattern_family" not in base_signals.columns:
        base_signals["pattern_family"] = "A"
    if "pattern" in signals_df.columns and "pattern" in predictions_df.columns:
        merge_keys.append("pattern")

    predictions_view = predictions_df.copy()
    predictions_view["signal_date"] = pd.to_datetime(predictions_view["signal_date"], errors="coerce")
    predictions_view = predictions_view.dropna(subset=["signal_date", "ticker", "pattern_family"]).copy()

    prediction_columns = merge_keys + [
        col
        for col in ["month", "signal_stop_risk", "signal_reliability_score", "stop_before_target"]
        if col in predictions_view.columns
    ]
    prediction_keys = predictions_view[prediction_columns].drop_duplicates(subset=merge_keys)

    scoped = base_signals.copy()
    scoped["signal_date"] = pd.to_datetime(scoped["signal_date"], errors="coerce")
    scoped = scoped.dropna(subset=["signal_date", "ticker", "pattern_family"]).copy()
    scoped = scoped.merge(prediction_keys, on=merge_keys, how="inner")

    if evaluation_mode == "holdout":
        note = (
            f"Tracker scope: holdout test rows after {summary.get('train_end_date', train_end_date)} "
            f"({len(scoped)} trades shown)."
        )
    else:
        note = f"Tracker scope: walk-forward unseen rows across {int(summary.get('months', 0) or 0)} months ({len(scoped)} trades shown)."
    return scoped, note


def _apply_tomorrow_score_method(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df

    out = rows_df.copy()
    method = _get_tomorrow_score_method()
    metric_column = str(method["column"])

    if "ui_score" not in out.columns:
        out["ui_score"] = pd.to_numeric(out.get("signal_score"), errors="coerce")
    out["signal_reliability_score"] = pd.to_numeric(out.get("signal_reliability_score"), errors="coerce")
    out["signal_stop_risk"] = pd.to_numeric(out.get("signal_stop_risk"), errors="coerce")

    preferred = pd.to_numeric(out.get(metric_column), errors="coerce")
    heuristic = pd.to_numeric(out.get("ui_score"), errors="coerce")
    out["selected_score_value"] = preferred.where(preferred.notna(), heuristic)
    out["selected_score_label"] = str(method["label"])
    out["selected_score_short_label"] = str(method["short_label"])
    out["selected_score_higher_is_better"] = bool(method["higher_is_better"])
    out["selected_score_display_value"] = out["selected_score_value"] * float(method["display_scale"])
    out["selected_score_display_suffix"] = str(method["display_suffix"])
    out["selected_score_source_column"] = metric_column
    return out


def _render_pattern_hit_summary(view_df: pd.DataFrame) -> None:
    if view_df is None or view_df.empty or "status" not in view_df.columns:
        return

    hits = view_df[view_df["status"].astype(str) == "Target Hit ✅"].copy()
    if hits.empty:
        st.caption("Pattern hit summary: no target hits in the current view.")
        return

    if "pattern" in hits.columns:
        labels = hits["pattern"].astype(str).map(_format_pattern_name)
    elif "pattern_family" in hits.columns:
        labels = hits["pattern_family"].astype(str).map(lambda value: f"Pattern {value.strip().upper()}" if value and value.strip() else "Unknown")
    else:
        return

    counts = labels.value_counts()
    summary_text = " | ".join(f"{label}: {int(count)}" for label, count in counts.items())
    st.caption(f"Pattern hit summary: {summary_text}")


_LAB_PATTERN_OPTIONS = [
    ("A", "Breakout"),
    ("B", "Pullback rebound"),
    ("C", "MACD crossover"),
    ("D", "RSI bounce"),
    ("E", "BB squeeze"),
    ("F", "VWAP reclaim"),
    ("G", "VCP breakout"),
]


@st.cache_data(show_spinner="Building Backtesting Lab signal history...")
def _build_lab_history_signals(
    prices_df: pd.DataFrame,
    *,
    use_pattern_a: bool,
    use_pattern_b: bool,
    use_pattern_c: bool,
    use_pattern_d: bool,
    use_pattern_e: bool,
    use_pattern_f: bool,
    use_pattern_g: bool,
    breakout_days: int = 40,
    volume_multiplier: float = 1.5,
    stop_pct: float = 7.0,
    pullback_buffer_pct: float = 1.5,
    rebound_min_pct: float = 0.2,
    consensus_bonus: float = 5.0,
) -> pd.DataFrame:
    if prices_df.empty:
        return pd.DataFrame()

    price_history = _exclude_benchmark_rows(prices_df)
    if price_history.empty:
        return pd.DataFrame()

    eligible_dates = sorted(pd.to_datetime(price_history["Date"], errors="coerce").dropna().dt.normalize().unique().tolist())
    if not eligible_dates:
        return pd.DataFrame()

    all_signals: list[pd.DataFrame] = []
    for signal_date in eligible_dates:
        hist_to_date = price_history[price_history["Date"] <= signal_date].copy()
        day_signals = compute_scored_signals_for_date(
            hist_to_date,
            as_of_date=pd.to_datetime(signal_date),
            breakout_days=int(breakout_days),
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            use_pattern_a=bool(use_pattern_a),
            use_pattern_b=bool(use_pattern_b),
            use_pattern_c=bool(use_pattern_c),
            use_pattern_d=bool(use_pattern_d),
            use_pattern_e=bool(use_pattern_e),
            use_pattern_f=bool(use_pattern_f),
            use_pattern_g=bool(use_pattern_g),
            pullback_buffer_pct=float(pullback_buffer_pct),
            rebound_min_pct=float(rebound_min_pct),
            min_signal_score=0.0,
            consensus_bonus=float(consensus_bonus),
        )
        if not day_signals.empty:
            all_signals.append(day_signals)

    if not all_signals:
        return pd.DataFrame()

    out = pd.concat(all_signals, ignore_index=True)
    out.sort_values(["signal_date", "ticker"], inplace=True)
    return out.reset_index(drop=True)


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

    # ── Candle-shape flags ──
    if prices_df is not None and not prices_df.empty:
        _tag_candle_shapes_fast(out, prices_df, ticker_col="ticker", date_col="signal_date")
        _tag_labels = {
            "candle_doji": "Doji",
            "candle_hammer": "Hammer",
            "candle_marubozu": "Bullish Marubozu",
            "candle_confirmed_hammer_a": "Confirmed Hammer + Pattern A",
            "candle_morning_star": "Morning Star",
            "candle_engulfing": "Engulfing",
            "candle_engulfing_trend_combo": "Engulf A/C/G",
            "candle_harami": "Harami",
            "candle_piercing_line": "Piercing Line",
            "candle_piercing_variant": "Piercing Variant",
            "candle_piercing_variant_b_combo": "Pierce V+B",
            "candle_inverted_hammer": "Inverted Hammer",
            "candle_belt_hold": "Belt Hold",
            "candle_three_white_soldiers": "Three White Soldiers",
        }
        for idx in out.index:
            for col, tag_label in _tag_labels.items():
                if out.at[idx, col]:
                    existing = out.at[idx, "tags"]
                    if isinstance(existing, list):
                        if tag_label not in existing:
                            existing.append(tag_label)
                    else:
                        out.at[idx, "tags"] = [tag_label]
    else:
        for c in (
            "candle_doji", "candle_hammer", "candle_marubozu", "candle_morning_star", "candle_engulfing",
            "candle_engulfing_trend_combo",
            "candle_harami", "candle_piercing_line", "candle_piercing_variant", "candle_inverted_hammer",
            "candle_piercing_variant_b_combo",
            "candle_belt_hold", "candle_three_white_soldiers", "candle_confirmed_hammer_a",
        ):
            out[c] = False

    return out


def _latest_signal_timestamp(signals_df: pd.DataFrame) -> pd.Timestamp | None:
    if signals_df.empty or "signal_date" not in signals_df.columns:
        return None

    latest = pd.to_datetime(signals_df["signal_date"], errors="coerce").dropna().max()
    if pd.isna(latest):
        return None
    return pd.to_datetime(latest).normalize()


def _select_tomorrow_signal_source(
    pattern_a_df: pd.DataFrame,
    all_pattern_df: pd.DataFrame,
) -> pd.DataFrame:
    pattern_a_latest = _latest_signal_timestamp(pattern_a_df)
    all_pattern_latest = _latest_signal_timestamp(all_pattern_df)

    if all_pattern_latest is None:
        return pattern_a_df
    if pattern_a_latest is None:
        return all_pattern_df
    if all_pattern_latest >= pattern_a_latest:
        return all_pattern_df
    return pattern_a_df


def _prepare_tomorrow_list(signals_df: pd.DataFrame, prices_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str | None]:
    if signals_df.empty:
        return pd.DataFrame(), None

    latest_signal_date = str(signals_df["signal_date"].max())
    base = signals_df[signals_df["signal_date"] == latest_signal_date].copy()
    if base.empty:
        return pd.DataFrame(), latest_signal_date

    return _decorate_stock_rows(base, prices_df), latest_signal_date


def _get_latest_market_date(prices_df: pd.DataFrame | None) -> pd.Timestamp | None:
    if prices_df is None or prices_df.empty:
        return None

    live_prices = _exclude_benchmark_rows(prices_df)
    if live_prices.empty:
        return None

    latest_market_date = pd.to_datetime(live_prices.get("Date"), errors="coerce").dropna().max()
    if pd.isna(latest_market_date):
        return None

    return pd.to_datetime(latest_market_date).normalize()


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


def _build_tomorrow_empty_note(
    recent_candidates_df: pd.DataFrame,
    *,
    latest_signal_date: str | None,
    min_score: float,
    recent_days: int = 7,
) -> str:
    threshold_label = f"{float(min_score):.0f}"
    if recent_candidates_df is None or recent_candidates_df.empty:
        if latest_signal_date:
            return (
                f"No live tomorrow picks from Patterns A-G on the latest market date ({latest_signal_date}) "
                f"and none in the last {int(recent_days)} days above {threshold_label}."
            )
        return f"No live tomorrow picks from Patterns A-G in the last {int(recent_days)} days above {threshold_label}."

    recent = recent_candidates_df.copy()
    recent["signal_score"] = pd.to_numeric(recent.get("signal_score"), errors="coerce")
    recent["signal_date_dt"] = pd.to_datetime(recent.get("signal_date"), errors="coerce")
    recent = recent[recent["signal_score"].notna()].copy()
    if recent.empty:
        return f"No live tomorrow picks from Patterns A-G above {threshold_label} in the last {int(recent_days)} days."

    recent.sort_values(["signal_score", "signal_date_dt"], ascending=[False, False], inplace=True)
    best_row = recent.iloc[0]
    best_ticker = str(best_row.get("ticker", "")).strip() or "N/A"
    best_date = pd.to_datetime(best_row.get("signal_date_dt"), errors="coerce")
    best_date_label = best_date.date().isoformat() if pd.notna(best_date) else str(best_row.get("signal_date", "N/A"))
    best_score = float(best_row.get("signal_score", 0.0))
    best_pattern = _format_pattern_name(str(best_row.get("pattern", "")))
    return f"No live tomorrow picks from Patterns A-G above {threshold_label}. Best recent signal was {best_ticker} ({best_pattern}) on {best_date_label} at {best_score:.1f}."


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
            top: 3.15rem;
            z-index: 50;
            background: rgba(248, 251, 255, 0.94);
            backdrop-filter: blur(6px);
            border: 1px solid #dbe4ef;
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
        }
        @media (max-width: 720px) {
            .tomorrow-sticky {
                top: 7.25rem;
            }
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
        .chip-candle {
            color: #6b21a8;
            background: #f3e8ff;
            border-color: #d8b4fe;
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
                considered_str = f" | Universe: {considered_val}"
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
            f"<div class='tomorrow-sub'>Signal date in data: {latest_signal_date or '-'}{considered_str} | Active picks: {total_count}</div>"
            f"<div class='tomorrow-sub'>{price_dot}Price file updated: {price_status}</div>"
            f"<div class='tomorrow-sub'>{sig_dot}Signal file updated: {sig_status}</div>"
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

    render_heading_with_help(
        "Ranking controls",
        "scoring_method",
        key="tomorrow_ranking_controls_help",
        level=4,
        caption="Choose the score, threshold, and sort order that shape Tomorrow's Picks.",
    )

    method = _get_tomorrow_score_method()
    h1, h2 = st.columns([1.45, 0.95])
    with h1:
        st.selectbox(
            "Scoring method",
            options=list(TOMORROW_SCORE_METHODS.keys()),
            key="score_method",
            help="Select which ranking metric drives the main Tomorrow's Picks list. Use the linked question marks for the longer in-app explanation.",
        )
    with h2:
        render_theme_toggle_control()

    h3, h4 = st.columns([1.35, 1.05])
    with h3:
        st.slider(
            str(method["filter_label"]),
            min_value=0,
            max_value=100,
            value=int(st.session_state.get("min_score", int(method["default_filter"]))),
            step=1,
            key="min_score",
            help="Filter the Tomorrow's Picks list using the active ranking method.",
        )
    with h4:
        st.selectbox(
            "Sort",
            options=["Selected method", "Trade risk", "Ticker (A to Z)"],
            key="sort_by",
            help="Change the display order of the filtered Tomorrow's Picks list.",
        )


def render_stock_card(row: pd.Series, *, selected: bool) -> bool:
    ticker = str(row.get("ticker", ""))
    score = float(row.get("selected_score_display_value", row.get("signal_score", 0.0)) or 0.0)
    score_label = str(row.get("selected_score_short_label", "Score"))
    score_suffix = str(row.get("selected_score_display_suffix", ""))
    higher_is_better = bool(row.get("selected_score_higher_is_better", True))
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
            # Candle-shape tags
            if t in {"doji", "hammer", "bullish marubozu", "confirmed hammer + pattern a", "morning star", "engulfing", "engulf a/c/g", "harami", "piercing line", "piercing variant", "pierce v+b", "inverted hammer", "belt hold", "three white soldiers"}:
                return "chip chip-candle"
            return "chip"

        chips = "".join([f"<span class='{_chip_class(t)}'>{t}</span>" for t in tags])
    else:
        chips = ""

    # Score color based on value
    if higher_is_better:
        if score >= 70:
            _sc_color = "#059669"
        elif score >= 45:
            _sc_color = "#d97706"
        else:
            _sc_color = "#dc2626"
    else:
        if score <= 20:
            _sc_color = "#059669"
        elif score <= 40:
            _sc_color = "#d97706"
        else:
            _sc_color = "#dc2626"

    card_css = "stock-card-meta stock-card-meta-selected" if selected else "stock-card-meta"
    st.markdown(
        (
            f"<div class='{card_css}'>"
            f"<div><strong>{ticker}</strong> | {pattern_simple}</div>"
            f"<div class='stock-card-line'>Recommended {recommended_date}</div>"
            f"<div class='stock-card-line'>"
            f"{score_label} <span style='font-weight:800; color:{_sc_color}; font-size:0.9rem;'>{score:.0f}{score_suffix}</span>"
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
    sort_col = "score_100" if "score_100" in scores_df.columns else "score"
    if sort_col in scores_df.columns:
        scores_df["_sort"] = pd.to_numeric(scores_df[sort_col], errors="coerce")
        scores_df.sort_values(["_sort", "ticker"], ascending=[False, True], inplace=True)

    tiles_html = []
    for _, r in scores_df.iterrows():
        ticker = str(r.get("ticker", "")).replace(".NS", "")
        score_val = r.get("score_100", r.get("score"))
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
    render_heading_with_help(
        "Tomorrow's Picks",
        "tomorrow_picks",
        key="tomorrow_picks_section_help",
    )
    fallback_note = st.session_state.get("tomorrow_fallback_note")
    _generating = st.session_state.get("_header_generating", False)

    st.markdown(
        "<style>"
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

    top_cols = st.columns([3.7, 1, 1, 0.3])
    with top_cols[1]:
        show_scores = st.toggle("📊 All scores", key="show_all_scores", value=False)
    with top_cols[2]:
        pass
    with top_cols[3]:
        render_help_button("tag_uptrend", key="tag_chips_glossary_help", tooltip="What do the coloured chips mean? Click for tag glossary.")
    with top_cols[2]:
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

    if fallback_note:
        # Styled fallback banner
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
            "</style>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div class='fallback-bar'><span class='fallback-bar-text'>⚠️ {fallback_note}</span></div>",
            unsafe_allow_html=True,
        )

    if stocks_df.empty:
        st.info("No active picks for the latest market date.")
        return

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
    render_heading_with_help(
        "Overview",
        "overview_metrics",
        key=f"overview_help_{str(selected_row.get('ticker', 'na'))}",
        level=4,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entry", f"{float(selected_row.get('entry_price', 0.0)):.2f}")
    c2.metric("Stop", f"{float(selected_row.get('stop_price', 0.0)):.2f}")
    risk_pct = float(selected_row.get("risk_pct", 0.0)) if pd.notna(selected_row.get("risk_pct")) else 0.0
    c3.metric("Risk", f"{risk_pct:.2f}%")
    score_label = str(selected_row.get("selected_score_short_label", "Score"))
    score_value = float(selected_row.get("selected_score_display_value", selected_row.get("signal_score", 0.0)) or 0.0)
    score_suffix = str(selected_row.get("selected_score_display_suffix", ""))
    c4.metric(score_label, f"{score_value:.1f}{score_suffix}")
    st.caption(f"Why this is here: {selected_row.get('reason_short', '')}")


def render_quick_check(selected_row: pd.Series, prices_df: pd.DataFrame) -> dict[str, str]:
    render_heading_with_help(
        "Quick check",
        "quick_check",
        key=f"quick_check_help_{str(selected_row.get('ticker', 'na'))}",
    )
    checks = _quick_check_data(str(selected_row.get("ticker", "")), prices_df, selected_row)
    show_df = pd.DataFrame(
        [{"Item": k, "Status": v} for k, v in checks.items()]
    )
    render_table(
        show_df,
        height=250,
        column_help=table_help_map("quick_check", show_df.columns),
        table_help_title="Quick check",
        table_help_key_prefix=f"quick_check_cols_{str(selected_row.get('ticker', 'na'))}",
    )
    return checks


def _slice_chart_window(
    hist: pd.DataFrame,
    *,
    signal_date: str | None = None,
    exit_date: str | None = None,
    pre_bars: int = 90,
    post_bars: int = 45,
    tail_bars: int = 180,
) -> pd.DataFrame:
    """Return a chart window centered around signal/exit dates when available."""
    if hist.empty:
        return hist

    window = hist.sort_values("Date").reset_index(drop=True)
    focus_dt = pd.to_datetime(signal_date, errors="coerce") if signal_date else pd.NaT
    exit_dt = pd.to_datetime(exit_date, errors="coerce") if exit_date else pd.NaT
    if pd.isna(focus_dt):
        return window.tail(int(tail_bars)).copy()

    focus_pos = window[window["Date"] <= focus_dt]
    focus_idx = int(focus_pos.index[-1]) if not focus_pos.empty else 0
    start_idx = max(0, focus_idx - int(pre_bars))
    end_idx = min(len(window), focus_idx + int(post_bars) + 1)

    if pd.notna(exit_dt):
        exit_pos = window[window["Date"] <= exit_dt]
        if not exit_pos.empty:
            exit_idx = int(exit_pos.index[-1])
            end_idx = min(len(window), max(end_idx, exit_idx + 16))

    return window.iloc[start_idx:end_idx].copy()


def _apply_chart_range(
    hist: pd.DataFrame,
    *,
    range_mode: str,
    signal_date: str | None = None,
    exit_date: str | None = None,
) -> pd.DataFrame:
    """Apply a user-selected chart range, anchored to signal context when available."""
    if hist.empty:
        return hist

    window = hist.sort_values("Date").copy()
    latest_dt = pd.to_datetime(window["Date"].max(), errors="coerce")
    signal_dt = pd.to_datetime(signal_date, errors="coerce") if signal_date else pd.NaT
    exit_dt = pd.to_datetime(exit_date, errors="coerce") if exit_date else pd.NaT

    if range_mode == "Full":
        return window
    if range_mode == "Around Signal":
        return _slice_chart_window(window, signal_date=signal_date, exit_date=exit_date)

    months_map = {"3M": 3, "6M": 6, "1Y": 12}
    months = months_map.get(range_mode)
    if months is None:
        return _slice_chart_window(window, signal_date=signal_date, exit_date=exit_date)

    anchor_dt = latest_dt
    if pd.notna(signal_dt):
        anchor_dt = exit_dt if pd.notna(exit_dt) else signal_dt + pd.Timedelta(days=30)
    start_dt = anchor_dt - pd.DateOffset(months=int(months))
    end_dt = anchor_dt + pd.Timedelta(days=7)
    ranged = window[(window["Date"] >= start_dt) & (window["Date"] <= end_dt)].copy()
    if ranged.empty:
        return window.tail(180).copy()
    return ranged


def render_chart(
    selected_row: pd.Series,
    prices_df: pd.DataFrame,
    *,
    signal_date: str | None = None,
    exit_date: str | None = None,
    chart_key: str | None = None,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ticker = str(selected_row.get("ticker", ""))
    t = prices_df[prices_df["Ticker"] == ticker].copy().sort_values("Date")
    if t.empty:
        st.info("No chart data for this stock.")
    else:
        t["SMA50"] = t["Close"].rolling(50).mean()
        t["SMA200"] = t["Close"].rolling(200).mean()
        range_options = ["3M", "6M", "1Y"]
        if signal_date and str(signal_date) not in ("", "nan", "None", "NaT"):
            range_options.append("Around Signal")
        range_options.append("Full")
        default_range = "Around Signal" if "Around Signal" in range_options else "6M"
        key_suffix = chart_key or f"{ticker}_{signal_date or 'latest'}"

        _ctrl_a, _ctrl_b = st.columns([2, 3])
        with _ctrl_a:
            selected_range = st.radio(
                "Chart range",
                options=range_options,
                index=range_options.index(default_range),
                horizontal=True,
                key=f"chart_range_{key_suffix}",
            )
        with _ctrl_b:
            _show_indicators = st.multiselect(
                "Indicators",
                options=["Bollinger Bands", "RSI", "MACD"],
                default=["Bollinger Bands", "RSI", "MACD"],
                key=f"chart_indicators_{key_suffix}",
                label_visibility="collapsed",
            )

        t = _apply_chart_range(t, range_mode=str(selected_range), signal_date=signal_date, exit_date=exit_date)

        # ── Compute indicators ────────────────────────────────────────────────
        _show_bb = "Bollinger Bands" in _show_indicators
        _show_rsi = "RSI" in _show_indicators
        _show_macd = "MACD" in _show_indicators

        # Bollinger Bands (20-period)
        t["BB_MID"] = t["Close"].rolling(20).mean()
        t["BB_STD"] = t["Close"].rolling(20).std()
        t["BB_UPPER"] = t["BB_MID"] + 2 * t["BB_STD"]
        t["BB_LOWER"] = t["BB_MID"] - 2 * t["BB_STD"]

        # RSI(14) via exponential smoothing
        _delta = t["Close"].diff()
        _gain = _delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        _loss = (-_delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        _loss_safe = _loss.replace(0, 1e-10)
        t["RSI14"] = 100 - (100 / (1 + _gain / _loss_safe))
        _rsi_ok = len(t) >= 14

        # MACD(12, 26, 9)
        t["EMA12"] = t["Close"].ewm(span=12, adjust=False).mean()
        t["EMA26"] = t["Close"].ewm(span=26, adjust=False).mean()
        t["MACD"] = t["EMA12"] - t["EMA26"]
        t["MACD_SIG"] = t["MACD"].ewm(span=9, adjust=False).mean()
        t["MACD_HIST"] = t["MACD"] - t["MACD_SIG"]
        _macd_ok = len(t) >= 26

        # ── Subplot layout ────────────────────────────────────────────────────
        _active_sub = (1 if _show_rsi else 0) + (1 if _show_macd else 0)
        if _active_sub == 2:
            _row_heights = [0.55, 0.15, 0.15, 0.15]
            _rows = 4
        elif _active_sub == 1:
            _row_heights = [0.60, 0.20, 0.20]
            _rows = 3
        else:
            _row_heights = [0.75, 0.25]
            _rows = 2

        _subplot_titles = ["Price"] + [""] * (_rows - 1)
        fig = make_subplots(
            rows=_rows, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=_row_heights,
            subplot_titles=_subplot_titles,
        )

        # ── Row 1: Candlestick + SMAs + BB ───────────────────────────────────
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

        if _show_bb:
            fig.add_trace(go.Scatter(
                x=t["Date"], y=t["BB_UPPER"], name="BB Upper",
                line=dict(color="#8b5cf6", width=1, dash="dot"),
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=t["Date"], y=t["BB_LOWER"], name="BB Lower",
                line=dict(color="#8b5cf6", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(139,92,246,0.06)",
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=t["Date"], y=t["BB_MID"], name="BB Mid",
                line=dict(color="#8b5cf6", width=0.8),
            ), row=1, col=1)

        # ── Row 2: Volume ─────────────────────────────────────────────────────
        colors = [
            "#22c55e" if c >= o else "#ef4444"
            for c, o in zip(t["Close"], t["Open"])
        ]
        fig.add_trace(go.Bar(
            x=t["Date"], y=t["Volume"], name="Volume",
            marker_color=colors, opacity=0.5,
        ), row=2, col=1)

        # ── Remaining rows: RSI then MACD ─────────────────────────────────────
        _next_row = 3
        if _show_rsi:
            if _rsi_ok:
                fig.add_trace(go.Scatter(
                    x=t["Date"], y=t["RSI14"], name="RSI(14)",
                    line=dict(color="#10b981", width=1.5),
                ), row=_next_row, col=1)
                fig.add_hline(y=70, line_width=0.8, line_dash="dash", line_color="#ef4444",
                              row=_next_row, col=1)
                fig.add_hline(y=50, line_width=0.6, line_dash="dot", line_color="#64748b",
                              row=_next_row, col=1)
                fig.add_hline(y=30, line_width=0.8, line_dash="dash", line_color="#22c55e",
                              row=_next_row, col=1)
                fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="rgba(239,68,68,0.06)",
                              row=_next_row, col=1)
                fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="rgba(34,197,94,0.06)",
                              row=_next_row, col=1)
                fig.update_yaxes(range=[0, 100], title_text="RSI", title_font_size=9,
                                 row=_next_row, col=1)
            else:
                fig.add_annotation(
                    x=0.5, y=0.5, xref="x domain", yref="y domain",
                    text="Not enough history for RSI",
                    showarrow=False, font=dict(color="#94a3b8", size=10),
                    row=_next_row, col=1,
                )
            _next_row += 1

        if _show_macd:
            if _macd_ok:
                hist_colors = [
                    "#22c55e" if v >= 0 else "#ef4444"
                    for v in t["MACD_HIST"].fillna(0)
                ]
                fig.add_trace(go.Bar(
                    x=t["Date"], y=t["MACD_HIST"], name="MACD Hist",
                    marker_color=hist_colors, opacity=0.6, showlegend=False,
                ), row=_next_row, col=1)
                fig.add_trace(go.Scatter(
                    x=t["Date"], y=t["MACD"], name="MACD",
                    line=dict(color="#38bdf8", width=1.5),
                ), row=_next_row, col=1)
                fig.add_trace(go.Scatter(
                    x=t["Date"], y=t["MACD_SIG"], name="Signal",
                    line=dict(color="#f59e0b", width=1.2),
                ), row=_next_row, col=1)
                fig.add_hline(y=0, line_width=0.8, line_dash="dot", line_color="#64748b",
                              row=_next_row, col=1)
                fig.update_yaxes(title_text="MACD", title_font_size=9,
                                 row=_next_row, col=1)
            else:
                fig.add_annotation(
                    x=0.5, y=0.5, xref="x domain", yref="y domain",
                    text="Not enough history for MACD",
                    showarrow=False, font=dict(color="#94a3b8", size=10),
                    row=_next_row, col=1,
                )

        # ── Vertical marker lines for signal/exit dates ───────────────────────
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

        _chart_height = 480 + (_active_sub * 100)
        _xaxis_kwargs = {f"xaxis{i}": dict(showgrid=False) for i in range(2, _rows + 1)}
        _yaxis_kwargs = {
            "yaxis": dict(showgrid=True, gridcolor="#1e293b"),
            **{f"yaxis{i}": dict(showgrid=True, gridcolor="#1e293b", zeroline=False) for i in range(2, _rows + 1)},
        }

        fig.update_layout(
            height=_chart_height,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            **_xaxis_kwargs,
            **_yaxis_kwargs,
        )

        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_past_results(selected_row: pd.Series, all_signals: pd.DataFrame, prices_df: pd.DataFrame) -> None:
    st.markdown("<div class='reveal-wrap'>", unsafe_allow_html=True)
    render_heading_with_help(
        "Past results",
        "past_results",
        key=f"past_results_help_{str(selected_row.get('ticker', 'na'))}",
    )
    ticker = str(selected_row.get("ticker", ""))
    hist = all_signals[all_signals["ticker"].astype(str) == ticker].copy().sort_values("signal_date")
    if hist.empty:
        st.info("No past rows for this stock.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.slider(
        "Hold days",
        min_value=5,
        max_value=60,
        step=1,
        key="hold_days",
        help="Controls how many forward trading days the quick past-results evaluator allows.",
    )
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
        render_table(
            view,
            height=240,
            column_help=table_help_map("past_results", view.columns),
            table_help_title="Past results",
            table_help_key_prefix=f"past_results_cols_{ticker}",
        )
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
        f"Heuristic score: {float(selected_row.get('signal_score', 0.0)):.1f}\n"
        f"Reliability score: {float(pd.to_numeric(selected_row.get('signal_reliability_score'), errors='coerce') if pd.notna(selected_row.get('signal_reliability_score')) else 0.0):.0f}\n"
        f"Stop risk: {float(pd.to_numeric(selected_row.get('signal_stop_risk'), errors='coerce') if pd.notna(selected_row.get('signal_stop_risk')) else 0.0) * 100.0:.1f}%"
    )
    if st.button("Send to Telegram", key=f"send_selected_{ticker}", disabled=not allow_actions):
        with st.spinner("Sending..."):
            ok, out = send_telegram_message(token, chat_id, msg)
        if ok:
            st.success("Sent.")
        else:
            st.error(out)


def render_score_breakdown(selected_row: pd.Series) -> None:
    render_heading_with_help(
        "Score breakdown",
        "score_breakdown",
        key=f"score_breakdown_help_{str(selected_row.get('ticker', 'na'))}",
        level=4,
    )
    selected_label = str(selected_row.get("selected_score_label", "Heuristic score"))
    selected_value = pd.to_numeric(selected_row.get("selected_score_display_value"), errors="coerce")
    selected_suffix = str(selected_row.get("selected_score_display_suffix", ""))
    if pd.notna(selected_value):
        st.markdown(f"- Active ranking metric: {selected_label} = {float(selected_value):.1f}{selected_suffix}")

    total_score = float(selected_row.get("signal_score", 0.0)) if pd.notna(selected_row.get("signal_score")) else 0.0
    trend = selected_row.get("score_trend")
    setup = selected_row.get("score_setup")
    volume = selected_row.get("score_volume")
    risk = selected_row.get("score_risk")
    rsi = selected_row.get("score_rsi")

    has_components = all(pd.notna(v) for v in [trend, setup, volume, risk])
    if not has_components:
        extra_lines = ["- Component scores are not available for this row.", f"- Heuristic score: {total_score:.1f}"]
        reliability = pd.to_numeric(selected_row.get("signal_reliability_score"), errors="coerce")
        stop_risk = pd.to_numeric(selected_row.get("signal_stop_risk"), errors="coerce")
        if pd.notna(reliability):
            extra_lines.append(f"- Reliability score: {float(reliability):.0f}")
        if pd.notna(stop_risk):
            extra_lines.append(f"- Stop risk: {float(stop_risk) * 100.0:.1f}%")
        st.markdown("\n".join(extra_lines))
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
    score_pattern = float(selected_row.get("score_pattern", 0.0) or 0.0) if pd.notna(selected_row.get("score_pattern")) else 0.0
    ma_slope_bonus = float(selected_row.get("ma_slope_bonus", 0.0) or 0.0) if pd.notna(selected_row.get("ma_slope_bonus")) else 0.0
    pattern_bonus = float(selected_row.get("pattern_bonus", 0.0) or 0.0) if pd.notna(selected_row.get("pattern_bonus")) else 0.0
    sma50_slope_pct = selected_row.get("sma50_slope_pct")

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

    if ma_slope_bonus > 0:
        running = round(running + ma_slope_bonus, 1)
        if pd.notna(sma50_slope_pct):
            lines.append(
                f"- SMA50 is sloping upward ({float(sma50_slope_pct):.2f}% over the last 5 bars), so MA slope adds +{ma_slope_bonus:.2f} as a score booster, running total {running:.1f}."
            )
        else:
            lines.append(
                f"- Moving-average slope is positive, so MA slope adds +{ma_slope_bonus:.2f} as a score booster, running total {running:.1f}."
            )
    elif pd.notna(sma50_slope_pct):
        lines.append(
            f"- SMA50 slope is {float(sma50_slope_pct):.2f}% over the last 5 bars, so no extra MA-slope bonus was added."
        )

    if pattern_bonus > 0 or score_pattern > 0:
        running = round(running + pattern_bonus, 1)
        pattern_family = str(selected_row.get("pattern_family", "") or selected_row.get("pattern", "Unknown")).strip()
        lines.append(
            f"- Pattern {pattern_family} carries a learned historical score of {score_pattern:.1f}/100, contributing +{pattern_bonus:.2f} out of the 30 pattern-weight points, running total {running:.1f}."
        )

    reliability = pd.to_numeric(selected_row.get("signal_reliability_score"), errors="coerce")
    stop_risk = pd.to_numeric(selected_row.get("signal_stop_risk"), errors="coerce")
    lines.append(f"- Heuristic score: {total_score:.1f}")
    if pd.notna(reliability):
        lines.append(f"- Reliability score: {float(reliability):.0f}")
    if pd.notna(stop_risk):
        lines.append(f"- Stop risk: {float(stop_risk) * 100.0:.1f}%")
    st.markdown("\n".join(lines))

    # Step 10: score component help chips
    _ticker_key = str(selected_row.get("ticker", "na"))
    _chip_cols = st.columns(7)
    _chip_defs = [
        ("Trend", "score_trend_comp"),
        ("Setup", "score_setup_comp"),
        ("Volume", "score_volume_comp"),
        ("Risk", "score_risk_comp"),
        ("RSI", "score_rsi_comp"),
        ("MA slope", "ma_slope_bonus"),
        ("Pattern", "pattern_family_bonus_formula"),
    ]
    for _ci, (_clabel, _ckey) in enumerate(_chip_defs):
        with _chip_cols[_ci]:
            render_help_button(_ckey, key=f"sbd_{_ckey}_{_ticker_key}", tooltip=_clabel)


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
    render_chart(selected_row, prices_df, chart_key=f"selected_stock_{ticker}")

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
        st.session_state["_nav_skip_sync"] = True
        st.rerun()

    # Step 8: Navigate to Lab pre-filtered for this ticker
    if st.button(
        "🔬 View all Lab signals for this ticker",
        key=f"tomorrow_nav_lab_{ticker}",
        help="Open the Backtesting Lab filtered to this ticker's trade history.",
    ):
        st.session_state["mode"] = "Backtest Lab"
        st.session_state["lab_prefill_ticker_filter"] = ticker
        st.session_state["_nav_skip_sync"] = True
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
    if not stocks_df.empty:
        stocks_df = _apply_tomorrow_score_method(stocks_df)

    min_score = min(100.0, max(0.0, float(st.session_state.get("min_score", 75))))
    score_method = _get_tomorrow_score_method()

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

    # Treat signals older than the latest available market date as stale.
    # This keeps holiday gaps and pre-market refreshes from hiding valid picks.
    stale_for_tomorrow = False
    latest_market_date = _get_latest_market_date(prices_df)
    if latest_signal_date:
        latest_dt = pd.to_datetime(latest_signal_date, errors="coerce")
        if pd.notna(latest_dt):
            if latest_market_date is not None:
                stale_for_tomorrow = latest_dt.normalize() < latest_market_date
            elif (date.today() - latest_dt.date()).days > 1:
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
        if stale_for_tomorrow:
            fallback_note = f"No active picks for today. Latest saved signals are from {latest_signal_date}."
        else:
            fallback_note = "No active picks for today."
        render_header(
            latest_signal_date=latest_signal_date,
            total_count=0,
            total_considered=total_considered,
            data_updated=data_updated,
            signals_generated=signals_generated,
            fallback_note=fallback_note,
        )
        render_pattern_bonus_expander()
        render_candle_enhancer_expander()
        st.session_state["tomorrow_fallback_note"] = fallback_note
        render_stock_list(pd.DataFrame())
        return
    else:
        selected_values = pd.to_numeric(stocks_df.get("selected_score_value"), errors="coerce")
        if bool(score_method["higher_is_better"]):
            stocks_df = stocks_df[selected_values >= min_score].copy()
            filter_note = f"above the minimum {score_method['label'].lower()} of {float(min_score):.0f}"
        else:
            stocks_df = stocks_df[(selected_values * float(score_method["display_scale"])) <= min_score].copy()
            filter_note = f"at or below the maximum {score_method['label'].lower()} of {float(min_score):.0f}%"
        if stocks_df.empty:
            fallback_note = f"No active picks for today {filter_note}."
            render_header(
                latest_signal_date=latest_signal_date,
                total_count=0,
                total_considered=total_considered,
                data_updated=data_updated,
                signals_generated=signals_generated,
                fallback_note=fallback_note,
            )
            render_pattern_bonus_expander()
            render_candle_enhancer_expander()
            st.session_state["tomorrow_fallback_note"] = fallback_note
            render_stock_list(pd.DataFrame())
            return

    # Re-render header with the correct total_count and optional fallback note.
    render_header(
        latest_signal_date=latest_signal_date,
        total_count=len(stocks_df),
        total_considered=total_considered,
        data_updated=data_updated,
        signals_generated=signals_generated,
        fallback_note=fallback_note,
    )
    render_pattern_bonus_expander()
    render_candle_enhancer_expander()
    # Store note for use directly above the Tomorrow's stock list section.
    st.session_state["tomorrow_fallback_note"] = fallback_note

    sort_by = str(st.session_state.get("sort_by", "Selected method"))
    if sort_by == "Trade risk":
        sort_cols = ["risk_pct"]
        asc = [True]
        if "selected_score_value" in stocks_df.columns:
            sort_cols.append("selected_score_value")
            asc.append(not bool(score_method["higher_is_better"]))
        sort_cols.append("ticker")
        asc.append(True)
        stocks_df.sort_values(sort_cols, ascending=asc, inplace=True)
    elif sort_by == "Ticker (A to Z)":
        stocks_df.sort_values(["ticker"], inplace=True)
    else:
        sort_cols = []
        asc = []
        if "selected_score_value" in stocks_df.columns:
            sort_cols.append("selected_score_value")
            asc.append(not bool(score_method["higher_is_better"]))
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


if st.session_state.get("mode") == "Documentation":
    _docs_header_spacer, _docs_theme_col = st.columns([4, 2])
    with _docs_theme_col:
        render_theme_toggle_control()
    render_documentation_page()
    st.stop()


signals = load_signals()
all_pattern_signals = load_all_pattern_signals()
sell_signals = load_sell_signals()
prices_all = load_prices()
benchmark_prices = _select_benchmark_rows(prices_all)
prices = _exclude_benchmark_rows(prices_all)

refresh_info = get_prices_refresh_info(prices)

# Single summary placeholder so refresh summary appears only once on page.
summary_panel = st.container()


def update_summary_panel(prices_df: pd.DataFrame, signals_df: pd.DataFrame) -> None:
    summary_panel.empty()
    with summary_panel:
        render_refresh_summary(prices_df, signals_df)


_init_tomorrow_ui_state()

today_str = date.today().isoformat()
last_refresh_date = st.session_state.get("last_refresh_date")

latest_trading_date_str = None
if not prices.empty:
    latest_trading_date_str = prices["Date"].max().date().isoformat()

# Keep tomorrow mode clean; legacy tabs remain available under other modes.
tomorrow_allow_actions = not IS_STREAMLIT_CLOUD
if st.session_state.get("mode") == "Tomorrow":
    render_whats_new_panel(context_label="Tomorrow's Picks")
    tomorrow_signals = _select_tomorrow_signal_source(signals, all_pattern_signals)
    render_tomorrow_screen(
        tomorrow_signals,
        prices,
        allow_actions=tomorrow_allow_actions,
        data_updated=refresh_info["file_updated"],
    )
    st.stop()

# Non-Tomorrow mode: set defaults
allow_actions = not IS_STREAMLIT_CLOUD

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
    render_whats_new_panel(context_label="Backtesting Lab")
    _lab_header_spacer, _lab_theme_col = st.columns([4, 2])
    with _lab_theme_col:
        render_theme_toggle_control()
    _render_backtest_lab_styles()
    _lab_summary_container = st.container()
    _lab_combo_container = st.container()
    _lab_charts_container = st.container()
    _lab_snapshot_container = st.container()
    with _lab_combo_container:
        _col_table, _col_setup = st.columns([3.0, 0.85])

    # --- Signal Performance Tracker ---
    if (not signals.empty or not all_pattern_signals.empty) and not prices.empty:
        with _col_setup:
            st.markdown("<div class='lab-setup-rail-cap'></div>", unsafe_allow_html=True)
            render_heading_with_help(
                "Analysis Setup",
                "analysis_setup",
                key="lab_analysis_setup_help",
            )
            st.markdown("<div class='lab-compact-panel'><div class='lab-compact-title'>Signal Scope</div></div>", unsafe_allow_html=True)
            _scope_action_a, _scope_action_b = st.columns(2)
            with _scope_action_a:
                _rescore_on = st.toggle(
                    "Recompute lab scores",
                    key="lab_rescore_toggle",
                    value=False,
                    help="Temporarily recalculate signal_score in the lab view using the current scoring logic. This does not update stock_scores.csv.",
                )
            with _scope_action_b:
                if st.button("Clear cache", key="lab_clear_cache_top", help="Clear cached Backtesting Lab results for this session."):
                    _clear_lab_session_cache()
                    st.rerun()

            _lab_pattern_labels = [f"{family} · {name}" for family, name in _LAB_PATTERN_OPTIONS]
            # Step 9: apply lab_family_filter from cross-nav (pattern chart click / docs button)
            _family_filter_prefill = st.session_state.pop("lab_family_filter", None)
            if _family_filter_prefill:
                _prefill_labels = [
                    lbl for lbl in _lab_pattern_labels
                    if lbl.split(" · ", 1)[0] in _family_filter_prefill
                ]
                if _prefill_labels:
                    st.session_state["lab_pattern_family_filter"] = _prefill_labels
            render_caption_with_help("Pattern families", "pattern_family", key="lab_pattern_families_help")
            _current_families = st.session_state.get("lab_pattern_family_filter", _lab_pattern_labels)
            _family_summary = f"{len(_current_families)} of {len(_lab_pattern_labels)} selected"
            with st.expander(_family_summary, expanded=False):
                _lab_pattern_selection = st.multiselect(
                    "Pattern families",
                    options=_lab_pattern_labels,
                    default=_lab_pattern_labels,
                    key="lab_pattern_family_filter",
                    label_visibility="collapsed",
                    help="The lab prefers saved signal history from signals_pattern_a.csv and signals_all_patterns.csv. It only rebuilds from price data if the all-pattern file is missing.",
                )

            st.markdown("<div class='lab-compact-panel'><div class='lab-compact-title'>Trade Rules</div></div>", unsafe_allow_html=True)
            render_caption_with_help("Stop mode", "stop_mode", key="lab_stop_mode_help")
            _lab_stop_mode = st.selectbox(
                "Stop mode",
                ["Structure + ATR", "ATR", "Fixed %", "Score >95 hold to target", "Score >90 hold to target"],
                index=0,
                key="lab_d_stop_mode",
                label_visibility="collapsed",
                help="Structure + ATR uses a recent swing low minus an ATR buffer, capped by Fixed stop %. ATR uses entry minus ATR multiple, also capped by Fixed stop %. Fixed % uses the legacy percent stop. Score >95 / >90 hold to target uses the fixed stop for risk display, but ignores stop exits on trades whose final signal score is above the threshold until target is hit.",
            )
            _tr_c1, _tr_c2 = st.columns(2)
            with _tr_c1:
                render_caption_with_help("Target %", "target_pct", key="lab_target_pct_help")
                _lab_tgt = st.number_input("Target %", min_value=1.0, max_value=50.0, value=6.0, step=0.5, key="lab_d_target", label_visibility="collapsed")
            with _tr_c2:
                render_caption_with_help("Stop %", "stop_mode", key="lab_stop_pct_help")
                _lab_stp = st.number_input("Stop %", min_value=1.0, max_value=50.0, value=9.0, step=0.5, key="lab_d_stop", label_visibility="collapsed")
            _tr_c3, _tr_c4 = st.columns(2)
            with _tr_c3:
                render_caption_with_help("₹ / trade", "capital_per_trade", key="lab_capital_help")
                _lab_cap = st.number_input("₹ / trade", min_value=1000.0, max_value=500000.0, value=10000.0, step=1000.0, key="lab_d_capital", label_visibility="collapsed")
            with _tr_c4:
                render_caption_with_help("Min score", "min_score_filter", key="lab_min_score_help")
                _lab_min_score = st.number_input("Min score", min_value=0, max_value=100, value=90, step=5, key="lab_d_min_score", label_visibility="collapsed")
            _tr_c5, _tr_c6 = st.columns(2)
            with _tr_c5:
                render_caption_with_help("ATR period", "stop_mode", key="lab_atr_period_help")
                _lab_atr_period = st.number_input("ATR period", min_value=5, max_value=50, value=14, step=1, key="lab_d_atr_period", label_visibility="collapsed")
            with _tr_c6:
                render_caption_with_help(
                    "ATR buffer" if _lab_stop_mode == "Structure + ATR" else "ATR x",
                    "atr_buffer",
                    key="lab_atr_buffer_help",
                )
                _lab_atr_mult = st.number_input(
                    "ATR buffer" if _lab_stop_mode == "Structure + ATR" else "ATR x",
                    min_value=0.1,
                    max_value=5.0,
                    value=2.5,
                    step=0.1,
                    key="lab_d_atr_mult",
                    label_visibility="collapsed",
                )
            render_caption_with_help("Max days held", "days_held", key="lab_max_days_help")
            _lab_max_days_held = st.number_input(
                "Max days held",
                min_value=1,
                max_value=365,
                value=60,
                step=1,
                key="lab_d_max_days_held",
                label_visibility="collapsed",
            )

            st.markdown("<div class='lab-compact-panel'><div class='lab-compact-title'>Record Filters</div></div>", unsafe_allow_html=True)
            _lab_filter_controls_container = st.container()

        _lab_pattern_keys = {label.split(" · ", 1)[0] for label in _lab_pattern_selection}
        if not _lab_pattern_keys:
            _lab_pattern_keys = {"A"}

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

        render_heading_with_help(
            "Model & Reference",
            "model_reference",
            key="lab_model_reference_help",
            caption="Evaluation settings affect the stop-risk snapshot and also define which out-of-sample records appear in the table.",
        )
        _render_backtest_evaluation_controls("lab_main")
        render_pattern_bonus_expander()
        render_candle_enhancer_expander()
        with st.expander("Advanced Scoring Inputs", expanded=False):
            st.caption("Keep these collapsed for day-to-day analysis. Open them when tuning score overlays or developer diagnostics.")
            _lab_c5, _lab_c6 = st.columns([1.0, 1.0])
            with _lab_c5:
                _lab_use_rs_bonus = st.checkbox(
                    "RS bonus",
                    value=False,
                    key="lab_use_rs_bonus",
                    help="Adds a small capped score bonus for stocks outperforming Nifty on RS20 and RS50.",
                )
            with _lab_c6:
                _lab_rs_bonus_max = st.number_input(
                    "RS cap",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.5,
                    format="%.1f",
                    key="lab_rs_bonus_max",
                    disabled=not _lab_use_rs_bonus,
                    help="Cap on the score bonus from stock-vs-Nifty relative strength.",
                )

            _lab_use_learned_candle_weights = st.checkbox(
                "Use learned family candle weights",
                value=True,
                key="lab_use_learned_candle_weights",
                help="Applies the learned candle contribution by pattern family, accounting for overlap and allowing negative candle effects where history supports it.",
            )
            _manual_candle_inputs_disabled = bool(_lab_use_learned_candle_weights)

            st.markdown("##### Candle-shape enhancer weights")
            st.caption("These fields show the learned global defaults from the candle model. Turn off learned family candle weights to edit and apply them as a flat manual fallback without double-counting.")
            _cw = _load_lab_default_candle_weights()
            _lab_candle_state_defaults = {
                "lab_d_doji_bonus": _cw["doji"],
                "lab_d_hammer_bonus": _cw["hammer"],
                "lab_d_marubozu_bonus": _cw["marubozu"],
                "lab_d_confirmed_hammer_a_bonus": _cw["confirmed_hammer_a"],
                "lab_d_mstar_bonus": _cw["morning_star"],
                "lab_d_engulf_bonus": _cw["engulfing"],
                "lab_d_engulf_trend_combo_bonus": _cw["engulfing_trend_combo"],
                "lab_d_harami_bonus": _cw["harami"],
                "lab_d_piercing_bonus": _cw["piercing_line"],
                "lab_d_piercing_variant_bonus": _cw["piercing_variant"],
                "lab_d_piercing_variant_b_combo_bonus": _cw["piercing_variant_b_combo"],
                "lab_d_inv_hammer_bonus": _cw["inverted_hammer"],
                "lab_d_belt_hold_bonus": _cw["belt_hold"],
                "lab_d_three_white_bonus": _cw["three_white_soldiers"],
            }
            _lab_candle_defaults_cfg = tuple(
                (key, float(value)) for key, value in sorted(_lab_candle_state_defaults.items())
            )
            if st.session_state.get("_lab_candle_defaults_cfg") != _lab_candle_defaults_cfg:
                for state_key, state_value in _lab_candle_state_defaults.items():
                    st.session_state[state_key] = float(state_value)
                st.session_state["_lab_candle_defaults_cfg"] = _lab_candle_defaults_cfg
            _e = st.columns(6)
            _lab_doji_bonus = _e[0].number_input("Doji", min_value=-20.0, max_value=20.0, value=_cw["doji"], step=0.5, format="%.1f", key="lab_d_doji_bonus", help=_candle_help("doji"), disabled=_manual_candle_inputs_disabled)
            _lab_hammer_bonus = _e[1].number_input("Hammer", min_value=-20.0, max_value=20.0, value=_cw["hammer"], step=0.5, format="%.1f", key="lab_d_hammer_bonus", help=_candle_help("hammer"), disabled=_manual_candle_inputs_disabled)
            _lab_marubozu_bonus = _e[2].number_input("Marubozu", min_value=-20.0, max_value=20.0, value=_cw["marubozu"], step=0.5, format="%.1f", key="lab_d_marubozu_bonus", help=_candle_help("marubozu"), disabled=_manual_candle_inputs_disabled)
            _lab_confirmed_hammer_a_bonus = _e[3].number_input("Ham+A", min_value=-20.0, max_value=20.0, value=_cw["confirmed_hammer_a"], step=0.5, format="%.1f", key="lab_d_confirmed_hammer_a_bonus", help=_candle_help("confirmed_hammer_a"), disabled=_manual_candle_inputs_disabled)
            _lab_mstar_bonus = _e[4].number_input("M.Star", min_value=-20.0, max_value=20.0, value=_cw["morning_star"], step=0.5, format="%.1f", key="lab_d_mstar_bonus", help=_candle_help("morning_star"), disabled=_manual_candle_inputs_disabled)
            _lab_engulf_bonus = _e[5].number_input("Engulf", min_value=-20.0, max_value=20.0, value=_cw["engulfing"], step=0.5, format="%.1f", key="lab_d_engulf_bonus", help=_candle_help("engulfing"), disabled=_manual_candle_inputs_disabled)
            _f = st.columns(6)
            _lab_engulf_trend_combo_bonus = _f[0].number_input("Engulf A/C/G", min_value=-20.0, max_value=20.0, value=_cw["engulfing_trend_combo"], step=0.5, format="%.1f", key="lab_d_engulf_trend_combo_bonus", help=_candle_help("engulfing_trend_combo"), disabled=_manual_candle_inputs_disabled)
            _lab_harami_bonus = _f[1].number_input("Harami", min_value=-20.0, max_value=20.0, value=_cw["harami"], step=0.5, format="%.1f", key="lab_d_harami_bonus", help=_candle_help("harami"), disabled=_manual_candle_inputs_disabled)
            _lab_piercing_bonus = _f[2].number_input("Pierce", min_value=-20.0, max_value=20.0, value=_cw["piercing_line"], step=0.5, format="%.1f", key="lab_d_piercing_bonus", help=_candle_help("piercing_line"), disabled=_manual_candle_inputs_disabled)
            _lab_piercing_variant_bonus = _f[3].number_input("Pierce V", min_value=-20.0, max_value=20.0, value=_cw["piercing_variant"], step=0.5, format="%.1f", key="lab_d_piercing_variant_bonus", help=_candle_help("piercing_variant"), disabled=_manual_candle_inputs_disabled)
            _lab_inv_hammer_bonus = _f[4].number_input("Inv Ham", min_value=-20.0, max_value=20.0, value=_cw["inverted_hammer"], step=0.5, format="%.1f", key="lab_d_inv_hammer_bonus", help=_candle_help("inverted_hammer"), disabled=_manual_candle_inputs_disabled)
            _lab_belt_hold_bonus = _f[5].number_input("Belt", min_value=-20.0, max_value=20.0, value=_cw["belt_hold"], step=0.5, format="%.1f", key="lab_d_belt_hold_bonus", help=_candle_help("belt_hold"), disabled=_manual_candle_inputs_disabled)
            _g = st.columns(6)
            _lab_piercing_variant_b_combo_bonus = _g[0].number_input("PierceV+B", min_value=-20.0, max_value=20.0, value=_cw["piercing_variant_b_combo"], step=0.5, format="%.1f", key="lab_d_piercing_variant_b_combo_bonus", help=_candle_help("piercing_variant_b_combo"), disabled=_manual_candle_inputs_disabled)
            _lab_three_white_bonus = _g[1].number_input("3 White", min_value=-20.0, max_value=20.0, value=_cw["three_white_soldiers"], step=0.5, format="%.1f", key="lab_d_three_white_bonus", help=_candle_help("three_white_soldiers"), disabled=_manual_candle_inputs_disabled)
            _lab_max_enh = st.number_input("Max total bonus", min_value=1.0, max_value=50.0, value=30.0, step=1.0, format="%.0f", key="lab_d_max_enh", help="Cap on combined enhancer bonus")

        _stop_mode_key = {
            "Fixed %": "fixed_pct",
            "ATR": "atr",
            "Structure + ATR": "structure_atr",
            "Score >95 hold to target": "score_gt_95_hold_to_target",
            "Score >90 hold to target": "score_gt_90_hold_to_target",
        }[_lab_stop_mode]
        _lab_source_signals = signals.copy()
        _using_saved_pattern_a = _lab_pattern_keys == {"A"}
        _lab_pattern_keys_sorted = tuple(sorted(_lab_pattern_keys))
        _lab_source_mode = "saved_pattern_a"
        if not _using_saved_pattern_a and not all_pattern_signals.empty and "pattern_family" in all_pattern_signals.columns:
            _lab_source_signals = all_pattern_signals[
                all_pattern_signals["pattern_family"].astype(str).isin(sorted(_lab_pattern_keys))
            ].copy()
            _lab_source_mode = "saved_all_patterns"
            st.caption(f"Using persisted all-pattern signal history for pattern families: {', '.join(sorted(_lab_pattern_keys))}.")
        elif not _using_saved_pattern_a:
            _lab_source_signals = _build_lab_history_signals(
                prices,
                use_pattern_a="A" in _lab_pattern_keys,
                use_pattern_b="B" in _lab_pattern_keys,
                use_pattern_c="C" in _lab_pattern_keys,
                use_pattern_d="D" in _lab_pattern_keys,
                use_pattern_e="E" in _lab_pattern_keys,
                use_pattern_f="F" in _lab_pattern_keys,
                use_pattern_g="G" in _lab_pattern_keys,
            )
            _lab_source_mode = "rebuilt_history"
            st.caption(f"Using rebuilt historical lab signals for pattern families: {', '.join(sorted(_lab_pattern_keys))}.")
        else:
            st.caption("Using saved Pattern A signal history from signals_pattern_a.csv.")

        _base_lab_signals = _apply_lab_stop_mode(
            _lab_source_signals,
            prices,
            stop_mode=_stop_mode_key,
            fixed_stop_pct=float(_lab_stp),
            atr_period=int(_lab_atr_period),
            atr_multiplier=float(_lab_atr_mult),
            structure_lookback=5,
            structure_atr_buffer=float(_lab_atr_mult),
        )
        _lab_signals = _rescore_signals(_base_lab_signals, prices) if _rescore_on else _base_lab_signals.copy()
        _lab_signals = _annotate_hold_to_target_only(_lab_signals, _stop_mode_key)
        _lab_idx_rs20 = build_ticker_index_rs_table(prices, benchmark_prices, lookback_days=20)
        _lab_idx_rs50 = build_ticker_index_rs_table(prices, benchmark_prices, lookback_days=50)
        if not _lab_idx_rs20.empty and not _lab_idx_rs50.empty:
            _lab_idx_rs = _lab_idx_rs20.merge(_lab_idx_rs50, on=["Date", "ticker"], how="outer")
        elif not _lab_idx_rs20.empty:
            _lab_idx_rs = _lab_idx_rs20.copy()
        elif not _lab_idx_rs50.empty:
            _lab_idx_rs = _lab_idx_rs50.copy()
        else:
            _lab_idx_rs = pd.DataFrame()
        if not _lab_idx_rs.empty and not _lab_signals.empty:
            _lab_signals = _attach_stock_index_rs(_lab_signals, _lab_idx_rs)
        _lab_signals = _apply_stock_rs_score_bonus(
            _lab_signals,
            enabled=bool(_lab_use_rs_bonus),
            max_bonus=float(_lab_rs_bonus_max),
        )

        # ── Apply candle-shape enhancer bonuses to scores (per signal date) ──
        _lab_enhanced = _lab_signals.copy()
        _learned_candle_payload = _load_candle_weights_payload() if _lab_use_learned_candle_weights else {}
        _manual_enh_bonuses = {
            "candle_doji": _lab_doji_bonus,
            "candle_hammer": _lab_hammer_bonus,
            "candle_marubozu": _lab_marubozu_bonus,
            "candle_confirmed_hammer_a": _lab_confirmed_hammer_a_bonus,
            "candle_morning_star": _lab_mstar_bonus,
            "candle_engulfing": _lab_engulf_bonus,
            "candle_engulfing_trend_combo": _lab_engulf_trend_combo_bonus,
            "candle_harami": _lab_harami_bonus,
            "candle_piercing_line": _lab_piercing_bonus,
            "candle_piercing_variant": _lab_piercing_variant_bonus,
            "candle_piercing_variant_b_combo": _lab_piercing_variant_b_combo_bonus,
            "candle_inverted_hammer": _lab_inv_hammer_bonus,
            "candle_belt_hold": _lab_belt_hold_bonus,
            "candle_three_white_soldiers": _lab_three_white_bonus,
        }
        if _lab_use_learned_candle_weights:
            _enh_bonuses = {key: 0.0 for key in _manual_enh_bonuses}
        else:
            _enh_bonuses = _manual_enh_bonuses
        _any_manual_bonus = any(abs(float(b)) > 0 for b in _enh_bonuses.values())
        if (_lab_use_learned_candle_weights or _any_manual_bonus) and not _lab_enhanced.empty:
            # Tag each signal row with pattern booleans at its signal date
            _tag_candle_shapes_fast(_lab_enhanced, prices, ticker_col="ticker", date_col="signal_date", add_ns_suffix=True)
            _enh_totals = pd.Series(0.0, index=_lab_enhanced.index)
            if _lab_use_learned_candle_weights:
                _enh_totals = _enh_totals + _compute_family_learned_candle_bonus(_lab_enhanced, _learned_candle_payload)
            for _col, _bonus in _enh_bonuses.items():
                if abs(float(_bonus)) > 0 and _col in _lab_enhanced.columns:
                    _enh_totals.loc[_lab_enhanced[_col].astype(bool)] += _bonus
            if _lab_max_enh > 0:
                _enh_totals = _enh_totals.clip(lower=-_lab_max_enh, upper=_lab_max_enh)
            _lab_enhanced["enhancer_bonus"] = _enh_totals
            _lab_enhanced["signal_score"] = (_lab_enhanced["signal_score"].astype(float) + _enh_totals).clip(0, 100)
            _n_boosted = int((_enh_totals > 0).sum())
            _n_penalized = int((_enh_totals < 0).sum())
            st.caption(f"🕯️ Candle model adjusted {_n_boosted + _n_penalized}/{len(_lab_enhanced)} signals: boosted={_n_boosted}, penalized={_n_penalized}. Use Min score to filter on the enhanced score.")

        _lab_enhanced = _annotate_hold_to_target_only(_lab_enhanced, _stop_mode_key)

        _tracker_cache_params = {
            "pattern_families": _lab_pattern_keys_sorted,
            "source_mode": _lab_source_mode,
            "evaluation_mode": str(st.session_state.get("lab_evaluation_mode", "walk-forward")),
            "train_end_date": _get_backtest_train_end_date().isoformat(),
            "evaluation_hold_days": int(st.session_state.get("lab_eval_hold_days", 30) or 30),
            "target_pct": float(_lab_tgt),
            "stop_mode": _lab_stop_mode,
            "capital_per_trade": float(_lab_cap),
            "min_score": int(_lab_min_score),
            "rescore": bool(_rescore_on),
            "rs_bonus": bool(_lab_use_rs_bonus),
            "rs_bonus_cap": float(_lab_rs_bonus_max),
            "use_learned_candle_weights": bool(_lab_use_learned_candle_weights),
            "stop_pct": float(_lab_stp),
            "atr_period": int(_lab_atr_period),
            "atr_mult": float(_lab_atr_mult),
            "doji_bonus": float(_lab_doji_bonus),
            "hammer_bonus": float(_lab_hammer_bonus),
            "marubozu_bonus": float(_lab_marubozu_bonus),
            "confirmed_hammer_a_bonus": float(_lab_confirmed_hammer_a_bonus),
            "morning_star_bonus": float(_lab_mstar_bonus),
            "engulf_bonus": float(_lab_engulf_bonus),
            "engulf_trend_combo_bonus": float(_lab_engulf_trend_combo_bonus),
            "harami_bonus": float(_lab_harami_bonus),
            "piercing_bonus": float(_lab_piercing_bonus),
            "piercing_variant_bonus": float(_lab_piercing_variant_bonus),
            "piercing_variant_b_combo_bonus": float(_lab_piercing_variant_b_combo_bonus),
            "inv_hammer_bonus": float(_lab_inv_hammer_bonus),
            "belt_hold_bonus": float(_lab_belt_hold_bonus),
            "three_white_bonus": float(_lab_three_white_bonus),
            "max_enh_bonus": float(_lab_max_enh),
        }
        _tracker_cache_key = _make_session_cache_key("lab_tracker", _tracker_cache_params)
        _tracker_input = _lab_enhanced if _lab_min_score == 0 else _lab_enhanced[_lab_enhanced["signal_score"].fillna(0) >= _lab_min_score]
        _tracker_input, _tracker_scope_note = _filter_lab_signals_for_evaluation_window(_tracker_input)
        _tracker = _session_cache_get_df("_lab_tracker_cache", _tracker_cache_key)
        if _tracker is None:
            _tracker = build_signal_tracker(
                _tracker_input,
                prices,
                target_pct=_lab_tgt,
                stop_pct=_lab_stp,
                capital_per_trade=_lab_cap,
            )
            if not _tracker.empty:
                _tag_candle_shapes_fast(_tracker, prices, ticker_col="ticker", date_col="signal_date", add_ns_suffix=True)
            _session_cache_set_df("_lab_tracker_cache", _tracker_cache_key, _tracker)
        if _tracker_scope_note:
            st.caption(_tracker_scope_note)
        if not _tracker.empty:
            with _lab_filter_controls_container:
                # Step 5: Score distribution histogram above the filter controls
                _render_score_distribution(_tracker_input, min_score=float(_lab_min_score))
                _rf_sf_cur = st.session_state.get("lab_d_sf", "All")
                _rf_candle_cur = st.session_state.get("lab_d_candle_filter", [])
                _rf_ticker_cur = (st.session_state.get("lab_d_ticker_filter") or "").strip().upper()
                _rf_parts = [str(_rf_sf_cur)]
                if _rf_candle_cur:
                    _rf_parts.append(f"{len(_rf_candle_cur)} candle(s)")
                if _rf_ticker_cur:
                    _rf_parts.append(_rf_ticker_cur)
                _rf_summary = " · ".join(_rf_parts)
                with st.expander(_rf_summary, expanded=False):
                    render_caption_with_help("Status", "status", key="lab_status_filter_help")
                    _lab_sf = st.selectbox("Status", ["All", "Target Hit ✅", "Stop Hit 🛑", "Holding"], key="lab_d_sf", label_visibility="collapsed")

                    _lf_mid1, _lf_mid2 = st.columns([1.15, 0.85])
                    with _lf_mid1:
                        render_caption_with_help("Sort by", "sort_order", key="lab_sort_by_help")
                        _lab_sort_by = st.selectbox(
                            "Sort by",
                            options=["signal_date", "signal_score", "stock_rs20", "stock_rs50", "return_pct", "pnl", "days_held", "ticker"],
                            index=1,
                            key="lab_d_sort_by",
                            label_visibility="collapsed",
                        )
                    with _lf_mid2:
                        render_caption_with_help("Sort direction", "sort_direction", key="lab_sort_direction_help")
                        _lab_sort_desc = st.checkbox("Descending", value=True, key="lab_d_sort_desc")

                    render_caption_with_help("Candle shape", "pattern", key="lab_candle_shape_help")
                    _nav_candle_sel = st.multiselect(
                        "Candle shape",
                        options=["Doji", "Hammer", "Bullish Marubozu", "Confirmed Hammer + Pattern A", "Morning Star", "Engulfing", "Engulf A/C/G", "Harami", "Piercing Line", "Piercing Variant", "Pierce V+B", "Inverted Hammer", "Belt Hold", "Three White Soldiers"],
                        key="lab_d_candle_filter",
                        label_visibility="collapsed",
                        help="Filter to signals that matched any selected candle pattern. The ? icons in the enhancer section explain each pattern in plain English.",
                    )

                    # Step 8: ticker text filter — pre-populated from lab_prefill_ticker_filter
                    _ticker_prefill = st.session_state.pop("lab_prefill_ticker_filter", None) or ""
                    if _ticker_prefill and not st.session_state.get("lab_d_ticker_filter"):
                        st.session_state["lab_d_ticker_filter"] = _ticker_prefill
                    render_caption_with_help("Ticker filter", "ticker", key="lab_ticker_filter_help")
                    _lab_ticker_filter = st.text_input(
                        "Ticker filter",
                        key="lab_d_ticker_filter",
                        placeholder="e.g. RELIANCE",
                        label_visibility="collapsed",
                    ).strip().upper()

            _view_cache_params = {
                **_tracker_cache_params,
                "status_filter": _lab_sf,
                "max_days_held": int(_lab_max_days_held),
                "candle_filter": tuple(_nav_candle_sel),
                "sort_by": _lab_sort_by,
                "sort_desc": bool(_lab_sort_desc),
                "ticker_filter": _lab_ticker_filter,
            }
            _view_cache_key = _make_session_cache_key("lab_view", _view_cache_params)
            _view = _session_cache_get_df("_lab_view_cache", _view_cache_key)
            if _view is None:
                _view = _filter_signal_tracker_view(
                    _tracker,
                    status_filter=_lab_sf,
                    candle_filters=_nav_candle_sel,
                    max_days_held=int(_lab_max_days_held),
                    sort_by=_lab_sort_by,
                    sort_desc=bool(_lab_sort_desc),
                )
                # Apply ticker text filter (Step 8)
                if _lab_ticker_filter and "ticker" in _view.columns:
                    _view = _view[_view["ticker"].astype(str).str.upper().str.contains(
                        _lab_ticker_filter, na=False
                    )].copy()
                _session_cache_set_df("_lab_view_cache", _view_cache_key, _view)

            _summary = summarize_signal_tracker(_view)
            _record_lab_session_snapshot(_view_cache_key, _view_cache_params, _summary, _view)
            _t_pnl = float(_summary["total_pnl"])
            _t_pnl_delta = f"-₹{abs(_t_pnl):,.0f}" if _t_pnl < 0 else f"₹{_t_pnl:,.0f}"
            _closed_pnl = float(_summary["closed_pnl"])
            _closed_pnl_delta = f"-₹{abs(_closed_pnl):,.0f}" if _closed_pnl < 0 else f"₹{_closed_pnl:,.0f}"
            _summary_metrics = [
                {"label": "Total signals", "value": int(_summary["n_total"]), "help": "Records remaining after the active filter set is applied."},
                {"label": "Target hit", "value": int(_summary["n_target"]), "tone": "positive", "help": "Trades that hit the target before stop or evaluation end."},
                {"label": "Stop hit", "value": int(_summary["n_stop"]), "tone": "warning", "help": "Trades that breached the configured stop before target."},
                {"label": "Holding", "value": int(_summary["n_holding"]), "help": "Trades still open at the latest available price."},
                {"label": "Overall return", "value": f"{float(_summary['overall_return']):.1f}%", "delta": _t_pnl_delta, "tone": "positive" if float(_summary['overall_return']) >= 0 else "negative", "help": "Includes open holding positions marked at latest close."},
                {"label": "Closed return", "value": f"{float(_summary['closed_return']):.1f}%", "delta": _closed_pnl_delta, "tone": "positive" if float(_summary['closed_return']) >= 0 else "negative", "help": "Only closed target-hit and stop-hit trades."},
                {"label": "Total invested", "value": f"₹{float(_summary['total_invested']):,.0f}", "help": "Capital allocated across the filtered trade set."},
                {"label": "Current value", "value": f"₹{float(_summary['total_current']):,.0f}", "help": "Marked-to-market value using the latest available close."},
                {"label": "Win rate", "value": f"{float(_summary['win_rate']):.0f}%", "tone": "positive" if float(_summary['win_rate']) >= 50.0 else "warning", "help": "Target hit divided by closed trades."},
            ]
            with _lab_summary_container:
                render_heading_with_help(
                    "Summary KPIs",
                    "summary_kpis",
                    key="lab_summary_kpis_help",
                )
                _render_summary_kpi_strip(_summary_metrics)
                _render_pattern_hit_summary(_view)

            _lab_tracker_cache_size = len(st.session_state.get("_lab_tracker_cache", {}))
            _lab_view_cache_size = len(st.session_state.get("_lab_view_cache", {}))
            _lab_dump_history = _build_lab_session_history_df()
            _lab_dump_rows = _build_lab_session_dump_df()
            _sc = [c for c in ["signal_date", "ticker", "entry_price", "status", "signal_score", "return_pct", "pattern", "enhancer_bonus",
                                "pattern_family", "qty", "invested", "target_price", "stop_price", "latest_close", "current_value", "pnl",
                                "days_held", "exit_date", "score_pattern", "sma50_slope_pct", "ma_slope_bonus", "pattern_bonus", "stock_rs20", "stock_rs50", "rs_bonus"] if c in _view.columns]
            _view_display = _view[_sc].copy()
            _float_cols = _view_display.select_dtypes(include=["float64", "float32"]).columns.tolist()
            for _fc in _float_cols:
                _view_display[_fc] = _view_display[_fc].round(2)
            _lab_export_filename = build_lab_export_filename(
                pattern_families=_lab_pattern_keys_sorted,
                status_filter=_lab_sf,
                candle_filters=_nav_candle_sel,
                min_score=int(_lab_min_score),
                max_days_held=int(_lab_max_days_held),
                sort_by=_lab_sort_by,
                sort_desc=bool(_lab_sort_desc),
                row_count=len(_view_display),
            )
            with _col_table:
                render_heading_with_help(
                    "Trade Records",
                    "trade_records",
                    key="lab_trade_records_help",
                    caption="The record table stays visible as the main drill-down surface. Select a row to open its chart view and export the filtered slice when needed.",
                )
                _trade_record_help = table_help_map("trade_records", _view_display.columns)
                render_table_help_glossary(
                    "Trade Records",
                    _trade_record_help,
                    key_prefix="lab_trade_record_cols",
                )
                _export_col, _export_meta_col = st.columns([1.2, 3.0])
                with _export_col:
                    st.download_button(
                        "Download filtered table CSV",
                        data=to_csv_bytes(_view_display),
                        file_name=_lab_export_filename,
                        mime="text/csv",
                        key="lab_filtered_table_download",
                        disabled=_view_display.empty,
                    )
                with _export_meta_col:
                    st.caption(f"Export file: {_lab_export_filename}")

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
                        column_config=build_dataframe_column_config(_trade_record_help),
                        on_select="rerun",
                        selection_mode="single-row",
                        key="lab_d_tracker_sel",
                    )
                _sel_rows = _sel_ev.selection.rows if _sel_ev and _sel_ev.selection else []
                if _sel_rows and _sel_rows[0] < len(_view):
                    st.session_state["_lab_d_had_sel"] = True
                    _picked = _view.iloc[_sel_rows[0]]
                    _chart_row = pd.Series({"ticker": str(_picked["ticker"]) + ".NS"})
                    if _chart_col is not None:
                        with _chart_col:
                            st.markdown(f"### 📈 {_picked['ticker']}")
                            render_chart(_chart_row, prices,
                                         signal_date=str(_picked.get("signal_date", "")),
                                         exit_date=str(_picked.get("exit_date", "")))
                            # Step 7: nav to Tomorrow's Picks for this ticker
                            if st.button(
                                "📌 View in Tomorrow's Picks",
                                key="lab_nav_to_tomorrow",
                                help="Switch to Tomorrow's Picks and pre-select this ticker.",
                            ):
                                st.session_state["mode"] = "Tomorrow"
                                st.session_state["selected_stock"] = str(_picked["ticker"])
                                st.session_state["_nav_skip_sync"] = True
                                st.rerun()
                    else:
                        st.rerun()
                else:
                    if _had_sel:
                        st.session_state["_lab_d_had_sel"] = False
                        st.rerun()

            with _lab_charts_container:
                _render_equity_curve(_view)
                _render_risk_return_scatter(_view)

            with _lab_snapshot_container:
                _render_backtest_stop_risk_results("lab_main")

            with st.expander(
                f"Advanced session diagnostics  ({_lab_tracker_cache_size} tracker configs, {_lab_view_cache_size} filtered views)",
                expanded=False,
            ):
                st.caption("Current session only. Repeated lab/filter combinations reuse cached outputs here, and the cache summary now includes hidden scope and scoring inputs that change row counts.")
                _dump_a, _dump_b, _dump_c = st.columns([1, 1, 1.2])
                with _dump_a:
                    st.download_button(
                        "Download cache summary",
                        data=to_csv_bytes(_lab_dump_history),
                        file_name="lab_session_cache_summary.csv",
                        mime="text/csv",
                        disabled=_lab_dump_history.empty,
                        key="lab_cache_summary_download",
                    )
                with _dump_b:
                    st.download_button(
                        "Download cached rows",
                        data=to_csv_bytes(_lab_dump_rows),
                        file_name="lab_session_cache_rows.csv",
                        mime="text/csv",
                        disabled=_lab_dump_rows.empty,
                        key="lab_cache_rows_download",
                    )
                with _dump_c:
                    if st.button("Clear session cache", key="lab_clear_session_cache"):
                        _clear_lab_session_cache()
                        st.rerun()
                if _lab_dump_history.empty:
                    st.info("No cached filter snapshots yet in this session.")
                else:
                    render_table(_lab_dump_history, height=220)
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
            render_heading_with_help(
                "Manual positions",
                "manual_positions",
                key="lab_manual_positions_help",
                level=4,
            )
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
            render_table(
                view_df.sort_values(["created_at", "ticker"], ascending=[False, True]),
                height=360,
                column_help=table_help_map("manual_positions", view_df.columns),
                table_help_title="Manual positions",
                table_help_key_prefix="manual_positions_cols",
            )

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

if st.session_state.get("mode") == "Coverage":
    _render_coverage_page(all_pattern_signals, prices)
    st.stop()

if "focus_ticker" not in st.session_state and not needs_action_rows.empty:
    st.session_state["focus_ticker"] = str(needs_action_rows.iloc[0]["ticker"])

# Legacy tabbed workspace removed from runtime. The app is now strictly
# navbar-driven: Tomorrow's Picks and Backtesting Lab.
st.stop()

market_tab, dashboard_tab, signals_tab, portfolio_tab, backtest_lab_tab, telegram_tab = st.tabs(["Market Dashboard", "Dashboard", "Signals", "Portfolio", "Backtesting Lab", "Telegram"])

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
                render_chart(pd.Series({"ticker": pick}), prices, chart_key=f"market_insight_{pick}")

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
                st.session_state["flow_step_3_date"] = date.today().isoformat()
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
        current_signal_date = latest_trading_date_str or date.today().isoformat()

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
                current_sell_date = latest_trading_date_str or date.today().isoformat()

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
                    render_chart(pd.Series({"ticker": chart_ticker}), prices, chart_key=f"signal_chart_{chart_ticker}")
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

with backtest_lab_tab:
    st.subheader("Backtesting Lab")
    st.caption("Auto-track every generated buy signal: buy 1 lot at entry, target +6%, stop −7%.")
    render_pattern_bonus_expander()
    render_candle_enhancer_expander()
    _render_backtest_evaluation_controls("lab_tab")
    _render_backtest_stop_risk_results("lab_tab")

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
              lab_stop = st.number_input("Stop %", min_value=1.0, max_value=50.0, value=9.0, step=0.5, key="lab_stop_pct")
        with lab_c3:
            lab_capital = st.number_input("₹ per trade", min_value=1000.0, max_value=500000.0, value=10000.0, step=1000.0, key="lab_capital")

        tracker_signals, tracker_scope_note = _filter_lab_signals_for_evaluation_window(signals)
        if tracker_scope_note:
            st.caption(tracker_scope_note)

        tracker_df = build_signal_tracker(
            tracker_signals, prices,
            target_pct=lab_target,
            stop_pct=lab_stop,
            capital_per_trade=lab_capital,
        )

        if tracker_df.empty:
            st.info("No signal data to track.")
        else:
            # ── Tag candle shapes on tracker rows ──
            _tag_candle_shapes_fast(tracker_df, prices, ticker_col="ticker", date_col="signal_date", add_ns_suffix=True)

            # Filters
            _lf1, _lf2, _lf3 = st.columns(3)
            with _lf1:
                status_opts = ["All", "Target Hit ✅", "Stop Hit 🛑", "Holding"]
                lab_status_filter = st.selectbox("Filter by status", options=status_opts, key="lab_status_filter")
            with _lf2:
                lab_max_days_held = st.number_input(
                    "Filter max days held",
                    min_value=1,
                    max_value=365,
                    value=60,
                    step=1,
                    key="lab_max_days_held",
                )
            with _lf3:
                _lab_candle_sel = st.multiselect(
                    "Filter by candle shape",
                    options=["Doji", "Hammer", "Bullish Marubozu", "Confirmed Hammer + Pattern A", "Morning Star", "Engulfing", "Engulf A/C/G", "Harami", "Piercing Line", "Piercing Variant", "Pierce V+B", "Inverted Hammer", "Belt Hold", "Three White Soldiers"],
                    key="lab_candle_filter",
                )

            view = tracker_df.copy()
            if lab_status_filter != "All":
                view = view[view["status"] == lab_status_filter]
            if "days_held" in view.columns:
                view = view[pd.to_numeric(view["days_held"], errors="coerce").fillna(10**9) <= int(lab_max_days_held)].copy()
            if _lab_candle_sel:
                _lab_cmap = {
                    "Doji": "candle_doji",
                    "Hammer": "candle_hammer",
                    "Bullish Marubozu": "candle_marubozu",
                    "Confirmed Hammer + Pattern A": "candle_confirmed_hammer_a",
                    "Morning Star": "candle_morning_star",
                    "Engulfing": "candle_engulfing",
                    "Engulf A/C/G": "candle_engulfing_trend_combo",
                    "Harami": "candle_harami",
                    "Piercing Line": "candle_piercing_line",
                    "Piercing Variant": "candle_piercing_variant",
                    "Pierce V+B": "candle_piercing_variant_b_combo",
                    "Inverted Hammer": "candle_inverted_hammer",
                    "Belt Hold": "candle_belt_hold",
                    "Three White Soldiers": "candle_three_white_soldiers",
                }
                _lab_cmask = pd.Series(False, index=view.index)
                for _lbl in _lab_candle_sel:
                    _col = _lab_cmap.get(_lbl)
                    if _col and _col in view.columns:
                        _lab_cmask = _lab_cmask | view[_col].astype(bool)
                view = view[_lab_cmask].copy()

            # Summary metrics (from filtered view)
            summary = summarize_signal_tracker(view)

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Total Signals", int(summary["n_total"]))
            m2.metric("Target Hit ✅", int(summary["n_target"]))
            m3.metric("Stop Hit 🛑", int(summary["n_stop"]))
            m4.metric("Holding", int(summary["n_holding"]))
            total_pnl = float(summary["total_pnl"])
            total_pnl_delta = f"-₹{abs(total_pnl):,.0f}" if total_pnl < 0 else f"₹{total_pnl:,.0f}"
            m5.metric("Overall Return incl. Holding", f"{float(summary['overall_return']):.1f}%", delta=total_pnl_delta)
            closed_pnl = float(summary["closed_pnl"])
            closed_pnl_delta = f"-₹{abs(closed_pnl):,.0f}" if closed_pnl < 0 else f"₹{closed_pnl:,.0f}"
            m6.metric("Closed Trades Return", f"{float(summary['closed_return']):.1f}%", delta=closed_pnl_delta)

            m7, m8, m9 = st.columns(3)
            m7.metric("Total Invested", f"₹{float(summary['total_invested']):,.0f}")
            m8.metric("Current Value", f"₹{float(summary['total_current']):,.0f}")
            m9.metric("Win Rate", f"{float(summary['win_rate']):.0f}%", help="Target hit / (Target hit + Stop hit)")
            _render_pattern_hit_summary(view)

            show_cols = [
                "signal_date", "ticker", "entry_price", "qty", "invested",
                "target_price", "stop_price", "latest_close", "current_value",
                "pnl", "return_pct", "days_held", "exit_date", "status", "signal_score", "score_pattern", "pattern_bonus",
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
    f"Production app: {PRODUCTION_APP_URL} | {_load_build_marker()}"
)
