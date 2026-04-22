"""Phase 2: UI utilities for catalyst mode selection and filtering.

Provides functions to:
1. Render catalyst mode selector in UI.
2. Apply catalyst-based filtering to signals based on selected mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CATALYST_MODES = {
    "baseline": {"label": "Baseline (No Catalysts)", "description": "All signals, no catalyst filtering"},
    "market_only": {"label": "Market Regime Soft Penalty", "description": "Keep all signals, but rank down stressed market regimes instead of filtering them out"},
    "market_and_events": {"label": "Market + Company Events", "description": "Apply soft market penalties and let event boosts rerank only the 70-84 score band"},
    "mid_band_event_rerank": {"label": "Mid-Band Event Rerank", "description": "Only let catalyst event boosts rerank signals whose pre-catalyst score is in the 70-84 band"},
}


def _numeric_series(signals: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in signals.columns:
        return pd.to_numeric(signals[column], errors="coerce").fillna(default)
    return pd.Series(default, index=signals.index, dtype=float)


def _bool_series(signals: pd.DataFrame, column: str) -> pd.Series:
    if column in signals.columns:
        return signals[column].fillna(False).astype(bool)
    return pd.Series(False, index=signals.index, dtype=bool)


def _apply_market_soft_penalty(
    signals: pd.DataFrame,
    *,
    vix_penalty: float = 0.5,
    flow_penalty: float = 0.25,
    energy_penalty: float = 0.25,
) -> pd.DataFrame:
    out = signals.copy()
    if "signal_score" not in out.columns:
        return out

    signal_score = _numeric_series(out, "signal_score")
    adjustment = pd.Series(0.0, index=out.index, dtype=float)
    adjustment += np.where(_bool_series(out, "vix_regime_high"), -float(vix_penalty), 0.0)
    adjustment += np.where(_bool_series(out, "flow_regime_weak"), -float(flow_penalty), 0.0)
    adjustment += np.where(_bool_series(out, "energy_regime_shock"), -float(energy_penalty), 0.0)

    out["score_adjustment_catalyst_mode_market"] = adjustment.round(2)
    out["signal_score"] = (signal_score + adjustment).clip(lower=0.0, upper=100.0).round(1)
    return out


def _apply_mid_band_event_rerank(
    signals: pd.DataFrame,
    *,
    band_min: float = 70.0,
    band_max: float = 85.0,
) -> pd.DataFrame:
    out = signals.copy()
    if "signal_score" not in out.columns:
        return out

    current_score = _numeric_series(out, "signal_score")
    base_score = _numeric_series(out, "signal_score_pre_catalyst_event")
    base_score = base_score.where(base_score.notna(), current_score)
    bonus = _numeric_series(out, "score_bonus_catalyst_event")
    penalty = _numeric_series(out, "score_penalty_catalyst_event")

    mid_band_mask = base_score.ge(float(band_min)) & base_score.lt(float(band_max))
    effective_bonus = bonus.where(mid_band_mask, 0.0)
    mode_adjustment = effective_bonus - penalty

    out["score_adjustment_catalyst_mode_events"] = mode_adjustment.round(2)
    out["signal_score"] = (base_score + mode_adjustment).clip(lower=0.0, upper=100.0).round(1)
    return out


def render_catalyst_mode_selector() -> str:
    """Render catalyst mode selector in Streamlit and return selected mode.
    
    Called from backtesting lab UI setup.
    """
    import streamlit as st
    
    st.subheader("🧬 Catalyst Filter Mode")
    
    mode = st.radio(
        "Select filtering mode:",
        options=list(CATALYST_MODES.keys()),
        format_func=lambda m: CATALYST_MODES[m]["label"],
        key="lab_catalyst_mode",
        index=0,  # Default: baseline (no filtering)
        help="Phase 2 feature: optionally filter eligible trade dates by market regime and/or company events.",
    )
    
    if mode != "baseline":
        st.caption(f"ℹ️ {CATALYST_MODES[mode]['description']}")
    
    return mode


def filter_signals_by_catalyst_mode(
    signals: pd.DataFrame,
    mode: str,
    *,
    market_vix_high_threshold: float | None = None,
    market_flow_weak_threshold: float | None = None,
) -> pd.DataFrame:
    """Apply catalyst-based ranking/selection logic to signals based on mode.
    
    Args:
        signals: Signal DataFrame with optional catalyst columns.
        mode: One of "baseline", "market_only", "market_and_events", "mid_band_event_rerank".
        market_vix_high_threshold: Reserved for future threshold overrides.
        market_flow_weak_threshold: Reserved for future threshold overrides.
    
    Returns:
        Signals DataFrame with mode-specific ranking applied.
    """
    if mode == "baseline":
        return signals.copy()

    ranked = signals.copy()
    if "signal_score" in ranked.columns and "signal_score_pre_catalyst_mode" not in ranked.columns:
        ranked["signal_score_pre_catalyst_mode"] = _numeric_series(ranked, "signal_score")

    if mode == "market_only":
        return _apply_market_soft_penalty(ranked)
    if mode == "mid_band_event_rerank":
        return _apply_mid_band_event_rerank(ranked)
    if mode == "market_and_events":
        ranked = _apply_mid_band_event_rerank(ranked)
        ranked = _apply_market_soft_penalty(ranked)
        return ranked

    return ranked


def summarize_catalyst_filtering(
    original_count: int,
    filtered_count: int,
    mode: str,
) -> str:
    """Generate UI summary text for catalyst filtering results."""
    if mode == "baseline":
        return f"Baseline mode: all {original_count} signals"

    excluded = original_count - filtered_count
    pct_excluded = (excluded / original_count * 100) if original_count > 0 else 0

    if mode in {"market_only", "market_and_events", "mid_band_event_rerank"}:
        return f"{CATALYST_MODES[mode]['label']}: reranked {filtered_count}/{original_count} signals ({pct_excluded:.1f}% excluded after downstream filters)"

    return f"{CATALYST_MODES[mode]['label']}: {filtered_count}/{original_count} signals retained ({pct_excluded:.1f}% excluded)"
