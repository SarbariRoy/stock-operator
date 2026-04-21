"""Phase 2: UI utilities for catalyst mode selection and filtering.

Provides functions to:
1. Render catalyst mode selector in UI.
2. Apply catalyst-based filtering to signals based on selected mode.
"""

from __future__ import annotations

import pandas as pd


CATALYST_MODES = {
    "baseline": {"label": "Baseline (No Catalysts)", "description": "All signals, no catalyst filtering"},
    "market_only": {"label": "Market Regime Only", "description": "Filter by VIX, flows, energy shocks"},
    "market_and_events": {"label": "Market + Company Events", "description": "Filter by both market conditions and earnings/dividend windows"},
}


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
    """Apply catalyst-based filtering to signals based on mode.
    
    Args:
        signals: Signal DataFrame with optional catalyst columns.
        mode: One of "baseline", "market_only", "market_and_events".
        market_vix_high_threshold: Exclude signals on high-VIX days if specified.
        market_flow_weak_threshold: Exclude signals on weak-flow days if specified.
    
    Returns:
        Filtered signals DataFrame.
    """
    if mode == "baseline":
        return signals.copy()
    
    filtered = signals.copy()
    
    # Market regime filters.
    if "market_only" in mode or "market_and_events" in mode:
        if "vix_regime_high" in filtered.columns:
            filtered = filtered[~(filtered["vix_regime_high"] == True)]
        if "energy_regime_shock" in filtered.columns:
            filtered = filtered[~(filtered["energy_regime_shock"] == True)]
        if "flow_regime_weak" in filtered.columns and filtered["flow_regime_weak"].any():
            filtered = filtered[~(filtered["flow_regime_weak"] == True)]
    
    # Event window filters.
    if mode == "market_and_events":
        if "within_earnings_window" in filtered.columns:
            # Exclude signals within earnings windows.
            filtered = filtered[~(filtered["within_earnings_window"] == True)]
        if "within_dividend_window" in filtered.columns:
            # Exclude signals within dividend windows.
            filtered = filtered[~(filtered["within_dividend_window"] == True)]
        if "post_event_gap_risk" in filtered.columns:
            # Exclude signals with post-event gap risk.
            filtered = filtered[~(filtered["post_event_gap_risk"] == True)]
    
    return filtered


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
    
    return f"{CATALYST_MODES[mode]['label']}: {filtered_count}/{original_count} signals retained ({pct_excluded:.1f}% excluded)"
