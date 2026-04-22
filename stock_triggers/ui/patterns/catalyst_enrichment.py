"""Phase 2: Catalyst feature enrichment utilities.

Joins market-regime and company-event factors onto signal rows.

Functions:
- enrich_signals_with_catalysts(): Attach market + event factors to signal DataFrame.
- compute_event_windows(): Derive event-proximity flags from event calendar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_EXTERNAL_FACTORS = DATA_DIR / "external_factors.csv"
DEFAULT_EVENT_CALENDAR = DATA_DIR / "event_calendar.csv"


def load_external_factors(path: Path | str = DEFAULT_EXTERNAL_FACTORS) -> pd.DataFrame:
    """Load market-regime factors (cached)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["Date", "india_vix_close", "usdinr_close", "brent_close", "fii_dii_net_cr"])
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")


def load_event_calendar(path: Path | str = DEFAULT_EVENT_CALENDAR) -> pd.DataFrame:
    """Load company event calendar (cached)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "event_date", "event_type", "amount_or_eps"])
    df = pd.read_csv(path, parse_dates=["event_date"])
    return df.sort_values(["ticker", "event_date"]).drop_duplicates(subset=["ticker", "event_date", "event_type"], keep="last")


def compute_event_windows(
    signals: pd.DataFrame,
    event_calendar: pd.DataFrame,
    *,
    window_days: int = 3,
) -> pd.DataFrame:
    """Compute event-window flags for each signal (±window_days around event).
    
    Returns signals with new columns:
    - within_earnings_window, within_dividend_window, post_event_gap_risk, event_proximity_score
    """
    signals = signals.copy()
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    event_calendar = event_calendar.copy()
    event_calendar["event_date"] = pd.to_datetime(event_calendar["event_date"])

    # Initialize flags.
    signals["within_earnings_window"] = False
    signals["within_dividend_window"] = False
    signals["post_event_gap_risk"] = False
    signals["event_proximity_score"] = 1.0  # Default: far from events.

    # Group events by ticker.
    for ticker in event_calendar["ticker"].unique():
        ticker_events = event_calendar[event_calendar["ticker"] == ticker].copy()
        ticker_signals = signals[signals["ticker"] == ticker].copy()

        for idx in ticker_signals.index:
            signal_date = signals.at[idx, "signal_date"]
            min_distance = float("inf")

            for _, event_row in ticker_events.iterrows():
                event_date = event_row["event_date"]
                event_type = event_row["event_type"]
                days_diff = (signal_date - event_date).days

                # Update proximity score (0=on event, 1=far).
                abs_diff = abs(days_diff)
                proximity = max(0, 1 - abs_diff / max(1, window_days))
                signals.at[idx, "event_proximity_score"] = max(
                    signals.at[idx, "event_proximity_score"],
                    proximity,
                )
                min_distance = min(min_distance, abs_diff)

                # Window flags (±window_days).
                if abs(days_diff) <= window_days:
                    if event_type.lower() == "earnings":
                        signals.at[idx, "within_earnings_window"] = True
                    elif event_type.lower() == "dividend":
                        signals.at[idx, "within_dividend_window"] = True

                # Post-event gap-risk: signal within 1 day after event AND >2% gap up.
                if 0 < days_diff <= 1 and "gap_pct" in signals.columns:
                    gap = signals.at[idx, "gap_pct"]
                    if pd.notna(gap) and gap > 2.0:
                        signals.at[idx, "post_event_gap_risk"] = True

    # Invert proximity so that 0=far, 1=on-event (for consistency with other flags).
    signals["event_proximity_score"] = 1 - signals["event_proximity_score"]
    signals["event_proximity_score"] = signals["event_proximity_score"].round(2)

    return signals


def compute_market_regimes(
    signals: pd.DataFrame,
    external_factors: pd.DataFrame,
    *,
    vix_percentile_window: int = 60,
    flow_percentile_window: int = 60,
) -> pd.DataFrame:
    """Compute derived market-regime flags from external factors.
    
    Signals should already have market factor columns (india_vix_close, usdinr_close, brent_close, fii_dii_net_cr).
    
    Returns signals with new columns:
    - vix_regime_high, flow_regime_weak, energy_regime_shock
    """
    signals = signals.copy()
    external_factors = external_factors.copy()
    external_factors["Date"] = pd.to_datetime(external_factors["Date"])

    # Compute percentile thresholds from recent history.
    recent_ef = external_factors[
        external_factors["Date"] <= external_factors["Date"].max()
    ].tail(vix_percentile_window).copy()

    vix_75 = recent_ef["india_vix_close"].quantile(0.90) if not recent_ef.empty else None
    flow_25 = recent_ef["fii_dii_net_cr"].quantile(0.25) if not recent_ef.empty else None

    # Regime flags. Market factor columns should already be in signals from previous join.
    if "india_vix_close" in signals.columns:
        signals["vix_regime_high"] = (signals["india_vix_close"] > vix_75) if vix_75 is not None else False
    if "fii_dii_net_cr" in signals.columns:
        signals["flow_regime_weak"] = (signals["fii_dii_net_cr"] < flow_25) if flow_25 is not None else False
    if "brent_close" in signals.columns:
        signals["energy_regime_shock"] = (signals["brent_close"].pct_change() < -0.05)
    
    # Fill in missing regime columns if they weren't created (in case external factors are empty).
    if "vix_regime_high" not in signals.columns:
        signals["vix_regime_high"] = False
    if "flow_regime_weak" not in signals.columns:
        signals["flow_regime_weak"] = False
    if "energy_regime_shock" not in signals.columns:
        signals["energy_regime_shock"] = False

    return signals


def apply_catalyst_event_score_adjustment(
    signals: pd.DataFrame,
    *,
    earnings_boost: float = 4.0,
    dividend_boost: float = 2.0,
    gap_risk_penalty: float = 2.0,
    total_cap: float = 6.0,
) -> pd.DataFrame:
    """Apply capped catalyst adjustment directly to signal_score.

    Heuristic intent: reward early positive event context while keeping caps tight.
    """
    out = signals.copy()
    if "signal_score" not in out.columns:
        return out

    # Make adjustment idempotent across repeated enrichment runs.
    def _series_or_default(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default)
        return pd.Series(default, index=out.index, dtype=float)

    existing_adj = _series_or_default("score_adjustment_catalyst_event", 0.0)
    current_score = _series_or_default("signal_score", 0.0)
    base_score = (current_score - existing_adj).clip(lower=0.0, upper=100.0)

    score_trend = _series_or_default("score_trend", 0.0)
    score_setup = _series_or_default("score_setup", 0.0)
    score_volume = _series_or_default("score_volume", 0.0)

    vix_high = out.get("vix_regime_high", False)
    if not isinstance(vix_high, pd.Series):
        vix_high = pd.Series(False, index=out.index)
    else:
        vix_high = vix_high.fillna(False).astype(bool)

    within_earnings = out.get("within_earnings_window", False)
    if not isinstance(within_earnings, pd.Series):
        within_earnings = pd.Series(False, index=out.index)
    else:
        within_earnings = within_earnings.fillna(False).astype(bool)

    within_dividend = out.get("within_dividend_window", False)
    if not isinstance(within_dividend, pd.Series):
        within_dividend = pd.Series(False, index=out.index)
    else:
        within_dividend = within_dividend.fillna(False).astype(bool)

    gap_risk = out.get("post_event_gap_risk", False)
    if not isinstance(gap_risk, pd.Series):
        gap_risk = pd.Series(False, index=out.index)
    else:
        gap_risk = gap_risk.fillna(False).astype(bool)

    # Positive reaction proxy around results date.
    positive_reaction = (
        ((score_trend >= 50.0) | (score_setup >= 50.0))
        & (score_volume >= 50.0)
    )

    earnings_adj = np.where(within_earnings & positive_reaction & ~vix_high, float(earnings_boost), 0.0)
    dividend_adj = np.where(within_dividend & (base_score >= 60.0) & ~vix_high, float(dividend_boost), 0.0)
    gap_adj = np.where(gap_risk, -float(gap_risk_penalty), 0.0)

    total_adj = pd.Series(earnings_adj + dividend_adj + gap_adj, index=out.index)
    total_adj = total_adj.clip(lower=-float(total_cap), upper=float(total_cap))

    out["signal_score_pre_catalyst_event"] = base_score.round(2)
    out["score_bonus_catalyst_event"] = pd.Series(earnings_adj + dividend_adj, index=out.index).round(2)
    out["score_penalty_catalyst_event"] = pd.Series(np.where(gap_risk, float(gap_risk_penalty), 0.0), index=out.index).round(2)
    out["score_adjustment_catalyst_event"] = total_adj.round(2)
    out["signal_score"] = (base_score + total_adj).clip(lower=0.0, upper=100.0).round(1)
    return out


def enrich_signals_with_catalysts(
    signals: pd.DataFrame,
    external_factors: pd.DataFrame | None = None,
    event_calendar: pd.DataFrame | None = None,
    *,
    include_market_regimes: bool = True,
    include_event_windows: bool = True,
    window_days: int = 3,
    apply_score_adjustment: bool = True,
) -> pd.DataFrame:
    """Attach catalyst features to signal DataFrame.
    
    Args:
        signals: Signal history DataFrame with signal_date and ticker columns.
        external_factors: Market-regime factors (loaded if None).
        event_calendar: Company event calendar (loaded if None).
        include_market_regimes: Whether to compute regime flags.
        include_event_windows: Whether to compute event-window flags.
        window_days: Event-window size (±days around event).
    
    Returns:
        Enriched signals DataFrame with catalyst columns.
    """
    signals = signals.copy()

    if external_factors is None:
        external_factors = load_external_factors()
    if event_calendar is None:
        event_calendar = load_event_calendar()

    # Market factors join.
    if not external_factors.empty:
        merge_columns = [
            "Date",
            "india_vix_close",
            "usdinr_close",
            "brent_close",
            "fii_dii_net_cr",
            "vix_change_1d_pct",
            "usdinr_ret_5d_pct",
            "brent_ret_5d_pct",
        ]
        signals.drop(columns=[col for col in merge_columns if col in signals.columns], inplace=True)
        signals = signals.merge(
            external_factors[merge_columns],
            left_on="signal_date",
            right_on="Date",
            how="left",
        )
        signals.drop(columns=["Date"], inplace=True, errors="ignore")

    # Event calendar joins and window computation.
    if not event_calendar.empty and include_event_windows:
        signals = compute_event_windows(signals, event_calendar, window_days=window_days)

    # Market regimes.
    if not external_factors.empty and include_market_regimes:
        signals = compute_market_regimes(signals, external_factors)

    # Apply capped catalyst score adjustment (boost/penalty) after catalyst flags are ready.
    if apply_score_adjustment:
        signals = apply_catalyst_event_score_adjustment(signals)

    return signals


def enrich_signal_history_file(
    signals_path: Path | str,
    external_factors_path: Path | str | None = None,
    event_calendar_path: Path | str | None = None,
    *,
    output_path: Path | str | None = None,
) -> None:
    """Batch enrich signal history file with catalyst features.
    
    Saves results to output_path or overwrites signals_path if output_path is None.
    """
    signals = pd.read_csv(signals_path, parse_dates=["signal_date"])
    external_factors = load_external_factors(external_factors_path or DEFAULT_EXTERNAL_FACTORS)
    event_calendar = load_event_calendar(event_calendar_path or DEFAULT_EVENT_CALENDAR)

    enriched = enrich_signals_with_catalysts(
        signals,
        external_factors=external_factors,
        event_calendar=event_calendar,
        include_market_regimes=True,
        include_event_windows=True,
    )

    output_path = Path(output_path or signals_path)
    enriched.to_csv(output_path, index=False)
    print(f"Saved enriched signals to {output_path}")
