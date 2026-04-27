"""Shared scoring utilities for all signal patterns."""

from __future__ import annotations

import pandas as pd


def clip_score(value: float) -> float:
    """Clamp *value* to [0, 100]."""
    return max(0.0, min(100.0, float(value)))


def score_rsi_sweet_spot(rsi_value: float | None) -> float:
    """Map RSI to a center-favored score with a 50-60 sweet spot.

    Returns 100 for RSI in [50, 60], then decays linearly toward 0 as RSI
    approaches 0 on the left or 100 on the right.
    """

    if rsi_value is None or pd.isna(rsi_value):
        return 50.0

    rsi = clip_score(float(rsi_value))
    if 50.0 <= rsi <= 60.0:
        return 100.0
    if rsi < 50.0:
        return clip_score((rsi / 50.0) * 100.0)
    return clip_score(((100.0 - rsi) / 40.0) * 100.0)


# Component weights for signal_score.
WEIGHT_TREND = 0.20
WEIGHT_SETUP = 0.20
WEIGHT_VOLUME = 0.13
WEIGHT_RISK = 0.14
WEIGHT_RSI = 0.03
WEIGHT_PATTERN = 0.30
MA_SLOPE_LOOKBACK_DAYS = 5
MA_SLOPE_BONUS_CAP = 3.0
PATTERN_WEIGHT_KEYS = ("A", "B", "C", "D", "E", "F", "G")
PATTERN_COMPONENT_CAP = round(WEIGHT_PATTERN * 100.0, 1)
LT_HEALTH_BONUS_STEP = 1.5
LT_HEALTH_BONUS_MAX_ABS = 3.0


def _coerce_pattern_contribution_map(pattern_weights: dict | None) -> dict[str, float]:
    weight_map = {key: 0.0 for key in PATTERN_WEIGHT_KEYS}
    if not pattern_weights:
        return weight_map
    for key in PATTERN_WEIGHT_KEYS:
        try:
            weight_map[key] = float(pattern_weights.get(key, 0.0))
        except (AttributeError, TypeError, ValueError):
            weight_map[key] = 0.0
    return weight_map


def _coerce_pattern_score_map(
    pattern_weights: dict | None,
    contribution_map: dict[str, float],
) -> dict[str, float]:
    score_map = {
        key: round((float(contribution_map.get(key, 0.0)) / PATTERN_COMPONENT_CAP) * 100.0, 1)
        if PATTERN_COMPONENT_CAP > 0
        else 0.0
        for key in PATTERN_WEIGHT_KEYS
    }
    if not pattern_weights:
        return score_map

    details = pattern_weights.get("details") if isinstance(pattern_weights, dict) else None
    if not isinstance(details, dict):
        return score_map

    for key in PATTERN_WEIGHT_KEYS:
        detail = details.get(key)
        if not isinstance(detail, dict):
            continue
        try:
            score_map[key] = float(detail.get("score_pattern", score_map[key]))
        except (TypeError, ValueError):
            continue
    return score_map


def compute_ma_slope_pct(series: pd.Series, *, lookback_days: int = MA_SLOPE_LOOKBACK_DAYS) -> float | None:
    """Return the percent change in a moving-average series over *lookback_days*."""
    if series is None:
        return None
    cleaned = pd.Series(series).dropna()
    if len(cleaned) <= int(lookback_days):
        return None
    latest = float(cleaned.iloc[-1])
    past = float(cleaned.iloc[-1 - int(lookback_days)])
    if past == 0:
        return None
    return ((latest / past) - 1.0) * 100.0


def compute_ma_slope_bonus(
    ma_slope_pct: float | None,
    *,
    bonus_cap: float = MA_SLOPE_BONUS_CAP,
) -> float:
    """Return an additive score bonus for positive moving-average slope."""
    if ma_slope_pct is None or pd.isna(ma_slope_pct):
        return 0.0
    slope = float(ma_slope_pct)
    if slope <= 0:
        return 0.0
    return round(min(float(bonus_cap), slope * 4.0), 2)


def apply_ma_slope_bonus(
    signal_score: float,
    ma_slope_pct: float | None,
    *,
    bonus_cap: float = MA_SLOPE_BONUS_CAP,
) -> tuple[float, float]:
    """Return (ma_slope_bonus, boosted_signal_score)."""
    bonus = compute_ma_slope_bonus(ma_slope_pct, bonus_cap=bonus_cap)
    return bonus, round(clip_score(float(signal_score) + bonus), 1)


def apply_pattern_family_bonus(
    signals_df: pd.DataFrame,
    pattern_weights: dict[str, float] | None,
) -> pd.DataFrame:
    """Apply or re-apply the historical pattern-family score contribution."""
    out = signals_df.copy()
    if out.empty:
        if "score_pattern" not in out.columns:
            out["score_pattern"] = pd.Series(dtype="float64")
        if "pattern_bonus" not in out.columns:
            out["pattern_bonus"] = pd.Series(dtype="float64")
        return out

    if "pattern_bonus" in out.columns:
        existing_bonus = pd.to_numeric(out["pattern_bonus"], errors="coerce").fillna(0.0)
    else:
        existing_bonus = pd.Series(0.0, index=out.index, dtype="float64")

    base_score = pd.to_numeric(out.get("signal_score"), errors="coerce")
    if not isinstance(base_score, pd.Series):
        base_score = pd.Series(0.0, index=out.index, dtype="float64")
    base_score = base_score.fillna(0.0) - existing_bonus

    weight_map = _coerce_pattern_contribution_map(pattern_weights)
    score_map = _coerce_pattern_score_map(pattern_weights, weight_map)

    if "pattern_family" in out.columns:
        families = out["pattern_family"].astype(str).str.strip().str.upper()
        new_bonus = families.map(weight_map).fillna(0.0).astype(float)
        score_pattern = families.map(score_map).fillna(0.0).astype(float)
    else:
        new_bonus = pd.Series(0.0, index=out.index, dtype="float64")
        score_pattern = pd.Series(0.0, index=out.index, dtype="float64")

    out["score_pattern"] = score_pattern.round(1)
    out["pattern_bonus"] = new_bonus.round(2)
    out["signal_score"] = (base_score + new_bonus).map(clip_score).round(1)
    return out


def apply_lt_health_modifier(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    bonus_step: float = LT_HEALTH_BONUS_STEP,
    max_abs_bonus: float = LT_HEALTH_BONUS_MAX_ABS,
) -> pd.DataFrame:
    """Apply a capped LT score adjustment from stock-health factors.

    Factors are evaluated on signal date with all data available:
    - SMA50 > SMA200
    - 20D return > 0
    - 60D return > 0
    - Distance from 52W high >= -12%

    The adjustment is intentionally small and bounded to avoid overwhelming
    the existing LT heuristic score.
    """

    out = signals_df.copy()
    if out.empty:
        return out
    if "signal_score" not in out.columns or "signal_date" not in out.columns or "ticker" not in out.columns:
        return out
    required_price_cols = {"Date", "Ticker", "Close"}
    if prices_df is None or prices_df.empty or not required_price_cols.issubset(set(prices_df.columns)):
        return out

    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.normalize()
    prices["Ticker"] = prices["Ticker"].astype(str).str.strip().str.upper()
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    if "High" in prices.columns:
        prices["High"] = pd.to_numeric(prices["High"], errors="coerce")

    prices.sort_values(["Ticker", "Date"], inplace=True)

    prices["sma50"] = prices.groupby("Ticker", sort=False)["Close"].transform(lambda s: s.rolling(50).mean())
    prices["sma200"] = prices.groupby("Ticker", sort=False)["Close"].transform(lambda s: s.rolling(200).mean())
    prices["ret_20d_pct"] = prices.groupby("Ticker", sort=False)["Close"].transform(
        lambda s: s.pct_change(20, fill_method=None) * 100.0
    )
    prices["ret_60d_pct"] = prices.groupby("Ticker", sort=False)["Close"].transform(
        lambda s: s.pct_change(60, fill_method=None) * 100.0
    )

    if "High" in prices.columns:
        rolling_high = prices.groupby("Ticker", sort=False)["High"].transform(lambda s: s.rolling(252).max())
    else:
        rolling_high = prices.groupby("Ticker", sort=False)["Close"].transform(lambda s: s.rolling(252).max())
    prices["dist_from_52w_high_pct"] = ((prices["Close"] / rolling_high) - 1.0) * 100.0

    feature_cols = [
        "Ticker",
        "Date",
        "sma50",
        "sma200",
        "ret_20d_pct",
        "ret_60d_pct",
        "dist_from_52w_high_pct",
    ]
    features = prices[feature_cols].copy()
    features.rename(columns={"Ticker": "_ticker", "Date": "_signal_date_dt"}, inplace=True)

    out["_ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["_signal_date_dt"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.normalize()
    out = out.merge(features, on=["_ticker", "_signal_date_dt"], how="left")

    valid = (
        out["sma50"].notna()
        & out["sma200"].notna()
        & out["ret_20d_pct"].notna()
        & out["ret_60d_pct"].notna()
        & out["dist_from_52w_high_pct"].notna()
    )

    points = pd.Series(2, index=out.index, dtype="int64")
    if valid.any():
        trend_ok = (out["sma50"] > out["sma200"]).astype("int64")
        ret20_ok = (out["ret_20d_pct"] > 0.0).astype("int64")
        ret60_ok = (out["ret_60d_pct"] > 0.0).astype("int64")
        high_ok = (out["dist_from_52w_high_pct"] >= -12.0).astype("int64")
        computed_points = trend_ok + ret20_ok + ret60_ok + high_ok
        points.loc[valid] = computed_points.loc[valid]

    bonus = ((points.astype(float) - 2.0) * float(bonus_step)).clip(
        lower=-abs(float(max_abs_bonus)),
        upper=abs(float(max_abs_bonus)),
    )
    bonus.loc[~valid] = 0.0

    base_score = pd.to_numeric(out.get("signal_score"), errors="coerce").fillna(0.0)
    out["signal_score_pre_lt_health"] = base_score.round(4)
    out["lt_health_points"] = points
    out["lt_health_bonus"] = bonus.round(2)
    out["signal_score"] = (base_score + bonus).map(clip_score).round(1)

    out.drop(
        columns=[
            "_ticker",
            "_signal_date_dt",
            "sma50",
            "sma200",
            "ret_20d_pct",
            "ret_60d_pct",
            "dist_from_52w_high_pct",
        ],
        inplace=True,
        errors="ignore",
    )
    return out


def build_score_components(
    *,
    trend_strength_pct: float,
    setup_strength_pct: float,
    volume_ratio: float,
    stop_pct_eff: float,
    rsi_value: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Return non-pattern score components and subtotal before family weighting."""
    score_trend = clip_score(50.0 + trend_strength_pct * 5.0)
    score_setup = clip_score(50.0 + setup_strength_pct * 8.0)
    score_volume = clip_score(40.0 + volume_ratio * 20.0)
    score_risk = clip_score(100.0 - stop_pct_eff * 6.0)

    score_rsi = score_rsi_sweet_spot(rsi_value)

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
