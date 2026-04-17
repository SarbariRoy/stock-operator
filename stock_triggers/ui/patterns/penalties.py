"""Feature extraction and learned penalty application for signal scores."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay

from .scoring import clip_score

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_SIGNAL_PENALTY_WEIGHTS_JSON = DATA_DIR / "signal_penalty_weights.json"
DEFAULT_RECENT_SIGNAL_LOOKBACK_DAYS = 20
DEFAULT_BREAKOUT_LOOKBACK_DAYS = 40
FEATURE_COLUMNS = [
    "feature_recent_signal_count",
    "feature_close_vs_prev_high_pct",
    "feature_close_vs_sma50_pct",
    "feature_gap_pct",
    "feature_range_vs_atr",
    "feature_gap_sequence_risk",
    "feature_exhaustion_risk",
]
PENALTY_COLUMNS = [
    "score_penalty_crowding",
    "score_penalty_extension",
    "score_penalty_gap_shock",
    "score_penalty_total",
]
FEATURE_TO_COMPONENT = {
    "feature_recent_signal_count": "score_penalty_crowding",
    "feature_close_vs_prev_high_pct": "score_penalty_extension",
    "feature_close_vs_sma50_pct": "score_penalty_extension",
    "feature_gap_pct": "score_penalty_gap_shock",
    "feature_range_vs_atr": "score_penalty_gap_shock",
    "feature_gap_sequence_risk": "score_penalty_gap_shock",
    "feature_exhaustion_risk": "score_penalty_extension",
}


def load_signal_penalty_weights(path: Path = DEFAULT_SIGNAL_PENALTY_WEIGHTS_JSON) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def get_recent_signal_lookback_days(payload: dict | None, *, default: int = DEFAULT_RECENT_SIGNAL_LOOKBACK_DAYS) -> int:
    if not isinstance(payload, dict):
        return int(default)
    try:
        return int(payload.get("recent_signal_lookback_days", default))
    except (TypeError, ValueError):
        return int(default)


def ensure_penalty_columns(signals_df: pd.DataFrame) -> pd.DataFrame:
    out = signals_df.copy()
    for column in FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    for column in PENALTY_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
    return out


def _prepare_price_feature_map(prices_df: pd.DataFrame, *, breakout_days: int) -> dict[str, pd.DataFrame]:
    if prices_df.empty:
        return {}

    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices.sort_values(["Ticker", "Date"], inplace=True)
    prices["PrevClose"] = prices.groupby("Ticker", sort=False)["Close"].shift(1)
    prices["PrevOpen"] = prices.groupby("Ticker", sort=False)["Open"].shift(1)
    prices["SMA20"] = prices.groupby("Ticker", sort=False)["Close"].transform(lambda s: s.rolling(20).mean())
    prices["SMA50"] = prices.groupby("Ticker", sort=False)["Close"].transform(lambda s: s.rolling(50).mean())
    prices["PrevBreakoutHighClose"] = prices.groupby("Ticker", sort=False)["Close"].transform(
        lambda s: s.shift(1).rolling(int(breakout_days)).max()
    )
    prices["Ret3dPct"] = prices.groupby("Ticker", sort=False)["Close"].pct_change(3) * 100.0
    prices["Ret5dPct"] = prices.groupby("Ticker", sort=False)["Close"].pct_change(5) * 100.0
    tr1 = prices["High"] - prices["Low"]
    tr2 = (prices["High"] - prices["PrevClose"]).abs()
    tr3 = (prices["Low"] - prices["PrevClose"]).abs()
    prices["TrueRange"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    prices["ATR14"] = prices.groupby("Ticker", sort=False)["TrueRange"].transform(lambda s: s.rolling(14).mean())
    prices["GapPct"] = ((prices["Open"] / prices["PrevClose"]) - 1.0) * 100.0
    prices["RangeVsATR"] = (prices["High"] - prices["Low"]) / prices["ATR14"]
    prices["PrevGapPct"] = prices.groupby("Ticker", sort=False)["GapPct"].shift(1)
    prices["PrevRangeVsATR"] = prices.groupby("Ticker", sort=False)["RangeVsATR"].shift(1)

    feature_map: dict[str, pd.DataFrame] = {}
    for ticker, grp in prices.groupby("Ticker", sort=False):
        feature_map[str(ticker)] = grp.set_index("Date").sort_index()
    return feature_map


def _resolve_price_history(feature_map: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame | None:
    clean = str(ticker).strip()
    if clean in feature_map:
        return feature_map[clean]
    if clean.endswith(".NS"):
        return feature_map.get(clean[:-3])
    return feature_map.get(clean + ".NS")


def _compute_recent_signal_counts(signals_df: pd.DataFrame, *, lookback_days: int) -> pd.Series:
    out = pd.Series(0, index=signals_df.index, dtype="int64")
    if signals_df.empty:
        return out

    working = signals_df.copy()
    working["signal_date_dt"] = pd.to_datetime(working["signal_date"], errors="coerce")
    working["ticker_key"] = working["ticker"].astype(str).str.strip().str.upper()
    working.sort_values(["ticker_key", "signal_date_dt"], inplace=True)

    for _, grp in working.groupby("ticker_key", sort=False):
        dates = list(grp["signal_date_dt"])
        indices = list(grp.index)
        left = 0
        for pos, current_date in enumerate(dates):
            if pd.isna(current_date):
                out.at[indices[pos]] = 0
                continue
            threshold = pd.Timestamp(current_date) - BDay(int(lookback_days))
            while left < pos and pd.Timestamp(dates[left]) <= threshold:
                left += 1
            out.at[indices[pos]] = max(0, pos - left)
    return out


def compute_signal_penalty_features(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    breakout_days: int = DEFAULT_BREAKOUT_LOOKBACK_DAYS,
    recent_signal_lookback_days: int = DEFAULT_RECENT_SIGNAL_LOOKBACK_DAYS,
) -> pd.DataFrame:
    out = ensure_penalty_columns(signals_df)
    if out.empty:
        return out

    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out["feature_recent_signal_count"] = _compute_recent_signal_counts(
        out,
        lookback_days=int(recent_signal_lookback_days),
    ).astype("Int64")

    feature_map = _prepare_price_feature_map(prices_df, breakout_days=int(breakout_days))
    for idx, row in out.iterrows():
        signal_date = pd.to_datetime(row.get("signal_date"), errors="coerce")
        if pd.isna(signal_date):
            continue
        hist = _resolve_price_history(feature_map, str(row.get("ticker", "")))
        if hist is None or signal_date not in hist.index:
            continue

        current = hist.loc[signal_date]
        if isinstance(current, pd.DataFrame):
            current = current.iloc[-1]

        close_value = pd.to_numeric(current.get("Close"), errors="coerce")
        prev_high = pd.to_numeric(current.get("PrevBreakoutHighClose"), errors="coerce")
        sma20 = pd.to_numeric(current.get("SMA20"), errors="coerce")
        sma50 = pd.to_numeric(current.get("SMA50"), errors="coerce")
        prev_close = pd.to_numeric(current.get("PrevClose"), errors="coerce")
        open_value = pd.to_numeric(current.get("Open"), errors="coerce")
        atr14 = pd.to_numeric(current.get("ATR14"), errors="coerce")
        high_value = pd.to_numeric(current.get("High"), errors="coerce")
        low_value = pd.to_numeric(current.get("Low"), errors="coerce")
        ret3d_pct = pd.to_numeric(current.get("Ret3dPct"), errors="coerce")
        ret5d_pct = pd.to_numeric(current.get("Ret5dPct"), errors="coerce")
        prev_gap_pct = pd.to_numeric(current.get("PrevGapPct"), errors="coerce")
        prev_range_vs_atr = pd.to_numeric(current.get("PrevRangeVsATR"), errors="coerce")

        close_vs_prev_high_pct = None
        if pd.notna(close_value) and pd.notna(prev_high) and float(prev_high) > 0:
            close_vs_prev_high_pct = ((float(close_value) / float(prev_high)) - 1.0) * 100.0
            out.at[idx, "feature_close_vs_prev_high_pct"] = round(close_vs_prev_high_pct, 4)
        close_vs_sma20_pct = None
        if pd.notna(close_value) and pd.notna(sma20) and float(sma20) > 0:
            close_vs_sma20_pct = ((float(close_value) / float(sma20)) - 1.0) * 100.0
        if pd.notna(close_value) and pd.notna(sma50) and float(sma50) > 0:
            out.at[idx, "feature_close_vs_sma50_pct"] = round(((float(close_value) / float(sma50)) - 1.0) * 100.0, 4)
        close_vs_sma50_pct = pd.to_numeric(out.at[idx, "feature_close_vs_sma50_pct"], errors="coerce")
        gap_pct_value = None
        if pd.notna(open_value) and pd.notna(prev_close) and float(prev_close) > 0:
            gap_pct_value = ((float(open_value) / float(prev_close)) - 1.0) * 100.0
            out.at[idx, "feature_gap_pct"] = round(gap_pct_value, 4)
        range_vs_atr_value = None
        if pd.notna(high_value) and pd.notna(low_value) and pd.notna(atr14) and float(atr14) > 0:
            range_vs_atr_value = (float(high_value) - float(low_value)) / float(atr14)
            out.at[idx, "feature_range_vs_atr"] = round(range_vs_atr_value, 4)

        gap_sequence_risk = (
            max(0.0, float(gap_pct_value or 0.0))
            + 0.7 * max(0.0, float(prev_gap_pct or 0.0))
            + max(0.0, float((range_vs_atr_value or 0.0) - 1.0))
            + 0.7 * max(0.0, float((prev_range_vs_atr or 0.0) - 1.0))
        )
        if gap_sequence_risk > 0.0:
            out.at[idx, "feature_gap_sequence_risk"] = round(gap_sequence_risk, 4)

        exhaustion_risk = (
            max(0.0, float(close_vs_prev_high_pct or 0.0))
            + 0.6 * max(0.0, float(close_vs_sma20_pct or 0.0))
            + 0.3 * max(0.0, float(close_vs_sma50_pct or 0.0))
            + 0.45 * max(0.0, float(ret3d_pct or 0.0))
            + 0.25 * max(0.0, float(ret5d_pct or 0.0))
        )
        if exhaustion_risk > 0.0:
            out.at[idx, "feature_exhaustion_risk"] = round(exhaustion_risk, 4)

    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.date.astype("string")
    return out


def _lookup_feature_penalty(payload: dict, *, feature_name: str, pattern_family: str, value: object) -> float:
    if not isinstance(payload, dict) or pd.isna(value):
        return 0.0
    features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    feature_spec = features.get(feature_name)
    if not isinstance(feature_spec, dict):
        return 0.0

    families = feature_spec.get("families") if isinstance(feature_spec.get("families"), dict) else {}
    family_spec = families.get(pattern_family) or families.get("__global__")
    if not isinstance(family_spec, dict):
        return 0.0

    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return 0.0

    for bucket in family_spec.get("buckets", []):
        if not isinstance(bucket, dict):
            continue
        lower = bucket.get("lower")
        upper = bucket.get("upper")
        include_upper = bool(bucket.get("include_upper", False))
        lower_ok = True if lower is None else float(numeric_value) >= float(lower)
        if upper is None:
            upper_ok = True
        elif include_upper:
            upper_ok = float(numeric_value) <= float(upper)
        else:
            upper_ok = float(numeric_value) < float(upper)
        if lower_ok and upper_ok:
            try:
                return float(bucket.get("penalty", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def apply_signal_penalty_weights(signals_df: pd.DataFrame, payload: dict | None) -> pd.DataFrame:
    out = ensure_penalty_columns(signals_df)
    if out.empty:
        return out

    existing_total = pd.to_numeric(out.get("score_penalty_total"), errors="coerce").fillna(0.0)
    base_score = pd.to_numeric(out.get("signal_score"), errors="coerce").fillna(0.0) - existing_total

    crowding_values: list[float] = []
    extension_values: list[float] = []
    gap_values: list[float] = []
    total_values: list[float] = []

    for _, row in out.iterrows():
        family = str(row.get("pattern_family", "")).strip().upper()
        crowding_penalty = _lookup_feature_penalty(
            payload or {},
            feature_name="feature_recent_signal_count",
            pattern_family=family,
            value=row.get("feature_recent_signal_count"),
        )
        extension_penalty = _lookup_feature_penalty(
            payload or {},
            feature_name="feature_close_vs_prev_high_pct",
            pattern_family=family,
            value=row.get("feature_close_vs_prev_high_pct"),
        ) + _lookup_feature_penalty(
            payload or {},
            feature_name="feature_close_vs_sma50_pct",
            pattern_family=family,
            value=row.get("feature_close_vs_sma50_pct"),
        ) + _lookup_feature_penalty(
            payload or {},
            feature_name="feature_exhaustion_risk",
            pattern_family=family,
            value=row.get("feature_exhaustion_risk"),
        )
        gap_penalty = _lookup_feature_penalty(
            payload or {},
            feature_name="feature_gap_pct",
            pattern_family=family,
            value=row.get("feature_gap_pct"),
        ) + _lookup_feature_penalty(
            payload or {},
            feature_name="feature_range_vs_atr",
            pattern_family=family,
            value=row.get("feature_range_vs_atr"),
        ) + _lookup_feature_penalty(
            payload or {},
            feature_name="feature_gap_sequence_risk",
            pattern_family=family,
            value=row.get("feature_gap_sequence_risk"),
        )
        total_penalty = crowding_penalty + extension_penalty + gap_penalty

        crowding_values.append(round(float(crowding_penalty), 2))
        extension_values.append(round(float(extension_penalty), 2))
        gap_values.append(round(float(gap_penalty), 2))
        total_values.append(round(float(total_penalty), 2))

    out["score_penalty_crowding"] = crowding_values
    out["score_penalty_extension"] = extension_values
    out["score_penalty_gap_shock"] = gap_values
    out["score_penalty_total"] = total_values
    out["signal_score"] = (base_score + pd.Series(total_values, index=out.index, dtype="float64")).map(clip_score).round(1)
    return out