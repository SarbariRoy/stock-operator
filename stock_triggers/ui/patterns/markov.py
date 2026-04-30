from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .scoring import clip_score

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_SIGNAL_MARKOV_MODEL_JSON = DATA_DIR / "st_lt_signal_markov_model.json"
MARKOV_OUTPUT_COLUMNS = [
    "markov_state",
    "markov_p_continuation",
    "markov_p_adverse",
    "score_markov_adjustment",
    "signal_score_pre_markov",
]
MARKOV_STATE_LEVELS = (
    "constructive_trend",
    "fresh_breakout",
    "extended_breakout",
    "sideways",
    "breakdown_risk",
)
DEFAULT_MARKOV_SCORE_POLICY = {
    "enabled": False,
    "horizon_days": 5,
    "max_bonus": 4.0,
    "max_penalty": 6.0,
    "continuation_center": 0.55,
    "adverse_center": 0.35,
    "continuation_states": ["constructive_trend", "fresh_breakout"],
    "adverse_states": ["extended_breakout", "breakdown_risk"],
}


def load_signal_markov_model(path: Path = DEFAULT_SIGNAL_MARKOV_MODEL_JSON) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def get_default_markov_score_policy() -> dict[str, object]:
    return dict(DEFAULT_MARKOV_SCORE_POLICY)


def ensure_markov_columns(signals_df: pd.DataFrame) -> pd.DataFrame:
    out = signals_df.copy()
    for column in MARKOV_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def _compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    no_loss_mask = avg_loss.eq(0.0) & avg_gain.notna()
    rsi.loc[no_loss_mask] = 100.0
    return rsi.astype("float64")


def build_markov_state_table(prices_df: pd.DataFrame) -> pd.DataFrame:
    if prices_df.empty:
        return pd.DataFrame(columns=["ticker", "signal_date", "markov_state"])

    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date"]).copy()
    if prices.empty:
        return pd.DataFrame(columns=["ticker", "signal_date", "markov_state"])

    prices.sort_values(["Ticker", "Date"], inplace=True)
    grouped = prices.groupby("Ticker", sort=False)
    prices["SMA20"] = grouped["Close"].transform(lambda s: s.rolling(20).mean())
    prices["SMA50"] = grouped["Close"].transform(lambda s: s.rolling(50).mean())
    prices["Prev20High"] = grouped["Close"].transform(lambda s: s.shift(1).rolling(20).max())
    prices["Ret20dPct"] = grouped["Close"].transform(lambda s: ((s / s.shift(20)) - 1.0) * 100.0)
    prices["CloseVsSMA20Pct"] = ((prices["Close"] / prices["SMA20"]) - 1.0) * 100.0
    prices["CloseVsSMA50Pct"] = ((prices["Close"] / prices["SMA50"]) - 1.0) * 100.0
    prices["BreakoutGapPct"] = ((prices["Close"] / prices["Prev20High"]) - 1.0) * 100.0
    prices["RSI14"] = grouped["Close"].transform(_compute_rsi_series)

    prices["markov_state"] = "sideways"

    breakdown_mask = (
        prices["CloseVsSMA50Pct"].le(-4.0)
        | ((prices["Ret20dPct"].le(-8.0)) & prices["CloseVsSMA50Pct"].lt(0.0))
    )
    fresh_breakout_mask = (
        prices["BreakoutGapPct"].ge(-0.5)
        & prices["CloseVsSMA50Pct"].ge(0.0)
        & prices["RSI14"].lt(72.0)
    )
    extended_breakout_mask = (
        prices["BreakoutGapPct"].ge(-0.5)
        & ((prices["CloseVsSMA20Pct"].ge(6.0)) | prices["RSI14"].ge(72.0))
    )
    constructive_trend_mask = (
        prices["CloseVsSMA50Pct"].ge(0.0)
        & prices["Ret20dPct"].ge(2.0)
        & prices["CloseVsSMA20Pct"].between(-2.0, 6.0, inclusive="both")
    )

    prices.loc[constructive_trend_mask, "markov_state"] = "constructive_trend"
    prices.loc[fresh_breakout_mask, "markov_state"] = "fresh_breakout"
    prices.loc[extended_breakout_mask, "markov_state"] = "extended_breakout"
    prices.loc[breakdown_mask, "markov_state"] = "breakdown_risk"

    state_table = prices[["Ticker", "Date", "markov_state"]].copy()
    state_table.rename(columns={"Ticker": "ticker"}, inplace=True)
    state_table["ticker"] = state_table["ticker"].astype(str).str.strip().str.upper()
    state_table["signal_date"] = pd.to_datetime(state_table["Date"], errors="coerce").dt.date.astype("string")
    state_table = state_table[["ticker", "signal_date", "markov_state"]].dropna(subset=["signal_date"])

    alias_rows = state_table[state_table["ticker"].str.endswith(".NS")].copy()
    if not alias_rows.empty:
        alias_rows["ticker"] = alias_rows["ticker"].str.replace(r"\.NS$", "", regex=True)
        state_table = pd.concat([state_table, alias_rows], ignore_index=True)

    state_table.drop_duplicates(subset=["ticker", "signal_date"], keep="first", inplace=True)
    return state_table


def _resolve_markov_mode(payload: dict | None, markov_mode: str) -> bool:
    mode = str(markov_mode or "auto").strip().lower()
    if mode == "on":
        return True
    if mode == "off":
        return False
    if isinstance(payload, dict):
        score_policy = payload.get("score_policy")
        if isinstance(score_policy, dict):
            return bool(score_policy.get("enabled", DEFAULT_MARKOV_SCORE_POLICY["enabled"]))
        return bool(payload.get("enabled", DEFAULT_MARKOV_SCORE_POLICY["enabled"]))
    return False


def _resolve_markov_score_policy(payload: dict | None) -> dict[str, object]:
    resolved = get_default_markov_score_policy()
    raw_policy = payload.get("score_policy") if isinstance(payload, dict) and isinstance(payload.get("score_policy"), dict) else {}
    resolved.update(raw_policy)
    resolved["enabled"] = bool(resolved.get("enabled", DEFAULT_MARKOV_SCORE_POLICY["enabled"]))
    resolved["horizon_days"] = max(1, int(resolved.get("horizon_days", DEFAULT_MARKOV_SCORE_POLICY["horizon_days"])))
    resolved["max_bonus"] = max(0.0, float(resolved.get("max_bonus", DEFAULT_MARKOV_SCORE_POLICY["max_bonus"])))
    resolved["max_penalty"] = max(0.0, float(resolved.get("max_penalty", DEFAULT_MARKOV_SCORE_POLICY["max_penalty"])))
    resolved["continuation_center"] = max(0.0, min(1.0, float(resolved.get("continuation_center", DEFAULT_MARKOV_SCORE_POLICY["continuation_center"]))))
    resolved["adverse_center"] = max(0.0, min(1.0, float(resolved.get("adverse_center", DEFAULT_MARKOV_SCORE_POLICY["adverse_center"]))))
    continuation_states = resolved.get("continuation_states", DEFAULT_MARKOV_SCORE_POLICY["continuation_states"])
    adverse_states = resolved.get("adverse_states", DEFAULT_MARKOV_SCORE_POLICY["adverse_states"])
    resolved["continuation_states"] = [str(state) for state in continuation_states if str(state)]
    resolved["adverse_states"] = [str(state) for state in adverse_states if str(state)]
    return resolved


def _transition_probabilities_for_state(
    state: str,
    transitions: dict[str, dict[str, float]],
    *,
    continuation_states: list[str],
    adverse_states: list[str],
) -> tuple[float, float]:
    row = transitions.get(str(state), {}) if isinstance(transitions, dict) else {}
    if not isinstance(row, dict):
        return 0.0, 0.0

    p_continuation = 0.0
    p_adverse = 0.0
    for next_state, raw_prob in row.items():
        try:
            prob = float(raw_prob)
        except (TypeError, ValueError):
            continue
        if next_state in continuation_states:
            p_continuation += prob
        if next_state in adverse_states:
            p_adverse += prob
    return float(max(0.0, min(1.0, p_continuation))), float(max(0.0, min(1.0, p_adverse)))


def _compute_markov_adjustment(
    p_continuation: float,
    p_adverse: float,
    *,
    max_bonus: float,
    max_penalty: float,
    continuation_center: float,
    adverse_center: float,
) -> float:
    continuation_push = max(0.0, float(p_continuation) - float(continuation_center))
    adverse_push = max(0.0, float(p_adverse) - float(adverse_center))
    continuation_scale = max(1e-6, 1.0 - float(continuation_center))
    adverse_scale = max(1e-6, 1.0 - float(adverse_center))
    bonus = (continuation_push / continuation_scale) * float(max_bonus)
    penalty = (adverse_push / adverse_scale) * float(max_penalty)
    return round(max(-float(max_penalty), min(float(max_bonus), bonus - penalty)), 2)


def apply_signal_markov_model(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    payload: dict | None,
    *,
    markov_mode: str = "auto",
) -> pd.DataFrame:
    out = ensure_markov_columns(signals_df)
    if out.empty:
        return out

    current_score = pd.to_numeric(out.get("signal_score"), errors="coerce").fillna(0.0)
    existing_adjustment = pd.to_numeric(out.get("score_markov_adjustment"), errors="coerce").fillna(0.0)
    base_score = (current_score - existing_adjustment).astype("float64")
    out["signal_score_pre_markov"] = base_score.round(4)

    enabled = _resolve_markov_mode(payload, markov_mode)
    if not enabled or not isinstance(payload, dict):
        out["markov_state"] = out.get("markov_state", pd.Series(pd.NA, index=out.index))
        out["markov_p_continuation"] = 0.0
        out["markov_p_adverse"] = 0.0
        out["score_markov_adjustment"] = 0.0
        out["signal_score"] = base_score.map(clip_score).round(1)
        return out

    transitions = payload.get("transitions") if isinstance(payload.get("transitions"), dict) else {}
    if not transitions:
        out["markov_p_continuation"] = 0.0
        out["markov_p_adverse"] = 0.0
        out["score_markov_adjustment"] = 0.0
        out["signal_score"] = base_score.map(clip_score).round(1)
        return out

    policy = _resolve_markov_score_policy(payload)
    state_table = build_markov_state_table(prices_df)
    if state_table.empty:
        out["markov_p_continuation"] = 0.0
        out["markov_p_adverse"] = 0.0
        out["score_markov_adjustment"] = 0.0
        out["signal_score"] = base_score.map(clip_score).round(1)
        return out

    merge_df = out.copy()
    merge_df["ticker"] = merge_df.get("ticker", pd.Series("", index=merge_df.index)).astype(str).str.strip().str.upper()
    merge_df["signal_date"] = pd.to_datetime(merge_df.get("signal_date"), errors="coerce").dt.date.astype("string")
    merge_df = merge_df.merge(state_table, on=["ticker", "signal_date"], how="left", suffixes=("", "_new"))
    if "markov_state_new" in merge_df.columns:
        merge_df["markov_state"] = merge_df["markov_state_new"].combine_first(merge_df.get("markov_state"))
        merge_df.drop(columns=["markov_state_new"], inplace=True)

    p_continuation: list[float] = []
    p_adverse: list[float] = []
    adjustments: list[float] = []
    continuation_states = list(policy.get("continuation_states", []))
    adverse_states = list(policy.get("adverse_states", []))
    for state in merge_df.get("markov_state", pd.Series(pd.NA, index=merge_df.index)).fillna("sideways"):
        state_name = str(state)
        p_cont, p_adv = _transition_probabilities_for_state(
            state_name,
            transitions,
            continuation_states=continuation_states,
            adverse_states=adverse_states,
        )
        p_continuation.append(round(p_cont, 4))
        p_adverse.append(round(p_adv, 4))
        adjustments.append(
            _compute_markov_adjustment(
                p_cont,
                p_adv,
                max_bonus=float(policy.get("max_bonus", DEFAULT_MARKOV_SCORE_POLICY["max_bonus"])),
                max_penalty=float(policy.get("max_penalty", DEFAULT_MARKOV_SCORE_POLICY["max_penalty"])),
                continuation_center=float(policy.get("continuation_center", DEFAULT_MARKOV_SCORE_POLICY["continuation_center"])),
                adverse_center=float(policy.get("adverse_center", DEFAULT_MARKOV_SCORE_POLICY["adverse_center"])),
            )
        )

    out["markov_state"] = merge_df.get("markov_state", pd.Series(pd.NA, index=out.index))
    out["markov_p_continuation"] = pd.Series(p_continuation, index=out.index, dtype="float64")
    out["markov_p_adverse"] = pd.Series(p_adverse, index=out.index, dtype="float64")
    out["score_markov_adjustment"] = pd.Series(adjustments, index=out.index, dtype="float64")
    out["signal_score"] = (base_score + out["score_markov_adjustment"].astype(float)).map(clip_score).round(1)
    return out