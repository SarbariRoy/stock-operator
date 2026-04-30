from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_triggers.scripts.generate_stock_scores import compute_rsi
from stock_triggers.ui.patterns import pattern_a, pattern_b, pattern_c_macd, pattern_d_rsi, pattern_e_boll, pattern_f_vwap, pattern_g_vcp


DEFAULT_TARGET_RETURN_PCT = 6.0
DEFAULT_FORWARD_DAYS = 60
DEFAULT_RECOGNITION_THRESHOLD = 80
DEFAULT_PATTERN_FAMILIES = ("A", "B", "C", "D", "E", "F", "G")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES_CSV = DATA_DIR / "st_lt_prices_eod.csv"
DEFAULT_SIGNALS_CSV = DATA_DIR / "st_signals_all_patterns.csv"
DEFAULT_CACHE_PKL = DATA_DIR / "coverage_default_cache.pkl"

RAW_COLS = [
    "signal_date",
    "ticker",
    "pattern",
    "pattern_family",
    "entry_price",
    "stop_pct",
    "stop_price",
]

SAVED_SIGNAL_COLS = [
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


def _source_signature(*paths: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for path in paths:
        if path.exists():
            stat = path.stat()
            out[str(path)] = {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}
        else:
            out[str(path)] = {"mtime_ns": -1, "size": -1}
    return out


def _append_if_not_empty(rows: list[pd.DataFrame], frame: pd.DataFrame) -> None:
    if not frame.empty:
        rows.append(frame[[col for col in RAW_COLS if col in frame.columns]].copy())


def scan_raw_breakouts_for_date(
    prices_df: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    pattern_families: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    selected = {str(family).strip().upper() for family in pattern_families if str(family).strip()}

    if "A" in selected:
        _append_if_not_empty(
            rows,
            pattern_a.detect(
                prices_df,
                as_of_date=as_of_date,
                breakout_days=40,
                volume_multiplier=1.0,
                stop_pct=7.0,
            ),
        )
    if "B" in selected:
        _append_if_not_empty(
            rows,
            pattern_b.detect(
                prices_df,
                as_of_date=as_of_date,
                volume_multiplier=1.0,
                stop_pct=7.0,
                pullback_buffer_pct=1.5,
                rebound_min_pct=0.2,
                compute_rsi_fn=compute_rsi,
            ),
        )
    if "C" in selected:
        _append_if_not_empty(
            rows,
            pattern_c_macd.detect(
                prices_df,
                as_of_date=as_of_date,
                volume_multiplier=1.0,
                stop_pct=7.0,
                compute_rsi_fn=compute_rsi,
            ),
        )
    if "D" in selected:
        _append_if_not_empty(
            rows,
            pattern_d_rsi.detect(
                prices_df,
                as_of_date=as_of_date,
                volume_multiplier=1.0,
                stop_pct=7.0,
                compute_rsi_fn=compute_rsi,
            ),
        )
    if "E" in selected:
        _append_if_not_empty(
            rows,
            pattern_e_boll.detect(
                prices_df,
                as_of_date=as_of_date,
                volume_multiplier=1.0,
                stop_pct=7.0,
                compute_rsi_fn=compute_rsi,
            ),
        )
    if "F" in selected:
        _append_if_not_empty(
            rows,
            pattern_f_vwap.detect(
                prices_df,
                as_of_date=as_of_date,
                volume_multiplier=1.0,
                stop_pct=7.0,
                compute_rsi_fn=compute_rsi,
            ),
        )
    if "G" in selected:
        _append_if_not_empty(
            rows,
            pattern_g_vcp.detect(
                prices_df,
                as_of_date=as_of_date,
                volume_multiplier=1.0,
                stop_pct=7.0,
                base_lookback=100,
                dryup_volume_ratio=1.0,
                compute_rsi_fn=compute_rsi,
            ),
        )

    if not rows:
        return pd.DataFrame(columns=RAW_COLS)
    return pd.concat(rows, ignore_index=True)


def scan_raw_breakout_candidates(
    prices_df: pd.DataFrame,
    pattern_families: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_dates = pd.to_datetime(prices_df["Date"], errors="coerce").dropna()
    all_dates = sorted(d for d in all_dates.unique() if start_ts <= pd.Timestamp(d) <= end_ts)

    rows: list[pd.DataFrame] = []
    for as_of_date in all_dates:
        detected = scan_raw_breakouts_for_date(
            prices_df,
            as_of_date=pd.Timestamp(as_of_date),
            pattern_families=pattern_families,
        )
        if not detected.empty:
            rows.append(detected)

    if not rows:
        return pd.DataFrame(columns=RAW_COLS)

    out = pd.concat(rows, ignore_index=True)
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out["pattern_family"] = out["pattern_family"].astype(str).str.strip().str.upper()
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out = out.dropna(subset=["signal_date", "ticker", "pattern_family", "entry_price"])
    out = out.drop_duplicates(subset=["signal_date", "ticker", "pattern_family", "pattern", "entry_price"])
    return out.reset_index(drop=True)


def evaluate_raw_breakout_targets(
    raw_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_return_pct: float,
    forward_days: int,
) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=[
            *RAW_COLS,
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
                **{col: row.get(col) for col in RAW_COLS},
                "target_return_pct": round(float(target_return_pct), 2),
                "target_price": round(float(target_price), 4) if pd.notna(target_price) else pd.NA,
                "bars_available_forward": bars_available_forward,
                "first_target_hit_day": first_target_hit_day,
                "is_breakout": bool(is_breakout),
                "is_pending": bool(is_pending),
            }
        )

    return pd.DataFrame(out_rows)


def build_coverage_view(
    prices_df: pd.DataFrame,
    all_signals_df: pd.DataFrame,
    *,
    pattern_families: tuple[str, ...] = DEFAULT_PATTERN_FAMILIES,
    start_date: str,
    end_date: str,
    target_return_pct: float = DEFAULT_TARGET_RETURN_PCT,
    forward_days: int = DEFAULT_FORWARD_DAYS,
    score_threshold: int = DEFAULT_RECOGNITION_THRESHOLD,
) -> tuple[pd.DataFrame, int]:
    raw_candidates = scan_raw_breakout_candidates(prices_df, pattern_families, start_date, end_date)
    evaluated = evaluate_raw_breakout_targets(
        raw_candidates,
        prices_df,
        target_return_pct=target_return_pct,
        forward_days=forward_days,
    )
    breakout_df = evaluated[evaluated["is_breakout"]].copy()
    pending_count = int(evaluated["is_pending"].sum()) if "is_pending" in evaluated.columns else 0
    if breakout_df.empty:
        return breakout_df, pending_count

    if all_signals_df.empty:
        saved_best = pd.DataFrame(columns=SAVED_SIGNAL_COLS)
    else:
        saved_best = all_signals_df.copy()
        saved_best["signal_date"] = pd.to_datetime(saved_best["signal_date"], errors="coerce")
        saved_best["pattern_family"] = saved_best["pattern_family"].astype(str).str.strip().str.upper()
        saved_best["ticker"] = saved_best["ticker"].astype(str).str.strip()
        saved_best["signal_score"] = pd.to_numeric(saved_best.get("signal_score"), errors="coerce")
        saved_best = saved_best[saved_best["pattern_family"].isin(pattern_families)]
        start_dt = pd.Timestamp(start_date).date()
        end_dt = pd.Timestamp(end_date).date()
        saved_best = saved_best[saved_best["signal_date"].dt.date >= start_dt]
        saved_best = saved_best[saved_best["signal_date"].dt.date <= end_dt]
        saved_best = saved_best.sort_values("signal_score", ascending=False, na_position="last")
        saved_best = saved_best[[col for col in SAVED_SIGNAL_COLS if col in saved_best.columns]].drop_duplicates(
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
    return df, pending_count


def build_and_save_default_cache(
    *,
    prices_path: Path = DEFAULT_PRICES_CSV,
    signals_path: Path = DEFAULT_SIGNALS_CSV,
    cache_path: Path = DEFAULT_CACHE_PKL,
) -> dict:
    prices_df = pd.read_csv(prices_path, parse_dates=["Date"])
    all_signals_df = pd.read_csv(signals_path) if signals_path.exists() else pd.DataFrame()
    all_dates = pd.to_datetime(prices_df["Date"], errors="coerce").dropna()
    if all_dates.empty:
        raise ValueError("Price history is empty or missing valid Date rows.")

    start_date = pd.Timestamp(all_dates.min()).date().isoformat()
    end_date = pd.Timestamp(all_dates.max()).date().isoformat()
    df, pending_count = build_coverage_view(
        prices_df,
        all_signals_df,
        pattern_families=DEFAULT_PATTERN_FAMILIES,
        start_date=start_date,
        end_date=end_date,
        target_return_pct=DEFAULT_TARGET_RETURN_PCT,
        forward_days=DEFAULT_FORWARD_DAYS,
        score_threshold=DEFAULT_RECOGNITION_THRESHOLD,
    )
    payload = {
        "meta": {
            "target_return_pct": DEFAULT_TARGET_RETURN_PCT,
            "forward_days": DEFAULT_FORWARD_DAYS,
            "score_threshold": DEFAULT_RECOGNITION_THRESHOLD,
            "pattern_families": list(DEFAULT_PATTERN_FAMILIES),
            "start_date": start_date,
            "end_date": end_date,
            "pending_count": pending_count,
            "source_signature": _source_signature(prices_path, signals_path),
        },
        "df": df,
    }
    pd.to_pickle(payload, cache_path)
    return payload


def load_default_cache_if_valid(
    *,
    prices_path: Path = DEFAULT_PRICES_CSV,
    signals_path: Path = DEFAULT_SIGNALS_CSV,
    cache_path: Path = DEFAULT_CACHE_PKL,
) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        payload = pd.read_pickle(cache_path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    meta = payload.get("meta") or {}
    if meta.get("target_return_pct") != DEFAULT_TARGET_RETURN_PCT:
        return None
    if meta.get("forward_days") != DEFAULT_FORWARD_DAYS:
        return None
    if meta.get("score_threshold") != DEFAULT_RECOGNITION_THRESHOLD:
        return None
    if tuple(meta.get("pattern_families") or ()) != DEFAULT_PATTERN_FAMILIES:
        return None
    if meta.get("source_signature") != _source_signature(prices_path, signals_path):
        return None
    return payload