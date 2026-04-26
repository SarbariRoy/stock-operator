"""Generate combined buy signals across Pattern A-G from OHLCV data.

Output is written to stock_triggers/data/signals_all_patterns.csv.
By default the script generates signals for the latest available date and
merges them into the output file. Use --backfill-history to build the full
available history.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_stock_scores import compute_rsi
from stock_triggers.ui.patterns import STANDARD_SIGNAL_COLS
from stock_triggers.ui.patterns import pattern_a, pattern_b, pattern_c_macd, pattern_d_rsi, pattern_e_boll, pattern_f_vwap, pattern_g_vcp
from stock_triggers.ui.patterns.catalyst_enrichment import enrich_signals_with_catalysts, load_external_factors, load_event_calendar
from stock_triggers.ui.patterns.markov import ensure_markov_columns, load_signal_markov_model
from stock_triggers.ui.patterns.penalties import ensure_penalty_columns, load_signal_penalty_weights
from stock_triggers.ui.patterns.publish import load_existing_signal_history, rescore_signal_history
from stock_triggers.ui.patterns.scoring import apply_ma_slope_bonus, apply_pattern_family_bonus, build_score_components, clip_score, compute_ma_slope_pct
from stock_triggers.ui.patterns.st_score import (
    DEFAULT_ST_SCORE_MODEL_JSON,
    DEFAULT_ST_SCORE_RF_MODEL_JSON,
    DEFAULT_ST_SCORE_SVM_MODEL_JSON,
    DEFAULT_ST_SCORE_XGB_MODEL_JSON,
    build_st_score_payload,
)
from stock_triggers.ui.patterns.stop_risk import ensure_stop_risk_columns, load_signal_stop_risk_model

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "prices_eod.csv"
DEFAULT_SIGNALS = DATA_DIR / "signals_all_patterns.csv"
DEFAULT_MARKOV_MODEL = DATA_DIR / "signal_markov_model.json"
PATTERN_WEIGHTS_JSON = DATA_DIR / "pattern_weights.json"
BENCHMARK_TICKERS = {"^NSEI"}
DEFAULT_PATTERN_FAMILIES = ("A", "B", "C", "D", "E", "F", "G")


def _is_benchmark_ticker(ticker: str) -> bool:
    return str(ticker).strip().upper() in BENCHMARK_TICKERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate combined Pattern A-G buy signals from OHLCV data")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES), help="Input prices CSV path")
    parser.add_argument("--out", type=str, default=str(DEFAULT_SIGNALS), help="Output signals CSV path")
    parser.add_argument("--as-of-date", type=str, default=None, help="Signal date YYYY-MM-DD (default: latest date)")
    parser.add_argument(
        "--backfill-history",
        action="store_true",
        help="Generate buy signals for all available dates in prices file (appended with de-dup).",
    )
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="Reload the existing output file and reapply learned family weights, penalties, and stop-risk columns.",
    )
    parser.add_argument("--breakout-days", type=int, default=40, help="Pattern A breakout lookback window in trading days")
    parser.add_argument("--volume-multiplier", type=float, default=1.5, help="Volume threshold versus 20-day average")
    parser.add_argument("--stop-pct", type=float, default=7.0, help="Initial stop loss percent below entry")
    parser.add_argument("--pullback-buffer-pct", type=float, default=1.5, help="Pattern B pullback proximity buffer percent")
    parser.add_argument("--rebound-min-pct", type=float, default=0.2, help="Pattern B rebound confirmation percent")
    parser.add_argument("--consensus-bonus", type=float, default=5.0, help="Score bonus when multiple pattern families agree")
    parser.add_argument(
        "--markov-mode",
        type=str,
        choices=("auto", "on", "off"),
        default="auto",
        help="Apply Markov regime adjustment: auto follows artifact enabled flag, on forces it, off disables it.",
    )
    parser.add_argument(
        "--markov-model",
        type=str,
        default=str(DEFAULT_MARKOV_MODEL),
        help="Path to the Markov regime model JSON artifact.",
    )
    parser.add_argument(
        "--pattern-families",
        type=str,
        default=",".join(DEFAULT_PATTERN_FAMILIES),
        help="Comma-separated pattern families to include (default: A,B,C,D,E,F,G)",
    )
    parser.add_argument(
        "--catalyst-enrichment",
        action="store_true",
        help="Attach market-regime and company-event catalyst features to signals (Phase 2).",
    )
    parser.add_argument(
        "--external-factors",
        type=str,
        default=str(DATA_DIR / "external_factors.csv"),
        help="Path to external factors CSV.",
    )
    parser.add_argument(
        "--event-calendar",
        type=str,
        default=str(DATA_DIR / "event_calendar.csv"),
        help="Path to event calendar CSV.",
    )
    parser.add_argument(
        "--st-model-mode",
        type=str,
        choices=("auto", "logistic", "svm", "rf", "xgboost", "hybrid", "hybrid3", "hybrid4"),
        default="hybrid4",
        help="ST model mode (default: hybrid4). hybrid4 blends logistic+svm+rf+xgboost.",
    )
    parser.add_argument(
        "--st-logistic-model",
        type=str,
        default=str(DEFAULT_ST_SCORE_MODEL_JSON),
        help="Path to logistic ST model artifact JSON.",
    )
    parser.add_argument(
        "--st-svm-model",
        type=str,
        default=str(DEFAULT_ST_SCORE_SVM_MODEL_JSON),
        help="Path to SVM ST model artifact JSON.",
    )
    parser.add_argument(
        "--st-rf-model",
        type=str,
        default=str(DEFAULT_ST_SCORE_RF_MODEL_JSON),
        help="Path to Random Forest ST model artifact JSON.",
    )
    parser.add_argument(
        "--st-xgb-model",
        type=str,
        default=str(DEFAULT_ST_SCORE_XGB_MODEL_JSON),
        help="Path to XGBoost ST model artifact JSON.",
    )
    parser.add_argument(
        "--st-svm-weight",
        type=float,
        default=0.25,
        help="Hybrid/hybrid3 blend weight for SVM probability in [0,1].",
    )
    parser.add_argument(
        "--st-rf-weight",
        type=float,
        default=0.25,
        help="Hybrid3 blend weight for RF probability in [0,1].",
    )
    parser.add_argument(
        "--st-xgb-weight",
        type=float,
        default=0.25,
        help="Hybrid4 blend weight for XGBoost probability in [0,1].",
    )
    return parser.parse_args()


def _resolve_pattern_families(raw_value: str) -> set[str]:
    requested = {part.strip().upper() for part in str(raw_value or "").split(",") if part.strip()}
    allowed = set(DEFAULT_PATTERN_FAMILIES)
    return requested & allowed if requested else allowed


def load_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Prices file not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in prices file: {missing}")
    df = df[~df["Ticker"].astype(str).map(_is_benchmark_ticker)].copy()
    df.sort_values(["Ticker", "Date"], inplace=True)
    return df


def load_pattern_weights(path: Path = PATTERN_WEIGHTS_JSON) -> dict[str, float]:
    defaults = {key: 0.0 for key in DEFAULT_PATTERN_FAMILIES}
    try:
        with open(path) as f:
            data = json.load(f)
        return {key: float(data.get(key, 0.0)) for key in defaults}
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return defaults


def _score_pattern_a_rows(
    a_df: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    breakout_days: int,
) -> pd.DataFrame:
    if a_df.empty:
        return a_df

    out = a_df.copy()
    for idx in out.index:
        ticker = str(out.at[idx, "ticker"])
        g = prices[prices["Ticker"] == ticker].copy().sort_values("Date")
        g = g[g["Date"] <= as_of_date].copy()
        if g.empty:
            continue
        g["SMA50"] = g["Close"].rolling(50).mean()
        g["SMA200"] = g["Close"].rolling(200).mean()
        g["VolAvg20"] = g["Volume"].rolling(20).mean()
        g["PrevNHighClose"] = g["Close"].shift(1).rolling(int(breakout_days)).max()
        r = g.iloc[-1]
        if any(pd.isna(r[col]) for col in ["SMA50", "SMA200", "VolAvg20", "PrevNHighClose"]):
            continue

        trend_strength_pct = ((float(r["SMA50"]) / float(r["SMA200"])) - 1.0) * 100.0
        setup_strength_pct = ((float(r["Close"]) / float(r["PrevNHighClose"])) - 1.0) * 100.0
        volume_ratio = float(r["Volume"]) / float(r["VolAvg20"])
        stop_pct_eff = float(out.at[idx, "stop_pct"])
        rsi_value = None
        try:
            hist_close = g["Close"].astype(float)
            rsi_value = compute_rsi(hist_close, period=14)
        except Exception:
            rsi_value = None

        score_trend, score_setup, score_volume, score_risk, score_rsi, signal_score = build_score_components(
            trend_strength_pct=trend_strength_pct,
            setup_strength_pct=setup_strength_pct,
            volume_ratio=volume_ratio,
            stop_pct_eff=stop_pct_eff,
            rsi_value=rsi_value,
        )
        sma50_slope_pct = compute_ma_slope_pct(g["SMA50"])
        ma_slope_bonus, signal_score = apply_ma_slope_bonus(signal_score, sma50_slope_pct)
        out.at[idx, "score_trend"] = score_trend
        out.at[idx, "score_setup"] = score_setup
        out.at[idx, "score_volume"] = score_volume
        out.at[idx, "score_risk"] = score_risk
        out.at[idx, "score_rsi"] = score_rsi
        out.at[idx, "sma50_slope_pct"] = round(float(sma50_slope_pct), 2) if sma50_slope_pct is not None else pd.NA
        out.at[idx, "ma_slope_bonus"] = ma_slope_bonus
        out.at[idx, "signal_score"] = signal_score
    return out


def compute_scored_signals_for_date(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
    pullback_buffer_pct: float,
    rebound_min_pct: float,
    consensus_bonus: float,
    pattern_families: set[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    if "A" in pattern_families:
        a_df = pattern_a.detect(
            prices,
            as_of_date=as_of_date,
            breakout_days=int(breakout_days),
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
        )
        a_df = _score_pattern_a_rows(a_df, prices, as_of_date=as_of_date, breakout_days=int(breakout_days))
        if not a_df.empty:
            rows.append(a_df)

    if "B" in pattern_families:
        b_df = pattern_b.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            pullback_buffer_pct=float(pullback_buffer_pct),
            rebound_min_pct=float(rebound_min_pct),
            compute_rsi_fn=compute_rsi,
        )
        if not b_df.empty:
            rows.append(b_df)

    if "C" in pattern_families:
        c_df = pattern_c_macd.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            compute_rsi_fn=compute_rsi,
        )
        if not c_df.empty:
            rows.append(c_df)

    if "D" in pattern_families:
        d_df = pattern_d_rsi.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            compute_rsi_fn=compute_rsi,
        )
        if not d_df.empty:
            rows.append(d_df)

    if "E" in pattern_families:
        e_df = pattern_e_boll.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            compute_rsi_fn=compute_rsi,
        )
        if not e_df.empty:
            rows.append(e_df)

    if "F" in pattern_families:
        f_df = pattern_f_vwap.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            compute_rsi_fn=compute_rsi,
        )
        if not f_df.empty:
            rows.append(f_df)

    if "G" in pattern_families:
        g_df = pattern_g_vcp.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=min(float(volume_multiplier), 1.2),
            stop_pct=float(stop_pct),
            base_lookback=100,
            dryup_volume_ratio=1.0,
            compute_rsi_fn=compute_rsi,
        )
        if not g_df.empty:
            rows.append(g_df)

    if not rows:
        return pd.DataFrame(columns=STANDARD_SIGNAL_COLS)

    out = pd.concat(rows, ignore_index=True)
    out = ensure_penalty_columns(out)
    out = ensure_markov_columns(out)
    out = ensure_stop_risk_columns(out)
    out["consensus_count"] = out.groupby(["signal_date", "ticker"])["pattern_family"].transform("nunique")

    if float(consensus_bonus) > 0:
        bonus_mask = out["consensus_count"] > 1
        out.loc[bonus_mask, "signal_score"] = (
            out.loc[bonus_mask, "signal_score"].astype(float) + float(consensus_bonus)
        ).map(clip_score)

    out = apply_pattern_family_bonus(out, load_pattern_weights())
    out = ensure_penalty_columns(out)
    out = ensure_stop_risk_columns(out)

    out.drop_duplicates(subset=["signal_date", "ticker", "pattern"], keep="last", inplace=True)
    out.sort_values(["signal_date", "ticker", "pattern"], inplace=True)
    return out[STANDARD_SIGNAL_COLS].copy()


def compute_signals_for_all_dates(
    prices: pd.DataFrame,
    *,
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
    pullback_buffer_pct: float,
    rebound_min_pct: float,
    consensus_bonus: float,
    pattern_families: set[str],
) -> pd.DataFrame:
    all_dates = sorted(prices["Date"].drop_duplicates())
    chunks: list[pd.DataFrame] = []
    for signal_date in all_dates:
        day = compute_scored_signals_for_date(
            prices,
            as_of_date=signal_date,
            breakout_days=int(breakout_days),
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            pullback_buffer_pct=float(pullback_buffer_pct),
            rebound_min_pct=float(rebound_min_pct),
            consensus_bonus=float(consensus_bonus),
            pattern_families=pattern_families,
        )
        if not day.empty:
            chunks.append(day)

    if not chunks:
        return pd.DataFrame(columns=STANDARD_SIGNAL_COLS)

    out = pd.concat(chunks, ignore_index=True)
    out.drop_duplicates(subset=["signal_date", "ticker", "pattern"], keep="last", inplace=True)
    out.sort_values(["signal_date", "ticker", "pattern"], inplace=True)
    return out[STANDARD_SIGNAL_COLS].copy()


def merge_buy_signals(existing_path: Path, new_signals: pd.DataFrame) -> pd.DataFrame:
    existing = pd.DataFrame(columns=STANDARD_SIGNAL_COLS)
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        for column in STANDARD_SIGNAL_COLS:
            if column not in existing.columns:
                existing[column] = pd.NA
        existing = existing[STANDARD_SIGNAL_COLS]

    if existing.empty:
        merged = new_signals[STANDARD_SIGNAL_COLS].copy()
    elif new_signals.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new_signals[STANDARD_SIGNAL_COLS]], ignore_index=True)

    merged.drop_duplicates(subset=["signal_date", "ticker", "pattern"], keep="last", inplace=True)
    merged.sort_values(["signal_date", "ticker", "pattern"], inplace=True)
    return merged[STANDARD_SIGNAL_COLS].copy()


def _print_pattern_counts(signals_df: pd.DataFrame, *, label: str) -> None:
    if signals_df.empty or "pattern_family" not in signals_df.columns:
        print(f"{label}: none")
        return
    counts = signals_df["pattern_family"].astype(str).value_counts().sort_index()
    rendered = ", ".join(f"{family}={int(count)}" for family, count in counts.items())
    print(f"{label}: {rendered}")


def main() -> None:
    args = parse_args()
    prices = load_prices(Path(args.prices))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.rescore_only:
        new_signals = pd.DataFrame(columns=STANDARD_SIGNAL_COLS)
        all_signals = load_existing_signal_history(out_path, required_columns=STANDARD_SIGNAL_COLS)
    else:
        as_of_date = pd.to_datetime(args.as_of_date) if args.as_of_date else prices["Date"].max()
        pattern_families = _resolve_pattern_families(args.pattern_families)

        if args.backfill_history:
            new_signals = compute_signals_for_all_dates(
                prices,
                breakout_days=args.breakout_days,
                volume_multiplier=args.volume_multiplier,
                stop_pct=args.stop_pct,
                pullback_buffer_pct=args.pullback_buffer_pct,
                rebound_min_pct=args.rebound_min_pct,
                consensus_bonus=args.consensus_bonus,
                pattern_families=pattern_families,
            )
        else:
            new_signals = compute_scored_signals_for_date(
                prices,
                as_of_date=as_of_date,
                breakout_days=args.breakout_days,
                volume_multiplier=args.volume_multiplier,
                stop_pct=args.stop_pct,
                pullback_buffer_pct=args.pullback_buffer_pct,
                rebound_min_pct=args.rebound_min_pct,
                consensus_bonus=args.consensus_bonus,
                pattern_families=pattern_families,
            )

        all_signals = merge_buy_signals(out_path, new_signals)

    pattern_families = _resolve_pattern_families(args.pattern_families)
    st_score_payload = build_st_score_payload(
        mode=str(args.st_model_mode),
        logistic_path=Path(args.st_logistic_model),
        svm_path=Path(args.st_svm_model),
        rf_path=Path(args.st_rf_model),
        xgb_path=Path(args.st_xgb_model),
        blend_weight_svm=float(args.st_svm_weight),
        blend_weight_rf=float(args.st_rf_weight),
        blend_weight_xgb=float(args.st_xgb_weight),
    )

    all_signals = rescore_signal_history(
        all_signals,
        prices,
        breakout_days=int(args.breakout_days),
        pattern_weights=load_pattern_weights(),
        penalty_payload=load_signal_penalty_weights(),
        markov_payload=load_signal_markov_model(Path(args.markov_model)),
        markov_mode=str(args.markov_mode),
        stop_risk_payload=load_signal_stop_risk_model(),
        st_score_payload=st_score_payload,
    )
    all_signals.to_csv(out_path, index=False)

    # Phase 2: Optionally attach catalyst features (market-regime + event windows).
    if args.catalyst_enrichment:
        print("Enriching signals with catalyst features (Phase 2)...")
        external_factors = load_external_factors(Path(args.external_factors))
        event_calendar = load_event_calendar(Path(args.event_calendar))
        all_signals = enrich_signals_with_catalysts(
            all_signals,
            external_factors=external_factors,
            event_calendar=event_calendar,
            include_market_regimes=True,
            include_event_windows=True,
        )
        all_signals.to_csv(out_path, index=False)
        print("  ✓ Catalyst enrichment complete")

    if args.rescore_only:
        print("Mode: rescore existing signal history")
        print(f"Signals rescored: {len(all_signals)}")
    elif args.backfill_history:
        print("As-of date: backfill all available dates")
        print(f"New all-pattern signals generated in backfill: {len(new_signals)}")
    else:
        print(f"As-of date: {as_of_date.date().isoformat()}")
        print(f"New all-pattern signals generated today: {len(new_signals)}")
    if not args.rescore_only:
        print(f"Pattern families included: {', '.join(sorted(pattern_families))}")
        _print_pattern_counts(new_signals, label="New signal counts by family")
    print(f"Total all-pattern buy signals tracked: {len(all_signals)}")
    resolved_st_mode = str(st_score_payload.get("model_type", "logistic") or "logistic")
    if resolved_st_mode == "hybrid":
        print(f"ST model mode: hybrid (svm_weight={float(st_score_payload.get('blend_weight_svm', 0.3)):.2f})")
    elif resolved_st_mode == "hybrid3":
        print(
            "ST model mode: hybrid3 "
            f"(logistic_weight={max(0.0, 1.0 - float(st_score_payload.get('blend_weight_svm', 0.3)) - float(st_score_payload.get('blend_weight_rf', 0.2))):.2f}, "
            f"svm_weight={float(st_score_payload.get('blend_weight_svm', 0.3)):.2f}, "
            f"rf_weight={float(st_score_payload.get('blend_weight_rf', 0.2)):.2f})"
        )
    elif resolved_st_mode == "hybrid4":
        _w_svm = float(st_score_payload.get("blend_weight_svm", 0.25))
        _w_rf = float(st_score_payload.get("blend_weight_rf", 0.25))
        _w_xgb = float(st_score_payload.get("blend_weight_xgb", 0.25))
        _w_log = max(0.0, 1.0 - _w_svm - _w_rf - _w_xgb)
        print(
            "ST model mode: hybrid4 "
            f"(logistic_weight={_w_log:.2f}, svm_weight={_w_svm:.2f}, rf_weight={_w_rf:.2f}, xgboost_weight={_w_xgb:.2f})"
        )
    else:
        print(f"ST model mode: {resolved_st_mode}")
    _print_pattern_counts(all_signals, label="Tracked signal counts by family")
    print(f"Buy signals saved to: {out_path}")


if __name__ == "__main__":
    main()