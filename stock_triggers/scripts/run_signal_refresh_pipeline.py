"""Run the shared signal refresh pipeline used by GitHub workflows.

This centralizes the workflow orchestration so the YAML files only need to pass
mode-specific options instead of duplicating long shell pipelines.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from stock_triggers.scoring_defaults import (
    build_scoring_defaults_snapshot,
    compute_scoring_defaults_hash,
    diff_scoring_snapshots,
)


SCRIPTS_DIR = ROOT / "stock_triggers" / "scripts"
LT_SCRIPTS_DIR = SCRIPTS_DIR / "long_term"
ST_SCRIPTS_DIR = SCRIPTS_DIR / "short_term"
DATA_DIR = ROOT / "stock_triggers" / "data"
UNIVERSE_FILE = DATA_DIR / "universe_tickers.txt"
TRAINING_DATA = DATA_DIR / "st_lt_training_signals_history.csv"
SIGNALS_ALL = DATA_DIR / "st_signals_all_patterns.csv"
SIGNALS_PATTERN_A = DATA_DIR / "lt_signals_pattern_a.csv"
UNIVERSE_SIGNAL_SCORES_CSV = DATA_DIR / "universe_signal_scores.csv"
SCORING_DEFAULTS_SNAPSHOT_JSON = DATA_DIR / "scoring_defaults_snapshot.json"
PIPELINE_ALERTS_JSON = DATA_DIR / "pipeline_alerts.json"
BENCHMARK_TICKERS = {"^NSEI"}


def _parse_bool(raw_value: str | bool) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off", ""}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw_value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared signal refresh pipeline")
    parser.add_argument("--mode", choices=("incremental", "daily"), required=True)
    parser.add_argument("--refresh-prices", type=_parse_bool, default=False)
    parser.add_argument("--recompute-pattern-weights", type=_parse_bool, default=False)
    parser.add_argument("--recompute-candle-weights", type=_parse_bool, default=True)
    parser.add_argument("--backfill-history", type=_parse_bool, default=False)
    parser.add_argument("--as-of-date", type=str, default="")
    parser.add_argument("--user-agent", type=str, default="Brilliant")
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--pause-seconds", type=float, default=0.8)
    return parser.parse_args()


def run_step(label: str, command: list[str]) -> None:
    print(f"\n==> {label}")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)


def _script_command(python_executable: str, script_name: str, *args: str) -> list[str]:
    return [python_executable, str(SCRIPTS_DIR / script_name), *args]


def _lt_script_command(python_executable: str, script_name: str, *args: str) -> list[str]:
    return [python_executable, str(LT_SCRIPTS_DIR / script_name), *args]


def _st_script_command(python_executable: str, script_name: str, *args: str) -> list[str]:
    return [python_executable, str(ST_SCRIPTS_DIR / script_name), *args]


def _maybe_refresh_prices(args: argparse.Namespace, python_executable: str) -> None:
    if not args.refresh_prices:
        return
    run_step(
        "Refresh prices",
        _script_command(
            python_executable,
            "update_prices_yf.py",
            "--user-agent",
            args.user_agent,
            "--days",
            str(args.days),
            "--pause-seconds",
            str(args.pause_seconds),
            "--overwrite",
            "--universe-file",
            str(UNIVERSE_FILE),
        ),
    )


def _incremental_extra_args(args: argparse.Namespace) -> list[str]:
    extra_args: list[str] = []
    if args.backfill_history:
        extra_args.append("--backfill-history")
    if args.as_of_date:
        extra_args.extend(["--as-of-date", args.as_of_date])
    return extra_args


def _daily_all_pattern_args(args: argparse.Namespace) -> list[str]:
    return ["--backfill-history"] if args.backfill_history or not SIGNALS_ALL.exists() else []


def _daily_pattern_a_args(args: argparse.Namespace) -> list[str]:
    return ["--backfill-history"] if args.backfill_history or not SIGNALS_PATTERN_A.exists() else []


def _is_benchmark_ticker(ticker: str) -> bool:
    return str(ticker).strip().upper() in BENCHMARK_TICKERS


def _load_json_dict(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _load_universe_tickers(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    tickers = [
        line.strip().upper()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        if _is_benchmark_ticker(ticker):
            continue
        if ticker not in seen:
            seen.add(ticker)
            deduped.append(ticker)
    return deduped


def _latest_scores_by_ticker(path: Path, score_column: str, out_score_col: str, out_date_col: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", out_score_col, out_date_col])
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["ticker", out_score_col, out_date_col])

    required = {"ticker", "signal_date", score_column}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["ticker", out_score_col, out_date_col])

    view = df[["ticker", "signal_date", score_column]].copy()
    view["ticker"] = view["ticker"].astype(str).str.strip().str.upper()
    view["signal_date"] = pd.to_datetime(view["signal_date"], errors="coerce")
    view[score_column] = pd.to_numeric(view[score_column], errors="coerce")
    view = view.dropna(subset=["ticker", "signal_date"]).copy()
    if view.empty:
        return pd.DataFrame(columns=["ticker", out_score_col, out_date_col])

    view.sort_values(["signal_date", score_column, "ticker"], ascending=[False, False, True], inplace=True)
    view = view.drop_duplicates(subset=["ticker"], keep="first")
    view.rename(columns={score_column: out_score_col, "signal_date": out_date_col}, inplace=True)
    return view[["ticker", out_score_col, out_date_col]].copy()


def _build_universe_signal_scores_artifact() -> int:
    universe = _load_universe_tickers(UNIVERSE_FILE)
    if not universe:
        pd.DataFrame(
            columns=[
                "ticker",
                "lt_score",
                "st_score",
                "lt_signal_date",
                "st_signal_date",
                "has_lt_signal",
                "has_st_signal",
                "as_of_date",
            ]
        ).to_csv(UNIVERSE_SIGNAL_SCORES_CSV, index=False)
        return 0

    base = pd.DataFrame({"ticker": universe})
    lt = _latest_scores_by_ticker(SIGNALS_PATTERN_A, "signal_score", "lt_score", "lt_signal_date")
    st = _latest_scores_by_ticker(SIGNALS_ALL, "st_score", "st_score", "st_signal_date")

    out = base.merge(lt, on="ticker", how="left").merge(st, on="ticker", how="left")
    out["has_lt_signal"] = out["lt_score"].notna()
    out["has_st_signal"] = out["st_score"].notna()
    out["lt_score"] = pd.to_numeric(out["lt_score"], errors="coerce").fillna(0.0).round(1)
    out["st_score"] = pd.to_numeric(out["st_score"], errors="coerce").fillna(0.0).round(1)

    lt_dates = pd.to_datetime(out["lt_signal_date"], errors="coerce")
    st_dates = pd.to_datetime(out["st_signal_date"], errors="coerce")
    latest_date = pd.concat([lt_dates, st_dates], axis=0).dropna().max()
    as_of_date = latest_date.date().isoformat() if pd.notna(latest_date) else ""
    out["as_of_date"] = as_of_date

    out.sort_values(["ticker"], inplace=True)
    out.to_csv(UNIVERSE_SIGNAL_SCORES_CSV, index=False)
    return int(len(out))


def _rescore_outputs(python_executable: str, *, include_pattern_a: bool) -> None:
    if include_pattern_a:
        run_step(
            "Re-score Pattern A signals",
                _lt_script_command(python_executable, "generate_lt_signals.py", "--rescore-only"),
        )
    run_step(
        "Re-score all-pattern signals",
            _st_script_command(python_executable, "generate_st_signals.py", "--rescore-only"),
    )


def _build_shared_training_history(python_executable: str) -> None:
    run_step(
        "Build shared training artifact",
        _script_command(python_executable, "build_training_signals_history.py"),
    )


def run_pipeline(args: argparse.Namespace) -> None:
    start_perf = time.perf_counter()
    python_executable = sys.executable

    previous_snapshot = _load_json_dict(SCORING_DEFAULTS_SNAPSHOT_JSON)
    previous_policy = previous_snapshot.get("policy") if isinstance(previous_snapshot.get("policy"), dict) else {}
    previous_hash = str(previous_snapshot.get("hash", "") or "")
    has_previous_snapshot = bool(previous_policy) and bool(previous_hash)

    current_policy = build_scoring_defaults_snapshot()
    current_hash = compute_scoring_defaults_hash(current_policy)
    policy_changes = diff_scoring_snapshots(previous_policy, current_policy) if has_previous_snapshot else []
    policy_changed = bool(policy_changes)

    _maybe_refresh_prices(args, python_executable)

    if args.mode == "incremental":
        extra_args = _incremental_extra_args(args)
        run_step(
            "Generate Pattern A triggers",
                _lt_script_command(python_executable, "generate_lt_signals.py", *extra_args),
        )
        run_step(
            "Generate all-pattern signals",
                _st_script_command(python_executable, "generate_st_signals.py", *extra_args),
        )
    else:
        all_pattern_args = _daily_all_pattern_args(args)
        run_step(
            "Generate all-pattern signals",
                _st_script_command(python_executable, "generate_st_signals.py", *all_pattern_args),
        )

    if args.recompute_pattern_weights:
        _build_shared_training_history(python_executable)
        run_step(
            "Compute pattern-family weights",
            _script_command(
                python_executable,
                "compute_pattern_weights.py",
                "--training-data",
                str(TRAINING_DATA),
            ),
        )
        _rescore_outputs(python_executable, include_pattern_a=args.mode == "incremental")

    _build_shared_training_history(python_executable)
    run_step(
        "Compute row-level signal penalties",
        _script_command(
            python_executable,
            "compute_signal_penalty_weights.py",
            "--training-data",
            str(TRAINING_DATA),
        ),
    )
    _rescore_outputs(python_executable, include_pattern_a=args.mode == "incremental")

    _build_shared_training_history(python_executable)
    run_step(
        "Compute monotonic stop-risk model",
        _script_command(
            python_executable,
            "compute_signal_stop_risk_model.py",
            "--feature-set",
            "scores_only",
            "--training-data",
            str(TRAINING_DATA),
        ),
    )

    # When backfilling with huge lookback (e.g. --days 5000), limit training data
    # to recent history (~2 years) to avoid O(months²) walk-forward evaluation cost
    train_start_arg = []
    if args.backfill_history:
        latest_date = pd.read_csv(SIGNALS_ALL, parse_dates=["signal_date"], usecols=["signal_date"])
        if not latest_date.empty:
            max_date = latest_date["signal_date"].max()
            train_start = max_date - timedelta(days=730)  # 2 years
            train_start_arg = ["--train-start-date", train_start.strftime("%Y-%m-%d")]

    run_step(
        "Evaluate stop-risk walk-forward (scores_only candidate)",
        _script_command(
            python_executable,
            "evaluate_stop_risk_walk_forward.py",
            "--evaluation-mode",
            "walk-forward",
            "--candidate",
            "scores_only",
            "--predictions-out",
                str(DATA_DIR / "lt_stop_risk_walk_forward_oos_complete.csv"),
            *train_start_arg,
        ),
    )

    if args.mode == "daily":
        pattern_a_args = _daily_pattern_a_args(args)
        run_step(
            "Generate Pattern A triggers",
              _lt_script_command(python_executable, "generate_lt_signals.py", *pattern_a_args),
        )
        _rescore_outputs(python_executable, include_pattern_a=False)
    else:
        _rescore_outputs(python_executable, include_pattern_a=True)

    if args.recompute_candle_weights:
        run_step(
            "Compute candle-shape enhancer weights",
            _script_command(
                python_executable,
                "compute_candle_weights.py",
                "--prices",
                str(DATA_DIR / "st_lt_prices_eod.csv"),
                "--signals",
                str(SIGNALS_ALL),
                "--out",
                str(DATA_DIR / "st_lt_candle_weights.json"),
            ),
        )
    else:
        print("\n==> Skip candle-shape enhancer weights (--recompute-candle-weights false)")

    run_step(
        "Build default Coverage cache",
        _script_command(
            python_executable,
            "build_coverage_cache.py",
            "--prices",
            str(DATA_DIR / "st_lt_prices_eod.csv"),
            "--signals",
            str(SIGNALS_ALL),
            "--out",
            str(DATA_DIR / "coverage_default_cache.pkl"),
        ),
    )

    print("\n==> Build full-universe LT/ST score artifact")
    universe_count = _build_universe_signal_scores_artifact()
    print(f"Universe LT/ST scores written to {UNIVERSE_SIGNAL_SCORES_CSV} ({universe_count} tickers)")

    run_step(
        "Build default LT/ST view artifacts",
        _script_command(
            python_executable,
            "build_default_view_artifacts.py",
        ),
    )

    elapsed_seconds = int(time.perf_counter() - start_perf)
    runtime_threshold_seconds = 3600
    runtime_breach = elapsed_seconds > runtime_threshold_seconds

    snapshot_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hash": current_hash,
        "policy": current_policy,
    }
    _write_json(SCORING_DEFAULTS_SNAPSHOT_JSON, snapshot_payload)

    alerts_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed_seconds,
        "runtime_threshold_seconds": runtime_threshold_seconds,
        "runtime_breach": runtime_breach,
        "default_policy_changed": policy_changed,
        "default_policy_hash_before": previous_hash,
        "default_policy_hash_after": current_hash,
        "default_policy_changes": policy_changes,
        "universe_score_rows": universe_count,
    }
    _write_json(PIPELINE_ALERTS_JSON, alerts_payload)

    if policy_changed:
        print(f"WARNING: default scoring policy changed ({len(policy_changes)} fields). Details in {PIPELINE_ALERTS_JSON}")
    else:
        print("Default scoring policy unchanged.")

    if runtime_breach:
        print(
            f"WARNING: refresh+scoring runtime breach: {elapsed_seconds}s > {runtime_threshold_seconds}s. "
            f"Pipeline continues by policy. Details in {PIPELINE_ALERTS_JSON}"
        )


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()