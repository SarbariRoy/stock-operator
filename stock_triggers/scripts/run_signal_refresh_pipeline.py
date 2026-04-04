"""Run the shared signal refresh pipeline used by GitHub workflows.

This centralizes the workflow orchestration so the YAML files only need to pass
mode-specific options instead of duplicating long shell pipelines.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "stock_triggers" / "scripts"
DATA_DIR = ROOT / "stock_triggers" / "data"
UNIVERSE_FILE = DATA_DIR / "universe_tickers.txt"
TRAINING_DATA = DATA_DIR / "training_signals_history.csv"
SIGNALS_ALL = DATA_DIR / "signals_all_patterns.csv"
SIGNALS_PATTERN_A = DATA_DIR / "signals_pattern_a.csv"


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
    parser.add_argument("--backfill-history", type=_parse_bool, default=False)
    parser.add_argument("--include-stock-scores", type=_parse_bool, default=False)
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


def _rescore_outputs(python_executable: str, *, include_pattern_a: bool) -> None:
    if include_pattern_a:
        run_step(
            "Re-score Pattern A signals",
            _script_command(python_executable, "generate_triggers_pattern_a.py", "--rescore-only"),
        )
    run_step(
        "Re-score all-pattern signals",
        _script_command(python_executable, "generate_signals_all_patterns.py", "--rescore-only"),
    )


def _build_shared_training_history(python_executable: str) -> None:
    run_step(
        "Build shared training artifact",
        _script_command(python_executable, "build_training_signals_history.py"),
    )


def run_pipeline(args: argparse.Namespace) -> None:
    python_executable = sys.executable

    _maybe_refresh_prices(args, python_executable)

    if args.mode == "incremental":
        extra_args = _incremental_extra_args(args)
        run_step(
            "Generate Pattern A triggers",
            _script_command(python_executable, "generate_triggers_pattern_a.py", *extra_args),
        )
        run_step(
            "Generate all-pattern signals",
            _script_command(python_executable, "generate_signals_all_patterns.py", *extra_args),
        )
    else:
        all_pattern_args = _daily_all_pattern_args(args)
        run_step(
            "Generate all-pattern signals",
            _script_command(python_executable, "generate_signals_all_patterns.py", *all_pattern_args),
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

    if args.mode == "daily":
        pattern_a_args = _daily_pattern_a_args(args)
        run_step(
            "Generate Pattern A triggers",
            _script_command(python_executable, "generate_triggers_pattern_a.py", *pattern_a_args),
        )
        _rescore_outputs(python_executable, include_pattern_a=False)
    else:
        _rescore_outputs(python_executable, include_pattern_a=True)

    run_step(
        "Compute candle-shape enhancer weights",
        _script_command(
            python_executable,
            "compute_candle_weights.py",
            "--prices",
            str(DATA_DIR / "prices_eod.csv"),
            "--signals",
            str(SIGNALS_ALL),
            "--out",
            str(DATA_DIR / "candle_weights.json"),
        ),
    )

    if args.include_stock_scores:
        run_step(
            "Generate stock health scores",
            _script_command(python_executable, "generate_stock_scores.py"),
        )


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()