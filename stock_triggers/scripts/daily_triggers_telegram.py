"""Run daily trigger pipeline and send results to Telegram.

Workflow:
1) Refresh prices (optional)
2) Generate Pattern A triggers
3) Send latest signal summary to Telegram

Credentials can be provided by any of these (highest priority first):
1) CLI args (--token, --chat-id)
2) Environment variables
3) secrets.yml in repo root

Environment variable names:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "stock_triggers" / "scripts"
DATA_DIR = ROOT / "stock_triggers" / "data"
SIGNALS_CSV = DATA_DIR / "signals_pattern_a.csv"
ALL_PATTERNS_SIGNALS_CSV = DATA_DIR / "signals_all_patterns.csv"
PATTERN_WEIGHTS_JSON = DATA_DIR / "pattern_weights.json"
SELL_SIGNALS_CSV = DATA_DIR / "sell_signals_pattern_a.csv"
SECRETS_FILE = ROOT / "secrets.yml"
PRODUCTION_APP_URL = "https://stock-operator-roy.streamlit.app/"


def _fmt_status(label: str, note: str = "") -> str:
    if label == "done":
        return "done"
    if label in {"skipped", "skipped_send_only"}:
        return "skipped"
    if label == "not_done":
        return f"failed ({note})" if note else "failed"
    return label


def _fmt_price(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "na"


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "na"


def is_remote_runtime() -> bool:
    """Allow Telegram sending only from hosted runtimes, never from local hosts."""
    return bool(os.getenv("GITHUB_ACTIONS")) or bool(os.getenv("STREAMLIT_CLOUD")) or bool(os.getenv("STREAMLIT_SHARING_MODE"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily triggers and notify Telegram")
    parser.add_argument("--skip-refresh", action="store_true", help="Skip price refresh step")
    parser.add_argument(
        "--send-only",
        action="store_true",
        help="Send Telegram message from existing signals file only (no refresh, no trigger generation).",
    )
    parser.add_argument("--breakout-days", type=int, default=40)
    parser.add_argument("--volume-multiplier", type=float, default=1.5)
    parser.add_argument("--stop-pct", type=float, default=7.0)
    parser.add_argument("--as-of-date", type=str, default=None, help="Optional YYYY-MM-DD")
    parser.add_argument("--token", type=str, default=None, help="Telegram bot token")
    parser.add_argument("--chat-id", type=str, default=None, help="Telegram chat id")
    parser.add_argument(
        "--secrets-file",
        type=str,
        default=str(SECRETS_FILE),
        help="Path to secrets.yml (default: repo-root secrets.yml)",
    )
    return parser.parse_args()


def load_secrets_file(path: Path) -> dict[str, str]:
    """Load a minimal key:value secrets.yml file.

    Supports lines like:
    TELEGRAM_BOT_TOKEN: "..."
    TELEGRAM_CHAT_ID: "..."
    """

    if not path.is_file():
        return {}

    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


def run_command(cmd: list[str]) -> tuple[bool, str]:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # pragma: no cover
        return False, str(exc)

    if res.returncode != 0:
        err = res.stderr.strip() or res.stdout.strip() or f"exit code {res.returncode}"
        return False, err
    return True, res.stdout.strip()


def refresh_prices() -> tuple[bool, str]:
    updater = SCRIPTS_DIR / "update_prices_yf.py"
    cmd = [
        sys.executable,
        str(updater),
        "--user-agent",
        "Brilliant",
        "--days",
        "1200",
        "--pause-seconds",
        "0.8",
        "--overwrite",
        "--universe-file",
        str(DATA_DIR / "universe_tickers.txt"),
    ]
    return run_command(cmd)


def generate_triggers(
    *,
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
    as_of_date: str | None,
) -> tuple[bool, str]:
    generator = SCRIPTS_DIR / "generate_triggers_pattern_a.py"
    cmd = [
        sys.executable,
        str(generator),
        "--breakout-days",
        str(breakout_days),
        "--volume-multiplier",
        str(volume_multiplier),
        "--stop-pct",
        str(stop_pct),
    ]
    if as_of_date:
        cmd.extend(["--as-of-date", as_of_date])
    return run_command(cmd)


def generate_all_pattern_signals(
    *,
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
    as_of_date: str | None,
) -> tuple[bool, str]:
    generator = SCRIPTS_DIR / "generate_signals_all_patterns.py"
    cmd = [
        sys.executable,
        str(generator),
        "--breakout-days",
        str(breakout_days),
        "--volume-multiplier",
        str(volume_multiplier),
        "--stop-pct",
        str(stop_pct),
    ]
    if as_of_date:
        cmd.extend(["--as-of-date", as_of_date])
    return run_command(cmd)


def compute_pattern_weights() -> tuple[bool, str]:
    generator = SCRIPTS_DIR / "compute_pattern_weights.py"
    cmd = [
        sys.executable,
        str(generator),
    ]
    return run_command(cmd)


def build_message(
    *,
    refresh_status: str,
    refresh_note: str,
    trigger_status: str,
    trigger_note: str,
    all_patterns_status: str,
    all_patterns_note: str,
    pattern_weights_status: str,
    pattern_weights_note: str,
) -> str:
    has_buy = SIGNALS_CSV.is_file()

    refresh_text = _fmt_status(refresh_status, refresh_note)
    trigger_text = _fmt_status(trigger_status, trigger_note)
    all_patterns_text = _fmt_status(all_patterns_status, all_patterns_note)
    pattern_weights_text = _fmt_status(pattern_weights_status, pattern_weights_note)
    today_text = date.today().isoformat()
    telegram_threshold = 60.0

    if not has_buy:
        return (
            f"Daily Stock Trigger Update | {today_text}\n\n"
            "No signal generated today.\n"
            f"Production: {PRODUCTION_APP_URL}\n\n"
            f"Pipeline: prices {refresh_text}, triggers {trigger_text}"
        )

    src_csv = ALL_PATTERNS_SIGNALS_CSV if ALL_PATTERNS_SIGNALS_CSV.is_file() else SIGNALS_CSV
    df = pd.read_csv(src_csv)
    if not df.empty and "signal_date" in df.columns:
        df["signal_date"] = df["signal_date"].astype(str)
        today_df = df[df["signal_date"] == today_text].copy()
    else:
        today_df = pd.DataFrame()

    sell_df = pd.DataFrame()
    if SELL_SIGNALS_CSV.is_file():
        try:
            sell_df = pd.read_csv(SELL_SIGNALS_CSV)
            if "sell_signal_date" in sell_df.columns:
                sell_df["sell_signal_date"] = sell_df["sell_signal_date"].astype(str)
        except Exception:
            sell_df = pd.DataFrame()
    today_exits = sell_df[sell_df["sell_signal_date"] == today_text].copy() if not sell_df.empty and "sell_signal_date" in sell_df.columns else pd.DataFrame()

    def _section(frame: pd.DataFrame, score_col: str, label: str) -> list[str]:
        if score_col not in frame.columns or frame.empty:
            return []
        col = pd.to_numeric(frame[score_col], errors="coerce")
        filtered = frame[col >= telegram_threshold].copy()
        if filtered.empty:
            return [f"{label}: none above {int(telegram_threshold)}", ""]
        filtered[score_col] = pd.to_numeric(filtered[score_col], errors="coerce")
        filtered.sort_values([score_col, "ticker"], ascending=[False, True], inplace=True)
        out = [f"{label} ({len(filtered)} signal{'s' if len(filtered) != 1 else ''})", ""]
        if score_col == "st_score":
            out.append("Exit strategy: Structure confluence stop (0.5% below lowest of swing low, EMA20, and VWAP reclaim; fallback to Stop %).")
            out.append("")
        for _, r in filtered.iterrows():
            score = int(round(float(r[score_col])))
            col_label = "ST" if score_col == "st_score" else "Score"
            pattern_text = str(r.get("pattern", "na"))
            stop_price = pd.to_numeric(r.get("stop_price"), errors="coerce")
            exit_text = f" | Exit {_fmt_price(stop_price)}" if pd.notna(stop_price) else ""
            out.append(
                f"- {r['ticker']} | {col_label} {score} | Entry {_fmt_price(r['entry_price'])}{exit_text} | {pattern_text}"
            )
        out.append("")
        return out

    st_lines = _section(today_df, "st_score", "Short term")
    lt_lines = _section(today_df, "signal_score", "Long term")

    lines = [f"Daily Stock Trigger Update | {today_text}", ""]

    exit_lines: list[str] = []
    if not today_exits.empty:
        today_exits_sorted = today_exits.sort_values("ticker") if "ticker" in today_exits.columns else today_exits
        n = len(today_exits_sorted)
        exit_lines.append(f"Exits today ({n} position{'s' if n != 1 else ''})")
        exit_lines.append("")
        for _, r in today_exits_sorted.iterrows():
            ret = float(pd.to_numeric(r.get("realized_return_pct"), errors="coerce") or 0.0)
            ret_sign = "+" if ret >= 0 else ""
            entry = float(pd.to_numeric(r.get("entry_price"), errors="coerce") or 0.0)
            exit_price = float(pd.to_numeric(r.get("sell_price"), errors="coerce") or 0.0)
            exit_lines.append(f"- SELL {r['ticker']} | {entry:.2f} \u2192 {exit_price:.2f} | {ret_sign}{ret:.1f}%")
        exit_lines.append("")

    if st_lines or lt_lines or exit_lines:
        lines.extend(st_lines)
        lines.extend(lt_lines)
        lines.extend(exit_lines)
    else:
        lines.append(f"No signal at or above Telegram threshold {int(telegram_threshold)} today.")
        lines.append("")

    lines.extend([f"Production: {PRODUCTION_APP_URL}"])

    if refresh_note and refresh_status == "not_done":
        lines.extend(["", f"Price refresh error: {refresh_note}"])
    if trigger_note and trigger_status == "not_done":
        lines.extend(["", f"Trigger generation error: {trigger_note}"])
    if all_patterns_note and all_patterns_status == "not_done":
        lines.extend(["", f"All-pattern generation error: {all_patterns_note}"])
    if pattern_weights_note and pattern_weights_status == "not_done":
        lines.extend(["", f"Pattern weight calibration error: {pattern_weights_note}"])

    lines.extend(["", f"Pipeline: prices {refresh_text}, triggers {trigger_text}, all-patterns {all_patterns_text}, pattern-weights {pattern_weights_text}"])

    return "\n".join(lines)


def send_telegram(token: str, chat_id: str, text: str) -> tuple[bool, str]:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
    except requests.RequestException as exc:
        return False, str(exc)

    if resp.status_code != 200:
        return False, f"Telegram API error {resp.status_code}: {resp.text[:500]}"

    return True, "ok"


def main() -> None:
    args = parse_args()
    allow_telegram_send = is_remote_runtime()

    refresh_status = "not_run"
    refresh_note = ""
    trigger_status = "not_run"
    trigger_note = ""
    all_patterns_status = "not_run"
    all_patterns_note = ""
    pattern_weights_status = "not_run"
    pattern_weights_note = ""

    if not args.send_only:
        if not args.skip_refresh:
            ok, out = refresh_prices()
            if ok:
                refresh_status = "done"
            else:
                refresh_status = "not_done"
                refresh_note = out
        else:
            refresh_status = "skipped"

        if refresh_status != "not_done":
            ok, out = generate_triggers(
                breakout_days=args.breakout_days,
                volume_multiplier=args.volume_multiplier,
                stop_pct=args.stop_pct,
                as_of_date=args.as_of_date,
            )
            if ok:
                trigger_status = "done"
            else:
                trigger_status = "not_done"
                trigger_note = out
        else:
            trigger_status = "skipped"
            trigger_note = "Skipped because refresh failed."

        if trigger_status == "done":
            ok, out = generate_all_pattern_signals(
                breakout_days=args.breakout_days,
                volume_multiplier=args.volume_multiplier,
                stop_pct=args.stop_pct,
                as_of_date=args.as_of_date,
            )
            if ok:
                all_patterns_status = "done"
            else:
                all_patterns_status = "not_done"
                all_patterns_note = out
        elif trigger_status == "not_done":
            all_patterns_status = "skipped"
            all_patterns_note = "Skipped because Pattern A trigger generation failed."
        else:
            all_patterns_status = "skipped"
            all_patterns_note = "Skipped because refresh failed."

        if all_patterns_status == "done":
            ok, out = compute_pattern_weights()
            if ok:
                pattern_weights_status = "done"
            else:
                pattern_weights_status = "not_done"
                pattern_weights_note = out
        elif all_patterns_status == "not_done":
            pattern_weights_status = "skipped"
            pattern_weights_note = "Skipped because all-pattern signal generation failed."
        elif trigger_status == "not_done":
            pattern_weights_status = "skipped"
            pattern_weights_note = "Skipped because Pattern A trigger generation failed."
        else:
            pattern_weights_status = "skipped"
            pattern_weights_note = "Skipped because upstream pipeline steps did not complete."
    else:
        refresh_status = "skipped_send_only"
        trigger_status = "skipped_send_only"
        all_patterns_status = "skipped_send_only"
        pattern_weights_status = "skipped_send_only"

    message = build_message(
        refresh_status=refresh_status,
        refresh_note=refresh_note,
        trigger_status=trigger_status,
        trigger_note=trigger_note,
        all_patterns_status=all_patterns_status,
        all_patterns_note=all_patterns_note,
        pattern_weights_status=pattern_weights_status,
        pattern_weights_note=pattern_weights_note,
    )

    if allow_telegram_send:
        secrets = load_secrets_file(Path(args.secrets_file))
        token = (
            args.token
            or os.getenv("TELEGRAM_BOT_TOKEN", "")
            or secrets.get("TELEGRAM_BOT_TOKEN", "")
        )
        chat_id = (
            args.chat_id
            or os.getenv("TELEGRAM_CHAT_ID", "")
            or secrets.get("TELEGRAM_CHAT_ID", "")
        )
        if not token or not chat_id:
            raise SystemExit(
                "Missing Telegram credentials. Set via CLI args, environment vars, or secrets.yml."
            )

        ok, out = send_telegram(token, chat_id, message)
        if not ok:
            raise SystemExit(f"Telegram send failed: {out}")
        print("Daily trigger notification sent.")
    else:
        print("Telegram send skipped by policy: local runtime is blocked.")

    if refresh_status == "not_done" or trigger_status == "not_done" or all_patterns_status == "not_done" or pattern_weights_status == "not_done":
        if allow_telegram_send:
            raise SystemExit("Pipeline had step failures, but Telegram status notification was sent.")
        raise SystemExit("Pipeline had step failures. Telegram notification was skipped by local policy.")


if __name__ == "__main__":
    main()
