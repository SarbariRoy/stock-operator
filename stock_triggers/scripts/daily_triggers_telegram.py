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
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "stock_triggers" / "scripts"
DATA_DIR = ROOT / "stock_triggers" / "data"
SIGNALS_CSV = DATA_DIR / "signals_pattern_a.csv"
SELL_SIGNALS_CSV = DATA_DIR / "sell_signals_pattern_a.csv"
SECRETS_FILE = ROOT / "secrets.yml"


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
        "365",
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


def build_message(
    *,
    refresh_status: str,
    refresh_note: str,
    trigger_status: str,
    trigger_note: str,
) -> str:
    has_buy = SIGNALS_CSV.is_file()
    has_sell = SELL_SIGNALS_CSV.is_file()

    if not has_buy and not has_sell:
        return (
            "Stock Trigger Update\n\n"
            f"Refresh: {refresh_status}\n"
            f"Trigger generation: {trigger_status}\n\n"
            "No signal files found."
        )

    df = pd.read_csv(SIGNALS_CSV) if has_buy else pd.DataFrame()
    sell_df = pd.read_csv(SELL_SIGNALS_CSV) if has_sell else pd.DataFrame()
    lines = [
        "Stock Trigger Update",
        "",
        f"Refresh: {refresh_status}",
        f"Trigger generation: {trigger_status}",
    ]

    if refresh_note:
        lines.append(f"Refresh note: {refresh_note}")
    if trigger_note:
        lines.append(f"Trigger note: {trigger_note}")

    lines.append("")

    if df.empty:
        lines.append("Data update done/not done is shown above.")
        lines.append("No buy trigger today.")
    else:
        latest_date = df["signal_date"].max()
        latest = df[df["signal_date"] == latest_date].copy()
        latest.sort_values(["ticker"], inplace=True)

        lines.extend(
            [
                f"Date: {latest_date}",
                f"Buy signals: {len(latest)}",
                "",
            ]
        )

        for _, r in latest.iterrows():
            lines.append(
                f"- BUY {r['ticker']} | {r['pattern']} | Entry {r['entry_price']} | Stop {r['stop_price']}"
            )

    lines.append("")
    if sell_df.empty:
        lines.append("No sell trigger today.")
        return "\n".join(lines)

    latest_sell_date = sell_df["sell_signal_date"].max()
    latest_sell = sell_df[sell_df["sell_signal_date"] == latest_sell_date].copy()
    latest_sell.sort_values(["ticker"], inplace=True)

    lines.extend(
        [
            f"Sell trigger date: {latest_sell_date}",
            f"Sell signals: {len(latest_sell)}",
            "",
        ]
    )

    for _, r in latest_sell.iterrows():
        lines.append(
            f"- SELL {r['ticker']} | {r['pattern']} | Exit {r['sell_price']} | Return {r['realized_return_pct']}%"
        )

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
    else:
        refresh_status = "skipped_send_only"
        trigger_status = "skipped_send_only"

    message = build_message(
        refresh_status=refresh_status,
        refresh_note=refresh_note,
        trigger_status=trigger_status,
        trigger_note=trigger_note,
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

    if refresh_status == "not_done" or trigger_status == "not_done":
        if allow_telegram_send:
            raise SystemExit("Pipeline had step failures, but Telegram status notification was sent.")
        raise SystemExit("Pipeline had step failures. Telegram notification was skipped by local policy.")


if __name__ == "__main__":
    main()
