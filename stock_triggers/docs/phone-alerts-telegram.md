# Daily Trigger Alerts On Phone (Telegram)

This setup sends daily Pattern A trigger updates to your phone using Telegram.

## 1) Create a Telegram bot

1. Open Telegram and chat with BotFather.
2. Run /newbot and follow prompts.
3. Save the bot token (looks like 123456:ABC...).

## 2) Get your chat ID

1. Send a message to your new bot from your phone.
2. Open this URL in browser (replace BOT_TOKEN):

https://api.telegram.org/botBOT_TOKEN/getUpdates

3. Find `chat` -> `id` in the JSON response. Save it.

## 3) Store credentials in secrets.yml (recommended)

In repo root, create/edit secrets.yml:

```yaml
TELEGRAM_BOT_TOKEN: "your_bot_token"
TELEGRAM_CHAT_ID: "your_chat_id"
```

`secrets.yml` is git-ignored.

Template file is available at:

- secrets.yml.example

## 4) Alternative: set credentials in terminal

From repo root:

```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

## 5) Run daily pipeline + send alert

```bash
stockpy11/bin/python stock_triggers/scripts/daily_triggers_telegram.py
```

This does:

- refresh prices from universe_tickers.txt
- generate Pattern A triggers
- send latest trigger summary to Telegram

If you want trigger generation only (skip price refresh):

```bash
stockpy11/bin/python stock_triggers/scripts/daily_triggers_telegram.py --skip-refresh
```

## 6) Optional schedule on macOS (launchd)

You can schedule this command daily (for example after market close).

Basic idea:

- Create a launch agent plist in ~/Library/LaunchAgents/
- Command should run:
  - /Users/.../stock-operator/stockpy11/bin/python
  - /Users/.../stock-operator/stock_triggers/scripts/daily_triggers_telegram.py
- Load it with:
  - launchctl load ~/Library/LaunchAgents/your.plist

If you want, we can add a ready-to-use plist file in this repo for your exact path/time.
