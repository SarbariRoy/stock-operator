# Stock Triggers UI

This is the Streamlit front end for the trigger engine.

The old description of this app as a simple Pattern A viewer is no longer accurate.

The current app is much closer to a small control room.

## What the app is for

The app helps with two main jobs:

1. deciding what to inspect for tomorrow
2. checking how the signals would have behaved historically

## Main navigation

The app's main nav currently has three primary pages:

1. Tomorrow's Picks
2. Backtesting Lab
3. Documentation

The UI is also moving toward linked help:

- major user-facing terms now get question-mark affordances that open the in-app Documentation page
- major tables use column help plus nearby column glossaries instead of a full custom table rewrite

## Run it

From the repo root:

```bash
streamlit run stock_triggers/ui/app.py
```

## What the app reads

Important files it uses:

- stock_triggers/data/prices_eod.csv
- stock_triggers/data/signals_pattern_a.csv
- stock_triggers/data/signals_all_patterns.csv
- stock_triggers/data/pattern_weights.json
- stock_triggers/data/candle_weights.json
- stock_triggers/data/stock_scores.csv
- stock_triggers/data/portfolio_positions.csv
- stock_triggers/data/external_factors.csv

## Tomorrow's Picks page

This is the fast “what should I review?” page.

What it does in the current code:

- builds a live A-G candidate list from the latest market date when prices are present
- otherwise falls back to saved signal files
- applies learned pattern-family bonuses
- shows a minimum score filter that defaults high
- can fall back to recent signals from the last few days if there is nothing fresh
- shows a score breakdown for a selected row
- shows the learned pattern weights in an expander

In plain language, this page is trying to stop you from staring at a giant raw CSV.

## Backtesting Lab page

This is the experimental and validation page.

What it does in the current code:

- tracks generated signals forward
- estimates returns, pnl, and days held
- filters by status and candle shapes
- can limit the displayed trades by max days held
- shows pattern-family and score fields directly in the tables
- uses saved all-pattern history if it exists
- only rebuilds from raw prices when the saved all-pattern file is missing
- shows the learned pattern weights in an expander here too

Important behavior detail:

there is currently a stop-exit lockout period before stop exits are allowed in the tracker logic.

## Score breakdown idea

The UI explains the score like a running total.

Very roughly, it does this:

$$
\\text{running score}
= \text{weighted components} + \text{MA slope bonus} + \text{pattern bonus} + \text{other bonuses}
$$

That is useful because you can see why a row is strong, not just that it is strong.

## Deployment notes

For hosted deployment:

- entry point is stock_triggers/ui/app.py
- Google login sessions can be remembered with a signed browser cookie; set GOOGLE_AUTH_COOKIE_SECRET in secrets or env for that to work reliably
- local runs do not require Google auth unless you explicitly set GOOGLE_AUTH_ENABLED to local, localhost, dev, always, or on for testing
- it is safest to keep write-heavy actions off in cloud mode
- let GitHub Actions refresh the data files
- let the app mostly read the committed outputs

That matches the way the current repo is set up.
