# Stock Triggers UI (Streamlit)

This folder is for Streamlit-based UIs on top of the stock_triggers engine.

## Current app

- app.py – simple UI to view Pattern A signals from
  stock_triggers/data/signals_pattern_a.csv with basic filters.

Run from the repo root (after generating signals_pattern_a.csv):

```bash
streamlit run stock_triggers/ui/app.py
```

Make sure your virtualenv has Streamlit installed (e.g.,
`pip install streamlit`).

## Streamlit Cloud deployment notes

- Deploy app entrypoint: `stock_triggers/ui/app.py`.
- Keep the app in read-only mode on cloud (default):
  - `Enable refresh/trigger actions` toggle stays OFF.
- Run refresh/generation/notification via GitHub Actions, and let the UI read
  committed CSV outputs.
