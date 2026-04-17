# Changelog

This is the complete in-app changelog for Stock Operator from repo inception.

The What's New panel stays intentionally short and recent. This page keeps the longer history in one place so you can trace how the app and pipeline evolved over time.

## 2026-04-17

### Recency-weighted training and stop-risk ranking controls

- Added recency-weighted training updates so the learned models can lean harder on newer signal history.
- Moved stop-risk pressure into ranking control flow instead of leaving it as a side metric.
- Exposed live Backtesting Lab controls for the stop-risk curve, thresholds, and hard gate so ranking behavior can be tuned and inspected in-app.

## 2026-04-15

### Persistent sign-in and automated What's New updates

- Added persistent Google auth sessions so production users do not need to re-authenticate as often.
- Added a pre-push hook that can auto-refresh the rolling What's New feed on pushes to master.
- Kept the release-note generation flow in repo so the in-app summary can stay aligned with shipped changes.

## 2026-04-14

### Learned candle context, coverage analysis, and production polish

- Switched candle scoring toward learned behavior by pattern family instead of relying on one global rule.
- Added coverage analysis and refreshed backtest defaults so the lab gives a broader view of what the engine is seeing.
- Introduced the first in-app What's New panel and release-note tracking data.
- Added production auth hardening and visible build markers for deployed app clarity.

## 2026-04-11 to 2026-04-12

### Documentation became a first-class app surface

- Added a dedicated Documentation page inside the Streamlit app.
- Added help chips across the UI so live screens can jump directly to deeper explanations.
- Expanded documentation with search, per-pattern deep dives, score-formula explanations, candle-enhancer diagrams, and pattern-map visuals.
- Fixed programmatic navigation edge cases so internal docs jumps do not get overwritten by navbar sync.

## 2026-04-04 to 2026-04-06

### Workflow expansion and Pattern G activation

- Activated Pattern G in signal generation and published the related signal artifacts.
- Added backfill support to the incremental workflow and reduced unnecessary retraining work in refresh paths.
- Refactored refresh pipeline sequencing and restored more reliable long-history behavior.
- Added real historical pattern charts to documentation and tightened mobile layout behavior.

## 2026-04-01 to 2026-04-03

### Pattern-family scoring and calibrated stop-risk pipeline

- Added calibrated stop-risk and learned penalty modeling so signal ranking could reflect downside pressure more explicitly.
- Extended scoring with pattern-family contribution logic instead of treating every family identically.
- Regenerated training artifacts and candle-weight workflows to keep saved outputs aligned with the newer model logic.
- Added confirmed hammer analysis as part of the candle-enhancer work.

## 2026-03-28 to 2026-03-30

### Candle enhancers and richer scoring inputs

- Added multiple bullish candlestick enhancers and later widened the enhancer total when history supported it.
- Improved scoring with RSI integration, moving-average slope metrics, and hold-to-target logic.
- Extended price-history retrieval and refreshed stock signals and scores with broader coverage.
- Improved Tomorrow's Picks presentation with status indicators, branding, and clearer signal display.

## 2026-03-24 to 2026-03-26

### Core ranking layer and Tomorrow's Picks refinement

- Added stock-health score generation and strengthened the trigger-scoring pipeline.
- Brought RSI calculation into the signal-scoring path.
- Refined the Tomorrow's Picks interface so the daily shortlist became easier to inspect quickly.

## 2026-03-16 to 2026-03-23

### Stock Triggers app foundation

- Added the initial stock-triggers documentation and Streamlit UI for reviewing Pattern A signals.
- Added Telegram integration and deployment notes for the daily trigger workflow.
- Expanded the data universe and pipeline plumbing used by the trigger engine.
- Continued iterating on the daily shortlist UI and operational workflow cleanup.

## 2026-01-03

### Repo inception point

- Added the mutual funds selector script, which is the earliest tracked commit in this repository.
- This marks the starting point before the repo grew into the broader Stock Operator and Stock Triggers workspace.

## Notes

- This first complete changelog is intentionally milestone-based instead of commit-by-commit.
- Recent day-to-day updates still appear in the rolling What's New panel.
- Future history can keep growing here without exposing raw git history inside the app UI.